# Stroma: Architecture

## Role

Stroma is the shared substrate under higher-order products that need a neutral corpus and retrieval layer.

Its job is narrow:

1. normalize records into a stable internal shape
2. chunk record bodies into retrievable sections, optionally with parent/leaf lineage
3. delegate vector generation to an embedder, optionally with a per-chunk context prefix
4. delegate text generation, and optional structured JSON decoding, to an OpenAI-compatible chat completion client over the same retry / Retry-After / failure-classification substrate as the embedder
5. persist records, chunks, vectors, FTS entries, and index metadata in SQLite
6. open immutable local snapshots for typed reads, hybrid semantic search with per-arm provenance, and parent/neighbor context expansion

Everything above that line belongs elsewhere.

## Product Boundary

Stroma owns:

- canonical corpus records
- content hashing and corpus fingerprints
- pluggable chunking (Markdown, kind-dispatched, hierarchical)
- embedding abstractions (`Embedder`, `ContextualEmbedder`) and an OpenAI-compatible HTTP embedder
- OpenAI-compatible chat completion HTTP transport (`chat.OpenAI`): request/response encoding, retry with capped `Retry-After`, response-size bounding, APIToken redaction, classified failures, and product-neutral structured JSON decoding
- a shared provider substrate (`provider`) that both `embed` and `chat` consume — retry policy, response-size cap, and the `FailureClass` taxonomy
- SQLite plus `sqlite-vec` storage across three quantization modes
- hybrid semantic retrieval: dense vector + FTS5 fused via a pluggable `FusionStrategy` with per-hit `HitProvenance`
- incremental `Update` with embedding reuse and forward-only schema migrations

Stroma does not own:

- specification semantics
- governance graphs
- source adapters such as GitHub, filesystem discovery, or MCP
- policy analysis such as overlap, compliance, drift, or impact
- prompt templates, system prompts, or semantic interpretation of structured chat responses
- product-layer diagnostic labels (`Runtime`, `Provider`, `RequestType`) — callers wrap `*provider.Error` to add those
- product-facing workflows, CLIs, or review/reporting layers

## Public Packages

- `corpus`
  - `Record` is the neutral artifact model.
  - `NewRecord(ref, title, body)` covers the common construction path.
  - `Record.Normalize()` fills safe defaults and computes a content hash when absent. `Record.Normalized()` is kept as a deprecated alias; `Validate()` remains exported but callers should run `Normalize` first.
  - `Fingerprint` / `FingerprintFromPairs` summarize the full indexed corpus deterministically and return an error on invalid inputs — records that cannot be normalized, or pairs with empty `Ref` / `ContentHash`. The injective encoding guarantee comes from `HashRecord`'s serialization, not from runtime collision detection.

- `chunk`
  - `Policy` is the chunking contract: `Chunk(ctx, Record) ([]SectionWithLineage, error)`. Every returned section is tagged with an optional parent index so hierarchical policies can persist parent/leaf lineage.
  - `MarkdownPolicy` (default) reproduces the pre-1.0 heading-aware flat pipeline byte-for-byte.
  - `KindRouterPolicy` dispatches to a per-`Kind` sub-policy.
  - `LateChunkPolicy` emits a parent span per heading-aware section, then leaf sub-sections beneath each parent; it may skip leaf emission when a parent already fits the child token budget.
  - `MarkdownWithOptions` returns `ErrTooManySections` when the body exceeds the caller's `MaxSections` cap (DoS guard).

- `embed`
  - `Embedder` is the minimum embedding contract the index depends on (`Fingerprint`, `Dimension`, `EmbedDocuments`, `EmbedQueries`).
  - `ContextualEmbedder` embeds `Embedder` and adds `EmbedDocumentChunks` for implementations that want document-aware chunk vectors.
  - `Fixture` provides deterministic embeddings for tests and offline use.
  - `OpenAI` provides a generic OpenAI-compatible HTTP embedder without product-layer env loading or diagnostics. `OpenAIConfig.MaxBatchSize` caps per-request batch size; the parent deadline scales with batch count so a slow early batch cannot starve later ones. `APIToken` is redacted in `String`, `GoString`, and `LogValue` so logs cannot leak the secret. Per-batch HTTP calls route through the shared `provider` core, inheriting its retry / Retry-After / failure-classification behaviour.

- `chat`
  - `OpenAI` provides a generic OpenAI-compatible chat completion client (`ChatCompletionText(ctx, messages, temperature, maxTokens)`). Mirrors `embed.OpenAIConfig`: `BaseURL`, `Model`, `Timeout`, `APIToken`, `MaxRetries`, `MaxResponseBytes`. Negative `MaxRetries` normalize to zero. `APIToken` is redacted in `String` / `GoString` / `LogValue`.
  - `ChatCompletionJSON(ctx, req, target)` is the product-neutral structured-output helper. It sends the same chat request shape, optionally carries caller-supplied JSON Schema through `response_format`, extracts a JSON object from plain/fenced/prose-wrapped assistant text, and decodes into the caller-owned target. Invalid JSON, missing JSON, non-object JSON, and typed decode failures classify as `FailureClassSchemaMismatch`; upstream provider errors keep their original class. One bounded repair pass is available through `JSONRetryPolicy`, capped at one pass.
  - Handles both the plain-string `content` shape and the multi-part array shape used by tool-capable models. Any other shape (object, number, bool, null) surfaces as `FailureClassSchemaMismatch` — the package refuses to persist raw JSON as assistant text.
  - Prompt templates, semantic output validation, and product-specific interpretation of failures stay at the caller.

- `provider`
  - Shared HTTP substrate used by `embed` and `chat`. Not a full HTTP client — it takes a caller-supplied `*http.Client`, a `Target` (method/URL/body/token), and a decode function, and handles retry, `Retry-After`, response-size bounding, and failure classification around them.
  - `FailureClass` taxonomy: `auth`, `rate_limit`, `timeout`, `server`, `transport`, `schema_mismatch`, `dependency_unavailable`. HTTP status classification wins over message parsing; message parsing is the fallback when the status carries no signal.
  - `*provider.Error` is the public error type: every `provider.Do` failure satisfies `errors.As(*provider.Error)`. Where a lower-level transport/read/decode cause exists, `Error.Unwrap()` preserves it so callers and retry classification can use `errors.As`. Callers branch on `Error.FailureClass()` and attach their own product-layer labels via wrapping.
  - `Policy.MaxRetryAfter` caps a server-supplied `Retry-After` header at 30s by default so a hostile or misconfigured upstream cannot park a goroutine for hours via a large header value. Callers with a larger budget override the cap explicitly.

- `store`
  - validates that SQLite and `sqlite-vec` are usable
  - opens read-only and read-write handles consistently
  - `QuantizationFloat32` (default), `QuantizationInt8` (4× smaller, minor precision loss), and `QuantizationBinary` (1-bit sign-packed `vec0` prefilter that is 32× smaller than float32 for that representation; the companion `chunks_vec_full` table retains full precision for cosine rescoring, so total snapshot size is not 32× smaller)
  - translates embeddings to and from `sqlite-vec` blob format for each quantization mode

- `index`
  - `Rebuild` writes the index atomically via a staging file + rename. `BuildResult.ReuseStatus` / `ReuseDisabledReason` report whether `ReuseFromPath` was active, unavailable, incompatible, or disabled without making reuse setup fatal by default.
  - `Update` mutates an existing snapshot in place, deleting removed records and re-embedding only changed sections; it chains forward schema migrations v2 → v3 → v4 → v5 in the same transaction.
  - Added/replaced records are chunked, contextualized, reuse-planned, and embedded before the write transaction starts. `UpdateOptions.MaxPlannedRecords` lets callers cap the number of resident pre-transaction record plans and drive large ingests through repeated bounded batches; over-cap updates fail before embedding or opening the transaction. `MaxChunkSections` remains the per-record section-expansion cap.
  - Embedding reuse is keyed on `(title, heading, body, context_prefix)` so both plain and contextual snapshots benefit.
  - `Snapshot` (from `OpenSnapshot`) exposes long-lived `Stats`, `Records`, `Sections`, `Search` (hybrid), `SearchVector` (dense only), and `ExpandContext` (parent + neighbor window).
  - Search retrieval paths run through `FusionStrategy` — `DefaultFusion()` returns `RRFFusion{K:60, PreserveSingleArmScore:true}`. Ordering matches the pre-#17 hardcoded RRF on every pre-change shape; `Score` matches on the multi-arm hybrid path, and on single-arm paths (legacy snapshots without `fts_chunks`, or vector-empty + FTS-non-empty) `DefaultFusion` preserves the arm-native score instead of rewriting to RRF. Callers that need both values read them off `HitProvenance`. Custom strategies receive `RetrievalArm{Name, Hits, Available, Err}` for each arm and must attach `HitProvenance` to every returned hit.
  - `Reranker` optionally refines the fused shortlist; it sees the full `HitProvenance` so score-aware rerankers can weight per-arm evidence.
  - `SearchParams.SearchDimension` enables a truncated-prefix vector prefilter (Matryoshka) with full-dim cosine rescore; only valid on float32 snapshots.
  - `DefaultSearchLimit = 10` fills zero-value `Limit` fields. `MaxSearchLimit = 250` is the explicit upper bound for `Search` and `SearchVector`; larger limits return an error rather than being silently capped.

## Retrieval Contract

The snapshot API is the public traceability boundary for retrieval consumers that persist search results outside the immediate call.

Stable identity fields:

- `Stats.ContentFingerprint` identifies the indexed content generation for an opened snapshot. `BuildResult.ContentFingerprint` is the same generation value immediately after `Rebuild` or `Update`.
- `SearchHit.ChunkID` identifies one chunk only within that content generation. It is safe to pass back to `Snapshot.ExpandContext` for the same snapshot generation; it is not a cross-rebuild identity.
- `SearchHit.Ref` is the normalized record ref. `SearchHit.SourceRef`, `Kind`, `Title`, `Heading`, `Content`, and `Metadata` are snapshot-local payload fields that callers may forward as evidence context.

Unstable ranking fields:

- `SearchHit.Score` is query-local. Its scale depends on the retrieval/fusion path.
- `HitProvenance` records which retrieval arms contributed to a fused hit, with each arm's native rank and score. It is audit/debug evidence for the query, not a durable identifier.

Safe expansion pattern:

1. Open the snapshot and read `Stats`.
2. Search through that `Snapshot` rather than reopening per query.
3. Persist `{content_fingerprint, chunk_id, ref}` plus any needed metadata/source fields.
4. Before expanding a saved handle later, compare the current `Stats.ContentFingerprint` to the saved fingerprint. If it differs, treat the handle as stale and rerun search or delegate generation tracking to a catalog above Stroma.
5. Call `ExpandContext(chunk_id, opts)` only after that generation check. A missing chunk returns `[]Section{}` with nil error; unsupported or malformed snapshots fail earlier at `OpenSnapshot` or on the affected read.

`ExpandContext` always includes the requested chunk when it exists. On flat snapshots, `IncludeParent` is a no-op and `NeighborWindow` walks same-record chunks by `chunk_index`. On hierarchical snapshots, `IncludeParent` includes one parent row when present, and neighbors are sibling chunks under the same parent group. Returned sections are in document order and never include embeddings.

Batch retrieval does not need a special batch API. Keep one `Snapshot` open for many `Search` and `ExpandContext` calls. `Snapshot` read methods are safe for concurrent use through `database/sql`; callers still own backpressure, cancellation, and goroutine counts. Results are bounded by `MaxSearchLimit` and by the requested context window, so memory growth is caller-controlled through limits and concurrency.

## On-Disk Schema

Schema version is `"5"`. Snapshots produced by older versions of Stroma open read-only and migrate to v5 atomically on the first `Update`.

- `records`
  - canonical normalized records + metadata
- `chunks`
  - retrievable sections keyed to a record, with a `context_prefix` column for contextual retrieval and a `parent_chunk_id` column for hierarchical lineage. Same-record lineage is enforced at the storage layer via BEFORE INSERT / BEFORE UPDATE triggers that block cross-record parents.
- `chunks_vec`
  - `vec0` virtual table storing chunk embeddings. Column type is `float[N]`, `int8[N]`, or `bit[N]` depending on the snapshot's quantization.
- `chunks_vec_full` (binary quantization only)
  - full-precision companion feeding the cosine rescore pass over the hamming shortlist.
- `fts_chunks`
  - `fts5` virtual table over `(title, heading, content)` driving the FTS retrieval arm.
- `metadata`
  - schema version, embedder fingerprint, embedder dimension, content fingerprint, quantization mode, creation time.

### Migration chain

`Update` walks v2 → v3 → v4 → v5 inside one transaction, so older snapshots upgrade in place without a separate tool:

| From | Path                     |
|------|--------------------------|
| v2   | Update → v3 → v4 → v5    |
| v3   | Update → v4 → v5         |
| v4   | Update → v5              |
| v5   | already current          |

`OpenSnapshot` accepts v2–v5 read-only. The v2 → v3 step adds the `chunks.context_prefix` column; the v3 → v4 step re-hashes `records.content_hash` in place using the current `HashRecord` encoding (no DDL); the v4 → v5 step adds the `chunks.parent_chunk_id` column, a partial index on it, and the same-record triggers.

The schema is product-neutral. There are no spec/document distinctions, no governance edges, and no transport-specific tables. Higher-order products should treat the snapshot as a Stroma-owned artifact and go through the library API instead of joining these tables directly.

## Extraction Principle

Stroma should stay boring.

If a feature introduces domain language such as spec status, applies-to rules, drift classes, review flows, prompt templates, or product transport contracts, it belongs in the consuming product, not in Stroma. The chat and embed packages ship HTTP transport primitives only — anything that looks like product semantics (which model to pick, what to prompt it with, how to validate its JSON output, which runtime label to attach to a diagnostic) stays at the caller.
