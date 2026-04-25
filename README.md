# Stroma

Stroma is a neutral corpus and indexing substrate.

It owns the lowest-level operations needed to ingest text artifacts, chunk them, embed them, persist them in SQLite plus `sqlite-vec`, retrieve semantically close sections, and call OpenAI-compatible embedding and chat completion endpoints over a shared HTTP substrate. Callers consume Stroma through its APIs and treat the SQLite snapshot as an opaque local artifact. It does not own governance, specifications, compliance, drift analysis, prompt templates, product-specific output semantics, MCP, or CLI workflows.

## Scope

Stroma is for products that need a reusable text corpus layer with:

- canonical records with deterministic content fingerprints
- pluggable chunking strategies (`chunk.Policy` — `MarkdownPolicy` default, `KindRouterPolicy` for per-record-kind dispatch, `LateChunkPolicy` for parent/leaf hierarchy)
- pluggable embedders (`Embedder` / `ContextualEmbedder`) with a deterministic fixture and an OpenAI-compatible HTTP embedder
- OpenAI-compatible chat completion client (`chat.OpenAI`) sharing the same substrate as `embed.OpenAI`: retry with `Retry-After` (capped), classified failures (`auth` / `rate_limit` / `timeout` / `server` / `transport` / `schema_mismatch` / `dependency_unavailable`), preserved lower-level causes on provider errors, APIToken redaction, and a product-neutral structured JSON helper
- hybrid retrieval: dense vector + FTS5, fused via a pluggable `FusionStrategy` (`RRFFusion` by default) with per-arm provenance surfaced to downstream rerankers
- quantization knobs: `float32` (default), `int8` (4× smaller), `binary` (1-bit sign-packed `vec0` prefilter that is 32× smaller for the prefilter representation; full-precision vectors are retained in a companion table for cosine rescoring, so total snapshot size is not 32× smaller)
- optional Matryoshka prefilter at a truncated dimension with full-dim cosine rescore (`SearchParams.SearchDimension`)
- atomic rebuilds and incremental `Update` with embedding reuse at the section level, chaining schema migrations v2 → v3 → v4 → v5 in one transaction

Stroma is not for:

- spec governance
- source discovery or repository scanning
- code compliance or doc drift analysis
- prompt templates, system prompts, or semantic interpretation of structured chat responses
- product-specific adapters and transports

## Packages

- `corpus` — canonical record model, `NewRecord` helper, `Normalize`, deterministic `Fingerprint`
- `chunk` — `Policy` interface with `MarkdownPolicy`, `KindRouterPolicy`, `LateChunkPolicy`; `MarkdownWithOptions` returns `ErrTooManySections` when a body exceeds the DoS cap
- `embed` — `Embedder` and `ContextualEmbedder` interfaces; deterministic `Fixture`; OpenAI-compatible HTTP embedder with `MaxBatchSize` batching, deadline scaling across batches, and `APIToken` redaction in `String`/`GoString`/`LogValue`
- `chat` — OpenAI-compatible chat completion client (`chat.OpenAI`, `chat.Message`, `ChatCompletionText`, `ChatCompletionJSON`); tolerates string and multi-part array content; structured JSON responses decode into caller-owned targets and malformed JSON returns `schema_mismatch`; `APIToken` redaction parity with `embed.OpenAIConfig`
- `provider` — shared HTTP substrate used by `embed` and `chat`: retry with capped `Retry-After`, response-size bounding, negative `MaxRetries` normalization to zero, and a stable `FailureClass` taxonomy surfaced via `*provider.Error`. Callers branch on `FailureClass` to retry / degrade / propagate, and can unwrap lower-level transport/decode causes where available
- `store` — SQLite readiness probes, `sqlite-vec` readiness, quantization blob helpers (`QuantizationFloat32` / `QuantizationInt8` / `QuantizationBinary`)
- `index` — atomic `Rebuild` with embedding reuse and explicit reuse diagnostics, incremental `Update`, long-lived `Snapshot` readers, `Stats`, hybrid `Search` with provenance and explicit `MaxSearchLimit`, `ExpandContext` for parent/neighbor walks

## Retrieval Evidence And Batch Use

Use `OpenSnapshot` when issuing many searches against one built index. A `Snapshot` is safe for concurrent reads; callers own the concurrency limit, so use a bounded worker pool or semaphore sized for the host and workload, then close the snapshot after all searches and context expansions finish.

For durable evidence handles, persist at least:

- `Stats.ContentFingerprint` from the opened snapshot, identifying the indexed content generation
- `SearchHit.ChunkID`, identifying a chunk only within that snapshot generation
- `SearchHit.Ref` plus any caller-needed record metadata or `SourceRef`

`ChunkID` is not a cross-rebuild identity. Before expanding a previously saved hit, reopen the snapshot, compare `Stats.ContentFingerprint` with the saved value, and rerun search if it differs. `SearchHit.Score` and `HitProvenance` are ranking evidence for the query that produced the hit; keep them for audit/debugging, but do not use them as identity fields.

`ExpandContext(hit.ChunkID, opts)` returns the hit chunk plus requested parent/neighbor sections in document order. On flat snapshots, parent expansion is a no-op and neighbors are same-record chunks. On hierarchical snapshots, parent expansion follows `parent_chunk_id` one level and neighbors stay in the same sibling group. A missing chunk returns an empty slice and nil error, which lets wrappers treat stale handles as "not found" after they have already checked the content fingerprint.

## Example

```go
ctx := context.Background()

records := []corpus.Record{
    corpus.NewRecord(
        "widget-overview",
        "Widget Overview",
        "# Overview\n\nWidgets are synchronized in batches.",
    ),
}

fixture, err := embed.NewFixture("fixture-demo", 16)
if err != nil {
    log.Fatal(err)
}

if _, err := index.Rebuild(ctx, records, index.BuildOptions{
    Path:     "stroma.db",
    Embedder: fixture,
}); err != nil {
    log.Fatal(err)
}

hits, err := index.Search(ctx, index.SearchQuery{
    Path: "stroma.db",
    SearchParams: index.SearchParams{
        Text:     "synchronized batches",
        Limit:    5,
        Embedder: fixture,
        // Fusion / Reranker / SearchDimension are optional; zero values
        // give hybrid RRF over vector+FTS with the full stored dimension.
    },
})
if err != nil {
    log.Fatal(err)
}

fmt.Println(hits[0].Ref)
```

See the [v2.0.0 release notes](https://github.com/dusk-network/stroma/releases/tag/v2.0.0) for the full API surface.

## Status

v2.0.0 (current) ships the stable substrate: hybrid retrieval, pluggable fusion, quantization, matryoshka, contextual retrieval, adaptive chunking, and incremental update. Higher-order products should consume the library rather than re-embedding their own indexing substrate.
