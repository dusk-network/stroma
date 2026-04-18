# Adaptive chunk policies and context-preserving retrieval (#16)

**Status:** Approved for implementation
**Target release:** v1.0
**Schema bump:** v4 → v5

## Problem

Stroma currently supports heading-aware Markdown chunking, optional token-budget splitting with overlap, and optional contextual chunk embeddings via `embed.ContextualEmbedder`. The index stores a single flat chunk view per record and search returns only that flat hit set. Downstream consumers (Hippocampus, Pituitary) want adaptive chunk policies and explicit context-preserving retrieval primitives without inventing the reconstruction layer above the substrate.

Issue #16 lists four sub-deliverables:

1. Per-record / per-kind chunk policy selection.
2. Chunk lineage metadata so a retrieval chunk can point to a broader parent span.
3. Snapshot/search APIs that return a hit together with expanded local context.
4. Default behavior remains identical unless callers opt in.

All four ship in the v1.0 release window so the schema bump can land once.

## Design decisions

| Question | Decision | Reasoning |
|----------|----------|-----------|
| Lineage model | Parent-pointer FK on `chunks` (NULLable `parent_chunk_id`) | Subsumes hierarchical, late-chunking, and neighbor-expansion patterns under one schema. Single FK shape. One nullable column added in the v4→v5 bump. |
| Chunker API shape | `chunk.Policy` strategy interface | Separates "what is a chunking strategy" from "what are its parameters." `chunk.Options` stays focused on the Markdown defaults. Per-record dispatch composes via `KindRouterPolicy`. Downstream products implement custom policies without forking. |
| Expansion API surface | Separate `Snapshot.ExpandContext` method only | Substrate-pure: do one thing per API. Consumers typically rerank/filter before showing results, so they want to expand only the survivors. Inline expansion on `Search` is a convenience that risks 5–10× the work in the common case. Search/SearchHit stay byte-identical. |
| Embedding parents | Leaves only; parents are storage-only | Matches the late-chunking research. Search behavior stays uniform across snapshots. Avoids the "parent always wins broad queries" surprise. Half the embedding cost vs. embedding everything. |
| Lineage storage on Section | New `chunk.SectionWithLineage` wrapper | Keeps `chunk.Section` byte-identical for direct callers of `chunk.MarkdownWithOptions`. The Policy interface returns the wrapper; the index unpacks it. |
| Schema migration | v4 → v5 chained from v2/v3 via existing `migrateSchemaToCurrent` | Consistent with prior bumps. Adds one nullable column + one index; no row rewrites needed. |

## Architecture

### `chunk` package additions

```go
// Section stays unchanged.
type Section struct {
    Heading string
    Body    string
}

// NoParent is the sentinel used by SectionWithLineage.ParentIndex to mark
// a chunk as a root (flat or top-of-hierarchy).
const NoParent = -1

// SectionWithLineage decorates a Section with parent linkage. ParentIndex
// holds either NoParent or the slice index of another SectionWithLineage
// in the same Policy result. Forward references and cycles are rejected
// at insert time.
type SectionWithLineage struct {
    Section
    ParentIndex int
}

// Policy decides how a record's body becomes chunks. The default
// MarkdownPolicy reproduces the pre-1.0 chunking pipeline exactly so
// callers that do not opt in see no behavior change.
type Policy interface {
    Chunk(ctx context.Context, record corpus.Record) ([]SectionWithLineage, error)
}

// MarkdownPolicy wraps the existing heading-aware + token-budget split
// pipeline. ParentIndex is always NoParent on the returned slice.
type MarkdownPolicy struct {
    Options Options
}

// KindRouterPolicy dispatches chunking by record.Kind, falling back to
// Default when no per-kind policy is registered. Useful when one corpus
// mixes kinds with different optimal chunking shapes.
type KindRouterPolicy struct {
    Default Policy
    ByKind  map[string]Policy
}

// LateChunkPolicy emits a parent span per heading-aware section, then
// token-budget-split children that point at the parent. Reference
// implementation of the late-chunking pattern from the research signal.
type LateChunkPolicy struct {
    ParentMaxTokens    int  // 0 = one parent per heading-aware section
    ChildMaxTokens     int  // required, must be > 0
    ChildOverlapTokens int  // overlap between adjacent leaves under the same parent
    MaxSections        int  // cap parents + leaves combined
}
```

The `chunk.Options.MaxSections` cap continues to bound parents + leaves combined when used inside Policy implementations.

### `index` package additions

#### Schema v5

```sql
ALTER TABLE chunks ADD COLUMN parent_chunk_id INTEGER
    REFERENCES chunks(id) ON DELETE CASCADE;
CREATE INDEX idx_chunks_parent ON chunks(parent_chunk_id);
UPDATE metadata SET value = '5' WHERE key = 'schema_version';
```

`parent_chunk_id` is NULL for flat chunks (the only kind today's `MarkdownPolicy` produces) and for parent rows. It points at the parent's `chunks.id` for leaf rows. The index supports the parent-walk path in `ExpandContext`.

`OpenSnapshot` accepts v2, v3, v4, v5 read-only — read paths decode all four versions directly without forcing an Update. `Update` chains v2 → v3 → v4 → v5 in one transaction. Existing v2/v3/v4 snapshots Open with `parent_chunk_id` absent; the existing `hasContextPrefix`-style cached probe pattern extends to a `hasParentChunkID` flag set at open time so read paths know whether to issue parent-aware SQL or skip the join.

#### `BuildOptions` / `UpdateOptions`

```go
type BuildOptions struct {
    // ... existing fields ...

    // ChunkPolicy selects the chunking strategy. Nil defaults to
    // chunk.MarkdownPolicy{Options: chunk.Options{
    //     MaxTokens: MaxChunkTokens,
    //     OverlapTokens: ChunkOverlapTokens,
    //     MaxSections: resolveMaxChunkSections(MaxChunkSections),
    // }} so existing callers see no behavior change.
    ChunkPolicy chunk.Policy
}
```

`UpdateOptions` mirrors the field with the same defaulting. When set, the policy must produce stable parent topology for the same record body so reuse keying still matches across rebuilds — the substrate does not validate this; consumers who use stochastic policies own the consequence.

#### Two-pass insert in `indexSession.processRecord`

The existing single-pass insert becomes two passes:

1. **Topology pass.** Resolve which sections are parents (any index referenced by `ParentIndex`) and assign provisional roles. Parents must precede their children in the slice (validated; out-of-order topology returns a clear error). Build a `parentIndices` set.
2. **Insert pass.**
    - For each section in order:
        - Insert the chunk row. If the section is a parent (`parentIndices` contains its index), `parent_chunk_id = NULL` and skip the embed call. If it is a leaf with `ParentIndex != NoParent`, resolve the FK to the parent's `chunks.id` from a slice-index → chunk-id map.
        - Parents skip both the FTS5 row and the vector blob — they are storage-only context and must not surface from either arm of hybrid search. `ExpandContext` reads `chunks` directly, so the parent's text is still available for parent-walk retrieval.
        - For non-parents, insert the FTS5 row, run the embedder, and insert the vector blob (and the binary companion if applicable).

Reuse keying stays leaf-centric. The reuse map continues to key on `(heading, content, context_prefix)`; parents get fresh rows on every rebuild because they have no embedding to reuse. If a record's body changes such that the leaves are stable but a parent's text shifts, the leaves correctly reuse their embeddings while the parent row is rewritten.

`ChunkContextualizer` continues to receive the full section slice. Implementations are expected to return empty prefixes for parent sections (parents are not embedded, so the prefix is irrelevant); the substrate does not enforce this.

#### `ExpandContext` API

```go
type ContextOptions struct {
    // IncludeParent walks parent_chunk_id one level up and includes
    // the parent row in the returned slice (when the chunk has a
    // parent). Multi-level ancestry walks are explicit recursion by
    // the caller.
    IncludeParent bool

    // NeighborWindow includes up to N sibling chunks on each side of
    // the requested chunk, ordered by chunk_index. For a leaf, siblings
    // share the same parent_chunk_id. For a flat chunk, siblings are
    // other flat chunks under the same record_ref. Zero means no
    // neighbors.
    NeighborWindow int
}

// ExpandContext returns the chunk identified by chunkID together with
// the caller-requested local context, in document order:
//   [parent (if requested + present), neighbors before, the chunk itself, neighbors after]
//
// The chunk itself is always included. Embeddings are never populated;
// callers needing vectors should use Sections().
func (s *Snapshot) ExpandContext(ctx context.Context, chunkID int64, opts ContextOptions) ([]Section, error)
```

Implementation: a small bounded sequence of parameterized reads per call — at most one to locate the requested chunk, one to fetch the parent (when `IncludeParent` is set and `parent_chunk_id` is non-NULL), and one range scan over the sibling window scoped by `record_ref`, lineage, and `chunk_index BETWEEN low AND high`. There is no per-result parameter expansion (no `WHERE id IN (?, ?, ?, ...)`), so the query never approaches SQLite's parameter cap regardless of `NeighborWindow`. Returns `Section` rows compatible with the existing `Section` type (Heading, Content, ContextPrefix, ChunkID, Ref, Kind, Title, SourceRef, Metadata) with `Embedding = nil`. Cross-record `parent_chunk_id` values are rejected at storage (v5 same-record triggers) and ignored at read time (the parent-load query bakes the same-record predicate into its WHERE clause), so a malformed snapshot cannot leak parent context across records.

When the snapshot was opened against a v4 (pre-lineage) file, `parent_chunk_id` is treated as universally NULL: `IncludeParent` is a no-op, `NeighborWindow` falls back to flat-chunk neighbors via `chunk_index`. ExpandContext stays useful against legacy snapshots — it just can't surface lineage that was never recorded.

### Default behavior preservation

`BuildOptions{}` and `UpdateOptions{}` with `ChunkPolicy == nil` produce byte-identical output to today: flat chunks with `parent_chunk_id` NULL on every row, same `chunk_index` ordering, same embeddings, same reuse keys. The only observable schema delta is the new (always-NULL) `parent_chunk_id` column and the same-record lineage triggers on `chunks` — invisible to the read API contract and to existing tests.

`Search`, `SearchHit`, `Sections`, `SectionQuery`, and `Snapshot.Records` are byte-identical. The new surface is strictly additive: `chunk.Policy` (interface), `chunk.SectionWithLineage`, `chunk.MarkdownPolicy`, `chunk.KindRouterPolicy`, `chunk.LateChunkPolicy`, `chunk.NoParent`, `index.BuildOptions.ChunkPolicy`, `index.UpdateOptions.ChunkPolicy`, `index.ContextOptions`, `Snapshot.ExpandContext`.

## Components and units

### `chunk` package

- `chunk.Section` — unchanged.
- `chunk.SectionWithLineage` — new, thin wrapper over `Section` with `ParentIndex`.
- `chunk.Policy` — new interface; one method.
- `chunk.MarkdownPolicy` — wraps `MarkdownWithOptions`; ~20 LOC.
- `chunk.KindRouterPolicy` — dispatches by `record.Kind`; ~20 LOC.
- `chunk.LateChunkPolicy` — reference implementation of late chunking; ~80 LOC.
- `chunk.NoParent` — sentinel constant.

Each is independently testable with table-driven tests on representative records.

### `index` package

- `index.ContextOptions` — new struct.
- `Snapshot.ExpandContext` — new method; one SELECT.
- Schema v5 migration — `migrateV4ToV5` chained through `migrateSchemaToCurrent`.
- `Snapshot.hasParentChunkID` cached flag at open time (mirrors `hasContextPrefix`, `hasFTS` patterns from #36 and #42).
- `indexSession.processRecord` — two-pass insert variant; gated on policy result.
- `BuildOptions.ChunkPolicy` / `UpdateOptions.ChunkPolicy` — new field with default-fallback construction.

## Data flow

### Build-time

```
records → policy.Chunk(record) → []SectionWithLineage
       → topology pass (identify parents, validate forward order)
       → for each section in order:
            insert chunks row (parent FK resolved via index→id map)
            insert fts_chunks row (parent and leaf alike)
            if non-parent:
                contextualize → embed → insert chunks_vec (+ chunks_vec_full for binary)
```

### Read-time (Search)

Unchanged. Returns `SearchHit` from leaves only (parents are not in `chunks_vec`).

### Read-time (ExpandContext)

```
chunkID → load (record_ref, parent_chunk_id, chunk_index)
       → optional: load parent row (if IncludeParent + parent_chunk_id IS NOT NULL)
       → load neighbor rows where:
            record_ref = ?
            COALESCE(parent_chunk_id, -1) = COALESCE(?, -1)
            chunk_index BETWEEN ?-N AND ?+N
       → assemble in document order
       → return []Section (no embeddings)
```

## Error handling

- `Policy.Chunk` errors propagate up unchanged through the existing per-record error chain.
- Forward parent references in `SectionWithLineage.ParentIndex` (parent appearing after its child in the slice) return a clear error at the topology pass: `"chunk policy emitted forward parent reference: section %d points to %d which has not yet been emitted"`.
- Cyclic parent references (a section points at itself or transitively at itself) are rejected by the same forward-only constraint — a child can only point at sections with strictly smaller indices.
- `ExpandContext` against a non-existent `chunkID` returns an empty slice + nil error (consistent with section reads).
- `ExpandContext` against a snapshot that has no `parent_chunk_id` column (v4 or earlier) silently degrades: `IncludeParent` returns no parent; `NeighborWindow` falls back to flat-chunk neighbors via `chunk_index`. The cached `hasParentChunkID` flag drives this branch.
- `LateChunkPolicy` with `ChildMaxTokens <= 0` returns a clear configuration error at first `Chunk` call.

## Testing strategy

### `chunk` package

- `MarkdownPolicy` golden tests: same inputs as the existing `MarkdownWithOptions` tests; assert byte-identical sections + all `ParentIndex == NoParent`.
- `KindRouterPolicy` table-driven tests: per-kind dispatch, fallback to Default, missing-Default error.
- `LateChunkPolicy` table-driven tests: parent + child counts, `ParentIndex` topology, `MaxSections` enforcement, `ChildOverlapTokens` semantics.

### `index` package

- v4 → v5 migration: `rewriteSnapshotToV4(path)` + `Update` chains forward; assert `parent_chunk_id` column exists, all NULL, search results unchanged.
- Two-pass insert with `LateChunkPolicy`: build a small corpus, assert parent rows have NULL embedding (confirm via `chunks_vec` row count = number of leaves), assert leaves carry resolved `parent_chunk_id`.
- Reuse keying with parents: rebuild against same body, assert leaves reuse embeddings, parents are rewritten fresh, fingerprint stays stable.
- Forward parent reference rejection: a synthetic policy that emits leaf-before-parent triggers the topology error.
- `Snapshot.ExpandContext`: parent-only, neighbor-only, both, against a leaf in a hierarchy, against a flat chunk, against a chunk that doesn't exist (empty result), against a v4 snapshot (degraded but useful).
- Default behavior preservation: existing test corpus + `BuildOptions{ChunkPolicy: nil}` produces a v5 snapshot whose chunks/vectors are byte-identical to today's v4 (modulo the schema_version metadata bump and the always-NULL `parent_chunk_id` column).

### Integration

- Existing search, fingerprint, reuse, and snapshot tests run unchanged against the default (nil) policy.
- New test asserting `Search` ignores parent rows (parents are absent from `chunks_vec` so they cannot surface).

## PR slicing

Two PRs against the v1.0 release window:

**PR-A: Storage + ExpandContext (~500 LOC + tests)**

- Schema v5 migration with chained v2/v3/v4 → v5 via `migrateSchemaToCurrent`.
- `parent_chunk_id` column + index + cached `hasParentChunkID` flag on `Snapshot`.
- `ExpandContext` API (works for neighbor expansion against any snapshot; parent expansion against v5).
- `OpenSnapshot` accepts v3/v4/v5 read-only; v4 snapshots opened cleanly.
- Tests: migration round-trip, `ExpandContext` against legacy and current snapshots, neighbor windows.
- No chunker changes. Default behavior unchanged.

**PR-B: Policy framework (~1500 LOC + tests)**

- `chunk.SectionWithLineage`, `chunk.Policy`, `chunk.NoParent`.
- `chunk.MarkdownPolicy`, `chunk.KindRouterPolicy`, `chunk.LateChunkPolicy`.
- `BuildOptions.ChunkPolicy`, `UpdateOptions.ChunkPolicy` with nil-defaulting.
- `indexSession.processRecord` two-pass insert.
- Topology validation (forward-only parent references).
- Reuse keying continues to work with parent rows in the mix.
- Tests: per-policy table-driven tests, two-pass insert assertions, reuse + parent interaction, default-behavior preservation.

PR-B depends on PR-A's schema. Both must merge before v1.0 is tagged.

## Out of scope

- Multi-level recursive ancestry walks in `ExpandContext` (caller can recurse via the parent's chunkID).
- Inline expansion on `Search` (deferred to v1.x if a consumer demonstrates the convenience case bites).
- Embedding parents (Q decision; can be revisited in v1.x via a new policy or a per-section flag).
- Cross-record neighbor expansion.
- Persisted chunk-policy metadata (the snapshot does not record which policy built it; reuse keying handles divergence implicitly via fingerprint mismatches).
