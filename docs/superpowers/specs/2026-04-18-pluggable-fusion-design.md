# Pluggable hybrid fusion and reranker provenance (#17)

**Status:** Draft — codex adversarial review resolved, ready for implementation
**Target release:** post-v1.0 (first minor after 1.0.0)
**Schema bump:** none

## Problem

`Snapshot.Search` fuses the vector arm and FTS5 arm with Reciprocal Rank Fusion
implemented in a private `mergeRRF`. The shape is frozen in three ways:

1. Fusion is hard-coded to RRF (`rrfK = 60`) with tie-breaks on source count and
   best cross-arm rank.
2. Per-arm candidate lists and arm-native scores (cosine for vector, negative
   bm25-equivalent for FTS) are discarded during fusion. The fused `SearchHit.Score`
   is the RRF score; no caller can recover the arm evidence.
3. The optional `Reranker` hook receives fused candidates only. Rerankers that
   want to weight, filter, or debug by arm provenance have to either infer it
   heuristically or reach around the substrate.

Downstream products (Hippocampus retrieval quality, Pituitary shortlist control)
need to experiment with hybrid weighting and stronger rerank stages without
forking the kernel. Stroma stays substrate — domain-specific fusion logic does
not belong inside `index` — but the composition points have to exist.

## Design decisions

| Question | Decision | Reasoning |
|----------|----------|-----------|
| Fusion extension point | `FusionStrategy` interface taking `[]RetrievalArm` | Keeps fusion strategy separate from arm production. Default `RRFFusion{K:60}` reproduces today's behavior byte-for-byte. Custom strategies do not have to touch `Snapshot.Search`. |
| Per-arm exposure | `RetrievalArm{Name, Hits}` passed to strategies; arm-native `Score` preserved on each `SearchHit` before fusion | Strategies get both rank order (slice position) and raw score. The reranker sees provenance downstream via `SearchHit.Provenance`. |
| Provenance shape | `HitProvenance{Arms map[string]ArmEvidence}` attached to fused `SearchHit`s | Map-by-arm-name is extensible to N arms without interface churn. `ArmEvidence{Rank, Score}` carries everything a rerank strategy actually needs. |
| Provenance population | Always populated on fusion-path hits | One small map allocation per hit is trivial next to embedding / SQL cost. Opting in would force callers to set a flag just to use the information; no measurable saving. |
| Single-arm degenerate case | `Snapshot.Search` always calls `Fuse`; `RRFFusion` preserves arm-native scores when exactly one arm has hits (via `PreserveSingleArmScore` defaulting to true) | One pipeline path is simpler to reason about and lets custom strategies observe every case. The default `RRFFusion` keeps arm-native scores on single-arm results so legacy snapshots (no `fts_chunks`) and zero-hit FTS queries remain byte-identical to pre-#17 behavior. Callers who want uniform RRF scoring set `PreserveSingleArmScore: false`. |
| Arm availability and errors | `RetrievalArm` carries `Available bool` and `Err error`; `Fuse` returns `([]SearchHit, error)` | Custom strategies must distinguish "arm unavailable on this snapshot" (legacy schemas without `fts_chunks`) from "arm ran and found nothing" and from "arm failed". Silent conflation under version skew or degraded dependencies is a long-term footgun, so the interface surfaces arm status explicitly and lets fusion fail closed. |
| Arm naming | Exported string constants `ArmVector = "vector"`, `ArmFTS = "fts"` | Plain strings keep map keying trivial and let callers introduce arms later without a typed-constant registry. |
| Reranker interface | Unchanged | `Reranker.Rerank(ctx, query, []SearchHit)` already flows the candidate list. Adding `Provenance` to `SearchHit` is the pass-through; no signature change. |
| Debug payload | Deferred to a follow-up | A first-class `SearchResult` return type is a bigger API surface than this spec warrants. Pluggable fusion already unlocks debugging via a custom `FusionStrategy` that captures arms; a richer debug surface can be added once real consumer needs concretize. |
| Out-of-band fusion impls | Out of scope | Blended RAG weighting, DAT dynamic alpha, learned fusion, etc. are consumer code. Shipping them in the kernel would reintroduce the coupling #17 is trying to break. |

## Architecture

### New types in the `index` package

```go
// Arm name constants. Custom strategies may introduce additional arms.
const (
    ArmVector = "vector"
    ArmFTS    = "fts"
)

// RetrievalArm is one candidate list from one retrieval path, ordered by the
// arm's own ranking. Hits[i].Score is the arm-native score (cosine distance
// derivative for vector, negative bm25-equivalent for FTS). Fusion strategies
// may use rank (slice index) or score or both.
//
// Available and Err let fusion distinguish three otherwise identical-looking
// states:
//   - Available=true, Err=nil, len(Hits)==0      -> arm ran, no matches.
//   - Available=false, Err=nil                   -> arm unavailable on this
//                                                    snapshot (e.g. a legacy
//                                                    snapshot without
//                                                    fts_chunks). Hits must
//                                                    be empty.
//   - Available=false, Err!=nil                  -> arm failed. Hits must be
//                                                    empty. Strategies may
//                                                    choose to propagate, log,
//                                                    or fail closed.
//
// Available=true with Err!=nil is invalid and fusion implementations should
// return an error.
type RetrievalArm struct {
    Name      string
    Hits      []SearchHit
    Available bool
    Err       error
}

// FusionStrategy combines one or more RetrievalArms into a single ranked list,
// truncated to limit. Implementations must be deterministic and must attach
// HitProvenance to each returned hit covering every arm that contributed.
// The returned SearchHit.Score is the fusion-native score, except on paths
// where a strategy opts into preserving arm-native scores (see
// RRFFusion.PreserveSingleArmScore).
//
// Fuse returns an error when inputs are malformed (e.g. Available=true with
// Err!=nil, or a nil arm name), or when the strategy fails closed on an
// upstream arm error. Callers treat errors the same way as a retrieval
// failure. Strategies that want to tolerate partial-arm failures must do so
// internally and return a nil error.
type FusionStrategy interface {
    Fuse(arms []RetrievalArm, limit int) ([]SearchHit, error)
}

// ArmEvidence is one arm's contribution to a fused hit.
type ArmEvidence struct {
    Rank  int     // zero-based rank within the arm
    Score float64 // arm-native score at the time the arm returned the hit
}

// HitProvenance records which arms found a fused hit. The map is keyed by arm
// name. Arms that did not return the hit are absent from the map, so absence
// (rather than a sentinel Rank value) is the signal that an arm did not
// contribute.
type HitProvenance struct {
    Arms map[string]ArmEvidence
}

// RRFFusion is the default fusion strategy. K controls the RRF constant;
// K <= 0 is treated as K = 60 for backward compatibility with the pre-#17
// mergeRRF helper.
//
// PreserveSingleArmScore controls what happens when exactly one arm is
// available-and-non-empty. When true (the default, matching pre-#17
// behavior), RRFFusion returns that arm's hits in arm-native order with
// their arm-native Score preserved and HitProvenance attached. When false,
// it returns the same hits in the same order but rewrites Score to the
// RRF-derived 1/(K+rank+1). Callers that want numerically uniform fused
// scores across all paths opt in by setting this to false.
type RRFFusion struct {
    K                      int
    PreserveSingleArmScore bool
}

func (r RRFFusion) Fuse(arms []RetrievalArm, limit int) ([]SearchHit, error) { /* ... */ }

// DefaultFusion returns the fusion strategy used when SearchQuery.Fusion is
// nil. It is byte-identical to pre-#17 Search behavior on every path:
// - Both arms non-empty: standard RRF with source-count and best-rank tie
//   breaks.
// - Only vector arm non-empty (legacy snapshot, zero FTS hits, or FTS
//   disabled): vector hits returned in arm order with cosine-derived Score
//   preserved, matching today's rerankCandidates(..., vectorHits, ...)
//   bypass.
// - Only FTS arm non-empty: FTS hits in arm order with bm25-derived Score
//   preserved.
// - Both arms empty or unavailable: empty result.
func DefaultFusion() FusionStrategy {
    return RRFFusion{K: 60, PreserveSingleArmScore: true}
}
```

### `SearchHit` extension

```go
type SearchHit struct {
    ChunkID   int64
    Ref       string
    Kind      string
    Title     string
    SourceRef string
    Heading   string
    Content   string
    Metadata  map[string]string
    Score     float64

    // Provenance is populated by fusion. Nil on paths that skip fusion
    // entirely (e.g. SearchVector, or searchFTS called in isolation by tests).
    Provenance *HitProvenance
}
```

### Query options

```go
type SearchQuery struct {
    // ... existing fields unchanged ...

    // Fusion optionally overrides the hybrid fusion strategy. Nil uses
    // DefaultFusion() (equivalent to RRFFusion{K: 60, PreserveSingleArmScore: true}),
    // which preserves pre-#17 ordering on every path and pre-#17 Score on
    // every path except the vector-empty + FTS-non-empty case described
    // in the backward-compatibility section.
    Fusion FusionStrategy

    Reranker Reranker
}

type SnapshotSearchQuery struct {
    // ... existing fields unchanged ...
    Fusion   FusionStrategy
    Reranker Reranker
}
```

### `Snapshot.Search` pipeline

Before:

```
vectorHits := searchVectorCandidates(...)
ftsHits    := searchFTS(...)
if len(ftsHits) == 0 {
    return rerankCandidates(..., vectorHits, reranker)
}
fused := mergeRRF(vectorHits, ftsHits, candidateCount)
return rerankCandidates(..., fused, reranker)
```

After:

```
vectorHits, vecErr := searchVectorCandidates(...)
if vecErr != nil { return nil, vecErr }

ftsHits, ftsErr := searchFTS(...)  // existing short-circuit when !s.hasFTS
                                   // stays; returns (nil, nil) in that case.

arms := []RetrievalArm{
    {
        Name:      ArmVector,
        Hits:      vectorHits,
        Available: true,
    },
    {
        Name:      ArmFTS,
        Hits:      ftsHits,
        Available: s.hasFTS && ftsErr == nil,
        Err:       ftsErr,
    },
}
strategy := query.Fusion
if strategy == nil {
    strategy = DefaultFusion()
}
fused, err := strategy.Fuse(arms, candidateCount)
if err != nil {
    return nil, fmt.Errorf("fuse candidates: %w", err)
}
return rerankCandidates(..., fused, reranker)
```

The "skip fusion when FTS is empty" shortcut is absorbed into
`RRFFusion.PreserveSingleArmScore=true`: with a single available-and-non-empty
arm, `DefaultFusion()` returns that arm's hits in arm order with arm-native
`Score` preserved, matching today's `rerankCandidates(..., vectorHits, ...)`
bypass exactly. `Provenance` is still attached so rerankers can observe arm
context.

`searchFTS` keeps its existing `!s.hasFTS` short-circuit returning
`(nil, nil)`; the pipeline translates that to `Available=false` on the FTS
arm so custom strategies can distinguish a legacy snapshot from a zero-hit
FTS result. The FTS "no such table" fallback logic in `searchFTS` is left
unchanged.

### RRF implementation detail

`RRFFusion.Fuse`:

1. Walk each arm in order; for each hit, accumulate RRF score `1 / (K + rank + 1)`,
   increment the arm-source count, track best rank across arms, and stash
   arm-native rank and score in an `ArmEvidence` keyed by arm name.
2. Sort by (fused score desc, source count desc, best rank asc) — same tie-break
   rules as today, generalized to N arms.
3. Truncate to limit.
4. Assign fused `Score`, attach `Provenance`, return.

The existing private `mergeRRF` is removed; tests that referenced it shift to
the `RRFFusion{}` public surface. The current `index/hybrid.go` file becomes the
new home for `RRFFusion` and its helpers.

## Backward compatibility

- `Snapshot.Search` and `Search` keep the same signatures.
- Default behavior is byte-identical on every path, including:
  - Both arms non-empty: same RRF math, same tie-break order as pre-#17.
  - Legacy snapshot without `fts_chunks` (`hasFTS == false`): vector-only
    hits flow through `DefaultFusion` with arm-native cosine-derived `Score`
    preserved. Identical scores and order to pre-#17.
  - FTS arm runs and returns zero hits: same as above. The prior bypass is
    reproduced by `PreserveSingleArmScore: true`.
  - Vector arm returns zero hits, FTS non-empty: FTS hits in arm order with
    their bm25-derived `Score` preserved (previously this path already fused
    via `mergeRRF`; `DefaultFusion` now preserves arm-native score instead,
    which is a small behavioral tightening but keeps ordering identical and
    avoids introducing RRF scores to code paths that never saw them pre-#17).
  - Both arms empty or unavailable: empty result, no error.
- `SearchHit.Provenance` is populated on every path that flows through
  `DefaultFusion`; it remains nil on `SearchVector` and the `searchFTS` test
  helpers because those do not fuse. Callers that ignore the field are
  unaffected.
- No schema change. No migration. No new dependency.
- Behavioral tightening noted: the vector-only-arm-returns-zero-but-FTS-non-
  empty case now preserves FTS bm25 score rather than rewriting to RRF,
  because `DefaultFusion` treats it as a single-arm result. This is the
  only score-math change vs pre-#17; ordering is unchanged and the previous
  behavior required the FTS arm to win the whole result set anyway. Callers
  relying on stable numeric scores on that path should read `Score` (now the
  arm-native FTS score) or `Provenance.Arms[ArmFTS].Score`.

## Acceptance criteria

- [ ] `FusionStrategy`, `RetrievalArm`, `HitProvenance`, `ArmEvidence`, `RRFFusion`
      exported in the `index` package; `ArmVector` and `ArmFTS` constants
      exported; `DefaultFusion()` helper exported.
- [ ] `SearchQuery.Fusion` and `SnapshotSearchQuery.Fusion` wired through to
      `Snapshot.Search`; nil falls back to `DefaultFusion()`.
- [ ] `Fuse` returns `([]SearchHit, error)`; `Snapshot.Search` wraps fusion
      errors as `fuse candidates: %w` and returns them to the caller.
- [ ] `RetrievalArm` carries `Available bool` and `Err error` with the
      documented state semantics; `Snapshot.Search` populates them correctly
      for both the legacy-no-`fts_chunks` path and the live-FTS path.
- [ ] `SearchHit.Provenance` populated on every hit returned through
      `DefaultFusion`, covering every arm that contributed; nil on non-fusion
      paths (`SearchVector`, direct `searchFTS` callers).
- [ ] Existing fusion tests (`TestSnapshotSearchHybridFusion`, the RRF test at
      `index/index_test.go:846`) continue to pass unchanged against the default
      strategy — same scores, same order.
- [ ] New regression test: legacy snapshot with `hasFTS=false` returns
      byte-identical hits and scores to a pre-#17 snapshot of the same data.
      FTS `RetrievalArm.Available` is false; custom fusion receives it as such.
- [ ] New regression test: query whose FTS arm runs and returns zero hits
      returns byte-identical hits and scores to today's vector-only bypass.
      FTS `RetrievalArm.Available` is true, `Hits` is empty.
- [ ] New regression test: callers that read `SearchHit.Score` as a cosine
      derivative on vector-only paths continue to see cosine-derived scores
      under `DefaultFusion()`.
- [ ] New test: `RRFFusion{PreserveSingleArmScore: false}` rewrites single-arm
      hits to RRF-derived scores, demonstrating the opt-out knob works.
- [ ] New test: custom `FusionStrategy` receives both arms with expected
      lengths, availability flags, and arm-native scores, and the returned
      hits flow to the reranker with the strategy's `Score`.
- [ ] New test: `SearchHit.Provenance` from the default path includes the
      expected `Rank` and `Score` per arm for a hit that appears in both arms,
      and a hit that appears in only one arm.
- [ ] New test: a `FusionStrategy` that returns a non-nil error surfaces
      through `Snapshot.Search` as a wrapped error and does not produce
      partial results.
- [ ] Reranker test confirms `Provenance` is visible inside `Rerank`.
- [ ] `go test ./...` green; `make analyze` clean.

## Out of scope (explicit)

- New fusion strategies beyond RRF (Blended RAG weighted fusion, DAT dynamic
  alpha, learned fusion) — consumer code.
- New retrieval arms (ColBERT-style late-interaction, SPLATE sparse late
  interaction) — requires a separate design around arm production and cost.
- ANN vector arm — tracked in #55.
- First-class debug-result return type from `Search` — deferred; callers can
  capture arms with a custom `FusionStrategy` in the meantime.
- Metadata filtering, per-arm `Kinds` overrides, or other query-shape changes.

## Files touched (expected)

- `index/index.go` — add `FusionStrategy`, `RetrievalArm`, `HitProvenance`,
  `ArmEvidence`, arm-name constants; extend `SearchHit`, `SearchQuery`,
  `SnapshotSearchQuery`.
- `index/hybrid.go` — `RRFFusion` implementation; `mergeRRF` removed.
- `index/snapshot.go` — pipeline change in `Snapshot.Search`.
- `index/index_test.go` or a new `index/fusion_test.go` — provenance and
  custom-strategy coverage; migrate the RRF unit test to the public surface.
