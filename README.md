# Stroma

Stroma is a neutral corpus and indexing substrate.

It owns the lowest-level operations needed to ingest text artifacts, chunk them, embed them, persist them in SQLite plus `sqlite-vec`, and retrieve semantically close sections. Callers consume Stroma through its APIs and treat the SQLite snapshot as an opaque local artifact. It does not own governance, specifications, compliance, drift analysis, MCP, or CLI workflows.

## Scope

Stroma is for products that need a reusable text corpus layer with:

- canonical records with deterministic content fingerprints
- pluggable chunking strategies (`chunk.Policy` — `MarkdownPolicy` default, `KindRouterPolicy` for per-record-kind dispatch, `LateChunkPolicy` for parent/leaf hierarchy)
- pluggable embedders (`Embedder` / `ContextualEmbedder`) with a deterministic fixture and an OpenAI-compatible HTTP embedder
- hybrid retrieval: dense vector + FTS5, fused via a pluggable `FusionStrategy` (`RRFFusion` by default) with per-arm provenance surfaced to downstream rerankers
- quantization knobs: `float32` (default), `int8` (4× smaller), `binary` (32× smaller via 1-bit sign packing with a full-precision rescore pass)
- optional Matryoshka prefilter at a truncated dimension with full-dim cosine rescore (`SearchParams.SearchDimension`)
- atomic rebuilds and incremental `Update` with embedding reuse at the section level, chaining schema migrations v2 → v3 → v4 → v5 in one transaction

Stroma is not for:

- spec governance
- source discovery or repository scanning
- code compliance or doc drift analysis
- product-specific adapters and transports

## Packages

- `corpus` — canonical record model, `NewRecord` helper, `Normalize`, deterministic `Fingerprint`
- `chunk` — `Policy` interface with `MarkdownPolicy`, `KindRouterPolicy`, `LateChunkPolicy`; `MarkdownWithOptions` returns `ErrTooManySections` when a body exceeds the DoS cap
- `embed` — `Embedder` and `ContextualEmbedder` interfaces; deterministic `Fixture`; OpenAI-compatible HTTP embedder with `MaxBatchSize` batching, deadline scaling across batches, and `APIToken` redaction in `String`/`GoString`/`LogValue`
- `store` — SQLite readiness probes, `sqlite-vec` readiness, quantization blob helpers (`QuantizationFloat32` / `QuantizationInt8` / `QuantizationBinary`)
- `index` — atomic `Rebuild` with embedding reuse, incremental `Update`, long-lived `Snapshot` readers, `Stats`, hybrid `Search` with provenance, `ExpandContext` for parent/neighbor walks

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
