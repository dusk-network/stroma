# Stroma

Stroma is a neutral corpus and indexing substrate.

It owns the lowest-level operations needed to ingest text artifacts, chunk them, embed them, persist them in SQLite plus `sqlite-vec`, and retrieve semantically close sections. Callers consume Stroma through its APIs and treat the SQLite snapshot as an opaque local artifact. It does not own governance, specifications, compliance, drift analysis, MCP, or CLI workflows.

## Scope

Stroma is for products that need a reusable text corpus layer with:

- canonical records
- deterministic content fingerprints
- heading-aware Markdown chunking
- pluggable embedders
- SQLite-backed semantic retrieval

Stroma is not for:

- spec governance
- source discovery or repository scanning
- code compliance or doc drift analysis
- product-specific adapters and transports

## Packages

- `corpus`: canonical record model and deterministic fingerprints
- `chunk`: heading-aware Markdown chunking
- `embed`: embedding interface and deterministic fixture embedder
- `store`: SQLite readiness and vector blob helpers
- `index`: index rebuild with embedding reuse, opened snapshot readers, stats, and semantic search

## Example

```go
ctx := context.Background()

records := []corpus.Record{
	{
		Ref:        "widget-overview",
		Kind:       "note",
		Title:      "Widget Overview",
		SourceRef:  "file://docs/widget-overview.md",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   "# Overview\n\nWidgets are synchronized in batches.",
	},
}

fixture, err := embed.NewFixture("fixture-demo", 16)
if err != nil {
	log.Fatal(err)
}

_, err = index.Rebuild(ctx, records, index.BuildOptions{
	Path:     "stroma.db",
	Embedder: fixture,
})
if err != nil {
	log.Fatal(err)
}

hits, err := index.Search(ctx, index.SearchQuery{
	Path:     "stroma.db",
	Text:     "synchronized batches",
	Limit:    5,
	Embedder: fixture,
})
if err != nil {
	log.Fatal(err)
}

fmt.Println(hits[0].Ref)
```

## Status

This repo currently provides the extracted micro-kernel only. Higher-order products should consume it rather than re-embedding their own indexing substrate.
