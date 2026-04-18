// Package embed defines embedder interfaces and a deterministic test fixture.
package embed

import "context"

// Embedder generates embeddings for stored records and live queries.
type Embedder interface {
	Fingerprint() string
	Dimension(ctx context.Context) (int, error)
	EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error)
	EmbedQueries(ctx context.Context, texts []string) ([][]float64, error)
}

// ContextualEmbedder extends Embedder with an optional chunk-aware entry
// point that sees the full source document alongside the chunk texts,
// enabling late-chunking or other context-aware strategies. The
// embedding of Embedder makes the relationship explicit in the type
// system: every ContextualEmbedder is an Embedder, and the index's
// runtime type assertion from an Embedder value to a
// ContextualEmbedder is a superset upgrade rather than a separate
// disjoint implementation.
type ContextualEmbedder interface {
	Embedder
	EmbedDocumentChunks(ctx context.Context, fullDoc string, chunks []string) ([][]float64, error)
}
