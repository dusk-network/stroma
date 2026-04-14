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

// ContextualEmbedder optionally embeds chunk texts with access to the full
// source document, enabling late-chunking or other context-aware strategies
// without changing the base Embedder contract.
type ContextualEmbedder interface {
	EmbedDocumentChunks(ctx context.Context, fullDoc string, chunks []string) ([][]float64, error)
}
