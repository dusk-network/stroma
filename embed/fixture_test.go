package embed

import (
	"context"
	"math"
	"reflect"
	"testing"
)

func TestNewFixtureRejectsNonPositiveDimensions(t *testing.T) {
	t.Parallel()

	if _, err := NewFixture("fixture", 0); err == nil {
		t.Fatal("NewFixture() error = nil, want invalid dimensions")
	}
}

func TestFixtureEmbeddingsAreDeterministic(t *testing.T) {
	t.Parallel()

	embedder, err := NewFixture("fixture", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	left, err := embedder.EmbedDocuments(context.Background(), []string{"burst handling"})
	if err != nil {
		t.Fatalf("EmbedDocuments() error = %v", err)
	}
	right, err := embedder.EmbedQueries(context.Background(), []string{"burst handling"})
	if err != nil {
		t.Fatalf("EmbedQueries() error = %v", err)
	}
	if !reflect.DeepEqual(left, right) {
		t.Fatalf("fixture embedder returned inconsistent vectors: %v != %v", left, right)
	}
}

func TestFixtureEmbeddingsAreNormalized(t *testing.T) {
	t.Parallel()

	embedder, err := NewFixture("fixture", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	vectors, err := embedder.EmbedDocuments(context.Background(), []string{"background workers batch records"})
	if err != nil {
		t.Fatalf("EmbedDocuments() error = %v", err)
	}
	if len(vectors) != 1 {
		t.Fatalf("vector count = %d, want 1", len(vectors))
	}
	var norm float64
	for _, value := range vectors[0] {
		norm += value * value
	}
	if math.Abs(norm-1) > 0.000001 {
		t.Fatalf("vector norm = %f, want 1", norm)
	}
}
