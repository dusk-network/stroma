package index

import (
	"context"
	"strings"
	"testing"

	"github.com/dusk-network/stroma/v2/corpus"
	"github.com/dusk-network/stroma/v2/embed"
)

// TestSnapshot_Search_RejectsLimitAboveMax reproduces issue #78: when a
// caller passes a SearchParams.Limit above the internal candidate cap,
// the old behaviour silently capped to 250 hits. The new contract is
// to reject with a clear error so pagination / reranking / export flows
// can rely on the public API.
func TestSnapshot_Search_RejectsLimitAboveMax(t *testing.T) {
	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture: %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        "limit-test-alpha",
		Kind:       testNoteKind,
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content lives here",
	}}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}
	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot: %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	cases := []struct {
		name  string
		limit int
	}{
		{"just_above_max", MaxSearchLimit + 1},
		{"thousand", 1000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := snap.Search(context.Background(), SnapshotSearchQuery{
				SearchParams: SearchParams{
					Text:     "alpha",
					Limit:    tc.limit,
					Embedder: fixture,
				},
			})
			if err == nil {
				t.Fatalf("expected error for Limit=%d, got nil", tc.limit)
			}
			if !strings.Contains(err.Error(), "MaxSearchLimit") {
				t.Errorf("error %q does not mention MaxSearchLimit", err.Error())
			}
		})
	}
}

// TestSnapshot_SearchVector_RejectsLimitAboveMax exercises the
// vector-only path (SearchVector) because it has an independent
// query.Limit validation site that must share the same contract.
func TestSnapshot_SearchVector_RejectsLimitAboveMax(t *testing.T) {
	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture: %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        "limit-test-vec",
		Kind:       testNoteKind,
		Title:      "Vec",
		SourceRef:  "file://vec.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "vec content lives here",
	}}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}
	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot: %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	vectors, err := fixture.EmbedQueries(context.Background(), []string{"vec"})
	if err != nil {
		t.Fatalf("EmbedQueries: %v", err)
	}

	_, err = snap.SearchVector(context.Background(), VectorSearchQuery{
		Embedding: vectors[0],
		Limit:     MaxSearchLimit + 1,
	})
	if err == nil {
		t.Fatalf("expected error for Limit=%d, got nil", MaxSearchLimit+1)
	}
	if !strings.Contains(err.Error(), "MaxSearchLimit") {
		t.Errorf("error %q does not mention MaxSearchLimit", err.Error())
	}
}
