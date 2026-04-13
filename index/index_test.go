package index

import (
	"context"
	"strings"
	"testing"

	"github.com/dusk-network/stroma/corpus"
	"github.com/dusk-network/stroma/embed"
)

func TestRebuildAndReadStats(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{
		{
			Ref:        "sync-guide",
			Kind:       "guide",
			Title:      "Sync Guide",
			SourceRef:  "file://docs/sync-guide.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Overview\n\nBackground workers process items in batches.\n\n## Scheduling\n\nRun every hour.",
		},
		{
			Ref:        "status-note",
			Kind:       "note",
			Title:      "Status Note",
			SourceRef:  "file://docs/status.txt",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Health checks run every minute.",
		},
	}

	result, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     path,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}
	if result.RecordCount != 2 {
		t.Fatalf("RecordCount = %d, want 2", result.RecordCount)
	}
	if result.ChunkCount != 3 {
		t.Fatalf("ChunkCount = %d, want 3", result.ChunkCount)
	}

	stats, err := ReadStats(context.Background(), path)
	if err != nil {
		t.Fatalf("ReadStats() error = %v", err)
	}
	if stats.RecordCount != 2 {
		t.Fatalf("stats.RecordCount = %d, want 2", stats.RecordCount)
	}
	if stats.ChunkCount != 3 {
		t.Fatalf("stats.ChunkCount = %d, want 3", stats.ChunkCount)
	}
	if stats.KindCounts["guide"] != 1 || stats.KindCounts["note"] != 1 {
		t.Fatalf("KindCounts = %#v, want guide=1 note=1", stats.KindCounts)
	}
	if stats.EmbedderFingerprint != fixture.Fingerprint() {
		t.Fatalf("EmbedderFingerprint = %q, want %q", stats.EmbedderFingerprint, fixture.Fingerprint())
	}
}

func TestSearchReturnsClosestHit(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{
		{
			Ref:        "sync-guide",
			Kind:       "guide",
			Title:      "Sync Guide",
			SourceRef:  "file://docs/sync-guide.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Overview\n\nBackground workers process items in batches.\n\n## Scheduling\n\nRun every hour.",
		},
		{
			Ref:        "status-note",
			Kind:       "note",
			Title:      "Status Note",
			SourceRef:  "file://docs/status.txt",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Health checks run every minute.",
		},
	}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	hits, err := Search(context.Background(), SearchQuery{
		Path:     path,
		Text:     "background workers batches",
		Limit:    3,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("Search() returned no hits")
	}
	if hits[0].Ref != "sync-guide" {
		t.Fatalf("first hit ref = %q, want sync-guide", hits[0].Ref)
	}
}

func TestSearchHybridBoostsExactMatch(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{
		{
			Ref:        "error-codes",
			Kind:       "guide",
			Title:      "Error Codes",
			SourceRef:  "file://docs/errors.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Error Codes\n\nERR_TIMEOUT_42 occurs when the background worker exceeds the deadline.",
		},
		{
			Ref:        "sync-guide",
			Kind:       "guide",
			Title:      "Sync Guide",
			SourceRef:  "file://docs/sync.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Sync\n\nBackground workers process items in batches every hour.",
		},
	}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	// Search for an underscore-delimited identifier — this must match via
	// FTS because the hash-bigram fixture embedder has no special affinity
	// for this token.
	hits, err := Search(context.Background(), SearchQuery{
		Path:     path,
		Text:     "ERR_TIMEOUT_42",
		Limit:    5,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("Search() returned no hits")
	}
	if hits[0].Ref != "error-codes" {
		t.Fatalf("first hit ref = %q, want error-codes (exact match via FTS)", hits[0].Ref)
	}
}

func TestRebuildWithChunkOptions(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	// Build a record with a section > 10 words.
	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{
		{
			Ref:        "long-doc",
			Kind:       "guide",
			Title:      "Long Doc",
			SourceRef:  "file://docs/long.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Section\n\nalpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango",
		},
	}

	result, err := Rebuild(context.Background(), records, BuildOptions{
		Path:           path,
		Embedder:       fixture,
		MaxChunkTokens: 8,
	})
	if err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}
	// 20 words at 8 per chunk → at least 3 chunks.
	if result.ChunkCount < 3 {
		t.Fatalf("ChunkCount = %d, want >= 3 with MaxChunkTokens=8", result.ChunkCount)
	}
}

func TestRebuildWithInt8Quantization(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{
		{
			Ref:        "alpha",
			Kind:       "artifact",
			Title:      "Alpha",
			SourceRef:  "file://alpha.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "Burst handling lives here.",
		},
	}

	result, err := Rebuild(context.Background(), records, BuildOptions{
		Path:         path,
		Embedder:     fixture,
		Quantization: "int8",
	})
	if err != nil {
		t.Fatalf("Rebuild(int8) error = %v", err)
	}
	if result.RecordCount != 1 {
		t.Fatalf("RecordCount = %d, want 1", result.RecordCount)
	}

	// Search should still work with int8.
	hits, err := Search(context.Background(), SearchQuery{
		Path:     path,
		Text:     "burst handling",
		Limit:    3,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search(int8) error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("Search(int8) returned no hits")
	}
	if hits[0].Ref != "alpha" {
		t.Fatalf("first hit ref = %q, want alpha", hits[0].Ref)
	}
}

func TestRebuildInt8SectionsWithEmbeddings(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{
		{
			Ref:        "alpha",
			Kind:       "artifact",
			Title:      "Alpha",
			SourceRef:  "file://alpha.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "Burst handling lives here.",
		},
	}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:         path,
		Embedder:     fixture,
		Quantization: "int8",
	}); err != nil {
		t.Fatalf("Rebuild(int8) error = %v", err)
	}

	snapshot, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	defer func() { _ = snapshot.Close() }()

	sections, err := snapshot.Sections(context.Background(), SectionQuery{
		IncludeEmbeddings: true,
	})
	if err != nil {
		t.Fatalf("Sections(IncludeEmbeddings=true) error = %v", err)
	}
	if len(sections) == 0 {
		t.Fatal("Sections() returned no sections")
	}
	if len(sections[0].Embedding) != 16 {
		t.Fatalf("embedding dimension = %d, want 16", len(sections[0].Embedding))
	}
}

func TestRebuildInt8RejectsFloat32Reuse(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	dir := t.TempDir()
	pathA := dir + "/a.db"
	pathB := dir + "/b.db"

	records := []corpus.Record{
		{
			Ref:        "alpha",
			Kind:       "artifact",
			Title:      "Alpha",
			SourceRef:  "file://alpha.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "Burst handling lives here.",
		},
	}

	// Build with float32.
	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     pathA,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild(float32) error = %v", err)
	}

	// Rebuild with int8 reusing float32 — should not reuse embeddings.
	result, err := Rebuild(context.Background(), records, BuildOptions{
		Path:          pathB,
		ReuseFromPath: pathA,
		Embedder:      fixture,
		Quantization:  "int8",
	})
	if err != nil {
		t.Fatalf("Rebuild(int8, reuse float32) error = %v", err)
	}
	if result.ReusedChunkCount != 0 {
		t.Fatalf("ReusedChunkCount = %d, want 0 (quantization changed)", result.ReusedChunkCount)
	}
}

func TestRebuildInt8PreservesTopHitQuality(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	dir := t.TempDir()
	floatPath := dir + "/float32.db"
	int8Path := dir + "/int8.db"

	records := []corpus.Record{
		{
			Ref:        "burst-guide",
			Kind:       "guide",
			Title:      "Burst Guide",
			SourceRef:  "file://docs/burst.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Burst Handling\n\nBurst handling spreads queue pressure across workers and protects deadline-sensitive tasks.",
		},
		{
			Ref:        "chunk-guide",
			Kind:       "guide",
			Title:      "Chunk Guide",
			SourceRef:  "file://docs/chunk.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Chunking\n\nToken budgets and overlap keep retrieval chunks small while preserving nearby context.",
		},
		{
			Ref:        "error-guide",
			Kind:       "guide",
			Title:      "Error Guide",
			SourceRef:  "file://docs/error.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Error Codes\n\nERR_TIMEOUT_42 marks deadline overruns in the background worker.",
		},
	}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     floatPath,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild(float32) error = %v", err)
	}
	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:         int8Path,
		Embedder:     fixture,
		Quantization: "int8",
	}); err != nil {
		t.Fatalf("Rebuild(int8) error = %v", err)
	}

	int8Snapshot, err := OpenSnapshot(context.Background(), int8Path)
	if err != nil {
		t.Fatalf("OpenSnapshot(int8) error = %v", err)
	}
	defer func() { _ = int8Snapshot.Close() }()

	gotQuantization := readMetadataValueDefault(context.Background(), int8Snapshot.db, "quantization", "")
	if gotQuantization != "int8" {
		t.Fatalf("quantization metadata = %q, want int8", gotQuantization)
	}

	testCases := []struct {
		query string
		want  string
	}{
		{query: "burst worker deadlines", want: "burst-guide"},
		{query: "token overlap context", want: "chunk-guide"},
		{query: "ERR_TIMEOUT_42 deadline", want: "error-guide"},
	}

	for _, tc := range testCases {
		floatHits, err := Search(context.Background(), SearchQuery{
			Path:     floatPath,
			Text:     tc.query,
			Limit:    3,
			Embedder: fixture,
		})
		if err != nil {
			t.Fatalf("Search(float32, %q) error = %v", tc.query, err)
		}
		int8Hits, err := Search(context.Background(), SearchQuery{
			Path:     int8Path,
			Text:     tc.query,
			Limit:    3,
			Embedder: fixture,
		})
		if err != nil {
			t.Fatalf("Search(int8, %q) error = %v", tc.query, err)
		}
		if len(floatHits) == 0 || len(int8Hits) == 0 {
			t.Fatalf("empty search results for query %q", tc.query)
		}
		if floatHits[0].Ref != tc.want {
			t.Fatalf("float32 top hit for %q = %q, want %q", tc.query, floatHits[0].Ref, tc.want)
		}
		if int8Hits[0].Ref != tc.want {
			t.Fatalf("int8 top hit for %q = %q, want %q", tc.query, int8Hits[0].Ref, tc.want)
		}
	}
}

func TestSearchFTSRequiresAllTokens(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{
		{
			Ref:        "exact-match",
			Kind:       "guide",
			Title:      "Exact Match",
			SourceRef:  "file://docs/exact.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Exact\n\nERR_TIMEOUT_42 occurs when the background worker exceeds the deadline.",
		},
		{
			Ref:        "partial-only",
			Kind:       "guide",
			Title:      "Partial Only",
			SourceRef:  "file://docs/partial.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Partial\n\nThe number 42 appears frequently in test fixtures but has no timeout relevance.",
		},
	}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	hits, err := Search(context.Background(), SearchQuery{
		Path:     path,
		Text:     "ERR_TIMEOUT_42",
		Limit:    5,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("Search() returned no hits")
	}
	if hits[0].Ref != "exact-match" {
		t.Fatalf("first hit ref = %q, want exact-match (AND semantics should prevent partial-token boost)", hits[0].Ref)
	}
}

func TestSearchRejectsEmbedderMismatch(t *testing.T) {
	t.Parallel()

	buildEmbedder, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	queryEmbedder, err := embed.NewFixture("fixture-b", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{
		{
			Ref:        "alpha",
			Kind:       "artifact",
			Title:      "Alpha",
			SourceRef:  "file://alpha.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "Burst handling lives here.",
		},
	}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     path,
		Embedder: buildEmbedder,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	_, err = Search(context.Background(), SearchQuery{
		Path:     path,
		Text:     "burst handling",
		Limit:    3,
		Embedder: queryEmbedder,
	})
	if err == nil {
		t.Fatal("Search() error = nil, want embedder mismatch")
	}
	if !strings.Contains(err.Error(), "embedder fingerprint mismatch") {
		t.Fatalf("Search() error = %v, want fingerprint mismatch", err)
	}
}
