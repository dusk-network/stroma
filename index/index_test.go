package index

import (
	"context"
	"errors"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/dusk-network/stroma/chunk"
	"github.com/dusk-network/stroma/corpus"
	"github.com/dusk-network/stroma/embed"
	"github.com/dusk-network/stroma/store"
)

const testSyncGuideRef = "sync-guide"

type reverseReranker struct {
	seenQuery      string
	seenCandidates []SearchHit
}

func (r *reverseReranker) Rerank(_ context.Context, query string, candidates []SearchHit) ([]SearchHit, error) {
	r.seenQuery = query
	r.seenCandidates = append([]SearchHit(nil), candidates...)

	reranked := append([]SearchHit(nil), candidates...)
	for i, j := 0, len(reranked)-1; i < j; i, j = i+1, j-1 {
		reranked[i], reranked[j] = reranked[j], reranked[i]
	}
	for i := range reranked {
		reranked[i].Score = float64(len(reranked) - i)
	}
	return reranked, nil
}

type errorReranker struct {
	err error
}

func (r errorReranker) Rerank(context.Context, string, []SearchHit) ([]SearchHit, error) {
	return nil, r.err
}

type contextualOnlyEmbedder struct {
	fixture         *embed.Fixture
	contextualCalls int
	documentCalls   int
	seenFullDocs    []string
	seenChunks      [][]string
}

func newContextualOnlyEmbedder(t *testing.T) *contextualOnlyEmbedder {
	t.Helper()

	fixture, err := embed.NewFixture("fixture-contextual", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	return &contextualOnlyEmbedder{fixture: fixture}
}

func (e *contextualOnlyEmbedder) Fingerprint() string {
	return "contextual|" + e.fixture.Fingerprint()
}

func (e *contextualOnlyEmbedder) Dimension(ctx context.Context) (int, error) {
	return e.fixture.Dimension(ctx)
}

func (e *contextualOnlyEmbedder) EmbedDocuments(context.Context, []string) ([][]float64, error) {
	e.documentCalls++
	return nil, errors.New("EmbedDocuments should not be called when ContextualEmbedder is available")
}

func (e *contextualOnlyEmbedder) EmbedQueries(ctx context.Context, texts []string) ([][]float64, error) {
	return e.fixture.EmbedQueries(ctx, texts)
}

func (e *contextualOnlyEmbedder) EmbedDocumentChunks(ctx context.Context, fullDoc string, chunks []string) ([][]float64, error) {
	e.contextualCalls++
	e.seenFullDocs = append(e.seenFullDocs, fullDoc)
	e.seenChunks = append(e.seenChunks, append([]string(nil), chunks...))
	return e.fixture.EmbedDocuments(ctx, chunks)
}

func TestRebuildAndReadStats(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{
		{
			Ref:        testSyncGuideRef,
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
			Ref:        testSyncGuideRef,
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
	if hits[0].Ref != testSyncGuideRef {
		t.Fatalf("first hit ref = %q, want %s", hits[0].Ref, testSyncGuideRef)
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
			Ref:        testSyncGuideRef,
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

func TestRebuildUsesContextualEmbedderWhenAvailable(t *testing.T) {
	t.Parallel()

	embedder := newContextualOnlyEmbedder(t)

	path := t.TempDir() + "/stroma.db"
	record := corpus.Record{
		Ref:        "long-note",
		Kind:       "note",
		Title:      "Long Note",
		SourceRef:  "file://docs/long-note.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima",
	}
	buildOpts := BuildOptions{
		Path:           path,
		Embedder:       embedder,
		MaxChunkTokens: 4,
	}

	result, err := Rebuild(context.Background(), []corpus.Record{record}, buildOpts)
	if err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	sections := sectionsForRecord(record, chunk.Options{
		MaxTokens:     buildOpts.MaxChunkTokens,
		OverlapTokens: buildOpts.ChunkOverlapTokens,
	})
	wantChunks := make([]string, 0, len(sections))
	for _, section := range sections {
		wantChunks = append(wantChunks, textForEmbedding(record.Title, section))
	}

	if result.EmbeddedChunkCount != len(wantChunks) {
		t.Fatalf("EmbeddedChunkCount = %d, want %d", result.EmbeddedChunkCount, len(wantChunks))
	}
	if embedder.contextualCalls != 1 {
		t.Fatalf("contextualCalls = %d, want 1", embedder.contextualCalls)
	}
	if embedder.documentCalls != 0 {
		t.Fatalf("documentCalls = %d, want 0", embedder.documentCalls)
	}
	if len(embedder.seenFullDocs) != 1 || embedder.seenFullDocs[0] != documentTextForEmbedding(record) {
		t.Fatalf("seenFullDocs = %v, want [%q]", embedder.seenFullDocs, documentTextForEmbedding(record))
	}
	if len(embedder.seenChunks) != 1 || !reflect.DeepEqual(embedder.seenChunks[0], wantChunks) {
		t.Fatalf("seenChunks = %#v, want %#v", embedder.seenChunks, wantChunks)
	}
}

func TestSearchAppliesRerankerBeforeLimitTruncation(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{
		{
			Ref:        testSyncGuideRef,
			Kind:       "guide",
			Title:      "Sync Guide",
			SourceRef:  "file://docs/sync-guide.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Overview\n\nBackground workers process items in batches every hour.",
		},
		{
			Ref:        "status-note",
			Kind:       "note",
			Title:      "Status Note",
			SourceRef:  "file://docs/status.txt",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Background workers publish heartbeat status every minute.",
		},
		{
			Ref:        "retry-guide",
			Kind:       "guide",
			Title:      "Retry Guide",
			SourceRef:  "file://docs/retry.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Retries\n\nBackground workers retry transient failures with exponential backoff.",
		},
	}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	baseline, err := Search(context.Background(), SearchQuery{
		Path:     path,
		Text:     "background workers",
		Limit:    3,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search(baseline) error = %v", err)
	}
	if len(baseline) != 3 {
		t.Fatalf("len(baseline) = %d, want 3", len(baseline))
	}

	reranker := &reverseReranker{}
	hits, err := Search(context.Background(), SearchQuery{
		Path:     path,
		Text:     "background workers",
		Limit:    2,
		Embedder: fixture,
		Reranker: reranker,
	})
	if err != nil {
		t.Fatalf("Search(reranker) error = %v", err)
	}
	if reranker.seenQuery != "background workers" {
		t.Fatalf("reranker query = %q, want background workers", reranker.seenQuery)
	}
	if len(reranker.seenCandidates) != 3 {
		t.Fatalf("reranker saw %d candidates, want 3 before final truncation", len(reranker.seenCandidates))
	}
	if len(hits) != 2 {
		t.Fatalf("len(reranked hits) = %d, want 2", len(hits))
	}
	if hits[0].Ref != reranker.seenCandidates[2].Ref || hits[1].Ref != reranker.seenCandidates[1].Ref {
		t.Fatalf(
			"reranked refs = [%q %q], want [%q %q] from reversed candidates",
			hits[0].Ref,
			hits[1].Ref,
			reranker.seenCandidates[2].Ref,
			reranker.seenCandidates[1].Ref,
		)
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

	gotQuantization, err := readMetadataValueOptional(context.Background(), int8Snapshot.db, "quantization", "")
	if err != nil {
		t.Fatalf("readMetadataValueOptional(quantization) error = %v", err)
	}
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

func TestSnapshotRejectsMalformedQuantizationMetadata(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{{
		Ref:        "alpha",
		Kind:       "artifact",
		Title:      "Alpha",
		SourceRef:  "file://alpha.md",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   "Burst handling lives here.",
	}}
	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:         path,
		Embedder:     fixture,
		Quantization: "int8",
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	// Corrupt the quantization metadata directly to simulate a stale or
	// tampered snapshot. Any non-supported value should be caught by
	// Snapshot's read paths instead of silently misdecoding int8 blobs as
	// float32.
	rwDB, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	if _, err := rwDB.ExecContext(context.Background(),
		`UPDATE metadata SET value = 'broken' WHERE key = 'quantization'`); err != nil {
		t.Fatalf("corrupt quantization metadata: %v", err)
	}
	if err := rwDB.Close(); err != nil {
		t.Fatalf("close rw db: %v", err)
	}

	snapshot, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	defer func() { _ = snapshot.Close() }()

	t.Run("Sections_with_embeddings", func(t *testing.T) {
		_, err := snapshot.Sections(context.Background(), SectionQuery{IncludeEmbeddings: true})
		if err == nil {
			t.Fatal("want error for malformed quantization, got nil")
		}
		if !strings.Contains(err.Error(), "quantization") {
			t.Errorf("error does not mention quantization: %v", err)
		}
	})

	t.Run("Search", func(t *testing.T) {
		_, err := snapshot.Search(context.Background(), SnapshotSearchQuery{
			Text:     "burst",
			Limit:    3,
			Embedder: fixture,
		})
		if err == nil {
			t.Fatal("want error for malformed quantization, got nil")
		}
		if !strings.Contains(err.Error(), "quantization") {
			t.Errorf("error does not mention quantization: %v", err)
		}
	})

	t.Run("SearchVector", func(t *testing.T) {
		_, err := snapshot.SearchVector(context.Background(), VectorSearchQuery{
			Embedding: make([]float64, 16),
			Limit:     3,
		})
		if err == nil {
			t.Fatal("want error for malformed quantization, got nil")
		}
		if !strings.Contains(err.Error(), "quantization") {
			t.Errorf("error does not mention quantization: %v", err)
		}
	})

	t.Run("Sections_without_embeddings_still_works", func(t *testing.T) {
		sections, err := snapshot.Sections(context.Background(), SectionQuery{})
		if err != nil {
			t.Fatalf("Sections(IncludeEmbeddings=false) should skip quantization; got %v", err)
		}
		if len(sections) == 0 {
			t.Fatal("expected at least one section")
		}
	})
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

func TestMergeRRFUsesFusedScores(t *testing.T) {
	t.Parallel()

	hits := mergeRRF(
		[]SearchHit{{ChunkID: 1, Ref: "vector-only", Score: 0.8}, {ChunkID: 2, Ref: "shared", Score: 0.7}},
		[]SearchHit{{ChunkID: 2, Ref: "shared", Score: -0.2}, {ChunkID: 3, Ref: "fts-only", Score: -0.1}},
		3,
	)
	if len(hits) != 3 {
		t.Fatalf("len(mergeRRF) = %d, want 3", len(hits))
	}
	for _, hit := range hits {
		if hit.Score <= 0 {
			t.Fatalf("hit %q has non-positive fused score %f", hit.Ref, hit.Score)
		}
	}
	if hits[0].Ref != "shared" {
		t.Fatalf("top fused hit = %q, want shared", hits[0].Ref)
	}
}

func TestReadMetadataValueOptionalPropagatesNonMissingErrors(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        "alpha",
		Kind:       "artifact",
		Title:      "Alpha",
		SourceRef:  "file://alpha.md",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   "Burst handling lives here.",
	}}, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	snapshot, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	defer func() { _ = snapshot.Close() }()

	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = readMetadataValueOptional(canceledCtx, snapshot.db, "quantization", store.QuantizationFloat32)
	if err == nil {
		t.Fatal("readMetadataValueOptional() error = nil, want propagated context error")
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

func TestSnapshotSearchPropagatesRerankerError(t *testing.T) {
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
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	snapshot, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	defer func() { _ = snapshot.Close() }()

	_, err = snapshot.Search(context.Background(), SnapshotSearchQuery{
		Text:     "burst handling",
		Limit:    3,
		Embedder: fixture,
		Reranker: errorReranker{err: errors.New("boom")},
	})
	if err == nil {
		t.Fatal("Snapshot.Search() error = nil, want reranker failure")
	}
	if !strings.Contains(err.Error(), "rerank candidates: boom") {
		t.Fatalf("Snapshot.Search() error = %v, want reranker context", err)
	}
}

func TestUpdateAddsRecord(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	initial := []corpus.Record{
		{
			Ref:        testSyncGuideRef,
			Kind:       "guide",
			Title:      "Sync Guide",
			SourceRef:  "file://docs/sync.md",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Background workers process items in batches.",
		},
	}
	added := corpus.Record{
		Ref:        "retry-guide",
		Kind:       "guide",
		Title:      "Retry Guide",
		SourceRef:  "file://docs/retry.md",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "Retry with exponential backoff after transient failures.",
	}

	if _, err := Rebuild(context.Background(), initial, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}
	initialStats, err := ReadStats(context.Background(), path)
	if err != nil {
		t.Fatalf("ReadStats(initial) error = %v", err)
	}

	result, err := Update(context.Background(), []corpus.Record{added}, nil, UpdateOptions{
		Path:     path,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Update() error = %v", err)
	}
	if result.UpsertedCount != 1 {
		t.Fatalf("UpsertedCount = %d, want 1", result.UpsertedCount)
	}
	if result.RecordCount != 2 {
		t.Fatalf("RecordCount = %d, want 2", result.RecordCount)
	}
	if result.ChunkCount != 2 {
		t.Fatalf("ChunkCount = %d, want 2", result.ChunkCount)
	}

	expectedRecords := append([]corpus.Record(nil), initial...)
	expectedRecords = append(expectedRecords, added)
	if result.ContentFingerprint != corpus.Fingerprint(expectedRecords) {
		t.Fatalf("ContentFingerprint = %q, want %q", result.ContentFingerprint, corpus.Fingerprint(expectedRecords))
	}

	stats, err := ReadStats(context.Background(), path)
	if err != nil {
		t.Fatalf("ReadStats() error = %v", err)
	}
	if stats.RecordCount != 2 || stats.ChunkCount != 2 {
		t.Fatalf("stats = %+v, want record_count=2 chunk_count=2", stats)
	}
	if stats.CreatedAt != initialStats.CreatedAt {
		t.Fatalf("CreatedAt = %q, want preserved %q", stats.CreatedAt, initialStats.CreatedAt)
	}

	hits, err := Search(context.Background(), SearchQuery{
		Path:     path,
		Text:     "exponential backoff",
		Limit:    3,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) == 0 || hits[0].Ref != "retry-guide" {
		t.Fatalf("top hit = %+v, want retry-guide", hits)
	}
}

func TestUpdateRemovesRecordWithoutEmbedder(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	initial := []corpus.Record{
		{
			Ref:        testSyncGuideRef,
			Kind:       "guide",
			Title:      "Sync Guide",
			SourceRef:  "file://docs/sync.md",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Background workers process items in batches.",
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

	if _, err := Rebuild(context.Background(), initial, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	result, err := Update(context.Background(), nil, []string{"status-note"}, UpdateOptions{
		Path: path,
	})
	if err != nil {
		t.Fatalf("Update(remove) error = %v", err)
	}
	if result.UpsertedCount != 0 {
		t.Fatalf("UpsertedCount = %d, want 0", result.UpsertedCount)
	}
	if result.RemovedCount != 1 {
		t.Fatalf("RemovedCount = %d, want 1", result.RemovedCount)
	}
	if result.RecordCount != 1 || result.ChunkCount != 1 {
		t.Fatalf("result = %+v, want record_count=1 chunk_count=1", result)
	}

	snapshot, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	defer func() { _ = snapshot.Close() }()

	records, err := snapshot.Records(context.Background(), RecordQuery{})
	if err != nil {
		t.Fatalf("Records() error = %v", err)
	}
	if len(records) != 1 || records[0].Ref != testSyncGuideRef {
		t.Fatalf("records = %+v, want only %s", records, testSyncGuideRef)
	}
}

func TestUpdateReplacesRecord(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	initial := corpus.Record{
		Ref:        testSyncGuideRef,
		Kind:       "guide",
		Title:      "Sync Guide",
		SourceRef:  "file://docs/sync.md",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   "# Overview\n\nBackground workers process items in batches.\n\n## Scheduling\n\nRun every hour.",
	}
	replacement := corpus.Record{
		Ref:        testSyncGuideRef,
		Kind:       "guide",
		Title:      "Sync Guide",
		SourceRef:  "file://docs/sync.md",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   "# Overview\n\nBackground workers process items in batches.\n\n## Scheduling\n\nRun every 30 minutes.",
	}

	if _, err := Rebuild(context.Background(), []corpus.Record{initial}, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	result, err := Update(context.Background(), []corpus.Record{replacement}, nil, UpdateOptions{
		Path:     path,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Update(replace) error = %v", err)
	}
	if result.UpsertedCount != 1 {
		t.Fatalf("UpsertedCount = %d, want 1", result.UpsertedCount)
	}
	if result.ReusedChunkCount != 1 {
		t.Fatalf("ReusedChunkCount = %d, want 1", result.ReusedChunkCount)
	}
	if result.EmbeddedChunkCount != 1 {
		t.Fatalf("EmbeddedChunkCount = %d, want 1", result.EmbeddedChunkCount)
	}

	snapshot, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	defer func() { _ = snapshot.Close() }()

	sections, err := snapshot.Sections(context.Background(), SectionQuery{})
	if err != nil {
		t.Fatalf("Sections() error = %v", err)
	}
	if len(sections) != 2 {
		t.Fatalf("len(sections) = %d, want 2", len(sections))
	}
	if !strings.Contains(sections[1].Content, "30 minutes") {
		t.Fatalf("updated section content = %q, want 30 minutes", sections[1].Content)
	}

	hits, err := Search(context.Background(), SearchQuery{
		Path:     path,
		Text:     "30 minutes",
		Limit:    3,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) == 0 || hits[0].Ref != testSyncGuideRef {
		t.Fatalf("top hit = %+v, want %s", hits, testSyncGuideRef)
	}
}

func TestUpdateMixedOperationsMatchEquivalentRebuild(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	dir := t.TempDir()
	updatePath := dir + "/update.db"
	rebuildPath := dir + "/rebuild.db"

	baseRecords := []corpus.Record{
		{
			Ref:        "alpha",
			Kind:       "guide",
			Title:      "Alpha",
			SourceRef:  "file://docs/alpha.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Overview\n\nAlpha handles queue admission.\n\n## Limits\n\nAlpha applies hourly quotas.",
		},
		{
			Ref:        "beta",
			Kind:       "guide",
			Title:      "Beta",
			SourceRef:  "file://docs/beta.md",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Beta performs background health checks.",
		},
	}
	alphaUpdated := corpus.Record{
		Ref:        "alpha",
		Kind:       "guide",
		Title:      "Alpha",
		SourceRef:  "file://docs/alpha.md",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   "# Overview\n\nAlpha handles queue admission.\n\n## Limits\n\nAlpha applies per-tenant quotas.",
	}
	gammaAdded := corpus.Record{
		Ref:        "gamma",
		Kind:       "note",
		Title:      "Gamma",
		SourceRef:  "file://docs/gamma.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "Gamma records retry budgets and backoff windows.",
	}
	finalRecords := []corpus.Record{alphaUpdated, gammaAdded}

	if _, err := Rebuild(context.Background(), baseRecords, BuildOptions{
		Path:     updatePath,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild(updatePath) error = %v", err)
	}
	if _, err := Update(context.Background(), []corpus.Record{alphaUpdated, gammaAdded}, []string{"beta"}, UpdateOptions{
		Path:     updatePath,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Update(mixed) error = %v", err)
	}

	if _, err := Rebuild(context.Background(), finalRecords, BuildOptions{
		Path:     rebuildPath,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild(rebuildPath) error = %v", err)
	}

	updateStats, err := ReadStats(context.Background(), updatePath)
	if err != nil {
		t.Fatalf("ReadStats(updatePath) error = %v", err)
	}
	rebuildStats, err := ReadStats(context.Background(), rebuildPath)
	if err != nil {
		t.Fatalf("ReadStats(rebuildPath) error = %v", err)
	}
	if updateStats.RecordCount != rebuildStats.RecordCount ||
		updateStats.ChunkCount != rebuildStats.ChunkCount ||
		updateStats.ContentFingerprint != rebuildStats.ContentFingerprint ||
		updateStats.EmbedderFingerprint != rebuildStats.EmbedderFingerprint {
		t.Fatalf("update stats = %+v rebuild stats = %+v, want matching core metadata", updateStats, rebuildStats)
	}

	updateRecords, updateSections := snapshotContents(t, updatePath)
	rebuildRecords, rebuildSections := snapshotContents(t, rebuildPath)
	if !reflect.DeepEqual(updateRecords, rebuildRecords) {
		t.Fatalf("updated records = %#v, want %#v", updateRecords, rebuildRecords)
	}
	if !reflect.DeepEqual(updateSections, rebuildSections) {
		t.Fatalf("updated sections = %#v, want %#v", updateSections, rebuildSections)
	}

	updateHits, err := Search(context.Background(), SearchQuery{
		Path:     updatePath,
		Text:     "retry budgets",
		Limit:    3,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search(updatePath) error = %v", err)
	}
	rebuildHits, err := Search(context.Background(), SearchQuery{
		Path:     rebuildPath,
		Text:     "retry budgets",
		Limit:    3,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search(rebuildPath) error = %v", err)
	}
	updateRefs := searchRefs(updateHits)
	rebuildRefs := searchRefs(rebuildHits)
	if !reflect.DeepEqual(updateRefs, rebuildRefs) {
		t.Fatalf("update search refs = %v, want %v", updateRefs, rebuildRefs)
	}
}

func TestRebuildStreamingReuseAcrossManyRecords(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	dir := t.TempDir()
	pathA := dir + "/a.db"
	pathB := dir + "/b.db"

	const (
		unchangedCount = 20
		editedCount    = 5
		addedCount     = 5
	)

	baseRecord := func(i int) corpus.Record {
		return corpus.Record{
			Ref:        "rec-" + strconv.Itoa(i),
			Kind:       "guide",
			Title:      "Record " + strconv.Itoa(i),
			SourceRef:  "file://docs/rec-" + strconv.Itoa(i) + ".md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText: "# Overview\n\nRecord " + strconv.Itoa(i) + " covers queue admission and limits.\n\n" +
				"## Limits\n\nRecord " + strconv.Itoa(i) + " enforces hourly quotas.",
		}
	}

	initial := make([]corpus.Record, 0, unchangedCount+editedCount)
	for i := 0; i < unchangedCount+editedCount; i++ {
		initial = append(initial, baseRecord(i))
	}

	if _, err := Rebuild(context.Background(), initial, BuildOptions{
		Path:     pathA,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild(A) error = %v", err)
	}

	// Build set B: keep the first `unchangedCount` records identical,
	// edit the second section on the next `editedCount`, and append
	// `addedCount` brand-new records.
	next := make([]corpus.Record, 0, unchangedCount+editedCount+addedCount)
	for i := 0; i < unchangedCount; i++ {
		next = append(next, baseRecord(i))
	}
	for i := unchangedCount; i < unchangedCount+editedCount; i++ {
		edited := baseRecord(i)
		edited.BodyText = "# Overview\n\nRecord " + strconv.Itoa(i) + " covers queue admission and limits.\n\n" +
			"## Limits\n\nRecord " + strconv.Itoa(i) + " enforces per-tenant quotas."
		next = append(next, edited)
	}
	for i := 0; i < addedCount; i++ {
		next = append(next, corpus.Record{
			Ref:        "new-" + strconv.Itoa(i),
			Kind:       "note",
			Title:      "Note " + strconv.Itoa(i),
			SourceRef:  "file://docs/new-" + strconv.Itoa(i) + ".txt",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Newly added note " + strconv.Itoa(i) + " records retry budgets.",
		})
	}

	result, err := Rebuild(context.Background(), next, BuildOptions{
		Path:          pathB,
		ReuseFromPath: pathA,
		Embedder:      fixture,
	})
	if err != nil {
		t.Fatalf("Rebuild(B, reuse A) error = %v", err)
	}

	// Chunks in B: unchanged (2 each) + edited (2 each) + added (1 each).
	wantChunks := unchangedCount*2 + editedCount*2 + addedCount
	if result.ChunkCount != wantChunks {
		t.Fatalf("ChunkCount = %d, want %d", result.ChunkCount, wantChunks)
	}
	// Reused chunks: both sections of each unchanged record plus the
	// unchanged "Overview" section on each edited record.
	wantReusedChunks := unchangedCount*2 + editedCount
	if result.ReusedChunkCount != wantReusedChunks {
		t.Fatalf("ReusedChunkCount = %d, want %d", result.ReusedChunkCount, wantReusedChunks)
	}
	// Embedded chunks: the rewritten "Limits" section on each edited
	// record plus the single section per added plaintext record.
	wantEmbedded := editedCount + addedCount
	if result.EmbeddedChunkCount != wantEmbedded {
		t.Fatalf("EmbeddedChunkCount = %d, want %d", result.EmbeddedChunkCount, wantEmbedded)
	}
	if result.ReusedRecordCount != unchangedCount {
		t.Fatalf("ReusedRecordCount = %d, want %d", result.ReusedRecordCount, unchangedCount)
	}
}

func TestRebuildBinaryQuantizationReturnsClosestHit(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{
		{
			Ref:        testSyncGuideRef,
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
		Path:         path,
		Embedder:     fixture,
		Quantization: "binary",
	})
	if err != nil {
		t.Fatalf("Rebuild(binary) error = %v", err)
	}
	if result.ChunkCount == 0 {
		t.Fatal("Rebuild(binary) ChunkCount = 0, want >0")
	}

	hits, err := Search(context.Background(), SearchQuery{
		Path:     path,
		Text:     "background workers batches",
		Limit:    3,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search(binary) error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("Search(binary) returned no hits")
	}
	if hits[0].Ref != testSyncGuideRef {
		t.Fatalf("first hit ref = %q, want %s (binary hamming prefilter + cosine rescore should still rank the topical guide first)",
			hits[0].Ref, testSyncGuideRef)
	}
}

func TestRebuildBinaryRejectsDimensionNotDivisibleByEight(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 15)
	if err != nil {
		t.Fatalf("NewFixture(15) error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        "alpha",
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:         path,
		Embedder:     fixture,
		Quantization: "binary",
	}); err == nil || !strings.Contains(err.Error(), "divisible by 8") {
		t.Fatalf("Rebuild(binary, dim=15) err = %v, want divisible-by-8 error", err)
	}
}

func TestRebuildBinaryRejectsCrossQuantizationReuse(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	dir := t.TempDir()
	floatPath := dir + "/float.db"
	binaryPath := dir + "/binary.db"

	records := []corpus.Record{{
		Ref:        "alpha",
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     floatPath,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild(float32) error = %v", err)
	}

	// Rebuild binary while reusing from a float32 snapshot — the
	// compatibility check must reject the reuse so we do not insert a
	// float32 blob into the bit-column vec0 table.
	result, err := Rebuild(context.Background(), records, BuildOptions{
		Path:          binaryPath,
		ReuseFromPath: floatPath,
		Embedder:      fixture,
		Quantization:  "binary",
	})
	if err != nil {
		t.Fatalf("Rebuild(binary, reuse float32) error = %v", err)
	}
	if result.ReusedChunkCount != 0 {
		t.Fatalf("ReusedChunkCount = %d, want 0 (quantization mismatch must disable reuse)",
			result.ReusedChunkCount)
	}
}

func TestRebuildBinaryReuseAcrossBinarySnapshots(t *testing.T) {
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
			Kind:       "guide",
			Title:      "Alpha",
			SourceRef:  "file://alpha.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Overview\n\nAlpha handles queue admission.\n\n## Limits\n\nAlpha applies hourly quotas.",
		},
		{
			Ref:        "beta",
			Kind:       "note",
			Title:      "Beta",
			SourceRef:  "file://beta.txt",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Beta performs background health checks.",
		},
	}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:         pathA,
		Embedder:     fixture,
		Quantization: "binary",
	}); err != nil {
		t.Fatalf("Rebuild(binary A) error = %v", err)
	}

	result, err := Rebuild(context.Background(), records, BuildOptions{
		Path:          pathB,
		ReuseFromPath: pathA,
		Embedder:      fixture,
		Quantization:  "binary",
	})
	if err != nil {
		t.Fatalf("Rebuild(binary B, reuse binary A) error = %v", err)
	}
	if result.ReusedChunkCount == 0 {
		t.Fatalf("ReusedChunkCount = 0, want all chunks reused across binary→binary rebuild")
	}
	if result.ReusedChunkCount != result.ChunkCount {
		t.Fatalf("ReusedChunkCount = %d, ChunkCount = %d, want equal", result.ReusedChunkCount, result.ChunkCount)
	}
}

func TestSnapshotSectionsBinaryIncludesFloatEmbeddings(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        "alpha",
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:         path,
		Embedder:     fixture,
		Quantization: "binary",
	}); err != nil {
		t.Fatalf("Rebuild(binary) error = %v", err)
	}

	snapshot, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	defer func() { _ = snapshot.Close() }()

	sections, err := snapshot.Sections(context.Background(), SectionQuery{IncludeEmbeddings: true})
	if err != nil {
		t.Fatalf("Sections(IncludeEmbeddings) error = %v", err)
	}
	if len(sections) == 0 {
		t.Fatal("Sections() returned no sections")
	}
	seenNonBinary := false
	for _, section := range sections {
		if len(section.Embedding) != 16 {
			t.Fatalf("section %s embedding len = %d, want 16", section.Ref, len(section.Embedding))
		}
		for _, v := range section.Embedding {
			if v != 1 && v != -1 {
				seenNonBinary = true
				break
			}
		}
	}
	if !seenNonBinary {
		t.Fatal("Sections(binary) returned only {-1, 1} values; expected float32 values from chunks_vec_full")
	}
}

func TestOpenSnapshotRejectsIncompleteBinaryCompanion(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        "alpha",
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:         path,
		Embedder:     fixture,
		Quantization: "binary",
	}); err != nil {
		t.Fatalf("Rebuild(binary) error = %v", err)
	}

	// Surgically drop a companion row behind stroma's back to simulate
	// the exact corruption mode Codex flagged: chunks_vec_full missing a
	// row that chunks still references. OpenSnapshot must refuse instead
	// of silently losing the chunk through the inner-join read path.
	db, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	if _, err := db.ExecContext(context.Background(), `DELETE FROM chunks_vec_full`); err != nil {
		t.Fatalf("DELETE chunks_vec_full error = %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("db.Close() error = %v", err)
	}

	_, err = OpenSnapshot(context.Background(), path)
	if err == nil {
		t.Fatal("OpenSnapshot(corrupt binary snapshot) err = nil, want completeness error")
	}
	if !strings.Contains(err.Error(), "chunks_vec_full") {
		t.Fatalf("OpenSnapshot err = %v, want reference to chunks_vec_full", err)
	}
}

func TestRebuildSelfReuseReplacesSnapshotInPlace(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	dir := t.TempDir()
	path := dir + "/index.db"

	records := []corpus.Record{
		{
			Ref:        "alpha",
			Kind:       "guide",
			Title:      "Alpha",
			SourceRef:  "file://docs/alpha.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Overview\n\nAlpha handles queue admission.\n\n## Limits\n\nAlpha applies hourly quotas.",
		},
		{
			Ref:        "beta",
			Kind:       "note",
			Title:      "Beta",
			SourceRef:  "file://docs/beta.txt",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Beta performs background health checks.",
		},
	}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild(initial) error = %v", err)
	}

	// Rebuilding into the same path while reusing from it must close the
	// read-only reuse handle before replaceFile renames the staged file
	// into place, otherwise the rebuild races an open handle against the
	// atomic swap (fatal on Windows).
	result, err := Rebuild(context.Background(), records, BuildOptions{
		Path:          path,
		ReuseFromPath: path,
		Embedder:      fixture,
	})
	if err != nil {
		t.Fatalf("Rebuild(self-reuse) error = %v", err)
	}
	if result.ReusedRecordCount != len(records) {
		t.Fatalf("ReusedRecordCount = %d, want %d (all records carry over on self-reuse)",
			result.ReusedRecordCount, len(records))
	}
	if result.EmbeddedChunkCount != 0 {
		t.Fatalf("EmbeddedChunkCount = %d, want 0 (all chunks reused on self-reuse)",
			result.EmbeddedChunkCount)
	}
	if result.ReusedChunkCount != result.ChunkCount {
		t.Fatalf("ReusedChunkCount = %d, ChunkCount = %d, want equal on self-reuse",
			result.ReusedChunkCount, result.ChunkCount)
	}
}

type comparableSection struct {
	Ref       string
	Kind      string
	Title     string
	SourceRef string
	Heading   string
	Content   string
	Metadata  map[string]string
}

func snapshotContents(t *testing.T, path string) ([]corpus.Record, []comparableSection) {
	t.Helper()

	snapshot, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot(%q) error = %v", path, err)
	}
	defer func() { _ = snapshot.Close() }()

	records, err := snapshot.Records(context.Background(), RecordQuery{})
	if err != nil {
		t.Fatalf("Records(%q) error = %v", path, err)
	}
	sections, err := snapshot.Sections(context.Background(), SectionQuery{})
	if err != nil {
		t.Fatalf("Sections(%q) error = %v", path, err)
	}

	resultSections := make([]comparableSection, 0, len(sections))
	for i := range sections {
		section := &sections[i]
		resultSections = append(resultSections, comparableSection{
			Ref:       section.Ref,
			Kind:      section.Kind,
			Title:     section.Title,
			SourceRef: section.SourceRef,
			Heading:   section.Heading,
			Content:   section.Content,
			Metadata:  section.Metadata,
		})
	}
	return records, resultSections
}

func searchRefs(hits []SearchHit) []string {
	result := make([]string, 0, len(hits))
	for _, hit := range hits {
		result = append(result, hit.Ref)
	}
	return result
}
