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

const (
	testSyncGuideRef  = "sync-guide"
	testAlphaRef      = "alpha"
	testNoteKind      = "note"
	testTargetNoteRef = "target-note"

	// testLegacyV3HashSentinel is the bogus content_hash migration tests
	// write into every row to simulate an old-algorithm pre-1.0 v3
	// snapshot. Any row still carrying this value after migration means
	// the migration skipped it.
	testLegacyV3HashSentinel = "legacy-v3-hash"
)

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
	expectedFingerprint, err := corpus.Fingerprint(expectedRecords)
	if err != nil {
		t.Fatalf("corpus.Fingerprint(expected) error = %v", err)
	}
	if result.ContentFingerprint != expectedFingerprint {
		t.Fatalf("ContentFingerprint = %q, want %q", result.ContentFingerprint, expectedFingerprint)
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

// prefixContextualizer is a deterministic fixture that emits a per-record
// prefix derived from the record ref, so tests can both reason about what
// gets persisted and detect when the contextualizer "changes" across
// rebuilds (by swapping the tag).
type prefixContextualizer struct {
	tag   string
	calls int
}

func (p *prefixContextualizer) ContextualizeChunks(_ context.Context, record corpus.Record, sections []chunk.Section) ([]string, error) {
	p.calls++
	out := make([]string, len(sections))
	for i := range sections {
		out[i] = p.tag + ":" + record.Ref
	}
	return out, nil
}

func TestRebuildContextualizerPersistsPrefix(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	contextualizer := &prefixContextualizer{tag: "ctx"}
	records := []corpus.Record{{
		Ref:        "alpha",
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "Alpha handles queue admission.",
	}}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:           path,
		Embedder:       fixture,
		Contextualizer: contextualizer,
	}); err != nil {
		t.Fatalf("Rebuild(contextualizer) error = %v", err)
	}
	if contextualizer.calls == 0 {
		t.Fatal("contextualizer.calls = 0, want >=1 (Rebuild must invoke Contextualizer when configured)")
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
	if len(sections) == 0 {
		t.Fatal("Sections() returned no sections")
	}
	want := "ctx:alpha"
	for _, section := range sections {
		if section.ContextPrefix != want {
			t.Fatalf("section %s ContextPrefix = %q, want %q", section.Ref, section.ContextPrefix, want)
		}
	}
}

func TestSearchFTSMatchesContextualPrefix(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{
		{
			Ref:        testAlphaRef,
			Kind:       "note",
			Title:      "Alpha",
			SourceRef:  "file://alpha.txt",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Widgets are processed in batches every hour.",
		},
		{
			Ref:        "beta",
			Kind:       "note",
			Title:      "Beta",
			SourceRef:  "file://beta.txt",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Unrelated text about retries.",
		},
	}

	// Contextualizer appends a distinct token ("contextAlphaToken") only
	// on the alpha record. An FTS search for that token must find alpha,
	// proving the prefix ended up in the FTS5 index even though it is
	// absent from the stored content column.
	contextualizer := contextualizerFunc(func(_ context.Context, record corpus.Record, sections []chunk.Section) ([]string, error) {
		out := make([]string, len(sections))
		for i := range sections {
			if record.Ref == testAlphaRef {
				out[i] = "contextAlphaToken"
			}
		}
		return out, nil
	})

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:           path,
		Embedder:       fixture,
		Contextualizer: contextualizer,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	hits, err := Search(context.Background(), SearchQuery{
		Path:     path,
		Text:     "contextAlphaToken",
		Limit:    5,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("Search() returned no hits for contextualizer-only token")
	}
	if hits[0].Ref != testAlphaRef {
		t.Fatalf("first hit ref = %q, want alpha (FTS must match the contextual prefix)", hits[0].Ref)
	}
	// The stored content column must stay the original body, not the
	// prefix-augmented text surfaced through FTS.
	if strings.Contains(hits[0].Content, "contextAlphaToken") {
		t.Fatalf("hit content leaks context prefix: %q", hits[0].Content)
	}
}

func TestRebuildContextualizerChangeInvalidatesReuse(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	dir := t.TempDir()
	pathA := dir + "/a.db"
	pathB := dir + "/b.db"

	records := []corpus.Record{{
		Ref:        "alpha",
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "Alpha handles queue admission.",
	}}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:           pathA,
		Embedder:       fixture,
		Contextualizer: &prefixContextualizer{tag: "v1"},
	}); err != nil {
		t.Fatalf("Rebuild(v1) error = %v", err)
	}

	// A contextualizer that emits a different prefix for the same record
	// changes the reuse key. Reuse must not carry over stale embeddings.
	resultSwap, err := Rebuild(context.Background(), records, BuildOptions{
		Path:           pathB,
		ReuseFromPath:  pathA,
		Embedder:       fixture,
		Contextualizer: &prefixContextualizer{tag: "v2"},
	})
	if err != nil {
		t.Fatalf("Rebuild(v2 reusing v1) error = %v", err)
	}
	if resultSwap.ReusedChunkCount != 0 {
		t.Fatalf("ReusedChunkCount = %d, want 0 (different prefix must miss the reuse key)",
			resultSwap.ReusedChunkCount)
	}

	// Sanity: the same contextualizer version reuses the same chunks.
	pathC := dir + "/c.db"
	resultSame, err := Rebuild(context.Background(), records, BuildOptions{
		Path:           pathC,
		ReuseFromPath:  pathA,
		Embedder:       fixture,
		Contextualizer: &prefixContextualizer{tag: "v1"},
	})
	if err != nil {
		t.Fatalf("Rebuild(v1 reusing v1) error = %v", err)
	}
	if resultSame.ReusedChunkCount != resultSame.ChunkCount {
		t.Fatalf("ReusedChunkCount = %d, ChunkCount = %d, want equal (matching prefix must hit reuse)",
			resultSame.ReusedChunkCount, resultSame.ChunkCount)
	}
}

func TestRebuildContextualizerRejectsWrongLengthPrefixes(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	broken := contextualizerFunc(func(_ context.Context, _ corpus.Record, _ []chunk.Section) ([]string, error) {
		return []string{"only-one-prefix"}, nil
	})

	_, err = Rebuild(context.Background(), []corpus.Record{{
		Ref:        "alpha",
		Kind:       "guide",
		Title:      "Alpha",
		SourceRef:  "file://alpha.md",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   "# One\n\nfirst.\n\n# Two\n\nsecond.",
	}}, BuildOptions{
		Path:           path,
		Embedder:       fixture,
		Contextualizer: broken,
	})
	if err == nil || !strings.Contains(err.Error(), "prefixes for") {
		t.Fatalf("Rebuild(broken contextualizer) err = %v, want length-mismatch error", err)
	}
}

// TestOpenSnapshotV2BackwardCompatibleSections locks the read-compat
// contract: pre-1.0 v2 snapshots still open for read-only workloads
// (Stats, Records, Sections, Search) under v1.0+ code. Read paths never
// recompute HashRecord, so the content_hash encoding bump does not affect
// them; only Update forces the migration chain. Sections() on a v2
// snapshot projects an empty string for the missing context_prefix column.
func TestOpenSnapshotV2BackwardCompatibleSections(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}
	if err := rewriteSnapshotToV2(path); err != nil {
		t.Fatalf("rewriteSnapshotToV2() error = %v", err)
	}

	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot(v2) error = %v", err)
	}
	defer func() { _ = snap.Close() }()

	sections, err := snap.Sections(context.Background(), SectionQuery{})
	if err != nil {
		t.Fatalf("Sections(v2) error = %v", err)
	}
	if len(sections) == 0 {
		t.Fatal("Sections(v2) returned no sections")
	}
	for _, section := range sections {
		if section.ContextPrefix != "" {
			t.Fatalf("v2 section %s ContextPrefix = %q, want empty", section.Ref, section.ContextPrefix)
		}
	}
}

// TestRebuildReuseFromV2SnapshotStillHits locks the read-compat contract
// for reuse. A pre-1.0 v2 snapshot remains a valid ReuseFromPath source:
// chunk-level reuse via reuseChunkKey (title/heading/body-keyed, algo-
// independent) still fires on unchanged sections even though record-level
// content_hash comparison can't (the v2 file's stored hashes are under the
// old encoding, while the new-side recomputes under the new encoding).
func TestRebuildReuseFromV2SnapshotStillHits(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	dir := t.TempDir()
	pathA := dir + "/a.db"
	pathB := dir + "/b.db"

	records := []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}
	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     pathA,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild(A) error = %v", err)
	}
	if err := rewriteSnapshotToV2(pathA); err != nil {
		t.Fatalf("rewriteSnapshotToV2() error = %v", err)
	}

	result, err := Rebuild(context.Background(), records, BuildOptions{
		Path:          pathB,
		ReuseFromPath: pathA,
		Embedder:      fixture,
	})
	if err != nil {
		t.Fatalf("Rebuild(B reusing v2 A) error = %v", err)
	}
	if result.ReusedChunkCount == 0 {
		t.Fatal("ReusedChunkCount = 0 against v2 reuse source; chunk-level reuse must stay compatible when no Contextualizer is configured")
	}
}

func TestUpdateMigratesV2ToCurrentSchemaChain(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}
	if err := rewriteSnapshotToV2(path); err != nil {
		t.Fatalf("rewriteSnapshotToV2() error = %v", err)
	}

	if _, err := Update(context.Background(), []corpus.Record{{
		Ref:        "beta",
		Kind:       "note",
		Title:      "Beta",
		SourceRef:  "file://beta.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "beta content",
	}}, nil, UpdateOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Update(v2 chain migration) error = %v", err)
	}

	stats, err := ReadStats(context.Background(), path)
	if err != nil {
		t.Fatalf("ReadStats() error = %v", err)
	}
	if stats.SchemaVersion != schemaVersion {
		t.Fatalf("SchemaVersion = %q after Update chain migration, want %q", stats.SchemaVersion, schemaVersion)
	}
}

// TestSchemaHasContextPrefixMapping locks the direct mapping read paths
// now rely on in place of runtime PRAGMA probes: v3 and v4 both carry
// context_prefix (the v3→v4 bump re-hashed records but kept the table
// shape), and legacy v2 predates the column. If a future schema bump
// adds another accept-listed version, this test forces the mapping to be
// reconsidered explicitly rather than silently inheriting one side's
// default.
func TestSchemaHasContextPrefixMapping(t *testing.T) {
	t.Parallel()

	if !schemaHasContextPrefix(schemaVersion) {
		t.Fatalf("schemaHasContextPrefix(%q) = false, want true (v4 carries context_prefix)", schemaVersion)
	}
	if !schemaHasContextPrefix(prevSchemaVersion) {
		t.Fatalf("schemaHasContextPrefix(%q) = false, want true (v3 carries context_prefix)", prevSchemaVersion)
	}
	if schemaHasContextPrefix(legacySchemaVersionV2) {
		t.Fatalf("schemaHasContextPrefix(%q) = true, want false (v2 predates context_prefix)", legacySchemaVersionV2)
	}
	// Whitespace tolerance matches the open-time check, which reads
	// schema_version via readMetadataValue and trims before compare.
	if !schemaHasContextPrefix("  " + schemaVersion + "  ") {
		t.Fatalf("schemaHasContextPrefix did not trim whitespace around %q", schemaVersion)
	}
}

// TestOpenSnapshotCachesContextPrefixFlag verifies that OpenSnapshot
// derives hasContextPrefix from the accept-listed schema_version at
// handle-open time, so Sections()/reuse paths never need to probe
// PRAGMA table_info(chunks) per call.
func TestOpenSnapshotCachesContextPrefixFlag(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	// Fresh rebuilds are always at the current schemaVersion (v4). Both
	// accept-listed schemas (v3 and v4) carry context_prefix, so this flag
	// is uniformly true for anything OpenSnapshot will accept.
	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	defer func() { _ = snap.Close() }()
	if !snap.hasContextPrefix {
		t.Fatalf("Snapshot.hasContextPrefix = false on %q snapshot, want true", schemaVersion)
	}
}

// TestOpenSnapshotRejectsSchemaTableShapeDivergence locks the open-time
// invariant that schema_version and chunks.context_prefix presence must
// agree. Both accept-listed schemas (v3, v4) carry the column, so the only
// divergence OpenSnapshot can still trip over is: metadata says an
// accept-listed schema but the column is missing (half-migration or
// external tampering). Without the open-time probe, Sections() would fail
// later with "no such column" on the first query instead of failing
// loudly at handle open.
func TestOpenSnapshotRejectsSchemaTableShapeDivergence(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	rwDB, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	if _, err := rwDB.ExecContext(context.Background(), `ALTER TABLE chunks DROP COLUMN context_prefix`); err != nil {
		t.Fatalf("drop context_prefix: %v", err)
	}
	if err := rwDB.Close(); err != nil {
		t.Fatalf("close rw db: %v", err)
	}

	_, err = OpenSnapshot(context.Background(), path)
	if err == nil {
		t.Fatal("OpenSnapshot() succeeded with metadata at current schema but missing column, want divergence error")
	}
	if !strings.Contains(err.Error(), "schema_version") || !strings.Contains(err.Error(), "context_prefix") {
		t.Fatalf("OpenSnapshot() err = %v, want divergence error mentioning schema_version and context_prefix", err)
	}
}

// rewriteSnapshotToV2 converts a freshly-built snapshot into a pristine
// v2 one by dropping the context_prefix column and rewinding
// schema_version to "2", so the v2-chain-migration tests can run against
// the same build surface that produced the snapshot in the first place.
func rewriteSnapshotToV2(path string) error {
	db, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		return err
	}
	defer func() { _ = db.Close() }()
	for _, stmt := range []string{
		`ALTER TABLE chunks DROP COLUMN context_prefix`,
		`UPDATE metadata SET value = '2' WHERE key = 'schema_version'`,
	} {
		if _, err := db.ExecContext(context.Background(), stmt); err != nil {
			return err
		}
	}
	return nil
}

func TestOpenSnapshotRejectsUnknownSchemaVersion(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	db, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	if _, err := db.ExecContext(context.Background(),
		`UPDATE metadata SET value = '99' WHERE key = 'schema_version'`); err != nil {
		_ = db.Close()
		t.Fatalf("rewrite schema_version = %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("db.Close() error = %v", err)
	}

	_, err = OpenSnapshot(context.Background(), path)
	if err == nil {
		t.Fatal("OpenSnapshot() succeeded on schema_version=99, want ErrUnsupportedSchemaVersion")
	}
	if !errors.Is(err, ErrUnsupportedSchemaVersion) {
		t.Fatalf("OpenSnapshot() err = %v, want wraps ErrUnsupportedSchemaVersion", err)
	}
	if !strings.Contains(err.Error(), `"99"`) {
		t.Fatalf("OpenSnapshot() err = %v, want message to include observed version %q", err, "99")
	}
}

func TestMigrateV2ToV3IsCrashSafe(t *testing.T) {
	// Not parallel: mutates the package-level migrateV2ToV3InjectFault hook.

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}
	if err := rewriteSnapshotToV2(path); err != nil {
		t.Fatalf("rewriteSnapshotToV2() error = %v", err)
	}

	injected := errors.New("simulated crash between ALTER and UPDATE")
	migrateV2ToV3InjectFault = func() error { return injected }
	t.Cleanup(func() { migrateV2ToV3InjectFault = nil })

	_, err = Update(context.Background(), []corpus.Record{{
		Ref:        "beta",
		Kind:       "note",
		Title:      "Beta",
		SourceRef:  "file://beta.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "beta content",
	}}, nil, UpdateOptions{
		Path:     path,
		Embedder: fixture,
	})
	if err == nil {
		t.Fatal("Update() succeeded with injected fault, want failure")
	}
	if !errors.Is(err, injected) {
		t.Fatalf("Update() err = %v, want wraps injected fault", err)
	}

	// Update's main transaction must roll back both the ALTER and the
	// schema_version bump, so the snapshot is back to a pristine v2 (no
	// context_prefix column yet and schema_version still "2").
	checkDB, err := store.OpenReadOnlyContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadOnlyContext() error = %v", err)
	}
	hasPrefix, err := hasChunkColumn(context.Background(), checkDB, "context_prefix")
	if err != nil {
		_ = checkDB.Close()
		t.Fatalf("hasChunkColumn() error = %v", err)
	}
	if hasPrefix {
		_ = checkDB.Close()
		t.Fatal("context_prefix column present after rolled-back migration; ALTER leaked past ROLLBACK")
	}
	schema, err := readMetadataValue(context.Background(), checkDB, "schema_version")
	if err != nil {
		_ = checkDB.Close()
		t.Fatalf("readMetadataValue() error = %v", err)
	}
	if strings.TrimSpace(schema) != legacySchemaVersionV2 {
		_ = checkDB.Close()
		t.Fatalf("schema_version = %q after rollback, want %q", schema, legacySchemaVersionV2)
	}
	if err := checkDB.Close(); err != nil {
		t.Fatalf("checkDB.Close() error = %v", err)
	}

	// Clear the fault and retry. A fresh Update must complete the migration
	// cleanly, proving the rollback left the file re-openable.
	migrateV2ToV3InjectFault = nil
	if _, err := Update(context.Background(), []corpus.Record{{
		Ref:        "beta",
		Kind:       "note",
		Title:      "Beta",
		SourceRef:  "file://beta.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "beta content",
	}}, nil, UpdateOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Update() retry error = %v", err)
	}

	stats, err := ReadStats(context.Background(), path)
	if err != nil {
		t.Fatalf("ReadStats() error = %v", err)
	}
	if stats.SchemaVersion != schemaVersion {
		t.Fatalf("SchemaVersion = %q after retry, want %q", stats.SchemaVersion, schemaVersion)
	}
}

// TestUpdateMigratesV3ToV4RehashesRecords verifies that when Update runs
// against a freshly-built v4 snapshot that has been forced back to v3 with
// old-algorithm content_hashes, migrateV3ToV4 rewrites every row's
// content_hash under the new injective HashRecord encoding and the
// published ContentFingerprint reflects the post-migration hashes.
func TestUpdateMigratesV3ToV4RehashesRecords(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	seed := []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
		Metadata:   map[string]string{"k": "v"},
	}}
	if _, err := Rebuild(context.Background(), seed, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	// Simulate a legacy v3 snapshot: fake-old content_hash on every row
	// (matches the "stored hashes were produced by the old encoding"
	// state that a pre-1.0 v3 file would exhibit) and rewind schema_version.
	rwDB, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	if _, err := rwDB.ExecContext(context.Background(),
		`UPDATE records SET content_hash = ?`, testLegacyV3HashSentinel); err != nil {
		t.Fatalf("overwrite content_hash: %v", err)
	}
	if _, err := rwDB.ExecContext(context.Background(),
		`UPDATE metadata SET value = ? WHERE key = 'schema_version'`, prevSchemaVersion); err != nil {
		t.Fatalf("rewind schema_version: %v", err)
	}
	if err := rwDB.Close(); err != nil {
		t.Fatalf("close rw db: %v", err)
	}

	// An Update with no changes still triggers the migration chain via
	// resolveUpdateSessionConfig; finalizeUpdate then republishes
	// ContentFingerprint from the post-migration (ref, content_hash) pairs.
	result, err := Update(context.Background(), nil, nil, UpdateOptions{
		Path:     path,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Update(v3→v4 migration) error = %v", err)
	}

	stats, err := ReadStats(context.Background(), path)
	if err != nil {
		t.Fatalf("ReadStats() error = %v", err)
	}
	if stats.SchemaVersion != schemaVersion {
		t.Fatalf("SchemaVersion = %q after migration, want %q", stats.SchemaVersion, schemaVersion)
	}

	// The fingerprint must match what corpus.Fingerprint produces over the
	// original normalized records — that's the byte-identity contract.
	normalized := make([]corpus.Record, 0, len(seed))
	for _, r := range seed {
		n, err := r.Normalized()
		if err != nil {
			t.Fatalf("Normalized(%q) error = %v", r.Ref, err)
		}
		normalized = append(normalized, n)
	}
	expected, err := corpus.Fingerprint(normalized)
	if err != nil {
		t.Fatalf("corpus.Fingerprint() error = %v", err)
	}
	if result.ContentFingerprint != expected {
		t.Fatalf("ContentFingerprint = %q after migration, want %q (rehash did not match new HashRecord encoding)",
			result.ContentFingerprint, expected)
	}

	// And the on-disk content_hash for every record must be the new-algo
	// hash, not the testLegacyV3HashSentinel sentinel we wrote.
	roDB, err := store.OpenReadOnlyContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadOnlyContext() error = %v", err)
	}
	defer func() { _ = roDB.Close() }()
	var storedHash string
	if err := roDB.QueryRowContext(context.Background(),
		`SELECT content_hash FROM records WHERE ref = ?`, testAlphaRef).Scan(&storedHash); err != nil {
		t.Fatalf("read content_hash: %v", err)
	}
	if storedHash == testLegacyV3HashSentinel {
		t.Fatal("content_hash still equal to legacy sentinel; migration did not rewrite rows")
	}
	if storedHash != corpus.HashRecord(normalized[0]) {
		t.Fatalf("stored content_hash = %q, want %q (new-algo HashRecord)",
			storedHash, corpus.HashRecord(normalized[0]))
	}
}

// TestMigrateV3ToV4BatchesAcrossMultiplePasses exercises the multi-batch
// path: if the per-batch limit is smaller than the record count, the
// migration advances via `WHERE ref > ?` seeks across several passes.
// Without proper seek continuity, the second pass could either reprocess
// the first-batch rows (wasted work, still correct) or skip the last
// seen ref (leaves stale old-algo content_hashes on disk). This test
// shrinks the batch to 1 so every row forces a fresh pass.
func TestMigrateV3ToV4BatchesAcrossMultiplePasses(t *testing.T) {
	// Not parallel: mutates the package-level migrateV3ToV4BatchSize.

	original := migrateV3ToV4BatchSize
	migrateV3ToV4BatchSize = 1
	t.Cleanup(func() { migrateV3ToV4BatchSize = original })

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	seed := []corpus.Record{
		{Ref: "alpha", Kind: "note", Title: "Alpha", SourceRef: "a", BodyFormat: corpus.FormatPlaintext, BodyText: "a"},
		{Ref: "bravo", Kind: "note", Title: "Bravo", SourceRef: "b", BodyFormat: corpus.FormatPlaintext, BodyText: "b"},
		{Ref: "charlie", Kind: "note", Title: "Charlie", SourceRef: "c", BodyFormat: corpus.FormatPlaintext, BodyText: "c"},
	}
	if _, err := Rebuild(context.Background(), seed, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	rwDB, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	if _, err := rwDB.ExecContext(context.Background(),
		`UPDATE records SET content_hash = ?`, testLegacyV3HashSentinel); err != nil {
		t.Fatalf("overwrite content_hash: %v", err)
	}
	if _, err := rwDB.ExecContext(context.Background(),
		`UPDATE metadata SET value = ? WHERE key = 'schema_version'`, prevSchemaVersion); err != nil {
		t.Fatalf("rewind schema_version: %v", err)
	}
	if err := rwDB.Close(); err != nil {
		t.Fatalf("close rw db: %v", err)
	}

	if _, err := Update(context.Background(), nil, nil, UpdateOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Update(v3→v4 with batchSize=1) error = %v", err)
	}

	// Every row must now carry a new-algo hash — none should still
	// have the sentinel a missed-seek batch would have left behind.
	roDB, err := store.OpenReadOnlyContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadOnlyContext() error = %v", err)
	}
	defer func() { _ = roDB.Close() }()
	rows, err := roDB.QueryContext(context.Background(),
		`SELECT ref, content_hash FROM records ORDER BY ref ASC`)
	if err != nil {
		t.Fatalf("query content_hash: %v", err)
	}
	defer func() { _ = rows.Close() }()

	seenByRef := make(map[string]string, len(seed))
	for rows.Next() {
		var ref, hash string
		if err := rows.Scan(&ref, &hash); err != nil {
			t.Fatalf("scan: %v", err)
		}
		if hash == testLegacyV3HashSentinel {
			t.Fatalf("record %q still has legacy sentinel; multi-batch migration missed this row", ref)
		}
		seenByRef[ref] = hash
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("rows.Err(): %v", err)
	}
	if len(seenByRef) != len(seed) {
		t.Fatalf("migrated %d records, want %d", len(seenByRef), len(seed))
	}
}

// TestMigrateV3ToV4IsCrashSafe mirrors the v2→v3 crash-safety test: if a
// fault fires between the per-row re-hash and the schema_version bump, the
// enclosing Update transaction must roll back and leave the snapshot
// fully at v3 with the original (old-algo) content_hashes intact.
func TestMigrateV3ToV4IsCrashSafe(t *testing.T) {
	// Not parallel: mutates the package-level migrateV3ToV4InjectFault hook.

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	// Force v3 state with a known-bogus content_hash we can detect after
	// rollback — the migration must not leak any UPDATE past the fault.
	rwDB, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	if _, err := rwDB.ExecContext(context.Background(),
		`UPDATE records SET content_hash = ?`, testLegacyV3HashSentinel); err != nil {
		t.Fatalf("overwrite content_hash: %v", err)
	}
	if _, err := rwDB.ExecContext(context.Background(),
		`UPDATE metadata SET value = ? WHERE key = 'schema_version'`, prevSchemaVersion); err != nil {
		t.Fatalf("rewind schema_version: %v", err)
	}
	if err := rwDB.Close(); err != nil {
		t.Fatalf("close rw db: %v", err)
	}

	injected := errors.New("simulated crash between rehash and metadata bump")
	migrateV3ToV4InjectFault = func() error { return injected }
	t.Cleanup(func() { migrateV3ToV4InjectFault = nil })

	_, err = Update(context.Background(), nil, nil, UpdateOptions{
		Path:     path,
		Embedder: fixture,
	})
	if err == nil {
		t.Fatal("Update() succeeded with injected fault, want failure")
	}
	if !errors.Is(err, injected) {
		t.Fatalf("Update() err = %v, want wraps injected fault", err)
	}

	// The rollback must leave schema_version at v3 and every
	// content_hash at the legacy sentinel — if either rewrote-per-row or
	// the metadata bump leaked, we would see otherwise here.
	checkDB, err := store.OpenReadOnlyContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadOnlyContext() error = %v", err)
	}
	defer func() { _ = checkDB.Close() }()
	schema, err := readMetadataValue(context.Background(), checkDB, "schema_version")
	if err != nil {
		t.Fatalf("readMetadataValue() error = %v", err)
	}
	if strings.TrimSpace(schema) != prevSchemaVersion {
		t.Fatalf("schema_version = %q after rolled-back v3→v4 migration, want %q",
			schema, prevSchemaVersion)
	}
	var storedHash string
	if err := checkDB.QueryRowContext(context.Background(),
		`SELECT content_hash FROM records WHERE ref = ?`, testAlphaRef).Scan(&storedHash); err != nil {
		t.Fatalf("read content_hash: %v", err)
	}
	if storedHash != testLegacyV3HashSentinel {
		t.Fatalf("content_hash = %q after rollback, want legacy sentinel preserved (rehash leaked past rollback)", storedHash)
	}
}

func TestUpdatePostCommitIntegrityFailureIsTyped(t *testing.T) {
	// Not parallel: mutates the package-level runIntegrityChecksInjectFault hook.

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	// Force the post-commit integrity check to fail. Because the fault
	// fires after tx.Commit has already returned, the "beta" record
	// write is durable; Update must surface the failure with
	// ErrUpdateCommittedIntegrityCheckFailed so callers can distinguish
	// it from a pre-commit rollback and route it to operator inspection
	// instead of a naive retry.
	integrityErr := errors.New("simulated post-commit integrity corruption")
	runIntegrityChecksInjectFault = func() error { return integrityErr }
	t.Cleanup(func() { runIntegrityChecksInjectFault = nil })

	_, err = Update(context.Background(), []corpus.Record{{
		Ref:        "beta",
		Kind:       "note",
		Title:      "Beta",
		SourceRef:  "file://beta.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "beta content",
	}}, nil, UpdateOptions{
		Path:     path,
		Embedder: fixture,
	})
	if err == nil {
		t.Fatal("Update() succeeded with integrity fault, want ErrUpdateCommittedIntegrityCheckFailed")
	}
	if !errors.Is(err, ErrUpdateCommittedIntegrityCheckFailed) {
		t.Fatalf("Update() err = %v, want wraps ErrUpdateCommittedIntegrityCheckFailed", err)
	}
	if !errors.Is(err, integrityErr) {
		t.Fatalf("Update() err = %v, want also wraps underlying injected integrity error", err)
	}

	// Clear the fault so ReadStats's own integrity machinery (if any)
	// does not trip; then confirm the Update really did commit durably
	// before the typed error was returned.
	runIntegrityChecksInjectFault = nil
	stats, err := ReadStats(context.Background(), path)
	if err != nil {
		t.Fatalf("ReadStats() error = %v", err)
	}
	if stats.RecordCount != 2 {
		t.Fatalf("RecordCount = %d after typed integrity failure, want 2; update was not durable", stats.RecordCount)
	}
}

// failingEmbedder mirrors the fingerprint and dimension of the wrapped
// fixture so it passes resolveUpdateSessionConfig's compatibility checks,
// but returns err from EmbedDocuments. Used to drive a post-migration
// failure in TestUpdateFailureAfterMigrationRollsBackSchema.
type failingEmbedder struct {
	inner embed.Embedder
	err   error
}

func (f failingEmbedder) Fingerprint() string { return f.inner.Fingerprint() }
func (f failingEmbedder) Dimension(ctx context.Context) (int, error) {
	return f.inner.Dimension(ctx)
}
func (f failingEmbedder) EmbedDocuments(context.Context, []string) ([][]float64, error) {
	return nil, f.err
}
func (f failingEmbedder) EmbedQueries(context.Context, []string) ([][]float64, error) {
	return nil, f.err
}

func TestUpdateFailureAfterMigrationRollsBackSchema(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}
	if err := rewriteSnapshotToV2(path); err != nil {
		t.Fatalf("rewriteSnapshotToV2() error = %v", err)
	}

	// Drive a failure AFTER resolveUpdateSessionConfig has already run
	// migrateV2ToV3 on the outer tx, by failing the embedder when Update
	// tries to embed the new record. The tx must roll back both the
	// schema bump and the half-written update, leaving the file at v2.
	embedErr := errors.New("simulated embedder failure")
	broken := failingEmbedder{inner: fixture, err: embedErr}

	_, err = Update(context.Background(), []corpus.Record{{
		Ref:        "beta",
		Kind:       "note",
		Title:      "Beta",
		SourceRef:  "file://beta.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "beta content",
	}}, nil, UpdateOptions{
		Path:     path,
		Embedder: broken,
	})
	if err == nil {
		t.Fatal("Update() succeeded despite failing embedder, want failure")
	}
	if !errors.Is(err, embedErr) {
		t.Fatalf("Update() err = %v, want wraps embedder failure", err)
	}

	checkDB, err := store.OpenReadOnlyContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadOnlyContext() error = %v", err)
	}
	defer func() { _ = checkDB.Close() }()
	schema, err := readMetadataValue(context.Background(), checkDB, "schema_version")
	if err != nil {
		t.Fatalf("readMetadataValue() error = %v", err)
	}
	if strings.TrimSpace(schema) != legacySchemaVersionV2 {
		t.Fatalf("schema_version = %q after failed Update, want %q; migration leaked past Update's rollback",
			schema, legacySchemaVersionV2)
	}
	hasPrefix, err := hasChunkColumn(context.Background(), checkDB, "context_prefix")
	if err != nil {
		t.Fatalf("hasChunkColumn() error = %v", err)
	}
	if hasPrefix {
		t.Fatal("context_prefix column present after failed Update; ALTER leaked past rollback")
	}
}

// contextualizerFunc adapts a function literal to ChunkContextualizer for
// test fixtures that only need a one-off implementation.
type contextualizerFunc func(ctx context.Context, record corpus.Record, sections []chunk.Section) ([]string, error)

func (f contextualizerFunc) ContextualizeChunks(ctx context.Context, record corpus.Record, sections []chunk.Section) ([]string, error) {
	return f(ctx, record, sections)
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

func TestSearchMatryoshkaTruncatedPrefilterRescoresAtFullDim(t *testing.T) {
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
		{
			Ref:        "retry-guide",
			Kind:       "guide",
			Title:      "Retry Guide",
			SourceRef:  "file://docs/retry.md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   "# Retries\n\nBackoff windows govern retry cadence.",
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
		Text:     "background workers batches",
		Limit:    3,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search(baseline) error = %v", err)
	}
	if len(baseline) == 0 {
		t.Fatal("baseline returned no hits")
	}

	// Truncated prefilter at half the stored dimension. The rescore still
	// runs full-dim cosine, so any chunk that survives the prefilter into
	// the rescored result should carry the same score the baseline path
	// assigned to it.
	truncated, err := Search(context.Background(), SearchQuery{
		Path:            path,
		Text:            "background workers batches",
		Limit:           3,
		Embedder:        fixture,
		SearchDimension: 8,
	})
	if err != nil {
		t.Fatalf("Search(SearchDimension=8) error = %v", err)
	}
	if len(truncated) == 0 {
		t.Fatal("Matryoshka path returned no hits")
	}

	baselineScores := make(map[int64]float64, len(baseline))
	for _, hit := range baseline {
		baselineScores[hit.ChunkID] = hit.Score
	}
	for _, hit := range truncated {
		want, ok := baselineScores[hit.ChunkID]
		if !ok {
			continue
		}
		if hit.Score != want {
			t.Fatalf("chunk %d Matryoshka score = %v, baseline = %v (full-dim rescore must match)",
				hit.ChunkID, hit.Score, want)
		}
	}

	// SearchDimension == stored dim is a no-op: it must collapse to the
	// same result set as the baseline search.
	noop, err := Search(context.Background(), SearchQuery{
		Path:            path,
		Text:            "background workers batches",
		Limit:           3,
		Embedder:        fixture,
		SearchDimension: 16,
	})
	if err != nil {
		t.Fatalf("Search(SearchDimension=16) error = %v", err)
	}
	if len(noop) != len(baseline) {
		t.Fatalf("no-op truncation returned %d hits, baseline returned %d", len(noop), len(baseline))
	}
	for i := range noop {
		if noop[i].ChunkID != baseline[i].ChunkID || noop[i].Score != baseline[i].Score {
			t.Fatalf("no-op truncation hit %d = (%d, %v), baseline = (%d, %v)",
				i, noop[i].ChunkID, noop[i].Score, baseline[i].ChunkID, baseline[i].Score)
		}
	}
}

func TestSearchMatryoshkaRejectsInvalidDimension(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{{
		Ref:        "alpha",
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}
	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	if _, err := Search(context.Background(), SearchQuery{
		Path:            path,
		Text:            "alpha",
		Limit:           3,
		Embedder:        fixture,
		SearchDimension: 17,
	}); err == nil || !strings.Contains(err.Error(), "exceeds stored embedder_dimension") {
		t.Fatalf("oversized SearchDimension err = %v, want exceeds-stored error", err)
	}
}

func TestSearchMatryoshkaRejectsInt8Index(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	records := []corpus.Record{{
		Ref:        "alpha",
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}
	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:         path,
		Embedder:     fixture,
		Quantization: "int8",
	}); err != nil {
		t.Fatalf("Rebuild(int8) error = %v", err)
	}

	if _, err := Search(context.Background(), SearchQuery{
		Path:            path,
		Text:            "alpha",
		Limit:           3,
		Embedder:        fixture,
		SearchDimension: 8,
	}); err == nil || !strings.Contains(err.Error(), "only supported for float32") {
		t.Fatalf("int8 SearchDimension err = %v, want float32-only error", err)
	}
}

func TestSearchMatryoshkaHonorsKindFilterInsidePrefilter(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"

	// Build a corpus where the "note" kind is a tiny minority against a
	// large block of "noise" chunks. If the Matryoshka prefilter applied
	// the kind filter after truncation, the single note chunk could
	// easily fall outside the shortlist and the search would silently
	// return zero hits even though a matching kind exists.
	records := []corpus.Record{{
		Ref:        testTargetNoteRef,
		Kind:       "note",
		Title:      "Target Note",
		SourceRef:  "file://notes/target.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "Background workers process items in batches every hour.",
	}}
	for i := 0; i < 50; i++ {
		records = append(records, corpus.Record{
			Ref:        "noise-" + strconv.Itoa(i),
			Kind:       "noise",
			Title:      "Noise " + strconv.Itoa(i),
			SourceRef:  "file://noise/" + strconv.Itoa(i) + ".txt",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Unrelated filler text number " + strconv.Itoa(i) + ".",
		})
	}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	hits, err := Search(context.Background(), SearchQuery{
		Path:            path,
		Text:            "background workers batches",
		Limit:           3,
		Kinds:           []string{"note"},
		Embedder:        fixture,
		SearchDimension: 8,
	})
	if err != nil {
		t.Fatalf("Search(Kinds=note, SearchDimension=8) error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("Matryoshka + kind filter returned no hits; the prefilter must apply the kind restriction before truncation, not after")
	}
	if hits[0].Ref != testTargetNoteRef || hits[0].Kind != testNoteKind {
		t.Fatalf("first hit = (%s, %s), want (%s, %s)", hits[0].Ref, hits[0].Kind, testTargetNoteRef, testNoteKind)
	}
	for _, hit := range hits {
		if hit.Kind != "note" {
			t.Fatalf("kind filter leaked: hit %s has kind %q", hit.Ref, hit.Kind)
		}
	}
}

// kindFilterStarvationRecords builds a corpus designed to expose the
// post-prefilter kind-filter starvation bug: the single "note" chunk
// uses body text that is *unrelated* to the query, while every "noise"
// chunk repeats the query tokens verbatim. Any prefilter that ignores
// the kind predicate fills its shortlist with noise, and the outer
// WHERE r.kind IN (...) strips every row. The corpus is sized so the
// prefilter shortlist (candidateLimit == 25 for the default path,
// candidateLimit * binaryRescoreMultiplier == 75 for the binary path)
// cannot swallow the whole corpus and mask the bug.
func kindFilterStarvationRecords(noiseCount int) []corpus.Record {
	records := []corpus.Record{{
		Ref:        testTargetNoteRef,
		Kind:       testNoteKind,
		Title:      "Target Note",
		SourceRef:  "file://notes/target.txt",
		BodyFormat: corpus.FormatPlaintext,
		// Deliberately lexically distant from the query so the target
		// note is ranked below the noise by the deterministic fixture
		// embedder. With the bug present the prefilter never surfaces
		// it; with the fix the in-CTE kind filter restricts the search
		// space to the note kind and the note is the only candidate.
		BodyText: "Reminder: review quarterly hiring plans on Monday.",
	}}
	for i := 0; i < noiseCount; i++ {
		records = append(records, corpus.Record{
			Ref:        "noise-" + strconv.Itoa(i),
			Kind:       "noise",
			Title:      "Noise " + strconv.Itoa(i),
			SourceRef:  "file://noise/" + strconv.Itoa(i) + ".txt",
			BodyFormat: corpus.FormatPlaintext,
			BodyText:   "Background workers process items in batches every hour.",
		})
	}
	return records
}

func TestSearchFloat32HonorsKindFilterInsidePrefilter(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), kindFilterStarvationRecords(120), BuildOptions{
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

	// SearchVector bypasses the FTS leg of hybrid search so the vec0
	// MATCH prefilter has to stand on its own. If the kind predicate is
	// applied after MATCH returned its k noise candidates, the result is
	// silently empty; the in-CTE kind predicate must widen the search
	// until the target note is reached.
	vectors, err := fixture.EmbedQueries(context.Background(), []string{"background workers batches"})
	if err != nil {
		t.Fatalf("EmbedQueries() error = %v", err)
	}
	hits, err := snapshot.SearchVector(context.Background(), VectorSearchQuery{
		Embedding: vectors[0],
		Limit:     3,
		Kinds:     []string{testNoteKind},
	})
	if err != nil {
		t.Fatalf("SearchVector(Kinds=note, default float32) error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("default float32 + kind filter returned no hits; the prefilter must apply the kind restriction inside the MATCH CTE, not after")
	}
	if hits[0].Ref != testTargetNoteRef || hits[0].Kind != testNoteKind {
		t.Fatalf("first hit = (%s, %s), want (%s, %s)", hits[0].Ref, hits[0].Kind, testTargetNoteRef, testNoteKind)
	}
	for _, hit := range hits {
		if hit.Kind != testNoteKind {
			t.Fatalf("kind filter leaked: hit %s has kind %q", hit.Ref, hit.Kind)
		}
	}
}

func TestSearchBinaryHonorsKindFilterInsidePrefilter(t *testing.T) {
	t.Parallel()

	// dim=32 is the smallest div-by-8 dim that spreads the fixture's
	// hashed bigrams across enough bits for the 1-bit prefilter to
	// separate the lexically-distant note from the query-token noise.
	// With dim=16 the binary signatures collide heavily and the target
	// note is accidentally captured by the hamming shortlist.
	fixture, err := embed.NewFixture("fixture-a", 32)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), kindFilterStarvationRecords(120), BuildOptions{
		Path:         path,
		Embedder:     fixture,
		Quantization: "binary",
	}); err != nil {
		t.Fatalf("Rebuild(binary) error = %v", err)
	}

	snapshot, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot(binary) error = %v", err)
	}
	defer func() { _ = snapshot.Close() }()

	// SearchVector isolates the binary two-stage path: hamming MATCH
	// prefilter over chunks_vec bit[N] feeding into a full-precision
	// cosine rescore against chunks_vec_full. If the kind predicate is
	// applied only after the hamming shortlist is built, a corpus where
	// the requested kind is a minority starves the rescore input.
	vectors, err := fixture.EmbedQueries(context.Background(), []string{"background workers batches"})
	if err != nil {
		t.Fatalf("EmbedQueries() error = %v", err)
	}
	hits, err := snapshot.SearchVector(context.Background(), VectorSearchQuery{
		Embedding: vectors[0],
		Limit:     3,
		Kinds:     []string{testNoteKind},
	})
	if err != nil {
		t.Fatalf("SearchVector(Kinds=note, binary) error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("binary + kind filter returned no hits; the prefilter must apply the kind restriction inside the hamming MATCH CTE, not after")
	}
	if hits[0].Ref != testTargetNoteRef || hits[0].Kind != testNoteKind {
		t.Fatalf("first hit = (%s, %s), want (%s, %s)", hits[0].Ref, hits[0].Kind, testTargetNoteRef, testNoteKind)
	}
	for _, hit := range hits {
		if hit.Kind != testNoteKind {
			t.Fatalf("kind filter leaked: hit %s has kind %q", hit.Ref, hit.Kind)
		}
	}
}

func TestSearchInt8HonorsKindFilterInsidePrefilter(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), kindFilterStarvationRecords(120), BuildOptions{
		Path:         path,
		Embedder:     fixture,
		Quantization: "int8",
	}); err != nil {
		t.Fatalf("Rebuild(int8) error = %v", err)
	}

	snapshot, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot(int8) error = %v", err)
	}
	defer func() { _ = snapshot.Close() }()

	// The int8 path shares buildSearchSQL with float32, so the same
	// starvation bug and fix apply — but the kind-filtered branch wraps
	// the query blob in vec_int8() to match the packed int8 column type.
	// Cover it explicitly: sqlite-vec rejects vec_slice over packed int8
	// bytes (hence no Matryoshka for int8), so the combination of
	// vec_distance_cosine + vec_int8(?) that this branch relies on is
	// genuinely a new operator pairing for this codebase.
	vectors, err := fixture.EmbedQueries(context.Background(), []string{"background workers batches"})
	if err != nil {
		t.Fatalf("EmbedQueries() error = %v", err)
	}
	hits, err := snapshot.SearchVector(context.Background(), VectorSearchQuery{
		Embedding: vectors[0],
		Limit:     3,
		Kinds:     []string{testNoteKind},
	})
	if err != nil {
		t.Fatalf("SearchVector(Kinds=note, int8) error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("int8 + kind filter returned no hits; the prefilter must apply the kind restriction inside the MATCH CTE, not after")
	}
	if hits[0].Ref != testTargetNoteRef || hits[0].Kind != testNoteKind {
		t.Fatalf("first hit = (%s, %s), want (%s, %s)", hits[0].Ref, hits[0].Kind, testTargetNoteRef, testNoteKind)
	}
	for _, hit := range hits {
		if hit.Kind != testNoteKind {
			t.Fatalf("kind filter leaked: hit %s has kind %q", hit.Ref, hit.Kind)
		}
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

// TestUpdateRefusesEmptyContentHashPair exercises the guard in
// loadCurrentRefHashes: the narrow fingerprint path in finalizeUpdate relies
// on every persisted row having a non-empty content_hash. If that invariant
// ever breaks (writer bypass, external tampering), the pair-based digest would
// silently diverge from corpus.Fingerprint([]Record), which re-derives the
// hash via HashRecord. Rather than diverge, Update must fail loudly.
func TestUpdateRefusesEmptyContentHashPair(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       "note",
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	// Blank the content_hash directly to simulate a writer bypass or a
	// tampered row — the only states in which the narrow (ref, content_hash)
	// path could silently disagree with corpus.Fingerprint([]Record).
	rwDB, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	if _, err := rwDB.ExecContext(context.Background(),
		`UPDATE records SET content_hash = '' WHERE ref = ?`, testAlphaRef); err != nil {
		t.Fatalf("blank content_hash: %v", err)
	}
	if err := rwDB.Close(); err != nil {
		t.Fatalf("close rw db: %v", err)
	}

	_, err = Update(context.Background(), []corpus.Record{{
		Ref:        "beta",
		Kind:       "note",
		Title:      "Beta",
		SourceRef:  "file://beta.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "beta content",
	}}, nil, UpdateOptions{
		Path:     path,
		Embedder: fixture,
	})
	if err == nil {
		t.Fatal("Update() succeeded with empty content_hash row, want error")
	}
	if !strings.Contains(err.Error(), "empty content_hash") {
		t.Fatalf("Update() err = %v, want error mentioning empty content_hash", err)
	}
}
