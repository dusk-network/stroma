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
