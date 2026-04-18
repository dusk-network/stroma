package index

import (
	"context"
	"strconv"
	"testing"

	"github.com/dusk-network/stroma/v2/corpus"
	"github.com/dusk-network/stroma/v2/embed"
	"github.com/dusk-network/stroma/v2/store"
)

// BenchmarkRebuildReuse measures the allocation cost of the reuse-probe
// hot path — loadStoredChunksForRecord. Pre-#60, every scanned blob was
// double-copied (once by database/sql bytes.Clone on *[]byte scan, again
// by the defensive append([]byte(nil), ...) in the loop); the Scan into
// []byte is the only copy that is actually required and is the only one
// retained post-#60.
//
// This benchmark populates a fixture-embedded snapshot with many chunks
// per record and then drives loadStoredChunksForRecord repeatedly. The
// -benchmem delta vs. a pre-#60 run is the gauge for D2's effectiveness.
func BenchmarkRebuildReuse(b *testing.B) {
	const (
		dim          = 16
		recordCount  = 4
		chunksPerRec = 10
	)

	fixture, err := embed.NewFixture("bench-fixture", dim)
	if err != nil {
		b.Fatal(err)
	}

	// Long-bodied records so each one produces multiple chunks through
	// the markdown chunker. The body pattern is deterministic so reuse
	// keys stay stable across runs.
	records := make([]corpus.Record, 0, recordCount)
	for i := 0; i < recordCount; i++ {
		body := ""
		for j := 0; j < chunksPerRec; j++ {
			body += "## Section " + strconv.Itoa(j) + "\n\nBody paragraph for record " +
				strconv.Itoa(i) + " section " + strconv.Itoa(j) +
				" with enough filler text to cross one chunk boundary " +
				"under the default markdown chunker.\n\n"
		}
		records = append(records, corpus.Record{
			Ref:        "rec-" + strconv.Itoa(i),
			Kind:       "guide",
			Title:      "Record " + strconv.Itoa(i),
			SourceRef:  "file://bench/rec-" + strconv.Itoa(i) + ".md",
			BodyFormat: corpus.FormatMarkdown,
			BodyText:   body,
		})
	}

	path := b.TempDir() + "/reuse.db"
	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		b.Fatal(err)
	}

	db, err := store.OpenReadOnlyContext(context.Background(), path)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = db.Close() }()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, rec := range records {
			chunks, err := loadStoredChunksForRecord(
				context.Background(),
				db,
				rec.Ref,
				rec.Title,
				store.QuantizationFloat32,
				true,
			)
			if err != nil {
				b.Fatal(err)
			}
			if len(chunks) == 0 {
				b.Fatalf("no chunks loaded for %s", rec.Ref)
			}
		}
	}
}
