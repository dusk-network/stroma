package index

import (
	"bytes"
	"context"
	"strconv"
	"testing"

	"github.com/dusk-network/stroma/v2/corpus"
	"github.com/dusk-network/stroma/v2/embed"
	"github.com/dusk-network/stroma/v2/store"
)

// TestReuseScanBlobSurvivesNext pins the contract that D2 (#60) depends
// on: once `database/sql.Rows.Scan(&embedding)` with a `*[]byte`
// destination returns, the scanned buffer is caller-owned and must
// remain bit-identical across subsequent rows.Next() iterations. If the
// ncruces wasm driver ever bypasses the stdlib convertAssign path
// (e.g. via a future RawBytes shortcut) or a Go toolchain upgrade
// changes the clone semantics, this test fails and forces an explicit
// review of the D2 decision.
//
// Without this guard the defensive `append([]byte(nil), embedding...)`
// removed in #60 could be silently required again under a driver or
// stdlib change, and only show up as corrupted embeddings under very
// specific reuse patterns.
func TestReuseScanBlobSurvivesNext(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("scan-survives-next", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	// Build a record with enough body to produce multiple chunks so the
	// loadStoredChunksForRecord rows iterator actually steps more than
	// once inside a single query.
	body := ""
	for i := 0; i < 6; i++ {
		body += "## Section " + strconv.Itoa(i) + "\n\nScan-lifetime probe body " +
			strconv.Itoa(i) + " with enough filler to cross the default chunk boundary.\n\n"
	}
	records := []corpus.Record{{
		Ref:        "rec-scan",
		Kind:       "guide",
		Title:      "Scan Lifetime",
		SourceRef:  "file://scan.md",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   body,
	}}

	path := t.TempDir() + "/scan.db"
	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	db, err := store.OpenReadOnlyContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadOnlyContext() error = %v", err)
	}
	defer func() { _ = db.Close() }()

	// First pass: load the chunks through the same helper #60 modified.
	// Capture the scanned blobs and a deep-copy snapshot alongside them.
	chunks, err := loadStoredChunksForRecord(
		context.Background(),
		db,
		"rec-scan",
		"Scan Lifetime",
		store.QuantizationFloat32,
		true,
	)
	if err != nil {
		t.Fatalf("loadStoredChunksForRecord() error = %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("need at least 2 chunks to exercise rows.Next() lifetime; got %d", len(chunks))
	}

	snapshots := make(map[string][]byte, len(chunks))
	for k, v := range chunks {
		snapshots[k] = append([]byte(nil), v...)
	}

	// Run the helper a second time in the same handle. Both runs share
	// the db handle and the prepared statement pool, so if the driver
	// aliased any scanned buffer to wasm memory reused across Step()
	// calls, the second run would stomp the first run's blobs.
	chunks2, err := loadStoredChunksForRecord(
		context.Background(),
		db,
		"rec-scan",
		"Scan Lifetime",
		store.QuantizationFloat32,
		true,
	)
	if err != nil {
		t.Fatalf("loadStoredChunksForRecord() second call error = %v", err)
	}
	if len(chunks2) != len(chunks) {
		t.Fatalf("second call chunk count = %d, want %d", len(chunks2), len(chunks))
	}

	// Every blob captured in the first pass must still match its
	// snapshot byte-for-byte after the second pass has completed.
	for k, original := range chunks {
		if !bytes.Equal(original, snapshots[k]) {
			t.Fatalf("blob for %q mutated after second rows.Next() pass; D2's no-copy assumption is broken", k)
		}
	}
}
