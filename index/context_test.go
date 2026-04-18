package index

import (
	"context"
	"strings"
	"testing"

	"github.com/dusk-network/stroma/v2/corpus"
	"github.com/dusk-network/stroma/v2/embed"
	"github.com/dusk-network/stroma/v2/store"
)

// TestExpandContextNonExistentChunkReturnsEmpty locks the contract that
// ExpandContext treats "no such chunk" as an empty result + nil error,
// matching the section-read APIs. A typed error here would force every
// caller to wrap a not-found check around the expansion call.
func TestExpandContextNonExistentChunkReturnsEmpty(t *testing.T) {
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
		SourceRef:  "a",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	got, err := snap.ExpandContext(context.Background(), 999_999, ContextOptions{
		IncludeParent:  true,
		NeighborWindow: 5,
	})
	if err != nil {
		t.Fatalf("ExpandContext(missing id) err = %v, want nil", err)
	}
	if len(got) != 0 {
		t.Fatalf("ExpandContext(missing id) returned %d sections, want 0", len(got))
	}
}

// TestExpandContextZeroOptionsReturnsJustTheChunk pins the zero-value
// behavior: no parent walk, no neighbors, just the chunk itself echoed
// back. Callers should be able to ask for "give me what you have for
// this chunk id" without thinking about lineage.
func TestExpandContextZeroOptionsReturnsJustTheChunk(t *testing.T) {
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
		SourceRef:  "a",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	chunkID := firstChunkID(t, snap)
	got, err := snap.ExpandContext(context.Background(), chunkID, ContextOptions{})
	if err != nil {
		t.Fatalf("ExpandContext(zero opts) error = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("ExpandContext(zero opts) returned %d sections, want 1", len(got))
	}
	if got[0].ChunkID != chunkID {
		t.Fatalf("ExpandContext(zero opts) returned chunk_id %d, want %d", got[0].ChunkID, chunkID)
	}
	if got[0].Embedding != nil {
		t.Fatalf("ExpandContext returned a non-nil Embedding (%d floats); embeddings must never be populated", len(got[0].Embedding))
	}
}

// TestExpandContextNeighborWindowOnFlatChunks builds a small Markdown
// record that produces multiple flat chunks (parent_chunk_id NULL on
// every row, the only shape PR-A actually emits) and asserts that
// NeighborWindow includes exactly the requested chunk_index neighbors,
// in document order.
func TestExpandContextNeighborWindowOnFlatChunks(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	body := "# Alpha\n\nintro\n\n# Beta\n\nbody two\n\n# Gamma\n\nbody three\n\n# Delta\n\nbody four\n\n# Epsilon\n\nbody five\n"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        "doc",
		Kind:       "note",
		Title:      "Doc",
		SourceRef:  "doc.md",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   body,
	}}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	if !snap.hasParentChunkID {
		t.Fatal("freshly-built snapshot should report hasParentChunkID = true")
	}

	allSections, err := snap.Sections(context.Background(), SectionQuery{Refs: []string{"doc"}})
	if err != nil {
		t.Fatalf("Sections() error = %v", err)
	}
	if len(allSections) < 5 {
		t.Fatalf("Sections() returned %d sections, want at least 5 for the 5-heading body", len(allSections))
	}

	// Pick the middle chunk and ask for 1 neighbor on each side.
	mid := allSections[2]
	got, err := snap.ExpandContext(context.Background(), mid.ChunkID, ContextOptions{NeighborWindow: 1})
	if err != nil {
		t.Fatalf("ExpandContext(NeighborWindow=1) error = %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("ExpandContext(NeighborWindow=1) returned %d sections, want 3", len(got))
	}
	for i, want := range []int64{allSections[1].ChunkID, allSections[2].ChunkID, allSections[3].ChunkID} {
		if got[i].ChunkID != want {
			t.Fatalf("ExpandContext sections[%d].ChunkID = %d, want %d (document order)", i, got[i].ChunkID, want)
		}
	}

	// IncludeParent on a flat chunk has no effect.
	gotWithParent, err := snap.ExpandContext(context.Background(), mid.ChunkID, ContextOptions{
		IncludeParent:  true,
		NeighborWindow: 1,
	})
	if err != nil {
		t.Fatalf("ExpandContext(IncludeParent on flat chunk) error = %v", err)
	}
	if len(gotWithParent) != len(got) {
		t.Fatalf("IncludeParent changed result on flat chunk: was %d sections, now %d", len(got), len(gotWithParent))
	}
}

// TestExpandContextWalksParentChunkID inserts a synthetic parent →
// child relationship via direct SQL (PR-A doesn't ship a Policy that
// emits parents; that lands in PR-B) and asserts the parent walk +
// sibling grouping behave correctly.
func TestExpandContextWalksParentChunkID(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	body := "# A\n\none\n\n# B\n\ntwo\n\n# C\n\nthree\n"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        "doc",
		Kind:       "note",
		Title:      "Doc",
		SourceRef:  "doc.md",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   body,
	}}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	// Pick the first chunk as the synthetic parent and re-parent the rest
	// onto it. This mimics the storage shape PR-B's LateChunkPolicy will
	// produce: one parent row, multiple leaves pointing at it.
	rwDB, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	rows, err := rwDB.QueryContext(context.Background(),
		`SELECT id FROM chunks WHERE record_ref = 'doc' ORDER BY chunk_index ASC`)
	if err != nil {
		_ = rwDB.Close()
		t.Fatalf("query chunk ids: %v", err)
	}
	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			_ = rows.Close()
			_ = rwDB.Close()
			t.Fatalf("scan id: %v", err)
		}
		ids = append(ids, id)
	}
	_ = rows.Close()
	if len(ids) < 3 {
		_ = rwDB.Close()
		t.Fatalf("got %d chunks, want at least 3", len(ids))
	}
	parentID := ids[0]
	for _, leaf := range ids[1:] {
		if _, err := rwDB.ExecContext(context.Background(),
			`UPDATE chunks SET parent_chunk_id = ? WHERE id = ?`, parentID, leaf); err != nil {
			_ = rwDB.Close()
			t.Fatalf("set parent_chunk_id: %v", err)
		}
	}
	if err := rwDB.Close(); err != nil {
		t.Fatalf("rwDB.Close() error = %v", err)
	}

	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	// IncludeParent on a leaf returns parent + leaf in document order.
	leaf := ids[1]
	got, err := snap.ExpandContext(context.Background(), leaf, ContextOptions{IncludeParent: true})
	if err != nil {
		t.Fatalf("ExpandContext(IncludeParent) error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("ExpandContext(IncludeParent) returned %d sections, want 2 (parent + leaf)", len(got))
	}
	if got[0].ChunkID != parentID {
		t.Fatalf("ExpandContext result[0].ChunkID = %d, want parent %d", got[0].ChunkID, parentID)
	}
	if got[1].ChunkID != leaf {
		t.Fatalf("ExpandContext result[1].ChunkID = %d, want leaf %d", got[1].ChunkID, leaf)
	}

	// NeighborWindow on a leaf returns leaves under the same parent only
	// (siblings share parent_chunk_id). With 2 sibling leaves total, a
	// window of 5 still only yields 2 sections.
	gotSiblings, err := snap.ExpandContext(context.Background(), leaf, ContextOptions{NeighborWindow: 5})
	if err != nil {
		t.Fatalf("ExpandContext(NeighborWindow=5) error = %v", err)
	}
	if len(gotSiblings) != len(ids)-1 {
		t.Fatalf("ExpandContext(NeighborWindow=5) returned %d sections, want %d (sibling leaves only)",
			len(gotSiblings), len(ids)-1)
	}
	for _, s := range gotSiblings {
		if s.ChunkID == parentID {
			t.Fatalf("ExpandContext(NeighborWindow only) returned the parent (id=%d); without IncludeParent, parent must be excluded", parentID)
		}
	}
}

// TestExpandContextOnLegacyV4SnapshotDegradesGracefully asserts that
// ExpandContext stays useful against pre-v5 snapshots: IncludeParent is
// a no-op (no column to walk), NeighborWindow falls back to record_ref-
// scoped grouping (no parent grouping is available).
func TestExpandContextOnLegacyV4SnapshotDegradesGracefully(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	body := "# A\n\none\n\n# B\n\ntwo\n\n# C\n\nthree\n"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        "doc",
		Kind:       "note",
		Title:      "Doc",
		SourceRef:  "doc.md",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   body,
	}}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}
	if err := rewriteSnapshotToV4(path); err != nil {
		t.Fatalf("rewriteSnapshotToV4() error = %v", err)
	}

	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot(v4) error = %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })
	if snap.hasParentChunkID {
		t.Fatal("v4 snapshot should report hasParentChunkID = false")
	}

	all, err := snap.Sections(context.Background(), SectionQuery{Refs: []string{"doc"}})
	if err != nil {
		t.Fatalf("Sections() error = %v", err)
	}
	if len(all) < 3 {
		t.Fatalf("Sections() returned %d, want >= 3", len(all))
	}

	// IncludeParent + NeighborWindow on the middle chunk: parent walk
	// is a no-op, neighbors come from the record_ref-scoped fallback.
	mid := all[1]
	got, err := snap.ExpandContext(context.Background(), mid.ChunkID, ContextOptions{
		IncludeParent:  true,
		NeighborWindow: 1,
	})
	if err != nil {
		t.Fatalf("ExpandContext on v4 snapshot error = %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("ExpandContext on v4 returned %d sections, want 3 (mid + one on each side via record_ref fallback)", len(got))
	}
	if got[1].ChunkID != mid.ChunkID {
		t.Fatalf("ExpandContext on v4 mid section = %d, want %d", got[1].ChunkID, mid.ChunkID)
	}
}

// TestParentChunkIDRejectsCrossRecordLineage proves the v5 same-record
// invariant: SQLite triggers reject any chunk whose parent_chunk_id
// points at a row in a different record. The blast radius without this
// guard is real — ExpandContext would walk a parent across record
// boundaries (privacy/isolation leak), and DELETE FROM records would
// CASCADE-delete child chunks while leaving their chunks_vec /
// fts_chunks rows orphaned (silent corruption).
//
// The test installs a synthetic cross-record link via direct SQL on a
// freshly-built snapshot and asserts both an INSERT-time and an
// UPDATE-time attempt are aborted.
func TestParentChunkIDRejectsCrossRecordLineage(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{
		{Ref: "doc-a", Kind: "note", Title: "A", SourceRef: "a", BodyFormat: corpus.FormatPlaintext, BodyText: "alpha"},
		{Ref: "doc-b", Kind: "note", Title: "B", SourceRef: "b", BodyFormat: corpus.FormatPlaintext, BodyText: "bravo"},
	}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	rwDB, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	defer func() { _ = rwDB.Close() }()

	var aChunk, bChunk int64
	if err := rwDB.QueryRowContext(context.Background(),
		`SELECT id FROM chunks WHERE record_ref = 'doc-a' LIMIT 1`).Scan(&aChunk); err != nil {
		t.Fatalf("read doc-a chunk: %v", err)
	}
	if err := rwDB.QueryRowContext(context.Background(),
		`SELECT id FROM chunks WHERE record_ref = 'doc-b' LIMIT 1`).Scan(&bChunk); err != nil {
		t.Fatalf("read doc-b chunk: %v", err)
	}

	// UPDATE: pointing doc-b's chunk at doc-a's chunk must abort.
	_, err = rwDB.ExecContext(context.Background(),
		`UPDATE chunks SET parent_chunk_id = ? WHERE id = ?`, aChunk, bChunk)
	if err == nil {
		t.Fatal("UPDATE setting cross-record parent_chunk_id succeeded; same-record trigger missing")
	}
	if !strings.Contains(err.Error(), "same record_ref") {
		t.Fatalf("UPDATE error = %v, want message mentioning same record_ref", err)
	}

	// INSERT: a fresh chunk under doc-b pointing at doc-a's chunk must abort.
	_, err = rwDB.ExecContext(context.Background(),
		`INSERT INTO chunks (record_ref, chunk_index, heading, content, parent_chunk_id) VALUES (?, ?, ?, ?, ?)`,
		"doc-b", 99, "synthetic", "synthetic body", aChunk)
	if err == nil {
		t.Fatal("INSERT with cross-record parent_chunk_id succeeded; same-record trigger missing")
	}
	if !strings.Contains(err.Error(), "same record_ref") {
		t.Fatalf("INSERT error = %v, want message mentioning same record_ref", err)
	}

	// Same-record link still works (sanity check that the trigger does
	// not over-reject the legitimate path PR-B will exercise).
	if _, err := rwDB.ExecContext(context.Background(),
		`INSERT INTO chunks (record_ref, chunk_index, heading, content, parent_chunk_id) VALUES (?, ?, ?, ?, ?)`,
		"doc-a", 99, "child", "child body", aChunk); err != nil {
		t.Fatalf("INSERT with same-record parent_chunk_id failed: %v (trigger over-rejects)", err)
	}
}

// TestExpandContextRejectsCrossRecordParentInLoad is defense in depth.
// The v5 same-record triggers (#16) reject cross-record parent links on
// INSERT/UPDATE, but a malformed or tampered snapshot could still carry
// one. ExpandContext must refuse to surface a parent that does not
// share the requested chunk's record_ref. This test bypasses the
// triggers (DROP TRIGGER + UPDATE), then asserts ExpandContext silently
// drops the cross-record parent — caller sees no parent in the result,
// no error, no leak.
func TestExpandContextRejectsCrossRecordParentInLoad(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{
		{Ref: "doc-a", Kind: "note", Title: "A", SourceRef: "a", BodyFormat: corpus.FormatPlaintext, BodyText: "alpha"},
		{Ref: "doc-b", Kind: "note", Title: "B", SourceRef: "b", BodyFormat: corpus.FormatPlaintext, BodyText: "bravo"},
	}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	rwDB, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	// Bypass the same-record triggers so we can plant a malformed link
	// that simulates a tampered snapshot.
	for _, stmt := range []string{
		`DROP TRIGGER IF EXISTS chunks_parent_same_record_insert`,
		`DROP TRIGGER IF EXISTS chunks_parent_same_record_update`,
	} {
		if _, err := rwDB.ExecContext(context.Background(), stmt); err != nil {
			_ = rwDB.Close()
			t.Fatalf("drop trigger: %v", err)
		}
	}
	var aChunk, bChunk int64
	if err := rwDB.QueryRowContext(context.Background(),
		`SELECT id FROM chunks WHERE record_ref = 'doc-a' LIMIT 1`).Scan(&aChunk); err != nil {
		_ = rwDB.Close()
		t.Fatalf("read doc-a chunk: %v", err)
	}
	if err := rwDB.QueryRowContext(context.Background(),
		`SELECT id FROM chunks WHERE record_ref = 'doc-b' LIMIT 1`).Scan(&bChunk); err != nil {
		_ = rwDB.Close()
		t.Fatalf("read doc-b chunk: %v", err)
	}
	if _, err := rwDB.ExecContext(context.Background(),
		`UPDATE chunks SET parent_chunk_id = ? WHERE id = ?`, aChunk, bChunk); err != nil {
		_ = rwDB.Close()
		t.Fatalf("plant cross-record parent: %v", err)
	}
	if err := rwDB.Close(); err != nil {
		t.Fatalf("close rwDB: %v", err)
	}

	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	got, err := snap.ExpandContext(context.Background(), bChunk, ContextOptions{IncludeParent: true})
	if err != nil {
		t.Fatalf("ExpandContext(IncludeParent on cross-record parent) error = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("ExpandContext returned %d sections, want 1 (just the chunk; cross-record parent must be dropped)", len(got))
	}
	if got[0].ChunkID != bChunk {
		t.Fatalf("ExpandContext returned ChunkID %d, want %d", got[0].ChunkID, bChunk)
	}
	if got[0].Ref != "doc-b" {
		t.Fatalf("ExpandContext returned Ref %q, want %q", got[0].Ref, "doc-b")
	}
}

// TestMigrateV4ToV5InstallsSameRecordTrigger proves upgraded snapshots
// also get the same-record lineage trigger. Without this, a v4 file
// migrated forward could accept cross-record parents from any later
// writer that runs against the upgraded handle.
func TestMigrateV4ToV5InstallsSameRecordTrigger(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{
		{Ref: "doc-a", Kind: "note", Title: "A", SourceRef: "a", BodyFormat: corpus.FormatPlaintext, BodyText: "alpha"},
		{Ref: "doc-b", Kind: "note", Title: "B", SourceRef: "b", BodyFormat: corpus.FormatPlaintext, BodyText: "bravo"},
	}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}
	if err := rewriteSnapshotToV4(path); err != nil {
		t.Fatalf("rewriteSnapshotToV4() error = %v", err)
	}
	if _, err := Update(context.Background(), nil, nil, UpdateOptions{
		Path:     path,
		Embedder: fixture,
	}); err != nil {
		t.Fatalf("Update(v4→v5) error = %v", err)
	}

	rwDB, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	defer func() { _ = rwDB.Close() }()

	var aChunk, bChunk int64
	if err := rwDB.QueryRowContext(context.Background(),
		`SELECT id FROM chunks WHERE record_ref = 'doc-a' LIMIT 1`).Scan(&aChunk); err != nil {
		t.Fatalf("read doc-a chunk: %v", err)
	}
	if err := rwDB.QueryRowContext(context.Background(),
		`SELECT id FROM chunks WHERE record_ref = 'doc-b' LIMIT 1`).Scan(&bChunk); err != nil {
		t.Fatalf("read doc-b chunk: %v", err)
	}
	_, err = rwDB.ExecContext(context.Background(),
		`UPDATE chunks SET parent_chunk_id = ? WHERE id = ?`, aChunk, bChunk)
	if err == nil {
		t.Fatal("UPDATE setting cross-record parent_chunk_id on upgraded snapshot succeeded; trigger missing on migration path")
	}
	if !strings.Contains(err.Error(), "same record_ref") {
		t.Fatalf("UPDATE error = %v, want message mentioning same record_ref", err)
	}
}

// firstChunkID returns one chunk_id from the snapshot. The snapshot must
// have at least one chunk.
func firstChunkID(t *testing.T, snap *Snapshot) int64 {
	t.Helper()
	row := snap.db.QueryRowContext(context.Background(), `SELECT id FROM chunks ORDER BY id ASC LIMIT 1`)
	var id int64
	if err := row.Scan(&id); err != nil {
		t.Fatalf("read first chunk_id: %v", err)
	}
	return id
}

// TestRebuildFreshSnapshotHasParentChunkIDColumnAllNULL pins the
// default-behavior-preservation contract: PR-A adds parent_chunk_id but
// no policy in PR-A emits parents, so every freshly-built chunk must
// have parent_chunk_id IS NULL. PR-B then activates non-NULL values
// behind ChunkPolicy opt-in.
func TestRebuildFreshSnapshotHasParentChunkIDColumnAllNULL(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{
		{Ref: "a", Kind: "note", Title: "A", SourceRef: "a", BodyFormat: corpus.FormatPlaintext, BodyText: "a"},
		{Ref: "b", Kind: "note", Title: "B", SourceRef: "b", BodyFormat: corpus.FormatPlaintext, BodyText: "b"},
	}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
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
	if strings.TrimSpace(schema) != schemaVersion {
		t.Fatalf("schema_version = %q on fresh snapshot, want %q", schema, schemaVersion)
	}

	var notNullCount int
	if err := checkDB.QueryRowContext(context.Background(),
		`SELECT COUNT(*) FROM chunks WHERE parent_chunk_id IS NOT NULL`).Scan(&notNullCount); err != nil {
		t.Fatalf("count parent_chunk_id IS NOT NULL: %v", err)
	}
	if notNullCount != 0 {
		t.Fatalf("fresh snapshot has %d chunk(s) with parent_chunk_id IS NOT NULL, want 0 (no PR-A policy emits parents)", notNullCount)
	}
}
