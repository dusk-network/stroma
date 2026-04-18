package index

import (
	"context"
	"strings"
	"testing"

	"github.com/dusk-network/stroma/chunk"
	"github.com/dusk-network/stroma/corpus"
	"github.com/dusk-network/stroma/embed"
	"github.com/dusk-network/stroma/store"
)

// brokenForwardPolicy emits a section that points forward at a later
// slice index. The index session must reject this via
// validateLineageTopology so callers cannot land an unsatisfiable FK
// chain.
type brokenForwardPolicy struct{}

func (brokenForwardPolicy) Chunk(_ context.Context, _ corpus.Record) ([]chunk.SectionWithLineage, error) {
	return []chunk.SectionWithLineage{
		{Section: chunk.Section{Heading: "leaf-first", Body: "body"}, ParentIndex: 1},
		{Section: chunk.Section{Heading: "parent-second", Body: "parent body"}, ParentIndex: chunk.NoParent},
	}, nil
}

// selfReferencingPolicy emits a section pointing at itself. Forward-only
// validation rejects ParentIndex >= i, so self-reference is also caught.
type selfReferencingPolicy struct{}

func (selfReferencingPolicy) Chunk(_ context.Context, _ corpus.Record) ([]chunk.SectionWithLineage, error) {
	return []chunk.SectionWithLineage{
		{Section: chunk.Section{Heading: "self", Body: "body"}, ParentIndex: 0},
	}, nil
}

func TestRebuildRejectsForwardParentReference(t *testing.T) {
	t.Parallel()
	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	_, err = Rebuild(context.Background(), []corpus.Record{
		{Ref: "doc", Kind: "note", Title: "Doc", BodyFormat: corpus.FormatPlaintext, BodyText: "body"},
	}, BuildOptions{
		Path:        path,
		Embedder:    fixture,
		ChunkPolicy: brokenForwardPolicy{},
	})
	if err == nil {
		t.Fatal("Rebuild() with forward parent reference succeeded; want topology error")
	}
	if !strings.Contains(err.Error(), "forward parent reference") {
		t.Fatalf("err = %v, want message about forward parent reference", err)
	}
}

func TestRebuildRejectsSelfReferencingParent(t *testing.T) {
	t.Parallel()
	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	_, err = Rebuild(context.Background(), []corpus.Record{
		{Ref: "doc", Kind: "note", Title: "Doc", BodyFormat: corpus.FormatPlaintext, BodyText: "body"},
	}, BuildOptions{
		Path:        path,
		Embedder:    fixture,
		ChunkPolicy: selfReferencingPolicy{},
	})
	if err == nil {
		t.Fatal("Rebuild() with self-referencing parent succeeded; want topology error")
	}
	if !strings.Contains(err.Error(), "forward parent reference") {
		t.Fatalf("err = %v, want message about forward parent reference (self-ref triggers ParentIndex >= i)", err)
	}
}

// TestRebuildWithLateChunkPolicyEmitsHierarchy is the end-to-end happy
// path for PR-B: build a snapshot with LateChunkPolicy, assert that
// chunks rows carry parent_chunk_id linking leaves to the parent, and
// assert that only leaves are embedded and added to FTS (parents have
// a chunks row but no chunks_vec or fts_chunks row, so they cannot
// surface from either arm of hybrid search).
func TestRebuildWithLateChunkPolicyEmitsHierarchy(t *testing.T) {
	t.Parallel()
	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	body := "# Heading\n\n" + strings.Repeat("alpha bravo charlie delta ", 8)
	_, err = Rebuild(context.Background(), []corpus.Record{
		{Ref: "doc", Kind: "note", Title: "Doc", BodyFormat: corpus.FormatMarkdown, BodyText: body},
	}, BuildOptions{
		Path:     path,
		Embedder: fixture,
		ChunkPolicy: chunk.LateChunkPolicy{
			ChildMaxTokens:     6,
			ChildOverlapTokens: 1,
		},
	})
	if err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	checkDB, err := store.OpenReadOnlyContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadOnlyContext() error = %v", err)
	}
	defer func() { _ = checkDB.Close() }()

	var totalChunks, parentRows, leafRows int
	if err := checkDB.QueryRowContext(context.Background(), `SELECT
        COUNT(*),
        SUM(CASE WHEN parent_chunk_id IS NULL THEN 1 ELSE 0 END),
        SUM(CASE WHEN parent_chunk_id IS NOT NULL THEN 1 ELSE 0 END)
        FROM chunks`).Scan(&totalChunks, &parentRows, &leafRows); err != nil {
		t.Fatalf("count chunks: %v", err)
	}
	if parentRows < 1 {
		t.Fatalf("got %d parent rows (parent_chunk_id NULL), want >= 1", parentRows)
	}
	if leafRows < 1 {
		t.Fatalf("got %d leaf rows (parent_chunk_id NOT NULL), want >= 1", leafRows)
	}

	// chunks_vec has one row per leaf only — parents skip embedding.
	var vecRows int
	if err := checkDB.QueryRowContext(context.Background(), `SELECT COUNT(*) FROM chunks_vec`).Scan(&vecRows); err != nil {
		t.Fatalf("count chunks_vec: %v", err)
	}
	if vecRows != leafRows {
		t.Fatalf("chunks_vec has %d rows, want %d (one per leaf only — parents must skip embedding)", vecRows, leafRows)
	}

	// fts_chunks has one row per leaf only — parents are storage-only
	// context and must not surface from the FTS arm of hybrid search.
	// ExpandContext reads chunks directly, so it does not need the
	// parents in FTS.
	var ftsRows int
	if err := checkDB.QueryRowContext(context.Background(), `SELECT COUNT(*) FROM fts_chunks`).Scan(&ftsRows); err != nil {
		t.Fatalf("count fts_chunks: %v", err)
	}
	if ftsRows != leafRows {
		t.Fatalf("fts_chunks has %d rows, want %d (one per leaf; parents must not surface from FTS arm)", ftsRows, leafRows)
	}

	// Every leaf's parent_chunk_id resolves to a parent row in the same
	// record (the v5 same-record trigger guarantees this; the test
	// just sanity-checks the data shape).
	var dangling int
	if err := checkDB.QueryRowContext(context.Background(), `
SELECT COUNT(*) FROM chunks c
LEFT JOIN chunks p ON p.id = c.parent_chunk_id
WHERE c.parent_chunk_id IS NOT NULL
  AND (p.id IS NULL OR p.record_ref != c.record_ref)`).Scan(&dangling); err != nil {
		t.Fatalf("scan dangling parent_chunk_id: %v", err)
	}
	if dangling != 0 {
		t.Fatalf("found %d leaves whose parent_chunk_id does not resolve to a same-record parent", dangling)
	}
}

// TestSearchSurfacesLeavesOnlyUnderLateChunkPolicy verifies that a
// hybrid search against a LateChunkPolicy snapshot returns leaves only
// (parents participate in neither chunks_vec nor fts_chunks, so they
// cannot surface from either arm of hybrid retrieval). This is the
// design intent of "leaves embed, parents stay storage-only context."
func TestSearchSurfacesLeavesOnlyUnderLateChunkPolicy(t *testing.T) {
	t.Parallel()
	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	body := "# Heading\n\n" + strings.Repeat("alpha bravo charlie delta ", 8)
	if _, err := Rebuild(context.Background(), []corpus.Record{
		{Ref: "doc", Kind: "note", Title: "Doc", BodyFormat: corpus.FormatMarkdown, BodyText: body},
	}, BuildOptions{
		Path:        path,
		Embedder:    fixture,
		ChunkPolicy: chunk.LateChunkPolicy{ChildMaxTokens: 6, ChildOverlapTokens: 1},
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	hits, err := snap.Search(context.Background(), SnapshotSearchQuery{
		SearchParams: SearchParams{
			Text:     "alpha bravo",
			Limit:    10,
			Embedder: fixture,
		},
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("Search() returned 0 hits, want >= 1 (leaves carry the embedding)")
	}

	// Cross-check every hit against the chunks table: each must have a
	// non-NULL parent_chunk_id (i.e., be a leaf, not the parent row).
	rwDB, err := store.OpenReadOnlyContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadOnlyContext() error = %v", err)
	}
	defer func() { _ = rwDB.Close() }()
	for _, h := range hits {
		var parentNotNull int
		if err := rwDB.QueryRowContext(context.Background(),
			`SELECT CASE WHEN parent_chunk_id IS NOT NULL THEN 1 ELSE 0 END FROM chunks WHERE id = ?`, h.ChunkID).Scan(&parentNotNull); err != nil {
			t.Fatalf("scan parent_chunk_id for hit %d: %v", h.ChunkID, err)
		}
		if parentNotNull != 1 {
			t.Fatalf("Search returned chunk_id %d which has parent_chunk_id IS NULL (parent row); search must surface leaves only", h.ChunkID)
		}
	}
}

// TestExpandContextWalksParentForLateChunkLeaf is the end-to-end test
// for the ExpandContext-on-real-hierarchy path: build a LateChunkPolicy
// snapshot, search for one of its leaves, then ExpandContext with
// IncludeParent. Result must include the parent row (the leaf's
// designed retrieval surface for "broader context").
func TestExpandContextWalksParentForLateChunkLeaf(t *testing.T) {
	t.Parallel()
	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	body := "# Heading\n\n" + strings.Repeat("alpha bravo charlie delta ", 8)
	if _, err := Rebuild(context.Background(), []corpus.Record{
		{Ref: "doc", Kind: "note", Title: "Doc", BodyFormat: corpus.FormatMarkdown, BodyText: body},
	}, BuildOptions{
		Path:        path,
		Embedder:    fixture,
		ChunkPolicy: chunk.LateChunkPolicy{ChildMaxTokens: 6, ChildOverlapTokens: 1},
	}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	hits, err := snap.Search(context.Background(), SnapshotSearchQuery{
		SearchParams: SearchParams{
			Text:     "alpha bravo",
			Limit:    1,
			Embedder: fixture,
		},
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("Search() returned 0 hits")
	}
	leafID := hits[0].ChunkID

	got, err := snap.ExpandContext(context.Background(), leafID, ContextOptions{IncludeParent: true})
	if err != nil {
		t.Fatalf("ExpandContext(IncludeParent) error = %v", err)
	}
	if len(got) < 2 {
		t.Fatalf("ExpandContext returned %d sections, want >= 2 (parent + leaf)", len(got))
	}
	// Document order: parent first.
	if got[0].ChunkID == leafID {
		t.Fatal("ExpandContext returned the leaf first; parent must precede leaf in document order")
	}
}

// TestRebuildPreservesByteCompatibilityWithoutChunkPolicy pins the
// default-behavior contract for PR-B: building with
// BuildOptions{ChunkPolicy: nil} produces a snapshot whose chunks
// rows (count, content, parent_chunk_id all-NULL) match what PR-A
// already produced. PR-B is strictly additive when no policy is set.
func TestRebuildPreservesByteCompatibilityWithoutChunkPolicy(t *testing.T) {
	t.Parallel()
	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	body := "# A\n\none two three\n\n# B\n\nfour five six seven eight\n\n# C\n\nnine ten\n"
	records := []corpus.Record{
		{Ref: "doc", Kind: "note", Title: "Doc", BodyFormat: corpus.FormatMarkdown, BodyText: body},
	}
	pathA := t.TempDir() + "/a.db"
	pathB := t.TempDir() + "/b.db"

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:           pathA,
		Embedder:       fixture,
		MaxChunkTokens: 4,
	}); err != nil {
		t.Fatalf("Rebuild(no policy) error = %v", err)
	}
	// Since #62 F2, setting both ChunkPolicy and the flat chunking
	// knobs is an explicit error (the flat knobs would be silently
	// ignored). Pass MaxTokens only through the policy here; the
	// byte-equivalence claim is still exercised because the first
	// Rebuild above uses the flat MaxChunkTokens path and this second
	// one uses the equivalent ChunkPolicy path.
	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path:        pathB,
		Embedder:    fixture,
		ChunkPolicy: chunk.MarkdownPolicy{Options: chunk.Options{MaxTokens: 4}},
	}); err != nil {
		t.Fatalf("Rebuild(explicit MarkdownPolicy) error = %v", err)
	}

	snapA, err := OpenSnapshot(context.Background(), pathA)
	if err != nil {
		t.Fatalf("OpenSnapshot(A) error = %v", err)
	}
	defer func() { _ = snapA.Close() }()
	snapB, err := OpenSnapshot(context.Background(), pathB)
	if err != nil {
		t.Fatalf("OpenSnapshot(B) error = %v", err)
	}
	defer func() { _ = snapB.Close() }()

	secsA, err := snapA.Sections(context.Background(), SectionQuery{})
	if err != nil {
		t.Fatalf("Sections(A) error = %v", err)
	}
	secsB, err := snapB.Sections(context.Background(), SectionQuery{})
	if err != nil {
		t.Fatalf("Sections(B) error = %v", err)
	}
	if len(secsA) != len(secsB) {
		t.Fatalf("A: %d sections, B: %d sections — explicit MarkdownPolicy did not preserve byte-equivalence",
			len(secsA), len(secsB))
	}
	for i := range secsA {
		if secsA[i].Heading != secsB[i].Heading || secsA[i].Content != secsB[i].Content {
			t.Fatalf("section %d differs: A=%q/%q B=%q/%q",
				i, secsA[i].Heading, secsA[i].Content, secsB[i].Heading, secsB[i].Content)
		}
	}

	// And neither snapshot should have any non-NULL parent_chunk_id
	// (default policy never emits parents).
	for _, p := range []string{pathA, pathB} {
		db, err := store.OpenReadOnlyContext(context.Background(), p)
		if err != nil {
			t.Fatalf("OpenReadOnlyContext(%s) error = %v", p, err)
		}
		var notNull int
		if err := db.QueryRowContext(context.Background(),
			`SELECT COUNT(*) FROM chunks WHERE parent_chunk_id IS NOT NULL`).Scan(&notNull); err != nil {
			_ = db.Close()
			t.Fatalf("count parent_chunk_id IS NOT NULL on %s: %v", p, err)
		}
		_ = db.Close()
		if notNull != 0 {
			t.Fatalf("snapshot %s has %d parent_chunk_id IS NOT NULL rows; default MarkdownPolicy must emit only flat chunks", p, notNull)
		}
	}
}

// TestRebuildRejectsChunkPolicyWithLegacyKnobs pins the #62 F2 fix:
// populating both BuildOptions.ChunkPolicy and any of the flat
// chunking knobs (MaxChunkTokens / ChunkOverlapTokens /
// MaxChunkSections) used to silently drop the knob value, which was
// a quiet foot-gun for callers who populated the struct by hand.
// Rebuild and Update must now reject the conflicting shape loudly.
func TestRebuildRejectsChunkPolicyWithLegacyKnobs(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	records := []corpus.Record{
		{Ref: "doc", Kind: "note", Title: "Doc", BodyFormat: corpus.FormatMarkdown, BodyText: "alpha bravo charlie"},
	}
	cases := []struct {
		name string
		opts BuildOptions
	}{
		{
			name: "MaxChunkTokens",
			opts: BuildOptions{Embedder: fixture, MaxChunkTokens: 4, ChunkPolicy: chunk.MarkdownPolicy{}},
		},
		{
			name: "ChunkOverlapTokens",
			opts: BuildOptions{Embedder: fixture, ChunkOverlapTokens: 2, ChunkPolicy: chunk.MarkdownPolicy{}},
		},
		{
			name: "MaxChunkSections",
			opts: BuildOptions{Embedder: fixture, MaxChunkSections: 5, ChunkPolicy: chunk.MarkdownPolicy{}},
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			tc.opts.Path = t.TempDir() + "/stroma.db"
			_, err := Rebuild(context.Background(), records, tc.opts)
			if err == nil {
				t.Fatalf("Rebuild() with %s + ChunkPolicy succeeded, want validation error", tc.name)
			}
			if !strings.Contains(err.Error(), tc.name) {
				t.Errorf("error does not mention the conflicting knob %q: %v", tc.name, err)
			}
		})
	}
}

// TestUpdateRejectsChunkPolicyWithLegacyKnobs mirrors the Rebuild
// regression for the UpdateOptions shape. The incremental-update path
// shares the same flat-knob vs ChunkPolicy ambiguity and must reject
// the same conflicting combinations. This pins that behavior directly
// so Rebuild's coverage does not silently imply Update's.
func TestUpdateRejectsChunkPolicyWithLegacyKnobs(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{
		{Ref: "seed", Kind: "note", Title: "Seed", BodyFormat: corpus.FormatMarkdown, BodyText: "seed body"},
	}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() seed error = %v", err)
	}

	cases := []struct {
		name string
		opts UpdateOptions
	}{
		{
			name: "MaxChunkTokens",
			opts: UpdateOptions{Path: path, Embedder: fixture, MaxChunkTokens: 4, ChunkPolicy: chunk.MarkdownPolicy{}},
		},
		{
			name: "ChunkOverlapTokens",
			opts: UpdateOptions{Path: path, Embedder: fixture, ChunkOverlapTokens: 2, ChunkPolicy: chunk.MarkdownPolicy{}},
		},
		{
			name: "MaxChunkSections",
			opts: UpdateOptions{Path: path, Embedder: fixture, MaxChunkSections: 5, ChunkPolicy: chunk.MarkdownPolicy{}},
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := Update(context.Background(), []corpus.Record{
				{Ref: "added", Kind: "note", Title: "Added", BodyFormat: corpus.FormatMarkdown, BodyText: "added body"},
			}, nil, tc.opts)
			if err == nil {
				t.Fatalf("Update() with %s + ChunkPolicy succeeded, want validation error", tc.name)
			}
			if !strings.Contains(err.Error(), tc.name) {
				t.Errorf("error does not mention the conflicting knob %q: %v", tc.name, err)
			}
		})
	}
}

// TestRebuildBinaryWithLateChunkPolicyOpensAndSearches is a Codex
// follow-up regression. checkChunksVecFullCompleteness used to require
// every chunks row to have a chunks_vec_full companion, which broke
// hierarchical binary snapshots: parent rows are storage-only context
// and intentionally have no companion vector. The fix exempts parent
// rows from the completeness check; this test pins the contract
// end-to-end by building a binary + LateChunkPolicy snapshot,
// reopening it (where the check fires), and exercising both Search
// and Sections(IncludeEmbeddings=true).
func TestRebuildBinaryWithLateChunkPolicyOpensAndSearches(t *testing.T) {
	t.Parallel()
	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	body := "# Heading\n\n" + strings.Repeat("alpha bravo charlie delta ", 8)
	if _, err := Rebuild(context.Background(), []corpus.Record{
		{Ref: "doc", Kind: "note", Title: "Doc", BodyFormat: corpus.FormatMarkdown, BodyText: body},
	}, BuildOptions{
		Path:         path,
		Embedder:     fixture,
		Quantization: store.QuantizationBinary,
		ChunkPolicy:  chunk.LateChunkPolicy{ChildMaxTokens: 6, ChildOverlapTokens: 1},
	}); err != nil {
		t.Fatalf("Rebuild(binary + LateChunkPolicy) error = %v", err)
	}

	// OpenSnapshot runs checkChunksVecFullCompleteness — without the
	// parent exemption this would fail with "N chunk(s) missing from
	// chunks_vec_full."
	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot(binary + LateChunkPolicy) error = %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	hits, err := snap.Search(context.Background(), SnapshotSearchQuery{
		SearchParams: SearchParams{
			Text:     "alpha bravo",
			Limit:    5,
			Embedder: fixture,
		},
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("Search() returned 0 hits on binary + LateChunkPolicy snapshot")
	}

	// Sections(IncludeEmbeddings=true) must return leaves with
	// populated Embedding via the chunks_vec_full join. Parents are
	// silently filtered (no companion row).
	secs, err := snap.Sections(context.Background(), SectionQuery{IncludeEmbeddings: true})
	if err != nil {
		t.Fatalf("Sections(IncludeEmbeddings) error = %v", err)
	}
	if len(secs) == 0 {
		t.Fatal("Sections(IncludeEmbeddings) returned 0; binary leaves should be visible")
	}
	for _, s := range secs {
		if len(s.Embedding) == 0 {
			t.Fatalf("section %d has empty Embedding under IncludeEmbeddings=true", s.ChunkID)
		}
	}
}

// TestReuseHonorsLateChunkPolicyLeaves rebuilds the same record body
// against itself with LateChunkPolicy. The leaves' (heading, content,
// prefix) keys are unchanged across rebuilds, so the leaves' embeddings
// are reused. Parents are not embedded so they cannot be "reused" in
// the embedding sense; they get rewritten fresh on every rebuild.
func TestReuseHonorsLateChunkPolicyLeaves(t *testing.T) {
	t.Parallel()
	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	src := t.TempDir() + "/src.db"
	dst := t.TempDir() + "/dst.db"

	body := "# Heading\n\n" + strings.Repeat("alpha bravo charlie delta ", 8)
	records := []corpus.Record{
		{Ref: "doc", Kind: "note", Title: "Doc", BodyFormat: corpus.FormatMarkdown, BodyText: body},
	}
	policy := chunk.LateChunkPolicy{ChildMaxTokens: 6, ChildOverlapTokens: 1}

	if _, err := Rebuild(context.Background(), records, BuildOptions{
		Path: src, Embedder: fixture, ChunkPolicy: policy,
	}); err != nil {
		t.Fatalf("Rebuild(src) error = %v", err)
	}

	res, err := Rebuild(context.Background(), records, BuildOptions{
		Path: dst, Embedder: fixture, ChunkPolicy: policy, ReuseFromPath: src,
	})
	if err != nil {
		t.Fatalf("Rebuild(dst, reuse) error = %v", err)
	}

	// Every leaf must have been reused; parents are not embedded so
	// they are not counted in either reused or embedded counters.
	if res.EmbeddedChunkCount != 0 {
		t.Fatalf("EmbeddedChunkCount = %d after reuse, want 0 (every leaf should reuse from prior snapshot)",
			res.EmbeddedChunkCount)
	}
	if res.ReusedChunkCount == 0 {
		t.Fatalf("ReusedChunkCount = 0 after reuse, want > 0 (LateChunkPolicy leaves must be reused)")
	}
}
