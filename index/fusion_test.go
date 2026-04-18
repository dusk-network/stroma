package index

import (
	"context"
	"errors"
	"testing"

	"github.com/dusk-network/stroma/corpus"
	"github.com/dusk-network/stroma/embed"
	"github.com/dusk-network/stroma/store"
)

type capturingFusion struct {
	received []RetrievalArm
	result   []SearchHit
	err      error
}

func (c *capturingFusion) Fuse(arms []RetrievalArm, _ int) ([]SearchHit, error) {
	c.received = append(c.received[:0], arms...)
	if c.err != nil {
		return nil, c.err
	}
	return c.result, nil
}

type provenanceRecordingReranker struct {
	seenProvenance []*HitProvenance
}

func (r *provenanceRecordingReranker) Rerank(_ context.Context, _ string, candidates []SearchHit) ([]SearchHit, error) {
	for i := range candidates {
		r.seenProvenance = append(r.seenProvenance, candidates[i].Provenance)
	}
	return candidates, nil
}

func TestRRFFusionSingleArmPreservesArmScore(t *testing.T) {
	t.Parallel()

	vectorHit := SearchHit{ChunkID: 1, Ref: "vector-only", Score: 0.8}
	arms := []RetrievalArm{
		{Name: ArmVector, Available: true, Hits: []SearchHit{vectorHit}},
		{Name: ArmFTS, Available: true, Hits: nil},
	}
	hits, err := DefaultFusion().Fuse(arms, 5)
	if err != nil {
		t.Fatalf("Fuse() error = %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("len(hits) = %d, want 1", len(hits))
	}
	if hits[0].Score != 0.8 {
		t.Fatalf("Score = %v, want 0.8 (arm-native preserved)", hits[0].Score)
	}
	if hits[0].Provenance == nil {
		t.Fatalf("Provenance is nil on single-arm default fusion")
	}
	ev, ok := hits[0].Provenance.Arms[ArmVector]
	if !ok {
		t.Fatalf("single-arm hit missing ArmVector provenance")
	}
	if ev.Rank != 0 || ev.Score != 0.8 {
		t.Fatalf("vector evidence = %+v, want {Rank:0 Score:0.8}", ev)
	}
	if _, ok := hits[0].Provenance.Arms[ArmFTS]; ok {
		t.Fatalf("empty-arm ArmFTS appeared in Provenance, want absent")
	}
}

func TestRRFFusionSingleArmRewriteWhenOptingOut(t *testing.T) {
	t.Parallel()

	vectorHit := SearchHit{ChunkID: 1, Ref: "vector-only", Score: 0.8}
	arms := []RetrievalArm{
		{Name: ArmVector, Available: true, Hits: []SearchHit{vectorHit}},
		{Name: ArmFTS, Available: true, Hits: nil},
	}
	hits, err := (RRFFusion{K: 60, PreserveSingleArmScore: false}).Fuse(arms, 5)
	if err != nil {
		t.Fatalf("Fuse() error = %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("len(hits) = %d, want 1", len(hits))
	}
	want := 1.0 / float64(60+0+1)
	if hits[0].Score != want {
		t.Fatalf("Score = %v, want RRF-derived %v", hits[0].Score, want)
	}
	if hits[0].Provenance == nil {
		t.Fatalf("Provenance is nil")
	}
	if hits[0].Provenance.Arms[ArmVector].Score != 0.8 {
		t.Fatalf("arm-native score should still be in Provenance")
	}
}

func TestRRFFusionUnavailableArmWithHitsIsInvalid(t *testing.T) {
	t.Parallel()

	arms := []RetrievalArm{
		{Name: ArmVector, Available: false, Hits: []SearchHit{{ChunkID: 1}}},
	}
	if _, err := DefaultFusion().Fuse(arms, 5); err == nil {
		t.Fatalf("Fuse() error = nil, want validation error")
	}
}

func TestRRFFusionAvailableWithErrorIsInvalid(t *testing.T) {
	t.Parallel()

	arms := []RetrievalArm{
		{Name: ArmVector, Available: true, Err: errors.New("boom")},
	}
	if _, err := DefaultFusion().Fuse(arms, 5); err == nil {
		t.Fatalf("Fuse() error = nil, want validation error")
	}
}

func TestRRFFusionEmptyArmNameIsInvalid(t *testing.T) {
	t.Parallel()

	arms := []RetrievalArm{{Name: "", Available: true}}
	if _, err := DefaultFusion().Fuse(arms, 5); err == nil {
		t.Fatalf("Fuse() error = nil, want validation error")
	}
}

func TestRRFFusionDuplicateArmNameIsInvalid(t *testing.T) {
	t.Parallel()

	arms := []RetrievalArm{
		{Name: ArmVector, Available: true, Hits: []SearchHit{{ChunkID: 1, Score: 0.5}}},
		{Name: ArmVector, Available: true, Hits: []SearchHit{{ChunkID: 2, Score: 0.4}}},
	}
	if _, err := DefaultFusion().Fuse(arms, 5); err == nil {
		t.Fatalf("Fuse() error = nil, want duplicate-name validation error")
	}
}

func TestRRFFusionPropagatesArmError(t *testing.T) {
	t.Parallel()

	armErr := errors.New("vector arm exploded")
	arms := []RetrievalArm{
		{Name: ArmVector, Available: false, Err: armErr},
		{Name: ArmFTS, Available: true, Hits: nil},
	}
	_, err := DefaultFusion().Fuse(arms, 5)
	if err == nil {
		t.Fatalf("Fuse() error = nil, want propagated arm error")
	}
	if !errors.Is(err, armErr) {
		t.Fatalf("Fuse() error = %v, want wraps armErr", err)
	}
}

func TestRRFFusionAllArmsUnavailable(t *testing.T) {
	t.Parallel()

	arms := []RetrievalArm{
		{Name: ArmVector, Available: false},
		{Name: ArmFTS, Available: false},
	}
	hits, err := DefaultFusion().Fuse(arms, 5)
	if err != nil {
		t.Fatalf("Fuse() error = %v", err)
	}
	if len(hits) != 0 {
		t.Fatalf("len(hits) = %d, want 0", len(hits))
	}
}

// TestSearchCustomFusionReceivesArmsFromSnapshot proves that Snapshot.Search
// hands a custom FusionStrategy the arms it expects, with the right names
// and availability flags; and that the strategy's returned hits flow on to
// the reranker with the strategy's scoring.
func TestSearchCustomFusionReceivesArmsFromSnapshot(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       testNoteKind,
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content lives here",
	}}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}

	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	fusion := &capturingFusion{result: []SearchHit{{ChunkID: 42, Ref: "synthetic", Score: 9.9}}}
	rr := &provenanceRecordingReranker{}

	hits, err := snap.Search(context.Background(), SnapshotSearchQuery{
		Text:     "alpha content",
		Limit:    5,
		Embedder: fixture,
		Fusion:   fusion,
		Reranker: rr,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if len(fusion.received) != 2 {
		t.Fatalf("custom Fuse received %d arms, want 2", len(fusion.received))
	}
	if fusion.received[0].Name != ArmVector || !fusion.received[0].Available {
		t.Fatalf("arm[0] = %+v, want ArmVector Available=true", fusion.received[0])
	}
	if fusion.received[1].Name != ArmFTS || !fusion.received[1].Available {
		t.Fatalf("arm[1] = %+v, want ArmFTS Available=true", fusion.received[1])
	}
	if len(fusion.received[0].Hits) == 0 {
		t.Fatalf("vector arm delivered zero hits to custom fusion")
	}
	if len(hits) != 1 || hits[0].Ref != "synthetic" || hits[0].Score != 9.9 {
		t.Fatalf("custom fusion result did not flow through: hits = %+v", hits)
	}
	if len(rr.seenProvenance) != 1 || rr.seenProvenance[0] != nil {
		// The synthetic hit from capturingFusion has no Provenance because
		// the strategy did not attach one; this asserts rerank sees whatever
		// the strategy returned without tampering.
		t.Fatalf("reranker provenance snapshot = %+v, want [nil]", rr.seenProvenance)
	}
}

// TestSearchCustomFusionSeesLegacyNoFTSArmUnavailable proves that a custom
// FusionStrategy observes the FTS arm as Available=false on a snapshot
// without fts_chunks, so it can distinguish "unavailable" from "ran empty".
func TestSearchCustomFusionSeesLegacyNoFTSArmUnavailable(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       testNoteKind,
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}
	db, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	if _, err := db.ExecContext(context.Background(), `DROP TABLE fts_chunks`); err != nil {
		_ = db.Close()
		t.Fatalf("drop fts_chunks: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("db.Close() error = %v", err)
	}
	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	fusion := &capturingFusion{result: []SearchHit{}}
	_, err = snap.Search(context.Background(), SnapshotSearchQuery{
		Text:     "alpha content",
		Limit:    5,
		Embedder: fixture,
		Fusion:   fusion,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(fusion.received) != 2 {
		t.Fatalf("custom Fuse received %d arms, want 2", len(fusion.received))
	}
	if fusion.received[1].Name != ArmFTS {
		t.Fatalf("arm[1].Name = %q, want ArmFTS", fusion.received[1].Name)
	}
	if fusion.received[1].Available {
		t.Fatalf("legacy-no-FTS snapshot: ArmFTS.Available = true, want false")
	}
	if len(fusion.received[1].Hits) != 0 {
		t.Fatalf("legacy-no-FTS snapshot: ArmFTS.Hits len = %d, want 0", len(fusion.received[1].Hits))
	}
}

// TestSearchDefaultFusionPreservesVectorScoreOnLegacyNoFTS proves the
// backward-compat claim: on a snapshot without fts_chunks, DefaultFusion
// returns vector-only hits with their arm-native (cosine-derived) Score
// preserved, not rewritten to RRF.
func TestSearchDefaultFusionPreservesVectorScoreOnLegacyNoFTS(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}

	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       testNoteKind,
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "alpha content",
	}}, BuildOptions{Path: path, Embedder: fixture}); err != nil {
		t.Fatalf("Rebuild() error = %v", err)
	}
	db, err := store.OpenReadWriteContext(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenReadWriteContext() error = %v", err)
	}
	if _, err := db.ExecContext(context.Background(), `DROP TABLE fts_chunks`); err != nil {
		_ = db.Close()
		t.Fatalf("drop fts_chunks: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("db.Close() error = %v", err)
	}
	snap, err := OpenSnapshot(context.Background(), path)
	if err != nil {
		t.Fatalf("OpenSnapshot() error = %v", err)
	}
	t.Cleanup(func() { _ = snap.Close() })

	hits, err := snap.Search(context.Background(), SnapshotSearchQuery{
		Text:     "alpha content",
		Limit:    5,
		Embedder: fixture,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) == 0 {
		t.Fatalf("Search() returned 0 hits")
	}
	// On the legacy-no-FTS path, Score must be the arm-native cosine
	// derivative (positive, and not the RRF 1/(60+rank+1) pattern).
	rrfFirst := 1.0 / float64(60+0+1)
	if hits[0].Score == rrfFirst {
		t.Fatalf("hits[0].Score = %v matches RRF-derived score; byte-identity regression on legacy snapshots", hits[0].Score)
	}
	if hits[0].Score <= 0 {
		t.Fatalf("hits[0].Score = %v, want positive cosine-derived value", hits[0].Score)
	}
	// Provenance is populated on the fusion path, even for single-arm results.
	if hits[0].Provenance == nil {
		t.Fatalf("Provenance is nil; default fusion should always attach")
	}
	if _, ok := hits[0].Provenance.Arms[ArmVector]; !ok {
		t.Fatalf("legacy-no-FTS Provenance missing ArmVector evidence")
	}
	if _, ok := hits[0].Provenance.Arms[ArmFTS]; ok {
		t.Fatalf("legacy-no-FTS Provenance should not carry ArmFTS evidence")
	}
}

// TestSearchCustomFusionErrorPropagates proves that Snapshot.Search wraps a
// FusionStrategy error and returns no partial result.
func TestSearchCustomFusionErrorPropagates(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       testNoteKind,
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
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

	sentinel := errors.New("fusion strategy refused")
	hits, err := snap.Search(context.Background(), SnapshotSearchQuery{
		Text:     "alpha content",
		Limit:    5,
		Embedder: fixture,
		Fusion:   &capturingFusion{err: sentinel},
	})
	if err == nil {
		t.Fatal("Search() error = nil, want fusion error")
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("Search() error = %v, want wraps sentinel", err)
	}
	if hits != nil {
		t.Fatalf("Search() hits = %+v, want nil on fusion error", hits)
	}
}

// TestRerankerSeesProvenance proves the reranker observes SearchHit.Provenance
// attached by the default fusion strategy.
func TestRerankerSeesProvenance(t *testing.T) {
	t.Parallel()

	fixture, err := embed.NewFixture("fixture-a", 16)
	if err != nil {
		t.Fatalf("NewFixture() error = %v", err)
	}
	path := t.TempDir() + "/stroma.db"
	if _, err := Rebuild(context.Background(), []corpus.Record{{
		Ref:        testAlphaRef,
		Kind:       testNoteKind,
		Title:      "Alpha",
		SourceRef:  "file://alpha.txt",
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

	rr := &provenanceRecordingReranker{}
	if _, err := snap.Search(context.Background(), SnapshotSearchQuery{
		Text:     "alpha content",
		Limit:    5,
		Embedder: fixture,
		Reranker: rr,
	}); err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(rr.seenProvenance) == 0 {
		t.Fatalf("reranker saw no candidates")
	}
	for i, prov := range rr.seenProvenance {
		if prov == nil {
			t.Fatalf("rr.seenProvenance[%d] = nil; default fusion must attach Provenance", i)
		}
		if len(prov.Arms) == 0 {
			t.Fatalf("rr.seenProvenance[%d].Arms is empty", i)
		}
	}
}
