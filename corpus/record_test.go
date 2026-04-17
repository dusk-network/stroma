package corpus

import "testing"

func TestRecordNormalizedAppliesDefaults(t *testing.T) {
	t.Parallel()

	record, err := (Record{
		Ref:      " alpha ",
		BodyText: " body ",
	}).Normalized()
	if err != nil {
		t.Fatalf("Normalized() error = %v", err)
	}
	if record.Kind != DefaultKind {
		t.Fatalf("Kind = %q, want %q", record.Kind, DefaultKind)
	}
	if record.Title != "alpha" {
		t.Fatalf("Title = %q, want alpha", record.Title)
	}
	if record.SourceRef != "alpha" {
		t.Fatalf("SourceRef = %q, want alpha", record.SourceRef)
	}
	if record.BodyFormat != FormatMarkdown {
		t.Fatalf("BodyFormat = %q, want %q", record.BodyFormat, FormatMarkdown)
	}
	if record.ContentHash == "" {
		t.Fatal("ContentHash = empty, want deterministic hash")
	}
}

func TestRecordNormalizedRejectsUnknownBodyFormat(t *testing.T) {
	t.Parallel()

	_, err := (Record{
		Ref:        "alpha",
		SourceRef:  "file://alpha.md",
		BodyFormat: "html",
	}).Normalized()
	if err == nil {
		t.Fatal("Normalized() error = nil, want unsupported body_format")
	}
}

func TestFingerprintIsOrderIndependent(t *testing.T) {
	t.Parallel()

	left := []Record{
		{Ref: "a", SourceRef: "a", BodyText: "one"},
		{Ref: "b", SourceRef: "b", BodyText: "two"},
	}
	right := []Record{
		{Ref: "b", SourceRef: "b", BodyText: "two"},
		{Ref: "a", SourceRef: "a", BodyText: "one"},
	}
	if Fingerprint(left) != Fingerprint(right) {
		t.Fatalf("Fingerprint() changed with record order")
	}
}

// TestFingerprintFromPairsMatchesRecords locks byte-identity between the
// record-based and pair-based digests for already-normalized records — the
// contract finalizeUpdate relies on when it reads only (ref, content_hash) to
// recompute ContentFingerprint. If this regresses, every persisted
// ContentFingerprint disagrees with what callers recompute from their records.
func TestFingerprintFromPairsMatchesRecords(t *testing.T) {
	t.Parallel()

	raw := []Record{
		{Ref: "alpha", Kind: "doc", SourceRef: "file://alpha.md", BodyFormat: FormatMarkdown, BodyText: "alpha body", Metadata: map[string]string{"k": "v"}},
		{Ref: "bravo", Kind: DefaultKind, SourceRef: "bravo", BodyFormat: FormatPlaintext, BodyText: "bravo body"},
		{Ref: "charlie", Kind: "note", SourceRef: "charlie", BodyFormat: FormatMarkdown, BodyText: ""},
	}

	normalized := make([]Record, 0, len(raw))
	pairs := make([]RefHash, 0, len(raw))
	for _, r := range raw {
		n, err := r.Normalized()
		if err != nil {
			t.Fatalf("Normalized(%q) error = %v", r.Ref, err)
		}
		normalized = append(normalized, n)
		pairs = append(pairs, RefHash{Ref: n.Ref, ContentHash: n.ContentHash})
	}

	want := Fingerprint(normalized)
	got := FingerprintFromPairs(pairs)
	if got != want {
		t.Fatalf("FingerprintFromPairs = %q, want %q (must be byte-identical to Fingerprint for normalized records)", got, want)
	}
}

func TestFingerprintFromPairsSkipsEmpty(t *testing.T) {
	t.Parallel()

	pairs := []RefHash{
		{Ref: "alpha", ContentHash: "a"},
		{Ref: "   ", ContentHash: "b"},
		{Ref: "gamma", ContentHash: "  "},
		{Ref: "bravo", ContentHash: "c"},
	}
	kept := []RefHash{
		{Ref: "alpha", ContentHash: "a"},
		{Ref: "bravo", ContentHash: "c"},
	}
	if FingerprintFromPairs(pairs) != FingerprintFromPairs(kept) {
		t.Fatal("FingerprintFromPairs did not skip pairs with empty Ref or ContentHash")
	}
}
