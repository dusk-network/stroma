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
	leftFP, err := Fingerprint(left)
	if err != nil {
		t.Fatalf("Fingerprint(left) error = %v", err)
	}
	rightFP, err := Fingerprint(right)
	if err != nil {
		t.Fatalf("Fingerprint(right) error = %v", err)
	}
	if leftFP != rightFP {
		t.Fatalf("Fingerprint() changed with record order")
	}
}

// TestFingerprintSurfacesMalformedRecords locks the loud-failure contract:
// previously, Fingerprint silently dropped records that failed Normalized(),
// so a corpus with an invalid record produced the same digest as the same
// corpus without that record. Combined with ReuseFromPath, that masked
// silent reuse of snapshots missing data the caller thought they had.
func TestFingerprintSurfacesMalformedRecords(t *testing.T) {
	t.Parallel()

	_, err := Fingerprint([]Record{
		{Ref: "alpha", SourceRef: "alpha", BodyText: "a"},
		{Ref: "bravo", SourceRef: "bravo", BodyFormat: "html"}, // unsupported body_format
	})
	if err == nil {
		t.Fatal("Fingerprint() succeeded on malformed record, want error")
	}
}

// TestHashRecordEncodingIsInjective guards the content_hash encoding fix in
// #39. The prior encoding ("key=value" lines joined by \n) let a metadata
// value containing \ntitle=X collide with a record whose Title was actually
// X. Quoted-pair encoding (%q=%q) makes every part self-delimiting, so no
// attacker-crafted metadata value can mimic another field's serialization.
func TestHashRecordEncodingIsInjective(t *testing.T) {
	t.Parallel()

	genuine := Record{
		Ref:        "alpha",
		Kind:       "note",
		Title:      "real-title",
		SourceRef:  "alpha",
		BodyFormat: FormatPlaintext,
		BodyText:   "body",
	}
	// In the old encoding, a metadata value that began with a newline and
	// continued with `title="real-title"` would be serialized as
	//   metadata.k=\ntitle=real-title
	// which joins into a part that parses identically to the genuine record
	// — different field, same serialization. The new %q=%q encoding quotes
	// the field name and value, so the attacker part cannot forge the
	// target part.
	spoofed := Record{
		Ref:        "alpha",
		Kind:       "note",
		Title:      "ignored",
		SourceRef:  "alpha",
		BodyFormat: FormatPlaintext,
		BodyText:   "body",
		Metadata: map[string]string{
			"k": "\ntitle=\"real-title\"",
		},
	}
	if HashRecord(genuine) == HashRecord(spoofed) {
		t.Fatal("HashRecord encoding is not injective: spoofed metadata collides with genuine field")
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

	want, err := Fingerprint(normalized)
	if err != nil {
		t.Fatalf("Fingerprint() error = %v", err)
	}
	got, err := FingerprintFromPairs(pairs)
	if err != nil {
		t.Fatalf("FingerprintFromPairs() error = %v", err)
	}
	if got != want {
		t.Fatalf("FingerprintFromPairs = %q, want %q (must be byte-identical to Fingerprint for normalized records)", got, want)
	}
}

// TestFingerprintFromPairsRejectsEmpty locks the loud-failure contract:
// every persisted row that flows through loadCurrentRefHashes has been
// guarded for non-empty content_hash, so a pair reaching this helper with
// an empty field is a bug worth surfacing, not silently skipping.
func TestFingerprintFromPairsRejectsEmpty(t *testing.T) {
	t.Parallel()

	for _, pair := range []RefHash{
		{Ref: "", ContentHash: "a"},
		{Ref: "   ", ContentHash: "b"},
		{Ref: "alpha", ContentHash: ""},
		{Ref: "alpha", ContentHash: "  "},
	} {
		if _, err := FingerprintFromPairs([]RefHash{pair}); err == nil {
			t.Fatalf("FingerprintFromPairs(%+v) succeeded, want error", pair)
		}
	}
}
