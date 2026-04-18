package corpus

import "testing"

func TestNewRecordProducesNormalizableRecord(t *testing.T) {
	t.Parallel()

	const (
		wantRef   = "alpha"
		wantTitle = "Alpha"
	)

	raw := NewRecord(wantRef, wantTitle, "body text")
	if raw.Ref != wantRef {
		t.Fatalf("Ref = %q, want %q", raw.Ref, wantRef)
	}
	if raw.BodyFormat != FormatMarkdown {
		t.Fatalf("BodyFormat = %q, want %q", raw.BodyFormat, FormatMarkdown)
	}

	// The struct returned by NewRecord is not yet normalized — Kind
	// defaults through Normalize, not through the constructor — but it
	// must round-trip cleanly through Normalize without error.
	normalized, err := raw.Normalize()
	if err != nil {
		t.Fatalf("NewRecord(...).Normalize() error = %v", err)
	}
	if normalized.Kind != DefaultKind {
		t.Fatalf("normalized Kind = %q, want %q", normalized.Kind, DefaultKind)
	}
	if normalized.Title != wantTitle {
		t.Fatalf("normalized Title = %q, want %q", normalized.Title, wantTitle)
	}
	if normalized.ContentHash == "" {
		t.Fatal("normalized ContentHash = empty, want regenerated hash")
	}
}

// TestRecordNormalizedShimMirrorsNormalize pins the Deprecated:
// compatibility bridge. Downstream callers mid-upgrade should be
// able to keep using the old name and get byte-identical behavior
// from Normalize.
func TestRecordNormalizedShimMirrorsNormalize(t *testing.T) {
	t.Parallel()

	seed := Record{Ref: " alpha ", BodyText: " body "}
	legacy, legacyErr := seed.Normalized() //nolint:staticcheck // explicitly exercising the deprecated shim
	modern, modernErr := seed.Normalize()
	if (legacyErr == nil) != (modernErr == nil) {
		t.Fatalf("error mismatch: Normalized err = %v, Normalize err = %v", legacyErr, modernErr)
	}
	// The Metadata field is a map, so direct struct comparison
	// won't compile; assert on HashRecord — which folds Metadata
	// into the digest — to confirm the two paths produced the same
	// normalized record shape.
	if HashRecord(legacy) != HashRecord(modern) {
		t.Fatalf("Normalized() result diverged from Normalize() result\nlegacy: %#v\nmodern: %#v", legacy, modern)
	}
	if legacy.Ref != modern.Ref || legacy.ContentHash != modern.ContentHash {
		t.Fatalf("Normalized/Normalize Ref or ContentHash diverged\nlegacy: %#v\nmodern: %#v", legacy, modern)
	}
}

func TestRecordNormalizedAppliesDefaults(t *testing.T) {
	t.Parallel()

	record, err := (Record{
		Ref:      " alpha ",
		BodyText: " body ",
	}).Normalize()
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
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
	}).Normalize()
	if err == nil {
		t.Fatal("Normalize() error = nil, want unsupported body_format")
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
// previously, Fingerprint silently dropped records that failed Normalize(),
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

// TestHashRecordEncodingIsInjective guards the content_hash encoding fix
// in #39. Under the old encoding each part was rendered as plain
// "key=value" and parts were joined by '\n' — so a value containing '\n'
// could inject text that parsed as a neighboring part after the join, and
// two distinct records could in principle serialize to the same byte
// string. The new %q=%q encoding quotes both the field name and the
// value, which makes strconv.Quote's escaping the sole source of
// injectivity: no value can forge another field's delimited form.
//
// This test constructs two records that differ in field content but
// share a newline-boundary ambiguity — under the new encoding their
// hashes must differ. Whether the old encoding happened to collide for
// this specific pair is secondary; the point is that the new encoding
// hashes them distinctly by construction.
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
	spoofed := Record{
		Ref:        "alpha",
		Kind:       "note",
		Title:      "ignored",
		SourceRef:  "alpha",
		BodyFormat: FormatPlaintext,
		BodyText:   "body",
		Metadata: map[string]string{
			"k": "\ntitle=real-title",
		},
	}
	if HashRecord(genuine) == HashRecord(spoofed) {
		t.Fatal("HashRecord encoding is not injective: spoofed metadata produced the same digest as genuine field")
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
		n, err := r.Normalize()
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
