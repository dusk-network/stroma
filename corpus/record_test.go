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
