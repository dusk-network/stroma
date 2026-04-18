package index

import (
	"testing"
)

// TestUnmarshalMetadataBytesInputIndependence verifies that the returned
// map does not alias the input buffer. database/sql reuses scan
// destinations across rows, so the unmarshalMetadata contract is that its
// result must outlive any subsequent mutation of the byte slice the caller
// scanned into.
func TestUnmarshalMetadataBytesInputIndependence(t *testing.T) {
	raw := []byte(`{"phase":"ingest","owner":"alpha"}`)
	md, err := unmarshalMetadata("rec", raw)
	if err != nil {
		t.Fatalf("unmarshalMetadata error = %v", err)
	}
	// Corrupt the backing slice in place; the returned map must keep its
	// decoded values.
	for i := range raw {
		raw[i] = 0
	}
	if md["phase"] != "ingest" || md["owner"] != "alpha" {
		t.Fatalf("map values aliased input buffer: %#v", md)
	}
}

// BenchmarkUnmarshalMetadataBytes exercises the hot-path shape used by
// searchVectorCandidates / searchFTS / Sections: database/sql scans the
// metadata column into []byte and unmarshalMetadata consumes that slice
// directly. The pre-#57 path forced a string->[]byte copy per row
// inside json.Unmarshal; this benchmark gives reviewers a measurable
// baseline for that copy's absence.
func BenchmarkUnmarshalMetadataBytes(b *testing.B) {
	raw := []byte(`{"phase":"ingest","owner":"alpha","pipeline":"stroma"}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := unmarshalMetadata("rec", raw); err != nil {
			b.Fatal(err)
		}
	}
}
