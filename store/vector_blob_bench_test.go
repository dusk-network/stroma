package store

import (
	"testing"
)

// BenchmarkDecodeVectorBlob exercises the float32 → float64 decode hot path
// hit on every stored chunk on the Sections(IncludeEmbeddings=true) and
// reuse-probe code paths. The allocs/op should land at 1 (the returned
// []float64); the pre-#60 implementation with binary.Read + bytes.Reader
// showed 2 extra allocations per call.
func BenchmarkDecodeVectorBlob(b *testing.B) {
	// 1536 is the OpenAI text-embedding-3-small / -large dimension.
	const dim = 1536
	vector := make([]float64, dim)
	for i := range vector {
		vector[i] = float64(i) / float64(dim)
	}
	blob, err := EncodeVectorBlob(vector)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := DecodeVectorBlob(blob)
		if err != nil {
			b.Fatal(err)
		}
		// Prevent the compiler from eliding the decode entirely.
		if out[0]+out[dim-1] < -1e9 {
			b.Fatal("unreachable")
		}
	}
}
