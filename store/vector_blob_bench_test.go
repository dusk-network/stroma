package store

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestDecodeVectorBlobEdgeBitPatterns pins the bit-exact semantics of the
// #60 D1 rewrite. The old binary.Read path preserved every float32 bit
// pattern (NaN payloads, infinities, subnormals, signed zero) on the
// round trip because binary.LittleEndian.Uint32 + Float32frombits is
// exactly what binary.Read emits for []float32. Any future
// "simplification" (e.g. Float32frombits via typed math conversions)
// that silently canonicalizes NaN payloads or flips signed-zero sign
// would regress bit identity — this test fails closed on that.
func TestDecodeVectorBlobEdgeBitPatterns(t *testing.T) {
	cases := []struct {
		name string
		bits uint32
	}{
		{"positive_zero", 0x00000000},
		{"negative_zero", 0x80000000},
		{"positive_infinity", 0x7f800000},
		{"negative_infinity", 0xff800000},
		{"quiet_nan_payload", 0x7fc00001},
		// Signaling NaNs are not preserved through the float32->float64->float32
		// roundtrip on most architectures (IEEE 754 allows the conversion to
		// quiet the NaN), and this was already the behavior of the pre-#60
		// binary.Read path. Do not reintroduce the signaling-NaN case here.
		{"smallest_subnormal", 0x00000001},
		{"largest_subnormal", 0x007fffff},
		{"smallest_normal", 0x00800000},
		{"largest_normal", 0x7f7fffff},
	}

	blob := make([]byte, 4*len(cases))
	for i, tc := range cases {
		binary.LittleEndian.PutUint32(blob[i*4:], tc.bits)
	}

	decoded, err := DecodeVectorBlob(blob)
	if err != nil {
		t.Fatalf("DecodeVectorBlob() error = %v", err)
	}
	if len(decoded) != len(cases) {
		t.Fatalf("len(decoded) = %d, want %d", len(decoded), len(cases))
	}

	for i, tc := range cases {
		got := math.Float32bits(float32(decoded[i]))
		if got != tc.bits {
			t.Errorf("%s: roundtripped bits = %#x, want %#x", tc.name, got, tc.bits)
		}
	}
}

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
