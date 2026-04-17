package store

import (
	"math"
	"testing"
)

func TestVectorBlobRoundTrip(t *testing.T) {
	t.Parallel()

	original := []float64{0.125, -0.5, 0.75}
	blob, err := EncodeVectorBlob(original)
	if err != nil {
		t.Fatalf("EncodeVectorBlob() error = %v", err)
	}
	decoded, err := DecodeVectorBlob(blob)
	if err != nil {
		t.Fatalf("DecodeVectorBlob() error = %v", err)
	}
	if len(decoded) != len(original) {
		t.Fatalf("decoded len = %d, want %d", len(decoded), len(original))
	}
	for index := range original {
		if math.Abs(decoded[index]-original[index]) > 0.000001 {
			t.Fatalf("decoded[%d] = %f, want %f", index, decoded[index], original[index])
		}
	}
}

func TestDecodeVectorBlobRejectsInvalidLength(t *testing.T) {
	t.Parallel()

	if _, err := DecodeVectorBlob([]byte{1, 2, 3}); err == nil {
		t.Fatal("DecodeVectorBlob() error = nil, want invalid length")
	}
}

func TestVectorBlobInt8RoundTrip(t *testing.T) {
	t.Parallel()

	// Already unit-normalized vector (L2 norm = 1.0).
	original := []float64{0.6, -0.8, 0.0}
	blob, err := EncodeVectorBlobInt8(original)
	if err != nil {
		t.Fatalf("EncodeVectorBlobInt8() error = %v", err)
	}
	if len(blob) != len(original) {
		t.Fatalf("blob len = %d, want %d", len(blob), len(original))
	}
	decoded, err := DecodeVectorBlobInt8(blob)
	if err != nil {
		t.Fatalf("DecodeVectorBlobInt8() error = %v", err)
	}
	if len(decoded) != len(original) {
		t.Fatalf("decoded len = %d, want %d", len(decoded), len(original))
	}
	for i := range original {
		if math.Abs(decoded[i]-original[i]) > 0.02 {
			t.Fatalf("decoded[%d] = %f, want ~%f", i, decoded[i], original[i])
		}
	}
}

func TestVectorBlobInt8NormalizesOutOfRange(t *testing.T) {
	t.Parallel()

	// A non-normalized vector should be L2-normalized before quantization,
	// so encoding must succeed and round-trip values must be in [-1, 1].
	original := []float64{3.0, -4.0}
	blob, err := EncodeVectorBlobInt8(original)
	if err != nil {
		t.Fatalf("EncodeVectorBlobInt8() error = %v", err)
	}
	decoded, err := DecodeVectorBlobInt8(blob)
	if err != nil {
		t.Fatalf("DecodeVectorBlobInt8() error = %v", err)
	}
	for i, v := range decoded {
		if v < -1 || v > 1 {
			t.Fatalf("decoded[%d] = %f, want in [-1, 1]", i, v)
		}
	}
	// 3/5 = 0.6, -4/5 = -0.8 — check approximate values.
	if math.Abs(decoded[0]-0.6) > 0.02 {
		t.Fatalf("decoded[0] = %f, want ~0.6", decoded[0])
	}
	if math.Abs(decoded[1]-(-0.8)) > 0.02 {
		t.Fatalf("decoded[1] = %f, want ~-0.8", decoded[1])
	}
}

func TestVectorBlobBinaryRoundTrip(t *testing.T) {
	t.Parallel()

	original := []float64{0.5, -0.1, 0.0, -2.0, 3.0, -0.2, 0.1, -0.9}
	blob, err := EncodeVectorBlobBinary(original)
	if err != nil {
		t.Fatalf("EncodeVectorBlobBinary() error = %v", err)
	}
	if len(blob) != 1 {
		t.Fatalf("blob len = %d, want 1 (8 dims → 1 byte)", len(blob))
	}
	decoded, err := DecodeVectorBlobBinary(blob)
	if err != nil {
		t.Fatalf("DecodeVectorBlobBinary() error = %v", err)
	}
	for i, v := range original {
		wantSign := 1.0
		if v < 0 {
			wantSign = -1
		}
		if decoded[i] != wantSign {
			t.Fatalf("decoded[%d] = %f, want %f (sign of %f)", i, decoded[i], wantSign, v)
		}
	}
}

func TestVectorBlobBinaryRejectsNonMultipleOfEight(t *testing.T) {
	t.Parallel()

	if _, err := EncodeVectorBlobBinary([]float64{0.1, -0.1, 0.1}); err == nil {
		t.Fatal("EncodeVectorBlobBinary(len=3) error = nil, want multiple-of-8 error")
	}
	if _, err := EncodeVectorBlobBinary(nil); err == nil {
		t.Fatal("EncodeVectorBlobBinary(nil) error = nil, want non-empty error")
	}
}

func TestCosineScoreFromDistanceClampsToRange(t *testing.T) {
	t.Parallel()

	if got := CosineScoreFromDistance(-0.25); got != 1 {
		t.Fatalf("CosineScoreFromDistance(-0.25) = %f, want 1", got)
	}
	if got := CosineScoreFromDistance(2); got != 0 {
		t.Fatalf("CosineScoreFromDistance(2) = %f, want 0", got)
	}
}
