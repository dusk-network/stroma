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

func TestCosineScoreFromDistanceClampsToRange(t *testing.T) {
	t.Parallel()

	if got := CosineScoreFromDistance(-0.25); got != 1 {
		t.Fatalf("CosineScoreFromDistance(-0.25) = %f, want 1", got)
	}
	if got := CosineScoreFromDistance(2); got != 0 {
		t.Fatalf("CosineScoreFromDistance(2) = %f, want 0", got)
	}
}
