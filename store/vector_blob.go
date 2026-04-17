package store

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

const (
	// QuantizationFloat32 is the default float32 vector quantization.
	QuantizationFloat32 = "float32"

	// QuantizationInt8 uses signed 8-bit scalar quantization.
	QuantizationInt8 = "int8"

	// QuantizationBinary uses sign-based 1-bit quantization for the
	// prefilter and pairs it with a full-precision float32 companion
	// column for rescoring. The embedder dimension must be a multiple of
	// 8 because each byte of the packed blob carries 8 consecutive dims.
	QuantizationBinary = "binary"
)

// EncodeVectorBlob encodes float64 embeddings into the sqlite-vec blob format.
func EncodeVectorBlob(vector []float64) ([]byte, error) {
	buf := make([]byte, len(vector)*4)
	for index, value := range vector {
		binary.LittleEndian.PutUint32(buf[index*4:], math.Float32bits(float32(value)))
	}
	return buf, nil
}

// DecodeVectorBlob decodes a sqlite-vec float32 blob into float64 values.
func DecodeVectorBlob(blob []byte) ([]float64, error) {
	if len(blob)%4 != 0 {
		return nil, fmt.Errorf("invalid vector blob length %d", len(blob))
	}
	decoded := make([]float32, len(blob)/4)
	if err := binary.Read(bytes.NewReader(blob), binary.LittleEndian, decoded); err != nil {
		return nil, fmt.Errorf("decode vector blob: %w", err)
	}
	vector := make([]float64, len(decoded))
	for index, value := range decoded {
		vector[index] = float64(value)
	}
	return vector, nil
}

// EncodeVectorBlobInt8 L2-normalizes the input vector and then quantizes it
// to signed 8-bit integers for sqlite-vec int8 vector columns.
// Normalizing before quantization ensures any embedder output is mapped into
// [-1, 1] without silent clamping or hidden preconditions.
func EncodeVectorBlobInt8(vector []float64) ([]byte, error) {
	normalized := l2Normalize(vector)
	buf := make([]byte, len(normalized))
	for i, v := range normalized {
		buf[i] = byte(int8(math.Round(v * 127))) //nolint:gosec // intentional signed-to-unsigned reinterpret
	}
	return buf, nil
}

// l2Normalize returns a unit-length copy of vector. A zero vector is returned
// unchanged (all zeros).
func l2Normalize(vector []float64) []float64 {
	var norm float64
	for _, v := range vector {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	out := make([]float64, len(vector))
	if norm == 0 {
		return out
	}
	for i, v := range vector {
		out[i] = v / norm
	}
	return out
}

// DecodeVectorBlobInt8 decodes a sqlite-vec int8 vector blob back to float64.
func DecodeVectorBlobInt8(blob []byte) ([]float64, error) {
	vector := make([]float64, len(blob))
	for i, b := range blob {
		vector[i] = float64(int8(b)) / 127.0 //nolint:gosec // intentional unsigned-to-signed reinterpret
	}
	return vector, nil
}

// EncodeVectorBlobBinary packs the vector into a sqlite-vec bit blob using
// sign-based quantization: each output bit is 1 when the corresponding
// component is non-negative, 0 otherwise. The dimension must be a multiple
// of 8. The bit ordering matches sqlite-vec's vec_bit() parser: within each
// byte, dimension k occupies bit (k % 8) counted from the LSB.
func EncodeVectorBlobBinary(vector []float64) ([]byte, error) {
	if len(vector) == 0 {
		return nil, fmt.Errorf("binary quantization requires at least one dimension")
	}
	if len(vector)%8 != 0 {
		return nil, fmt.Errorf("binary quantization requires dimension divisible by 8, got %d", len(vector))
	}
	buf := make([]byte, len(vector)/8)
	for i, v := range vector {
		if v >= 0 {
			buf[i/8] |= 1 << (i % 8)
		}
	}
	return buf, nil
}

// DecodeVectorBlobBinary expands a bit-packed vector blob back to {-1, 1}
// float64 values. The returned vector has length blob-bytes * 8.
func DecodeVectorBlobBinary(blob []byte) ([]float64, error) {
	if len(blob) == 0 {
		return nil, fmt.Errorf("empty binary vector blob")
	}
	vector := make([]float64, len(blob)*8)
	for byteIdx, b := range blob {
		for bit := 0; bit < 8; bit++ {
			if b&(1<<bit) != 0 {
				vector[byteIdx*8+bit] = 1
			} else {
				vector[byteIdx*8+bit] = -1
			}
		}
	}
	return vector, nil
}

// CosineScoreFromDistance converts sqlite-vec cosine distance into a clamped score.
func CosineScoreFromDistance(distance float64) float64 {
	score := 1 - distance
	switch {
	case score < 0:
		return 0
	case score > 1:
		return 1
	default:
		return score
	}
}
