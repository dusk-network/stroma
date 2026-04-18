package embed

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"strings"
	"unicode"
)

const fixtureStrategy = "hash-bigram-v1"

// Fixture is a deterministic embedder for tests and offline use.
//
// Safe for concurrent use by multiple goroutines once constructed:
// Fixture is immutable post-NewFixture and every embed path operates
// on local state only.
type Fixture struct {
	model      string
	dimensions int
}

// NewFixture returns a deterministic hash-based embedder.
func NewFixture(model string, dimensions int) (*Fixture, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("fixture dimensions must be positive")
	}
	model = strings.TrimSpace(model)
	if model == "" {
		model = "fixture"
	}
	return &Fixture{
		model:      model,
		dimensions: dimensions,
	}, nil
}

// Fingerprint returns a stable identifier for the fixture configuration.
func (f *Fixture) Fingerprint() string {
	return fmt.Sprintf("fixture|%s|%s|%d", f.model, fixtureStrategy, f.dimensions)
}

// Dimension reports the embedding dimensionality.
func (f *Fixture) Dimension(ctx context.Context) (int, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	return f.dimensions, ctx.Err()
}

// EmbedDocuments returns deterministic embeddings for indexed document texts.
func (f *Fixture) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	return f.embedTexts(ctx, texts)
}

// EmbedQueries returns deterministic embeddings for query texts.
func (f *Fixture) EmbedQueries(ctx context.Context, texts []string) ([][]float64, error) {
	return f.embedTexts(ctx, texts)
}

func (f *Fixture) embedTexts(ctx context.Context, texts []string) ([][]float64, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	vectors := make([][]float64, 0, len(texts))
	for _, text := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		vectors = append(vectors, fixtureVector(text, f.dimensions))
	}
	return vectors, nil
}

func fixtureVector(text string, dimensions int) []float64 {
	vector := make([]float64, dimensions)
	if dimensions <= 0 {
		return vector
	}

	tokens := tokenize(text)
	if len(tokens) == 0 {
		return vector
	}

	for index, token := range tokens {
		vector[tokenBucket(token, dimensions)] += 1.0
		if index > 0 {
			bigram := tokens[index-1] + "_" + token
			vector[tokenBucket(bigram, dimensions)] += 1.5
		}
	}

	return normalize(vector)
}

func tokenize(text string) []string {
	var builder strings.Builder
	builder.Grow(len(text))
	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			builder.WriteRune(r)
			continue
		}
		builder.WriteByte(' ')
	}
	return strings.Fields(builder.String())
}

func tokenBucket(token string, dimensions int) int {
	hasher := fnv.New32a()
	_, _ = hasher.Write([]byte(token))
	return int(int64(hasher.Sum32()) % int64(dimensions))
}

func normalize(vector []float64) []float64 {
	var norm float64
	for _, value := range vector {
		norm += value * value
	}
	if norm == 0 {
		return vector
	}
	norm = math.Sqrt(norm)
	for index := range vector {
		vector[index] /= norm
	}
	return vector
}
