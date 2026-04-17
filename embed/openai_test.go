package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestOpenAIUnconfigured(t *testing.T) {
	t.Parallel()

	e := NewOpenAI(OpenAIConfig{})
	if !e.Unconfigured() {
		t.Fatalf("expected Unconfigured() = true")
	}
	if _, err := e.EmbedQueries(context.Background(), []string{"hello"}); err == nil {
		t.Fatal("EmbedQueries() error = nil, want unconfigured failure")
	}
}

func TestOpenAIConfigNormalizesBaseURL(t *testing.T) {
	t.Parallel()

	e := NewOpenAI(OpenAIConfig{BaseURL: "http://example.test", Model: "mami"})
	if got := e.Config().BaseURL; got != "http://example.test/v1" {
		t.Fatalf("Config().BaseURL = %q, want %q", got, "http://example.test/v1")
	}

	e = NewOpenAI(OpenAIConfig{BaseURL: "http://example.test/custom/", Model: "mami"})
	if got := e.Config().BaseURL; got != "http://example.test/custom" {
		t.Fatalf("Config().BaseURL = %q, want %q", got, "http://example.test/custom")
	}

	e = NewOpenAI(OpenAIConfig{BaseURL: "http://example.test/v1?x=y#frag", Model: "mami"})
	if got := e.Config().BaseURL; got != "http://example.test/v1" {
		t.Fatalf("Config().BaseURL = %q, want %q", got, "http://example.test/v1")
	}
	if got := e.Config().Timeout; got != defaultOpenAITimeout {
		t.Fatalf("Config().Timeout = %s, want %s", got, defaultOpenAITimeout)
	}
}

func TestOpenAIFingerprintDependsOnModelAndStrategy(t *testing.T) {
	t.Parallel()

	plain := NewOpenAI(OpenAIConfig{BaseURL: "http://x", Model: "mami"})
	nomic := NewOpenAI(OpenAIConfig{BaseURL: "http://x", Model: "nomic-embed-text-v1.5"})
	if plain.Fingerprint() == nomic.Fingerprint() {
		t.Fatalf("distinct models should produce distinct fingerprints: %s", plain.Fingerprint())
	}
	if !strings.Contains(nomic.Fingerprint(), openAIStrategyNomicPrefix) {
		t.Fatalf("nomic fingerprint should encode strategy: %s", nomic.Fingerprint())
	}
	if !strings.Contains(plain.Fingerprint(), openAIStrategyPlain) {
		t.Fatalf("plain fingerprint should encode strategy: %s", plain.Fingerprint())
	}
}

func TestOpenAIFingerprintDistinguishesBaseURL(t *testing.T) {
	t.Parallel()

	a := NewOpenAI(OpenAIConfig{BaseURL: "http://host-a/v1", Model: "mami"})
	b := NewOpenAI(OpenAIConfig{BaseURL: "http://host-b/v1", Model: "mami"})
	if a.Fingerprint() == b.Fingerprint() {
		t.Fatalf("distinct base URLs must yield distinct fingerprints; got %q for both", a.Fingerprint())
	}
}

func TestOpenAIRoundTripHappyPath(t *testing.T) {
	t.Parallel()

	server, received := startOpenAIEmbedderStub(t, stubResponse{dimension: 4})
	defer server.Close()

	e := NewOpenAI(OpenAIConfig{
		BaseURL: server.URL + "/v1",
		Model:   "mami",
		Timeout: 2 * time.Second,
	})

	vectors, err := e.EmbedDocuments(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("EmbedDocuments() error = %v", err)
	}
	if len(vectors) != 2 {
		t.Fatalf("vector count = %d, want 2", len(vectors))
	}
	for i, vector := range vectors {
		if len(vector) != 4 {
			t.Fatalf("vector[%d] len = %d, want 4", i, len(vector))
		}
	}

	dimension, err := e.Dimension(context.Background())
	if err != nil {
		t.Fatalf("Dimension() error = %v", err)
	}
	if dimension != 4 {
		t.Fatalf("Dimension() = %d, want 4", dimension)
	}
	if got := received.Model(); got != "mami" {
		t.Fatalf("stub saw model %q, want %q", got, "mami")
	}
	if got := received.Inputs(); len(got) != 2 || got[0] != "hello" || got[1] != "world" {
		t.Fatalf("stub saw inputs %v, want [hello world]", got)
	}
}

func TestOpenAIAppliesNomicPrefixes(t *testing.T) {
	t.Parallel()

	server, received := startOpenAIEmbedderStub(t, stubResponse{dimension: 4})
	defer server.Close()

	e := NewOpenAI(OpenAIConfig{
		BaseURL: server.URL + "/v1",
		Model:   "nomic-embed-text-v1.5",
		Timeout: 2 * time.Second,
	})

	if _, err := e.EmbedQueries(context.Background(), []string{"how does this work"}); err != nil {
		t.Fatalf("EmbedQueries() error = %v", err)
	}
	if got := received.Inputs(); len(got) != 1 || !strings.HasPrefix(got[0], "search_query: ") {
		t.Fatalf("query input = %v, want search_query: prefix", got)
	}

	received.Reset()
	if _, err := e.EmbedDocuments(context.Background(), []string{"body text"}); err != nil {
		t.Fatalf("EmbedDocuments() error = %v", err)
	}
	if got := received.Inputs(); len(got) != 1 || !strings.HasPrefix(got[0], "search_document: ") {
		t.Fatalf("document input = %v, want search_document: prefix", got)
	}
}

func TestOpenAISurfacesHTTPErrors(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "upstream boom", http.StatusBadGateway)
	}))
	defer server.Close()

	e := NewOpenAI(OpenAIConfig{
		BaseURL: server.URL + "/v1",
		Model:   "mami",
		Timeout: time.Second,
	})
	if _, err := e.EmbedQueries(context.Background(), []string{"probe"}); err == nil {
		t.Fatal("EmbedQueries() error = nil, want HTTP failure")
	}
}

func TestOpenAIDetectsDimensionDrift(t *testing.T) {
	t.Parallel()

	dim := 4
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		out := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			out[i] = map[string]any{
				"embedding": fillVector(dim),
				"index":     i,
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
		dim = 8
	}))
	defer server.Close()

	e := NewOpenAI(OpenAIConfig{
		BaseURL: server.URL + "/v1",
		Model:   "mami",
		Timeout: time.Second,
	})
	if _, err := e.EmbedQueries(context.Background(), []string{"first"}); err != nil {
		t.Fatalf("first EmbedQueries() error = %v", err)
	}
	if _, err := e.EmbedQueries(context.Background(), []string{"second"}); err == nil {
		t.Fatal("second EmbedQueries() error = nil, want dimension drift failure")
	}
}

type capturedOpenAIRequest struct {
	model  string
	inputs []string
}

type receivedOpenAIRequests struct {
	requests []capturedOpenAIRequest
}

func (r *receivedOpenAIRequests) Model() string {
	if len(r.requests) == 0 {
		return ""
	}
	return r.requests[0].model
}

func (r *receivedOpenAIRequests) Inputs() []string {
	if len(r.requests) == 0 {
		return nil
	}
	return r.requests[len(r.requests)-1].inputs
}

func (r *receivedOpenAIRequests) Reset() {
	r.requests = nil
}

type stubResponse struct {
	dimension int
}

func startOpenAIEmbedderStub(t *testing.T, resp stubResponse) (*httptest.Server, *receivedOpenAIRequests) {
	t.Helper()
	received := &receivedOpenAIRequests{}

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)
			return
		}
		var body struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		received.requests = append(received.requests, capturedOpenAIRequest{model: body.Model, inputs: body.Input})

		out := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			out[i] = map[string]any{
				"embedding": fillVector(resp.dimension),
				"index":     i,
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	})
	return httptest.NewServer(handler), received
}

// TestOpenAIBoundedResponseBody locks the byte-side guard: a misconfigured
// or hostile upstream cannot stream an unbounded body through json.Decode
// and OOM the host. The fake server writes more than maxEmbedResponseBytes;
// EmbedDocuments must abort with a clear error rather than buffer it all.
func TestOpenAIBoundedResponseBody(t *testing.T) {
	t.Parallel()

	// Stream a payload larger than the cap. We start a valid JSON
	// document (so decoder doesn't bail early on a content-type
	// mismatch) and then pad inside a string field. The handler does
	// NOT set Content-Length so the client cannot short-circuit; it
	// has to actually attempt the read.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if _, err := fmt.Fprint(w, `{"data":[{"embedding":[0.1],"index":0,"pad":"`); err != nil {
			return
		}
		// Write pad in 64 KiB chunks until we exceed the cap.
		chunk := strings.Repeat("a", 64<<10)
		for written := 0; written <= maxEmbedResponseBytes; written += len(chunk) {
			if _, err := fmt.Fprint(w, chunk); err != nil {
				return
			}
		}
	}))
	defer server.Close()

	e := NewOpenAI(OpenAIConfig{
		BaseURL: server.URL + "/v1",
		Model:   "mami",
		Timeout: 10 * time.Second,
	})
	_, err := e.EmbedDocuments(context.Background(), []string{"probe"})
	if err == nil {
		t.Fatal("EmbedDocuments() with oversized body succeeded, want error")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("EmbedDocuments() err = %v, want error mentioning the size cap", err)
	}
}

// TestOpenAIConfigRedactsAPIToken locks the log-facing redaction paths:
// String, GoString, MarshalText, and slog.LogValue all hide APIToken.
// json.Marshal deliberately stays canonical (round-trip preserving), so
// it's not part of the redaction contract — callers that want a
// redacted JSON view should marshal cfg.String() or build a view type.
func TestOpenAIConfigRedactsAPIToken(t *testing.T) {
	t.Parallel()

	const secret = "sk-live-xxxxxxxxxxxxxxxxxxxxxx"
	cfg := OpenAIConfig{
		BaseURL:  "http://example.test/v1",
		Model:    "mami",
		Timeout:  2 * time.Second,
		APIToken: secret,
	}

	for _, rendering := range []struct {
		name string
		got  string
	}{
		{"String", cfg.String()},
		{"GoString", cfg.GoString()},
	} {
		if strings.Contains(rendering.got, secret) {
			t.Errorf("%s leaked raw token: %q", rendering.name, rendering.got)
		}
		if !strings.Contains(rendering.got, redactedTokenPlaceholder) {
			t.Errorf("%s missing redaction placeholder: %q", rendering.name, rendering.got)
		}
	}

	// slog.JSONHandler consults LogValuer before falling back to
	// json.Marshal, so logging the config through any slog handler
	// must not leak the token. Capture the JSON line and assert.
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))
	logger.Info("probe", "cfg", cfg)
	if strings.Contains(buf.String(), secret) {
		t.Errorf("slog.JSONHandler leaked raw token: %s", buf.String())
	}
	if !strings.Contains(buf.String(), redactedTokenPlaceholder) {
		t.Errorf("slog.JSONHandler missing redaction placeholder: %s", buf.String())
	}

	// Direct field access still returns the raw token — redaction is
	// defense-in-depth, not info-hiding, so the embedder can sign
	// requests. Locks the contract so future refactors don't
	// accidentally remove field access.
	if cfg.APIToken != secret {
		t.Fatalf("APIToken field = %q, want raw %q", cfg.APIToken, secret)
	}

	// json.Marshal deliberately stays canonical (see MarshalJSON
	// absence). This assertion documents the design: redaction on
	// json.Marshal would silently break round-trip persistence for any
	// downstream caller.
	marshaled, err := json.Marshal(cfg) //nolint:gosec // canonical round-trippable encoding is the intent
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if !strings.Contains(string(marshaled), secret) {
		t.Fatalf("json.Marshal should preserve APIToken for round-trip, got %s", marshaled)
	}

	// And a config without a token must not emit the placeholder —
	// "unset" stays distinguishable from "set but hidden".
	empty := OpenAIConfig{BaseURL: "http://example.test/v1", Model: "mami"}
	if strings.Contains(empty.String(), redactedTokenPlaceholder) {
		t.Errorf("empty-token config should not emit redaction marker: %q", empty.String())
	}
}

// TestOpenAIChunksOverMaxBatchSize locks the client-side batching
// contract: EmbedDocuments with more inputs than MaxBatchSize must
// split into ordered sub-requests of at most MaxBatchSize each and
// concatenate the results in input order. The fake server records
// every request's input slice so the test can assert both the
// per-batch cap and the overall ordering.
func TestOpenAIChunksOverMaxBatchSize(t *testing.T) {
	t.Parallel()

	server, received := startOpenAIEmbedderStub(t, stubResponse{dimension: 4})
	defer server.Close()

	e := NewOpenAI(OpenAIConfig{
		BaseURL:      server.URL + "/v1",
		Model:        "mami",
		Timeout:      5 * time.Second,
		MaxBatchSize: 3,
	})

	inputs := []string{"a", "b", "c", "d", "e", "f", "g"}
	vectors, err := e.EmbedDocuments(context.Background(), inputs)
	if err != nil {
		t.Fatalf("EmbedDocuments() error = %v", err)
	}
	if len(vectors) != len(inputs) {
		t.Fatalf("returned %d vectors, want %d", len(vectors), len(inputs))
	}

	// Three batches expected at batch size 3: [a b c], [d e f], [g].
	if len(received.requests) != 3 {
		t.Fatalf("server saw %d requests, want 3 (batches of 3/3/1)", len(received.requests))
	}
	for i, req := range received.requests {
		if len(req.inputs) > 3 {
			t.Fatalf("batch %d size = %d, want <= 3", i, len(req.inputs))
		}
	}
	// Flatten in arrival order and assert it equals input order.
	flat := make([]string, 0, len(inputs))
	for _, req := range received.requests {
		flat = append(flat, req.inputs...)
	}
	for i, want := range inputs {
		if flat[i] != want {
			t.Fatalf("flattened[%d] = %q, want %q (order preservation across batches)", i, flat[i], want)
		}
	}
}

// TestOpenAIMaxBatchSizeDefaultsToConservativeValue locks the default
// behaviour: a caller that doesn't set MaxBatchSize gets
// defaultOpenAIMaxBatchSize (512), so a single large EmbedDocuments
// call can't silently exceed the upstream's per-request limit.
func TestOpenAIMaxBatchSizeDefaultsToConservativeValue(t *testing.T) {
	t.Parallel()

	e := NewOpenAI(OpenAIConfig{BaseURL: "http://example.test", Model: "mami"})
	if got := e.Config().MaxBatchSize; got != defaultOpenAIMaxBatchSize {
		t.Fatalf("Config().MaxBatchSize = %d, want %d (conservative default)", got, defaultOpenAIMaxBatchSize)
	}
}

// TestOpenAIMaxBatchSizeExactBoundary covers the edge case where the
// input count equals MaxBatchSize — a single request, not two (one
// with the batch and one empty).
func TestOpenAIMaxBatchSizeExactBoundary(t *testing.T) {
	t.Parallel()

	server, received := startOpenAIEmbedderStub(t, stubResponse{dimension: 4})
	defer server.Close()

	e := NewOpenAI(OpenAIConfig{
		BaseURL:      server.URL + "/v1",
		Model:        "mami",
		Timeout:      2 * time.Second,
		MaxBatchSize: 3,
	})
	if _, err := e.EmbedDocuments(context.Background(), []string{"a", "b", "c"}); err != nil {
		t.Fatalf("EmbedDocuments() error = %v", err)
	}
	if len(received.requests) != 1 {
		t.Fatalf("server saw %d requests, want 1 (exact boundary should not spill to a second call)", len(received.requests))
	}
}

func fillVector(dim int) []float64 {
	out := make([]float64, dim)
	for i := range out {
		out[i] = float64(i) + 0.1
	}
	return out
}
