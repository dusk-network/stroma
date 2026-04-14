package embed

import (
	"context"
	"encoding/json"
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

func fillVector(dim int) []float64 {
	out := make([]float64, dim)
	for i := range out {
		out[i] = float64(i) + 0.1
	}
	return out
}
