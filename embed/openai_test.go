package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/dusk-network/stroma/v2/provider"
)

// testEmbedPath is the path stub servers match to accept real
// embeddings POSTs — BaseURL normalization appends "/v1" and the
// client POSTs to BaseURL+"/embeddings", so every fake handler that
// cares about routing uses this exact path.
const testEmbedPath = "/v1/embeddings"

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
		if r.URL.Path != testEmbedPath {
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
	if len(flat) != len(inputs) {
		t.Fatalf("flattened captured inputs has %d entries, want %d (batching dropped or duplicated an input)", len(flat), len(inputs))
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

// TestOpenAIMultiBatchBoundsTotalTimeout verifies that a multi-batch
// EmbedDocuments call is still bounded by the scaled parent deadline
// (batches × Timeout) — i.e. a slow upstream does not let the loop
// run unboundedly even when every individual request would fit inside
// the per-request Timeout.
//
// The per-request http.Client.Timeout is disabled in this test so the
// scaled parent ctx deadline is the ONLY timeout in play. Otherwise
// http.Client.Timeout trips on the first request (because server
// work > Timeout) and we would not actually be exercising the parent
// deadline at all — which was Copilot's observation on the initial
// revision of this test.
//
// Shape: server work per request = 500ms, Timeout = 200ms, 4 batches.
// Parent budget = 4*200ms = 800ms. Batch 1 takes 500ms; after it the
// remaining budget is ~300ms, which is less than the next batch's
// 500ms of work, so the ctx trips during batch 2. We expect:
//   - EmbedDocuments returns a non-nil error,
//   - at least one request was fully delivered (proving http.Client
//     did not trip first),
//   - not all batches ran (proving the parent deadline did bound the
//     loop),
//   - elapsed stays within 2x the scaled budget as a sanity check
//     against an unbounded run.
func TestOpenAIMultiBatchBoundsTotalTimeout(t *testing.T) {
	t.Parallel()

	const (
		timeout         = 200 * time.Millisecond
		perRequestDelay = 500 * time.Millisecond
		batchSize       = 2
		inputCount      = 8
	)

	var (
		mu   sync.Mutex
		seen int
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		select {
		case <-time.After(perRequestDelay):
		case <-r.Context().Done():
			return
		}
		mu.Lock()
		seen++
		mu.Unlock()
		out := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			out[i] = map[string]any{"embedding": fillVector(4), "index": i}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
	defer server.Close()

	e := NewOpenAI(OpenAIConfig{
		BaseURL:      server.URL + "/v1",
		Model:        "mami",
		Timeout:      timeout,
		MaxBatchSize: batchSize,
	})
	// Disable the per-request http.Client.Timeout so the scaled
	// parent ctx deadline is the ONLY bound in play. Without this,
	// http.Client.Timeout (=200ms) trips before the server finishes
	// any single 500ms request and the test would not actually
	// validate the parent deadline.
	e.client.Timeout = 0

	start := time.Now()
	inputs := make([]string, inputCount)
	for i := range inputs {
		inputs[i] = "t" + strconv.Itoa(i)
	}
	_, err := e.EmbedDocuments(context.Background(), inputs)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("EmbedDocuments() succeeded on slow upstream, want deadline-exceeded error")
	}

	batches := (inputCount + batchSize - 1) / batchSize
	maxBudget := time.Duration(batches) * timeout

	mu.Lock()
	observed := seen
	mu.Unlock()

	if observed == 0 {
		t.Fatal("server saw 0 completed requests; http.Client.Timeout tripped before any request delivered — parent-deadline bound not actually exercised")
	}
	if observed >= batches {
		t.Fatalf("server saw all %d requests despite a bounded parent deadline; scaled deadline did not actually cap the loop", observed)
	}
	// Upper bound: the total budget is batches × Timeout plus
	// scheduling jitter. If we blew past 2× the expected maximum,
	// the parent ctx deadline was not bounding the loop.
	if elapsed > 2*maxBudget {
		t.Fatalf("EmbedDocuments() took %s, want bounded by batches*Timeout (~%s + slack)", elapsed, maxBudget)
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

// TestScaledMultiBatchBudgetSaturatesOnOverflow guards the overflow
// case on the #63 deadline math. time.Duration is int64 nanoseconds
// (max ~292 years); a caller-configured large Timeout combined with
// a large batch count can make `Timeout * batches` wrap. The raw
// multiplication would yield a negative or wildly smaller duration
// that cancels the call immediately; the helper must clamp to
// math.MaxInt64 so the derived deadline behaves like "effectively no
// upper bound" instead.
func TestScaledMultiBatchBudgetSaturatesOnOverflow(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		perRequest time.Duration
		batches    int
		wantMax    bool
	}{
		{"zero_batches_returns_perRequest", time.Hour, 0, false},
		{"zero_timeout_returns_perRequest", 0, 10, false},
		{"small_values_multiply_cleanly", time.Second, 10, false},
		// int64 nanoseconds: MaxInt64 ≈ 9.22e18. One hour ≈ 3.6e12 ns.
		// 3.6e12 * 3e6 ≈ 1.08e19 > MaxInt64 → overflow without the guard.
		{"hour_times_three_million_overflows", time.Hour, 3_000_000, true},
		// Extreme case: a year-long Timeout with modest batches also
		// overflows quickly. One year ≈ 3.15e16 ns; 300 batches ≈ 9.45e18
		// which is already on the edge of MaxInt64.
		{"year_times_thousand_overflows", 365 * 24 * time.Hour, 1000, true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := scaledMultiBatchBudget(tc.perRequest, tc.batches)
			if got < 0 {
				t.Fatalf("scaledMultiBatchBudget(%s, %d) = %s, want a non-negative duration (overflow wrapped into negative)",
					tc.perRequest, tc.batches, got)
			}
			if tc.wantMax && got != time.Duration(math.MaxInt64) {
				t.Fatalf("scaledMultiBatchBudget(%s, %d) = %s, want saturated math.MaxInt64",
					tc.perRequest, tc.batches, got)
			}
			if !tc.wantMax {
				// Identity paths (zero batches or zero timeout) return
				// perRequest unchanged; the multiplying path returns
				// perRequest * batches exactly.
				want := tc.perRequest
				if tc.batches > 0 && tc.perRequest > 0 {
					want = tc.perRequest * time.Duration(tc.batches)
				}
				if got != want {
					t.Fatalf("scaledMultiBatchBudget(%s, %d) = %s, want %s",
						tc.perRequest, tc.batches, got, want)
				}
			}
		})
	}
}

// TestOpenAIMultiBatchDeadlineScalesWithBatchCount locks the #63 fix:
// the overall deadline imposed on a multi-batch EmbedDocuments call
// must scale with the batch count, so later batches are not starved by
// the cumulative time spent on earlier ones. Pre-#63, EmbedDocuments
// set one flat `time.Now().Add(Timeout)` deadline around the whole
// batch loop. Each individual sub-request still fit inside the
// per-request Timeout, but their *sum* exceeded it — so the shared
// context deadline tripped mid-loop and later sub-requests failed
// with `context deadline exceeded` even though every request was
// individually well-behaved.
//
// The test configuration is tuned to discriminate the two shapes:
// every batch takes ~65ms (well under the 150ms per-request Timeout
// enforced by http.Client.Timeout), and the three batches sum to
// ~195ms — strictly greater than Timeout, strictly less than the
// post-fix budget of 3*Timeout = 450ms.
func TestOpenAIMultiBatchDeadlineScalesWithBatchCount(t *testing.T) {
	t.Parallel()

	const (
		timeout      = 150 * time.Millisecond
		perBatchWork = 65 * time.Millisecond
		dimension    = 4
	)

	var (
		mu        sync.Mutex
		seen      int
		allInputs [][]string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != testEmbedPath {
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

		mu.Lock()
		seen++
		allInputs = append(allInputs, append([]string(nil), body.Input...))
		mu.Unlock()

		// Every batch takes the same modest amount of server-side
		// work — individually well within http.Client.Timeout, but
		// three of them sum to more than one Timeout window. Pre-#63
		// the shared ctx deadline trips on the second or third
		// request; post-#63 the scaled deadline covers all three.
		select {
		case <-time.After(perBatchWork):
		case <-r.Context().Done():
			return
		}

		out := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			out[i] = map[string]any{
				"embedding": fillVector(dimension),
				"index":     i,
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
	defer server.Close()

	e := NewOpenAI(OpenAIConfig{
		BaseURL:      server.URL + "/v1",
		Model:        "mami",
		Timeout:      timeout,
		MaxBatchSize: 2,
	})

	inputs := []string{"a", "b", "c", "d", "e", "f"}
	vectors, err := e.EmbedDocuments(context.Background(), inputs)
	if err != nil {
		t.Fatalf("EmbedDocuments() error = %v (multi-batch deadline should scale with batch count)", err)
	}
	if len(vectors) != len(inputs) {
		t.Fatalf("len(vectors) = %d, want %d", len(vectors), len(inputs))
	}

	mu.Lock()
	defer mu.Unlock()
	if seen != 3 {
		t.Fatalf("server saw %d requests, want 3 (batches of 2/2/2)", seen)
	}
	flat := make([]string, 0, len(inputs))
	for _, batch := range allInputs {
		flat = append(flat, batch...)
	}
	for i, want := range inputs {
		if flat[i] != want {
			t.Fatalf("flattened[%d] = %q, want %q (ordering broke across scaled deadline)", i, flat[i], want)
		}
	}
}

// TestOpenAIEmbedDecodeErrorClassifiesAsSchemaMismatch confirms that a
// malformed 2xx body — whether unparsable JSON or a count mismatch
// from parseEmbedResponse — surfaces as a classified provider error
// callers can branch on. Before the provider port, decode failures
// returned plain fmt.Errorf values and silently bypassed any caller
// FailureClass check.
func TestOpenAIEmbedDecodeErrorClassifiesAsSchemaMismatch(t *testing.T) {
	t.Parallel()

	cases := map[string]string{
		"not json":        `<html>nope</html>`,
		"count mismatch":  `{"data":[{"embedding":[0.1],"index":0},{"embedding":[0.2],"index":1}]}`,
		"empty embedding": `{"data":[{"embedding":[],"index":0}]}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = fmt.Fprint(w, body)
			}))
			t.Cleanup(server.Close)

			e := NewOpenAI(OpenAIConfig{BaseURL: server.URL, Model: "mami"})
			_, err := e.EmbedQueries(context.Background(), []string{"probe"})
			if err == nil {
				t.Fatalf("err = nil, want classified schema_mismatch")
			}
			var perr *provider.Error
			if !errors.As(err, &perr) {
				t.Fatalf("err = %v, want *provider.Error via errors.As", err)
			}
			if perr.FailureClass() != provider.FailureClassSchemaMismatch {
				t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), provider.FailureClassSchemaMismatch)
			}
		})
	}
}

// TestOpenAIEmbedRetriesOn5xxThenSucceeds locks the #73 wiring: with
// MaxRetries > 0, a retryable upstream 5xx must trigger another
// attempt through the shared provider core rather than surfacing
// immediately. Mirrors the chat-side coverage so the two surfaces
// share the same contract.
func TestOpenAIEmbedRetriesOn5xxThenSucceeds(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != testEmbedPath {
			http.NotFound(w, r)
			return
		}
		n := attempts.Add(1)
		if n < 2 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		out := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			out[i] = map[string]any{"embedding": fillVector(4), "index": i}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
	t.Cleanup(server.Close)

	e := NewOpenAI(OpenAIConfig{
		BaseURL:    server.URL + "/v1",
		Model:      "mami",
		Timeout:    2 * time.Second,
		MaxRetries: 2,
	})
	vectors, err := e.EmbedQueries(context.Background(), []string{"probe"})
	if err != nil {
		t.Fatalf("EmbedQueries() error = %v", err)
	}
	if len(vectors) != 1 || len(vectors[0]) != 4 {
		t.Fatalf("vectors = %v, want one 4-dim vector", vectors)
	}
	if got := attempts.Load(); got != 2 {
		t.Errorf("attempts = %d, want 2 (one 502 + one success)", got)
	}
}

// TestOpenAIEmbedDoesNotRetryOnNon5xx4xx confirms classification-aware
// retrying: an auth failure (401) is non-retryable, so the client must
// stop after the first attempt even when MaxRetries > 0. This guards
// against an accidental regression that would retry every non-2xx.
func TestOpenAIEmbedDoesNotRetryOnNon5xx4xx(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = fmt.Fprint(w, `{"error":{"message":"invalid key"}}`)
	}))
	t.Cleanup(server.Close)

	e := NewOpenAI(OpenAIConfig{
		BaseURL:    server.URL + "/v1",
		Model:      "mami",
		Timeout:    2 * time.Second,
		APIToken:   "bad",
		MaxRetries: 3,
	})
	_, err := e.EmbedQueries(context.Background(), []string{"probe"})
	if err == nil {
		t.Fatal("EmbedQueries() error = nil, want auth failure")
	}
	var perr *provider.Error
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *provider.Error", err)
	}
	if perr.FailureClass() != provider.FailureClassAuth {
		t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), provider.FailureClassAuth)
	}
	if got := attempts.Load(); got != 1 {
		t.Errorf("attempts = %d, want 1 (auth failures are non-retryable even with MaxRetries=3)", got)
	}
}

// TestOpenAIEmbedHonoursRetryAfterHeader verifies that a 429 response
// with a Retry-After header makes the retry loop wait at least the
// server-supplied interval before re-issuing the request. Defers the
// bulk of Retry-After coverage to provider/retry_test.go; this test
// confirms the embed surface actually routes through the shared loop.
func TestOpenAIEmbedHonoursRetryAfterHeader(t *testing.T) {
	t.Parallel()

	var (
		attempts        atomic.Int32
		firstAttemptAt  time.Time
		secondAttemptAt time.Time
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != testEmbedPath {
			http.NotFound(w, r)
			return
		}
		n := attempts.Add(1)
		if n == 1 {
			firstAttemptAt = time.Now()
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		secondAttemptAt = time.Now()
		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		out := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			out[i] = map[string]any{"embedding": fillVector(4), "index": i}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
	t.Cleanup(server.Close)

	e := NewOpenAI(OpenAIConfig{
		BaseURL:    server.URL + "/v1",
		Model:      "mami",
		Timeout:    5 * time.Second,
		MaxRetries: 2,
	})
	if _, err := e.EmbedQueries(context.Background(), []string{"probe"}); err != nil {
		t.Fatalf("EmbedQueries() error = %v", err)
	}
	gap := secondAttemptAt.Sub(firstAttemptAt)
	if gap < 900*time.Millisecond {
		t.Errorf("Retry-After honoured gap = %s, want >=900ms (1s header minus jitter slack)", gap)
	}
}

// TestOpenAIEmbedMultiBatchBudgetScalesWithRetries locks the retry-
// aware multi-batch deadline: when batch 0 spends more than one
// Timeout window on a retry chain, batch 1 must still get its own
// full Timeout-sized per-attempt window. Pre-fix the parent ctx
// deadline was batches*Timeout = 400ms total; batch 0's initial 502
// + 200ms backoff + 100ms retry work consumed ~305ms of that, and
// batch 1's 100ms server work then tripped the parent ctx at 400ms.
// Post-fix the deadline scales by (MaxRetries+1) so batch 1 has a
// full Timeout worth of budget regardless of what batch 0 did.
func TestOpenAIEmbedMultiBatchBudgetScalesWithRetries(t *testing.T) {
	t.Parallel()

	const (
		timeout      = 200 * time.Millisecond
		perBatchWork = 100 * time.Millisecond
	)

	var (
		mu      sync.Mutex
		attempt int
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != testEmbedPath {
			http.NotFound(w, r)
			return
		}
		mu.Lock()
		attempt++
		n := attempt
		mu.Unlock()

		// Attempt 1 is batch 0's first shot → 502 immediately.
		// Attempts 2 (batch 0 retry) and 3 (batch 1) do real work
		// bounded by perBatchWork, which stays under Timeout so the
		// per-attempt http.Client.Timeout never trips.
		if n == 1 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		select {
		case <-time.After(perBatchWork):
		case <-r.Context().Done():
			return
		}

		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		out := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			out[i] = map[string]any{"embedding": fillVector(4), "index": i}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
	defer server.Close()

	e := NewOpenAI(OpenAIConfig{
		BaseURL:      server.URL + "/v1",
		Model:        "mami",
		Timeout:      timeout,
		MaxRetries:   1,
		MaxBatchSize: 1,
	})

	start := time.Now()
	vectors, err := e.EmbedDocuments(context.Background(), []string{"a", "b"})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("EmbedDocuments() error = %v (retry-aware multi-batch budget regressed; batch 1 likely starved by batch 0's retry chain)", err)
	}
	if len(vectors) != 2 {
		t.Fatalf("len(vectors) = %d, want 2", len(vectors))
	}

	mu.Lock()
	got := attempt
	mu.Unlock()
	if got != 3 {
		t.Errorf("server saw %d attempts, want 3 (batch 0 initial 502 + batch 0 retry + batch 1)", got)
	}

	// Lower bound: backoff (200ms) + at least one 100ms batch of
	// real work. Proves retries actually fired and we are not just
	// racing through three instant 2xx responses.
	if elapsed < 250*time.Millisecond {
		t.Errorf("elapsed = %s, want >=250ms (backoff+work must have actually happened)", elapsed)
	}
	// Upper bound: generous slack for CI jitter. Post-fix total is
	// ~405ms (5ms + 200ms backoff + 100ms + 100ms); pre-fix failed
	// outright, so passing at all is the strongest signal.
	if elapsed > 2*time.Second {
		t.Errorf("elapsed = %s, want well under 2s (indicates unexpected wall-clock inflation)", elapsed)
	}
}

// TestOpenAINormalizesNegativeMaxRetries locks the construction-time
// clamp: a pathological negative MaxRetries (e.g. from a bad env
// parse) must not short-circuit provider.Do's retry loop. Without
// normalization, provider.Do's `for attempt := 0; attempt <=
// policy.MaxRetries; attempt++` executes zero times when MaxRetries
// is negative, so the embedder would hard-fail every call with a
// generic error before any HTTP round trip. After normalization the
// raw request still fires and either succeeds or returns a
// classified transport/protocol error — the normal path.
func TestOpenAINormalizesNegativeMaxRetries(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != testEmbedPath {
			http.NotFound(w, r)
			return
		}
		attempts.Add(1)
		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		out := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			out[i] = map[string]any{"embedding": fillVector(4), "index": i}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
	t.Cleanup(server.Close)

	e := NewOpenAI(OpenAIConfig{
		BaseURL:    server.URL + "/v1",
		Model:      "mami",
		Timeout:    2 * time.Second,
		MaxRetries: -1, // pathological; must be clamped to 0
	})
	if got := e.Config().MaxRetries; got != 0 {
		t.Errorf("Config().MaxRetries = %d after NewOpenAI, want 0 (negative values must be clamped)", got)
	}

	vectors, err := e.EmbedQueries(context.Background(), []string{"probe"})
	if err != nil {
		t.Fatalf("EmbedQueries() error = %v (negative MaxRetries must not brick the embedder)", err)
	}
	if len(vectors) != 1 || len(vectors[0]) != 4 {
		t.Fatalf("vectors = %v, want one 4-dim vector", vectors)
	}
	if got := attempts.Load(); got != 1 {
		t.Errorf("attempts = %d, want 1 (normalized to 0 retries → single attempt)", got)
	}
}

// TestOpenAIEmbedMultiBatchBudgetIncludesRetryAfterSlack locks the
// retry-wait slack in the parent deadline: a 429 with Retry-After on
// batch 0 sleeps well beyond a single Timeout, but batch 1 must still
// complete because the deadline budgets MaxRetries * embedMaxRetryAfter
// of inter-retry wait time on top of per-attempt Timeout. Pre-fix the
// budget was batches*(MaxRetries+1)*Timeout = 2*2*200ms = 800ms, and a
// 1s Retry-After alone would consume that before batch 1 even started.
// Post-fix the budget adds MaxRetries * embedMaxRetryAfter = 30s of
// slack per batch, so batch 1 still receives its full Timeout window.
func TestOpenAIEmbedMultiBatchBudgetIncludesRetryAfterSlack(t *testing.T) {
	t.Parallel()

	const (
		timeout         = 200 * time.Millisecond
		retryAfterValue = 1 * time.Second
	)

	var (
		mu      sync.Mutex
		attempt int
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != testEmbedPath {
			http.NotFound(w, r)
			return
		}
		mu.Lock()
		attempt++
		n := attempt
		mu.Unlock()

		// Attempt 1 is batch 0's first shot → 429 with a
		// Retry-After header that substantially exceeds Timeout,
		// forcing the retry loop to sleep beyond a single Timeout
		// window. Attempts 2 (batch 0 retry) and 3 (batch 1) both
		// succeed fast — the test is about the safety-net budget,
		// not about timing individual responses.
		if n == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		out := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			out[i] = map[string]any{"embedding": fillVector(4), "index": i}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
	defer server.Close()

	e := NewOpenAI(OpenAIConfig{
		BaseURL:      server.URL + "/v1",
		Model:        "mami",
		Timeout:      timeout,
		MaxRetries:   1,
		MaxBatchSize: 1,
	})

	start := time.Now()
	vectors, err := e.EmbedDocuments(context.Background(), []string{"a", "b"})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("EmbedDocuments() error = %v (retry-wait slack missing from safety-net deadline — batch 1 starved by batch 0's Retry-After sleep)", err)
	}
	if len(vectors) != 2 {
		t.Fatalf("len(vectors) = %d, want 2", len(vectors))
	}

	mu.Lock()
	got := attempt
	mu.Unlock()
	if got != 3 {
		t.Errorf("server saw %d attempts, want 3 (batch 0 initial 429 + batch 0 retry + batch 1)", got)
	}

	// Lower bound: honouring the Retry-After header means we must
	// have waited at least ~1s before batch 0's retry. Assert with
	// slack for CI clock coarseness.
	if elapsed < 900*time.Millisecond {
		t.Errorf("elapsed = %s, want >=900ms (Retry-After=1s must have been honoured before batch 0 retry)", elapsed)
	}
	// Upper bound: real wall-clock is ~1s Retry-After plus two fast
	// successful requests. 5s is generous slack against CI jitter
	// without masking runaway retry behavior.
	if elapsed > 5*time.Second {
		t.Errorf("elapsed = %s, want well under 5s (indicates unexpected wall-clock inflation)", elapsed)
	}
}
