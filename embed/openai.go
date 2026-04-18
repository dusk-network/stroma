package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

const (
	openAIStrategyPlain       = "plain_v1"
	openAIStrategyNomicPrefix = "nomic_search_prefix_v1"
	defaultOpenAITimeout      = 15 * time.Second

	// defaultOpenAIMaxBatchSize is the per-request cap applied when a
	// caller leaves OpenAIConfig.MaxBatchSize unset. Real OpenAI
	// embeddings accept up to 2048 inputs per request, but self-hosted
	// OpenAI-compatible gateways vary; 512 is a conservative default
	// that fits every deployment we've seen in the wild and keeps a
	// single embed batch well under the 32 MiB response cap.
	defaultOpenAIMaxBatchSize = 512

	// maxEmbedResponseBytes caps how much of an embedder response body
	// stroma will buffer before aborting the decode. Real OpenAI
	// responses for a single embed batch are well under 1 MiB; 32 MiB
	// is generous headroom for self-hosted gateways with larger batches,
	// while still preventing a misconfigured or hostile upstream from
	// streaming GiBs of payload and OOMing the host. The client Timeout
	// bounds wall time but not bytes, so this is the byte-side guard.
	maxEmbedResponseBytes = 32 << 20 // 32 MiB

	// redactedTokenPlaceholder replaces APIToken in every text/JSON
	// representation of OpenAIConfig so accidental structured logging
	// or pretty-printing cannot leak the credential.
	redactedTokenPlaceholder = "[REDACTED]"
)

// OpenAIConfig configures an OpenAI-compatible embedder.
//
// APIToken is redacted from every log/display representation — String,
// GoString, and slog.LogValuer all render the token as "[REDACTED]".
// Direct field access still yields the raw value so the embedder
// itself can sign requests; redaction is defense-in-depth against
// accidental fmt.Printf / slog.Info / log line disclosure.
//
// json.Marshal and encoding.TextMarshaler deliberately stay canonical:
// OpenAIConfig is a public configuration type, so overriding either
// would silently break any caller persisting or round-tripping the
// config (credential would vanish on encode, and TextMarshaler also
// redirects json.Marshal output through its redacted form). Callers
// that need a redacted view should marshal cfg.String() or build a
// dedicated view type.
type OpenAIConfig struct {
	BaseURL string
	Model   string

	// Timeout bounds a single embeddings sub-request. For inputs that
	// fit in a single batch (len(texts) <= MaxBatchSize) this is also
	// the total wall-clock budget. For multi-batch calls the total
	// budget is Timeout * ceil(len(texts)/MaxBatchSize) — each
	// sub-request gets its own Timeout-sized window, and a slow early
	// batch cannot starve later batches. A zero or negative value
	// selects defaultOpenAITimeout (15s) at NewOpenAI time. Callers
	// that want a tighter global cap should pass a ctx with their own
	// deadline; the embedder honors whichever deadline trips first.
	Timeout  time.Duration
	APIToken string

	// MaxBatchSize caps how many inputs are sent in a single embeddings
	// request. EmbedDocuments and EmbedQueries chunk their input into
	// sub-requests of at most this size and concatenate the results in
	// order. Zero or negative values select defaultOpenAIMaxBatchSize
	// (512), which is conservative for self-hosted gateways; real OpenAI
	// accepts up to 2048 per request, and operators who know their
	// upstream can raise this explicitly.
	MaxBatchSize int
}

// Enabled reports whether the config is usable for requests.
func (c OpenAIConfig) Enabled() bool {
	return strings.TrimSpace(c.BaseURL) != "" && strings.TrimSpace(c.Model) != ""
}

// String returns a redacted, human-readable rendering of the config.
// fmt verbs %v and %s route through this method.
func (c OpenAIConfig) String() string {
	return fmt.Sprintf("OpenAIConfig{BaseURL:%q Model:%q Timeout:%s MaxBatchSize:%d APIToken:%q}",
		c.BaseURL, c.Model, c.Timeout, c.MaxBatchSize, redactedToken(c.APIToken))
}

// GoString returns a redacted Go-syntax rendering of the config for %#v.
// Without this, %#v falls back to reflection and surfaces the raw
// APIToken field value. Timeout is formatted as time.Duration(ns) so
// the result stays a valid Go composite literal a reader could paste
// back into source (Duration's default %s form "2s" is human-readable
// but not Go-parseable).
func (c OpenAIConfig) GoString() string {
	return fmt.Sprintf("embed.OpenAIConfig{BaseURL:%q, Model:%q, Timeout:time.Duration(%d), MaxBatchSize:%d, APIToken:%q}",
		c.BaseURL, c.Model, int64(c.Timeout), c.MaxBatchSize, redactedToken(c.APIToken))
}

// LogValue implements slog.LogValuer so slog handlers — including the
// default JSONHandler — render the config with APIToken redacted.
// slog consults LogValuer before falling back to json.Marshal, so this
// covers the structured-logging path without hijacking json.Marshal
// itself (which stays canonical for callers that round-trip the
// config through JSON).
func (c OpenAIConfig) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("base_url", c.BaseURL),
		slog.String("model", c.Model),
		slog.Duration("timeout", c.Timeout),
		slog.Int("max_batch_size", c.MaxBatchSize),
		slog.String("api_token", redactedToken(c.APIToken)),
	)
}

// redactedToken returns the placeholder when a token is set, or "" when
// the config never carried one — empty stays empty so the rendering
// distinguishes "unset" from "set but hidden".
func redactedToken(token string) string {
	if token == "" {
		return ""
	}
	return redactedTokenPlaceholder
}

// OpenAI implements Embedder against an OpenAI-compatible HTTP embeddings API.
type OpenAI struct {
	config   OpenAIConfig
	strategy string
	client   *http.Client

	mu        sync.Mutex
	dimension int
}

// NewOpenAI returns an OpenAI-compatible embedder bound to cfg.
func NewOpenAI(cfg OpenAIConfig) *OpenAI {
	cfg.BaseURL = normalizeOpenAIBaseURL(cfg.BaseURL)
	cfg.Model = strings.TrimSpace(cfg.Model)
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = defaultOpenAITimeout
	}
	cfg.Timeout = timeout
	if cfg.MaxBatchSize <= 0 {
		cfg.MaxBatchSize = defaultOpenAIMaxBatchSize
	}
	return &OpenAI{
		config:   cfg,
		strategy: openAIStrategyForModel(cfg.Model),
		client:   &http.Client{Timeout: timeout},
	}
}

// Config returns the normalized embedder config.
func (e *OpenAI) Config() OpenAIConfig { return e.config }

// Unconfigured reports whether the embedder is missing a base URL or model.
func (e *OpenAI) Unconfigured() bool { return !e.config.Enabled() }

// Fingerprint returns a deterministic identifier for the endpoint, model, and
// input-preparation strategy.
func (e *OpenAI) Fingerprint() string {
	return "openai:" + strings.TrimSpace(e.config.BaseURL) + ":" + strings.TrimSpace(e.config.Model) + ":" + e.strategy
}

// Dimension probes the upstream on first use and then reuses the cached value.
func (e *OpenAI) Dimension(ctx context.Context) (int, error) {
	if d := e.cachedDimension(); d > 0 {
		return d, nil
	}
	vectors, err := e.embed(ctx, "query", []string{"dimension probe"})
	if err != nil {
		return 0, err
	}
	if len(vectors) != 1 || len(vectors[0]) == 0 {
		return 0, fmt.Errorf("stroma/embed: openai returned no embedding dimensions")
	}
	return len(vectors[0]), nil
}

// EmbedDocuments generates embeddings for indexed document texts.
func (e *OpenAI) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	return e.embed(ctx, "document", texts)
}

// EmbedQueries generates embeddings for query texts.
func (e *OpenAI) EmbedQueries(ctx context.Context, texts []string) ([][]float64, error) {
	return e.embed(ctx, "query", texts)
}

func (e *OpenAI) embed(ctx context.Context, purpose string, texts []string) ([][]float64, error) {
	if !e.config.Enabled() {
		return nil, fmt.Errorf("stroma/embed: openai embedder is not configured")
	}
	if len(texts) == 0 {
		return [][]float64{}, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}

	// NewOpenAI ensures MaxBatchSize > 0; guard here for the "someone
	// constructed OpenAI{} directly" path and for tests that mutate
	// config post-construction.
	batchSize := e.config.MaxBatchSize
	if batchSize <= 0 {
		batchSize = defaultOpenAIMaxBatchSize
	}
	if len(texts) <= batchSize {
		return e.embedBatch(ctx, purpose, texts)
	}

	// Derive an overall deadline that scales with batch count so every
	// sub-request gets a full Timeout window. A single flat deadline
	// (the pre-#63 shape) collapsed the entire loop under one Timeout:
	// one slow early batch shrank the remaining budget and made later
	// sub-requests fail with `context deadline exceeded` even though
	// each individually fit inside Timeout. Scaling by batch count
	// keeps the per-request budget honest while preserving an upper
	// bound on total wall time for callers who rely on Timeout to
	// bound batch runs. context.WithDeadline honors any tighter
	// caller-supplied deadline, so callers that want a smaller global
	// cap pass it via ctx.
	batches := (len(texts) + batchSize - 1) / batchSize
	if e.config.Timeout > 0 && batches > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, time.Now().Add(scaledMultiBatchBudget(e.config.Timeout, batches)))
		defer cancel()
	}

	// Chunk into sub-requests of at most batchSize and concatenate the
	// results in input order. Any sub-request failure fails the whole
	// call; callers retry at the EmbedDocuments/EmbedQueries boundary.
	// Note: earlier-batch vectors that already succeeded upstream are
	// discarded on retry — embedders are idempotent so this is correct
	// but wasteful. Richer per-batch progress/retry handling is out of
	// scope for #41 part 1.
	result := make([][]float64, 0, len(texts))
	for start := 0; start < len(texts); start += batchSize {
		end := start + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		vectors, err := e.embedBatch(ctx, purpose, texts[start:end])
		if err != nil {
			return nil, fmt.Errorf("stroma/embed: batch [%d:%d] of %d: %w", start, end, len(texts), err)
		}
		result = append(result, vectors...)
	}
	return result, nil
}

// scaledMultiBatchBudget returns perRequest * batches, saturating at
// math.MaxInt64 (~292 years) instead of wrapping. Without this guard a
// caller-reachable combination of a large Timeout and a large batch
// count (Timeout > MaxInt64 / batches) would overflow the int64 nanos
// in time.Duration and yield a negative / tiny derived deadline that
// cancels the call instantly. Saturating is effectively "no upper
// bound" for practical purposes, which matches the caller's intent:
// if they configured a huge Timeout they do not want batch scaling
// to surprise them by collapsing it.
func scaledMultiBatchBudget(perRequest time.Duration, batches int) time.Duration {
	if batches <= 0 || perRequest <= 0 {
		return perRequest
	}
	if int64(perRequest) > int64(math.MaxInt64)/int64(batches) {
		return time.Duration(math.MaxInt64)
	}
	return perRequest * time.Duration(batches)
}

// embedBatch issues a single embeddings request. Callers guarantee
// len(texts) is within the server's per-request limit; embed() handles
// that via MaxBatchSize chunking.
func (e *OpenAI) embedBatch(ctx context.Context, purpose string, texts []string) ([][]float64, error) {
	req, inputCount, err := e.buildEmbedRequest(ctx, purpose, texts)
	if err != nil {
		return nil, err
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("stroma/embed: call openai endpoint %s: %w", req.URL.String(), err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("stroma/embed: openai endpoint %s returned status %s", req.URL.String(), resp.Status)
	}

	return e.parseEmbedResponse(resp.Body, inputCount)
}

func (e *OpenAI) buildEmbedRequest(ctx context.Context, purpose string, texts []string) (*http.Request, int, error) {
	input := make([]string, 0, len(texts))
	for _, text := range texts {
		input = append(input, prepareOpenAIEmbeddingInput(e.strategy, purpose, text))
	}

	requestBody, err := json.Marshal(map[string]any{
		"model": e.config.Model,
		"input": input,
	})
	if err != nil {
		return nil, 0, fmt.Errorf("stroma/embed: encode openai request: %w", err)
	}

	endpoint := e.config.BaseURL + "/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(requestBody))
	if err != nil {
		return nil, 0, fmt.Errorf("stroma/embed: build openai request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.config.APIToken != "" {
		req.Header.Set("Authorization", "Bearer "+e.config.APIToken)
	}
	return req, len(input), nil
}

func (e *OpenAI) parseEmbedResponse(body io.Reader, inputCount int) ([][]float64, error) {
	// Cap how much we read before decoding. An unbounded json.Decode
	// against a hostile or misconfigured upstream can stream GiBs and
	// OOM the host — the client Timeout bounds wall time but not
	// bytes. LimitReader(max+1) lets us detect an overflow without
	// swallowing it silently.
	raw, err := io.ReadAll(io.LimitReader(body, maxEmbedResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("stroma/embed: read openai response: %w", err)
	}
	if int64(len(raw)) > maxEmbedResponseBytes {
		return nil, fmt.Errorf("stroma/embed: openai response exceeds %d bytes; aborting decode to avoid OOM", maxEmbedResponseBytes)
	}

	var payload struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}
	// json.Decoder matches the previous json.NewDecoder(body).Decode
	// behaviour — it tolerates trailing whitespace or filler after the
	// first complete JSON value, which nonconforming self-hosted
	// gateways sometimes emit. json.Unmarshal would reject those.
	if err := json.NewDecoder(bytes.NewReader(raw)).Decode(&payload); err != nil {
		return nil, fmt.Errorf("stroma/embed: decode openai response: %w", err)
	}
	if len(payload.Data) != inputCount {
		return nil, fmt.Errorf("stroma/embed: openai returned %d embedding(s) for %d input(s)", len(payload.Data), inputCount)
	}

	vectors := make([][]float64, inputCount)
	for i, item := range payload.Data {
		idx := item.Index
		if idx < 0 || idx >= inputCount {
			idx = i
		}
		if len(item.Embedding) == 0 {
			return nil, fmt.Errorf("stroma/embed: openai returned empty embedding for input %d", idx)
		}
		if err := e.cacheDimension(len(item.Embedding)); err != nil {
			return nil, err
		}
		vectors[idx] = item.Embedding
	}
	for i, vec := range vectors {
		if len(vec) == 0 {
			return nil, fmt.Errorf("stroma/embed: openai omitted embedding for input %d", i)
		}
	}
	return vectors, nil
}

func (e *OpenAI) cachedDimension() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.dimension
}

func (e *OpenAI) cacheDimension(dim int) error {
	if dim <= 0 {
		return fmt.Errorf("stroma/embed: openai returned a non-positive embedding dimension")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.dimension == 0 {
		e.dimension = dim
		return nil
	}
	if e.dimension != dim {
		return fmt.Errorf("stroma/embed: openai changed embedding dimension from %d to %d", e.dimension, dim)
	}
	return nil
}

func openAIStrategyForModel(model string) string {
	if strings.Contains(strings.ToLower(model), "nomic-embed-text") {
		return openAIStrategyNomicPrefix
	}
	return openAIStrategyPlain
}

func prepareOpenAIEmbeddingInput(strategy, purpose, text string) string {
	switch strategy {
	case openAIStrategyNomicPrefix:
		if purpose == "query" {
			return "search_query: " + text
		}
		return "search_document: " + text
	default:
		return text
	}
}

func normalizeOpenAIBaseURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return strings.TrimRight(raw, "/")
	}
	parsed.RawQuery = ""
	parsed.Fragment = ""
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	if parsed.Path == "" {
		parsed.Path = "/v1"
	}
	return strings.TrimRight(parsed.String(), "/")
}

// Compile-time assertion that *OpenAI satisfies Embedder.
var _ Embedder = (*OpenAI)(nil)
