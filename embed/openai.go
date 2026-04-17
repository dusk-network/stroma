package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
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
// APIToken is redacted from every standard text/JSON representation —
// String, GoString, MarshalJSON, MarshalText all return the config with
// the token replaced by "[REDACTED]". Direct field access still yields
// the raw value so the embedder itself can sign requests; redaction is
// defense-in-depth against accidental fmt/slog/json.Marshal disclosure.
type OpenAIConfig struct {
	BaseURL  string
	Model    string
	Timeout  time.Duration
	APIToken string
}

// Enabled reports whether the config is usable for requests.
func (c OpenAIConfig) Enabled() bool {
	return strings.TrimSpace(c.BaseURL) != "" && strings.TrimSpace(c.Model) != ""
}

// String returns a redacted, human-readable rendering of the config.
// fmt verbs %v and %s route through this method.
func (c OpenAIConfig) String() string {
	return fmt.Sprintf("OpenAIConfig{BaseURL:%q Model:%q Timeout:%s APIToken:%q}",
		c.BaseURL, c.Model, c.Timeout, redactedToken(c.APIToken))
}

// GoString returns a redacted Go-syntax rendering of the config for %#v.
// Without this, %#v falls back to reflection and surfaces the raw
// APIToken field value.
func (c OpenAIConfig) GoString() string {
	return fmt.Sprintf("embed.OpenAIConfig{BaseURL:%q, Model:%q, Timeout:%s, APIToken:%q}",
		c.BaseURL, c.Model, c.Timeout, redactedToken(c.APIToken))
}

// MarshalJSON emits a JSON document with APIToken redacted. Any caller
// that pretty-prints the config via json.Marshal — structured loggers,
// diagnostics dumps, API responses — gets the redacted value. Uses a
// map rather than an anonymous struct so no intermediate type carries
// a field named APIToken that a static-analysis pattern matcher could
// misclassify as a real secret.
func (c OpenAIConfig) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]any{
		"base_url":  c.BaseURL,
		"model":     c.Model,
		"timeout":   c.Timeout,
		"api_token": redactedToken(c.APIToken),
	})
}

// MarshalText emits a redacted text rendering for encoding.TextMarshaler
// consumers (some log libraries prefer this path over Stringer).
func (c OpenAIConfig) MarshalText() ([]byte, error) {
	return []byte(c.String()), nil
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
	if err := json.Unmarshal(raw, &payload); err != nil {
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
