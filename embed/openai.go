package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/dusk-network/stroma/v2/provider"
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

	// embedMaxRetryAfter caps how long a single retry gap will block
	// on a server-supplied Retry-After header. Set on every request's
	// provider.Policy, and also used by the multi-batch deadline math
	// so the safety-net deadline and the actual retry loop agree on
	// the worst-case per-retry wait. A hostile or misconfigured
	// upstream returning `Retry-After: 86400` would otherwise park
	// the goroutine for hours.
	embedMaxRetryAfter = 30 * time.Second

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

	// Timeout bounds a single embeddings HTTP attempt. With
	// MaxRetries=0 this is also the total wall-clock budget for a
	// single-batch call. With MaxRetries > 0 the wall-clock extends
	// beyond Timeout: each attempt is still bounded by Timeout, but
	// retries and Retry-After / backoff waits stack on top — see
	// MaxRetries for details.
	//
	// For multi-batch calls the embedder schedules a safety-net
	// parent deadline of approximately:
	//
	//   batches * ((MaxRetries+1) * Timeout + MaxRetries * 30s)
	//
	// where batches = ceil(len(texts)/MaxBatchSize) and 30s is the
	// embed-side cap on each inter-retry Retry-After / backoff wait.
	// Each sub-request keeps a full per-attempt window even when an
	// earlier batch retried through a rate limit, so a slow or
	// retrying early batch cannot starve later batches. The formula
	// saturates at math.MaxInt64 nanoseconds on overflow.
	//
	// A zero or negative value selects defaultOpenAITimeout (15s) at
	// NewOpenAI time. Callers that need a hard global cap should pass
	// a ctx with their own deadline; the embedder honors whichever
	// deadline trips first.
	Timeout  time.Duration
	APIToken string

	// MaxRetries caps the number of retry attempts after a retryable
	// failure (429, 5xx, connection reset, timeout). Zero disables
	// retries. A negative value is normalized to zero at NewOpenAI
	// time so a config typo or bad env parse cannot silently
	// short-circuit every request. Retry-After is always honoured
	// when present, up to an embed-side cap of embedMaxRetryAfter
	// (30s) per gap. Mirrors chat.OpenAIConfig.MaxRetries semantics
	// so consumers that swap between chat and embed get the same
	// retry surface.
	//
	// Retries replay the embeddings POST unchanged — no idempotency
	// key is sent — so on ambiguous transport failures (timeout,
	// connection reset) an upstream that already processed the first
	// attempt may bill both. Self-hosted gateways generally have no
	// billing concern; callers hitting real OpenAI with cost ceilings
	// should leave this at zero or cap it tightly.
	MaxRetries int

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
	return fmt.Sprintf("OpenAIConfig{BaseURL:%q Model:%q Timeout:%s MaxRetries:%d MaxBatchSize:%d APIToken:%q}",
		c.BaseURL, c.Model, c.Timeout, c.MaxRetries, c.MaxBatchSize, redactedToken(c.APIToken))
}

// GoString returns a redacted Go-syntax rendering of the config for %#v.
// Without this, %#v falls back to reflection and surfaces the raw
// APIToken field value. Timeout is formatted as time.Duration(ns) so
// the result stays a valid Go composite literal a reader could paste
// back into source (Duration's default %s form "2s" is human-readable
// but not Go-parseable).
func (c OpenAIConfig) GoString() string {
	return fmt.Sprintf("embed.OpenAIConfig{BaseURL:%q, Model:%q, Timeout:time.Duration(%d), MaxRetries:%d, MaxBatchSize:%d, APIToken:%q}",
		c.BaseURL, c.Model, int64(c.Timeout), c.MaxRetries, c.MaxBatchSize, redactedToken(c.APIToken))
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
		slog.Int("max_retries", c.MaxRetries),
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
//
// Safe for concurrent use by multiple goroutines once constructed: the
// cached dimension is guarded by a sync.Mutex, and http.Client is
// goroutine-safe by contract. Callers may share a single *OpenAI
// across a long-lived service without additional synchronization.
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
	if cfg.MaxRetries < 0 {
		// Normalize pathological negative MaxRetries at construction so
		// both the multi-batch deadline math and provider.Policy agree
		// on attempt counts. Without this, a negative value (e.g. from
		// a bad env parse) would short-circuit provider.Do's retry loop
		// and hard-fail every request with a generic error before any
		// HTTP round trip.
		cfg.MaxRetries = 0
	}
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

	// Derive an overall safety-net deadline that covers every batch's
	// full retry chain — attempts AND inter-retry waits — so one slow
	// or rate-limited early batch cannot starve later batches.
	//
	// Three historical footguns informed this shape:
	//   - Pre-#63: one flat Timeout around the whole loop collapsed
	//     the budget; later batches got ctx.DeadlineExceeded.
	//   - #63: scaling by batches fixed the per-attempt starvation
	//     but left no room for retries — which #73 then introduced.
	//   - #73 + first adversarial-review fix: scaling by
	//     batches * (retries+1) covered per-attempt time but not the
	//     retry-wait sleeps (backoff + Retry-After), so a 429 with
	//     Retry-After could still starve batch 1.
	//
	// Current shape accounts for all three: each batch gets
	// (MaxRetries+1) * Timeout of attempt time plus MaxRetries *
	// embedMaxRetryAfter of inter-retry wait budget. NewOpenAI has
	// already clamped MaxRetries >= 0, so attemptsPerBatch >= 1.
	// context.WithDeadline honors any tighter caller-supplied
	// deadline, so callers that want a smaller global cap pass it via
	// ctx.
	batches := (len(texts) + batchSize - 1) / batchSize
	attemptsPerBatch := 1 + e.config.MaxRetries
	if e.config.Timeout > 0 && batches > 0 {
		perBatch := perBatchWallClockBudget(e.config.Timeout, attemptsPerBatch, embedMaxRetryAfter)
		totalBudget := scaledMultiBatchBudget(perBatch, batches)
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, time.Now().Add(totalBudget))
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

// perBatchWallClockBudget returns an upper bound on wall-clock time
// one batch can consume including retries and retry waits, saturating
// at math.MaxInt64 nanoseconds on overflow. attemptsPerBatch covers
// (1 + MaxRetries) HTTP attempts, each bounded by perAttempt. With R =
// attemptsPerBatch-1 retries, the inter-attempt sleep budget is
// bounded by R * maxRetryWait — that covers both exponential backoff
// (capped at 2s per gap in provider) and a server-supplied Retry-After
// (capped at maxRetryWait in provider). Scaling by batches is the
// caller's responsibility; this helper stops at one batch so the
// retry-wait term is not double-counted.
func perBatchWallClockBudget(perAttempt time.Duration, attemptsPerBatch int, maxRetryWait time.Duration) time.Duration {
	attempts := scaledMultiBatchBudget(perAttempt, attemptsPerBatch)
	retries := attemptsPerBatch - 1
	if retries <= 0 || maxRetryWait <= 0 {
		return attempts
	}
	waits := scaledMultiBatchBudget(maxRetryWait, retries)
	// Saturating add: on overflow return the int64 max so the derived
	// deadline behaves as "effectively unbounded", matching
	// scaledMultiBatchBudget's own overflow policy.
	if attempts > time.Duration(math.MaxInt64)-waits {
		return time.Duration(math.MaxInt64)
	}
	return attempts + waits
}

// embedBatch issues a single embeddings request through the shared
// provider core. Callers guarantee len(texts) is within the server's
// per-request limit; embed() handles that via MaxBatchSize chunking.
func (e *OpenAI) embedBatch(ctx context.Context, purpose string, texts []string) ([][]float64, error) {
	input := make([]string, 0, len(texts))
	for _, text := range texts {
		input = append(input, prepareOpenAIEmbeddingInput(e.strategy, purpose, text))
	}
	requestBody, err := json.Marshal(map[string]any{
		"model": e.config.Model,
		"input": input,
	})
	if err != nil {
		return nil, fmt.Errorf("stroma/embed: encode openai request: %w", err)
	}

	inputCount := len(input)
	endpoint := e.config.BaseURL + "/embeddings"
	details := provider.FailureDetails{
		Model:      e.config.Model,
		Endpoint:   e.config.BaseURL,
		TimeoutMS:  int(e.config.Timeout / time.Millisecond),
		MaxRetries: e.config.MaxRetries,
		BatchSize:  inputCount,
		InputCount: inputCount,
	}
	target := provider.Target{
		Method: http.MethodPost,
		URL:    endpoint,
		Body:   requestBody,
		Token:  e.config.APIToken,
	}
	// Embedding responses are larger than chat completions — a full
	// batch of 512 inputs × ~1536-dim floats fills several MiB. Keep
	// the historical 32 MiB cap rather than falling to the provider
	// default (4 MiB) so self-hosted gateways that concatenate batches
	// still decode cleanly.
	policy := provider.Policy{
		MaxRetries: e.config.MaxRetries,
		// MaxRetryAfter is set explicitly so the provider retry loop
		// and the multi-batch deadline math in embed() agree on the
		// worst-case per-retry wait. Without this they would drift
		// apart if provider ever changed its default and the safety-
		// net deadline would under-bound retry chains that honored
		// the new provider cap.
		MaxRetryAfter:    embedMaxRetryAfter,
		MaxResponseBytes: maxEmbedResponseBytes,
	}

	return provider.Do(ctx, e.client, target, details, policy,
		func(_ *http.Response, body []byte) ([][]float64, error) {
			return e.parseEmbedResponse(body, inputCount)
		},
	)
}

func (e *OpenAI) parseEmbedResponse(body []byte, inputCount int) ([][]float64, error) {
	var payload struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}
	// json.Decoder tolerates trailing whitespace or filler after the
	// first complete JSON value, which nonconforming self-hosted
	// gateways sometimes emit. json.Unmarshal would reject those.
	if err := json.NewDecoder(bytes.NewReader(body)).Decode(&payload); err != nil {
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
