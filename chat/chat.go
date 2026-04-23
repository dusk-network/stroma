// Package chat is stroma's OpenAI-compatible chat completion primitive,
// the sibling of embed.OpenAI. It ships the narrow HTTP surface —
// request shape, retry / Retry-After, classified errors, response-size
// bounding — and nothing else. Prompt templating, JSON-shape
// enforcement of model outputs, and runtime-specific labelling stay at
// the caller.
//
// Constructed like embed.OpenAI:
//
//	client := chat.NewOpenAI(chat.OpenAIConfig{
//	    BaseURL:    "https://api.openai.com/v1",
//	    Model:      "gpt-4o-mini",
//	    APIToken:   os.Getenv("OPENAI_API_KEY"),
//	    Timeout:    30 * time.Second,
//	    MaxRetries: 2,
//	})
//	text, err := client.ChatCompletionText(ctx, []chat.Message{
//	    {Role: "system", Content: "..."},
//	    {Role: "user", Content: question},
//	}, 0, 512)
package chat

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/dusk-network/stroma/v2/provider"
)

const (
	defaultChatTimeout       = 30 * time.Second
	redactedTokenPlaceholder = "[REDACTED]"
)

// OpenAIConfig configures an OpenAI-compatible chat completion client.
//
// APIToken is redacted from every log/display representation — String,
// GoString, and slog.LogValuer all render the token as "[REDACTED]".
// Direct field access still yields the raw value so the client can
// sign requests.
//
// json.Marshal and encoding.TextMarshaler stay canonical: OpenAIConfig
// is a public configuration type, and overriding either would silently
// break callers that persist or round-trip the config. Callers that
// need a redacted JSON view should marshal cfg.String() or build a
// dedicated view type.
type OpenAIConfig struct {
	BaseURL string
	Model   string

	// Timeout bounds a single chat completion request. A zero or
	// negative value selects defaultChatTimeout (30s) at NewOpenAI
	// time. Callers that want a tighter global cap should pass a ctx
	// with their own deadline; the client honours whichever deadline
	// trips first.
	Timeout  time.Duration
	APIToken string

	// MaxRetries caps the number of retry attempts after a retryable
	// failure (429, 5xx, connection reset, timeout). Zero disables
	// retries. Retry-After is always honoured when present.
	MaxRetries int

	// MaxResponseBytes bounds the response body; zero selects the
	// provider default (4 MiB). Chat completions fit comfortably in
	// the default.
	MaxResponseBytes int64
}

// Enabled reports whether the config is usable for requests.
func (c OpenAIConfig) Enabled() bool {
	return strings.TrimSpace(c.BaseURL) != "" && strings.TrimSpace(c.Model) != ""
}

// String returns a redacted, human-readable rendering of the config.
func (c OpenAIConfig) String() string {
	return fmt.Sprintf("OpenAIConfig{BaseURL:%q Model:%q Timeout:%s MaxRetries:%d APIToken:%q}",
		c.BaseURL, c.Model, c.Timeout, c.MaxRetries, redactedToken(c.APIToken))
}

// GoString returns a redacted Go-syntax rendering for %#v. Timeout is
// emitted as time.Duration(ns) so the result stays a valid Go literal.
func (c OpenAIConfig) GoString() string {
	return fmt.Sprintf("chat.OpenAIConfig{BaseURL:%q, Model:%q, Timeout:time.Duration(%d), MaxRetries:%d, APIToken:%q}",
		c.BaseURL, c.Model, int64(c.Timeout), c.MaxRetries, redactedToken(c.APIToken))
}

// LogValue implements slog.LogValuer so slog handlers render the
// config with APIToken redacted.
func (c OpenAIConfig) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("base_url", c.BaseURL),
		slog.String("model", c.Model),
		slog.Duration("timeout", c.Timeout),
		slog.Int("max_retries", c.MaxRetries),
		slog.String("api_token", redactedToken(c.APIToken)),
	)
}

func redactedToken(token string) string {
	if token == "" {
		return ""
	}
	return redactedTokenPlaceholder
}

// Message is one item in a chat completion request. Matches the
// OpenAI chat-completion protocol.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OpenAI is an OpenAI-compatible chat completion client. Safe for
// concurrent use; callers may share a single *OpenAI across a
// long-lived service.
type OpenAI struct {
	config OpenAIConfig
	client *http.Client
}

// NewOpenAI returns a client bound to cfg.
func NewOpenAI(cfg OpenAIConfig) *OpenAI {
	cfg.BaseURL = normalizeBaseURL(cfg.BaseURL)
	cfg.Model = strings.TrimSpace(cfg.Model)
	if cfg.Timeout <= 0 {
		cfg.Timeout = defaultChatTimeout
	}
	return &OpenAI{
		config: cfg,
		client: &http.Client{Timeout: cfg.Timeout},
	}
}

// Config returns the normalized config.
func (c *OpenAI) Config() OpenAIConfig { return c.config }

// Unconfigured reports whether the client is missing a base URL or model.
func (c *OpenAI) Unconfigured() bool { return !c.config.Enabled() }

type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

type chatResponse struct {
	Choices []chatChoice    `json:"choices"`
	Err     json.RawMessage `json:"error,omitempty"`
}

type chatChoice struct {
	Message chatChoiceMessage `json:"message"`
}

type chatChoiceMessage struct {
	Content json.RawMessage `json:"content"`
}

// ChatCompletionText sends messages to the chat completion endpoint
// and returns the assistant text. Retries, Retry-After, and failure
// classification are handled by the shared provider core. Decode
// failures, empty choices, and empty message text are all reported as
// FailureClassSchemaMismatch — they are chat-protocol invariants, not
// product-JSON-shape concerns.
func (c *OpenAI) ChatCompletionText(ctx context.Context, messages []Message, temperature float64, maxTokens int) (string, error) {
	if !c.config.Enabled() {
		return "", fmt.Errorf("stroma/chat: openai client is not configured")
	}
	body, err := json.Marshal(chatRequest{
		Model:       c.config.Model,
		Messages:    messages,
		Temperature: temperature,
		MaxTokens:   maxTokens,
	})
	if err != nil {
		return "", fmt.Errorf("stroma/chat: encode request: %w", err)
	}

	details := provider.FailureDetails{
		Model:      c.config.Model,
		Endpoint:   c.config.BaseURL,
		TimeoutMS:  int(c.config.Timeout / time.Millisecond),
		MaxRetries: c.config.MaxRetries,
		InputCount: len(messages),
	}

	target := provider.Target{
		Method: http.MethodPost,
		URL:    c.config.BaseURL + "/chat/completions",
		Body:   body,
		Token:  c.config.APIToken,
	}
	policy := provider.Policy{
		MaxRetries:       c.config.MaxRetries,
		MaxResponseBytes: c.config.MaxResponseBytes,
	}

	return provider.Do(ctx, c.client, target, details, policy,
		func(_ *http.Response, responseBody []byte) (string, error) {
			var payload chatResponse
			if err := json.Unmarshal(responseBody, &payload); err != nil {
				failure := details
				failure.FailureClass = provider.FailureClassSchemaMismatch
				return "", provider.NewError(failure, "decode response: %v", err)
			}
			if message := provider.ExtractErrorValue(payload.Err); message != "" {
				failure := details
				failure.FailureClass = provider.Classify(0, nil, message)
				return "", provider.NewError(failure, "%s returned an error: %s", target.URL, message)
			}
			if len(payload.Choices) == 0 {
				failure := details
				failure.FailureClass = provider.FailureClassSchemaMismatch
				return "", provider.NewError(failure, "response contained no choices")
			}
			text, ok := ExtractMessageText(payload.Choices[0].Message.Content)
			if !ok {
				failure := details
				failure.FailureClass = provider.FailureClassSchemaMismatch
				return "", provider.NewError(failure, "response message content is not a recognized shape")
			}
			if text == "" {
				failure := details
				failure.FailureClass = provider.FailureClassSchemaMismatch
				return "", provider.NewError(failure, "response contained an empty message")
			}
			return text, nil
		},
	)
}

// ExtractMessageText returns the text content of a chat choice
// message and whether the content had a recognised shape. The chat
// protocol defines content as either a plain string
// ({"content":"..."}) or a multi-part array used by tool-capable
// models ({"content":[{"type":"text","text":"..."}]}); any other
// shape (object, number, bool, null) returns ("", false) so the
// caller can classify the response as schema_mismatch rather than
// persist raw JSON as assistant text.
func ExtractMessageText(raw json.RawMessage) (string, bool) {
	// Literal JSON `null` unmarshals into a string as "" with nil
	// error, which would otherwise pass as a recognised-but-empty
	// text shape. Null is not a valid content shape for this protocol
	// (some gateways emit it when a response is tool_call-only); flag
	// it as unrecognised so the caller classifies schema_mismatch.
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return "", false
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return strings.TrimSpace(text), true
	}

	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var builder strings.Builder
		for _, part := range parts {
			if strings.TrimSpace(part.Text) == "" {
				continue
			}
			if builder.Len() > 0 {
				builder.WriteString("\n")
			}
			builder.WriteString(strings.TrimSpace(part.Text))
		}
		return strings.TrimSpace(builder.String()), true
	}

	return "", false
}

func normalizeBaseURL(raw string) string {
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
