// Package chat is stroma's OpenAI-compatible chat completion primitive,
// the sibling of embed.OpenAI. It ships the narrow HTTP surface —
// request shape, retry / Retry-After, classified errors, response-size
// bounding — plus a product-neutral JSON object decoding helper.
// Prompt templating, semantic validation of model outputs, and
// runtime-specific labelling stay at the caller.
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
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"reflect"
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
	// retries; negative values normalize to zero. Retry-After is always
	// honoured when present.
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
	if cfg.MaxRetries < 0 {
		cfg.MaxRetries = 0
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
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Temperature    float64         `json:"temperature"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	ResponseFormat json.RawMessage `json:"response_format,omitempty"`
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

// JSONCallRequest configures one structured JSON chat completion call.
//
// Schema is optional JSON Schema. When present, it is sent to the
// OpenAI-compatible endpoint as response_format.type=json_schema with a
// neutral schema name; Stroma still validates the assistant response
// client-side and returns schema_mismatch for malformed, missing, or
// non-object JSON.
type JSONCallRequest struct {
	Messages    []Message
	Temperature float64
	MaxTokens   int
	Schema      json.RawMessage
	Retry       JSONRetryPolicy
}

// JSONRetryPolicy controls the bounded client-side repair pass for
// ChatCompletionJSON. MaxRepairs values above one are capped at one so a
// malformed response cannot turn into an autonomous loop.
type JSONRetryPolicy struct {
	MaxRepairs int
	// RepairMessage overrides the generic repair instruction appended
	// after a malformed JSON response.
	RepairMessage string
}

// ChatCompletionText sends messages to the chat completion endpoint
// and returns the assistant text. Retries, Retry-After, and failure
// classification are handled by the shared provider core. Decode
// failures, empty choices, and empty message text are all reported as
// FailureClassSchemaMismatch — they are chat-protocol invariants, not
// product-JSON-shape concerns.
func (c *OpenAI) ChatCompletionText(ctx context.Context, messages []Message, temperature float64, maxTokens int) (string, error) {
	return c.chatCompletionText(ctx, messages, temperature, maxTokens, nil)
}

// ChatCompletionJSON sends messages to the chat completion endpoint,
// extracts a JSON object from the assistant response, and decodes it
// into target. Invalid JSON, missing JSON, non-object JSON, and typed
// decode failures are classified as provider.FailureClassSchemaMismatch.
//
// The helper is deliberately product-neutral: callers own prompts,
// schema design, and any semantic validation after decoding.
func (c *OpenAI) ChatCompletionJSON(ctx context.Context, req JSONCallRequest, target any) error {
	if err := validateJSONTarget(target); err != nil {
		return err
	}
	responseFormat, err := jsonResponseFormat(req.Schema)
	if err != nil {
		return err
	}

	details := c.failureDetails(len(req.Messages))
	maxRepairs := req.Retry.MaxRepairs
	if maxRepairs > 1 {
		maxRepairs = 1
	}
	if maxRepairs < 0 {
		maxRepairs = 0
	}

	messages := cloneMessages(req.Messages)
	var lastErr error
	for attempt := 0; attempt <= maxRepairs; attempt++ {
		text, err := c.chatCompletionText(ctx, messages, req.Temperature, req.MaxTokens, responseFormat)
		if err != nil {
			if shouldRepairSchemaMismatch(err, attempt, maxRepairs) {
				messages = append(messages, Message{Role: "user", Content: repairMessage(req.Retry)})
				lastErr = err
				continue
			}
			return err
		}
		raw, err := ExtractJSONObject(text)
		if err != nil {
			lastErr = provider.NewErrorCause(schemaMismatchDetails(details), err, "structured JSON response: %v", err)
		} else if err := decodeJSONTarget(raw, target); err != nil {
			lastErr = provider.NewErrorCause(schemaMismatchDetails(details), err, "decode structured JSON response: %v", err)
		} else {
			return nil
		}
		if attempt >= maxRepairs {
			break
		}
		messages = append(messages,
			Message{Role: "assistant", Content: text},
			Message{Role: "user", Content: repairMessage(req.Retry)},
		)
	}
	return lastErr
}

func validateJSONTarget(target any) error {
	if target == nil {
		return fmt.Errorf("stroma/chat: JSON target is nil")
	}
	value := reflect.ValueOf(target)
	if value.Kind() != reflect.Ptr || value.IsNil() {
		return fmt.Errorf("stroma/chat: JSON target must be a non-nil pointer")
	}
	return nil
}

func decodeJSONTarget(raw json.RawMessage, target any) error {
	value := reflect.ValueOf(target)
	scratch := reflect.New(value.Elem().Type())
	if err := json.Unmarshal(raw, scratch.Interface()); err != nil {
		return err
	}
	value.Elem().Set(scratch.Elem())
	return nil
}

func shouldRepairSchemaMismatch(err error, attempt, maxRepairs int) bool {
	if attempt >= maxRepairs {
		return false
	}
	var perr *provider.Error
	return errors.As(err, &perr) && perr.FailureClass() == provider.FailureClassSchemaMismatch
}

func (c *OpenAI) chatCompletionText(ctx context.Context, messages []Message, temperature float64, maxTokens int, responseFormat json.RawMessage) (string, error) {
	if !c.config.Enabled() {
		return "", fmt.Errorf("stroma/chat: openai client is not configured")
	}
	body, err := json.Marshal(chatRequest{
		Model:          c.config.Model,
		Messages:       messages,
		Temperature:    temperature,
		MaxTokens:      maxTokens,
		ResponseFormat: responseFormat,
	})
	if err != nil {
		return "", fmt.Errorf("stroma/chat: encode request: %w", err)
	}

	details := c.failureDetails(len(messages))

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

func (c *OpenAI) failureDetails(inputCount int) provider.FailureDetails {
	return provider.FailureDetails{
		Model:      c.config.Model,
		Endpoint:   c.config.BaseURL,
		TimeoutMS:  int(c.config.Timeout / time.Millisecond),
		MaxRetries: c.config.MaxRetries,
		InputCount: inputCount,
	}
}

func schemaMismatchDetails(details provider.FailureDetails) provider.FailureDetails {
	details.FailureClass = provider.FailureClassSchemaMismatch
	return details
}

func cloneMessages(messages []Message) []Message {
	if len(messages) == 0 {
		return nil
	}
	out := make([]Message, len(messages))
	copy(out, messages)
	return out
}

func repairMessage(policy JSONRetryPolicy) string {
	if msg := strings.TrimSpace(policy.RepairMessage); msg != "" {
		return msg
	}
	return "Return only one valid JSON object matching the requested schema. Do not include prose, Markdown fences, arrays, or multiple JSON values."
}

func jsonResponseFormat(schema json.RawMessage) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(schema)
	if len(trimmed) == 0 {
		return nil, nil
	}
	if !json.Valid(trimmed) {
		return nil, fmt.Errorf("stroma/chat: invalid JSON schema")
	}
	responseFormat := struct {
		Type       string `json:"type"`
		JSONSchema struct {
			Name   string          `json:"name"`
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	}{
		Type: "json_schema",
	}
	responseFormat.JSONSchema.Name = "stroma_json_response"
	responseFormat.JSONSchema.Schema = append(json.RawMessage(nil), trimmed...)
	return json.Marshal(responseFormat)
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

var (
	errJSONRootNotObject = errors.New("JSON root is not an object")
	errJSONExtraContent  = errors.New("JSON response contains extra content")
)

// ExtractJSONObject extracts a single JSON object from assistant text.
// It accepts a bare object, a Markdown-fenced object, or prose wrapped
// around an object. Valid JSON whose root is not an object is rejected
// rather than mining nested objects out of the wrong shape.
func ExtractJSONObject(text string) (json.RawMessage, error) {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil, fmt.Errorf("empty response")
	}
	if raw, err := decodeExactJSONObject(trimmed); err == nil {
		return raw, nil
	} else if errors.Is(err, errJSONRootNotObject) || errors.Is(err, errJSONExtraContent) {
		return nil, err
	}

	for start := 0; start < len(trimmed); start++ {
		if trimmed[start] != '{' {
			continue
		}
		if raw, ok := extractObjectFrom(trimmed[start:]); ok {
			return raw, nil
		}
	}
	return nil, fmt.Errorf("response did not contain a JSON object")
}

func extractObjectFrom(text string) (json.RawMessage, bool) {
	depth := 0
	inString := false
	escaped := false
	for i := 0; i < len(text); i++ {
		ch := text[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			switch ch {
			case '\\':
				escaped = true
			case '"':
				inString = false
			}
			continue
		}
		switch ch {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				raw, err := decodeExactJSONObject(text[:i+1])
				return raw, err == nil
			}
			if depth < 0 {
				return nil, false
			}
		}
	}
	return nil, false
}

func decodeExactJSONObject(text string) (json.RawMessage, error) {
	dec := json.NewDecoder(strings.NewReader(text))
	dec.UseNumber()
	var raw json.RawMessage
	if err := dec.Decode(&raw); err != nil {
		return nil, err
	}
	if len(bytes.TrimSpace(raw)) == 0 || bytes.TrimSpace(raw)[0] != '{' {
		return nil, errJSONRootNotObject
	}
	var extra json.RawMessage
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return nil, errJSONExtraContent
		}
		return nil, err
	}
	return append(json.RawMessage(nil), raw...), nil
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
