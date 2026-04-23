package chat

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/dusk-network/stroma/v2/provider"
)

func TestOpenAIConfigRedactsToken(t *testing.T) {
	cfg := OpenAIConfig{
		BaseURL:  "http://x/v1",
		Model:    "gpt",
		APIToken: "sk-supersecret",
	}
	// Direct calls exercise the Stringer / GoStringer implementations;
	// the stdlib contract for fmt %v / %#v routes through them, so
	// duplicating that check here would be redundant.
	for _, got := range []string{cfg.String(), cfg.GoString()} {
		if strings.Contains(got, "sk-supersecret") {
			t.Errorf("config rendering leaked token: %s", got)
		}
		if !strings.Contains(got, "[REDACTED]") {
			t.Errorf("config rendering missing redaction marker: %s", got)
		}
	}

	// slog JSON path should also redact.
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))
	logger.Info("config", "cfg", cfg)
	line := buf.String()
	if strings.Contains(line, "sk-supersecret") {
		t.Errorf("slog leaked token: %s", line)
	}
	if !strings.Contains(line, "[REDACTED]") {
		t.Errorf("slog missing redaction marker: %s", line)
	}
}

func TestOpenAIConfigRedactsEmptyTokenStaysEmpty(t *testing.T) {
	// Distinguish "unset" from "set but hidden".
	cfg := OpenAIConfig{BaseURL: "http://x/v1", Model: "gpt"}
	s := cfg.String()
	if strings.Contains(s, "[REDACTED]") {
		t.Errorf("empty token should not render REDACTED: %s", s)
	}
}

func TestNewOpenAINormalisesDefaults(t *testing.T) {
	cfg := NewOpenAI(OpenAIConfig{BaseURL: "http://x/v1/", Model: "  gpt-4o-mini "}).Config()
	if cfg.BaseURL != "http://x/v1" {
		t.Errorf("BaseURL = %q, want http://x/v1", cfg.BaseURL)
	}
	if cfg.Model != "gpt-4o-mini" {
		t.Errorf("Model = %q, want gpt-4o-mini", cfg.Model)
	}
	if cfg.Timeout != defaultChatTimeout {
		t.Errorf("Timeout = %s, want %s", cfg.Timeout, defaultChatTimeout)
	}
}

func TestNewOpenAIBaseURLBareHostAddsV1(t *testing.T) {
	cfg := NewOpenAI(OpenAIConfig{BaseURL: "http://x", Model: "m"}).Config()
	if cfg.BaseURL != "http://x/v1" {
		t.Errorf("BaseURL = %q, want http://x/v1", cfg.BaseURL)
	}
}

func TestUnconfiguredReportsMissingConfig(t *testing.T) {
	cases := []OpenAIConfig{
		{},
		{BaseURL: "http://x/v1"},
		{Model: "gpt"},
	}
	for _, cfg := range cases {
		c := NewOpenAI(cfg)
		if !c.Unconfigured() {
			t.Errorf("Unconfigured() = false for %+v", cfg)
		}
	}
	c := NewOpenAI(OpenAIConfig{BaseURL: "http://x/v1", Model: "gpt"})
	if c.Unconfigured() {
		t.Errorf("Unconfigured() = true for fully-configured client")
	}
}

func TestChatCompletionTextSuccess(t *testing.T) {
	var captured struct {
		Model       string
		Temperature float64
		MaxTokens   int
		Messages    []Message
		Auth        string
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body chatRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		captured.Model = body.Model
		captured.Temperature = body.Temperature
		captured.MaxTokens = body.MaxTokens
		captured.Messages = body.Messages
		captured.Auth = r.Header.Get("Authorization")

		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"hi"}}]}`)
	}))
	t.Cleanup(server.Close)

	c := NewOpenAI(OpenAIConfig{BaseURL: server.URL, Model: "gpt", APIToken: "sk"})
	out, err := c.ChatCompletionText(context.Background(),
		[]Message{{Role: "user", Content: "hello"}},
		0.25, 128,
	)
	if err != nil {
		t.Fatalf("ChatCompletionText err = %v", err)
	}
	if out != "hi" {
		t.Errorf("out = %q, want hi", out)
	}
	if captured.Model != "gpt" {
		t.Errorf("request model = %q, want gpt", captured.Model)
	}
	if captured.Temperature != 0.25 {
		t.Errorf("request temperature = %v, want 0.25", captured.Temperature)
	}
	if captured.MaxTokens != 128 {
		t.Errorf("request max_tokens = %d, want 128", captured.MaxTokens)
	}
	if len(captured.Messages) != 1 || captured.Messages[0].Content != "hello" {
		t.Errorf("request messages = %+v", captured.Messages)
	}
	if captured.Auth != "Bearer sk" {
		t.Errorf("Authorization = %q, want Bearer sk", captured.Auth)
	}
}

func TestChatCompletionTextHandlesPartArrayContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":[{"type":"text","text":"alpha"},{"type":"text","text":"beta"}]}}]}`)
	}))
	t.Cleanup(server.Close)

	out, err := NewOpenAI(OpenAIConfig{BaseURL: server.URL, Model: "gpt"}).
		ChatCompletionText(context.Background(), []Message{{Role: "user", Content: "q"}}, 0, 0)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if out != "alpha\nbeta" {
		t.Errorf("out = %q, want alpha\\nbeta", out)
	}
}

func TestChatCompletionTextSchemaMismatchOnEmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{"choices":[]}`)
	}))
	t.Cleanup(server.Close)

	_, err := NewOpenAI(OpenAIConfig{BaseURL: server.URL, Model: "gpt"}).
		ChatCompletionText(context.Background(), []Message{{Role: "user", Content: "q"}}, 0, 0)
	var perr *provider.Error
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *provider.Error", err)
	}
	if perr.FailureClass() != provider.FailureClassSchemaMismatch {
		t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), provider.FailureClassSchemaMismatch)
	}
}

func TestChatCompletionTextSchemaMismatchOnEmptyMessageText(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":""}}]}`)
	}))
	t.Cleanup(server.Close)

	_, err := NewOpenAI(OpenAIConfig{BaseURL: server.URL, Model: "gpt"}).
		ChatCompletionText(context.Background(), []Message{{Role: "user", Content: "q"}}, 0, 0)
	var perr *provider.Error
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *provider.Error", err)
	}
	if perr.FailureClass() != provider.FailureClassSchemaMismatch {
		t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), provider.FailureClassSchemaMismatch)
	}
}

func TestChatCompletionTextSchemaMismatchOnUnsupportedContentShape(t *testing.T) {
	// Guards against API version skew or a faulty gateway emitting
	// `content` as an object, number, bool, or null. Raw JSON must not
	// leak as assistant text — the failure must classify as schema_mismatch.
	cases := map[string]string{
		"object":  `{"choices":[{"message":{"content":{"type":"foo"}}}]}`,
		"number":  `{"choices":[{"message":{"content":42}}]}`,
		"boolean": `{"choices":[{"message":{"content":true}}]}`,
		"null":    `{"choices":[{"message":{"content":null}}]}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = fmt.Fprint(w, body)
			}))
			t.Cleanup(server.Close)

			out, err := NewOpenAI(OpenAIConfig{BaseURL: server.URL, Model: "gpt"}).
				ChatCompletionText(context.Background(), []Message{{Role: "user", Content: "q"}}, 0, 0)
			if err == nil {
				t.Fatalf("err = nil, want schema_mismatch (out=%q)", out)
			}
			if out != "" {
				t.Errorf("out = %q, want empty — raw JSON must not leak as assistant text", out)
			}
			var perr *provider.Error
			if !errors.As(err, &perr) {
				t.Fatalf("err = %v, want *provider.Error", err)
			}
			if perr.FailureClass() != provider.FailureClassSchemaMismatch {
				t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), provider.FailureClassSchemaMismatch)
			}
		})
	}
}

func TestExtractMessageTextDirectly(t *testing.T) {
	got, ok := ExtractMessageText(json.RawMessage(`"  hello  "`))
	if !ok || got != "hello" {
		t.Errorf("string content: got (%q, %v), want (hello, true)", got, ok)
	}

	got, ok = ExtractMessageText(json.RawMessage(`[{"type":"text","text":"a"},{"type":"text","text":""},{"type":"text","text":"b"}]`))
	if !ok || got != "a\nb" {
		t.Errorf("parts array: got (%q, %v), want (a\\nb, true)", got, ok)
	}

	for _, raw := range []string{`{"type":"foo"}`, `42`, `true`, `null`} {
		got, ok = ExtractMessageText(json.RawMessage(raw))
		if ok {
			t.Errorf("ExtractMessageText(%s): ok=true, want false", raw)
		}
		if got != "" {
			t.Errorf("ExtractMessageText(%s): got=%q, want empty", raw, got)
		}
	}
}

func TestChatCompletionTextSchemaMismatchOnDecodeFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `<html>not json</html>`)
	}))
	t.Cleanup(server.Close)

	_, err := NewOpenAI(OpenAIConfig{BaseURL: server.URL, Model: "gpt"}).
		ChatCompletionText(context.Background(), []Message{{Role: "user", Content: "q"}}, 0, 0)
	var perr *provider.Error
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *provider.Error", err)
	}
	if perr.FailureClass() != provider.FailureClassSchemaMismatch {
		t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), provider.FailureClassSchemaMismatch)
	}
}

func TestChatCompletionTextRetriesOn5xxThenSucceeds(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n := attempts.Add(1)
		if n < 2 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	t.Cleanup(server.Close)

	out, err := NewOpenAI(OpenAIConfig{BaseURL: server.URL, Model: "gpt", MaxRetries: 2}).
		ChatCompletionText(context.Background(), []Message{{Role: "user", Content: "q"}}, 0, 0)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if out != "ok" {
		t.Errorf("out = %q, want ok", out)
	}
	if attempts.Load() != 2 {
		t.Errorf("attempts = %d, want 2", attempts.Load())
	}
}

func TestChatCompletionTextAuthErrorClassified(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = fmt.Fprint(w, `{"error":{"message":"invalid key"}}`)
	}))
	t.Cleanup(server.Close)

	_, err := NewOpenAI(OpenAIConfig{BaseURL: server.URL, Model: "gpt", APIToken: "bad"}).
		ChatCompletionText(context.Background(), []Message{{Role: "user", Content: "q"}}, 0, 0)
	var perr *provider.Error
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *provider.Error", err)
	}
	if perr.FailureClass() != provider.FailureClassAuth {
		t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), provider.FailureClassAuth)
	}
}

func TestChatCompletionTextUnconfiguredRejects(t *testing.T) {
	c := NewOpenAI(OpenAIConfig{})
	_, err := c.ChatCompletionText(context.Background(), nil, 0, 0)
	if err == nil || !strings.Contains(err.Error(), "not configured") {
		t.Errorf("err = %v, want not configured", err)
	}
}

func TestChatCompletionTextPopulatesDiagnosticFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	t.Cleanup(server.Close)

	c := NewOpenAI(OpenAIConfig{
		BaseURL:    server.URL,
		Model:      "gpt-4o-mini",
		MaxRetries: 0,
		Timeout:    5 * time.Second,
	})
	_, err := c.ChatCompletionText(context.Background(),
		[]Message{{Role: "user", Content: "q"}}, 0, 0)
	var perr *provider.Error
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *provider.Error", err)
	}
	fields := perr.DiagnosticFields()
	if fields["model"] != "gpt-4o-mini" {
		t.Errorf("model = %v, want gpt-4o-mini", fields["model"])
	}
	if fields["failure_class"] != provider.FailureClassServer {
		t.Errorf("failure_class = %v, want %v", fields["failure_class"], provider.FailureClassServer)
	}
	if fields["http_status"] != http.StatusBadGateway {
		t.Errorf("http_status = %v, want %d", fields["http_status"], http.StatusBadGateway)
	}
	if fields["input_count"] != 1 {
		t.Errorf("input_count = %v, want 1", fields["input_count"])
	}
}
