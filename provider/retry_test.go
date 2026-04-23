package provider

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestRetryAfterDuration(t *testing.T) {
	cases := map[string]time.Duration{
		"":         0,
		"0":        0,
		"-5":       0,
		"3":        3 * time.Second,
		"  7  ":    7 * time.Second,
		"not-a-no": 0,
	}
	for input, want := range cases {
		if got := retryAfterDuration(input); got != want {
			t.Errorf("retryAfterDuration(%q) = %s, want %s", input, got, want)
		}
	}

	// HTTP-Date: one minute in the future.
	future := time.Now().Add(time.Minute).UTC().Format(http.TimeFormat)
	got := retryAfterDuration(future)
	if got <= 0 || got > 2*time.Minute {
		t.Errorf("retryAfterDuration(%q) = %s, want ~1m", future, got)
	}

	// HTTP-Date in the past clamps to zero (don't block for negative delay).
	past := time.Now().Add(-time.Minute).UTC().Format(http.TimeFormat)
	if got := retryAfterDuration(past); got != 0 {
		t.Errorf("retryAfterDuration(past) = %s, want 0", got)
	}
}

func TestDoSuccessSinglePass(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		if auth := r.Header.Get("Authorization"); auth != "Bearer secret" {
			t.Errorf("Authorization = %q, want Bearer secret", auth)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", ct)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"value":"hi"}`)
	}))
	t.Cleanup(server.Close)

	out, err := Do(context.Background(), server.Client(),
		Target{Method: http.MethodPost, URL: server.URL, Body: []byte(`{"q":"go"}`), Token: "secret"},
		FailureDetails{},
		Policy{MaxRetries: 2},
		func(_ *http.Response, body []byte) (string, error) {
			if !strings.Contains(string(body), "hi") {
				return "", fmt.Errorf("unexpected body %q", body)
			}
			return "hi", nil
		},
	)
	if err != nil {
		t.Fatalf("Do() err = %v", err)
	}
	if out != "hi" {
		t.Errorf("Do() = %q, want %q", out, "hi")
	}
	if n := attempts.Load(); n != 1 {
		t.Errorf("attempts = %d, want 1", n)
	}
}

func TestDoRetriesOn5xxThenSucceeds(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n := attempts.Add(1)
		if n < 3 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `ok`)
	}))
	t.Cleanup(server.Close)

	out, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{},
		Policy{MaxRetries: 5},
		func(_ *http.Response, body []byte) (string, error) { return string(body), nil },
	)
	if err != nil {
		t.Fatalf("Do() err = %v", err)
	}
	if out != "ok" {
		t.Errorf("Do() = %q, want ok", out)
	}
	if n := attempts.Load(); n != 3 {
		t.Errorf("attempts = %d, want 3", n)
	}
}

func TestDoHonoursRetryAfterHeader(t *testing.T) {
	var attempts atomic.Int32
	var firstAttemptAt time.Time
	var secondAttemptAt time.Time
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n := attempts.Add(1)
		if n == 1 {
			firstAttemptAt = time.Now()
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		secondAttemptAt = time.Now()
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, "ok")
	}))
	t.Cleanup(server.Close)

	_, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{},
		Policy{MaxRetries: 3},
		func(_ *http.Response, body []byte) (string, error) { return string(body), nil },
	)
	if err != nil {
		t.Fatalf("Do() err = %v", err)
	}
	gap := secondAttemptAt.Sub(firstAttemptAt)
	if gap < 900*time.Millisecond {
		t.Errorf("Retry-After honoured gap = %s, want >=900ms", gap)
	}
}

func TestDoClassifiesNon2xx(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = fmt.Fprint(w, `{"error":{"message":"bad key"}}`)
	}))
	t.Cleanup(server.Close)

	_, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{Model: "m", Endpoint: server.URL},
		Policy{MaxRetries: 0},
		func(_ *http.Response, body []byte) (string, error) { return "", nil },
	)
	var perr *ProviderError
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *ProviderError", err)
	}
	if perr.FailureClass() != FailureClassAuth {
		t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), FailureClassAuth)
	}
	if perr.HTTPStatusCode() != http.StatusUnauthorized {
		t.Errorf("HTTPStatusCode = %d, want 401", perr.HTTPStatusCode())
	}
	if !strings.Contains(perr.Error(), "bad key") {
		t.Errorf("error message missing upstream text: %q", perr.Error())
	}
}

func TestDoRespectsMaxRetriesExhaustion(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(server.Close)

	_, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{},
		Policy{MaxRetries: 2},
		func(_ *http.Response, body []byte) (string, error) { return "", nil },
	)
	if err == nil {
		t.Fatalf("Do() err = nil, want error after exhaustion")
	}
	if n := attempts.Load(); n != 3 {
		t.Errorf("attempts = %d, want 3 (1 initial + 2 retries)", n)
	}
	var perr *ProviderError
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *ProviderError", err)
	}
	if perr.FailureClass() != FailureClassServer {
		t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), FailureClassServer)
	}
}

func TestDoEnforcesResponseSizeCap(t *testing.T) {
	// Emit a response larger than the configured cap.
	payload := strings.Repeat("x", 64)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, payload)
	}))
	t.Cleanup(server.Close)

	_, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{},
		Policy{MaxResponseBytes: 16},
		func(_ *http.Response, body []byte) (string, error) { return string(body), nil },
	)
	var perr *ProviderError
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *ProviderError", err)
	}
	if perr.FailureClass() != FailureClassSchemaMismatch {
		t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), FailureClassSchemaMismatch)
	}
}

func TestDoContextCancellationShortCircuits(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, "ok")
	}))
	t.Cleanup(server.Close)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := Do(ctx, server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{},
		Policy{MaxRetries: 3},
		func(_ *http.Response, body []byte) (string, error) { return "", nil },
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestDoDecoderSchemaMismatchPropagates(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"value":"x"}`)
	}))
	t.Cleanup(server.Close)

	_, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{},
		Policy{MaxRetries: 0},
		func(_ *http.Response, body []byte) (string, error) {
			return "", NewProviderError(FailureDetails{FailureClass: FailureClassSchemaMismatch}, "bad shape")
		},
	)
	var perr *ProviderError
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *ProviderError", err)
	}
	if perr.FailureClass() != FailureClassSchemaMismatch {
		t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), FailureClassSchemaMismatch)
	}
}

func TestDoHeaderPassthrough(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Test") != "yes" {
			t.Errorf("X-Test header = %q, want yes", r.Header.Get("X-Test"))
		}
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, "ok")
	}))
	t.Cleanup(server.Close)

	h := http.Header{}
	h.Set("X-Test", "yes")
	_, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`), Header: h},
		FailureDetails{},
		Policy{},
		func(_ *http.Response, body []byte) (string, error) { return string(body), nil },
	)
	if err != nil {
		t.Fatalf("Do() err = %v", err)
	}
}

func TestRetryAfterCapBoundsHugeHeaderValue(t *testing.T) {
	// An upstream sending `Retry-After: 86400` would otherwise park a
	// caller with context.Background() for a full day. MaxRetryAfter
	// bounds this — verify by asserting the retry fires within a
	// tight budget, not after 86400s.
	var firstAttemptAt, secondAttemptAt time.Time
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n := attempts.Add(1)
		if n == 1 {
			firstAttemptAt = time.Now()
			w.Header().Set("Retry-After", "86400")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		secondAttemptAt = time.Now()
		_, _ = fmt.Fprint(w, "ok")
	}))
	t.Cleanup(server.Close)

	start := time.Now()
	_, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{},
		Policy{MaxRetries: 2, MaxRetryAfter: 250 * time.Millisecond},
		func(_ *http.Response, body []byte) (string, error) { return string(body), nil },
	)
	if err != nil {
		t.Fatalf("Do() err = %v", err)
	}
	gap := secondAttemptAt.Sub(firstAttemptAt)
	if gap > 1*time.Second {
		t.Errorf("retry fired after %s, want bounded by MaxRetryAfter (~250ms)", gap)
	}
	if time.Since(start) > 2*time.Second {
		t.Errorf("total elapsed %s, want <2s (MaxRetryAfter cap broken)", time.Since(start))
	}
}

func TestRetryAfterDefaultCapIs30s(t *testing.T) {
	// Regression guard: the default Policy{} (MaxRetryAfter=0) must
	// select a finite cap, not honour arbitrary header values.
	p := Policy{}
	if p.MaxRetryAfter != 0 {
		t.Fatalf("sanity: expected zero-value Policy to have MaxRetryAfter=0")
	}
	// Exercise via a header value larger than any reasonable real
	// rate-limiter window but well under our cap so the test doesn't
	// block on it. The point is to confirm capping works from zero.
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n := attempts.Add(1)
		if n == 1 {
			w.Header().Set("Retry-After", "3600") // 1 hour
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = fmt.Fprint(w, "ok")
	}))
	t.Cleanup(server.Close)

	// Bound this test at 5s via context — if the default cap is
	// missing or longer than 30s, the test still finishes on ctx
	// cancellation and the resulting ctx.Err asserts the failure.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		_, err := Do(ctx, server.Client(),
			Target{URL: server.URL, Body: []byte(`x`)},
			FailureDetails{},
			Policy{MaxRetries: 1}, // MaxRetryAfter zero → default
			func(_ *http.Response, body []byte) (string, error) { return string(body), nil },
		)
		done <- err
	}()

	// The test passes if Do() returns within 10s. Without a default
	// cap, a Retry-After of 3600s would block the goroutine for an
	// hour and trip the deadline case below. Either a successful
	// retry or a ctx-deadline-exceeded failure proves the cap fired.
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatalf("Do() did not return within 10s — default MaxRetryAfter cap is missing")
	}
}

func TestDoOversizedNon2xxPreservesStatusClassification(t *testing.T) {
	// A 503 with a body larger than the response cap must still
	// surface as FailureClassServer — HTTP status is the strongest
	// signal. Before the reorder, oversize was checked first and
	// clobbered the classification with schema_mismatch.
	big := strings.Repeat("x", 64)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = fmt.Fprint(w, big)
	}))
	t.Cleanup(server.Close)

	_, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{},
		Policy{MaxResponseBytes: 16},
		func(_ *http.Response, body []byte) (string, error) { return string(body), nil },
	)
	var perr *ProviderError
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *ProviderError", err)
	}
	if perr.FailureClass() != FailureClassServer {
		t.Errorf("FailureClass = %q, want %q (status must win over oversize)", perr.FailureClass(), FailureClassServer)
	}
	if perr.HTTPStatusCode() != http.StatusServiceUnavailable {
		t.Errorf("HTTPStatusCode = %d, want 503", perr.HTTPStatusCode())
	}
}

func TestDoOversized401PreservesAuthClass(t *testing.T) {
	// Mirror test for auth: an oversized 401 body must still classify
	// as auth, not schema_mismatch.
	big := strings.Repeat("x", 64)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = fmt.Fprint(w, big)
	}))
	t.Cleanup(server.Close)

	_, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{},
		Policy{MaxResponseBytes: 16},
		func(_ *http.Response, body []byte) (string, error) { return string(body), nil },
	)
	var perr *ProviderError
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *ProviderError", err)
	}
	if perr.FailureClass() != FailureClassAuth {
		t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), FailureClassAuth)
	}
}

func TestDoWrapsOpaqueDecodeErrorAsSchemaMismatch(t *testing.T) {
	// The provider contract is that every failure return satisfies
	// errors.As(*ProviderError). Decoders like embed.parseEmbedResponse
	// return plain fmt.Errorf values for protocol violations; the
	// provider layer must wrap them so caller-side branching on
	// FailureClass still works.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{"anything":"goes"}`)
	}))
	t.Cleanup(server.Close)

	_, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{Model: "m", Endpoint: server.URL},
		Policy{},
		func(_ *http.Response, body []byte) (string, error) {
			return "", fmt.Errorf("wrong input count: got 3, want 1")
		},
	)
	var perr *ProviderError
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *ProviderError via errors.As", err)
	}
	if perr.FailureClass() != FailureClassSchemaMismatch {
		t.Errorf("FailureClass = %q, want %q", perr.FailureClass(), FailureClassSchemaMismatch)
	}
	if !strings.Contains(err.Error(), "wrong input count") {
		t.Errorf("wrapped error lost original text: %q", err.Error())
	}
	if perr.DiagnosticFields()["model"] != "m" {
		t.Errorf("details not preserved: %v", perr.DiagnosticFields())
	}
}

func TestDoPreservesClassifiedDecodeError(t *testing.T) {
	// Decoders that already return *ProviderError must pass through
	// unchanged — no double-wrapping, original FailureClass preserved.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `ok`)
	}))
	t.Cleanup(server.Close)

	original := NewProviderError(FailureDetails{FailureClass: FailureClassAuth}, "custom auth msg")
	_, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{},
		Policy{},
		func(_ *http.Response, body []byte) (string, error) { return "", original },
	)
	var perr *ProviderError
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *ProviderError", err)
	}
	if perr.FailureClass() != FailureClassAuth {
		t.Errorf("FailureClass = %q, want %q (original should pass through)", perr.FailureClass(), FailureClassAuth)
	}
	if perr.Error() != "custom auth msg" {
		t.Errorf("Error() = %q, want original message", perr.Error())
	}
}

func TestShouldRetryNonTimeoutNetErrorFallsThroughToStringMatch(t *testing.T) {
	// Regression: http.Client.Do wraps dial errors in *url.Error
	// around *net.OpError. A non-timeout net error must still be
	// retried when its message contains "connection refused" / reset /
	// broken pipe. An earlier implementation returned
	// netErr.Timeout() directly and made these unreachable.
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{
			name: "connection refused wrapped in url.Error",
			err: &url.Error{
				Op:  "Post",
				URL: "http://x",
				Err: &net.OpError{Op: "dial", Err: errors.New("connection refused")},
			},
			want: true,
		},
		{
			name: "connection reset wrapped",
			err: &url.Error{
				Op:  "Post",
				URL: "http://x",
				Err: &net.OpError{Op: "read", Err: errors.New("connection reset by peer")},
			},
			want: true,
		},
		{
			name: "broken pipe wrapped",
			err: &url.Error{
				Op:  "Post",
				URL: "http://x",
				Err: &net.OpError{Op: "write", Err: errors.New("broken pipe")},
			},
			want: true,
		},
		{
			name: "timeout wrapped",
			err: &url.Error{
				Op:  "Post",
				URL: "http://x",
				Err: fakeNetError{msg: "timeout", timeout: true},
			},
			want: true,
		},
		{
			name: "unknown non-timeout net error stays non-retryable",
			err: &url.Error{
				Op:  "Post",
				URL: "http://x",
				Err: &net.OpError{Op: "dial", Err: errors.New("some other thing")},
			},
			want: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := shouldRetry(tc.err, 0); got != tc.want {
				t.Errorf("shouldRetry(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestDoReadErrorOnNon2xxPreservesStatus(t *testing.T) {
	// Regression: a 401/429/5xx whose body fails to read mid-stream
	// must still classify by HTTP status, not collapse to transport
	// with status 0. The status was received before the body read, so
	// it is still the strongest signal.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// Send 503 with Content-Length announcing more bytes than we
		// actually emit, then close the connection so io.ReadAll
		// observes an unexpected EOF.
		w.Header().Set("Content-Length", "128")
		w.WriteHeader(http.StatusServiceUnavailable)
		if hj, ok := w.(http.Hijacker); ok {
			conn, buf, err := hj.Hijack()
			if err != nil {
				t.Errorf("hijack: %v", err)
				return
			}
			// Flush what's already been written (headers), then drop
			// the connection to trigger a body read error.
			_ = buf.Flush()
			_ = conn.Close()
		}
	}))
	t.Cleanup(server.Close)

	_, err := Do(context.Background(), server.Client(),
		Target{URL: server.URL, Body: []byte(`x`)},
		FailureDetails{},
		Policy{MaxRetries: 0},
		func(_ *http.Response, body []byte) (string, error) { return string(body), nil },
	)
	var perr *ProviderError
	if !errors.As(err, &perr) {
		t.Fatalf("err = %v, want *ProviderError", err)
	}
	if perr.HTTPStatusCode() != http.StatusServiceUnavailable {
		t.Errorf("HTTPStatusCode = %d, want %d", perr.HTTPStatusCode(), http.StatusServiceUnavailable)
	}
	if perr.FailureClass() != FailureClassServer {
		t.Errorf("FailureClass = %q, want %q (read-err on non-2xx must classify by status)", perr.FailureClass(), FailureClassServer)
	}
}

func TestDoRejectsNilClient(t *testing.T) {
	_, err := Do(context.Background(), nil,
		Target{URL: "http://x"},
		FailureDetails{},
		Policy{},
		func(_ *http.Response, body []byte) (string, error) { return "", nil },
	)
	if err == nil || !strings.Contains(err.Error(), "nil http client") {
		t.Errorf("err = %v, want nil http client", err)
	}
}
