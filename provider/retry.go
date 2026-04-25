package provider

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Target describes an idempotent request that provider.Do may resend
// on retry. Body is stored as a byte slice (rather than an io.Reader)
// so the transport can rebuild a fresh http.Request per attempt.
//
// An empty Token skips the Authorization header. Header is optional
// and merged after the core sets Content-Type and Authorization, so
// callers can set e.g. Accept without clobbering either.
type Target struct {
	Method string
	URL    string
	Body   []byte
	Token  string
	Header http.Header
}

// Policy controls retry and response-size behaviour. A zero Policy is
// valid: MaxRetries 0 means "try once, no retries", negative
// MaxRetries values normalize to zero, and a zero MaxResponseBytes
// selects defaultMaxResponseBytes (4 MiB) — generous for chat
// completions, tight enough to avoid OOMing the host on a
// misconfigured upstream. Embedders with larger expected payloads set
// MaxResponseBytes explicitly.
type Policy struct {
	MaxRetries       int
	MaxResponseBytes int64

	// MaxRetryAfter bounds how long waitBeforeRetry will honour a
	// server-supplied Retry-After header. A hostile or misconfigured
	// upstream sending `Retry-After: 86400` (or a far-future HTTP
	// date) would otherwise park the goroutine for that entire
	// duration when the caller's context has no deadline. A zero or
	// negative value selects defaultMaxRetryAfter (30s) — generous for
	// real rate limits, tight enough that a pathological header does
	// not silently consume hours of wall time. Callers with a larger
	// budget set this explicitly.
	MaxRetryAfter time.Duration
}

// DecodeFunc parses a successful 2xx response body into T. Decoders
// may return a *Error directly (e.g. for schema_mismatch) to
// signal a caller-classified failure; other errors are treated as
// opaque and wrapped as dependency_unavailable.
type DecodeFunc[T any] func(resp *http.Response, body []byte) (T, error)

const (
	// defaultMaxResponseBytes bounds the response body when Policy
	// leaves MaxResponseBytes at zero. 4 MiB fits every OpenAI-compat
	// chat / small-batch payload we've observed and keeps the host
	// safe against a hostile or misconfigured upstream that streams
	// GiBs. Embedding batch responses are larger by design; embed
	// configures its own higher cap.
	defaultMaxResponseBytes = 4 << 20

	// backoffInitialDelay is the starting backoff when Retry-After is
	// absent. Doubled per attempt and capped at backoffMaxDelay.
	backoffInitialDelay = 200 * time.Millisecond
	backoffMaxDelay     = 2 * time.Second

	// defaultMaxRetryAfter bounds a server-supplied Retry-After header
	// when Policy leaves MaxRetryAfter unset. 30s covers real rate
	// limiter windows while preventing a hostile or misconfigured
	// upstream from parking a goroutine for hours via a huge Retry-After.
	defaultMaxRetryAfter = 30 * time.Second
)

// Do sends target, applies retry + Retry-After behaviour per policy,
// bounds the response body, classifies non-2xx responses and transport
// failures, and hands the raw body to decode on success. details is
// the baseline FailureDetails used to enrich every returned
// *Error; HTTPStatus and FailureClass are populated by Do.
//
// Retries fire on: 429 Too Many Requests, any 5xx, connection
// refused / reset / broken pipe, and net.Error.Timeout(). Retry-After
// (integer seconds or HTTP-Date) is honoured when present; otherwise
// the delay is exponential 200ms → 400ms → ... → 2s cap. Context
// cancellation or deadline exceeded aborts immediately without retry.
func Do[T any](
	ctx context.Context,
	client *http.Client,
	target Target,
	details FailureDetails,
	policy Policy,
	decode DecodeFunc[T],
) (T, error) {
	var zero T
	if client == nil {
		return zero, fmt.Errorf("stroma/provider: nil http client")
	}
	if decode == nil {
		return zero, fmt.Errorf("stroma/provider: nil decode function")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	maxResponseBytes := policy.MaxResponseBytes
	if maxResponseBytes <= 0 {
		maxResponseBytes = defaultMaxResponseBytes
	}

	maxRetryAfter := policy.MaxRetryAfter
	if maxRetryAfter <= 0 {
		maxRetryAfter = defaultMaxRetryAfter
	}

	maxRetries := policy.MaxRetries
	if maxRetries < 0 {
		maxRetries = 0
	}
	if details.MaxRetries < 0 {
		details.MaxRetries = 0
	}

	method := target.Method
	if method == "" {
		method = http.MethodPost
	}

	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		value, retryAfter, statusCode, err := doAttempt(ctx, client, method, target, details, maxResponseBytes, decode)
		if err == nil {
			return value, nil
		}
		// Context cancellation propagates immediately — retrying a
		// cancelled caller would waste cycles and surprise them.
		if errors.Is(err, context.Canceled) ||
			(errors.Is(err, context.DeadlineExceeded) && ctx.Err() != nil) {
			return zero, err
		}
		lastErr = err
		if shouldRetry(err, statusCode) && attempt < maxRetries {
			if waitErr := waitBeforeRetry(ctx, attempt, min(retryAfter, maxRetryAfter)); waitErr != nil {
				return zero, waitErr
			}
			continue
		}
		return zero, err
	}

	if lastErr == nil {
		lastErr = NewError(details, "request failed")
	}
	return zero, lastErr
}

// doAttempt performs one HTTP attempt and classifies the outcome.
// On success it returns (value, 0, statusCode, nil). On failure it
// returns (_, retryAfter, statusCode, *Error) — or the raw
// ctx error when the context was cancelled, which Do propagates
// without retry.
func doAttempt[T any](
	ctx context.Context,
	client *http.Client,
	method string,
	target Target,
	details FailureDetails,
	maxResponseBytes int64,
	decode DecodeFunc[T],
) (value T, retryAfter time.Duration, statusCode int, err error) {
	req, err := buildRequest(ctx, method, target)
	if err != nil {
		return value, 0, 0, err
	}

	resp, err := client.Do(req)
	if err != nil {
		if errors.Is(err, context.Canceled) ||
			(errors.Is(err, context.DeadlineExceeded) && ctx.Err() != nil) {
			return value, 0, 0, err
		}
		failure := classifiedFailureDetails(details, 0, err, err.Error())
		return value, 0, 0, NewErrorCause(failure, err, "call %s %s: %v", method, target.URL, err)
	}

	retryAfter = retryAfterDuration(resp.Header.Get("Retry-After"))
	body, readErr := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes+1))
	closeErr := resp.Body.Close()

	value, err = interpretResponse(resp, method, target, body, readErr, closeErr, details, maxResponseBytes, decode)
	return value, retryAfter, resp.StatusCode, err
}

// buildRequest assembles the http.Request including headers so
// doAttempt stays focused on transport and classification.
func buildRequest(ctx context.Context, method string, target Target) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, target.URL, bytes.NewReader(target.Body))
	if err != nil {
		return nil, fmt.Errorf("stroma/provider: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if target.Token != "" {
		req.Header.Set("Authorization", "Bearer "+target.Token)
	}
	for key, values := range target.Header {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}
	return req, nil
}

// interpretResponse classifies a received HTTP response into a
// decoded value or a *Error. HTTP-status classification wins
// over body-read and oversize concerns so the strongest signal
// (auth / rate_limit / server) survives truncation or oversized error
// payloads.
func interpretResponse[T any](
	resp *http.Response,
	method string,
	target Target,
	body []byte,
	readErr error,
	closeErr error,
	details FailureDetails,
	maxResponseBytes int64,
	decode DecodeFunc[T],
) (T, error) {
	var zero T
	oversized := int64(len(body)) > maxResponseBytes
	non2xx := resp.StatusCode < 200 || resp.StatusCode >= 300

	if non2xx {
		return zero, interpretNon2xx(resp, method, target, body, readErr, oversized, details, maxResponseBytes)
	}
	if readErr != nil {
		failure := classifiedFailureDetails(details, 0, readErr, readErr.Error())
		return zero, NewErrorCause(failure, readErr, "read response: %v", readErr)
	}
	if oversized {
		failure := details
		failure.FailureClass = FailureClassSchemaMismatch
		return zero, NewError(failure, "response exceeds %d bytes; aborting decode", maxResponseBytes)
	}

	value, decodeErr := decode(resp, body)
	if decodeErr == nil {
		if closeErr != nil {
			failure := classifiedFailureDetails(details, 0, closeErr, closeErr.Error())
			return zero, NewErrorCause(failure, closeErr, "close response: %v", closeErr)
		}
		return value, nil
	}
	// Decoders that already return a classified *Error pass
	// through; opaque errors are wrapped as schema_mismatch so every
	// Do return satisfies errors.As(*Error).
	var perr *Error
	if !errors.As(decodeErr, &perr) {
		failure := details
		failure.FailureClass = FailureClassSchemaMismatch
		decodeErr = NewErrorCause(failure, decodeErr, "decode response: %v", decodeErr)
	}
	return zero, decodeErr
}

// interpretNon2xx handles any non-2xx response: classify by status
// even when the body read failed or the body exceeded the cap.
func interpretNon2xx(
	resp *http.Response,
	method string,
	target Target,
	body []byte,
	readErr error,
	oversized bool,
	details FailureDetails,
	maxResponseBytes int64,
) error {
	if readErr != nil {
		failure := classifiedFailureDetails(details, resp.StatusCode, nil, readErr.Error())
		return NewErrorStatusCause(failure, resp.StatusCode, readErr,
			"%s %s returned %s (response body read failed: %v)",
			method, target.URL, resp.Status, readErr)
	}
	message := ExtractErrorMessage(body)
	if message == "" {
		message = strings.TrimSpace(string(body))
	}
	if message == "" {
		message = http.StatusText(resp.StatusCode)
	}
	if oversized {
		message = fmt.Sprintf("%s (response body exceeds %d bytes; truncated)", http.StatusText(resp.StatusCode), maxResponseBytes)
	}
	failure := classifiedFailureDetails(details, resp.StatusCode, nil, message)
	return NewErrorStatus(failure, resp.StatusCode,
		"%s %s returned %s: %s", method, target.URL, resp.Status, message)
}

func shouldRetry(err error, statusCode int) bool {
	if statusCode == http.StatusTooManyRequests || statusCode >= 500 {
		return true
	}

	if err == nil {
		return false
	}

	// http.Client.Do wraps dial/read failures in *url.Error around a
	// *net.OpError — both satisfy net.Error. Timeouts are retryable
	// directly; non-timeout net errors still need to fall through to
	// message-based matching so connection-refused / reset / broken-pipe
	// paths stay retryable (returning immediately on the Timeout check
	// above was the regression Codex flagged).
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}

	lower := strings.ToLower(err.Error())
	return strings.Contains(lower, "connection refused") ||
		strings.Contains(lower, "connection reset") ||
		strings.Contains(lower, "broken pipe")
}

// retryAfterDuration parses a Retry-After header value as either an
// integer number of seconds or an HTTP-Date. It returns the derived
// delay, clamped to a positive duration; invalid or past values return
// zero so the caller falls back to exponential backoff.
func retryAfterDuration(value string) time.Duration {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0
	}
	if seconds, err := strconv.Atoi(value); err == nil {
		if seconds > 0 {
			return time.Duration(seconds) * time.Second
		}
		return 0
	}
	if when, err := http.ParseTime(value); err == nil {
		if delay := time.Until(when); delay > 0 {
			return delay
		}
	}
	return 0
}

// waitBeforeRetry blocks until retryAfter elapses, or — when
// retryAfter is zero — until an exponential backoff interval elapses.
// Returns early with ctx.Err() if the context is cancelled.
func waitBeforeRetry(ctx context.Context, attempt int, retryAfter time.Duration) error {
	delay := retryAfter
	if delay <= 0 {
		delay = backoffInitialDelay
		for range attempt {
			delay *= 2
			if delay >= backoffMaxDelay {
				delay = backoffMaxDelay
				break
			}
		}
	}

	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
