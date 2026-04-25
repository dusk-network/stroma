// Package provider is stroma's shared OpenAI-compatible HTTP substrate.
//
// It owns the per-request mechanics common to embed and chat: retry
// with capped exponential backoff, Retry-After parsing, response-size
// bounding, and classification of transport / HTTP / decode failures
// into a public FailureClass taxonomy. Callers (embed.OpenAI,
// chat.OpenAI, and downstream wrappers) consume provider.Do with their
// own request shape and response decoder.
//
// Product-specific labels (Runtime, RequestType, etc.) are intentionally
// out of scope — callers that need those wrap Error on return.
package provider

import (
	"encoding/json"
	"fmt"
	"strings"
)

// FailureClass enumerates the substrate-level failure modes a provider
// call can surface. Callers branch on these to decide whether to retry,
// degrade, or propagate. The string values are stable and safe to log.
const (
	FailureClassAuth                  = "auth"
	FailureClassDependencyUnavailable = "dependency_unavailable"
	FailureClassRateLimit             = "rate_limit"
	FailureClassSchemaMismatch        = "schema_mismatch"
	FailureClassServer                = "server"
	FailureClassTimeout               = "timeout"
	FailureClassTransport             = "transport"
)

// FailureDetails is the substrate-level diagnostic payload attached to
// every Error. It carries fields provider.Do can populate from
// the request and response itself. Callers that need product-layer
// labels (Runtime, Provider, RequestType) wrap the error on return
// rather than adding those fields here.
type FailureDetails struct {
	Model        string
	Endpoint     string
	FailureClass string
	HTTPStatus   int
	TimeoutMS    int
	MaxRetries   int
	BatchSize    int
	InputCount   int
}

// Map returns the details as a JSON-friendly object with empty fields
// omitted so log sinks can emit a minimal diagnostic payload.
func (d FailureDetails) Map() map[string]any {
	values := map[string]any{}
	if value := strings.TrimSpace(d.Model); value != "" {
		values["model"] = value
	}
	if value := strings.TrimSpace(d.Endpoint); value != "" {
		values["endpoint"] = value
	}
	if value := strings.TrimSpace(d.FailureClass); value != "" {
		values["failure_class"] = value
	}
	if d.HTTPStatus > 0 {
		values["http_status"] = d.HTTPStatus
	}
	if d.TimeoutMS > 0 {
		values["timeout_ms"] = d.TimeoutMS
	}
	if d.MaxRetries > 0 {
		values["max_retries"] = d.MaxRetries
	}
	if d.BatchSize > 0 {
		values["batch_size"] = d.BatchSize
	}
	if d.InputCount > 0 {
		values["input_count"] = d.InputCount
	}
	if len(values) == 0 {
		return nil
	}
	return values
}

// Error is the classified-failure type returned by every
// provider.Do path. Callers branch on Details.FailureClass (or use
// errors.As to reach it) to decide retry / degrade / propagate.
type Error struct {
	Message    string
	HTTPStatus int
	Details    *FailureDetails
	cause      error
}

// Error implements the error interface.
func (e *Error) Error() string {
	if e == nil {
		return ""
	}
	return e.Message
}

// Unwrap returns the original lower-level cause, when one exists.
func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

// HTTPStatusCode returns the associated HTTP status when the failure
// came from an HTTP response, or zero otherwise.
func (e *Error) HTTPStatusCode() int {
	if e == nil {
		return 0
	}
	return e.HTTPStatus
}

// FailureClass returns the classified failure mode, or "" when the
// error carries no details.
func (e *Error) FailureClass() string {
	if e == nil || e.Details == nil {
		return ""
	}
	return strings.TrimSpace(e.Details.FailureClass)
}

// DiagnosticFields returns a JSON-friendly diagnostic payload built
// from the attached details, or nil when none are set.
func (e *Error) DiagnosticFields() map[string]any {
	if e == nil || e.Details == nil {
		return nil
	}
	details := *e.Details
	if details.HTTPStatus == 0 {
		details.HTTPStatus = e.HTTPStatus
	}
	return details.Map()
}

// NewError formats a classified failure without HTTP context.
func NewError(details FailureDetails, format string, args ...any) *Error {
	return NewErrorCause(details, nil, format, args...)
}

// NewErrorCause formats a classified failure and preserves a lower-level cause.
func NewErrorCause(details FailureDetails, cause error, format string, args ...any) *Error {
	return &Error{
		Message: fmt.Sprintf(format, args...),
		Details: &details,
		cause:   cause,
	}
}

// NewErrorStatus formats a classified failure and records the
// associated HTTP status code.
func NewErrorStatus(details FailureDetails, status int, format string, args ...any) *Error {
	return NewErrorStatusCause(details, status, nil, format, args...)
}

// NewErrorStatusCause formats a classified HTTP failure and preserves
// a lower-level cause.
func NewErrorStatusCause(details FailureDetails, status int, cause error, format string, args ...any) *Error {
	details.HTTPStatus = status
	return &Error{
		Message:    fmt.Sprintf(format, args...),
		HTTPStatus: status,
		Details:    &details,
		cause:      cause,
	}
}

// ExtractErrorMessage returns a human-readable error message from a
// full OpenAI-compatible response body. It tolerates both
// `{"error":"..."}` and `{"error":{"message":"..."}}` shapes as well as
// a top-level `{"message":"..."}` used by some self-hosted gateways.
func ExtractErrorMessage(body []byte) string {
	var payload struct {
		Error   json.RawMessage `json:"error"`
		Message string          `json:"message"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return ""
	}
	if strings.TrimSpace(payload.Message) != "" {
		return strings.TrimSpace(payload.Message)
	}
	return ExtractErrorValue(payload.Error)
}

// ExtractErrorValue returns a human-readable error message from an
// OpenAI-compatible `error` field that may be either a string or an
// object with message/error/detail keys.
func ExtractErrorValue(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return strings.TrimSpace(text)
	}

	var payload struct {
		Message string `json:"message"`
		Error   string `json:"error"`
		Detail  string `json:"detail"`
	}
	if err := json.Unmarshal(raw, &payload); err == nil {
		switch {
		case strings.TrimSpace(payload.Message) != "":
			return strings.TrimSpace(payload.Message)
		case strings.TrimSpace(payload.Error) != "":
			return strings.TrimSpace(payload.Error)
		case strings.TrimSpace(payload.Detail) != "":
			return strings.TrimSpace(payload.Detail)
		}
	}

	return ""
}
