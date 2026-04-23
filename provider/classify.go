package provider

import (
	"context"
	"errors"
	"net"
	"net/http"
	"strings"
)

// Classify returns the FailureClass best describing the given HTTP
// status, Go error, and upstream error message. It is called by
// provider.Do to label every failure the substrate surfaces; callers
// may also call it directly when synthesising a FailureDetails for a
// caller-side error path.
//
// Precedence: HTTP status mapping wins first (auth / rate_limit /
// timeout / server); otherwise the Go error and message are inspected
// for auth / rate-limit / timeout / transport hints, defaulting to
// dependency_unavailable.
func Classify(statusCode int, err error, message string) string {
	switch statusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		return FailureClassAuth
	case http.StatusTooManyRequests:
		return FailureClassRateLimit
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		return FailureClassTimeout
	}
	if statusCode >= http.StatusInternalServerError {
		return FailureClassServer
	}

	lower := strings.ToLower(strings.TrimSpace(message))
	switch {
	case mentionsAuthFailure(lower):
		return FailureClassAuth
	case mentionsRateLimit(lower):
		return FailureClassRateLimit
	case isTimeoutFailure(err, lower):
		return FailureClassTimeout
	case isTransportFailure(err, lower):
		return FailureClassTransport
	default:
		return FailureClassDependencyUnavailable
	}
}

// classifiedFailureDetails returns a copy of details with HTTPStatus
// and FailureClass populated from the request outcome.
func classifiedFailureDetails(details FailureDetails, statusCode int, err error, message string) FailureDetails {
	details.HTTPStatus = statusCode
	details.FailureClass = Classify(statusCode, err, message)
	return details
}

func mentionsAuthFailure(message string) bool {
	return strings.Contains(message, "api key") ||
		strings.Contains(message, "apikey") ||
		strings.Contains(message, "auth") ||
		strings.Contains(message, "unauthorized") ||
		strings.Contains(message, "forbidden")
}

func mentionsRateLimit(message string) bool {
	return strings.Contains(message, "rate limit") ||
		strings.Contains(message, "too many requests") ||
		strings.Contains(message, "quota exceeded")
}

func isTimeoutFailure(err error, message string) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	return strings.Contains(message, "context deadline exceeded") ||
		strings.Contains(message, "client.timeout exceeded") ||
		strings.Contains(message, "timeout awaiting response headers") ||
		strings.Contains(message, "i/o timeout") ||
		strings.Contains(message, "timed out")
}

func isTransportFailure(err error, message string) bool {
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	return strings.Contains(message, "connection refused") ||
		strings.Contains(message, "connection reset") ||
		strings.Contains(message, "broken pipe") ||
		strings.Contains(message, "couldn't connect to server") ||
		strings.Contains(message, "failed to connect") ||
		strings.Contains(message, "no such host") ||
		strings.Contains(message, "network is unreachable") ||
		strings.Contains(message, "unexpected eof") ||
		message == "eof"
}
