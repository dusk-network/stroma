package provider

import (
	"context"
	"errors"
	"net"
	"net/http"
	"testing"
)

type fakeNetError struct {
	msg     string
	timeout bool
}

func (e fakeNetError) Error() string   { return e.msg }
func (e fakeNetError) Timeout() bool   { return e.timeout }
func (e fakeNetError) Temporary() bool { return false }

func TestClassifyStatusCode(t *testing.T) {
	cases := map[int]string{
		http.StatusUnauthorized:        FailureClassAuth,
		http.StatusForbidden:           FailureClassAuth,
		http.StatusTooManyRequests:     FailureClassRateLimit,
		http.StatusRequestTimeout:      FailureClassTimeout,
		http.StatusGatewayTimeout:      FailureClassTimeout,
		http.StatusInternalServerError: FailureClassServer,
		http.StatusBadGateway:          FailureClassServer,
		http.StatusServiceUnavailable:  FailureClassServer,
	}
	for status, want := range cases {
		if got := Classify(status, nil, ""); got != want {
			t.Errorf("Classify(%d) = %q, want %q", status, got, want)
		}
	}
}

func TestClassifyMessageFallbacks(t *testing.T) {
	cases := []struct {
		name    string
		message string
		want    string
	}{
		{"auth from message", "missing API key", FailureClassAuth},
		{"rate limit from message", "rate limit reached", FailureClassRateLimit},
		{"timeout from message", "i/o timeout", FailureClassTimeout},
		{"transport from message", "connection refused", FailureClassTransport},
		{"unknown defaults to dependency", "something odd happened", FailureClassDependencyUnavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(0, nil, tc.message); got != tc.want {
				t.Errorf("Classify(0,_,%q) = %q, want %q", tc.message, got, tc.want)
			}
		})
	}
}

func TestClassifyTimeoutFromError(t *testing.T) {
	if got := Classify(0, context.DeadlineExceeded, ""); got != FailureClassTimeout {
		t.Errorf("Classify(_, DeadlineExceeded, _) = %q, want %q", got, FailureClassTimeout)
	}

	netErr := fakeNetError{msg: "dial tcp: i/o timeout", timeout: true}
	if got := Classify(0, netErr, ""); got != FailureClassTimeout {
		t.Errorf("Classify(_, net.Error{timeout:true}, _) = %q, want %q", got, FailureClassTimeout)
	}
}

func TestClassifyTransportFromError(t *testing.T) {
	netErr := fakeNetError{msg: "dial tcp: refused", timeout: false}
	if got := Classify(0, netErr, ""); got != FailureClassTransport {
		t.Errorf("Classify(_, net.Error, _) = %q, want %q", got, FailureClassTransport)
	}
}

func TestClassifyStatusWinsOverMessage(t *testing.T) {
	// A 401 that happens to mention "rate limit" is still classified as
	// auth — HTTP status is the stronger signal.
	if got := Classify(http.StatusUnauthorized, nil, "rate limit hint"); got != FailureClassAuth {
		t.Errorf("Classify(401, _, %q) = %q, want %q", "rate limit hint", got, FailureClassAuth)
	}
}

// Sanity check that net.OpError satisfies the net.Error interface path we
// rely on in isTransportFailure.
func TestClassifyRecognizesNetOpError(t *testing.T) {
	opErr := &net.OpError{Op: "dial", Err: errors.New("refused")}
	if got := Classify(0, opErr, ""); got != FailureClassTransport {
		t.Errorf("Classify(_, *net.OpError, _) = %q, want %q", got, FailureClassTransport)
	}
}

// Guard against future churn on the list literal.
func TestClassifyEOFFallback(t *testing.T) {
	if got := Classify(0, nil, "eof"); got != FailureClassTransport {
		t.Errorf("Classify(_, _, eof) = %q, want %q", got, FailureClassTransport)
	}
}

// Ensure classifiedFailureDetails populates both HTTPStatus and FailureClass.
func TestClassifiedFailureDetailsPopulates(t *testing.T) {
	base := FailureDetails{Model: "m", Endpoint: "http://x"}
	got := classifiedFailureDetails(base, 503, nil, "")
	if got.HTTPStatus != 503 {
		t.Errorf("HTTPStatus = %d, want 503", got.HTTPStatus)
	}
	if got.FailureClass != FailureClassServer {
		t.Errorf("FailureClass = %q, want %q", got.FailureClass, FailureClassServer)
	}
	// Base fields preserved.
	if got.Model != "m" || got.Endpoint != "http://x" {
		t.Errorf("base fields not preserved: %+v", got)
	}
}

// Compile-time check that fakeNetError satisfies net.Error — the test
// relies on errors.As landing there.
var _ net.Error = fakeNetError{}
