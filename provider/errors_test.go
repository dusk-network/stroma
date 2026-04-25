package provider

import (
	"encoding/json"
	"errors"
	"net"
	"testing"
)

func TestFailureDetailsMapOmitsEmpty(t *testing.T) {
	empty := FailureDetails{}
	if got := empty.Map(); got != nil {
		t.Fatalf("Map() for empty details = %v, want nil", got)
	}

	populated := FailureDetails{
		Model:        "gpt",
		Endpoint:     "http://x/v1",
		FailureClass: FailureClassTimeout,
		HTTPStatus:   504,
		TimeoutMS:    250,
		MaxRetries:   3,
		BatchSize:    32,
		InputCount:   8,
	}
	got := populated.Map()
	want := map[string]any{
		"model":         "gpt",
		"endpoint":      "http://x/v1",
		"failure_class": FailureClassTimeout,
		"http_status":   504,
		"timeout_ms":    250,
		"max_retries":   3,
		"batch_size":    32,
		"input_count":   8,
	}
	for key, value := range want {
		if got[key] != value {
			t.Errorf("Map()[%q] = %v, want %v", key, got[key], value)
		}
	}
	if len(got) != len(want) {
		t.Errorf("Map() extra keys: got %v, want %v", got, want)
	}
}

func TestProviderErrorIsStandardError(t *testing.T) {
	err := NewError(FailureDetails{FailureClass: FailureClassAuth}, "missing key for %s", "openai")
	if err.Error() != "missing key for openai" {
		t.Errorf("Error() = %q", err.Error())
	}
	if err.FailureClass() != FailureClassAuth {
		t.Errorf("FailureClass() = %q, want %q", err.FailureClass(), FailureClassAuth)
	}

	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("errors.As(Error) = false")
	}
}

func TestProviderErrorUnwrapsCause(t *testing.T) {
	cause := fakeNetError{msg: "temporary upstream stall", timeout: true}
	err := NewErrorCause(FailureDetails{FailureClass: FailureClassTimeout}, cause, "call failed: %v", cause)

	if !errors.Is(err, cause) {
		t.Fatalf("errors.Is(err, cause) = false")
	}
	var netErr net.Error
	if !errors.As(err, &netErr) {
		t.Fatalf("errors.As(err, net.Error) = false")
	}
	if !netErr.Timeout() {
		t.Fatalf("netErr.Timeout() = false, want true")
	}
}

func TestProviderErrorStatusRoundtripsHTTPStatus(t *testing.T) {
	err := NewErrorStatus(FailureDetails{}, 429, "rate limited")
	if err.HTTPStatusCode() != 429 {
		t.Errorf("HTTPStatusCode() = %d, want 429", err.HTTPStatusCode())
	}
	fields := err.DiagnosticFields()
	if fields["http_status"] != 429 {
		t.Errorf("DiagnosticFields()[http_status] = %v, want 429", fields["http_status"])
	}
}

func TestExtractErrorMessage(t *testing.T) {
	for name, tc := range map[string]struct {
		body string
		want string
	}{
		"top-level message": {
			body: `{"message":"bad"}`,
			want: "bad",
		},
		"error as string": {
			body: `{"error":"rate limit reached"}`,
			want: "rate limit reached",
		},
		"error object with message": {
			body: `{"error":{"message":"invalid api key"}}`,
			want: "invalid api key",
		},
		"error object with detail": {
			body: `{"error":{"detail":"upstream down"}}`,
			want: "upstream down",
		},
		"empty": {
			body: `{}`,
			want: "",
		},
		"garbage": {
			body: `not json`,
			want: "",
		},
	} {
		t.Run(name, func(t *testing.T) {
			if got := ExtractErrorMessage([]byte(tc.body)); got != tc.want {
				t.Errorf("ExtractErrorMessage(%q) = %q, want %q", tc.body, got, tc.want)
			}
		})
	}
}

func TestExtractErrorValueNilRaw(t *testing.T) {
	if got := ExtractErrorValue(json.RawMessage(nil)); got != "" {
		t.Errorf("ExtractErrorValue(nil) = %q, want empty", got)
	}
}

func TestProviderErrorNilSafe(t *testing.T) {
	var err *Error
	if err.Error() != "" {
		t.Errorf("nil.Error() = %q, want empty", err.Error())
	}
	if err.HTTPStatusCode() != 0 {
		t.Errorf("nil.HTTPStatusCode() = %d, want 0", err.HTTPStatusCode())
	}
	if err.FailureClass() != "" {
		t.Errorf("nil.FailureClass() = %q, want empty", err.FailureClass())
	}
	if err.DiagnosticFields() != nil {
		t.Errorf("nil.DiagnosticFields() = %v, want nil", err.DiagnosticFields())
	}
}
