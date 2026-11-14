package anthropic

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/maktoufzied/livekit-agents-go/providers"
)

// APIError is the error object returned by the Anthropic API, either as the
// body of a failed HTTP response or as an "error" event inside a stream. It
// is wrapped in a *providers.Error; reach it with errors.As.
type APIError struct {
	// Type is the API error type, e.g. "rate_limit_error".
	Type string
	// Message is the human-readable message of the API.
	Message string
	// RequestID identifies the request for Anthropic support, when known.
	RequestID string
}

func (e *APIError) Error() string {
	msg := e.Type
	if e.Message != "" {
		if msg != "" {
			msg += ": "
		}
		msg += e.Message
	}
	if msg == "" {
		msg = "unknown API error"
	}
	if e.RequestID != "" {
		msg += " (request " + e.RequestID + ")"
	}
	return msg
}

// errorEnvelope is the wire form of an API error.
type errorEnvelope struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
	RequestID string `json:"request_id"`
}

// newError builds a provider error whose chain cannot contain the API key.
// An error that had to be scrubbed loses its chain, see providers.Scrub; an
// *APIError never does, its fields are redacted when it is built.
func (l *LLM) newError(op string, status int, retryable bool, err error) *providers.Error {
	return &providers.Error{
		Provider:   providerName,
		Kind:       providers.KindLLM,
		Op:         op,
		StatusCode: status,
		Retryable:  retryable,
		Err:        providers.Scrub(err, providers.Secret(l.apiKey())),
	}
}

// closedError is returned by Recv once the caller has closed the stream.
func (l *LLM) closedError() *providers.Error {
	return l.newError("recv", 0, false, providers.ErrStreamClosed)
}

// readAPIError turns the body of a failed HTTP response into an error.
func (l *LLM) readAPIError(resp *http.Response) error {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
	apiErr := &APIError{RequestID: l.redact(resp.Header.Get("request-id"))}

	var env errorEnvelope
	if json.Unmarshal(raw, &env) == nil && (env.Error.Type != "" || env.Error.Message != "") {
		apiErr.Type = l.redact(env.Error.Type)
		apiErr.Message = l.redact(env.Error.Message)
		if apiErr.RequestID == "" {
			apiErr.RequestID = l.redact(env.RequestID)
		}
		return apiErr
	}
	apiErr.Message = l.redact(strings.TrimSpace(string(raw)))
	if apiErr.Message == "" {
		apiErr.Message = http.StatusText(resp.StatusCode)
	}
	return apiErr
}

// streamAPIError maps an "error" event of a stream. The HTTP status of the
// response was 200, so StatusCode carries the status the API documents for
// the error type, or 0 when there is none.
func (l *LLM) streamAPIError(env errorEnvelope) *providers.Error {
	apiErr := &APIError{Type: l.redact(env.Error.Type), Message: l.redact(env.Error.Message), RequestID: l.redact(env.RequestID)}
	status, retryable := 0, false
	switch env.Error.Type {
	case "overloaded_error":
		status, retryable = 529, true
	case "rate_limit_error":
		status, retryable = http.StatusTooManyRequests, true
	case "api_error":
		status, retryable = http.StatusInternalServerError, true
	case "timeout_error":
		status, retryable = http.StatusGatewayTimeout, true
	case "invalid_request_error":
		status = http.StatusBadRequest
	case "authentication_error":
		status = http.StatusUnauthorized
	case "permission_error":
		status = http.StatusForbidden
	case "not_found_error":
		status = http.StatusNotFound
	}
	return l.newError("recv", status, retryable, apiErr)
}

// redact removes the API key from a string that is about to enter an error.
func (l *LLM) redact(s string) string {
	key := l.apiKey()
	if key == "" {
		return s
	}
	return strings.ReplaceAll(s, key, "[REDACTED]")
}
