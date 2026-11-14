// Package providers holds the types shared by provider integrations: the
// typed provider error and the Secret wrapper for credentials.
//
// Provider packages live in sub-directories and depend only on the interface
// packages (stt, llm, tts), never on the orchestration core.
package providers

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
)

// ErrStreamClosed is the cause of the error returned by Recv (and Push)
// once the caller has closed a provider stream. Providers wrap it in an
// *Error; test with errors.Is.
var ErrStreamClosed = errors.New("providers: stream closed")

// Kind identifies the capability a provider error relates to.
type Kind string

const (
	KindSTT Kind = "stt"
	KindLLM Kind = "llm"
	KindTTS Kind = "tts"
)

// Error is returned by provider integrations. It never contains credentials.
type Error struct {
	Provider   string // e.g. "deepgram"
	Kind       Kind
	Op         string // e.g. "connect", "recv"
	StatusCode int    // HTTP or WebSocket close status, 0 if not applicable
	Retryable  bool   // a new attempt may succeed
	Err        error
}

func (e *Error) Error() string {
	msg := fmt.Sprintf("%s %s %s", e.Provider, e.Kind, e.Op)
	if e.StatusCode != 0 {
		msg += fmt.Sprintf(" (status %d)", e.StatusCode)
	}
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

func (e *Error) Unwrap() error { return e.Err }

// IsRetryable reports whether err is a provider error flagged retryable.
// Context cancellation and deadline errors are never retryable.
func IsRetryable(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var pe *Error
	return errors.As(err, &pe) && pe.Retryable
}

// RetryableStatus reports whether an HTTP status is worth retrying:
// 408, 409, 429 and 5xx.
func RetryableStatus(code int) bool {
	switch code {
	case http.StatusRequestTimeout, http.StatusConflict, http.StatusTooManyRequests:
		return true
	}
	return code >= 500
}

// Secret wraps a credential so it cannot leak through fmt or slog. Use
// Reveal at the single point where the value is put on the wire.
type Secret string

// Reveal returns the underlying value.
func (s Secret) Reveal() string { return string(s) }

// String implements fmt.Stringer and never returns the value.
func (s Secret) String() string {
	if s == "" {
		return ""
	}
	return "[REDACTED]"
}

// GoString implements fmt.GoStringer.
func (s Secret) GoString() string { return s.String() }

// LogValue implements slog.LogValuer.
func (s Secret) LogValue() slog.Value { return slog.StringValue(s.String()) }

// Scrub returns an error with the same message as err, minus every
// occurrence of the given secrets. It deliberately drops the error chain:
// keeping the original reachable through Unwrap would keep the secret
// reachable too. Use it on errors built from provider responses or URLs
// before they leave a provider package. A nil err yields nil.
func Scrub(err error, secrets ...Secret) error {
	if err == nil {
		return nil
	}
	msg := err.Error()
	changed := false
	for _, s := range secrets {
		if s != "" && strings.Contains(msg, string(s)) {
			msg = strings.ReplaceAll(msg, string(s), "[REDACTED]")
			changed = true
		}
	}
	if !changed {
		return err
	}
	return errors.New(msg)
}

// MarshalText keeps secrets out of JSON and text encodings. A Secret
// therefore does not survive an encoding round trip: load credentials from
// the environment or a secret store, not from re-serialized configuration.
func (s Secret) MarshalText() ([]byte, error) { return []byte(s.String()), nil }

// SecretFromEnv reads a secret from the first non-empty environment variable.
func SecretFromEnv(names ...string) Secret {
	for _, n := range names {
		if v := os.Getenv(n); v != "" {
			return Secret(v)
		}
	}
	return ""
}
