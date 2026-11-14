package openai

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/maktoufzied/livekit-agents-go/providers"
)

const testKey = "sk-test-secret-0123456789"

// newServer starts a test server that is closed with the test.
func newServer(t *testing.T, h http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv
}

// providerError asserts that err is a *providers.Error of this package.
func providerError(t *testing.T, err error, kind providers.Kind) *providers.Error {
	t.Helper()
	var pe *providers.Error
	if !errors.As(err, &pe) {
		t.Fatalf("error %v (%T) is not a *providers.Error", err, err)
	}
	if pe.Provider != "openai" || pe.Kind != kind || pe.Op == "" {
		t.Fatalf("provider/kind/op = %q/%q/%q, want openai/%s and an op", pe.Provider, pe.Kind, pe.Op, kind)
	}
	assertNoKey(t, err)
	return pe
}

// assertNoKey checks that the key appears nowhere in err: neither in its
// message, nor in its formatting, nor in any error of its Unwrap chain.
func assertNoKey(t *testing.T, err error) {
	t.Helper()
	for e := err; e != nil; e = errors.Unwrap(e) {
		for _, s := range []string{e.Error(), fmt.Sprintf("%v", e), fmt.Sprintf("%+v", e), fmt.Sprintf("%#v", e)} {
			if strings.Contains(s, testKey) {
				t.Errorf("error leaks the API key (%T in the chain): %s", e, strings.ReplaceAll(s, testKey, "<KEY>"))
			}
		}
	}
}

// assertPlainError checks the contract of constructor errors: a plain error
// prefixed with the package name, never a *providers.Error.
func assertPlainError(t *testing.T, err error, contains string) {
	t.Helper()
	if err == nil {
		t.Fatalf("want an error mentioning %q, got nil", contains)
	}
	var pe *providers.Error
	if errors.As(err, &pe) {
		t.Errorf("constructor error is a *providers.Error: %v", err)
	}
	if !strings.HasPrefix(err.Error(), "openai: ") || !strings.Contains(err.Error(), contains) {
		t.Errorf("error = %q, want an \"openai: \" prefix and a mention of %q", err, contains)
	}
	assertNoKey(t, err)
}

// assertCause checks the contract of use-after-Close and context errors: a
// non-retryable *providers.Error that unwraps to cause.
func assertCause(t *testing.T, err error, kind providers.Kind, cause error) {
	t.Helper()
	pe := providerError(t, err, kind)
	if !errors.Is(err, cause) {
		t.Errorf("error %v does not unwrap to %v", err, cause)
	}
	if pe.Retryable || providers.IsRetryable(err) {
		t.Errorf("error %v must not be retryable", err)
	}
}

func ptr[T any](v T) *T { return &v }

// waitGone blocks a handler until the client goes away. The request body is
// drained first: the server only watches the connection once it is read.
func waitGone(r *http.Request) {
	io.Copy(io.Discard, r.Body)
	<-r.Context().Done()
}
