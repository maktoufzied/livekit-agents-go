// Package openai implements the llm.LLM and tts.TTS interfaces on top of the
// OpenAI HTTP API.
//
// The LLM uses the Chat Completions streaming endpoint
// (POST {BaseURL}/chat/completions), which is also the de-facto compatible
// API of many other vendors and local servers: point BaseURL at them. The
// TTS uses POST {BaseURL}/audio/speech with raw PCM output.
//
// The package depends only on the standard library and on this module's
// interface packages.
//
// # Errors
//
// NewLLM and NewTTS return plain errors that describe the invalid option.
// Every other error is a *providers.Error, except the io.EOF that Recv
// returns at the normal end of a stream. A stream used after Close fails
// with an error matching providers.ErrStreamClosed, a cancelled or expired
// context with one matching the context error; neither is retryable. No
// error ever contains the API key.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/maktoufzied/livekit-agents-go/providers"
)

const (
	providerName = "openai"

	// DefaultBaseURL is the OpenAI API root used when no BaseURL is set. It
	// includes the /v1 prefix, see LLMOptions.BaseURL.
	DefaultBaseURL = "https://api.openai.com/v1"

	// apiKeyEnv is the environment variable read when no key is given.
	apiKeyEnv = "OPENAI_API_KEY"

	// maxErrorBody bounds how much of an error response is read.
	maxErrorBody = 64 << 10
)

// defaultHTTPClient is used when no HTTPClient is configured. It has no
// overall Timeout: responses are streams bounded by the caller's context.
var defaultHTTPClient = &http.Client{}

// client holds what the LLM and the TTS share: where to send requests and
// how to authenticate them. It is always held by pointer: fmt does not
// follow nested pointers, so printing an LLM or a TTS with %+v cannot reveal
// the key (Secret's own redaction does not apply to unexported fields).
type client struct {
	kind    providers.Kind
	baseURL string
	key     providers.Secret
	http    *http.Client
}

// newClient resolves the options shared by the constructors. keyField names
// the key option in the error, e.g. "LLMOptions.APIKey".
func newClient(kind providers.Kind, key providers.Secret, keyField string, allowNoKey bool, baseURL string, hc *http.Client) (*client, error) {
	if key == "" {
		key = providers.SecretFromEnv(apiKeyEnv)
	}
	if key == "" && !allowNoKey {
		return nil, errors.New("openai: API key is required (" + keyField + " or " + apiKeyEnv + ")")
	}
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	if hc == nil {
		hc = defaultHTTPClient
	}
	return &client{kind: kind, baseURL: strings.TrimRight(baseURL, "/"), key: key, http: hc}, nil
}

// post sends a JSON body and returns the response when its status is 200.
// Any other outcome is returned as a *providers.Error and the response body
// is closed.
func (c *client) post(ctx context.Context, path, accept string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, c.errorf("request", false, err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", accept)
	if c.key != "" {
		req.Header.Set("Authorization", "Bearer "+c.key.Reveal())
	}
	resp, err := c.http.Do(req)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, c.errorf("request", false, ctxErr)
		}
		return nil, c.errorf("request", true, err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
		pe := c.errorf("request", providers.RetryableStatus(resp.StatusCode), errors.New(apiErrorMessage(raw, resp.StatusCode)))
		pe.StatusCode = resp.StatusCode
		return nil, pe
	}
	return resp, nil
}

// errorf builds a provider error whose chain cannot contain the API key: the
// key may show up in an error when a server echoes the request headers back.
// An error that had to be scrubbed loses its chain, see providers.Scrub.
func (c *client) errorf(op string, retryable bool, err error) *providers.Error {
	return &providers.Error{
		Provider: providerName, Kind: c.kind, Op: op,
		Retryable: retryable, Err: providers.Scrub(err, c.key),
	}
}

// closedError is returned by Recv once Close was called.
func (c *client) closedError() *providers.Error {
	return c.errorf("recv", false, providers.ErrStreamClosed)
}

// streamError maps a failure to read a response body. parent is the context
// the caller passed in; closed reports whether Close was called.
func (c *client) streamError(parent context.Context, closed bool, err error) *providers.Error {
	switch {
	case closed:
		return c.closedError()
	case parent.Err() != nil:
		return c.errorf("recv", false, parent.Err())
	default:
		return c.errorf("recv", true, err)
	}
}

// apiError is the error object of the OpenAI API: {"error": {...}}.
type apiError struct {
	Message string          `json:"message"`
	Type    string          `json:"type"`
	Code    json.RawMessage `json:"code"`
}

func (e *apiError) String() string {
	msg := e.Message
	if msg == "" {
		msg = "unknown error"
	}
	if e.Type != "" {
		msg += " (" + e.Type + ")"
	}
	return msg
}

// apiErrorMessage extracts a human-readable message from an error response.
func apiErrorMessage(raw []byte, status int) string {
	var body struct {
		Error *apiError `json:"error"`
	}
	if json.Unmarshal(raw, &body) == nil && body.Error != nil && body.Error.Message != "" {
		return body.Error.String()
	}
	text := strings.TrimSpace(string(raw))
	if text == "" {
		return http.StatusText(status)
	}
	const limit = 512
	if len(text) > limit {
		text = text[:limit] + "..."
	}
	return fmt.Sprintf("%s: %s", http.StatusText(status), text)
}
