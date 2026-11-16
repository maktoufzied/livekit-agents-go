// Package deepgram implements the stt.STT and tts.TTS interfaces on top of
// the Deepgram API: live streaming transcription over WebSocket
// (/v1/listen) and text-to-speech, streaming over WebSocket or one request
// at a time over REST (/v1/speak).
//
// # Errors
//
// NewSTT and NewTTS return plain errors that describe the invalid option.
// Every other error is a *providers.Error, except the io.EOF that Recv
// returns at the normal end of a stream. A stream used after Close fails
// with an error matching providers.ErrStreamClosed, a cancelled or expired
// context with one matching the context error; neither is retryable. No
// error ever contains the API key.
package deepgram

import (
	"context"
	"errors"
	"net/http"

	"github.com/maktoufzied/livekit-agents-go/providers"
)

const (
	// DefaultBaseURL is the Deepgram API origin used when no BaseURL is set.
	DefaultBaseURL = "https://api.deepgram.com"
	// DefaultSTTModel is the transcription model used when STTOptions.Model
	// is empty.
	DefaultSTTModel = "nova-3"
	// DefaultTTSModel is the voice used when TTSOptions.Model is empty.
	DefaultTTSModel = "aura-2-thalia-en"

	providerName = "deepgram"
	// apiKeyEnv is the environment variable read when no key is configured.
	apiKeyEnv = "DEEPGRAM_API_KEY"
)

// defaultHTTPClient is used when no HTTPClient is configured. It has no
// overall Timeout: responses are streams bounded by the caller's context.
var defaultHTTPClient = &http.Client{}

// resolveKey returns the configured key or the one from the environment.
// field names the option in the error, e.g. "STTOptions.APIKey".
func resolveKey(key providers.Secret, field string) (providers.Secret, error) {
	if key == "" {
		key = providers.SecretFromEnv(apiKeyEnv)
	}
	if key == "" {
		return "", errors.New("deepgram: API key is required (" + field + " or " + apiKeyEnv + ")")
	}
	return key, nil
}

// newError builds a provider error whose chain cannot contain the API key.
func newError(kind providers.Kind, op string, status int, retryable bool, err error, key providers.Secret) *providers.Error {
	return &providers.Error{
		Provider:   providerName,
		Kind:       kind,
		Op:         op,
		StatusCode: status,
		Retryable:  retryable,
		Err:        providers.Scrub(err, key),
	}
}

// closedError is returned by stream methods called after Close.
func closedError(kind providers.Kind, op string) *providers.Error {
	return &providers.Error{Provider: providerName, Kind: kind, Op: op, Err: providers.ErrStreamClosed}
}

// ctxError reports that the context bounding a stream is done. It unwraps
// to the context error and is never retryable.
func ctxError(ctx context.Context, kind providers.Kind, op string) *providers.Error {
	return &providers.Error{Provider: providerName, Kind: kind, Op: op, Err: ctx.Err()}
}

// The clients hold the API key in an unexported field, where fmt would
// print it raw with %v or %+v (Secret's own redaction is not consulted for
// unexported fields). These methods keep it out of any formatted output.

// String implements fmt.Stringer without revealing the API key.
func (s STT) String() string { return "deepgram.STT{model: " + s.opts.Model + "}" }

// GoString implements fmt.GoStringer without revealing the API key.
func (s STT) GoString() string { return s.String() }

// String implements fmt.Stringer without revealing the API key.
func (t TTS) String() string { return "deepgram.TTS{model: " + t.opts.Model + "}" }

// GoString implements fmt.GoStringer without revealing the API key.
func (t TTS) GoString() string { return t.String() }
