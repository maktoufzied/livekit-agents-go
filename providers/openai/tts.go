package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/providers"
	"github.com/maktoufzied/livekit-agents-go/tts"
)

const (
	// DefaultTTSModel is the speech model used when TTSOptions.Model is
	// empty.
	DefaultTTSModel = "gpt-4o-mini-tts"
	// DefaultTTSVoice is the voice used when TTSOptions.Voice is empty.
	DefaultTTSVoice = "marin"

	// defaultTTSFrame is the duration of the frames returned by Recv.
	defaultTTSFrame = 20 * time.Millisecond
)

// ttsFormat is what response_format "pcm" returns: headerless 24 kHz
// signed 16-bit little-endian mono samples.
var ttsFormat = audio.Format{SampleRate: 24000, Channels: 1}

// TTSOptions configures a TTS.
type TTSOptions struct {
	// APIKey authenticates the requests. It defaults to the OPENAI_API_KEY
	// environment variable.
	APIKey providers.Secret
	// AllowNoAPIKey permits an empty key, for local compatible servers
	// that do not authenticate and gateways that inject the credentials.
	// No Authorization header is sent then. It defaults to false.
	AllowNoAPIKey bool
	// Model is the model ID. It defaults to DefaultTTSModel.
	Model string
	// Voice is a built-in voice name. It defaults to DefaultTTSVoice.
	Voice string
	// Instructions steers the delivery (tone, pace, accent). It defaults to
	// empty and is sent only when not empty; tts-1 and tts-1-hd do not
	// support it.
	Instructions string
	// Speed is the speech rate, between 0.25 and 4. It defaults to nil,
	// which leaves the provider default (1).
	Speed *float64
	// BaseURL is the API root including the version prefix, for example
	// "http://localhost:8000/v1": the paths appended to it are
	// /chat/completions and /audio/speech. Unlike the other providers of
	// this module it is not a bare origin, because the /v1 prefix is part
	// of what OpenAI-compatible servers document as their base URL. A
	// trailing slash is ignored. It defaults to DefaultBaseURL.
	BaseURL string
	// FrameDuration is the duration of the returned frames. It defaults
	// to 20 ms.
	FrameDuration time.Duration
	// HTTPClient sends the requests. It defaults to nil, which means a
	// client shared by the package with no overall Timeout, because
	// streams are long-lived: bound each stream with its context instead.
	HTTPClient *http.Client
}

// TTS synthesizes speech with the audio/speech endpoint. It is safe for
// concurrent use.
type TTS struct {
	c    *client
	opts TTSOptions
}

var _ tts.TTS = (*TTS)(nil)

// NewTTS validates opts and returns a TTS. It fails when no API key is
// configured, unless opts.AllowNoAPIKey is set, or an option is invalid.
func NewTTS(opts TTSOptions) (*TTS, error) {
	c, err := newClient(providers.KindTTS, opts.APIKey, "TTSOptions.APIKey", opts.AllowNoAPIKey, opts.BaseURL, opts.HTTPClient)
	if err != nil {
		return nil, err
	}
	// Written so that NaN is rejected too.
	if v := opts.Speed; v != nil && !(*v >= 0.25 && *v <= 4) {
		return nil, fmt.Errorf("openai: Speed must be between 0.25 and 4, got %v", *v)
	}
	if opts.Model == "" {
		opts.Model = DefaultTTSModel
	}
	if opts.Voice == "" {
		opts.Voice = DefaultTTSVoice
	}
	if opts.FrameDuration <= 0 {
		opts.FrameDuration = defaultTTSFrame
	}
	opts.APIKey = "" // the client holds the only copy
	return &TTS{c: c, opts: opts}, nil
}

// Format returns 24 kHz mono, the format of every frame.
func (t *TTS) Format() audio.Format { return ttsFormat }

// speechRequest is the body of POST /audio/speech.
type speechRequest struct {
	Model          string   `json:"model"`
	Input          string   `json:"input"`
	Voice          string   `json:"voice"`
	Instructions   string   `json:"instructions,omitempty"`
	ResponseFormat string   `json:"response_format"`
	Speed          *float64 `json:"speed,omitempty"`
}

// Synthesize starts synthesizing text. It returns once the response
// headers are received; the audio is then read incrementally by Recv. ctx
// bounds the whole stream.
func (t *TTS) Synthesize(ctx context.Context, text string) (tts.Stream, error) {
	body, err := json.Marshal(speechRequest{
		Model:          t.opts.Model,
		Input:          text,
		Voice:          t.opts.Voice,
		Instructions:   t.opts.Instructions,
		ResponseFormat: "pcm",
		Speed:          t.opts.Speed,
	})
	if err != nil {
		return nil, t.c.errorf("request", false, err)
	}
	reqCtx, cancel := context.WithCancel(ctx)
	resp, err := t.c.post(reqCtx, "/audio/speech", "application/octet-stream", body)
	if err != nil {
		cancel()
		return nil, err
	}
	return &speechStream{
		c:       t.c,
		parent:  ctx,
		cancel:  cancel,
		body:    resp.Body,
		chunker: audio.NewChunker(ttsFormat, t.opts.FrameDuration),
		buf:     make([]byte, 4096),
	}, nil
}

// speechStream re-frames the PCM response body. Recv reads from the body
// on the caller's goroutine: the stream owns no goroutine.
type speechStream struct {
	c      *client
	parent context.Context
	cancel context.CancelFunc
	body   io.ReadCloser

	closeOnce sync.Once
	closed    atomic.Bool

	// Owned by the Recv goroutine.
	chunker *audio.Chunker
	buf     []byte
	frames  []audio.Frame // decoded, not yet returned
	err     error         // sticky terminal state, io.EOF at the normal end
}

// Recv returns the next frame. Every frame has the configured duration,
// except the last one which may be shorter.
func (s *speechStream) Recv() (audio.Frame, error) {
	if s.closed.Load() {
		return audio.Frame{}, s.c.closedError()
	}
	for len(s.frames) == 0 {
		if s.err != nil {
			return audio.Frame{}, s.err
		}
		n, err := s.body.Read(s.buf)
		s.frames = s.chunker.Write(s.buf[:n])
		switch {
		case err == io.EOF:
			if last, ok := s.chunker.Flush(); ok {
				s.frames = append(s.frames, last)
			}
			s.err = io.EOF
		case err != nil:
			// Frames decoded before the failure are still delivered.
			s.err = s.c.streamError(s.parent, s.closed.Load(), err)
		}
	}
	f := s.frames[0]
	s.frames = s.frames[1:]
	return f, nil
}

// Close aborts the request and unblocks Recv.
func (s *speechStream) Close() error {
	s.closeOnce.Do(func() {
		s.closed.Store(true)
		s.cancel()
		s.body.Close()
	})
	return nil
}
