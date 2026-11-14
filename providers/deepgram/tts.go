package deepgram

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/providers"
	"github.com/maktoufzied/livekit-agents-go/tts"
)

const (
	defaultTTSSampleRate = 24000
	defaultFrameDuration = 20 * time.Millisecond
	speakPath            = "/v1/speak"
	// maxErrorBody bounds how much of an error response is read.
	maxErrorBody = 4 << 10
)

// ttsSampleRates are the rates Deepgram supports for linear16 output.
var ttsSampleRates = []int{8000, 16000, 24000, 32000, 48000}

// TTSOptions configures the Deepgram text-to-speech client.
type TTSOptions struct {
	// APIKey authenticates the requests. It defaults to the
	// DEEPGRAM_API_KEY environment variable.
	APIKey providers.Secret
	// Model is the voice. It defaults to DefaultTTSModel.
	Model string
	// SampleRate is the rate of the returned PCM: 8000, 16000, 24000,
	// 32000 or 48000. It defaults to 24000.
	SampleRate int
	// BaseURL is the bare origin of the API (scheme and host, plus an
	// optional path prefix of a proxy), without the /v1/speak path. It
	// defaults to DefaultBaseURL. The scheme must be http or https.
	BaseURL string
	// FrameDuration is the duration of the frames returned by Recv. It
	// defaults to 20 ms.
	FrameDuration time.Duration
	// HTTPClient sends the requests. It defaults to nil, which means a
	// client shared by the package with no overall Timeout, because
	// streams are long-lived: bound each stream with its context instead.
	HTTPClient *http.Client
	// Header is added to every request. It defaults to none.
	Header http.Header
}

// TTS is a Deepgram text-to-speech client. It is safe for concurrent use.
type TTS struct {
	opts   TTSOptions
	format audio.Format
	url    string
}

var _ tts.TTS = (*TTS)(nil)

// NewTTS validates opts and returns a synthesizer. It fails when no API key
// is configured or the sample rate is not supported.
func NewTTS(opts TTSOptions) (*TTS, error) {
	key, err := resolveKey(opts.APIKey, "TTSOptions.APIKey")
	if err != nil {
		return nil, err
	}
	opts.APIKey = key
	if opts.Model == "" {
		opts.Model = DefaultTTSModel
	}
	if opts.SampleRate == 0 {
		opts.SampleRate = defaultTTSSampleRate
	}
	if !slices.Contains(ttsSampleRates, opts.SampleRate) {
		return nil, fmt.Errorf("deepgram: unsupported TTS sample rate %d (want one of %v)", opts.SampleRate, ttsSampleRates)
	}
	if opts.BaseURL == "" {
		opts.BaseURL = DefaultBaseURL
	}
	if opts.FrameDuration <= 0 {
		opts.FrameDuration = defaultFrameDuration
	}
	if opts.HTTPClient == nil {
		opts.HTTPClient = defaultHTTPClient
	}
	u, err := url.Parse(opts.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("deepgram: invalid BaseURL: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, fmt.Errorf("deepgram: unsupported BaseURL scheme %q", u.Scheme)
	}
	u.Path = strings.TrimSuffix(u.Path, "/") + speakPath
	q := url.Values{}
	q.Set("model", opts.Model)
	q.Set("encoding", "linear16")
	q.Set("sample_rate", strconv.Itoa(opts.SampleRate))
	q.Set("container", "none") // raw PCM, no WAV header
	u.RawQuery = q.Encode()
	return &TTS{
		opts:   opts,
		format: audio.Format{SampleRate: opts.SampleRate, Channels: 1},
		url:    u.String(),
	}, nil
}

// Format returns mono PCM at the configured sample rate.
func (t *TTS) Format() audio.Format { return t.format }

// Synthesize requests the audio for text and returns once Deepgram has
// accepted the request; audio is then read from the response as it arrives.
// ctx bounds the whole stream.
func (t *TTS) Synthesize(ctx context.Context, text string) (tts.Stream, error) {
	ctx, cancel := context.WithCancel(ctx)
	st := &ttsStream{ctx: ctx, cancel: cancel, chunker: audio.NewChunker(t.format, t.opts.FrameDuration)}
	if strings.TrimSpace(text) == "" {
		st.eof = true // nothing to say: an empty stream, not a 400
		return st, nil
	}

	body, err := json.Marshal(struct {
		Text string `json:"text"`
	}{text})
	if err != nil {
		cancel()
		return nil, t.newError("synthesize", 0, false, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.url, bytes.NewReader(body))
	if err != nil {
		cancel()
		return nil, t.newError("synthesize", 0, false, err)
	}
	for k, v := range t.opts.Header {
		req.Header[k] = append([]string(nil), v...)
	}
	req.Header.Set("Authorization", "Token "+t.opts.APIKey.Reveal())
	req.Header.Set("Content-Type", "application/json")

	resp, err := t.opts.HTTPClient.Do(req)
	if err != nil {
		defer cancel()
		if ctx.Err() != nil {
			return nil, ctxError(ctx, providers.KindTTS, "synthesize")
		}
		return nil, t.newError("synthesize", 0, true, err)
	}
	if resp.StatusCode != http.StatusOK {
		defer cancel()
		defer resp.Body.Close()
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
		return nil, t.newError("synthesize", resp.StatusCode, providers.RetryableStatus(resp.StatusCode), errors.New(errorMessage(raw, resp.StatusCode)))
	}
	st.body = resp.Body
	st.newError = t.newError
	return st, nil
}

func (t *TTS) newError(op string, status int, retryable bool, err error) error {
	return newError(providers.KindTTS, op, status, retryable, err, t.opts.APIKey)
}

// errorMessage extracts the human-readable part of a Deepgram error body,
// which is either {"err_code","err_msg","request_id"},
// {"category","message","details","request_id"} or plain text.
func errorMessage(raw []byte, status int) string {
	var e struct {
		ErrCode  string `json:"err_code"`
		ErrMsg   string `json:"err_msg"`
		Category string `json:"category"`
		Message  string `json:"message"`
		Details  string `json:"details"`
	}
	if json.Unmarshal(raw, &e) == nil {
		var parts []string
		for _, p := range []string{e.ErrCode, e.ErrMsg, e.Category, e.Message, e.Details} {
			if p != "" && !slices.Contains(parts, p) {
				parts = append(parts, p)
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, ": ")
		}
	}
	if s := strings.TrimSpace(string(raw)); s != "" {
		return s
	}
	return http.StatusText(status)
}

// ttsStream reads PCM from the response body on the caller's goroutine, so
// it owns no goroutine of its own.
type ttsStream struct {
	ctx      context.Context
	cancel   context.CancelFunc
	body     io.ReadCloser // nil for an empty stream
	newError func(op string, status int, retryable bool, err error) error

	closeOnce sync.Once
	closed    atomic.Bool

	// Recv-only state.
	chunker *audio.Chunker
	pending []audio.Frame
	buf     [4096]byte
	eof     bool
	err     error
}

// Recv returns the next frame, io.EOF after the last one.
func (s *ttsStream) Recv() (audio.Frame, error) {
	for {
		if s.closedNow() {
			return audio.Frame{}, closedError(providers.KindTTS, "recv")
		}
		if len(s.pending) > 0 {
			f := s.pending[0]
			s.pending = s.pending[1:]
			return f, nil
		}
		if s.err != nil {
			return audio.Frame{}, s.err
		}
		if s.eof {
			return audio.Frame{}, io.EOF
		}
		n, err := s.body.Read(s.buf[:])
		if n > 0 {
			s.pending = s.chunker.Write(s.buf[:n])
		}
		switch {
		case err == nil:
		case errors.Is(err, io.EOF):
			s.eof = true
			if f, ok := s.chunker.Flush(); ok {
				s.pending = append(s.pending, f)
			}
		case s.closedNow():
			s.err = closedError(providers.KindTTS, "recv")
		case s.ctx.Err() != nil:
			s.err = ctxError(s.ctx, providers.KindTTS, "recv")
		default:
			s.err = s.newError("recv", 0, true, err)
		}
	}
}

func (s *ttsStream) closedNow() bool { return s.closed.Load() }

// Close aborts the request and unblocks Recv.
func (s *ttsStream) Close() error {
	s.closeOnce.Do(func() {
		s.closed.Store(true)
		s.cancel()
		if s.body != nil {
			s.body.Close()
		}
	})
	return nil
}
