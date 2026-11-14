package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/providers"
	"github.com/maktoufzied/livekit-agents-go/tts"
)

func newTestTTS(t *testing.T, url string, mod func(*TTSOptions)) *TTS {
	t.Helper()
	opts := TTSOptions{APIKey: testKey, BaseURL: url}
	if mod != nil {
		mod(&opts)
	}
	s, err := NewTTS(opts)
	if err != nil {
		t.Fatalf("NewTTS: %v", err)
	}
	return s
}

// pcmHandler serves pcm in pieces of the given size, flushing each one.
func pcmHandler(pcm []byte, piece int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/pcm")
		for rest := pcm; len(rest) > 0; {
			n := min(piece, len(rest))
			w.Write(rest[:n])
			w.(http.Flusher).Flush()
			rest = rest[n:]
		}
	}
}

func drain(s tts.Stream) ([]audio.Frame, error) {
	var frames []audio.Frame
	for {
		f, err := s.Recv()
		if err == io.EOF {
			return frames, nil
		}
		if err != nil {
			return frames, err
		}
		frames = append(frames, f)
	}
}

func TestNewTTSAPIKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	_, err := NewTTS(TTSOptions{})
	assertPlainError(t, err, "API key is required (TTSOptions.APIKey or OPENAI_API_KEY)")
	if _, err := NewTTS(TTSOptions{AllowNoAPIKey: true}); err != nil {
		t.Errorf("AllowNoAPIKey: %v", err)
	}
	for _, speed := range []float64{0, 0.2, 4.5, math.NaN()} {
		_, err := NewTTS(TTSOptions{APIKey: testKey, Speed: &speed})
		assertPlainError(t, err, "Speed must be between 0.25 and 4")
	}
	for _, speed := range []float64{0.25, 1, 4} {
		if _, err := NewTTS(TTSOptions{APIKey: testKey, Speed: &speed}); err != nil {
			t.Errorf("Speed %v: %v", speed, err)
		}
	}
	s, err := NewTTS(TTSOptions{APIKey: testKey})
	if err != nil {
		t.Fatalf("NewTTS with key: %v", err)
	}
	if out := fmt.Sprintf("%+v %#v", s, s); strings.Contains(out, testKey) {
		t.Errorf("formatted TTS leaks the key: %s", out)
	}
	if got := s.Format(); got != (audio.Format{SampleRate: 24000, Channels: 1}) {
		t.Errorf("Format = %+v", got)
	}
	if s.c.http != defaultHTTPClient {
		t.Error("a nil HTTPClient must select the shared client")
	}
}

func TestSynthesizeRequestShape(t *testing.T) {
	tests := []struct {
		name string
		mod  func(*TTSOptions)
		want string
	}{
		{
			name: "defaults",
			want: `{"model":"gpt-4o-mini-tts","voice":"marin","input":"Hello there.","response_format":"pcm"}`,
		},
		{
			name: "all options",
			mod: func(o *TTSOptions) {
				o.Model, o.Voice, o.Instructions, o.Speed = "tts-x", "cedar", "Speak calmly.", ptr(1.25)
			},
			want: `{"model":"tts-x","voice":"cedar","input":"Hello there.","response_format":"pcm",
				"instructions":"Speak calmly.","speed":1.25}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				got     any
				headers http.Header
				target  string
			)
			srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				target, headers = r.Method+" "+r.URL.Path, r.Header.Clone()
				json.NewDecoder(r.Body).Decode(&got)
			})
			s, err := newTestTTS(t, srv.URL, tt.mod).Synthesize(context.Background(), "Hello there.")
			if err != nil {
				t.Fatalf("Synthesize: %v", err)
			}
			defer s.Close()
			if frames, err := drain(s); err != nil || len(frames) != 0 {
				t.Errorf("empty body: frames = %d, err = %v", len(frames), err)
			}
			if target != "POST /audio/speech" {
				t.Errorf("request = %s", target)
			}
			if h := headers.Get("Authorization"); h != "Bearer "+testKey {
				t.Errorf("Authorization = %q", h)
			}
			var want any
			if err := json.Unmarshal([]byte(tt.want), &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				gotJSON, _ := json.Marshal(got)
				t.Errorf("request body = %s", gotJSON)
			}
		})
	}
}

func TestSynthesizeFrames(t *testing.T) {
	format := audio.Format{SampleRate: 24000, Channels: 1}
	// 1.23 s plus a dangling sample: not a whole number of frames.
	tone := audio.Tone(format, 440, 1230*time.Millisecond, 0.5)
	tone.Data = append(tone.Data, 1234)
	pcm := tone.BytesLE()

	tests := []struct {
		name  string
		piece int // bytes per server write; odd sizes split samples
		frame time.Duration
	}{
		{"single write", len(pcm), 0},
		{"odd pieces", 333, 0},
		{"tiny pieces, frames of 50ms", 7, 50 * time.Millisecond},
		{"large pieces", 9001, 30 * time.Millisecond},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newServer(t, pcmHandler(pcm, tt.piece))
			s, err := newTestTTS(t, srv.URL, func(o *TTSOptions) { o.FrameDuration = tt.frame }).
				Synthesize(context.Background(), "hi")
			if err != nil {
				t.Fatalf("Synthesize: %v", err)
			}
			defer s.Close()
			frames, err := drain(s)
			if err != nil {
				t.Fatalf("Recv: %v", err)
			}

			frameDur := tt.frame
			if frameDur == 0 {
				frameDur = 20 * time.Millisecond
			}
			var samples []int16
			var total time.Duration
			for i, f := range frames {
				if f.Format() != format {
					t.Fatalf("frame %d format = %+v", i, f.Format())
				}
				if i < len(frames)-1 && f.Duration() != frameDur {
					t.Fatalf("frame %d duration = %v, want %v", i, f.Duration(), frameDur)
				}
				if f.Duration() > frameDur || len(f.Data) == 0 {
					t.Fatalf("frame %d duration = %v", i, f.Duration())
				}
				total += f.Duration()
				samples = append(samples, f.Data...)
			}
			if want := tone.Duration(); total != want && want-total > time.Millisecond {
				t.Errorf("total duration = %v, want %v", total, want)
			}
			if len(samples)*2 != len(pcm) {
				t.Errorf("served %d bytes, got %d bytes of audio", len(pcm), len(samples)*2)
			}
			if !reflect.DeepEqual(samples, tone.Data) {
				t.Error("audio differs from what was served")
			}
			if _, err := s.Recv(); err != io.EOF {
				t.Errorf("Recv after EOF = %v", err)
			}
		})
	}
}

func TestSynthesizeIsIncremental(t *testing.T) {
	// 100 ms are served, then the server stalls: the frames must be
	// available without waiting for the end of the response.
	pcm := audio.Tone(audio.Format{SampleRate: 24000, Channels: 1}, 440, 100*time.Millisecond, 0.5).BytesLE()
	srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write(pcm)
		w.(http.Flusher).Flush()
		waitGone(r)
	})
	s, err := newTestTTS(t, srv.URL, nil).Synthesize(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	for i := range 5 {
		if f, err := s.Recv(); err != nil || f.Duration() != 20*time.Millisecond {
			t.Fatalf("frame %d = %v, %v", i, f.Duration(), err)
		}
	}

	errc := make(chan error, 1)
	go func() {
		_, err := s.Recv()
		errc <- err
	}()
	time.Sleep(20 * time.Millisecond) // let Recv block on the body
	s.Close()
	select {
	case err := <-errc:
		if err == nil || err == io.EOF || providers.IsRetryable(err) {
			t.Fatalf("Recv after Close = %v, want a non-retryable error", err)
		}
		providerError(t, err, providers.KindTTS)
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not unblock Recv")
	}
}

func TestSynthesizeHTTPErrors(t *testing.T) {
	tests := []struct {
		status    int
		body      string
		retryable bool
		contains  string
	}{
		{429, `{"error":{"message":"Rate limit reached","type":"rate_limit_error"}}`, true, "Rate limit reached"},
		{500, `oops`, true, "oops"},
		{400, `{"error":{"message":"Invalid voice","type":"invalid_request_error"}}`, false, "Invalid voice"},
		{401, `{"error":{"message":"Incorrect API key provided: ` + testKey + `","type":"invalid_request_error"}}`, false, "Incorrect API key"},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprint(tt.status), func(t *testing.T) {
			srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				io.WriteString(w, tt.body)
			})
			s, err := newTestTTS(t, srv.URL, nil).Synthesize(context.Background(), "hi")
			if err == nil {
				s.Close()
				t.Fatal("Synthesize succeeded")
			}
			pe := providerError(t, err, providers.KindTTS)
			if pe.StatusCode != tt.status || pe.Retryable != tt.retryable || pe.Op != "request" {
				t.Errorf("error = %+v", pe)
			}
			if !strings.Contains(err.Error(), tt.contains) {
				t.Errorf("error = %q", err)
			}
		})
	}
}

func TestSynthesizeTruncatedBody(t *testing.T) {
	// The server announces more bytes than it sends, then drops the
	// connection: the frames received are delivered, then a retryable error.
	pcm := audio.Silence(audio.Format{SampleRate: 24000, Channels: 1}, 40*time.Millisecond).BytesLE()
	srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprint(len(pcm)*2))
		w.Write(pcm)
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler)
	})
	s, err := newTestTTS(t, srv.URL, nil).Synthesize(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	frames, err := drain(s)
	if len(frames) != 2 {
		t.Errorf("frames = %d, want 2", len(frames))
	}
	if err == nil {
		t.Fatal("truncated body ended with io.EOF")
	}
	if pe := providerError(t, err, providers.KindTTS); !pe.Retryable || pe.Op != "recv" {
		t.Errorf("error = %+v", pe)
	}
}

func TestSynthesizeContextCancel(t *testing.T) {
	srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.(http.Flusher).Flush()
		waitGone(r)
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s, err := newTestTTS(t, srv.URL, nil).Synthesize(ctx, "hi")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	time.AfterFunc(20*time.Millisecond, cancel)
	_, err = s.Recv()
	assertCause(t, err, providers.KindTTS, context.Canceled)

	expiring, cancel2 := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel2()
	s2, err := newTestTTS(t, srv.URL, nil).Synthesize(expiring, "hi")
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()
	_, err = s2.Recv()
	assertCause(t, err, providers.KindTTS, context.DeadlineExceeded)
}

func TestSynthesizeClose(t *testing.T) {
	srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.(http.Flusher).Flush()
		waitGone(r)
	})
	s, err := newTestTTS(t, srv.URL, nil).Synthesize(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	errc := make(chan error, 1)
	go func() {
		_, err := s.Recv()
		errc <- err
	}()
	time.Sleep(20 * time.Millisecond) // let Recv block on the body
	if err := s.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	select {
	case err := <-errc:
		assertCause(t, err, providers.KindTTS, providers.ErrStreamClosed)
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not unblock Recv")
	}
	if err := s.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	_, err = s.Recv()
	assertCause(t, err, providers.KindTTS, providers.ErrStreamClosed)
}
