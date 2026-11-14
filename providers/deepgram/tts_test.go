package deepgram

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/providers"
)

func newTestTTS(t *testing.T, baseURL string, mod func(*TTSOptions)) *TTS {
	t.Helper()
	opts := TTSOptions{APIKey: testKey, BaseURL: baseURL}
	if mod != nil {
		mod(&opts)
	}
	s, err := NewTTS(opts)
	if err != nil {
		t.Fatalf("NewTTS: %v", err)
	}
	return s
}

func TestNewTTSValidation(t *testing.T) {
	t.Setenv(apiKeyEnv, "")
	_, err := NewTTS(TTSOptions{})
	assertPlainError(t, err, "API key is required (TTSOptions.APIKey or DEEPGRAM_API_KEY)")
	_, err = NewTTS(TTSOptions{APIKey: testKey, SampleRate: 44100})
	assertPlainError(t, err, "unsupported TTS sample rate 44100")
	_, err = NewTTS(TTSOptions{APIKey: testKey, BaseURL: "wss://example.test"})
	assertPlainError(t, err, "unsupported BaseURL scheme")
	s, err := NewTTS(TTSOptions{APIKey: testKey})
	if err != nil {
		t.Fatal(err)
	}
	if s.opts.HTTPClient != defaultHTTPClient || defaultHTTPClient.Timeout != 0 {
		t.Error("a nil HTTPClient must select the shared client, which has no Timeout")
	}
	if DefaultBaseURL != "https://api.deepgram.com" || DefaultTTSModel != "aura-2-thalia-en" {
		t.Errorf("defaults changed: %s %s", DefaultBaseURL, DefaultTTSModel)
	}
	if got := s.Format(); got != (audio.Format{SampleRate: 24000, Channels: 1}) {
		t.Errorf("Format = %+v", got)
	}
	if want := "https://api.deepgram.com/v1/speak?container=none&encoding=linear16&model=aura-2-thalia-en&sample_rate=24000"; s.url != want {
		t.Errorf("url = %s, want %s", s.url, want)
	}
}

func TestTTSStreamsFrames(t *testing.T) {
	format := audio.Format{SampleRate: 16000, Channels: 1}
	// 1.01 s of audio: 50 full 20 ms frames plus a 10 ms tail.
	pcm := audio.Tone(format, 440, 1010*time.Millisecond, 0.5).BytesLE()
	firstFrameRead := make(chan struct{})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/speak" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		q := r.URL.Query()
		for k, v := range map[string]string{"model": "aura-2-apollo-en", "encoding": "linear16", "sample_rate": "16000", "container": "none"} {
			if q.Get(k) != v {
				t.Errorf("query %s = %q, want %q", k, q.Get(k), v)
			}
		}
		if got := r.Header.Get("Authorization"); got != "Token "+testKey {
			t.Errorf("Authorization = %q", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q", got)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("body: %v", err)
		}
		if len(body) != 1 || body["text"] != `Hello "world".` {
			t.Errorf("body = %v", body)
		}

		w.Header().Set("Content-Type", "audio/l16;rate=16000")
		fl := w.(http.Flusher)
		// First an odd-sized chunk holding a bit more than one frame...
		w.Write(pcm[:701])
		fl.Flush()
		// ...and nothing else until the client has consumed a frame, which
		// proves the response is not buffered whole.
		select {
		case <-firstFrameRead:
		case <-time.After(5 * time.Second):
			t.Error("client did not yield a frame before the response ended")
		}
		for rest := pcm[701:]; len(rest) > 0; {
			n := min(999, len(rest))
			w.Write(rest[:n])
			fl.Flush()
			rest = rest[n:]
		}
	}))
	defer srv.Close()

	s := newTestTTS(t, srv.URL, func(o *TTSOptions) {
		o.Model = "aura-2-apollo-en"
		o.SampleRate = 16000
	})
	st, err := s.Synthesize(context.Background(), `Hello "world".`)
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	defer st.Close()

	var got []byte
	var total time.Duration
	frames := 0
	for {
		f, err := st.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if frames == 0 {
			close(firstFrameRead)
		}
		frames++
		if f.Format() != s.Format() {
			t.Fatalf("frame format = %+v", f.Format())
		}
		if d := f.Duration(); d > 20*time.Millisecond || d <= 0 {
			t.Errorf("frame %d duration = %v", frames, d)
		}
		total += f.Duration()
		got = append(got, f.BytesLE()...)
	}
	if frames != 51 {
		t.Errorf("frames = %d, want 51", frames)
	}
	if total != 1010*time.Millisecond {
		t.Errorf("total duration = %v, want 1.01s", total)
	}
	if string(got) != string(pcm) {
		t.Error("audio bytes differ from what the server sent")
	}
	if _, err := st.Recv(); err != io.EOF {
		t.Errorf("Recv after EOF = %v", err)
	}
}

func TestTTSEmptyText(t *testing.T) {
	s := newTestTTS(t, "http://127.0.0.1:1", nil) // must not be contacted
	st, err := s.Synthesize(context.Background(), "  \n")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if _, err := st.Recv(); err != io.EOF {
		t.Errorf("Recv = %v, want io.EOF", err)
	}
}

func TestTTSHTTPErrors(t *testing.T) {
	tests := []struct {
		status    int
		body      string
		retryable bool
		contains  string
	}{
		{http.StatusTooManyRequests, `{"err_code":"TOO_MANY_REQUESTS","err_msg":"Too many requests. Please try again later","request_id":"r"}`, true, "Too many requests"},
		{http.StatusServiceUnavailable, ``, true, "Service Unavailable"},
		{http.StatusBadRequest, `{"category":"INVALID_JSON","message":"Invalid JSON submitted.","details":"bad","request_id":"r"}`, false, "Invalid JSON submitted."},
		{http.StatusUnauthorized, `{"err_code":"INVALID_AUTH","err_msg":"Invalid credentials for ` + testKey + `","request_id":"r"}`, false, "INVALID_AUTH"},
		{http.StatusForbidden, `plain ` + testKey, false, "plain"},
	}
	for _, tc := range tests {
		t.Run(http.StatusText(tc.status), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				io.WriteString(w, tc.body)
			}))
			defer srv.Close()
			s := newTestTTS(t, srv.URL, nil)
			_, err := s.Synthesize(context.Background(), "hi")
			assertProviderError(t, err, providers.KindTTS, "synthesize", tc.status, tc.retryable)
			if !strings.Contains(err.Error(), tc.contains) {
				t.Errorf("error %q does not mention %q", err, tc.contains)
			}
		})
	}
}

func TestTTSNetworkError(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	s := newTestTTS(t, url, nil)
	_, err := s.Synthesize(context.Background(), "hi")
	assertProviderError(t, err, providers.KindTTS, "synthesize", 0, true)
}

// hangingTTSServer sends one frame of audio and then stalls until the client
// goes away, which it reports on the returned channel.
func hangingTTSServer(t *testing.T) (*httptest.Server, <-chan struct{}) {
	t.Helper()
	aborted := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Write(make([]byte, 960)) // 20 ms at 24 kHz
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(aborted)
	}))
	t.Cleanup(srv.Close)
	return srv, aborted
}

func TestTTSCloseAbortsRequest(t *testing.T) {
	srv, aborted := hangingTTSServer(t)
	s := newTestTTS(t, srv.URL, nil)
	st, err := s.Synthesize(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := st.Recv(); err != nil {
		t.Fatalf("first Recv: %v", err)
	}
	recvErr := make(chan error, 1)
	go func() {
		_, err := st.Recv()
		recvErr <- err
	}()
	time.Sleep(20 * time.Millisecond)
	st.Close()
	select {
	case err := <-recvErr:
		assertCause(t, err, providers.KindTTS, providers.ErrStreamClosed)
	case <-time.After(5 * time.Second):
		t.Fatal("Recv still blocked after Close")
	}
	select {
	case <-aborted:
	case <-time.After(5 * time.Second):
		t.Fatal("server never saw the request aborted")
	}
	// The error is sticky and Close is idempotent.
	_, err = st.Recv()
	assertCause(t, err, providers.KindTTS, providers.ErrStreamClosed)
	if err := st.Close(); err != nil {
		t.Errorf("second Close = %v", err)
	}

	// A stream that reached io.EOF also reports the Close.
	st, err = s.Synthesize(context.Background(), " ")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := st.Recv(); err != io.EOF {
		t.Fatalf("Recv = %v, want io.EOF", err)
	}
	st.Close()
	_, err = st.Recv()
	assertCause(t, err, providers.KindTTS, providers.ErrStreamClosed)
}

func TestTTSContextCancel(t *testing.T) {
	srv, aborted := hangingTTSServer(t)
	s := newTestTTS(t, srv.URL, nil)

	ctx, cancel := context.WithCancel(context.Background())
	st, err := s.Synthesize(ctx, "hi")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if _, err := st.Recv(); err != nil {
		t.Fatalf("first Recv: %v", err)
	}
	time.AfterFunc(20*time.Millisecond, cancel)
	_, err = st.Recv()
	assertCause(t, err, providers.KindTTS, context.Canceled)
	select {
	case <-aborted:
	case <-time.After(5 * time.Second):
		t.Fatal("server never saw the request aborted")
	}

	cancelled, cancel2 := context.WithCancel(context.Background())
	cancel2()
	_, err = s.Synthesize(cancelled, "hi")
	assertCause(t, err, providers.KindTTS, context.Canceled)

	expiring, cancel3 := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel3()
	srv2, _ := hangingTTSServer(t)
	st2, err := newTestTTS(t, srv2.URL, nil).Synthesize(expiring, "hi")
	if err != nil {
		t.Fatal(err)
	}
	defer st2.Close()
	if _, err := st2.Recv(); err != nil {
		t.Fatalf("first Recv: %v", err)
	}
	_, err = st2.Recv()
	assertCause(t, err, providers.KindTTS, context.DeadlineExceeded)
}

func TestTTSTruncatedResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "100000")
		w.Write(make([]byte, 960))
		w.(http.Flusher).Flush()
		// Returning now leaves the body short: the client sees a broken stream.
	}))
	defer srv.Close()
	s := newTestTTS(t, srv.URL, nil)
	st, err := s.Synthesize(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if _, err := st.Recv(); err != nil {
		t.Fatalf("first Recv: %v", err)
	}
	_, err = st.Recv()
	assertProviderError(t, err, providers.KindTTS, "recv", 0, true)
}
