package deepgram

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/providers"
	"github.com/maktoufzied/livekit-agents-go/stt"
)

const testKey = "dg-secret-test-key-123"

var testFormat = audio.Format{SampleRate: 16000, Channels: 1}

// wsServer starts a server that upgrades /v1/listen and runs handle with the
// connection. Handlers are waited for on cleanup.
func wsServer(t *testing.T, handle func(r *http.Request, c *websocket.Conn)) *httptest.Server {
	t.Helper()
	var wg sync.WaitGroup
	up := websocket.Upgrader{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		wg.Add(1)
		defer wg.Done()
		c, err := up.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade: %v", err)
			return
		}
		defer c.Close()
		handle(r, c)
	}))
	t.Cleanup(func() {
		srv.Close()
		wg.Wait()
	})
	return srv
}

func newTestSTT(t *testing.T, baseURL string, mod func(*STTOptions)) *STT {
	t.Helper()
	opts := STTOptions{APIKey: testKey, BaseURL: baseURL}
	if mod != nil {
		mod(&opts)
	}
	s, err := NewSTT(opts)
	if err != nil {
		t.Fatalf("NewSTT: %v", err)
	}
	return s
}

// closeNormally sends a normal close frame and waits for the peer to hang up.
func closeNormally(c *websocket.Conn, code int, text string) {
	c.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(code, text), time.Now().Add(time.Second))
	c.SetReadDeadline(time.Now().Add(2 * time.Second))
	for {
		if _, _, err := c.ReadMessage(); err != nil {
			return
		}
	}
}

// drain blocks until the client goes away.
func drain(c *websocket.Conn) {
	for {
		if _, _, err := c.ReadMessage(); err != nil {
			return
		}
	}
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
	if !strings.HasPrefix(err.Error(), "deepgram: ") || !strings.Contains(err.Error(), contains) {
		t.Errorf("error = %q, want a \"deepgram: \" prefix and a mention of %q", err, contains)
	}
	assertNoKey(t, err)
}

// assertCause checks the contract of use-after-Close and context errors: a
// non-retryable *providers.Error that unwraps to cause.
func assertCause(t *testing.T, err error, kind providers.Kind, cause error) {
	t.Helper()
	var pe *providers.Error
	if !errors.As(err, &pe) {
		t.Fatalf("want *providers.Error, got %T: %v", err, err)
	}
	if pe.Provider != "deepgram" || pe.Kind != kind || pe.Op == "" {
		t.Errorf("got %+v, want provider deepgram, kind %s and an op", pe, kind)
	}
	if !errors.Is(err, cause) {
		t.Errorf("error %v does not unwrap to %v", err, cause)
	}
	if pe.Retryable || providers.IsRetryable(err) {
		t.Errorf("error %v must not be retryable", err)
	}
	assertNoKey(t, err)
}

func assertProviderError(t *testing.T, err error, kind providers.Kind, op string, status int, retryable bool) {
	t.Helper()
	var pe *providers.Error
	if !errors.As(err, &pe) {
		t.Fatalf("want *providers.Error, got %T: %v", err, err)
	}
	if pe.Provider != "deepgram" || pe.Kind != kind || pe.Op != op || pe.StatusCode != status || pe.Retryable != retryable {
		t.Errorf("got %+v, want kind=%s op=%s status=%d retryable=%v", pe, kind, op, status, retryable)
	}
	if providers.IsRetryable(err) != retryable {
		t.Errorf("IsRetryable = %v, want %v", !retryable, retryable)
	}
	assertNoKey(t, err)
}

func TestNewSTTRequiresKey(t *testing.T) {
	t.Setenv(apiKeyEnv, "")
	_, err := NewSTT(STTOptions{})
	assertPlainError(t, err, "API key is required (STTOptions.APIKey or DEEPGRAM_API_KEY)")
	t.Setenv(apiKeyEnv, "from-env")
	s, err := NewSTT(STTOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if s.opts.APIKey.Reveal() != "from-env" || s.opts.Model != "nova-3" || DefaultSTTModel != "nova-3" {
		t.Errorf("unexpected defaults: key set=%v model=%q", s.opts.APIKey != "", s.opts.Model)
	}
	if got, want := s.listenURL(testFormat), "wss://api.deepgram.com/v1/listen?"; !strings.HasPrefix(got, want) {
		t.Errorf("default listen URL = %s, want prefix %s", got, want)
	}
}

func TestNewSTTValidation(t *testing.T) {
	tests := []struct {
		name     string
		opts     STTOptions
		contains string
	}{
		{"utterance end without interim", STTOptions{UtteranceEnd: time.Second, InterimResults: new(false)}, "UtteranceEnd requires InterimResults"},
		// A bare integer is nanoseconds: the classic mistake after EndpointingMS.
		{"endpointing below 1ms", STTOptions{Endpointing: 300}, "Endpointing"},
		{"utterance end below 1ms", STTOptions{UtteranceEnd: 1000}, "UtteranceEnd"},
		{"bad scheme", STTOptions{BaseURL: "ftp://example.test"}, "unsupported BaseURL scheme"},
		{"bad url", STTOptions{BaseURL: "https://exa mple.test"}, "invalid BaseURL"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.opts.APIKey = testKey
			_, err := NewSTT(tc.opts)
			assertPlainError(t, err, tc.contains)
		})
	}
}

func TestSTTDurationsOnTheWire(t *testing.T) {
	tests := []struct {
		endpointing, utteranceEnd  time.Duration
		wantEndpoint, wantUtterEnd string
	}{
		{0, 0, "", ""},
		{-1, 0, "false", ""},
		{-time.Second, 0, "false", ""},
		{time.Millisecond, time.Second, "1", "1000"},
		{300 * time.Millisecond, 1500 * time.Millisecond, "300", "1500"},
		{2 * time.Second, 0, "2000", ""},
	}
	for _, tc := range tests {
		s := newTestSTT(t, "", func(o *STTOptions) { o.Endpointing, o.UtteranceEnd = tc.endpointing, tc.utteranceEnd })
		u, err := url.Parse(s.listenURL(testFormat))
		if err != nil {
			t.Fatal(err)
		}
		q := u.Query()
		if q.Get("endpointing") != tc.wantEndpoint || q.Get("utterance_end_ms") != tc.wantUtterEnd {
			t.Errorf("Endpointing %v, UtteranceEnd %v: endpointing=%q utterance_end_ms=%q, want %q and %q",
				tc.endpointing, tc.utteranceEnd, q.Get("endpointing"), q.Get("utterance_end_ms"), tc.wantEndpoint, tc.wantUtterEnd)
		}
	}
}

func TestSTTNetDialContext(t *testing.T) {
	srv := wsServer(t, func(r *http.Request, c *websocket.Conn) { drain(c) })
	var dialed []string
	s := newTestSTT(t, srv.URL, func(o *STTOptions) {
		o.NetDialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
			dialed = append(dialed, network+" "+addr)
			var d net.Dialer
			return d.DialContext(ctx, network, addr)
		}
	})
	st, err := s.Stream(context.Background(), stt.StreamOptions{Format: testFormat})
	if err != nil {
		t.Fatal(err)
	}
	st.Close()
	if want := "tcp " + strings.TrimPrefix(srv.URL, "http://"); len(dialed) != 1 || dialed[0] != want {
		t.Errorf("NetDialContext calls = %v, want [%s]", dialed, want)
	}

	s = newTestSTT(t, srv.URL, func(o *STTOptions) {
		o.NetDialContext = func(context.Context, string, string) (net.Conn, error) {
			return nil, errors.New("no route for " + testKey)
		}
	})
	_, err = s.Stream(context.Background(), stt.StreamOptions{Format: testFormat})
	assertProviderError(t, err, providers.KindSTT, "connect", 0, true)
	if !strings.Contains(err.Error(), "no route for [REDACTED]") {
		t.Errorf("error %q does not carry the scrubbed dial error", err)
	}
}

func TestSTTStreamInvalidFormat(t *testing.T) {
	s := newTestSTT(t, "http://127.0.0.1:1", nil) // must not be contacted
	_, err := s.Stream(context.Background(), stt.StreamOptions{})
	assertProviderError(t, err, providers.KindSTT, "connect", 0, false)
}

func TestSTTRequestAndEventMapping(t *testing.T) {
	frames := []audio.Frame{
		audio.Tone(testFormat, 440, 20*time.Millisecond, 0.5),
		audio.Tone(testFormat, 880, 20*time.Millisecond, 0.5),
		audio.Tone(testFormat, 220, 20*time.Millisecond, 0.5),
	}
	var want []byte
	for _, f := range frames {
		want = append(want, f.BytesLE()...)
	}

	srv := wsServer(t, func(r *http.Request, c *websocket.Conn) {
		if r.URL.Path != "/v1/listen" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Token "+testKey {
			t.Errorf("Authorization = %q", got)
		}
		if got := r.Header.Get("X-Extra"); got != "1" {
			t.Errorf("X-Extra = %q", got)
		}
		q := r.URL.Query()
		for k, v := range map[string]string{
			"model": "nova-3", "language": "en-US", "encoding": "linear16",
			"sample_rate": "16000", "channels": "1", "interim_results": "true",
			"punctuate": "true", "smart_format": "true", "endpointing": "300",
			"utterance_end_ms": "1000", "vad_events": "true",
		} {
			if q.Get(k) != v {
				t.Errorf("query %s = %q, want %q", k, q.Get(k), v)
			}
		}

		// Audio must arrive as binary frames, then CloseStream as text.
		var got []byte
		for {
			mt, data, err := c.ReadMessage()
			if err != nil {
				t.Errorf("server read: %v", err)
				return
			}
			if mt == websocket.BinaryMessage {
				got = append(got, data...)
				continue
			}
			if string(data) != `{"type":"CloseStream"}` {
				t.Errorf("unexpected text message %s", data)
			}
			break
		}
		if !bytes.Equal(got, want) {
			t.Errorf("server received %d audio bytes, want %d (or content differs)", len(got), len(want))
		}

		for _, m := range []string{
			// Silence before any speech: no EndOfSpeech must come out.
			`{"type":"Results","start":0,"duration":0.5,"is_final":true,"speech_final":true,"channel":{"alternatives":[{"transcript":"","confidence":0}]}}`,
			`{"type":"SpeechStarted","channel":[0,1],"timestamp":0.5}`,
			`{"type":"Results","start":0.5,"duration":1,"is_final":false,"speech_final":false,"channel":{"alternatives":[{"transcript":"hello","confidence":0.8}]}}`,
			`{"type":"Results","start":0.5,"duration":1.5,"is_final":true,"speech_final":false,"channel":{"alternatives":[{"transcript":"Hello there.","confidence":0.97}]}}`,
			`{"type":"Results","start":2,"duration":1,"is_final":true,"speech_final":true,"channel":{"alternatives":[{"transcript":"How are you?","confidence":0.9,"languages":["en"]}]}}`,
			// Duplicate end-of-speech signals are collapsed.
			`{"type":"UtteranceEnd","channel":[0,1],"last_word_end":2.9}`,
			`{"type":"Results","start":3,"duration":0.5,"is_final":true,"speech_final":true,"channel":{"alternatives":[{"transcript":"","confidence":0}]}}`,
			// A new utterance ended only by UtteranceEnd.
			`{"type":"Results","start":4,"duration":1,"is_final":true,"speech_final":false,"channel":{"alternatives":[{"transcript":"Bye.","confidence":0.5}]}}`,
			`{"type":"UtteranceEnd","channel":[0,1],"last_word_end":4.8}`,
			`{"type":"SomethingNew","foo":1}`,
			`{"type":"Metadata","request_id":"r","duration":5,"channels":1}`,
		} {
			if err := c.WriteMessage(websocket.TextMessage, []byte(m)); err != nil {
				t.Errorf("server write: %v", err)
				return
			}
		}
		closeNormally(c, websocket.CloseNormalClosure, "")
	})

	s := newTestSTT(t, srv.URL, func(o *STTOptions) {
		o.Language = "en-US"
		o.SmartFormat = true
		o.Endpointing = 300 * time.Millisecond
		o.UtteranceEnd = time.Second
		o.Header = http.Header{"X-Extra": {"1"}}
	})
	st, err := s.Stream(context.Background(), stt.StreamOptions{Format: testFormat})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer st.Close()

	for _, f := range frames {
		if err := st.Push(f); err != nil {
			t.Fatalf("Push: %v", err)
		}
	}
	err = st.Push(audio.Tone(audio.Format{SampleRate: 8000, Channels: 1}, 440, 20*time.Millisecond, 0.5))
	assertProviderError(t, err, providers.KindSTT, "send", 0, false)
	if err := st.CloseSend(); err != nil {
		t.Fatalf("CloseSend: %v", err)
	}
	err = st.Push(frames[0])
	assertProviderError(t, err, providers.KindSTT, "send", 0, false)

	var got []stt.Event
	for {
		ev, err := st.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		got = append(got, ev)
	}
	wantEvents := []stt.Event{
		{Type: stt.SpeechStarted, Start: 500 * time.Millisecond},
		{Type: stt.Interim, Text: "hello", Language: "en-US", Confidence: 0.8, Start: 500 * time.Millisecond, End: 1500 * time.Millisecond},
		{Type: stt.Final, Text: "Hello there.", Language: "en-US", Confidence: 0.97, Start: 500 * time.Millisecond, End: 2 * time.Second},
		{Type: stt.Final, Text: "How are you?", Language: "en", Confidence: 0.9, Start: 2 * time.Second, End: 3 * time.Second},
		{Type: stt.EndOfSpeech, End: 3 * time.Second},
		{Type: stt.Final, Text: "Bye.", Language: "en-US", Confidence: 0.5, Start: 4 * time.Second, End: 5 * time.Second},
		{Type: stt.EndOfSpeech, End: 4800 * time.Millisecond},
	}
	if len(got) != len(wantEvents) {
		t.Fatalf("got %d events %+v, want %d", len(got), got, len(wantEvents))
	}
	for i := range got {
		if got[i] != wantEvents[i] {
			t.Errorf("event %d = %+v, want %+v", i, got[i], wantEvents[i])
		}
	}
	// EOF is sticky.
	if _, err := st.Recv(); err != io.EOF {
		t.Errorf("second Recv after EOF = %v", err)
	}
}

func TestSTTQueryDefaults(t *testing.T) {
	s := newTestSTT(t, "https://example.test/prefix/", func(o *STTOptions) {
		o.InterimResults = new(false)
		o.Punctuate = new(false)
		o.Endpointing = -1
	})
	got := s.listenURL(audio.Format{SampleRate: 48000, Channels: 2})
	want := "wss://example.test/prefix/v1/listen?channels=2&encoding=linear16&endpointing=false&interim_results=false&model=nova-3&punctuate=false&sample_rate=48000&vad_events=true"
	if got != want {
		t.Errorf("listenURL =\n %s\nwant\n %s", got, want)
	}
}

func TestSTTKeepAlive(t *testing.T) {
	gotKeepAlive := make(chan struct{})
	srv := wsServer(t, func(r *http.Request, c *websocket.Conn) {
		mt, data, err := c.ReadMessage()
		if err != nil {
			t.Errorf("server read: %v", err)
			return
		}
		if mt != websocket.TextMessage || string(data) != `{"type":"KeepAlive"}` {
			t.Errorf("got message type %d %q, want KeepAlive text frame", mt, data)
		}
		close(gotKeepAlive)
		drain(c)
	})
	s := newTestSTT(t, srv.URL, func(o *STTOptions) { o.KeepAliveInterval = 10 * time.Millisecond })
	st, err := s.Stream(context.Background(), stt.StreamOptions{Format: testFormat})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	select {
	case <-gotKeepAlive:
	case <-time.After(5 * time.Second):
		t.Fatal("no KeepAlive received")
	}
}

func TestSTTCloseUnblocksRecvAndPush(t *testing.T) {
	release := make(chan struct{})
	// The server never reads, so the client's send path eventually backs up.
	srv := wsServer(t, func(r *http.Request, c *websocket.Conn) { <-release })
	defer close(release)

	s := newTestSTT(t, srv.URL, nil)
	st, err := s.Stream(context.Background(), stt.StreamOptions{Format: testFormat})
	if err != nil {
		t.Fatal(err)
	}

	recvErr := make(chan error, 1)
	go func() {
		_, err := st.Recv()
		recvErr <- err
	}()
	pushErr := make(chan error, 1)
	go func() {
		big := audio.Silence(testFormat, 2*time.Second)
		for {
			if err := st.Push(big); err != nil {
				pushErr <- err
				return
			}
		}
	}()

	time.Sleep(100 * time.Millisecond)
	closed := make(chan struct{})
	go func() {
		st.Close()
		close(closed)
	}()
	for name, ch := range map[string]chan error{"Recv": recvErr, "Push": pushErr} {
		select {
		case err := <-ch:
			assertCause(t, err, providers.KindSTT, providers.ErrStreamClosed)
		case <-time.After(5 * time.Second):
			t.Fatalf("%s still blocked after Close", name)
		}
	}
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not return (goroutine leak)")
	}
	// Every method keeps failing the same way, and Close stays a no-op.
	assertCause(t, st.CloseSend(), providers.KindSTT, providers.ErrStreamClosed)
	assertCause(t, st.Push(audio.Silence(testFormat, 20*time.Millisecond)), providers.KindSTT, providers.ErrStreamClosed)
	_, err = st.Recv()
	assertCause(t, err, providers.KindSTT, providers.ErrStreamClosed)
	if err := st.Close(); err != nil {
		t.Errorf("second Close = %v", err)
	}
}

func TestSTTServerFailures(t *testing.T) {
	tests := []struct {
		name      string
		handle    func(c *websocket.Conn)
		status    int
		retryable bool
		contains  string
	}{
		{
			name: "timeout close 1011",
			handle: func(c *websocket.Conn) {
				closeNormally(c, websocket.CloseInternalServerErr, "NET-0001")
			},
			status: 1011, retryable: true, contains: "NET-0001",
		},
		{
			name: "bad audio close 1008",
			handle: func(c *websocket.Conn) {
				closeNormally(c, websocket.ClosePolicyViolation, "DATA-0000")
			},
			status: 1008, retryable: false, contains: "DATA-0000",
		},
		{
			name:   "connection dropped",
			handle: func(c *websocket.Conn) { c.UnderlyingConn().Close() },
			status: websocket.CloseAbnormalClosure, retryable: true,
		},
		{
			name: "error message",
			handle: func(c *websocket.Conn) {
				c.WriteMessage(websocket.TextMessage, []byte(`{"type":"Error","variant":"x","description":"boom"}`))
				drain(c)
			},
			status: 0, retryable: false, contains: "boom",
		},
		{
			name: "error message echoing the key",
			handle: func(c *websocket.Conn) {
				c.WriteMessage(websocket.TextMessage, []byte(`{"type":"Error","description":"bad token `+testKey+`"}`))
				drain(c)
			},
			status: 0, retryable: false, contains: "bad token [REDACTED]",
		},
		{
			name: "close reason echoing the key",
			handle: func(c *websocket.Conn) {
				closeNormally(c, websocket.CloseInternalServerErr, "denied "+testKey)
			},
			status: 1011, retryable: true, contains: "denied [REDACTED]",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := wsServer(t, func(r *http.Request, c *websocket.Conn) {
				// One result before the failure must still be delivered.
				c.WriteMessage(websocket.TextMessage, []byte(`{"type":"Results","start":0,"duration":1,"is_final":true,"channel":{"alternatives":[{"transcript":"hi","confidence":1}]}}`))
				tc.handle(c)
			})
			s := newTestSTT(t, srv.URL, nil)
			st, err := s.Stream(context.Background(), stt.StreamOptions{Format: testFormat})
			if err != nil {
				t.Fatal(err)
			}
			defer st.Close()
			ev, err := st.Recv()
			if err != nil || ev.Type != stt.Final || ev.Text != "hi" {
				t.Fatalf("first Recv = %+v, %v", ev, err)
			}
			_, err = st.Recv()
			assertProviderError(t, err, providers.KindSTT, "recv", tc.status, tc.retryable)
			if tc.contains != "" && !strings.Contains(err.Error(), tc.contains) {
				t.Errorf("error %q does not mention %q", err, tc.contains)
			}
			// Push fails promptly once the stream is dead.
			deadline := time.After(5 * time.Second)
			for {
				if err := st.Push(audio.Silence(testFormat, 20*time.Millisecond)); err != nil {
					assertProviderError(t, err, providers.KindSTT, "recv", tc.status, tc.retryable)
					break
				}
				select {
				case <-deadline:
					t.Fatal("Push keeps succeeding on a failed stream")
				default:
				}
			}
		})
	}
}

func TestSTTHandshakeErrors(t *testing.T) {
	tests := []struct {
		status    int
		retryable bool
	}{
		{http.StatusUnauthorized, false},
		{http.StatusForbidden, false},
		{http.StatusBadRequest, false},
		{http.StatusTooManyRequests, true},
		{http.StatusServiceUnavailable, true},
	}
	for _, tc := range tests {
		t.Run(http.StatusText(tc.status), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// A hostile or sloppy server echoing the credential.
				w.Header().Set("dg-error", "Invalid credentials. "+strings.TrimPrefix(r.Header.Get("Authorization"), "Token "))
				w.WriteHeader(tc.status)
			}))
			defer srv.Close()
			s := newTestSTT(t, srv.URL, nil)
			_, err := s.Stream(context.Background(), stt.StreamOptions{Format: testFormat})
			assertProviderError(t, err, providers.KindSTT, "connect", tc.status, tc.retryable)
			if !strings.Contains(err.Error(), "Invalid credentials") {
				t.Errorf("error %q does not carry dg-error", err)
			}
		})
	}
}

func TestSTTConnectNetworkError(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close() // nothing listens any more
	s := newTestSTT(t, url, nil)
	_, err := s.Stream(context.Background(), stt.StreamOptions{Format: testFormat})
	assertProviderError(t, err, providers.KindSTT, "connect", 0, true)
}

func TestSTTContextCancel(t *testing.T) {
	srv := wsServer(t, func(r *http.Request, c *websocket.Conn) { drain(c) })
	s := newTestSTT(t, srv.URL, nil)

	t.Run("before dial", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := s.Stream(ctx, stt.StreamOptions{Format: testFormat})
		assertCause(t, err, providers.KindSTT, context.Canceled)
	})

	t.Run("deadline during stream", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()
		st, err := s.Stream(ctx, stt.StreamOptions{Format: testFormat})
		if err != nil {
			t.Fatal(err)
		}
		defer st.Close()
		_, err = st.Recv()
		assertCause(t, err, providers.KindSTT, context.DeadlineExceeded)
	})

	t.Run("during stream", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		st, err := s.Stream(ctx, stt.StreamOptions{Format: testFormat})
		if err != nil {
			t.Fatal(err)
		}
		defer st.Close()
		recvErr := make(chan error, 1)
		go func() {
			_, err := st.Recv()
			recvErr <- err
		}()
		time.Sleep(20 * time.Millisecond)
		cancel()
		select {
		case err := <-recvErr:
			assertCause(t, err, providers.KindSTT, context.Canceled)
		case <-time.After(5 * time.Second):
			t.Fatal("Recv still blocked after cancel")
		}
		assertCause(t, st.Push(audio.Silence(testFormat, 20*time.Millisecond)), providers.KindSTT, context.Canceled)
	})
}
