package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/providers"
	"github.com/maktoufzied/livekit-agents-go/tts"
)

// speakServer is a fake /v1/speak WebSocket: it answers every Speak with
// 20 ms of audio per character and every Flush with Flushed.
type speakServer struct {
	mu       sync.Mutex
	conns    int
	requests []*http.Request
	messages [][]string // per connection, the "type:text" of each message
	closes   atomic.Int32
}

func (s *speakServer) handle(r *http.Request, c *websocket.Conn) {
	s.mu.Lock()
	id := s.conns
	s.conns++
	s.requests = append(s.requests, r)
	s.messages = append(s.messages, nil)
	s.mu.Unlock()
	for {
		_, data, err := c.ReadMessage()
		if err != nil {
			return
		}
		var msg struct{ Type, Text string }
		if json.Unmarshal(data, &msg) != nil {
			continue
		}
		s.mu.Lock()
		s.messages[id] = append(s.messages[id], msg.Type+":"+msg.Text)
		s.mu.Unlock()
		switch msg.Type {
		case "Speak":
			// 16 kHz mono: 640 bytes are 20 ms.
			_ = c.WriteMessage(websocket.BinaryMessage, make([]byte, 640*len(msg.Text)))
		case "Flush":
			_ = c.WriteMessage(websocket.TextMessage, []byte(`{"type":"Metadata","request_id":"r"}`))
			_ = c.WriteMessage(websocket.TextMessage, []byte(`{"type":"Flushed","sequence_id":0}`))
		case "Close":
			s.closes.Add(1)
			return
		}
	}
}

func (s *speakServer) log(conn int) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if conn >= len(s.messages) {
		return nil
	}
	return append([]string(nil), s.messages[conn]...)
}

func (s *speakServer) connections() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conns
}

func drainStream(t *testing.T, st tts.Stream) time.Duration {
	t.Helper()
	var total time.Duration
	for {
		f, err := st.Recv()
		if errors.Is(err, io.EOF) {
			return total
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if f.SampleRate != 16000 || f.Channels != 1 {
			t.Fatalf("frame format = %v", f.Format())
		}
		total += f.Duration()
	}
}

func newStreamTTS(t *testing.T, srv *speakServer) *TTS {
	t.Helper()
	ws := wsServer(t, srv.handle)
	voice := newTestTTS(t, ws.URL, func(o *TTSOptions) { o.SampleRate = 16000 })
	t.Cleanup(func() { voice.Close() })
	return voice
}

func TestTTSStream(t *testing.T) {
	srv := &speakServer{}
	voice := newStreamTTS(t, srv)
	ctx := context.Background()

	st, err := voice.SynthesizeStream(ctx)
	if err != nil {
		t.Fatal(err)
	}
	// Text cut in the middle of words, as a model produces it.
	for _, chunk := range []string{"Hel", "lo  the", "re.\nHow are", " you", "?"} {
		if err := st.Push(chunk); err != nil {
			t.Fatal(err)
		}
	}
	if err := st.Flush(); err != nil {
		t.Fatal(err)
	}
	if err := st.Push("more"); err == nil {
		t.Fatal("no error for text pushed after Flush")
	}
	got := drainStream(t, st)
	// "Hello " "there. " "How " "are " "you? ": 26 characters.
	if want := 26 * 20 * time.Millisecond; got != want {
		t.Fatalf("audio = %v, want %v", got, want)
	}
	if err := st.Close(); err != nil {
		t.Fatal(err)
	}

	want := []string{"Speak:Hello ", "Speak:there. ", "Speak:How ", "Speak:are ", "Speak:you? ", "Flush:"}
	if log := srv.log(0); strings.Join(log, "|") != strings.Join(want, "|") {
		t.Fatalf("messages = %q, want %q", log, want)
	}
	r := srv.requests[0]
	if r.URL.Path != "/v1/speak" || r.Header.Get("Authorization") != "Token "+testKey {
		t.Fatalf("request: %s, authorization %q", r.URL.Path, r.Header.Get("Authorization"))
	}
	q := r.URL.Query()
	if q.Get("model") != DefaultTTSModel || q.Get("encoding") != "linear16" || q.Get("sample_rate") != "16000" || q.Has("container") {
		t.Fatalf("query: %v", q)
	}

	// The connection is reused by the next stream, and by the one after.
	for i := 0; i < 2; i++ {
		st, err := voice.SynthesizeStream(ctx)
		if err != nil {
			t.Fatal(err)
		}
		_ = st.Push("Again.")
		_ = st.Flush()
		if got := drainStream(t, st); got != 7*20*time.Millisecond {
			t.Fatalf("audio = %v", got)
		}
		st.Close()
	}
	if n := srv.connections(); n != 1 {
		t.Fatalf("%d connections for three streams, want 1", n)
	}
}

// A stream closed before the end gives its connection up, after telling the
// server, and the next stream connects again.
func TestTTSStreamCloseBeforeTheEnd(t *testing.T) {
	srv := &speakServer{}
	voice := newStreamTTS(t, srv)
	st, err := voice.SynthesizeStream(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	_ = st.Push("Some words that will not be heard. ")
	if _, err := st.Recv(); err != nil {
		t.Fatal(err)
	}
	st.Close()
	st.Close() // idempotent
	if _, err := st.Recv(); !errors.Is(err, providers.ErrStreamClosed) {
		t.Fatalf("Recv after Close = %v", err)
	}
	if err := st.Push("x "); !errors.Is(err, providers.ErrStreamClosed) {
		t.Fatalf("Push after Close = %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for srv.closes.Load() != 1 {
		if time.Now().After(deadline) {
			t.Fatal("the server was not told that the session ended")
		}
		time.Sleep(5 * time.Millisecond)
	}

	st2, err := voice.SynthesizeStream(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	_ = st2.Push("Next.")
	_ = st2.Flush()
	drainStream(t, st2)
	st2.Close()
	if n := srv.connections(); n != 2 {
		t.Fatalf("%d connections, want 2", n)
	}
}

func TestTTSStreamPrewarm(t *testing.T) {
	srv := &speakServer{}
	voice := newStreamTTS(t, srv)
	voice.Prewarm()
	deadline := time.Now().Add(2 * time.Second)
	for voice.pool.size() != 1 {
		if time.Now().After(deadline) {
			t.Fatal("no connection opened ahead")
		}
		time.Sleep(5 * time.Millisecond)
	}
	voice.Prewarm() // one is idle: nothing to do
	st, err := voice.SynthesizeStream(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	_ = st.Push("Hi.")
	_ = st.Flush()
	drainStream(t, st)
	st.Close()
	if n := srv.connections(); n != 1 {
		t.Fatalf("%d connections, want the one opened ahead", n)
	}
}

func TestTTSStreamContextCancel(t *testing.T) {
	srv := &speakServer{}
	voice := newStreamTTS(t, srv)
	ctx, cancel := context.WithCancel(context.Background())
	st, err := voice.SynthesizeStream(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	done := make(chan error, 1)
	go func() {
		_, err := st.Recv() // nothing was pushed: it waits
		done <- err
	}()
	time.Sleep(20 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Recv = %v, want the context error", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Recv not unblocked by the context")
	}
	if _, err := voice.SynthesizeStream(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("SynthesizeStream with a cancelled context = %v", err)
	}
}

func TestTTSStreamServerFailures(t *testing.T) {
	t.Run("error message", func(t *testing.T) {
		ws := wsServer(t, func(_ *http.Request, c *websocket.Conn) {
			_, _, _ = c.ReadMessage()
			_ = c.WriteMessage(websocket.TextMessage, []byte(`{"type":"Error","description":"bad voice `+testKey+`"}`))
			_, _, _ = c.ReadMessage()
		})
		voice := newTestTTS(t, ws.URL, nil)
		defer voice.Close()
		st, err := voice.SynthesizeStream(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		defer st.Close()
		_ = st.Push("Hi. ")
		_, err = st.Recv()
		var pe *providers.Error
		if !errors.As(err, &pe) || pe.Retryable || !strings.Contains(err.Error(), "bad voice") {
			t.Fatalf("Recv = %v", err)
		}
		if strings.Contains(err.Error(), testKey) {
			t.Fatal("the error contains the API key")
		}
		if voice.pool.size() != 0 {
			t.Fatal("a failed connection went back to the pool")
		}
	})
	t.Run("connection dropped", func(t *testing.T) {
		ws := wsServer(t, func(_ *http.Request, c *websocket.Conn) {
			_, _, _ = c.ReadMessage()
			c.Close()
		})
		voice := newTestTTS(t, ws.URL, nil)
		defer voice.Close()
		st, err := voice.SynthesizeStream(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		defer st.Close()
		_ = st.Push("Hi. ")
		_, err = st.Recv()
		var pe *providers.Error
		if !errors.As(err, &pe) || !pe.Retryable {
			t.Fatalf("Recv = %v, want a retryable error", err)
		}
	})
	t.Run("handshake refused", func(t *testing.T) {
		voice := newTestTTS(t, "http://127.0.0.1:1", nil)
		defer voice.Close()
		_, err := voice.SynthesizeStream(context.Background())
		var pe *providers.Error
		if !errors.As(err, &pe) || !pe.Retryable {
			t.Fatalf("SynthesizeStream = %v, want a retryable error", err)
		}
	})
}

// The frames keep the configured duration whatever the size of the
// messages of the server.
func TestTTSStreamFrameDuration(t *testing.T) {
	srv := &speakServer{}
	voice := newStreamTTS(t, srv)
	st, err := voice.SynthesizeStream(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	_ = st.Push("abc ")
	_ = st.Flush()
	var frames []audio.Frame
	for f, err := range tts.Frames(st) {
		if err != nil {
			t.Fatal(err)
		}
		frames = append(frames, f)
	}
	if len(frames) != 4 {
		t.Fatalf("%d frames, want 4 of 20 ms", len(frames))
	}
	for _, f := range frames {
		if f.Duration() != 20*time.Millisecond {
			t.Fatalf("frame of %v", f.Duration())
		}
	}
}
