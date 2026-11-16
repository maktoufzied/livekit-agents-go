package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"

	"github.com/gorilla/websocket"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/providers"
	"github.com/maktoufzied/livekit-agents-go/tts"
)

// The streaming path follows the upstream Python plugin
// (livekit-plugins-deepgram 1.8.12): one WebSocket per stream taken from a
// pool, the text sent word by word, one Flush at the end of the text.
const (
	// ttsMaxSession is how long a pooled connection is reused.
	ttsMaxSession = time.Hour
	// ttsRecvTimeout bounds the wait for one message of the server.
	ttsRecvTimeout = 10 * time.Second
	// ttsCloseWait is how long a closing connection waits for the server
	// to acknowledge.
	ttsCloseWait = time.Second

	ttsFlushMsg = `{"type":"Flush"}`
	ttsCloseMsg = `{"type":"Close"}`
)

var (
	_ tts.Streamer  = (*TTS)(nil)
	_ tts.Prewarmer = (*TTS)(nil)
)

// ttsConn is one WebSocket of the pool.
type ttsConn struct {
	ws   *websocket.Conn
	born time.Time
	wmu  sync.Mutex // serializes writes
	once sync.Once
}

func (c *ttsConn) write(msg string) error {
	c.wmu.Lock()
	defer c.wmu.Unlock()
	_ = c.ws.SetWriteDeadline(time.Now().Add(ttsRecvTimeout))
	return c.ws.WriteMessage(websocket.TextMessage, []byte(msg))
}

// shutdown ends the session on the server and closes the socket. Deepgram
// counts sessions that linger, so the server is told and given a moment to
// close its side before the socket goes. A Recv in progress returns at
// once. It does not block the caller.
func (c *ttsConn) shutdown() {
	c.once.Do(func() {
		go func() {
			told := c.write(ttsFlushMsg) == nil && c.write(ttsCloseMsg) == nil
			// Only the stream reads from the socket; the deadline of the
			// underlying connection is the one thing safe to touch here.
			_ = c.ws.NetConn().SetReadDeadline(time.Now())
			if told {
				time.Sleep(ttsCloseWait)
			}
			c.ws.Close()
		}()
	})
}

// ttsPool keeps idle connections for reuse.
type ttsPool struct {
	mu     sync.Mutex
	idle   []*ttsConn
	closed bool
}

// get returns an idle connection, or nil.
func (p *ttsPool) get() *ttsConn {
	p.mu.Lock()
	defer p.mu.Unlock()
	for len(p.idle) > 0 {
		c := p.idle[len(p.idle)-1]
		p.idle = p.idle[:len(p.idle)-1]
		if time.Since(c.born) < ttsMaxSession {
			return c
		}
		c.shutdown()
	}
	return nil
}

func (p *ttsPool) put(c *ttsConn) {
	p.mu.Lock()
	if p.closed || time.Since(c.born) >= ttsMaxSession {
		p.mu.Unlock()
		c.shutdown()
		return
	}
	p.idle = append(p.idle, c)
	p.mu.Unlock()
}

func (p *ttsPool) size() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.idle)
}

func (p *ttsPool) close() {
	p.mu.Lock()
	idle := p.idle
	p.idle, p.closed = nil, true
	p.mu.Unlock()
	for _, c := range idle {
		c.shutdown()
	}
}

func (t *TTS) dial(ctx context.Context) (*ttsConn, error) {
	h := http.Header{}
	for k, v := range t.opts.Header {
		h[k] = append([]string(nil), v...)
	}
	h.Set("Authorization", "Token "+t.opts.APIKey.Reveal())
	ws, resp, err := t.dialer.DialContext(ctx, t.wsURL, h)
	if err != nil {
		if resp != nil && resp.Body != nil {
			resp.Body.Close()
		}
		if ctx.Err() != nil {
			return nil, ctxError(ctx, providers.KindTTS, "connect")
		}
		if resp == nil {
			return nil, t.newError("connect", 0, true, err)
		}
		if msg := resp.Header.Get("dg-error"); msg != "" {
			err = errors.New(msg)
		}
		return nil, t.newError("connect", resp.StatusCode, providers.RetryableStatus(resp.StatusCode), err)
	}
	return &ttsConn{ws: ws, born: time.Now()}, nil
}

// Prewarm opens a connection in the background, so that the first stream
// does not wait for it. It does nothing when one is idle already.
func (t *TTS) Prewarm() {
	if t.pool.size() > 0 {
		return
	}
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), ttsRecvTimeout)
		defer cancel()
		if c, err := t.dial(ctx); err == nil {
			t.pool.put(c)
		}
	}()
}

// Close closes the pooled connections. Streams in use are not affected,
// and the TTS must not be used afterwards.
func (t *TTS) Close() error {
	t.pool.close()
	return nil
}

// SynthesizeStream opens a stream fed with text. It uses a pooled
// connection when one is idle, and connects otherwise. ctx bounds the whole
// stream.
func (t *TTS) SynthesizeStream(ctx context.Context) (tts.InputStream, error) {
	if err := ctx.Err(); err != nil {
		return nil, ctxError(ctx, providers.KindTTS, "connect")
	}
	conn := t.pool.get()
	if conn == nil {
		var err error
		if conn, err = t.dial(ctx); err != nil {
			return nil, err
		}
	}
	st := &ttsInputStream{t: t, ctx: ctx, conn: conn, chunker: audio.NewChunker(t.format, t.opts.FrameDuration)}
	st.stop = context.AfterFunc(ctx, st.abort)
	return st, nil
}

// ttsInputStream is one use of a connection: Speak messages, one Flush,
// audio until the server confirms the flush.
type ttsInputStream struct {
	t    *TTS
	ctx  context.Context
	conn *ttsConn
	stop func() bool

	closed   atomic.Bool
	released sync.Once // the connection went back to the pool or was shut down

	// Push and Flush state.
	word    strings.Builder // a word not yet complete
	spoke   bool            // at least one Speak was sent
	flushed bool

	// Recv state.
	chunker *audio.Chunker
	pending []audio.Frame
	eof     bool
	err     error
}

// Push sends the complete words of text; the end of the last word is known
// when more text or Flush follows.
func (s *ttsInputStream) Push(text string) error {
	if s.closed.Load() {
		return closedError(providers.KindTTS, "push")
	}
	if s.flushed {
		return s.t.newError("push", 0, false, errors.New("text pushed after Flush"))
	}
	for _, r := range text {
		if !unicode.IsSpace(r) {
			s.word.WriteRune(r)
			continue
		}
		if err := s.speak(); err != nil {
			return err
		}
	}
	return nil
}

// speak sends the pending word, if any.
func (s *ttsInputStream) speak() error {
	if s.word.Len() == 0 {
		return nil
	}
	msg, err := json.Marshal(struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}{"Speak", s.word.String() + " "})
	s.word.Reset()
	if err != nil {
		return s.t.newError("push", 0, false, err)
	}
	if err := s.conn.write(string(msg)); err != nil {
		return s.writeError("push", err)
	}
	s.spoke = true
	return nil
}

// Flush sends the last word and asks for the audio of everything sent.
func (s *ttsInputStream) Flush() error {
	if s.closed.Load() {
		return closedError(providers.KindTTS, "flush")
	}
	if s.flushed {
		return nil
	}
	s.flushed = true
	if err := s.speak(); err != nil {
		return err
	}
	if err := s.conn.write(ttsFlushMsg); err != nil {
		return s.writeError("flush", err)
	}
	return nil
}

func (s *ttsInputStream) writeError(op string, err error) error {
	switch {
	case s.closed.Load():
		return closedError(providers.KindTTS, op)
	case s.ctx.Err() != nil:
		return ctxError(s.ctx, providers.KindTTS, op)
	}
	return s.t.newError(op, 0, true, err)
}

// Recv returns the next frame, io.EOF once the flush is confirmed.
func (s *ttsInputStream) Recv() (audio.Frame, error) {
	for {
		if s.closed.Load() {
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
		_ = s.conn.ws.SetReadDeadline(time.Now().Add(ttsRecvTimeout))
		kind, data, err := s.conn.ws.ReadMessage()
		if err != nil {
			s.fail(s.readError(err))
			continue
		}
		if kind == websocket.BinaryMessage {
			s.pending = s.chunker.Write(data)
			continue
		}
		var msg struct {
			Type        string `json:"type"`
			Description string `json:"description"`
			ErrMsg      string `json:"err_msg"`
			WarnMsg     string `json:"warn_msg"`
		}
		if json.Unmarshal(data, &msg) != nil {
			continue
		}
		switch msg.Type {
		case "Flushed":
			s.eof = true
			if f, ok := s.chunker.Flush(); ok {
				s.pending = append(s.pending, f)
			}
			// The connection is clean again: the next stream may use it.
			s.release(true)
		case "Error", "error":
			text := msg.Description
			if text == "" {
				text = msg.ErrMsg
			}
			if text == "" {
				text = "server error"
			}
			s.fail(s.t.newError("recv", 0, false, errors.New(text)))
		}
		// Metadata and Warning carry nothing the caller acts on.
	}
}

func (s *ttsInputStream) readError(err error) error {
	switch {
	case s.closed.Load():
		return closedError(providers.KindTTS, "recv")
	case s.ctx.Err() != nil:
		return ctxError(s.ctx, providers.KindTTS, "recv")
	}
	var ce *websocket.CloseError
	if errors.As(err, &ce) {
		return s.t.newError("recv", ce.Code, true, fmt.Errorf("connection closed by the server: %w", err))
	}
	return s.t.newError("recv", 0, true, err)
}

// fail records a terminal error and gives the connection up.
func (s *ttsInputStream) fail(err error) {
	s.err = err
	s.release(false)
}

// release returns the connection to the pool when it is reusable, and
// shuts it down otherwise.
func (s *ttsInputStream) release(reusable bool) {
	s.released.Do(func() {
		s.stop()
		if reusable {
			s.t.pool.put(s.conn)
		} else {
			s.conn.shutdown()
		}
	})
}

// abort is run when the context ends: it unblocks Recv.
func (s *ttsInputStream) abort() {
	s.released.Do(func() { s.conn.shutdown() })
}

// Close aborts the stream and unblocks Recv. A stream closed before the
// server confirmed the flush gives its connection up: audio may still be
// on its way.
func (s *ttsInputStream) Close() error {
	if s.closed.CompareAndSwap(false, true) {
		s.release(false)
	}
	return nil
}
