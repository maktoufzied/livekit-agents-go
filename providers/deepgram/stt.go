package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/providers"
	"github.com/maktoufzied/livekit-agents-go/stt"
)

const (
	listenPath = "/v1/listen"
	// handshakeTimeout bounds the WebSocket handshake, like
	// websocket.DefaultDialer does.
	handshakeTimeout = 45 * time.Second

	// Deepgram closes a connection that receives neither audio nor a
	// KeepAlive for 10 seconds and recommends a KeepAlive every 3-5 seconds.
	defaultKeepAlive = 4 * time.Second
	writeTimeout     = 10 * time.Second
	// sendQueueFrames bounds the audio waiting to be written (about 1.3 s
	// of 20 ms frames); Push blocks once it is full.
	sendQueueFrames = 64
	eventQueueSize  = 64
)

// Client control messages, sent as text frames.
const (
	msgKeepAlive   = `{"type":"KeepAlive"}`
	msgCloseStream = `{"type":"CloseStream"}`
)

// STTOptions configures the Deepgram streaming recognizer.
type STTOptions struct {
	// APIKey authenticates the requests. It defaults to the
	// DEEPGRAM_API_KEY environment variable.
	APIKey providers.Secret
	// Model is the transcription model. It defaults to DefaultSTTModel.
	Model string
	// Language is a BCP-47 tag such as "en-US". It defaults to empty, which
	// leaves the choice to Deepgram (English).
	Language string
	// BaseURL is the bare origin of the API (scheme and host, plus an
	// optional path prefix of a proxy), without the /v1/listen path. It
	// defaults to DefaultBaseURL. The http and https schemes are mapped to
	// ws and wss; ws and wss are accepted as they are.
	BaseURL string
	// InterimResults enables stt.Interim events. It defaults to true (nil);
	// pass new(false) to disable.
	InterimResults *bool
	// Punctuate adds punctuation and capitalization. It defaults to true
	// (nil).
	Punctuate *bool
	// SmartFormat enables Deepgram's smart formatting. It defaults to
	// false.
	SmartFormat bool
	// Endpointing is the silence after which Deepgram marks a result
	// speech_final (reported as stt.EndOfSpeech). It defaults to zero,
	// which keeps Deepgram's default (10 ms); a negative value disables
	// endpointing. It is sent in whole milliseconds, so a positive value
	// must be at least one millisecond.
	Endpointing time.Duration
	// UtteranceEnd, when positive, asks for UtteranceEnd messages after
	// that gap between words (also reported as stt.EndOfSpeech). It
	// defaults to zero, which does not ask for them. Deepgram requires
	// interim results and recommends one second or more. It is sent in
	// whole milliseconds, so it must be at least one millisecond.
	UtteranceEnd time.Duration
	// KeepAliveInterval is how often a KeepAlive is sent while no audio is
	// flowing. It defaults to 4 seconds.
	KeepAliveInterval time.Duration
	// Header is added to the WebSocket handshake request. It defaults to
	// none.
	Header http.Header
	// NetDialContext opens the TCP connection of the WebSocket; TLS is
	// added on top of it for wss. It defaults to nil: a direct connection
	// made with a net.Dialer, through the proxy named by the environment
	// (HTTP_PROXY and friends) if there is one.
	NetDialContext func(ctx context.Context, network, addr string) (net.Conn, error)
}

// STT is a Deepgram streaming speech-to-text client. It is safe for
// concurrent use.
type STT struct {
	opts    STTOptions
	interim bool
	punct   bool
	base    *url.URL
	dialer  *websocket.Dialer
}

var _ stt.STT = (*STT)(nil)

// NewSTT validates opts and returns a recognizer. It fails when no API key
// is configured or an option is invalid.
func NewSTT(opts STTOptions) (*STT, error) {
	key, err := resolveKey(opts.APIKey, "STTOptions.APIKey")
	if err != nil {
		return nil, err
	}
	opts.APIKey = key
	if opts.Model == "" {
		opts.Model = DefaultSTTModel
	}
	if opts.BaseURL == "" {
		opts.BaseURL = DefaultBaseURL
	}
	if opts.KeepAliveInterval <= 0 {
		opts.KeepAliveInterval = defaultKeepAlive
	}
	s := &STT{opts: opts, interim: true, punct: true}
	// Apart from NetDialContext this is websocket.DefaultDialer.
	s.dialer = &websocket.Dialer{
		Proxy:            http.ProxyFromEnvironment,
		HandshakeTimeout: handshakeTimeout,
		NetDialContext:   opts.NetDialContext,
	}
	if opts.InterimResults != nil {
		s.interim = *opts.InterimResults
	}
	if opts.Punctuate != nil {
		s.punct = *opts.Punctuate
	}
	if opts.Endpointing > 0 && opts.Endpointing < time.Millisecond {
		return nil, fmt.Errorf("deepgram: Endpointing %v is shorter than one millisecond (it is a time.Duration, e.g. 300*time.Millisecond)", opts.Endpointing)
	}
	if opts.UtteranceEnd > 0 && opts.UtteranceEnd < time.Millisecond {
		return nil, fmt.Errorf("deepgram: UtteranceEnd %v is shorter than one millisecond (it is a time.Duration, e.g. time.Second)", opts.UtteranceEnd)
	}
	if opts.UtteranceEnd > 0 && !s.interim {
		return nil, errors.New("deepgram: UtteranceEnd requires InterimResults")
	}
	u, err := url.Parse(opts.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("deepgram: invalid BaseURL: %w", err)
	}
	switch u.Scheme {
	case "http":
		u.Scheme = "ws"
	case "https":
		u.Scheme = "wss"
	case "ws", "wss":
	default:
		return nil, fmt.Errorf("deepgram: unsupported BaseURL scheme %q", u.Scheme)
	}
	u.Path = strings.TrimSuffix(u.Path, "/") + listenPath
	s.base = u
	return s, nil
}

// listenURL builds the handshake URL for a stream of the given format.
func (s *STT) listenURL(f audio.Format) string {
	q := url.Values{}
	q.Set("model", s.opts.Model)
	if s.opts.Language != "" {
		q.Set("language", s.opts.Language)
	}
	q.Set("encoding", "linear16")
	q.Set("sample_rate", strconv.Itoa(f.SampleRate))
	q.Set("channels", strconv.Itoa(f.Channels))
	q.Set("interim_results", strconv.FormatBool(s.interim))
	q.Set("punctuate", strconv.FormatBool(s.punct))
	if s.opts.SmartFormat {
		q.Set("smart_format", "true")
	}
	switch {
	case s.opts.Endpointing > 0:
		q.Set("endpointing", strconv.FormatInt(s.opts.Endpointing.Milliseconds(), 10))
	case s.opts.Endpointing < 0:
		q.Set("endpointing", "false")
	}
	if s.opts.UtteranceEnd > 0 {
		q.Set("utterance_end_ms", strconv.FormatInt(s.opts.UtteranceEnd.Milliseconds(), 10))
	}
	q.Set("vad_events", "true")
	u := *s.base
	u.RawQuery = q.Encode()
	return u.String()
}

// Stream opens a live transcription session. ctx bounds the whole stream.
func (s *STT) Stream(ctx context.Context, opts stt.StreamOptions) (stt.Stream, error) {
	key := s.opts.APIKey
	if !opts.Format.Valid() {
		return nil, newError(providers.KindSTT, "connect", 0, false, fmt.Errorf("invalid stream format %+v", opts.Format), key)
	}
	h := http.Header{}
	for k, v := range s.opts.Header {
		h[k] = append([]string(nil), v...)
	}
	h.Set("Authorization", "Token "+key.Reveal())

	conn, resp, err := s.dialer.DialContext(ctx, s.listenURL(opts.Format), h)
	if err != nil {
		if resp != nil && resp.Body != nil {
			resp.Body.Close()
		}
		if ctx.Err() != nil {
			return nil, ctxError(ctx, providers.KindSTT, "connect")
		}
		if resp == nil {
			return nil, newError(providers.KindSTT, "connect", 0, true, err, key)
		}
		// Failed upgrades carry the reason in the dg-error header.
		if msg := resp.Header.Get("dg-error"); msg != "" {
			err = errors.New(msg)
		}
		return nil, newError(providers.KindSTT, "connect", resp.StatusCode, providers.RetryableStatus(resp.StatusCode), err, key)
	}

	st := &sttStream{
		ctx:       ctx,
		key:       key,
		conn:      conn,
		format:    opts.Format,
		language:  s.opts.Language,
		keepAlive: s.opts.KeepAliveInterval,
		sendq:     make(chan []byte, sendQueueFrames),
		endCh:     make(chan struct{}),
		events:    make(chan stt.Event, eventQueueSize),
		done:      make(chan struct{}),
		closed:    make(chan struct{}),
	}
	st.wg.Go(st.readLoop)
	st.wg.Go(st.writeLoop)
	return st, nil
}

// sttStream is one /v1/listen session. readLoop owns reads and the events
// channel; writeLoop owns every write and watches the context.
type sttStream struct {
	ctx       context.Context
	key       providers.Secret // only used to scrub errors
	conn      *websocket.Conn
	format    audio.Format
	language  string
	keepAlive time.Duration

	sendq chan []byte
	endCh chan struct{} // closed by CloseSend
	ended atomic.Bool
	// gotMetadata is set when the summary Metadata message that follows
	// CloseStream has been received.
	gotMetadata bool

	events chan stt.Event // closed by readLoop after err is set

	failOnce sync.Once
	err      error         // terminal error, valid once done is closed
	done     chan struct{} // closed when the stream has failed or ended

	closeOnce sync.Once
	endOnce   sync.Once
	closed    chan struct{} // closed by Close
	wg        sync.WaitGroup

	// speechOpen is true between the first sign of speech and the
	// EndOfSpeech that follows it; it suppresses duplicate EndOfSpeech
	// events. Only readLoop touches it.
	speechOpen bool
}

// fail records the terminal error; only the first call wins.
func (s *sttStream) fail(err error) {
	s.failOnce.Do(func() {
		s.err = err
		close(s.done)
	})
}

func (s *sttStream) isClosed() bool {
	select {
	case <-s.closed:
		return true
	default:
		return false
	}
}

// Push queues one frame for sending. It blocks while the queue is full and
// returns once the stream is closed, cancelled or failed.
func (s *sttStream) Push(frame audio.Frame) error {
	if s.isClosed() {
		return closedError(providers.KindSTT, "send")
	}
	if s.ended.Load() {
		return s.newError("send", 0, false, errors.New("Push after CloseSend"))
	}
	if len(frame.Data) == 0 {
		return nil
	}
	if frame.Format() != s.format {
		return s.newError("send", 0, false, fmt.Errorf("frame format %+v does not match stream format %+v", frame.Format(), s.format))
	}
	select {
	case <-s.done:
		return s.err
	default:
	}
	select {
	case s.sendq <- frame.BytesLE():
		return nil
	case <-s.closed:
		return closedError(providers.KindSTT, "send")
	case <-s.done:
		return s.err
	case <-s.ctx.Done():
		return ctxError(s.ctx, providers.KindSTT, "send")
	}
}

func (s *sttStream) newError(op string, status int, retryable bool, err error) *providers.Error {
	return newError(providers.KindSTT, op, status, retryable, err, s.key)
}

// CloseSend asks Deepgram to flush pending audio and close the stream; Recv
// returns io.EOF once the remaining results are drained.
func (s *sttStream) CloseSend() error {
	if s.isClosed() {
		return closedError(providers.KindSTT, "send")
	}
	s.endOnce.Do(func() {
		s.ended.Store(true)
		close(s.endCh)
	})
	return nil
}

// Recv returns the next event in provider order.
func (s *sttStream) Recv() (stt.Event, error) {
	if s.isClosed() {
		return stt.Event{}, closedError(providers.KindSTT, "recv")
	}
	select {
	case ev, ok := <-s.events:
		if !ok {
			return stt.Event{}, s.err
		}
		return ev, nil
	case <-s.closed:
		return stt.Event{}, closedError(providers.KindSTT, "recv")
	}
}

// Close releases the connection, unblocks Push and Recv and waits for the
// stream's goroutines to exit.
func (s *sttStream) Close() error {
	s.closeOnce.Do(func() {
		close(s.closed)
		s.conn.Close()
		s.wg.Wait()
	})
	return nil
}

func (s *sttStream) write(messageType int, data []byte) error {
	s.conn.SetWriteDeadline(time.Now().Add(writeTimeout))
	return s.conn.WriteMessage(messageType, data)
}

// writeLoop sends audio, KeepAlive and CloseStream, and tears the connection
// down when the context is cancelled.
func (s *sttStream) writeLoop() {
	ticker := time.NewTicker(s.keepAlive)
	defer ticker.Stop()
	endCh := s.endCh
	lastAudio := time.Now()

	writeFailed := func(err error) {
		switch {
		case s.isClosed():
			s.fail(closedError(providers.KindSTT, "send"))
		case s.ctx.Err() != nil:
			s.fail(ctxError(s.ctx, providers.KindSTT, "send"))
		default:
			s.fail(s.newError("send", 0, true, err))
		}
		s.conn.Close()
	}

	for {
		select {
		case <-s.closed:
			return
		case <-s.done:
			return
		case <-s.ctx.Done():
			s.fail(ctxError(s.ctx, providers.KindSTT, "recv"))
			s.conn.Close() // unblocks readLoop
			return
		case b := <-s.sendq:
			if err := s.write(websocket.BinaryMessage, b); err != nil {
				writeFailed(err)
				return
			}
			lastAudio = time.Now()
		case <-endCh:
			endCh = nil
			// Push and CloseSend come from one goroutine, so everything
			// pushed is already queued: flush it before CloseStream.
			for drained := false; !drained; {
				select {
				case b := <-s.sendq:
					if err := s.write(websocket.BinaryMessage, b); err != nil {
						writeFailed(err)
						return
					}
				default:
					drained = true
				}
			}
			if err := s.write(websocket.TextMessage, []byte(msgCloseStream)); err != nil {
				writeFailed(err)
				return
			}
		case <-ticker.C:
			if endCh == nil || time.Since(lastAudio) < s.keepAlive/2 {
				continue
			}
			if err := s.write(websocket.TextMessage, []byte(msgKeepAlive)); err != nil {
				writeFailed(err)
				return
			}
		}
	}
}

// readLoop turns server messages into events until the connection ends.
func (s *sttStream) readLoop() {
	defer close(s.events)
	defer s.conn.Close()
	for {
		mt, data, err := s.conn.ReadMessage()
		if err != nil {
			s.fail(s.readError(err))
			return
		}
		if mt != websocket.TextMessage {
			continue
		}
		if err := s.handle(data); err != nil {
			s.fail(err)
			return
		}
	}
}

// readError classifies the error that ended the read loop.
func (s *sttStream) readError(err error) error {
	if s.isClosed() {
		return closedError(providers.KindSTT, "recv")
	}
	if s.ctx.Err() != nil {
		return ctxError(s.ctx, providers.KindSTT, "recv")
	}
	ended := s.ended.Load()
	var ce *websocket.CloseError
	if errors.As(err, &ce) {
		if ce.Code == websocket.CloseNormalClosure || (ended && ce.Code == websocket.CloseNoStatusReceived) {
			return io.EOF
		}
		if ce.Text != "" {
			err = errors.New(ce.Text)
		}
		// 1008 (DATA-0000) means the audio could not be decoded; sending
		// it again will not help.
		return s.newError("recv", ce.Code, ce.Code != websocket.ClosePolicyViolation, err)
	}
	if ended && s.gotMetadata {
		// Everything was delivered; only the close handshake is missing.
		return io.EOF
	}
	return s.newError("recv", 0, true, err)
}

// serverMessage is the union of the /v1/listen server messages we consume.
type serverMessage struct {
	Type string `json:"type"`

	// Results
	Start       float64 `json:"start"`
	Duration    float64 `json:"duration"`
	IsFinal     bool    `json:"is_final"`
	SpeechFinal bool    `json:"speech_final"`

	// Error
	Description string `json:"description"`
	Message     string `json:"message"`
	Variant     string `json:"variant"`

	// SpeechStarted
	Timestamp float64 `json:"timestamp"`
	// UtteranceEnd
	LastWordEnd float64 `json:"last_word_end"`
}

// resultsChannel is the "channel" object of a Results message. It is decoded
// separately because SpeechStarted and UtteranceEnd reuse the "channel" key
// for an array of integers.
type resultsChannel struct {
	Channel struct {
		Alternatives []struct {
			Transcript string   `json:"transcript"`
			Confidence float64  `json:"confidence"`
			Languages  []string `json:"languages"`
		} `json:"alternatives"`
	} `json:"channel"`
}

func seconds(v float64) time.Duration { return time.Duration(v * float64(time.Second)) }

// handle decodes one text message and emits the matching events. A non-nil
// error ends the stream.
func (s *sttStream) handle(data []byte) error {
	var m serverMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return nil // not ours to interpret; ignore unknown payloads
	}
	switch m.Type {
	case "Results":
		var rc resultsChannel
		if err := json.Unmarshal(data, &rc); err != nil {
			return nil
		}
		end := seconds(m.Start + m.Duration)
		if alts := rc.Channel.Alternatives; len(alts) > 0 && strings.TrimSpace(alts[0].Transcript) != "" {
			ev := stt.Event{
				Type:       stt.Interim,
				Text:       alts[0].Transcript,
				Language:   s.language,
				Confidence: alts[0].Confidence,
				Start:      seconds(m.Start),
				End:        end,
			}
			if m.IsFinal {
				ev.Type = stt.Final
			}
			if len(alts[0].Languages) > 0 {
				ev.Language = alts[0].Languages[0]
			}
			if err := s.emit(ev); err != nil {
				return err
			}
		}
		if m.SpeechFinal {
			return s.emit(stt.Event{Type: stt.EndOfSpeech, End: end})
		}
	case "SpeechStarted":
		return s.emit(stt.Event{Type: stt.SpeechStarted, Start: seconds(m.Timestamp)})
	case "UtteranceEnd":
		return s.emit(stt.Event{Type: stt.EndOfSpeech, End: seconds(m.LastWordEnd)})
	case "Metadata":
		s.gotMetadata = true
	case "Error":
		msg := m.Description
		if msg == "" {
			msg = m.Message
		}
		if msg == "" {
			msg = m.Variant
		}
		return s.newError("recv", 0, false, errors.New(msg))
	}
	return nil
}

// emit delivers ev unless it is an EndOfSpeech with no speech before it.
func (s *sttStream) emit(ev stt.Event) error {
	if ev.Type == stt.EndOfSpeech {
		if !s.speechOpen {
			return nil
		}
		s.speechOpen = false
	} else {
		s.speechOpen = true
	}
	select {
	case s.events <- ev:
		return nil
	case <-s.closed:
		return closedError(providers.KindSTT, "recv")
	}
}
