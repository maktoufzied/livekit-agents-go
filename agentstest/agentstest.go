// Package agentstest provides deterministic STT, LLM, TTS and audio I/O
// implementations for tests and for running the pipeline without provider
// credentials.
package agentstest

import (
	"context"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers"
	"github.com/maktoufzied/livekit-agents-go/stt"
	"github.com/maktoufzied/livekit-agents-go/tts"
)

// ---------------------------------------------------------------------------
// STT

// STT is a scripted speech-to-text provider. Events sent with Send are
// delivered to the currently open stream; audio pushed to it is counted.
type STT struct {
	mu      sync.Mutex
	streams []*sttStream
	opens   int
	frames  int
	// OpenErrs are returned, in order, by the next calls to Stream.
	OpenErrs []error
	// PushDelay makes every Push block that long (a slow provider).
	PushDelay time.Duration
	events    chan sttItem
}

type sttItem struct {
	ev  stt.Event
	err error
}

// NewSTT returns a scripted STT.
func NewSTT() *STT { return &STT{events: make(chan sttItem, 256)} }

// Send queues an event for delivery by the open stream.
func (s *STT) Send(ev stt.Event) { s.events <- sttItem{ev: ev} }

// Final queues a final transcript.
func (s *STT) Final(text string) { s.Send(stt.Event{Type: stt.Final, Text: text}) }

// Interim queues an interim transcript.
func (s *STT) Interim(text string) { s.Send(stt.Event{Type: stt.Interim, Text: text}) }

// Fail makes the open stream's Recv return err.
func (s *STT) Fail(err error) { s.events <- sttItem{err: err} }

// Opens returns how many streams were opened.
func (s *STT) Opens() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.opens
}

// Frames returns how many audio frames were pushed.
func (s *STT) Frames() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.frames
}

// OpenStreams returns the number of streams not yet closed.
func (s *STT) OpenStreams() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, st := range s.streams {
		select {
		case <-st.closed:
		default:
			n++
		}
	}
	return n
}

// Stream implements stt.STT.
func (s *STT) Stream(ctx context.Context, _ stt.StreamOptions) (stt.Stream, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.opens++
	if len(s.OpenErrs) > 0 {
		err := s.OpenErrs[0]
		s.OpenErrs = s.OpenErrs[1:]
		if err != nil {
			return nil, err
		}
	}
	st := &sttStream{parent: s, ctx: ctx, closed: make(chan struct{}), ended: make(chan struct{})}
	s.streams = append(s.streams, st)
	return st, nil
}

type sttStream struct {
	parent    *STT
	ctx       context.Context
	closeOnce sync.Once
	endOnce   sync.Once
	closed    chan struct{}
	ended     chan struct{}
}

func (st *sttStream) Push(audio.Frame) error {
	select {
	case <-st.closed:
		return fmt.Errorf("agentstest: %w", providers.ErrStreamClosed)
	default:
	}
	if d := st.parent.PushDelay; d > 0 {
		t := time.NewTimer(d)
		defer t.Stop()
		select {
		case <-t.C:
		case <-st.closed:
			return fmt.Errorf("agentstest: %w", providers.ErrStreamClosed)
		case <-st.ctx.Done():
			return st.ctx.Err()
		}
	}
	st.parent.mu.Lock()
	st.parent.frames++
	st.parent.mu.Unlock()
	return nil
}

func (st *sttStream) CloseSend() error {
	st.endOnce.Do(func() { close(st.ended) })
	return nil
}

func (st *sttStream) Recv() (stt.Event, error) {
	select {
	case it := <-st.parent.events:
		return it.ev, it.err
	case <-st.ended:
		// Deliver what is already queued, then end.
		select {
		case it := <-st.parent.events:
			return it.ev, it.err
		default:
			return stt.Event{}, io.EOF
		}
	case <-st.closed:
		return stt.Event{}, fmt.Errorf("agentstest: %w", providers.ErrStreamClosed)
	case <-st.ctx.Done():
		return stt.Event{}, st.ctx.Err()
	}
}

func (st *sttStream) Close() error {
	st.closeOnce.Do(func() { close(st.closed) })
	return nil
}

// ---------------------------------------------------------------------------
// LLM

// Response is one scripted model response.
type Response struct {
	// Chunks are streamed in order.
	Chunks []llm.Chunk
	// Delay is waited before every chunk.
	Delay time.Duration
	// Err, when set, is returned by Chat (OpenErr) or after the chunks.
	OpenErr error
	Err     error
	// Hang blocks after the chunks until the stream is closed or cancelled.
	Hang bool
}

// Text builds a Response streaming text word by word.
func Text(text string) Response {
	var chunks []llm.Chunk
	start := 0
	for i, r := range text {
		if r == ' ' {
			chunks = append(chunks, llm.Chunk{Text: text[start : i+1]})
			start = i + 1
		}
	}
	if start < len(text) {
		chunks = append(chunks, llm.Chunk{Text: text[start:]})
	}
	return Response{Chunks: chunks}
}

// ToolCalls builds a Response requesting tool calls.
func ToolCalls(calls ...llm.ToolCall) Response {
	return Response{Chunks: []llm.Chunk{{ToolCalls: calls}}}
}

// LLM is a scripted model. Each Chat consumes the next Response; when the
// script is exhausted, Default is used.
type LLM struct {
	mu        sync.Mutex
	responses []Response
	requests  []llm.Request
	open      int
	// Default answers once the script is exhausted.
	Default Response
	// Handler, when set, computes the response and overrides the script.
	Handler func(req llm.Request) Response
}

// NewLLM returns a scripted LLM.
func NewLLM(responses ...Response) *LLM {
	return &LLM{responses: responses, Default: Text("Okay.")}
}

// Requests returns every request received so far.
func (l *LLM) Requests() []llm.Request {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]llm.Request(nil), l.requests...)
}

// OpenStreams returns the number of streams not yet closed.
func (l *LLM) OpenStreams() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.open
}

// Chat implements llm.LLM.
func (l *LLM) Chat(ctx context.Context, req llm.Request) (llm.Stream, error) {
	l.mu.Lock()
	l.requests = append(l.requests, req)
	var r Response
	switch {
	case l.Handler != nil:
		r = l.Handler(req)
	case len(l.responses) > 0:
		r = l.responses[0]
		l.responses = l.responses[1:]
	default:
		r = l.Default
	}
	if r.OpenErr != nil {
		l.mu.Unlock()
		return nil, r.OpenErr
	}
	l.open++
	l.mu.Unlock()
	return &llmStream{parent: l, ctx: ctx, r: r, closed: make(chan struct{})}, nil
}

type llmStream struct {
	parent    *LLM
	ctx       context.Context
	r         Response
	i         int
	closeOnce sync.Once
	closed    chan struct{}
}

func (s *llmStream) wait(d time.Duration) error {
	var timer <-chan time.Time
	if d >= 0 {
		t := time.NewTimer(d)
		defer t.Stop()
		timer = t.C
	}
	select {
	case <-timer:
		return nil
	case <-s.closed:
		return fmt.Errorf("agentstest: %w", providers.ErrStreamClosed)
	case <-s.ctx.Done():
		return s.ctx.Err()
	}
}

func (s *llmStream) Recv() (llm.Chunk, error) {
	if s.i < len(s.r.Chunks) {
		if err := s.wait(s.r.Delay); err != nil {
			return llm.Chunk{}, err
		}
		c := s.r.Chunks[s.i]
		s.i++
		return c, nil
	}
	if s.r.Hang {
		return llm.Chunk{}, s.wait(-1)
	}
	if err := s.wait(0); err != nil {
		return llm.Chunk{}, err
	}
	if s.r.Err != nil {
		return llm.Chunk{}, s.r.Err
	}
	return llm.Chunk{}, io.EOF
}

func (s *llmStream) Close() error {
	s.closeOnce.Do(func() {
		close(s.closed)
		s.parent.mu.Lock()
		s.parent.open--
		s.parent.mu.Unlock()
	})
	return nil
}

// ---------------------------------------------------------------------------
// TTS

// TTS synthesizes a tone whose length is proportional to the text length.
// Every sample of a sentence carries the same Marker value, so tests can
// tell which sentence a played frame belongs to.
type TTS struct {
	// AudioFormat defaults to 24 kHz mono.
	AudioFormat audio.Format
	// PerChar is the audio duration per character. Default 20ms.
	PerChar time.Duration
	// FrameDuration defaults to 20ms.
	FrameDuration time.Duration
	// FrameDelay is waited before each frame (synthesis speed).
	FrameDelay time.Duration
	// FirstFrameDelay is waited before the first frame.
	FirstFrameDelay time.Duration
	// Tone generates an audible sine wave instead of constant markers.
	Tone bool

	mu     sync.Mutex
	texts  []string
	errs   []error
	open   int
	marker int16
}

// NewTTS returns a fake TTS with defaults.
func NewTTS() *TTS { return &TTS{} }

// Format implements tts.TTS.
func (t *TTS) Format() audio.Format {
	if t.AudioFormat.Valid() {
		return t.AudioFormat
	}
	return audio.Format{SampleRate: 24000, Channels: 1}
}

// FailNext makes the next Synthesize calls fail with errs, in order. A nil
// entry lets that call succeed.
func (t *TTS) FailNext(errs ...error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.errs = append(t.errs, errs...)
}

// Texts returns every text synthesized so far, in request order.
func (t *TTS) Texts() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.texts...)
}

// OpenStreams returns the number of streams not yet closed.
func (t *TTS) OpenStreams() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.open
}

// Duration returns the audio duration Synthesize produces for text.
func (t *TTS) Duration(text string) time.Duration {
	perChar, frame := t.PerChar, t.FrameDuration
	if perChar <= 0 {
		perChar = 20 * time.Millisecond
	}
	if frame <= 0 {
		frame = 20 * time.Millisecond
	}
	n := (time.Duration(len(text))*perChar + frame - 1) / frame
	return n * frame
}

// Synthesize implements tts.TTS.
func (t *TTS) Synthesize(ctx context.Context, text string) (tts.Stream, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.errs) > 0 {
		err := t.errs[0]
		t.errs = t.errs[1:]
		if err != nil {
			return nil, err
		}
	}
	t.texts = append(t.texts, text)
	t.marker++
	t.open++
	frame := t.FrameDuration
	if frame <= 0 {
		frame = 20 * time.Millisecond
	}
	return &ttsStream{
		parent: t, ctx: ctx, marker: t.marker, frame: frame,
		remaining: int(t.Duration(text) / frame),
		closed:    make(chan struct{}),
	}, nil
}

type ttsStream struct {
	parent    *TTS
	ctx       context.Context
	marker    int16
	frame     time.Duration
	remaining int
	sent      int
	closeOnce sync.Once
	closed    chan struct{}
}

func (s *ttsStream) Recv() (audio.Frame, error) {
	d := s.parent.FrameDelay
	if s.sent == 0 {
		d += s.parent.FirstFrameDelay
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
	case <-s.closed:
		return audio.Frame{}, fmt.Errorf("agentstest: %w", providers.ErrStreamClosed)
	case <-s.ctx.Done():
		return audio.Frame{}, s.ctx.Err()
	}
	if s.remaining == 0 {
		return audio.Frame{}, io.EOF
	}
	s.remaining--
	s.sent++
	format := s.parent.Format()
	if s.parent.Tone {
		return audio.Tone(format, 440, s.frame, 0.5), nil
	}
	f := audio.Silence(format, s.frame)
	for i := range f.Data {
		f.Data[i] = s.marker
	}
	return f, nil
}

func (s *ttsStream) Close() error {
	s.closeOnce.Do(func() {
		close(s.closed)
		s.parent.mu.Lock()
		s.parent.open--
		s.parent.mu.Unlock()
	})
	return nil
}

// ---------------------------------------------------------------------------
// Audio I/O

// Input is an in-memory audio source.
type Input struct {
	format audio.Format
	ch     chan audio.Frame
	once   sync.Once
	closed chan struct{}
}

// NewInput returns an input with the given format (16 kHz mono if zero).
func NewInput(format audio.Format) *Input {
	if !format.Valid() {
		format = audio.Format{SampleRate: 16000, Channels: 1}
	}
	return &Input{format: format, ch: make(chan audio.Frame, 1024), closed: make(chan struct{})}
}

// Format implements the session's AudioInput.
func (in *Input) Format() audio.Format { return in.format }

// Push queues a frame for the session.
func (in *Input) Push(f audio.Frame) { in.ch <- f }

// PushTone queues d of loud audio in 20ms frames.
func (in *Input) PushTone(d time.Duration) {
	for ; d > 0; d -= 20 * time.Millisecond {
		in.Push(audio.Tone(in.format, 300, 20*time.Millisecond, 0.5))
	}
}

// PushSilence queues d of silence in 20ms frames.
func (in *Input) PushSilence(d time.Duration) {
	for ; d > 0; d -= 20 * time.Millisecond {
		in.Push(audio.Silence(in.format, 20*time.Millisecond))
	}
}

// Close ends the input: ReadFrame returns io.EOF once drained.
func (in *Input) Close() { in.once.Do(func() { close(in.closed) }) }

// ReadFrame implements the session's AudioInput.
func (in *Input) ReadFrame(ctx context.Context) (audio.Frame, error) {
	select {
	case f := <-in.ch:
		return f, nil
	default:
	}
	select {
	case f := <-in.ch:
		return f, nil
	case <-in.closed:
		return audio.Frame{}, io.EOF
	case <-ctx.Done():
		return audio.Frame{}, ctx.Err()
	}
}

// Written describes one frame accepted by Output.
type Written struct {
	Marker int16 // first sample of the frame
	At     time.Time
}

// Output is an in-memory audio sink that simulates real-time playback: it
// keeps a playhead and lets at most MaxQueued of audio wait in its queue.
type Output struct {
	format audio.Format
	// Speed is the playback speed relative to real time. Default 1.
	Speed float64
	// MaxQueued is the look-ahead the output accepts. Default 60ms.
	MaxQueued time.Duration

	mu       sync.Mutex
	playhead time.Time // wall-clock time at which queued audio ends
	written  []Written
	total    time.Duration
	dropped  time.Duration
	clears   int
}

// NewOutput returns an output with the given format.
func NewOutput(format audio.Format) *Output { return &Output{format: format} }

// Format implements the session's AudioOutput.
func (o *Output) Format() audio.Format { return o.format }

func (o *Output) scale(d time.Duration) time.Duration {
	if o.Speed > 0 {
		return time.Duration(float64(d) / o.Speed)
	}
	return d
}

// WriteFrame implements the session's AudioOutput.
func (o *Output) WriteFrame(ctx context.Context, f audio.Frame) error {
	maxQueued := o.MaxQueued
	if maxQueued <= 0 {
		maxQueued = 60 * time.Millisecond
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		o.mu.Lock()
		now := time.Now()
		if o.playhead.Before(now) {
			o.playhead = now
		}
		ahead := o.playhead.Sub(now)
		if ahead <= o.scale(maxQueued) {
			o.playhead = o.playhead.Add(o.scale(f.Duration()))
			var marker int16
			if len(f.Data) > 0 {
				marker = f.Data[0]
			}
			o.written = append(o.written, Written{Marker: marker, At: now})
			o.total += f.Duration()
			o.mu.Unlock()
			return nil
		}
		o.mu.Unlock()
		t := time.NewTimer(ahead - o.scale(maxQueued))
		select {
		case <-t.C:
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		}
	}
}

// Flush implements the session's AudioOutput.
func (o *Output) Flush(ctx context.Context) error {
	o.mu.Lock()
	wait := time.Until(o.playhead)
	o.mu.Unlock()
	if wait <= 0 {
		return nil
	}
	t := time.NewTimer(wait)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Clear implements the session's AudioOutput.
func (o *Output) Clear() time.Duration {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.clears++
	now := time.Now()
	if !o.playhead.After(now) {
		return 0
	}
	queued := o.playhead.Sub(now)
	if o.Speed > 0 {
		queued = time.Duration(float64(queued) * o.Speed)
	}
	o.playhead = now
	o.dropped += queued
	return queued
}

// Frames returns every frame written so far.
func (o *Output) Frames() []Written {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]Written(nil), o.written...)
}

// Written returns the total audio duration accepted.
func (o *Output) Written() time.Duration {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.total
}

// Played returns the audio duration that was not dropped by Clear.
func (o *Output) Played() time.Duration {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.total - o.dropped
}
