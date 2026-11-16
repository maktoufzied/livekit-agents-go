package agents

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers"
	"github.com/maktoufzied/livekit-agents-go/stt"
	"github.com/maktoufzied/livekit-agents-go/tts"
	"github.com/maktoufzied/livekit-agents-go/vad"
)

// Session runs one voice conversation: it listens to the user, detects
// turns, generates replies with the LLM and tools, and speaks them.
//
// A Session is created with NewSession, bound to its audio transport with
// Start and released with Close. All methods are safe for concurrent use.
type Session struct {
	opts  SessionOptions
	log   *slog.Logger
	conv  *llm.Conversation
	bus   *eventBus
	tools map[string]llm.Tool
	specs []llm.ToolSpec

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	in     AudioInput
	out    AudioOutput

	loopCh    chan func()
	speechCh  chan *SpeechHandle
	sttFrames chan audio.Frame

	mu          sync.Mutex
	started     bool
	closing     bool
	agentState  AgentState
	userState   UserState
	speeches    []*SpeechHandle // queued and running, in order
	closeReason CloseReason
	closeErr    error
	done        chan struct{}

	warmupOnce    sync.Once
	warmupUntil   atomic.Int64 // end of the echo warm-up, Unix nanoseconds; 0 before it starts
	speechSeq     atomic.Int64
	msgSeq        atomic.Int64
	droppedFrames atomic.Int64

	// turn is owned by the loop goroutine.
	turn turnState
	// turnAudio is the audio of the user turn for the TurnDetector; nil
	// without one.
	turnAudio *turnAudio
}

type turnState struct {
	userSpeaking  bool
	speechEndedAt time.Time
	// speechStartedAt is when the sound the VAD last reported began.
	speechStartedAt time.Time
	finals          []string
	interim         string
	eouGen          uint64 // invalidates pending end-of-turn timers
	eouExtended     bool
	bargeGen        uint64 // invalidates pending interruption timers
	wantInterrupt   bool   // speech lasted long enough, waiting for words

	// held is the speech put on hold by a sound that may or may not be an
	// interruption; nil when none. The speech may have ended since.
	held    *SpeechHandle
	heldAt  time.Time
	heldGen uint64 // invalidates pending resume timers
	// heldFor is the silence after which the speech on hold resumes.
	heldFor time.Duration

	// preempt is the model request started ahead of the end of the turn;
	// nil when none. preemptCount counts those started in this turn.
	preempt      *preemptive
	preemptAt    time.Time
	preemptCount int
}

// NewSession validates the options and returns an idle session.
func NewSession(opts SessionOptions) (*Session, error) {
	if err := opts.setDefaults(); err != nil {
		return nil, err
	}
	s := &Session{
		opts:       opts,
		log:        opts.Logger,
		conv:       opts.History,
		tools:      map[string]llm.Tool{},
		agentState: AgentInitializing,
		userState:  UserListening,
		done:       make(chan struct{}),
	}
	for _, t := range opts.Agent.Tools {
		s.tools[t.Name] = t
		s.specs = append(s.specs, t.Spec())
	}
	s.bus = newEventBus(opts.Buffers.Events, s.log)
	if opts.TurnDetector != nil {
		s.turnAudio = newTurnAudio()
	}
	return s, nil
}

// OutputFormat is the audio format the session writes to its AudioOutput.
func (s *Session) OutputFormat() audio.Format { return s.opts.TTS.Format() }

// History returns the live conversation, not a copy: it is updated as turns
// are committed, and messages appended to it are seen by the next model
// request. Use History().Messages() for a snapshot.
func (s *Session) History() *llm.Conversation { return s.conv }

// On registers an event handler and returns a function that removes it.
//
// Handlers are called one at a time, in emission order, from a dedicated
// goroutine, so a slow handler delays other handlers but never the audio
// pipeline. Events are dropped (and counted in Stats) when the queue is
// full. A handler must not call Close; use Shutdown from handlers.
func (s *Session) On(handler func(Event)) (remove func()) { return s.bus.on(handler) }

// Stats returns cumulative counters.
func (s *Session) Stats() SessionStats {
	return SessionStats{DroppedInputFrames: s.droppedFrames.Load(), DroppedEvents: s.bus.dropped.Load()}
}

// AgentState returns the current agent state.
func (s *Session) AgentState() AgentState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.agentState
}

// UserState returns the current user state.
func (s *Session) UserState() UserState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.userState
}

// Done is closed once the session is fully closed and every goroutine it
// started has exited.
func (s *Session) Done() <-chan struct{} { return s.done }

// Start binds the session to its transport and starts the pipeline. It
// returns as soon as the pipeline is running.
//
// ctx is the lifetime of the session, not a deadline for starting it: the
// session closes when ctx is cancelled. Pass the job context (or
// context.Background()), never a request-scoped or timeout context.
func (s *Session) Start(ctx context.Context, tr IO) error {
	if tr.Output == nil {
		return errors.New("agents: IO.Output is required")
	}
	if got, want := tr.Output.Format(), s.opts.TTS.Format(); got != want {
		return &FormatError{Stage: SourceOutput, Want: want, Got: got}
	}
	if tr.Input != nil {
		if s.opts.STT == nil {
			return errors.New("agents: SessionOptions.STT is required with an audio input")
		}
		if !tr.Input.Format().Valid() {
			return errors.New("agents: audio input reports an invalid format")
		}
	}

	s.mu.Lock()
	if s.closing {
		s.mu.Unlock()
		return ErrSessionClosed
	}
	if s.started {
		s.mu.Unlock()
		return ErrAlreadyStarted
	}
	s.started = true
	s.ctx, s.cancel = context.WithCancel(ctx)
	s.in, s.out = tr.Input, tr.Output
	s.loopCh = make(chan func(), 256)
	s.speechCh = make(chan *SpeechHandle, s.opts.Buffers.Speeches)
	s.sttFrames = make(chan audio.Frame, s.opts.Buffers.STTFrames)
	if pw, ok := s.opts.TTS.(tts.Prewarmer); ok {
		pw.Prewarm() // open the connection before the first reply needs it
	}
	// Started under the lock so that a concurrent shutdown waits for them.
	s.goRun(s.loop)
	s.goRun(s.speechLoop)
	if s.in != nil {
		s.goRun(s.inputLoop)
		s.goRun(s.sttLoop)
	}
	s.mu.Unlock()

	// Close the session when the caller's context ends.
	go func() {
		select {
		case <-ctx.Done():
			s.shutdown(CloseContextDone, nil)
		case <-s.ctx.Done():
		}
	}()
	s.setAgentState(AgentListening)
	return nil
}

func (s *Session) goRun(fn func()) {
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		fn()
	}()
}

// Close interrupts any speech, stops the pipeline and waits until every
// goroutine has exited and every provider stream is closed. It returns the
// error that closed the session, if any (see Err), and is idempotent. It
// must not be called from an event handler: use Shutdown there.
func (s *Session) Close() error {
	s.shutdown(CloseRequested, nil)
	<-s.done
	return s.Err()
}

// Shutdown starts closing the session without waiting. Use Done to wait.
func (s *Session) Shutdown() { s.shutdown(CloseRequested, nil) }

// Drain waits for queued and running speeches to finish, then closes the
// session. If ctx ends first the session is closed immediately.
func (s *Session) Drain(ctx context.Context) error {
	for {
		s.mu.Lock()
		var last *SpeechHandle
		if n := len(s.speeches); n > 0 {
			last = s.speeches[n-1]
		}
		s.mu.Unlock()
		if last == nil {
			break
		}
		select {
		case <-last.Done():
		case <-ctx.Done():
			s.Close()
			return ctx.Err()
		case <-s.done:
			return s.Err()
		}
	}
	return s.Close()
}

func (s *Session) shutdown(reason CloseReason, err error) {
	s.mu.Lock()
	if s.closing {
		s.mu.Unlock()
		return
	}
	s.closing = true
	s.closeReason, s.closeErr = reason, err
	started := s.started
	s.mu.Unlock()

	go func() {
		if started {
			s.cancel()
			s.wg.Wait()
			s.out.Clear()
		}
		s.mu.Lock()
		pending := s.speeches
		s.speeches = nil
		s.mu.Unlock()
		for _, sp := range pending {
			sp.finish(ErrSessionClosed)
		}
		if err != nil {
			s.log.Error("session closed with error", "reason", string(reason), "error", err)
		} else {
			s.log.Debug("session closed", "reason", string(reason))
		}
		s.bus.emit(SessionClosed{Reason: reason, Err: err})
		s.bus.close()
		close(s.done)
	}()
}

// Err returns the error that closed the session, if any. It is meaningful
// once Done is closed.
func (s *Session) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closeErr
}

// ---------------------------------------------------------------------------
// Public speech API

// SayOptions customizes Say.
type SayOptions struct {
	// DisableInterruptions prevents the user from interrupting this speech.
	DisableInterruptions bool
	// SkipHistory keeps the text out of the conversation history.
	SkipHistory bool
}

// Say queues fixed text to be spoken. It returns as soon as the speech is
// queued; use the handle to wait for it or interrupt it.
func (s *Session) Say(text string, opts SayOptions) (*SpeechHandle, error) {
	if strings.TrimSpace(text) == "" {
		return nil, errors.New("agents: Say needs non-empty text")
	}
	return s.enqueue(&SpeechHandle{
		source:             SpeechFromSay,
		text:               text,
		allowInterruptions: !opts.DisableInterruptions && !s.opts.Turn.DisableInterruptions,
		addToHistory:       !opts.SkipHistory,
	})
}

// ReplyOptions customizes GenerateReply.
type ReplyOptions struct {
	// UserInput, when set, is appended to the history as a user message
	// when the reply starts, as if the user had said it. Nothing is
	// appended if GenerateReply returns an error.
	UserInput string
	// Instructions are extra system instructions for this reply only.
	Instructions string
	// DisableInterruptions prevents the user from interrupting this reply.
	DisableInterruptions bool
}

// GenerateReply queues a model-generated reply.
func (s *Session) GenerateReply(opts ReplyOptions) (*SpeechHandle, error) {
	sp := &SpeechHandle{
		source:             SpeechFromReply,
		instructions:       opts.Instructions,
		allowInterruptions: !opts.DisableInterruptions && !s.opts.Turn.DisableInterruptions,
		addToHistory:       true,
	}
	if strings.TrimSpace(opts.UserInput) != "" {
		sp.userMsg = &llm.Message{Role: llm.RoleUser, Content: opts.UserInput, CreatedAt: time.Now()}
	}
	return s.enqueue(sp)
}

// Interrupt stops the current speech and discards queued ones, including
// speeches that do not allow user interruptions.
func (s *Session) Interrupt() { s.interruptSpeeches(true) }

// CommitUserTurn ends the user turn now, using the transcripts received so
// far. It is the only way to end a turn in TurnManual.
func (s *Session) CommitUserTurn() {
	if s.isRunning() {
		s.post(s.commitUserTurn)
	}
}

// ClearUserTurn discards the transcripts of the current user turn.
func (s *Session) ClearUserTurn() {
	if s.isRunning() {
		s.post(func() {
			s.turn.finals, s.turn.interim = nil, ""
			s.turn.eouGen++
			s.turn.preemptCount = 0
			s.dropPreemptive()
			s.resetTurnAudio()
		})
	}
}

func (s *Session) isRunning() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.started && !s.closing
}

func (s *Session) enqueue(sp *SpeechHandle) (*SpeechHandle, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closing {
		return nil, ErrSessionClosed
	}
	if !s.started {
		return nil, ErrNotStarted
	}
	sp.id = fmt.Sprintf("speech_%d", s.speechSeq.Add(1))
	sp.createdAt = time.Now()
	sp.done = make(chan struct{})
	sp.ctx, sp.cancel = context.WithCancel(s.ctx)
	select {
	case s.speechCh <- sp:
	default:
		sp.cancel()
		return nil, ErrSpeechQueueFull
	}
	s.speeches = append(s.speeches, sp)
	return sp, nil
}

// interruptSpeeches interrupts running and queued speeches. Without force,
// speeches that disallow interruptions are left alone. It reports whether
// anything was interrupted.
func (s *Session) interruptSpeeches(force bool) bool {
	s.mu.Lock()
	list := append([]*SpeechHandle(nil), s.speeches...)
	s.mu.Unlock()
	any := false
	for _, sp := range list {
		if (force || sp.allowInterruptions) && !sp.Interrupted() {
			sp.Interrupt()
			any = true
		}
	}
	return any
}

func (s *Session) hasInterruptibleSpeech() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, sp := range s.speeches {
		if sp.allowInterruptions && !sp.Interrupted() {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// State and history helpers

func (s *Session) emit(ev Event) { s.bus.emit(ev) }

func (s *Session) setAgentState(st AgentState) {
	s.mu.Lock()
	old := s.agentState
	s.agentState = st
	s.mu.Unlock()
	if old != st {
		s.emit(AgentStateChanged{Old: old, New: st})
	}
}

func (s *Session) setUserState(st UserState) {
	s.mu.Lock()
	old := s.userState
	s.userState = st
	s.mu.Unlock()
	if old != st {
		s.emit(UserStateChanged{Old: old, New: st})
	}
}

// commit appends a message to the history and announces it.
func (s *Session) commit(msgs ...llm.Message) {
	for i := range msgs {
		if msgs[i].ID == "" {
			msgs[i].ID = fmt.Sprintf("item_%d", s.msgSeq.Add(1))
		}
		if msgs[i].CreatedAt.IsZero() {
			msgs[i].CreatedAt = time.Now()
		}
	}
	s.conv.Append(msgs...)
	for _, m := range msgs {
		s.emit(ConversationItemAdded{Message: m})
	}
}

func (s *Session) reportError(source ErrorSource, err error, recoverable bool) {
	if err == nil || errors.Is(err, context.Canceled) {
		return
	}
	level := slog.LevelWarn
	if !recoverable {
		level = slog.LevelError
	}
	s.log.Log(context.Background(), level, "pipeline error", "source", string(source), "recoverable", recoverable, "error", err)
	s.emit(ErrorEvent{Source: source, Err: err, Recoverable: recoverable})
}

func retryable(err error) bool {
	var te *TimeoutError
	return providers.IsRetryable(err) || errors.As(err, &te)
}

// sleep waits for d or until ctx ends, reporting whether the full delay
// elapsed.
func sleep(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

func (s *Session) backoff(attempt int) time.Duration {
	d := s.opts.Retry.Backoff << (attempt - 1)
	if d > 5*time.Second || d <= 0 {
		d = 5 * time.Second
	}
	return d
}

func fmtFormat(f audio.Format) string { return fmt.Sprintf("%dHz/%dch", f.SampleRate, f.Channels) }

// ---------------------------------------------------------------------------
// Event loop: owns turn state. Everything that touches s.turn runs here.

func (s *Session) loop() {
	for {
		select {
		case fn := <-s.loopCh:
			fn()
		case <-s.ctx.Done():
			return
		}
	}
}

func (s *Session) post(fn func()) {
	select {
	case s.loopCh <- fn:
	case <-s.ctx.Done():
	}
}

// after runs fn on the loop after d.
func (s *Session) after(d time.Duration, fn func()) {
	// Always hop through a timer goroutine: the loop must never block
	// posting to its own queue.
	if d < 0 {
		d = 0
	}
	time.AfterFunc(d, func() { s.post(fn) })
}

// ---------------------------------------------------------------------------
// Audio input: VAD + bounded hand-off to STT

func (s *Session) inputLoop() {
	defer close(s.sttFrames)
	var det vad.Detector
	if s.opts.VAD != nil {
		d, err := s.opts.VAD.NewDetector()
		if err != nil {
			s.reportError(SourceInput, fmt.Errorf("create VAD detector: %w", err), false)
			s.shutdown(CloseError, err)
			return
		}
		det = d
		defer det.Close()
	}
	for {
		f, err := s.in.ReadFrame(s.ctx)
		if err != nil {
			if !errors.Is(err, io.EOF) && s.ctx.Err() == nil {
				s.reportError(SourceInput, err, false)
				s.shutdown(CloseError, err)
			}
			return
		}
		var pos int64 // where the audio stands, for events of this frame
		if s.turnAudio != nil {
			s.turnAudio.write(f)
			pos = s.turnAudio.position()
		}
		if det != nil {
			events, err := det.Process(f)
			if err != nil {
				s.reportError(SourceInput, fmt.Errorf("vad: %w", err), true)
			}
			for _, ev := range events {
				ev := ev
				s.post(func() { s.onVAD(ev, pos) })
			}
		}
		if s.echoWarmupActive() {
			// The recognizer must not transcribe the agent's own voice.
			f = audio.Frame{Data: make([]int16, len(f.Data)), SampleRate: f.SampleRate, Channels: f.Channels}
		}
		// Bounded hand-off: never block the audio source on a slow STT
		// provider; drop the oldest frame instead.
		select {
		case s.sttFrames <- f:
		default:
			select {
			case <-s.sttFrames:
				if s.droppedFrames.Add(1) == 1 {
					s.log.Warn("stt cannot keep up with audio input, dropping oldest frames")
				}
			default:
			}
			select {
			case s.sttFrames <- f:
			default:
			}
		}
	}
}

// sttLoop keeps one recognition stream open, reconnecting with bounded
// retries when it fails.
func (s *Session) sttLoop() {
	failures := 0
	for {
		gotEvent, err := s.runSTTStream()
		if err == nil || s.ctx.Err() != nil {
			return
		}
		if gotEvent {
			failures = 0
		}
		failures++
		fatal := failures > s.opts.Retry.STTMaxReconnects || !retryable(err)
		s.reportError(SourceSTT, err, !fatal)
		if fatal {
			s.shutdown(CloseError, err)
			return
		}
		if !sleep(s.ctx, s.backoff(failures)) {
			return
		}
	}
}

// runSTTStream runs one stream until the input ends (nil error), the
// session closes, or the stream fails.
func (s *Session) runSTTStream() (gotEvent bool, err error) {
	stream, err := s.opts.STT.Stream(s.ctx, stt.StreamOptions{Format: s.in.Format()})
	if err != nil {
		return false, err
	}
	stop := make(chan struct{})
	inputEnded := make(chan struct{})
	var pushWG sync.WaitGroup
	pushWG.Add(1)
	go func() {
		defer pushWG.Done()
		for {
			select {
			case f, ok := <-s.sttFrames:
				if !ok {
					close(inputEnded)
					_ = stream.CloseSend()
					return
				}
				if err := stream.Push(f); err != nil {
					return // Recv reports the failure
				}
			case <-stop:
				return
			case <-s.ctx.Done():
				return
			}
		}
	}()
	defer func() {
		close(stop)
		_ = stream.Close()
		pushWG.Wait()
	}()

	for {
		ev, rerr := stream.Recv()
		if rerr != nil {
			if s.ctx.Err() != nil {
				return gotEvent, nil
			}
			if errors.Is(rerr, io.EOF) {
				select {
				case <-inputEnded:
					return gotEvent, nil
				default:
					return gotEvent, &providers.Error{Provider: "stt", Kind: providers.KindSTT, Op: "recv", Retryable: true, Err: errors.New("stream ended unexpectedly")}
				}
			}
			return gotEvent, rerr
		}
		gotEvent = true
		s.post(func() { s.onSTT(ev) })
	}
}

// ---------------------------------------------------------------------------
// Turn detection and interruptions (loop goroutine only)

// onVAD handles a VAD event raised when the input stood at position pos.
func (s *Session) onVAD(ev vad.Event, pos int64) {
	t := &s.turn
	s.log.Debug("vad", "event", ev.Type.String(), "speech", ev.SpeechDuration.Round(time.Millisecond), "silence", ev.SilenceDuration.Round(time.Millisecond))
	switch ev.Type {
	case vad.SpeechStart:
		t.heldGen++ // the user makes a sound again: do not resume yet
		t.speechStartedAt = time.Now().Add(-ev.SpeechDuration)
		s.markTurnAudio(pos, ev.SpeechDuration)
		if s.turnAudio != nil {
			s.turnAudio.resumed()
		}
		t.userSpeaking = true
		t.eouGen++ // the user resumed: cancel any pending end of turn
		s.setUserState(UserSpeaking)
		if s.opts.Turn.DisableInterruptions || !s.hasInterruptibleSpeech() {
			return
		}
		if s.AgentState() != AgentSpeaking {
			// A reply is being prepared and the user makes a sound: its
			// audio waits until the sound is over, however short it is.
			if s.opts.Turn.FalseInterruptionTimeout >= 0 {
				s.hold(0)
			}
		}
		t.bargeGen++
		gen := t.bargeGen
		var qualify func()
		qualify = func() {
			if gen != t.bargeGen || !t.userSpeaking {
				return
			}
			if left := s.echoWarmupLeft(); left > 0 {
				// Most likely the agent's own voice. A sound that outlasts
				// the warm-up is looked at again.
				s.after(left, qualify)
				return
			}
			if s.opts.Turn.MinInterruptionWords > 0 {
				t.wantInterrupt = true
				s.maybeInterruptByWords()
				return
			}
			s.holdOrInterrupt()
		}
		s.after(s.opts.Turn.MinInterruptionDuration-ev.SpeechDuration, qualify)
	case vad.SpeechEnd:
		if s.turnAudio != nil {
			s.turnAudio.stopped(pos, ev.SilenceDuration)
		}
		t.userSpeaking = false
		// The speech ended before the silence that confirmed it: delays
		// count from there, so the VAD's silence is not waited twice.
		t.speechEndedAt = time.Now().Add(-ev.SilenceDuration)
		t.bargeGen++
		t.wantInterrupt = false
		s.setUserState(UserListening)
		if len(t.finals) > 0 {
			s.scheduleEOU(s.opts.Turn.MinEndpointingDelay - time.Since(t.speechEndedAt))
		}
		s.scheduleResume()
	}
}

func (s *Session) onSTT(ev stt.Event) {
	t := &s.turn
	hasVAD := s.opts.VAD != nil
	switch ev.Type {
	case stt.SpeechStarted:
		if !hasVAD {
			s.markTurnAudio(s.turnAudioPosition(), 0)
			t.userSpeaking = true
			t.eouGen++
			s.setUserState(UserSpeaking)
		}
	case stt.Interim:
		if strings.TrimSpace(ev.Text) == "" {
			return
		}
		t.interim = ev.Text
		s.emit(UserTranscript{Text: ev.Text, Language: ev.Language})
		if !hasVAD {
			if !t.userSpeaking {
				// The words were said before they were transcribed.
				s.markTurnAudio(s.turnAudioPosition(), sttStartLag)
			}
			t.userSpeaking = true
			t.eouGen++
			s.setUserState(UserSpeaking)
		}
		s.maybeInterruptByWords()
	case stt.Final:
		text := strings.TrimSpace(ev.Text)
		if text == "" {
			return
		}
		t.finals = append(t.finals, text)
		t.interim = ""
		s.log.Debug("final transcript", "text", text, "user_speaking", t.userSpeaking)
		s.emit(UserTranscript{Text: text, Final: true, Language: ev.Language})
		if hasVAD {
			s.interruptByTranscript()
		}
		if hasVAD && !t.userSpeaking && !t.speechEndedAt.IsZero() {
			s.emit(MetricsCollected{Metric: TranscriptionMetrics{Delay: time.Since(t.speechEndedAt)}})
		}
		s.maybeInterruptByWords()
		if !hasVAD {
			t.userSpeaking = false
			t.speechEndedAt = time.Now()
			s.setUserState(UserListening)
		}
		if t.userSpeaking {
			// The VAD speech end will schedule the end of turn. The reply
			// is prepared already: the user may well be done.
			s.preempt()
			return
		}
		switch s.opts.Turn.Mode {
		case TurnSTT:
			s.scheduleEOU(s.opts.Turn.MaxEndpointingDelay)
		default:
			s.scheduleEOU(s.opts.Turn.MinEndpointingDelay - time.Since(t.speechEndedAt))
		}
	case stt.EndOfSpeech:
		if !hasVAD {
			t.userSpeaking = false
			if t.speechEndedAt.IsZero() {
				t.speechEndedAt = time.Now()
			}
			s.setUserState(UserListening)
		}
		if s.opts.Turn.Mode == TurnSTT && len(t.finals) > 0 {
			s.scheduleEOU(0)
		}
	}
}

// maybeInterruptByWords interrupts the agent once enough words were
// transcribed. It is the only interruption path without a VAD.
func (s *Session) maybeInterruptByWords() {
	t := &s.turn
	if s.opts.Turn.DisableInterruptions || (s.opts.VAD != nil && !t.wantInterrupt) {
		return
	}
	words := len(strings.Fields(t.interim))
	for _, f := range t.finals {
		words += len(strings.Fields(f))
	}
	if words >= max(1, s.opts.Turn.MinInterruptionWords) {
		t.wantInterrupt = false
		s.bargeIn()
	}
}

func (s *Session) bargeIn() {
	s.turn.held = nil
	if s.interruptSpeeches(false) {
		s.log.Debug("user interrupted the agent")
	}
}

// holdOrInterrupt reacts to a sound the VAD took for speech and that lasted
// long enough to interrupt, before any final transcript. The running speech
// is put on hold: a final transcript or the end of the user turn will
// interrupt it for good, FalseInterruptionTimeout of silence will resume it.
// When holding is disabled, or nothing is running that could be resumed,
// speeches are interrupted at once.
func (s *Session) holdOrInterrupt() {
	if s.opts.Turn.FalseInterruptionTimeout < 0 || !s.hold(s.opts.Turn.FalseInterruptionTimeout) {
		s.bargeIn()
	}
}

// hold puts the running speech on hold until resumeAfter of silence, and
// reports whether it is on hold. A speech already on hold keeps the longer
// of its delays.
func (s *Session) hold(resumeAfter time.Duration) bool {
	t := &s.turn
	sp := s.runningSpeech()
	if sp == nil || !sp.allowInterruptions || sp.Interrupted() {
		return false
	}
	if t.held == sp {
		t.heldFor = max(t.heldFor, resumeAfter)
		return true
	}
	if !sp.hold.set(true) {
		return false
	}
	t.held, t.heldAt, t.heldFor = sp, time.Now(), resumeAfter
	t.heldGen++
	s.log.Debug("agent speech on hold", "speech_id", sp.id)
	return true
}

// interruptByTranscript is called on a final transcript: the user said
// words, so whatever the agent says or prepares is interrupted, whether the
// VAD noticed the speech or not.
func (s *Session) interruptByTranscript() {
	t := &s.turn
	if s.opts.Turn.DisableInterruptions || s.echoWarmupLeft() > 0 {
		return
	}
	if n := s.opts.Turn.MinInterruptionWords; n > 0 {
		words := 0
		for _, f := range t.finals {
			words += len(strings.Fields(f))
		}
		if words < n {
			return
		}
	}
	s.bargeIn()
}

// scheduleResume arms the timer that resumes the speech on hold, when the
// user stops making sound.
func (s *Session) scheduleResume() {
	t := &s.turn
	if t.held == nil {
		return
	}
	t.heldGen++
	gen := t.heldGen
	s.after(t.heldFor, func() {
		if gen != t.heldGen || t.held == nil || t.userSpeaking {
			return
		}
		if len(t.finals) > 0 {
			return // the turn is about to end and will interrupt the speech
		}
		sp := t.held
		t.held = nil
		if sp.hold.set(false) && !sp.Interrupted() {
			s.log.Debug("false interruption, agent speech resumed", "speech_id", sp.id)
			s.emit(FalseInterruption{SpeechID: sp.id, Paused: time.Since(t.heldAt)})
		}
	})
}

// startEchoWarmup opens the window during which the agent cannot be
// interrupted by sound, at its first audio.
func (s *Session) startEchoWarmup() {
	if d := s.opts.Turn.EchoWarmup; d > 0 {
		s.warmupOnce.Do(func() {
			s.warmupUntil.Store(time.Now().Add(d).UnixNano())
			s.log.Debug("echo warm-up, the agent cannot be interrupted by sound", "for", d)
		})
	}
}

// echoWarmupLeft returns how much of the echo warm-up is left; zero when it
// has not started or is over.
func (s *Session) echoWarmupLeft() time.Duration {
	until := s.warmupUntil.Load()
	if until == 0 {
		return 0
	}
	return max(0, time.Until(time.Unix(0, until)))
}

// echoWarmupActive reports whether what the microphone picks up now may be
// the agent's own voice, not yet cancelled on the user's side.
func (s *Session) echoWarmupActive() bool {
	return s.echoWarmupLeft() > 0 && s.AgentState() == AgentSpeaking
}

// runningSpeech returns the speech at the head of the queue, the only one
// that can be playing.
func (s *Session) runningSpeech() *SpeechHandle {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.speeches) == 0 {
		return nil
	}
	return s.speeches[0]
}

// sttStartLag is the assumed delay between the start of speech and the
// first interim transcript, used to place the turn start without a VAD.
const sttStartLag = 500 * time.Millisecond

// markTurnAudio notes that the user started speaking ago before the input
// stood at position at. The
// first speech of a turn sets the start of the audio given to the
// TurnDetector. Speech that follows a sound of which nothing was transcribed
// (a noise, or a turn that was cleared) starts over.
func (s *Session) markTurnAudio(at int64, ago time.Duration) {
	if s.turnAudio == nil {
		return
	}
	t := &s.turn
	stale := len(t.finals) == 0 && t.interim == ""
	s.turnAudio.mark(at, ago, stale)
}

// turnAudioPosition is the current input position, for speech reported by
// the STT provider, which carries no position.
func (s *Session) turnAudioPosition() int64 {
	if s.turnAudio == nil {
		return 0
	}
	return s.turnAudio.position()
}

func (s *Session) resetTurnAudio() {
	if s.turnAudio != nil {
		s.turnAudio.reset()
	}
}

// scheduleEOU (re)arms the end-of-turn timer.
func (s *Session) scheduleEOU(d time.Duration) {
	if s.opts.Turn.Mode == TurnManual {
		return
	}
	t := &s.turn
	t.eouGen++
	t.eouExtended = false
	s.armEOU(d)
}

func (s *Session) armEOU(d time.Duration) {
	t := &s.turn
	gen := t.eouGen
	ask := s.opts.TurnDetector != nil && !t.eouExtended && len(t.finals) > 0

	// These are only touched by the closures below, which run on the loop.
	var (
		waiting = ask // the end-of-turn model has not answered yet
		elapsed bool  // the delay is over
		prob    float64
	)
	decide := func() {
		if gen != t.eouGen || t.userSpeaking {
			return
		}
		if ask && prob < s.opts.Turn.EndOfTurnThreshold {
			t.eouExtended = true
			wait := s.opts.Turn.MaxEndpointingDelay - s.opts.Turn.MinEndpointingDelay
			if s.opts.VAD != nil && !t.speechEndedAt.IsZero() {
				wait = s.opts.Turn.MaxEndpointingDelay - time.Since(t.speechEndedAt)
			}
			s.armEOU(wait)
			return
		}
		s.commitUserTurn()
	}

	if !t.eouExtended {
		// The reply is prepared during the delay too.
		s.preempt()
	}
	if ask {
		// The model runs during the delay, so that its answer costs no time
		// when the user is done, and off the loop: it must not stall audio.
		in := TurnInput{
			History: append(s.conv.Messages(), llm.Message{Role: llm.RoleUser, Content: strings.Join(t.finals, " ")}),
			Audio:   s.turnAudio.snapshot(),
		}
		// Tracked so that Close waits for the call to return. Adding to the
		// WaitGroup is safe here: the loop goroutine itself is tracked.
		s.goRun(func() {
			ctx, cancel := context.WithTimeout(s.ctx, s.opts.Timeouts.TurnDetector)
			defer cancel()
			p, err := s.opts.TurnDetector.EndOfTurnProbability(ctx, in)
			if err != nil {
				s.reportError(SourceTurnDetector, err, true)
				p = 1 // without an answer the silence decides
			}
			s.log.Debug("end-of-turn model", "probability", p, "finished", p >= s.opts.Turn.EndOfTurnThreshold,
				"audio", in.Audio.Duration().Round(10*time.Millisecond), "text", in.History[len(in.History)-1].Content)
			s.post(func() {
				prob, waiting = p, false
				if elapsed {
					decide()
				}
			})
		})
	}
	s.after(d, func() {
		elapsed = true
		if !waiting {
			decide()
		}
	})
}

// commitUserTurn turns the pending transcripts into a user message and
// requests a reply.
func (s *Session) commitUserTurn() {
	t := &s.turn
	text := strings.TrimSpace(strings.Join(t.finals, " "))
	t.finals, t.interim = nil, ""
	t.eouGen++
	t.wantInterrupt = false
	s.resetTurnAudio()
	pre := t.preempt
	t.preempt, t.preemptCount = nil, 0
	if pre != nil && pre.text != text {
		pre.discard()
		pre = nil
	}
	if text == "" {
		return
	}
	stoppedAt := t.speechEndedAt
	t.speechEndedAt = time.Time{}
	if !stoppedAt.IsZero() {
		s.emit(MetricsCollected{Metric: EndOfTurnMetrics{Delay: time.Since(stoppedAt)}})
	}
	// The user said something new: anything the agent is still saying or
	// about to say answers an older turn.
	if !s.opts.Turn.DisableInterruptions {
		t.held = nil
		s.interruptSpeeches(false)
	}
	// The message travels with the reply and is committed when the reply
	// starts, after the interrupted speech recorded what it played.
	msg := llm.Message{Role: llm.RoleUser, Content: text, CreatedAt: time.Now()}
	if _, err := s.enqueue(&SpeechHandle{
		source:             SpeechFromReply,
		allowInterruptions: !s.opts.Turn.DisableInterruptions,
		addToHistory:       true,
		userMsg:            &msg,
		preempt:            pre,
		preemptAt:          t.preemptAt,
		userStoppedAt:      stoppedAt,
	}); err != nil {
		if pre != nil {
			pre.discard()
		}
		if errors.Is(err, ErrSessionClosed) {
			return
		}
		// No reply will run (queue full): keep what the user said.
		s.commit(msg)
		s.reportError(SourceLLM, fmt.Errorf("queue reply: %w", err), true)
	}
}
