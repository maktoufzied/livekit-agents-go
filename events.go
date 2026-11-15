package agents

import (
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/maktoufzied/livekit-agents-go/llm"
)

// AgentState is the coarse state of the agent.
type AgentState string

const (
	AgentInitializing AgentState = "initializing"
	AgentListening    AgentState = "listening"
	AgentThinking     AgentState = "thinking"
	AgentSpeaking     AgentState = "speaking"
)

// UserState is the coarse state of the user.
type UserState string

const (
	UserListening UserState = "listening"
	UserSpeaking  UserState = "speaking"
)

// Event is implemented by every session event. Use a type switch.
type Event interface{ sessionEvent() }

// AgentStateChanged reports an agent state transition.
type AgentStateChanged struct{ Old, New AgentState }

// UserStateChanged reports a user state transition.
type UserStateChanged struct{ Old, New UserState }

// UserTranscript carries interim and final user transcripts.
type UserTranscript struct {
	Text     string
	Final    bool
	Language string
}

// AgentTranscript is emitted when a sentence starts playing.
type AgentTranscript struct {
	SpeechID string
	Text     string
}

// ConversationItemAdded is emitted when a message is committed to history.
type ConversationItemAdded struct{ Message llm.Message }

// SpeechSource says what produced a speech.
type SpeechSource string

const (
	SpeechFromReply SpeechSource = "reply" // LLM generation
	SpeechFromSay   SpeechSource = "say"   // fixed text
)

// SpeechStarted is emitted when a speech begins executing.
type SpeechStarted struct {
	SpeechID string
	Source   SpeechSource
}

// SpeechFinished is emitted when a speech ends, whatever the reason.
type SpeechFinished struct {
	SpeechID    string
	Interrupted bool
	// PlayedDuration is the audio the user actually heard.
	PlayedDuration time.Duration
	// Err is nil on normal completion and on interruption.
	Err error
}

// FalseInterruption is emitted when a speech that was paused because the
// user seemed to interrupt is resumed: the sound was not followed by any
// transcript.
type FalseInterruption struct {
	SpeechID string
	// Paused is how long the speech was on hold.
	Paused time.Duration
}

// ToolCallStarted is emitted before a tool handler runs.
type ToolCallStarted struct {
	SpeechID string
	Call     llm.ToolCall
}

// ToolCallFinished is emitted when a tool call ends.
type ToolCallFinished struct {
	SpeechID string
	Call     llm.ToolCall
	// Output is what was sent to the model.
	Output   string
	IsError  bool
	Err      error // the handler's raw error, never sent to the model as-is
	Duration time.Duration
}

// MetricsCollected carries one latency or usage measurement.
type MetricsCollected struct{ Metric Metric }

// ErrorSource identifies the pipeline stage an ErrorEvent comes from.
type ErrorSource string

const (
	SourceSTT          ErrorSource = "stt"
	SourceLLM          ErrorSource = "llm"
	SourceTTS          ErrorSource = "tts"
	SourceInput        ErrorSource = "input"  // audio input or VAD
	SourceOutput       ErrorSource = "output" // audio output
	SourceTurnDetector ErrorSource = "turn_detector"
	SourceSpeech       ErrorSource = "speech" // none of the above
)

// ErrorEvent reports a pipeline error. Recoverable errors are retried or
// only affect the current speech; a non-recoverable one closes the session.
// Tool failures are reported by ToolCallFinished, not here.
type ErrorEvent struct {
	Source      ErrorSource
	Err         error
	Recoverable bool
}

// CloseReason says why a session closed.
type CloseReason string

const (
	CloseRequested     CloseReason = "requested"      // Close or Shutdown was called
	CloseContextDone   CloseReason = "context_done"   // the Start context was cancelled
	CloseError         CloseReason = "error"          // unrecoverable pipeline error
	CloseInputFinished CloseReason = "input_finished" // reserved for transports
)

// SessionClosed is the last event of a session.
type SessionClosed struct {
	Reason CloseReason
	Err    error
}

func (AgentStateChanged) sessionEvent()     {}
func (UserStateChanged) sessionEvent()      {}
func (UserTranscript) sessionEvent()        {}
func (AgentTranscript) sessionEvent()       {}
func (ConversationItemAdded) sessionEvent() {}
func (SpeechStarted) sessionEvent()         {}
func (SpeechFinished) sessionEvent()        {}
func (FalseInterruption) sessionEvent()     {}
func (ToolCallStarted) sessionEvent()       {}
func (ToolCallFinished) sessionEvent()      {}
func (MetricsCollected) sessionEvent()      {}
func (ErrorEvent) sessionEvent()            {}
func (SessionClosed) sessionEvent()         {}

// eventBus delivers events to handlers in emission order from a single
// goroutine. Emitting never blocks the pipeline.
//
// Lifecycle, history, speech, tool and error events are never dropped.
// Metrics and interim transcripts are dropped (and counted) once more than
// limit events are waiting, so a stuck handler cannot make the backlog of
// high-frequency events grow without bound.
type eventBus struct {
	log *slog.Logger

	hmu      sync.RWMutex
	handlers map[int]func(Event)
	nextID   int

	mu      sync.Mutex
	cond    *sync.Cond
	q       []Event
	closed  bool
	limit   int
	done    chan struct{}
	dropped atomic.Int64
}

func newEventBus(limit int, log *slog.Logger) *eventBus {
	b := &eventBus{log: log, handlers: map[int]func(Event){}, limit: limit, done: make(chan struct{})}
	b.cond = sync.NewCond(&b.mu)
	go b.run()
	return b
}

// droppable reports whether an event may be discarded under backlog.
func droppable(ev Event) bool {
	switch e := ev.(type) {
	case MetricsCollected:
		return true
	case UserTranscript:
		return !e.Final
	}
	return false
}

func (b *eventBus) run() {
	defer close(b.done)
	for {
		b.mu.Lock()
		for len(b.q) == 0 && !b.closed {
			b.cond.Wait()
		}
		if len(b.q) == 0 {
			b.mu.Unlock()
			return
		}
		ev := b.q[0]
		b.q[0] = nil
		b.q = b.q[1:]
		b.mu.Unlock()

		b.hmu.RLock()
		hs := make([]func(Event), 0, len(b.handlers))
		for id := 0; id < b.nextID; id++ {
			if h, ok := b.handlers[id]; ok {
				hs = append(hs, h)
			}
		}
		b.hmu.RUnlock()
		for _, h := range hs {
			b.call(h, ev)
		}
	}
}

func (b *eventBus) call(h func(Event), ev Event) {
	defer func() {
		if r := recover(); r != nil {
			b.log.Error("event handler panicked", "panic", r)
		}
	}()
	h(ev)
}

func (b *eventBus) on(h func(Event)) func() {
	b.hmu.Lock()
	defer b.hmu.Unlock()
	id := b.nextID
	b.nextID++
	b.handlers[id] = h
	return func() {
		b.hmu.Lock()
		defer b.hmu.Unlock()
		delete(b.handlers, id)
	}
}

func (b *eventBus) emit(ev Event) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return
	}
	if len(b.q) >= b.limit && droppable(ev) {
		if b.dropped.Add(1) == 1 {
			b.log.Warn("event backlog full, dropping metrics and interim transcripts: an event handler is too slow")
		}
		return
	}
	b.q = append(b.q, ev)
	b.cond.Signal()
}

// close delivers the queued events to the handlers and stops the dispatcher.
func (b *eventBus) close() {
	b.mu.Lock()
	b.closed = true
	b.cond.Broadcast()
	b.mu.Unlock()
	<-b.done
}
