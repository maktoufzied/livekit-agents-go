package agents

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/maktoufzied/livekit-agents-go/llm"
)

// SpeechHandle tracks one agent utterance: a model reply or a Say.
type SpeechHandle struct {
	id     string
	source SpeechSource

	ctx    context.Context
	cancel context.CancelFunc

	allowInterruptions bool
	addToHistory       bool
	text               string // Say
	instructions       string // GenerateReply
	// userMsg is the user message this reply answers. It is committed to
	// the history when the speech reaches the head of the queue.
	userMsg   *llm.Message
	createdAt time.Time
	// preempt is the model request started before the turn ended, if any.
	// The speech goroutine takes it for the first model step or discards it.
	preempt   *preemptive
	preemptAt time.Time

	interrupted   atomic.Bool
	interruptOnce sync.Once
	interruptedAt time.Time

	doneOnce sync.Once
	done     chan struct{}
	err      error

	hold holdGate
}

// holdGate pauses the playout of a speech while the session finds out
// whether the user really interrupted. The loop goroutine sets it, the
// playout goroutine obeys it.
type holdGate struct {
	mu      sync.Mutex
	held    bool
	changed chan struct{}      // closed on every change of held
	abort   context.CancelFunc // cancels the output write in flight
}

// set changes the state and reports whether it changed. Holding aborts the
// write in flight, so that playout stops at once.
func (g *holdGate) set(held bool) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.held == held {
		return false
	}
	g.held = held
	if g.changed != nil {
		close(g.changed)
		g.changed = nil
	}
	if held && g.abort != nil {
		g.abort()
	}
	return true
}

// state returns whether playout is held and a channel closed on the next
// change.
func (g *holdGate) state() (held bool, changed <-chan struct{}) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.changed == nil {
		g.changed = make(chan struct{})
	}
	return g.held, g.changed
}

// beginWrite registers the cancel function of an output write. It returns
// false, without registering, when playout is held.
func (g *holdGate) beginWrite(abort context.CancelFunc) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.held {
		return false
	}
	g.abort = abort
	return true
}

func (g *holdGate) endWrite() {
	g.mu.Lock()
	g.abort = nil
	g.mu.Unlock()
}

// ID returns the unique speech identifier.
func (h *SpeechHandle) ID() string { return h.id }

// Done is closed when the speech finished, was interrupted, or failed.
func (h *SpeechHandle) Done() <-chan struct{} { return h.done }

// Wait blocks until the speech is done or ctx is cancelled. It returns nil
// on normal completion, ErrInterrupted when the speech was cut short, and
// the pipeline error otherwise.
func (h *SpeechHandle) Wait(ctx context.Context) error {
	select {
	case <-h.done:
		return h.err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Interrupted reports whether the speech was interrupted.
func (h *SpeechHandle) Interrupted() bool { return h.interrupted.Load() }

// Interrupt stops the speech, even when it does not allow user
// interruptions. It is safe to call from any goroutine, any number of times.
func (h *SpeechHandle) Interrupt() {
	h.interruptOnce.Do(func() {
		h.interruptedAt = time.Now()
		h.interrupted.Store(true)
		h.cancel()
	})
}

func (h *SpeechHandle) finish(err error) {
	h.doneOnce.Do(func() {
		h.err = err
		h.cancel()
		close(h.done)
	})
}
