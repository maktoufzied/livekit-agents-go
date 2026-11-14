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

	interrupted   atomic.Bool
	interruptOnce sync.Once
	interruptedAt time.Time

	doneOnce sync.Once
	done     chan struct{}
	err      error
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
