package agents

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
)

var (
	// ErrSessionClosed is returned by operations on a closed session.
	ErrSessionClosed = errors.New("agents: session closed")
	// ErrNotStarted is returned when the session has not been started.
	ErrNotStarted = errors.New("agents: session not started")
	// ErrAlreadyStarted is returned by a second call to Session.Start.
	ErrAlreadyStarted = errors.New("agents: session already started")
	// ErrSpeechQueueFull is returned when too many speeches are queued.
	ErrSpeechQueueFull = errors.New("agents: speech queue full")
	// ErrInterrupted is the error of a speech cut short by an interruption.
	ErrInterrupted = errors.New("agents: speech interrupted")
	// ErrRequestAnswered is returned by JobRequest.Accept and Reject when
	// the request was already accepted or rejected.
	ErrRequestAnswered = errors.New("agents: job request already answered")
	// ErrAlreadyConnected is returned by a second call to Job.Connect.
	ErrAlreadyConnected = errors.New("agents: job already connected to its room")
	// ErrWorkerStopping is returned when a job is accepted while the worker
	// is shutting down.
	ErrWorkerStopping = errors.New("agents: worker is stopping")
)

// TimeoutError reports that a pipeline stage exceeded a configured timeout.
// It is treated as retryable where retries apply.
type TimeoutError struct {
	Op string // e.g. "llm first token"
	// After is the timeout that elapsed.
	After time.Duration
}

func (e *TimeoutError) Error() string {
	return fmt.Sprintf("agents: %s timed out after %s", e.Op, e.After)
}

// Timeout reports true, so that a TimeoutError satisfies the usual
// interface{ Timeout() bool } check.
func (e *TimeoutError) Timeout() bool { return true }

// Is makes errors.Is(err, context.DeadlineExceeded) true.
func (e *TimeoutError) Is(target error) bool { return target == context.DeadlineExceeded }

// FormatError reports an audio format mismatch between pipeline stages.
type FormatError struct {
	Stage     ErrorSource
	Want, Got audio.Format
}

func (e *FormatError) Error() string {
	return fmt.Sprintf("agents: %s audio format mismatch: want %s, got %s", e.Stage, fmtFormat(e.Want), fmtFormat(e.Got))
}

// RegistrationError reports that the worker could not register with the
// LiveKit server after all retries.
type RegistrationError struct {
	Attempts int
	Err      error
}

func (e *RegistrationError) Error() string {
	return fmt.Sprintf("agents: worker failed to connect to LiveKit after %d attempts: %v", e.Attempts, e.Err)
}

func (e *RegistrationError) Unwrap() error { return e.Err }
