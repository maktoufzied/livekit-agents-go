// Package stt defines the streaming speech-to-text interface.
package stt

import (
	"context"
	"errors"
	"io"
	"iter"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
)

// EventType identifies a recognition event.
type EventType int

const (
	// Interim carries a transcript that may still change.
	Interim EventType = iota + 1
	// Final carries a transcript segment that will not change. A user turn
	// may be made of several Final events.
	Final
	// SpeechStarted reports that the provider detected the start of speech.
	// Optional.
	SpeechStarted
	// EndOfSpeech reports that the provider's own endpointing decided the
	// utterance is over. Optional.
	EndOfSpeech
)

// Event is a recognition result.
type Event struct {
	Type       EventType
	Text       string
	Language   string
	Confidence float64
	// Start and End locate the segment in the audio stream when the
	// provider reports timing; both are zero otherwise.
	Start, End time.Duration
}

// StreamOptions are per-stream settings.
type StreamOptions struct {
	// Format is the format of every frame that will be pushed.
	Format audio.Format
}

// STT opens recognition streams. Implementations must be safe for
// concurrent use.
type STT interface {
	// Stream opens a recognition stream. The context bounds the lifetime
	// of the whole stream, not only the handshake.
	Stream(ctx context.Context, opts StreamOptions) (Stream, error)
}

// Stream is one recognition session.
//
// Ownership: the caller owns the stream and must call Close, which releases
// the connection and unblocks Push and Recv. Close is idempotent.
//
// Cancellation: the context passed to STT.Stream bounds the whole stream.
// Push and Recv take no context of their own and return once that context
// is cancelled or Close is called.
//
// Concurrency: one goroutine may call Push/CloseSend while another calls
// Recv. Push may block when the provider applies backpressure.
//
// Ordering: events are delivered by Recv in the order the provider produced
// them. Recv returns io.EOF after CloseSend once all results are drained, and
// a non-nil error (a *providers.Error for provider failures) otherwise.
type Stream interface {
	Push(frame audio.Frame) error
	// CloseSend signals that no more audio will be pushed. Results for
	// audio already pushed are still delivered.
	CloseSend() error
	Recv() (Event, error)
	Close() error
}

// Events returns an iterator over the stream's events. Iteration ends at
// io.EOF; any other error is yielded once, as the last element. It does not
// close the stream.
func Events(s Stream) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		for {
			v, err := s.Recv()
			if errors.Is(err, io.EOF) {
				return
			}
			if !yield(v, err) || err != nil {
				return
			}
		}
	}
}
