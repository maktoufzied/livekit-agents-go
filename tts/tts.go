// Package tts defines the text-to-speech interface.
package tts

import (
	"context"
	"errors"
	"io"
	"iter"

	"github.com/maktoufzied/livekit-agents-go/audio"
)

// TTS synthesizes speech. Implementations must be safe for concurrent use.
//
// The session splits model output into sentences and calls Synthesize once
// per sentence, in order, with a small look-ahead. Synthesizing per sentence
// lets the session know exactly which text each audio frame belongs to,
// which is what makes interruption-accurate history possible. Providers
// with a persistent connection may pool it behind Synthesize.
type TTS interface {
	// Format is the PCM format of every frame returned by any stream.
	Format() audio.Format
	// Synthesize starts synthesizing text. The context bounds the whole
	// stream. Audio should be returned incrementally, as it is produced.
	Synthesize(ctx context.Context, text string) (Stream, error)
}

// Stream yields the audio of one synthesis request.
//
// Ownership: the caller must call Close, which aborts the request and
// unblocks Recv. Close is idempotent. The context passed to Synthesize
// bounds the whole stream. Recv is called from a single goroutine and
// returns io.EOF after the last frame.
type Stream interface {
	Recv() (audio.Frame, error)
	Close() error
}

// Frames returns an iterator over the stream's audio. Iteration ends at
// io.EOF; any other error is yielded once, as the last element. It does not
// close the stream.
func Frames(s Stream) iter.Seq2[audio.Frame, error] {
	return func(yield func(audio.Frame, error) bool) {
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
