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
// With a plain TTS the session splits model output into sentences and calls
// Synthesize once per sentence, in order, with a small look-ahead. A TTS
// that also implements Streamer receives the text as the model produces
// it, which starts the audio sooner.
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

// Streamer is a TTS that accepts text incrementally, usually over a
// persistent connection. The session prefers it over Synthesize: it opens
// one stream per model response, pushes the text as it arrives and flushes
// at the end.
type Streamer interface {
	TTS
	// SynthesizeStream opens a stream. The context bounds its whole life.
	SynthesizeStream(ctx context.Context) (InputStream, error)
}

// InputStream is a synthesis stream fed with text.
//
// Push and Flush are called from one goroutine, Recv from another. Recv
// returns audio as it is produced, also before Flush, and io.EOF once the
// audio of everything pushed before Flush was returned. Close aborts the
// stream and unblocks Recv; it is idempotent and must always be called.
type InputStream interface {
	Stream
	// Push adds text. The text may be cut anywhere, even inside a word.
	Push(text string) error
	// Flush marks the end of the text. Nothing may be pushed after it.
	Flush() error
}

// Prewarmer is implemented by a TTS that can open its connection ahead of
// the first request. The session calls it when it starts.
type Prewarmer interface {
	Prewarm()
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
