package agentstest

import (
	"context"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/providers"
	"github.com/maktoufzied/livekit-agents-go/stt"
)

// EnergySTT is a fake recognizer driven by real audio: every burst of loud
// audio followed by silence yields one final transcript taken from Script
// (or "utterance N" once the script is exhausted). It lets the whole
// pipeline run over a real transport without a speech provider.
type EnergySTT struct {
	// Script holds the transcripts returned, in order.
	Script []string
	// MinSpeech is the loud audio needed for an utterance. Default 200ms.
	MinSpeech time.Duration
	// Silence is the trailing silence that ends an utterance. Default 300ms.
	Silence time.Duration
	// Threshold is the RMS level counted as speech. Default 0.02.
	Threshold float64

	mu sync.Mutex
	n  int
}

// Stream implements stt.STT.
func (e *EnergySTT) Stream(ctx context.Context, _ stt.StreamOptions) (stt.Stream, error) {
	return &energyStream{parent: e, ctx: ctx, events: make(chan stt.Event, 64), closed: make(chan struct{}), ended: make(chan struct{})}, nil
}

func (e *EnergySTT) next() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.n++
	if e.n <= len(e.Script) {
		return e.Script[e.n-1]
	}
	return fmt.Sprintf("utterance %d", e.n)
}

type energyStream struct {
	parent          *EnergySTT
	ctx             context.Context
	events          chan stt.Event
	speech, silence time.Duration
	closeOnce       sync.Once
	endOnce         sync.Once
	closed, ended   chan struct{}
}

func (s *energyStream) emit(ev stt.Event) {
	select {
	case s.events <- ev:
	default:
	}
}

func (s *energyStream) Push(f audio.Frame) error {
	select {
	case <-s.closed:
		return fmt.Errorf("agentstest: %w", providers.ErrStreamClosed)
	default:
	}
	minSpeech, silence, threshold := s.parent.MinSpeech, s.parent.Silence, s.parent.Threshold
	if minSpeech <= 0 {
		minSpeech = 200 * time.Millisecond
	}
	if silence <= 0 {
		silence = 300 * time.Millisecond
	}
	if threshold <= 0 {
		threshold = 0.02
	}
	if audio.RMS(f) >= threshold {
		if s.speech == 0 {
			s.emit(stt.Event{Type: stt.SpeechStarted})
		}
		s.speech += f.Duration()
		s.silence = 0
		return nil
	}
	if s.speech == 0 {
		return nil
	}
	s.silence += f.Duration()
	if s.silence >= silence {
		if s.speech >= minSpeech {
			s.emit(stt.Event{Type: stt.Final, Text: s.parent.next(), Confidence: 1})
			s.emit(stt.Event{Type: stt.EndOfSpeech})
		}
		s.speech, s.silence = 0, 0
	}
	return nil
}

func (s *energyStream) CloseSend() error {
	s.endOnce.Do(func() { close(s.ended) })
	return nil
}

func (s *energyStream) Recv() (stt.Event, error) {
	select {
	case ev := <-s.events:
		return ev, nil
	case <-s.ended:
		return stt.Event{}, io.EOF
	case <-s.closed:
		return stt.Event{}, fmt.Errorf("agentstest: %w", providers.ErrStreamClosed)
	case <-s.ctx.Done():
		return stt.Event{}, s.ctx.Err()
	}
}

func (s *energyStream) Close() error {
	s.closeOnce.Do(func() { close(s.closed) })
	return nil
}
