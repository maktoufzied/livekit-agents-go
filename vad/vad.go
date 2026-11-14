// Package vad defines the voice-activity-detection interface used for turn
// detection and interruptions, plus a dependency-free energy detector.
package vad

import (
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
)

// EventType identifies a VAD transition.
type EventType int

const (
	// SpeechStart is emitted once speech has lasted MinSpeechDuration.
	SpeechStart EventType = iota + 1
	// SpeechEnd is emitted once silence has lasted MinSilenceDuration.
	SpeechEnd
)

func (t EventType) String() string {
	switch t {
	case SpeechStart:
		return "speech_start"
	case SpeechEnd:
		return "speech_end"
	}
	return "unknown"
}

// Event is a speech boundary.
type Event struct {
	Type EventType
	// SpeechDuration is the speech observed so far (SpeechStart) or the
	// length of the finished utterance (SpeechEnd).
	SpeechDuration time.Duration
	// SilenceDuration is the trailing silence that confirmed a SpeechEnd.
	SilenceDuration time.Duration
}

// VAD creates detectors. Implementations must be safe for concurrent use.
type VAD interface {
	// NewDetector returns a detector for one audio stream.
	NewDetector() (Detector, error)
}

// Detector processes one audio stream. It is owned by a single goroutine.
//
// Process is synchronous: it returns the events caused by the frame, in
// order. It must not retain the frame. Close releases model resources.
type Detector interface {
	Process(f audio.Frame) ([]Event, error)
	Close() error
}
