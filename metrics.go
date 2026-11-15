package agents

import (
	"time"

	"github.com/maktoufzied/livekit-agents-go/llm"
)

// Metric is implemented by every measurement carried by MetricsCollected.
type Metric interface{ metric() }

// TranscriptionMetrics measures speech-to-text latency for one final
// transcript.
type TranscriptionMetrics struct {
	// Delay is the time between the end of user speech (VAD) and the final
	// transcript. Zero when no VAD is configured.
	Delay time.Duration
}

// EndOfTurnMetrics is emitted when a user turn is committed.
type EndOfTurnMetrics struct {
	// Delay is the time between the end of user speech and the commit.
	Delay time.Duration
}

// LLMMetrics measures one model request.
type LLMMetrics struct {
	SpeechID string
	// FirstToken is the time to the first output chunk.
	FirstToken time.Duration
	Duration   time.Duration
	Usage      llm.Usage
	Attempts   int
	// Preemptive is how long before the reply started its model request
	// was sent, when it was sent ahead of the end of the user turn; zero
	// otherwise. FirstToken and Duration count from the start of the reply.
	Preemptive time.Duration
}

// TTSMetrics measures the synthesis of one sentence.
type TTSMetrics struct {
	SpeechID string
	// FirstAudio is the time to the first audio frame.
	FirstAudio    time.Duration
	Duration      time.Duration
	AudioDuration time.Duration
	Characters    int
	Attempts      int
}

// InterruptionMetrics measures one interruption.
type InterruptionMetrics struct {
	SpeechID string
	// Latency is the time between the interruption decision and the moment
	// queued audio was dropped from the output.
	Latency time.Duration
	// Dropped is the queued audio that was discarded.
	Dropped time.Duration
}

// ResponseMetrics measures the user-perceived response time of a reply.
type ResponseMetrics struct {
	SpeechID string
	// FirstAudio is the time between the reply being requested (end of the
	// user turn) and its first audio frame being queued for playback.
	FirstAudio time.Duration
}

func (TranscriptionMetrics) metric() {}
func (EndOfTurnMetrics) metric()     {}
func (LLMMetrics) metric()           {}
func (TTSMetrics) metric()           {}
func (InterruptionMetrics) metric()  {}
func (ResponseMetrics) metric()      {}

// SessionStats are cumulative counters of a session.
type SessionStats struct {
	// DroppedInputFrames counts audio frames dropped because the STT
	// provider could not keep up.
	DroppedInputFrames int64
	// DroppedEvents counts events dropped because handlers were too slow.
	DroppedEvents int64
}
