package agents

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers"
	"github.com/maktoufzied/livekit-agents-go/stt"
	"github.com/maktoufzied/livekit-agents-go/tts"
	"github.com/maktoufzied/livekit-agents-go/vad"
)

// Agent is the personality of a session: its instructions and its tools.
type Agent struct {
	// Instructions is the system prompt.
	Instructions string
	// Tools are the functions the model may call.
	Tools []llm.Tool
}

// AudioInput is the source of user audio.
//
// The session is the only reader. ReadFrame blocks until a frame is
// available, returns io.EOF when the source ended for good (participant
// left) and ctx.Err() when cancelled. Implementations must bound their own
// buffering and drop the oldest audio rather than block the network.
type AudioInput interface {
	Format() audio.Format
	ReadFrame(ctx context.Context) (audio.Frame, error)
}

// AudioOutput is the sink for agent audio.
//
// The session guarantees a single writer at a time and never interleaves
// audio from two speeches.
//
// WriteFrame provides backpressure and pacing: it returns once the frame is
// queued for playback and blocks while more than a small look-ahead is
// already queued, so that the amount of audio that can still be cancelled
// stays bounded. Flush blocks until everything written has been played.
// Clear drops audio that was queued but not played yet and returns its
// duration; the session subtracts it to know what the user really heard.
type AudioOutput interface {
	Format() audio.Format
	WriteFrame(ctx context.Context, f audio.Frame) error
	Flush(ctx context.Context) error
	Clear() time.Duration
}

// IO binds a session to its audio transport. Input may be nil for an
// output-only session driven by Say and GenerateReply.
type IO struct {
	Input  AudioInput
	Output AudioOutput
}

// TurnMode selects how the end of a user turn is detected.
type TurnMode int

const (
	// TurnAuto uses TurnVAD when a VAD is configured and TurnSTT otherwise.
	TurnAuto TurnMode = iota
	// TurnVAD ends the turn when the VAD reports silence and a final
	// transcript is available, after the endpointing delay.
	TurnVAD
	// TurnSTT relies on the STT provider's own endpointing (stt.EndOfSpeech),
	// with MaxEndpointingDelay after the last final transcript as fallback.
	TurnSTT
	// TurnManual never ends a turn automatically; call Session.CommitUserTurn.
	TurnManual
)

func (m TurnMode) String() string {
	switch m {
	case TurnAuto:
		return "auto"
	case TurnVAD:
		return "vad"
	case TurnSTT:
		return "stt"
	case TurnManual:
		return "manual"
	}
	return fmt.Sprintf("TurnMode(%d)", int(m))
}

// TurnDetector is an optional end-of-turn model. When set, the session asks
// it once the minimum endpointing delay elapsed; a probability below
// TurnOptions.EndOfTurnThreshold postpones the turn end up to
// MaxEndpointingDelay. The SDK ships no implementation.
type TurnDetector interface {
	// EndOfTurnProbability returns the probability in [0, 1] that the user
	// finished speaking. history ends with the pending user message.
	EndOfTurnProbability(ctx context.Context, history []llm.Message) (float64, error)
}

// TurnOptions configures turn detection and interruptions.
type TurnOptions struct {
	Mode TurnMode
	// MinEndpointingDelay is the silence required after the user stops
	// before the turn is committed. Default 500ms.
	MinEndpointingDelay time.Duration
	// MaxEndpointingDelay is the longest wait when the TurnDetector thinks
	// the user is not done, and the fallback in TurnSTT. Default 3s.
	MaxEndpointingDelay time.Duration
	// EndOfTurnThreshold is the TurnDetector probability under which the
	// turn end is postponed. Default 0.5.
	EndOfTurnThreshold float64
	// DisableInterruptions prevents the user from interrupting the agent.
	DisableInterruptions bool
	// MinInterruptionDuration is the user speech required to interrupt the
	// agent. Default 500ms.
	MinInterruptionDuration time.Duration
	// MinInterruptionWords additionally requires that many transcribed
	// words before interrupting. Default 0 (disabled).
	MinInterruptionWords int
}

// ToolOptions configures tool execution.
type ToolOptions struct {
	// MaxSteps bounds the model/tool round trips of one reply. After that
	// many rounds the model is asked to answer without tools. Default 3.
	MaxSteps int
	// DefaultTimeout bounds a tool call that sets no timeout. Default 30s.
	DefaultTimeout time.Duration
	// Sequential runs the calls of one model step one after the other, in
	// order. By default they run concurrently.
	Sequential bool
}

// TimeoutOptions bounds provider latency.
type TimeoutOptions struct {
	// LLMFirstToken is the longest wait for the first model output. Default 15s.
	LLMFirstToken time.Duration
	// TTSFirstAudio is the longest wait for the first audio of a sentence. Default 10s.
	TTSFirstAudio time.Duration
	// TurnDetector bounds one TurnDetector call. Default 1s.
	TurnDetector time.Duration
}

// RetryOptions configures bounded retries. LLM and TTS requests are retried
// only when they fail before producing any output and the error is flagged
// retryable; tool calls are never retried.
type RetryOptions struct {
	// MaxRetries is the number of extra attempts. Default 2; negative disables.
	MaxRetries int
	// Backoff is the first delay; it doubles on every attempt. Default 200ms.
	Backoff time.Duration
	// STTMaxReconnects bounds consecutive failed attempts to (re)open the
	// STT stream before the session closes with an error. Default 3.
	STTMaxReconnects int
}

// BufferOptions bounds every queue of the pipeline.
type BufferOptions struct {
	// STTFrames is the number of audio frames buffered towards the STT
	// provider. When full the oldest frame is dropped. Default 150.
	STTFrames int
	// Sentences is the number of sentences buffered between the LLM and
	// TTS; when full, reading from the LLM stream blocks. Default 8.
	Sentences int
	// SynthesisLookahead is how many sentences may be synthesized ahead of
	// the one being played. Default 2.
	SynthesisLookahead int
	// SentenceFrames is the number of audio frames buffered per sentence;
	// when full, reading from the TTS stream blocks. Default 512.
	SentenceFrames int
	// Speeches is the maximum number of queued speeches. Default 16.
	Speeches int
	// Events is the event queue size. Default 1024.
	Events int
}

// SessionOptions configures a Session. Zero values select the defaults.
type SessionOptions struct {
	Agent *Agent
	// LLM and TTS are required. STT is required when the session has an
	// audio input. VAD is optional but recommended: without it,
	// interruptions and turn ends rely on STT events only.
	STT stt.STT
	LLM llm.LLM
	TTS tts.TTS
	VAD vad.VAD

	TurnDetector TurnDetector
	Turn         TurnOptions
	Tools        ToolOptions
	Timeouts     TimeoutOptions
	Retry        RetryOptions
	Buffers      BufferOptions

	// History seeds the conversation. The session appends to it.
	History *llm.Conversation
	// Logger defaults to slog.Default().
	Logger *slog.Logger
}

func (o *SessionOptions) setDefaults() error {
	if o.Agent == nil {
		o.Agent = &Agent{}
	}
	if o.LLM == nil {
		return errors.New("agents: SessionOptions.LLM is required")
	}
	if o.TTS == nil {
		return errors.New("agents: SessionOptions.TTS is required")
	}
	if !o.TTS.Format().Valid() {
		return errors.New("agents: TTS reports an invalid audio format")
	}
	seen := map[string]bool{}
	for _, t := range o.Agent.Tools {
		if err := t.Validate(); err != nil {
			return err
		}
		if seen[t.Name] {
			return fmt.Errorf("agents: duplicate tool name %q", t.Name)
		}
		seen[t.Name] = true
	}
	def := func(d *time.Duration, v time.Duration) {
		if *d <= 0 {
			*d = v
		}
	}
	defInt := func(d *int, v int) {
		if *d <= 0 {
			*d = v
		}
	}
	def(&o.Turn.MinEndpointingDelay, 500*time.Millisecond)
	def(&o.Turn.MaxEndpointingDelay, 3*time.Second)
	if o.Turn.MaxEndpointingDelay < o.Turn.MinEndpointingDelay {
		o.Turn.MaxEndpointingDelay = o.Turn.MinEndpointingDelay
	}
	def(&o.Turn.MinInterruptionDuration, 500*time.Millisecond)
	if o.Turn.EndOfTurnThreshold <= 0 {
		o.Turn.EndOfTurnThreshold = 0.5
	}
	if o.Turn.Mode == TurnAuto {
		if o.VAD != nil {
			o.Turn.Mode = TurnVAD
		} else {
			o.Turn.Mode = TurnSTT
		}
	}
	if o.Turn.Mode == TurnVAD && o.VAD == nil {
		return errors.New("agents: TurnVAD requires SessionOptions.VAD")
	}
	defInt(&o.Tools.MaxSteps, 3)
	def(&o.Tools.DefaultTimeout, 30*time.Second)
	def(&o.Timeouts.LLMFirstToken, 15*time.Second)
	def(&o.Timeouts.TTSFirstAudio, 10*time.Second)
	def(&o.Timeouts.TurnDetector, time.Second)
	if o.Retry.MaxRetries == 0 {
		o.Retry.MaxRetries = 2
	} else if o.Retry.MaxRetries < 0 {
		o.Retry.MaxRetries = 0
	}
	def(&o.Retry.Backoff, 200*time.Millisecond)
	defInt(&o.Retry.STTMaxReconnects, 3)
	defInt(&o.Buffers.STTFrames, 150)
	defInt(&o.Buffers.Sentences, 8)
	defInt(&o.Buffers.SynthesisLookahead, 2)
	defInt(&o.Buffers.SentenceFrames, 512)
	defInt(&o.Buffers.Speeches, 16)
	defInt(&o.Buffers.Events, 1024)
	if o.History == nil {
		o.History = llm.NewConversation()
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	return nil
}

// Secret wraps a credential so it cannot leak through fmt or slog.
type Secret = providers.Secret
