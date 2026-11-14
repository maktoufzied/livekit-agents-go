// Package realtime defines the extension point for speech-to-speech models:
// models that take audio in and produce audio out over one bidirectional
// connection, doing their own turn detection.
//
// Status: interfaces only. The session in package agents runs the
// STT → LLM → TTS pipeline and does not accept a realtime Model yet, and no
// provider implements these interfaces. They are published so that
// provider work can start against a stable shape; expect changes until a
// first implementation lands.
package realtime

import (
	"context"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/llm"
)

// Capabilities describes what a model does on its own.
type Capabilities struct {
	// ServerTurnDetection: the model detects user turns and interruptions.
	ServerTurnDetection bool
	// UserTranscription: the model emits transcripts of user audio.
	UserTranscription bool
	// AudioTruncation: the model supports Truncate.
	AudioTruncation bool
}

// SessionOptions configures a model session.
type SessionOptions struct {
	Instructions string
	Tools        []llm.ToolSpec
	// History seeds the conversation.
	History []llm.Message
}

// Model opens sessions. Implementations must be safe for concurrent use.
type Model interface {
	Capabilities() Capabilities
	// InputFormat and OutputFormat are the PCM formats of the session.
	InputFormat() audio.Format
	OutputFormat() audio.Format
	// Connect opens a session; ctx bounds its whole lifetime.
	Connect(ctx context.Context, opts SessionOptions) (Session, error)
}

// Session is one live connection.
//
// Ownership: the caller must call Close exactly once; Close unblocks Recv.
// One goroutine may send (PushAudio and the control methods) while another
// calls Recv. Events are delivered in the order the model produced them;
// Recv returns io.EOF after a clean close.
type Session interface {
	// PushAudio sends user audio. It may block under backpressure.
	PushAudio(f audio.Frame) error
	// GenerateReply asks for a response outside of server turn detection.
	GenerateReply(instructions string) error
	// SendToolResult answers a ToolCall event.
	SendToolResult(result llm.Message) error
	// Interrupt cancels the response being generated.
	Interrupt() error
	// Truncate tells the model how much of a response's audio was actually
	// played, so that its own history matches what the user heard.
	Truncate(responseID string, played time.Duration) error
	Recv() (Event, error)
	Close() error
}

// EventType identifies a session event.
type EventType int

const (
	// UserSpeechStarted and UserSpeechStopped come from server turn detection.
	UserSpeechStarted EventType = iota + 1
	UserSpeechStopped
	// UserTranscript carries a transcript of user audio in Text.
	UserTranscript
	// ResponseStarted opens a response identified by ResponseID.
	ResponseStarted
	// AudioDelta carries response audio in Audio.
	AudioDelta
	// TextDelta carries the transcript of response audio in Text.
	TextDelta
	// ToolCall carries a complete tool call in Call.
	ToolCall
	// ResponseDone closes a response; Usage may be set.
	ResponseDone
)

// Event is a session event.
type Event struct {
	Type       EventType
	ResponseID string
	Text       string
	Final      bool
	Audio      audio.Frame
	Call       llm.ToolCall
	Usage      *llm.Usage
}
