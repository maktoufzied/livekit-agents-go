// Package llm defines the chat model interface, the conversation history
// and tool definitions.
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"sync"
	"time"
)

// Role is the author of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ToolCall is a structured request from the model to run a tool.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// Message is one entry of the conversation history.
type Message struct {
	ID      string `json:"id,omitempty"`
	Role    Role   `json:"role"`
	Content string `json:"content,omitempty"`
	// ToolCalls is set on assistant messages that request tools.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID and Name are set on tool messages (the result of a call).
	ToolCallID string `json:"tool_call_id,omitempty"`
	Name       string `json:"name,omitempty"`
	// IsError marks a tool result that reports a failure.
	IsError bool `json:"is_error,omitempty"`
	// Interrupted marks an assistant message cut short by the user. Its
	// Content holds only what was actually played.
	Interrupted bool      `json:"interrupted,omitempty"`
	CreatedAt   time.Time `json:"created_at,omitempty"`
}

// Conversation is a role-based message history, safe for concurrent use.
//
// A session reads the conversation when it starts a model request and
// appends to it as turns are committed. The application may append too (for
// example to inject context); a message appended while a reply is being
// generated is seen by the next model request, not the current one.
type Conversation struct {
	mu   sync.RWMutex
	msgs []Message
}

// NewConversation returns a conversation seeded with msgs.
func NewConversation(msgs ...Message) *Conversation {
	c := &Conversation{}
	c.Append(msgs...)
	return c
}

// Append adds messages at the end of the history.
func (c *Conversation) Append(msgs ...Message) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.msgs = append(c.msgs, msgs...)
}

// Messages returns a copy of the history.
func (c *Conversation) Messages() []Message {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return append([]Message(nil), c.msgs...)
}

// Len returns the number of messages.
func (c *Conversation) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.msgs)
}

// ToolSpec is the model-facing description of a tool.
type ToolSpec struct {
	Name        string
	Description string
	// Parameters is a JSON Schema object describing the arguments.
	Parameters json.RawMessage
}

// ToolChoice controls whether the model may call tools.
type ToolChoice string

const (
	ToolChoiceAuto     ToolChoice = ""         // model decides
	ToolChoiceNone     ToolChoice = "none"     // tools are not offered for this request
	ToolChoiceRequired ToolChoice = "required" // model must call a tool
)

// Request is one model invocation. Instructions are passed as leading
// messages with RoleSystem; providers map them to their native field.
type Request struct {
	Messages   []Message
	Tools      []ToolSpec
	ToolChoice ToolChoice
}

// Usage reports token consumption when the provider exposes it.
type Usage struct {
	InputTokens  int
	OutputTokens int
}

// Chunk is an increment of model output. Text holds a delta. ToolCalls holds
// calls whose arguments are complete: providers buffer partial argument
// deltas and emit each call exactly once.
type Chunk struct {
	Text      string
	ToolCalls []ToolCall
	Usage     *Usage
}

// LLM is a streaming chat model. Implementations must be safe for
// concurrent use.
type LLM interface {
	// Chat starts a generation. The context bounds the whole stream.
	Chat(ctx context.Context, req Request) (Stream, error)
}

// Stream yields the output of one generation.
//
// Ownership: the caller must call Close, which aborts the request and
// unblocks Recv. Close is idempotent. The context passed to Chat bounds the
// whole stream. Recv is called from a single goroutine, returns chunks in
// generation order, io.EOF at the normal end, and a non-nil error (a
// *providers.Error for provider failures) otherwise.
type Stream interface {
	Recv() (Chunk, error)
	Close() error
}

// Chunks returns an iterator over the stream's output. Iteration ends at
// io.EOF; any other error is yielded once, as the last element. It does not
// close the stream.
//
//	stream, err := model.Chat(ctx, req)
//	if err != nil {
//		return err
//	}
//	defer stream.Close()
//	for chunk, err := range llm.Chunks(stream) {
//		if err != nil {
//			return err
//		}
//		fmt.Print(chunk.Text)
//	}
func Chunks(s Stream) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
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
