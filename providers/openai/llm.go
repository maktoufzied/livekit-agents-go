package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers"
)

// DefaultLLMModel is the chat model used when LLMOptions.Model is empty.
const DefaultLLMModel = "gpt-6-luna"

// LLMOptions configures an LLM.
type LLMOptions struct {
	// APIKey authenticates the requests. It defaults to the OPENAI_API_KEY
	// environment variable.
	APIKey providers.Secret
	// AllowNoAPIKey permits an empty key, for local compatible servers
	// that do not authenticate and gateways that inject the credentials.
	// No Authorization header is sent then. It defaults to false.
	AllowNoAPIKey bool
	// Model is the model ID. It defaults to DefaultLLMModel.
	Model string
	// BaseURL is the API root including the version prefix, for example
	// "http://localhost:8000/v1": the paths appended to it are
	// /chat/completions and /audio/speech. Unlike the other providers of
	// this module it is not a bare origin, because the /v1 prefix is part
	// of what OpenAI-compatible servers document as their base URL. A
	// trailing slash is ignored. It defaults to DefaultBaseURL.
	BaseURL string
	// Temperature is the sampling temperature, between 0 and 2. It defaults
	// to nil, which leaves the provider default.
	Temperature *float64
	// MaxCompletionTokens caps the generated tokens, reasoning included
	// (max_completion_tokens). It defaults to zero, which leaves the
	// provider default.
	MaxCompletionTokens int
	// ParallelToolCalls allows several tool calls in one generation. It
	// defaults to nil, which leaves the provider default (true). It is
	// only sent with tools.
	ParallelToolCalls *bool
	// ReasoningEffort is sent as reasoning_effort when not empty ("none",
	// "low", "medium", ...; support depends on the model).
	//
	// When Model is left empty it defaults to "none": the default model
	// reasons at "medium" otherwise, which adds latency, and it only
	// accepts function tools on this endpoint with reasoning disabled.
	// When Model is set nothing is sent unless this field is, so that
	// compatible servers never see a parameter they may reject.
	ReasoningEffort string
	// HTTPClient sends the requests. It defaults to nil, which means a
	// client shared by the package with no overall Timeout, because
	// streams are long-lived: bound each stream with its context instead.
	HTTPClient *http.Client
}

// LLM is a streaming chat model backed by the Chat Completions API. It is
// safe for concurrent use.
type LLM struct {
	c    *client
	opts LLMOptions
}

var _ llm.LLM = (*LLM)(nil)

// NewLLM validates opts and returns an LLM. It fails when no API key is
// configured, unless opts.AllowNoAPIKey is set, or an option is invalid.
func NewLLM(opts LLMOptions) (*LLM, error) {
	c, err := newClient(providers.KindLLM, opts.APIKey, "LLMOptions.APIKey", opts.AllowNoAPIKey, opts.BaseURL, opts.HTTPClient)
	if err != nil {
		return nil, err
	}
	// Written so that NaN is rejected too.
	if t := opts.Temperature; t != nil && !(*t >= 0 && *t <= 2) {
		return nil, fmt.Errorf("openai: Temperature must be between 0 and 2, got %v", *t)
	}
	if opts.MaxCompletionTokens < 0 {
		return nil, errors.New("openai: MaxCompletionTokens must not be negative")
	}
	if opts.Model == "" {
		opts.Model = DefaultLLMModel
		if opts.ReasoningEffort == "" {
			opts.ReasoningEffort = "none"
		}
	}
	opts.APIKey = "" // the client holds the only copy
	return &LLM{c: c, opts: opts}, nil
}

// Wire types of the request. Field names follow the Chat Completions API
// reference.
type (
	chatRequest struct {
		Model               string         `json:"model"`
		Messages            []chatMessage  `json:"messages"`
		Tools               []chatTool     `json:"tools,omitempty"`
		ToolChoice          string         `json:"tool_choice,omitempty"`
		ParallelToolCalls   *bool          `json:"parallel_tool_calls,omitempty"`
		Temperature         *float64       `json:"temperature,omitempty"`
		MaxCompletionTokens int            `json:"max_completion_tokens,omitempty"`
		ReasoningEffort     string         `json:"reasoning_effort,omitempty"`
		Stream              bool           `json:"stream"`
		StreamOptions       *streamOptions `json:"stream_options,omitempty"`
	}

	streamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	}

	chatMessage struct {
		Role string `json:"role"`
		// Content is omitted on assistant messages that only carry tool
		// calls.
		Content    *string        `json:"content,omitempty"`
		ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
		ToolCallID string         `json:"tool_call_id,omitempty"`
	}

	chatToolCall struct {
		ID       string           `json:"id"`
		Type     string           `json:"type"`
		Function chatFunctionCall `json:"function"`
	}

	chatFunctionCall struct {
		Name string `json:"name"`
		// Arguments is a JSON document encoded as a string.
		Arguments string `json:"arguments"`
	}

	chatTool struct {
		Type     string       `json:"type"`
		Function chatFunction `json:"function"`
	}

	chatFunction struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
	}
)

// chatChunk is one streamed chat.completion.chunk object.
type chatChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	// Error is not part of the documented chunk schema, but the API and
	// compatible servers report failures that happen after the response
	// started as a data event holding the usual error object.
	Error *apiError `json:"error"`
}

// buildRequest maps a llm.Request to the wire format.
func (l *LLM) buildRequest(req llm.Request) chatRequest {
	out := chatRequest{
		Model:               l.opts.Model,
		Messages:            make([]chatMessage, 0, len(req.Messages)),
		Temperature:         l.opts.Temperature,
		MaxCompletionTokens: l.opts.MaxCompletionTokens,
		ReasoningEffort:     l.opts.ReasoningEffort,
		Stream:              true,
		StreamOptions:       &streamOptions{IncludeUsage: true},
	}
	for _, m := range req.Messages {
		// IsError, Interrupted and the tool Name have no field in this API.
		msg := chatMessage{Role: string(m.Role)}
		switch m.Role {
		case llm.RoleAssistant:
			for _, tc := range m.ToolCalls {
				msg.ToolCalls = append(msg.ToolCalls, chatToolCall{
					ID:   tc.ID,
					Type: "function",
					Function: chatFunctionCall{
						Name:      tc.Name,
						Arguments: string(normalizeArguments(tc.Arguments)),
					},
				})
			}
			if m.Content != "" || len(msg.ToolCalls) == 0 {
				msg.Content = &m.Content
			}
		case llm.RoleTool:
			msg.Content = &m.Content
			msg.ToolCallID = m.ToolCallID
		default:
			msg.Content = &m.Content
		}
		out.Messages = append(out.Messages, msg)
	}
	if len(req.Tools) == 0 {
		return out
	}
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, chatTool{
			Type:     "function",
			Function: chatFunction{Name: t.Name, Description: t.Description, Parameters: t.Parameters},
		})
	}
	// ToolChoiceAuto is the API default when tools are present.
	out.ToolChoice = string(req.ToolChoice)
	out.ParallelToolCalls = l.opts.ParallelToolCalls
	return out
}

// normalizeArguments returns "{}" for calls without arguments so that the
// result is always a JSON document.
func normalizeArguments(args json.RawMessage) json.RawMessage {
	if len(bytes.TrimSpace(args)) == 0 {
		return json.RawMessage("{}")
	}
	return args
}

// Chat starts a generation. It returns once the response headers are
// received; ctx bounds the whole stream.
func (l *LLM) Chat(ctx context.Context, req llm.Request) (llm.Stream, error) {
	body, err := json.Marshal(l.buildRequest(req))
	if err != nil {
		return nil, l.c.errorf("request", false, err)
	}
	reqCtx, cancel := context.WithCancel(ctx)
	resp, err := l.c.post(reqCtx, "/chat/completions", "text/event-stream", body)
	if err != nil {
		cancel()
		return nil, err
	}
	return &chatStream{
		c:      l.c,
		parent: ctx,
		cancel: cancel,
		body:   resp.Body,
		r:      bufio.NewReader(resp.Body),
		calls:  make(map[int]*pendingCall),
	}, nil
}

// pendingCall accumulates the fragments of one tool call.
type pendingCall struct {
	id   string
	name string
	args strings.Builder
}

// chatStream reads the server-sent events of one generation. Recv reads
// from the response body on the caller's goroutine: the stream owns no
// goroutine.
type chatStream struct {
	c      *client
	parent context.Context
	cancel context.CancelFunc
	body   io.ReadCloser
	r      *bufio.Reader

	closeOnce sync.Once
	closed    atomic.Bool

	// Owned by the Recv goroutine.
	calls map[int]*pendingCall
	err   error // sticky terminal state, io.EOF at the normal end
}

// Recv returns the next chunk: text deltas as they arrive, the completed
// tool calls once, in index order, when the model finishes, and the usage
// when the API reports it.
func (s *chatStream) Recv() (llm.Chunk, error) {
	if s.closed.Load() {
		return llm.Chunk{}, s.c.closedError()
	}
	for s.err == nil {
		chunk, err := s.next()
		s.err = err
		if chunk.Text != "" || len(chunk.ToolCalls) > 0 || chunk.Usage != nil {
			return chunk, nil
		}
	}
	return llm.Chunk{}, s.err
}

// Close aborts the request and unblocks Recv.
func (s *chatStream) Close() error {
	s.closeOnce.Do(func() {
		s.closed.Store(true)
		s.cancel()
		s.body.Close()
	})
	return nil
}

// next reads one event. It may return a chunk together with a terminal
// error; the chunk is then delivered first.
func (s *chatStream) next() (llm.Chunk, error) {
	line, readErr := s.r.ReadBytes('\n')
	data, ok := sseData(line)
	done := ok && string(data) == "[DONE]"
	if readErr != nil && !done {
		// A line cut before its terminator is never interpreted.
		if readErr == io.EOF {
			readErr = fmt.Errorf("stream ended before [DONE]: %w", io.ErrUnexpectedEOF)
		}
		return llm.Chunk{}, s.c.streamError(s.parent, s.closed.Load(), readErr)
	}
	if !ok {
		return llm.Chunk{}, nil
	}
	if done {
		// Servers that omit finish_reason still get their calls flushed.
		calls, err := s.flushCalls()
		if err != nil {
			return llm.Chunk{}, err
		}
		return llm.Chunk{ToolCalls: calls}, io.EOF
	}

	var ev chatChunk
	if err := json.Unmarshal(data, &ev); err != nil {
		return llm.Chunk{}, s.c.errorf("recv", false, fmt.Errorf("decode chunk: %w", err))
	}
	if ev.Error != nil {
		return llm.Chunk{}, s.c.errorf("recv", true, errors.New(ev.Error.String()))
	}

	var out llm.Chunk
	if ev.Usage != nil {
		out.Usage = &llm.Usage{InputTokens: ev.Usage.PromptTokens, OutputTokens: ev.Usage.CompletionTokens}
	}
	if len(ev.Choices) == 0 {
		return out, nil
	}
	choice := ev.Choices[0]
	out.Text = choice.Delta.Content
	for _, d := range choice.Delta.ToolCalls {
		pc := s.calls[d.Index]
		if pc == nil {
			pc = &pendingCall{}
			s.calls[d.Index] = pc
		}
		if d.ID != "" {
			pc.id = d.ID
		}
		if d.Function.Name != "" {
			pc.name = d.Function.Name
		}
		pc.args.WriteString(d.Function.Arguments)
	}
	if choice.FinishReason != nil {
		calls, err := s.flushCalls()
		if err != nil {
			return llm.Chunk{Text: out.Text}, err
		}
		out.ToolCalls = calls
	}
	return out, nil
}

// flushCalls returns the buffered tool calls in index order and forgets
// them, so each call is emitted exactly once.
func (s *chatStream) flushCalls() ([]llm.ToolCall, error) {
	if len(s.calls) == 0 {
		return nil, nil
	}
	indexes := make([]int, 0, len(s.calls))
	for i := range s.calls {
		indexes = append(indexes, i)
	}
	sort.Ints(indexes)
	calls := make([]llm.ToolCall, 0, len(indexes))
	for _, i := range indexes {
		pc := s.calls[i]
		args := normalizeArguments(json.RawMessage(pc.args.String()))
		if !json.Valid(args) {
			// Typically a generation cut by the token limit.
			return nil, s.c.errorf("recv", false, fmt.Errorf("tool call %q has incomplete JSON arguments", pc.name))
		}
		calls = append(calls, llm.ToolCall{ID: pc.id, Name: pc.name, Arguments: args})
	}
	clear(s.calls)
	return calls, nil
}

// sseData returns the payload of a server-sent event "data:" line. Blank
// lines, comments and other fields are skipped: every event of this API is
// a single data line.
func sseData(line []byte) ([]byte, bool) {
	line = bytes.TrimRight(line, "\r\n")
	rest, ok := bytes.CutPrefix(line, []byte("data:"))
	if !ok {
		return nil, false
	}
	return bytes.TrimPrefix(rest, []byte(" ")), true
}
