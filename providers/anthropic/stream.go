package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/maktoufzied/livekit-agents-go/llm"
)

// stream reads the server-sent events of one generation.
//
// Recv is driven by a single goroutine and owns every field except closed,
// which Close may set from another goroutine.
type stream struct {
	ctx    context.Context // the context passed to Chat
	llm    *LLM
	body   io.ReadCloser
	reader *bufio.Reader
	cancel context.CancelFunc

	releaseOnce sync.Once
	closed      atomic.Bool

	// tools holds the tool_use blocks being received, by content block index.
	tools      map[int]*toolBlock
	usage      llm.Usage
	stopReason string
	// final is the sticky result of Recv once the stream has ended.
	final error
}

type toolBlock struct {
	id, name string
	input    bytes.Buffer
}

var _ llm.Stream = (*stream)(nil)

func newStream(ctx context.Context, l *LLM, body io.ReadCloser, cancel context.CancelFunc) *stream {
	return &stream{
		ctx:    ctx,
		llm:    l,
		body:   body,
		reader: bufio.NewReader(body),
		cancel: cancel,
		tools:  make(map[int]*toolBlock),
	}
}

// Close aborts the request and unblocks a pending Recv. It is safe to call
// more than once and from any goroutine.
func (s *stream) Close() error {
	s.closed.Store(true)
	s.release()
	return nil
}

// release aborts the request and frees the connection.
func (s *stream) release() {
	s.releaseOnce.Do(func() {
		s.cancel()
		_ = s.body.Close()
	})
}

// Recv returns the next chunk: a text delta, one completed tool call, or the
// token usage. It returns io.EOF after the message_stop event.
func (s *stream) Recv() (llm.Chunk, error) {
	for {
		if s.closed.Load() {
			return llm.Chunk{}, s.llm.closedError()
		}
		if s.final != nil {
			return llm.Chunk{}, s.final
		}
		event, data, err := s.readEvent()
		if err != nil {
			s.final = s.readError(err)
			s.release()
			if s.closed.Load() {
				return llm.Chunk{}, s.llm.closedError()
			}
			return llm.Chunk{}, s.final
		}
		chunk, ok, err := s.handle(event, data)
		if err != nil {
			s.final = err
			s.release()
			return llm.Chunk{}, err
		}
		if ok {
			return chunk, nil
		}
	}
}

// readError classifies a failure to read the next event.
func (s *stream) readError(err error) error {
	if ctxErr := s.ctx.Err(); ctxErr != nil {
		return s.llm.newError("recv", 0, false, ctxErr)
	}
	if errors.Is(err, io.EOF) {
		err = errors.New("stream ended before message_stop")
	}
	return s.llm.newError("recv", 0, true, err)
}

// readEvent reads one server-sent event and returns its name and data. It
// skips comments and events without data. At the end of the body it returns
// io.EOF, after a last event that was not terminated by a blank line.
func (s *stream) readEvent() (event string, data []byte, err error) {
	var buf bytes.Buffer
	hasData := false
	for {
		line, err := s.reader.ReadString('\n')
		eof := errors.Is(err, io.EOF)
		if err != nil && !eof {
			return "", nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		switch {
		case line == "":
			if hasData {
				return event, buf.Bytes(), nil
			}
			event = ""
		case strings.HasPrefix(line, ":"):
		default:
			field, value, _ := strings.Cut(line, ":")
			value = strings.TrimPrefix(value, " ")
			switch field {
			case "event":
				event = value
			case "data":
				if hasData {
					buf.WriteByte('\n')
				}
				buf.WriteString(value)
				hasData = true
			}
		}
		if eof {
			if hasData {
				return event, buf.Bytes(), nil
			}
			return "", nil, io.EOF
		}
	}
}

// wireEvent is the union of the stream event payloads this package reads.
type wireEvent struct {
	Type    string `json:"type"`
	Index   int    `json:"index"`
	Message *struct {
		Usage *wireUsage `json:"usage"`
	} `json:"message"`
	ContentBlock *struct {
		Type string `json:"type"`
		ID   string `json:"id"`
		Name string `json:"name"`
		Text string `json:"text"`
	} `json:"content_block"`
	Delta *struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Usage *wireUsage `json:"usage"`
}

type wireUsage struct {
	InputTokens              *int `json:"input_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
	OutputTokens             *int `json:"output_tokens"`
}

// handle processes one event. It reports whether chunk must be returned to
// the caller; a non-nil error ends the stream (io.EOF at the normal end).
func (s *stream) handle(event string, data []byte) (chunk llm.Chunk, ok bool, err error) {
	var ev wireEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return chunk, false, s.llm.newError("recv", 0, true, fmt.Errorf("decode %q event: %w", event, err))
	}
	if ev.Type == "" {
		ev.Type = event
	}

	switch ev.Type {
	case "message_start":
		if ev.Message != nil {
			s.mergeUsage(ev.Message.Usage)
		}

	case "content_block_start":
		if ev.ContentBlock == nil {
			break
		}
		switch ev.ContentBlock.Type {
		case "tool_use":
			s.tools[ev.Index] = &toolBlock{id: ev.ContentBlock.ID, name: ev.ContentBlock.Name}
		case "text":
			if ev.ContentBlock.Text != "" {
				return llm.Chunk{Text: ev.ContentBlock.Text}, true, nil
			}
		}

	case "content_block_delta":
		if ev.Delta == nil {
			break
		}
		switch ev.Delta.Type {
		case "text_delta":
			if ev.Delta.Text != "" {
				return llm.Chunk{Text: ev.Delta.Text}, true, nil
			}
		case "input_json_delta":
			if tb := s.tools[ev.Index]; tb != nil {
				tb.input.WriteString(ev.Delta.PartialJSON)
			}
		}

	case "content_block_stop":
		tb := s.tools[ev.Index]
		if tb == nil {
			break
		}
		delete(s.tools, ev.Index)
		args := bytes.TrimSpace(tb.input.Bytes())
		if len(args) == 0 {
			args = []byte(emptyObjectArgument)
		}
		if args[0] != '{' || !json.Valid(args) {
			// Typically an input cut off by max_tokens: a new attempt
			// may produce a complete call.
			return chunk, false, s.llm.newError("recv", 0, true,
				fmt.Errorf("tool call %q: input is not a complete JSON object", tb.name))
		}
		call := llm.ToolCall{ID: tb.id, Name: tb.name, Arguments: json.RawMessage(args)}
		return llm.Chunk{ToolCalls: []llm.ToolCall{call}}, true, nil

	case "message_delta":
		if ev.Delta != nil && ev.Delta.StopReason != "" {
			s.stopReason = ev.Delta.StopReason
		}
		if ev.Usage != nil {
			s.mergeUsage(ev.Usage)
			usage := s.usage
			return llm.Chunk{Usage: &usage}, true, nil
		}

	case "message_stop":
		if s.stopReason == "refusal" {
			return chunk, false, s.llm.newError("recv", 0, false,
				errors.New("the model declined to answer (stop_reason \"refusal\"); discard any partial output"))
		}
		return chunk, false, io.EOF

	case "error":
		var env errorEnvelope
		_ = json.Unmarshal(data, &env)
		return chunk, false, s.llm.streamAPIError(env)
	}
	// ping, thinking and any event type added later are ignored.
	return chunk, false, nil
}

// mergeUsage folds a usage object into the running totals. The counters of
// the API are cumulative, so a field that is present replaces the previous
// value. Input tokens include the tokens written to and read from the cache.
func (s *stream) mergeUsage(u *wireUsage) {
	if u == nil {
		return
	}
	if u.InputTokens != nil {
		in := *u.InputTokens
		if u.CacheCreationInputTokens != nil {
			in += *u.CacheCreationInputTokens
		}
		if u.CacheReadInputTokens != nil {
			in += *u.CacheReadInputTokens
		}
		s.usage.InputTokens = in
	}
	if u.OutputTokens != nil {
		s.usage.OutputTokens = *u.OutputTokens
	}
}
