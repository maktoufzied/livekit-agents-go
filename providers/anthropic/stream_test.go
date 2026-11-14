package anthropic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers"
)

// sse renders events as a server-sent event stream. Each event is a
// "name|data" pair.
func sse(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		name, data, _ := strings.Cut(e, "|")
		b.WriteString("event: " + name + "\ndata: " + data + "\n\n")
	}
	return b.String()
}

// replay sends one request to a server answering body and returns the stream.
func replay(t *testing.T, body string) llm.Stream {
	t.Helper()
	srv, _ := newServer(t, http.StatusOK, body)
	st, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), llm.Request{Messages: []llm.Message{user("hi")}})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	t.Cleanup(func() { st.Close() })
	return st
}

const (
	evStart = `message_start|{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":25,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}`
	evPing  = `ping|{"type":"ping"}`
	evStop  = `message_stop|{"type":"message_stop"}`
)

func TestStreamText(t *testing.T) {
	st := replay(t, ": keep-alive comment\n\n"+sse(
		evStart,
		// A thinking block, as the default model emits before its answer.
		`content_block_start|{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`content_block_delta|{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc"}}`,
		`content_block_stop|{"type":"content_block_stop","index":0}`,
		`content_block_start|{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		evPing,
		`content_block_delta|{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}`,
		`content_block_delta|{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":""}}`,
		`content_block_delta|{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":", wor"}}`,
		`some_future_event|{"type":"some_future_event","value":1}`,
		`content_block_delta|{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ld!"}}`,
		`content_block_stop|{"type":"content_block_stop","index":1}`,
		`message_delta|{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":12}}`,
		evStop,
	))
	chunks := drain(t, st)

	want := []llm.Chunk{
		{Text: "Hello"},
		{Text: ", wor"},
		{Text: "ld!"},
		{Usage: &llm.Usage{InputTokens: 25, OutputTokens: 12}},
	}
	if !reflect.DeepEqual(chunks, want) {
		t.Fatalf("chunks = %s, want %s", dump(chunks), dump(want))
	}
	// The end of the stream is sticky.
	if _, err := st.Recv(); !errors.Is(err, io.EOF) {
		t.Errorf("Recv after the end = %v, want io.EOF", err)
	}
}

func TestStreamToolCalls(t *testing.T) {
	st := replay(t, sse(
		evStart,
		`content_block_start|{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`content_block_delta|{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Checking."}}`,
		`content_block_stop|{"type":"content_block_stop","index":0}`,
		`content_block_start|{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`,
		`content_block_delta|{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}`,
		`content_block_delta|{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"ci"}}`,
		`content_block_delta|{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"ty\": \"Par"}}`,
		`content_block_delta|{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"is\"}"}}`,
		`content_block_stop|{"type":"content_block_stop","index":1}`,
		`content_block_start|{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_2","name":"get_weather","input":{}}}`,
		`content_block_delta|{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
		`content_block_delta|{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"Rome\",\"days\":2}"}}`,
		`content_block_stop|{"type":"content_block_stop","index":2}`,
		// A tool without arguments streams no input at all.
		`content_block_start|{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_3","name":"now","input":{}}}`,
		`content_block_stop|{"type":"content_block_stop","index":3}`,
		`message_delta|{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"input_tokens":40,"cache_read_input_tokens":10,"output_tokens":55}}`,
		evStop,
	))
	chunks := drain(t, st)

	want := []llm.Chunk{
		{Text: "Checking."},
		{ToolCalls: []llm.ToolCall{{ID: "toolu_1", Name: "get_weather", Arguments: []byte(`{"city": "Paris"}`)}}},
		{ToolCalls: []llm.ToolCall{{ID: "toolu_2", Name: "get_weather", Arguments: []byte(`{"city":"Rome","days":2}`)}}},
		{ToolCalls: []llm.ToolCall{{ID: "toolu_3", Name: "now", Arguments: []byte(`{}`)}}},
		{Usage: &llm.Usage{InputTokens: 50, OutputTokens: 55}},
	}
	if !reflect.DeepEqual(chunks, want) {
		t.Fatalf("chunks = %s, want %s", dump(chunks), dump(want))
	}
}

func TestStreamCRLFAndUnterminatedEvent(t *testing.T) {
	body := strings.ReplaceAll(sse(
		evStart,
		`content_block_delta|{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
	), "\n", "\r\n") + "event: message_stop\r\ndata: {\"type\":\"message_stop\"}"
	chunks := drain(t, replay(t, body))
	if len(chunks) != 1 || chunks[0].Text != "Hi" {
		t.Fatalf("chunks = %s", dump(chunks))
	}
}

func TestStreamTruncated(t *testing.T) {
	st := replay(t, sse(
		evStart,
		`content_block_delta|{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`,
	))
	chunks, err := collect(st)
	if len(chunks) != 1 || chunks[0].Text != "Hel" {
		t.Errorf("chunks = %s", dump(chunks))
	}
	var pe *providers.Error
	if !errors.As(err, &pe) || !pe.Retryable || pe.Op != "recv" || errors.Is(err, io.EOF) {
		t.Fatalf("err = %v, want a retryable recv error", err)
	}
	if _, again := st.Recv(); again != err {
		t.Errorf("the error is not sticky: %v", again)
	}
}

func TestStreamIncompleteToolInput(t *testing.T) {
	st := replay(t, sse(
		evStart,
		`content_block_start|{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`,
		`content_block_delta|{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\": \"Par"}}`,
		`content_block_stop|{"type":"content_block_stop","index":0}`,
		`message_delta|{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":4096}}`,
		evStop,
	))
	chunks, err := collect(st)
	if len(chunks) != 0 {
		t.Errorf("an incomplete tool call was emitted: %s", dump(chunks))
	}
	if !providers.IsRetryable(err) {
		t.Fatalf("err = %v, want a retryable error", err)
	}
}

func TestStreamErrorEvent(t *testing.T) {
	cases := []struct {
		errType   string
		status    int
		retryable bool
	}{
		{"overloaded_error", 529, true},
		{"api_error", 500, true},
		{"invalid_request_error", 400, false},
	}
	for _, tc := range cases {
		t.Run(tc.errType, func(t *testing.T) {
			st := replay(t, sse(
				evStart,
				`content_block_delta|{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`,
				`error|{"type":"error","error":{"type":"`+tc.errType+`","message":"Something broke for `+testKey+`"}}`,
			))
			chunks, err := collect(st)
			if len(chunks) != 1 {
				t.Errorf("chunks = %s", dump(chunks))
			}
			pe := providerError(t, err)
			if pe.Op != "recv" ||
				pe.StatusCode != tc.status || pe.Retryable != tc.retryable {
				t.Errorf("unexpected error: %+v", pe)
			}
			var apiErr *APIError
			if !errors.As(err, &apiErr) || apiErr.Type != tc.errType {
				t.Errorf("APIError = %+v", apiErr)
			}
			if msg := err.Error(); !strings.Contains(msg, "Something broke") || strings.Contains(msg, testKey) {
				t.Errorf("error message %q", msg)
			}
		})
	}
}

func TestStreamRefusal(t *testing.T) {
	st := replay(t, sse(
		evStart,
		`message_delta|{"type":"message_delta","delta":{"stop_reason":"refusal"},"usage":{"output_tokens":0}}`,
		evStop,
	))
	chunks, err := collect(st)
	if len(chunks) != 1 || chunks[0].Usage == nil {
		t.Errorf("chunks = %s", dump(chunks))
	}
	var pe *providers.Error
	if !errors.As(err, &pe) || pe.Retryable || !strings.Contains(err.Error(), "refusal") {
		t.Fatalf("err = %v, want a non-retryable refusal error", err)
	}
}

func TestStreamMalformedEvent(t *testing.T) {
	_, err := collect(replay(t, sse(evStart, `content_block_delta|{not json`)))
	if !providers.IsRetryable(err) {
		t.Fatalf("err = %v, want a retryable error", err)
	}
	providerError(t, err)

	// Text taken from the stream cannot carry the key into an error either.
	_, err = collect(replay(t, sse(evStart, `evt_`+testKey+`|{not json`)))
	if pe := providerError(t, err); !strings.Contains(err.Error(), "evt_[REDACTED]") {
		t.Errorf("error = %v (%+v)", err, pe)
	}
	_, err = collect(replay(t, sse(evStart,
		`content_block_start|{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"tool_`+testKey+`"}}`,
		`content_block_delta|{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}`,
		`content_block_stop|{"type":"content_block_stop","index":0}`,
	)))
	if pe := providerError(t, err); !strings.Contains(err.Error(), "tool_[REDACTED]") {
		t.Errorf("error = %v (%+v)", err, pe)
	}
}

// hangingServer sends the start of a stream and then keeps it open until the
// client goes away.
func hangingServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "text/event-stream")
		_, _ = io.WriteString(w, sse(
			evStart,
			`content_block_delta|{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`,
		))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)
	return srv
}

// startBlockedRecv opens a stream on a hanging server, reads the first chunk
// and returns a channel carrying the result of the next, blocked, Recv.
func startBlockedRecv(t *testing.T, ctx context.Context) (llm.Stream, <-chan error) {
	t.Helper()
	srv := hangingServer(t)
	st, err := newTestLLM(t, srv.URL, nil).Chat(ctx, llm.Request{Messages: []llm.Message{user("hi")}})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	t.Cleanup(func() { st.Close() })
	if c, err := st.Recv(); err != nil || c.Text != "Hel" {
		t.Fatalf("first Recv = %+v, %v", c, err)
	}
	done := make(chan error, 1)
	go func() {
		_, err := st.Recv()
		done <- err
	}()
	return st, done
}

func TestCloseUnblocksRecv(t *testing.T) {
	st, done := startBlockedRecv(t, context.Background())
	if err := st.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	err := waitErr(t, done)
	assertCause(t, err, providers.ErrStreamClosed)
	if err := st.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	_, err = st.Recv()
	assertCause(t, err, providers.ErrStreamClosed)
}

func TestRecvAfterEOFAndClose(t *testing.T) {
	st := replay(t, okStream)
	drain(t, st)
	if _, err := st.Recv(); err != io.EOF {
		t.Fatalf("Recv = %v, want a sticky io.EOF", err)
	}
	st.Close()
	_, err := st.Recv()
	assertCause(t, err, providers.ErrStreamClosed)
}

func TestContextCancelUnblocksRecv(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, done := startBlockedRecv(t, ctx)
	cancel()
	err := waitErr(t, done)
	assertCause(t, err, context.Canceled)
}

func TestContextDeadlineUnblocksRecv(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	_, done := startBlockedRecv(t, ctx)
	assertCause(t, waitErr(t, done), context.DeadlineExceeded)
}

func dump(chunks []llm.Chunk) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("\n  ")
		switch {
		case c.Usage != nil:
			fmt.Fprintf(&b, "usage %d/%d", c.Usage.InputTokens, c.Usage.OutputTokens)
		case len(c.ToolCalls) > 0:
			for _, tc := range c.ToolCalls {
				b.WriteString("tool " + tc.ID + " " + tc.Name + " " + string(tc.Arguments))
			}
		default:
			b.WriteString("text " + c.Text)
		}
	}
	return b.String()
}
