package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers"
)

// sse renders payloads as server-sent "data:" events.
func sse(payloads ...string) string {
	var b strings.Builder
	for _, p := range payloads {
		b.WriteString("data: " + p + "\n\n")
	}
	return b.String()
}

// sseHandler replies with a fixed event stream.
func sseHandler(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, body)
	}
}

func newTestLLM(t *testing.T, url string, mod func(*LLMOptions)) *LLM {
	t.Helper()
	opts := LLMOptions{APIKey: testKey, BaseURL: url}
	if mod != nil {
		mod(&opts)
	}
	l, err := NewLLM(opts)
	if err != nil {
		t.Fatalf("NewLLM: %v", err)
	}
	return l
}

// collect drains a stream until io.EOF or an error.
func collect(t *testing.T, s llm.Stream) ([]llm.Chunk, error) {
	t.Helper()
	var chunks []llm.Chunk
	for {
		c, err := s.Recv()
		if err == io.EOF {
			return chunks, nil
		}
		if err != nil {
			return chunks, err
		}
		chunks = append(chunks, c)
	}
}

func userRequest(text string) llm.Request {
	return llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: text}}}
}

func TestNewLLMAPIKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	_, err := NewLLM(LLMOptions{})
	assertPlainError(t, err, "API key is required (LLMOptions.APIKey or OPENAI_API_KEY)")
	if _, err := NewLLM(LLMOptions{AllowNoAPIKey: true}); err != nil {
		t.Errorf("AllowNoAPIKey: %v", err)
	}

	t.Setenv("OPENAI_API_KEY", "sk-from-env")
	l, err := NewLLM(LLMOptions{})
	if err != nil {
		t.Fatalf("NewLLM with env key: %v", err)
	}
	if got := l.c.key.Reveal(); got != "sk-from-env" {
		t.Errorf("key = %q, want the env value", got)
	}
	if l.c.baseURL != DefaultBaseURL || l.opts.Model != DefaultLLMModel {
		t.Errorf("defaults = %q, %q", l.c.baseURL, l.opts.Model)
	}
	if s := fmt.Sprintf("%+v %#v", l, l); strings.Contains(s, "sk-from-env") {
		t.Errorf("formatted LLM leaks the key: %s", s)
	}
	if l.c.http != defaultHTTPClient || defaultHTTPClient.Timeout != 0 {
		t.Error("a nil HTTPClient must select the shared client, which has no Timeout")
	}
}

func TestNewLLMValidation(t *testing.T) {
	for _, temp := range []float64{-0.1, 2.1, math.NaN(), math.Inf(1)} {
		_, err := NewLLM(LLMOptions{APIKey: testKey, Temperature: &temp})
		assertPlainError(t, err, "Temperature must be between 0 and 2")
	}
	for _, temp := range []float64{0, 1, 2} {
		if _, err := NewLLM(LLMOptions{APIKey: testKey, Temperature: &temp}); err != nil {
			t.Errorf("Temperature %v: %v", temp, err)
		}
	}
	_, err := NewLLM(LLMOptions{APIKey: testKey, MaxCompletionTokens: -1})
	assertPlainError(t, err, "MaxCompletionTokens must not be negative")
}

func TestChatRequestShape(t *testing.T) {
	var (
		got     map[string]any
		headers http.Header
		path    string
		method  string
	)
	srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		method, path, headers = r.Method, r.URL.Path, r.Header.Clone()
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode request: %v", err)
		}
		sseHandler(sse("[DONE]"))(w, r)
	})
	l := newTestLLM(t, srv.URL+"/", func(o *LLMOptions) {
		o.Model = "test-model"
		o.Temperature = ptr(0.3)
		o.MaxCompletionTokens = 128
		o.ParallelToolCalls = ptr(false)
		o.ReasoningEffort = "low"
	})

	req := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "Be brief."},
			{Role: llm.RoleUser, Content: "Weather in Paris?"},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
				{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Paris"}`)},
				{ID: "call_2", Name: "get_time"},
			}},
			{Role: llm.RoleTool, ToolCallID: "call_1", Name: "get_weather", Content: "18C", IsError: true},
			{Role: llm.RoleTool, ToolCallID: "call_2", Name: "get_time", Content: "noon"},
			{Role: llm.RoleAssistant, Content: "It is 18", Interrupted: true},
		},
		Tools: []llm.ToolSpec{
			{Name: "get_weather", Description: "Current weather", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)},
			{Name: "get_time"},
		},
		ToolChoice: llm.ToolChoiceRequired,
	}
	s, err := l.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	defer s.Close()
	if _, err := s.Recv(); err != io.EOF {
		t.Fatalf("Recv = %v, want io.EOF", err)
	}

	if method != http.MethodPost || path != "/chat/completions" {
		t.Errorf("request = %s %s", method, path)
	}
	if h := headers.Get("Authorization"); h != "Bearer "+testKey {
		t.Errorf("Authorization = %q", h)
	}
	if h := headers.Get("Content-Type"); h != "application/json" {
		t.Errorf("Content-Type = %q", h)
	}

	var want map[string]any
	wantJSON := `{
		"model": "test-model",
		"stream": true,
		"stream_options": {"include_usage": true},
		"temperature": 0.3,
		"max_completion_tokens": 128,
		"parallel_tool_calls": false,
		"reasoning_effort": "low",
		"tool_choice": "required",
		"messages": [
			{"role": "system", "content": "Be brief."},
			{"role": "user", "content": "Weather in Paris?"},
			{"role": "assistant", "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}},
				{"id": "call_2", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "18C"},
			{"role": "tool", "tool_call_id": "call_2", "content": "noon"},
			{"role": "assistant", "content": "It is 18"}
		],
		"tools": [
			{"type": "function", "function": {"name": "get_weather", "description": "Current weather",
				"parameters": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
			{"type": "function", "function": {"name": "get_time"}}
		]
	}`
	if err := json.Unmarshal([]byte(wantJSON), &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		gotJSON, _ := json.MarshalIndent(got, "", "  ")
		t.Errorf("request body mismatch, got:\n%s", gotJSON)
	}
}

func TestChatRequestDefaults(t *testing.T) {
	tests := []struct {
		name string
		mod  func(*LLMOptions)
		req  llm.Request
		want string
	}{
		{
			name: "default model disables reasoning",
			req:  userRequest("hi"),
			want: `{"model":"` + DefaultLLMModel + `","reasoning_effort":"none","stream":true,
				"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name: "explicit model sends no optional field",
			mod:  func(o *LLMOptions) { o.Model = "local"; o.ParallelToolCalls = ptr(true) },
			req:  llm.Request{Messages: userRequest("hi").Messages, ToolChoice: llm.ToolChoiceRequired},
			want: `{"model":"local","stream":true,"stream_options":{"include_usage":true},
				"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name: "tool choice none keeps the tools",
			mod:  func(o *LLMOptions) { o.Model = "local" },
			req: llm.Request{
				Messages:   []llm.Message{{Role: llm.RoleAssistant}},
				Tools:      []llm.ToolSpec{{Name: "f"}},
				ToolChoice: llm.ToolChoiceNone,
			},
			want: `{"model":"local","stream":true,"stream_options":{"include_usage":true},
				"messages":[{"role":"assistant","content":""}],
				"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"none"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got any
			srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				json.NewDecoder(r.Body).Decode(&got)
				sseHandler(sse("[DONE]"))(w, r)
			})
			s, err := newTestLLM(t, srv.URL, tt.mod).Chat(context.Background(), tt.req)
			if err != nil {
				t.Fatalf("Chat: %v", err)
			}
			s.Close()
			var want any
			if err := json.Unmarshal([]byte(tt.want), &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				gotJSON, _ := json.Marshal(got)
				t.Errorf("request body = %s", gotJSON)
			}
		})
	}
}

func TestChatNoAuthorizationWithoutKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	var auth []string
	srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Values("Authorization")
		sseHandler(sse("[DONE]"))(w, r)
	})
	l, err := NewLLM(LLMOptions{AllowNoAPIKey: true, BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	s, err := l.Chat(context.Background(), userRequest("hi"))
	if err != nil {
		t.Fatal(err)
	}
	s.Close()
	if len(auth) != 0 {
		t.Errorf("Authorization = %q, want none", auth)
	}
}

func TestChatStreamText(t *testing.T) {
	srv := newServer(t, sseHandler(
		": keep-alive comment\n\n"+
			sse(
				`{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],"usage":null}`,
				`{"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}],"usage":null}`,
				`{"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}],"usage":null}`,
				`{"choices":[{"index":0,"delta":{"content":null},"finish_reason":null}],"usage":null}`,
				`{"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}],"usage":null}`,
				`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}`,
				`{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}`,
				`[DONE]`,
			)))
	s, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), userRequest("hi"))
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	defer s.Close()
	chunks, err := collect(t, s)
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	want := []llm.Chunk{
		{Text: "Hel"}, {Text: "lo"}, {Text: " world"},
		{Usage: &llm.Usage{InputTokens: 11, OutputTokens: 3}},
	}
	if !reflect.DeepEqual(chunks, want) {
		t.Errorf("chunks = %+v, want %+v", chunks, want)
	}
	// The end of the stream is sticky.
	if _, err := s.Recv(); err != io.EOF {
		t.Errorf("Recv after EOF = %v, want io.EOF", err)
	}
}

func TestChatStreamParallelToolCalls(t *testing.T) {
	// Two calls whose fragments are interleaved and listed out of order.
	srv := newServer(t, sseHandler(sse(
		`{"choices":[{"index":0,"delta":{"role":"assistant","content":"Checking."},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"get_time","arguments":""}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"get_weather","arguments":"{\"ci"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"zone\":"}},{"index":0,"function":{"arguments":"ty\":\"Par"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"is\"}"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"CET\"}"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`{"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":22,"total_tokens":62}}`,
		`[DONE]`,
	)))
	s, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), userRequest("hi"))
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	defer s.Close()
	chunks, err := collect(t, s)
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	want := []llm.Chunk{
		{Text: "Checking."},
		{ToolCalls: []llm.ToolCall{
			{ID: "call_a", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Paris"}`)},
			{ID: "call_b", Name: "get_time", Arguments: json.RawMessage(`{"zone":"CET"}`)},
		}},
		{Usage: &llm.Usage{InputTokens: 40, OutputTokens: 22}},
	}
	if !reflect.DeepEqual(chunks, want) {
		t.Errorf("chunks = %+v\nwant     %+v", chunks, want)
	}
}

func TestChatStreamToolCallEdgeCases(t *testing.T) {
	t.Run("no arguments and no finish_reason", func(t *testing.T) {
		srv := newServer(t, sseHandler(
			`data:{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"ping"}}]}}]}`+"\r\n\r\n"+
				"data: [DONE]")) // no trailing newline
		s, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), userRequest("hi"))
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		chunks, err := collect(t, s)
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		want := []llm.Chunk{{ToolCalls: []llm.ToolCall{{ID: "call_a", Name: "ping", Arguments: json.RawMessage(`{}`)}}}}
		if !reflect.DeepEqual(chunks, want) {
			t.Errorf("chunks = %+v, want %+v", chunks, want)
		}
	})

	t.Run("arguments cut by the token limit", func(t *testing.T) {
		srv := newServer(t, sseHandler(sse(
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"get_weather","arguments":"{\"city\":\"Pa"}}]},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
			`[DONE]`,
		)))
		s, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), userRequest("hi"))
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		chunks, err := collect(t, s)
		if err == nil || len(chunks) != 0 {
			t.Fatalf("chunks = %+v, err = %v; want an error and no call", chunks, err)
		}
		if pe := providerError(t, err, providers.KindLLM); pe.Retryable || pe.Op != "recv" {
			t.Errorf("error = %+v", pe)
		}
	})
}

func TestChatHTTPErrors(t *testing.T) {
	tests := []struct {
		status    int
		body      string
		retryable bool
		contains  string
	}{
		{429, `{"error":{"message":"Rate limit reached","type":"rate_limit_error","param":null,"code":"rate_limit_exceeded"}}`, true, "Rate limit reached"},
		{500, `{"error":{"message":"The server had an error","type":"server_error","param":null,"code":null}}`, true, "The server had an error"},
		{503, `upstream unavailable`, true, "upstream unavailable"},
		{400, `{"error":{"message":"Invalid value for 'model'","type":"invalid_request_error","param":"model","code":null}}`, false, "Invalid value for 'model' (invalid_request_error)"},
		{401, `{"error":{"message":"Incorrect API key provided: ` + testKey + `","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}`, false, "Incorrect API key provided"},
		{404, ``, false, "Not Found"},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprint(tt.status), func(t *testing.T) {
			srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				io.WriteString(w, tt.body)
			})
			s, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), userRequest("hi"))
			if err == nil {
				s.Close()
				t.Fatal("Chat succeeded")
			}
			pe := providerError(t, err, providers.KindLLM)
			if pe.StatusCode != tt.status || pe.Op != "request" {
				t.Errorf("status/op = %d/%q", pe.StatusCode, pe.Op)
			}
			if pe.Retryable != tt.retryable || providers.IsRetryable(err) != tt.retryable {
				t.Errorf("retryable = %v, want %v", pe.Retryable, tt.retryable)
			}
			if !strings.Contains(err.Error(), tt.contains) {
				t.Errorf("error %q does not contain %q", err, tt.contains)
			}
		})
	}
}

func TestChatKeyNeverInErrors(t *testing.T) {
	// A server that echoes the credentials, on the status and stream paths.
	echo := func(w http.ResponseWriter, r *http.Request) string {
		return strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	}
	t.Run("status", func(t *testing.T) {
		srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusForbidden)
			fmt.Fprintf(w, "denied for key %s", echo(w, r))
		})
		_, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), userRequest("hi"))
		providerError(t, err, providers.KindLLM)
		if !strings.Contains(err.Error(), "denied for key [REDACTED]") {
			t.Errorf("error = %v", err)
		}
	})
	t.Run("stream", func(t *testing.T) {
		srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprintf(w, "data: {\"error\":{\"message\":\"overloaded, key %s\",\"type\":\"server_error\"}}\n\n", echo(w, r))
		})
		s, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), userRequest("hi"))
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		_, err = s.Recv()
		providerError(t, err, providers.KindLLM)
		if !strings.Contains(err.Error(), "overloaded, key [REDACTED]") {
			t.Errorf("error = %v", err)
		}
		if !providers.IsRetryable(err) {
			t.Errorf("in-stream error should be retryable: %v", err)
		}
	})
	t.Run("transport", func(t *testing.T) {
		srv := newServer(t, func(http.ResponseWriter, *http.Request) {})
		addr := srv.URL
		srv.Close()
		_, err := newTestLLM(t, addr, nil).Chat(context.Background(), userRequest("hi"))
		pe := providerError(t, err, providers.KindLLM)
		if !pe.Retryable || pe.StatusCode != 0 {
			t.Errorf("transport error = %+v", pe)
		}
		// A clean error keeps its chain.
		var ue *url.Error
		if !errors.As(err, &ue) {
			t.Errorf("transport error lost its chain: %v", err)
		}
	})
	t.Run("transport error holding the key", func(t *testing.T) {
		// The *url.Error of the transport quotes the URL, which holds the
		// key here. Scrubbing the message is not enough: the original error
		// must not stay reachable through Unwrap.
		srv := newServer(t, func(http.ResponseWriter, *http.Request) {})
		addr := srv.URL
		srv.Close()
		_, err := newTestLLM(t, addr+"/"+testKey, nil).Chat(context.Background(), userRequest("hi"))
		pe := providerError(t, err, providers.KindLLM)
		if !pe.Retryable || !strings.Contains(err.Error(), "/[REDACTED]/chat/completions") {
			t.Errorf("transport error = %v", err)
		}
	})
}

func TestChatTruncatedStream(t *testing.T) {
	bodies := map[string]string{
		"between events": sse(`{"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}`),
		"inside an event": sse(`{"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}`) +
			`data: {"choices":[{"index":0,"delta":{"content":"lo`,
		"after finish_reason": sse(
			`{"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`),
	}
	for name, body := range bodies {
		t.Run(name, func(t *testing.T) {
			srv := newServer(t, sseHandler(body))
			s, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), userRequest("hi"))
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			chunks, err := collect(t, s)
			if !reflect.DeepEqual(chunks, []llm.Chunk{{Text: "Hel"}}) {
				t.Errorf("chunks = %+v", chunks)
			}
			if err == nil {
				t.Fatal("truncated stream ended with io.EOF")
			}
			pe := providerError(t, err, providers.KindLLM)
			if !pe.Retryable || pe.Op != "recv" || !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Errorf("error = %v (%+v)", err, pe)
			}
			if _, again := s.Recv(); again != err {
				t.Errorf("error is not sticky: %v", again)
			}
		})
	}
}

// stallingHandler sends one text delta and then keeps the stream open until
// the client goes away.
func stallingHandler(w http.ResponseWriter, r *http.Request) {
	io.WriteString(w, sse(`{"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}`))
	w.(http.Flusher).Flush()
	waitGone(r)
}

func TestChatCloseUnblocksRecv(t *testing.T) {
	srv := newServer(t, stallingHandler)
	s, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), userRequest("hi"))
	if err != nil {
		t.Fatal(err)
	}
	if c, err := s.Recv(); err != nil || c.Text != "Hel" {
		t.Fatalf("Recv = %+v, %v", c, err)
	}
	errc := make(chan error, 1)
	go func() {
		_, err := s.Recv()
		errc <- err
	}()
	time.Sleep(20 * time.Millisecond) // let Recv block on the body
	if err := s.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	select {
	case err := <-errc:
		assertCause(t, err, providers.KindLLM, providers.ErrStreamClosed)
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not unblock Recv")
	}
	if err := s.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	_, err = s.Recv()
	assertCause(t, err, providers.KindLLM, providers.ErrStreamClosed)
}

func TestChatRecvAfterEOFAndClose(t *testing.T) {
	srv := newServer(t, sseHandler(sse(`{"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}`, `[DONE]`)))
	s, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), userRequest("hi"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collect(t, s); err != nil {
		t.Fatalf("stream ended with %v", err)
	}
	if _, err := s.Recv(); err != io.EOF {
		t.Fatalf("Recv = %v, want a sticky io.EOF", err)
	}
	s.Close()
	_, err = s.Recv()
	assertCause(t, err, providers.KindLLM, providers.ErrStreamClosed)
}

func TestChatContextCancel(t *testing.T) {
	t.Run("during the stream", func(t *testing.T) {
		srv := newServer(t, stallingHandler)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		s, err := newTestLLM(t, srv.URL, nil).Chat(ctx, userRequest("hi"))
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		if _, err := s.Recv(); err != nil {
			t.Fatal(err)
		}
		time.AfterFunc(20*time.Millisecond, cancel)
		_, err = s.Recv()
		assertCause(t, err, providers.KindLLM, context.Canceled)
	})

	t.Run("before the response", func(t *testing.T) {
		srv := newServer(t, func(w http.ResponseWriter, r *http.Request) {
			waitGone(r)
		})
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
		defer cancel()
		_, err := newTestLLM(t, srv.URL, nil).Chat(ctx, userRequest("hi"))
		assertCause(t, err, providers.KindLLM, context.DeadlineExceeded)
	})
}
