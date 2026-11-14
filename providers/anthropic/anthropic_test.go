package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	neturl "net/url"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers"
)

const testKey = "sk-ant-test-TOPSECRET"

// okStream is a minimal successful response.
const okStream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":3,"output_tokens":1}}}

event: message_stop
data: {"type":"message_stop"}

`

// capture records the last request received by a test server.
type capture struct {
	mu     sync.Mutex
	header http.Header
	path   string
	body   map[string]any
}

func (c *capture) get() (http.Header, string, map[string]any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.header, c.path, c.body
}

// newServer returns a server that records the request and replies with the
// given status and body.
func newServer(t *testing.T, status int, body string) (*httptest.Server, *capture) {
	t.Helper()
	c := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		c.mu.Lock()
		c.header, c.path, c.body = r.Header.Clone(), r.URL.Path, parsed
		c.mu.Unlock()
		if status == http.StatusOK {
			w.Header().Set("content-type", "text/event-stream")
		} else {
			w.Header().Set("content-type", "application/json")
			w.Header().Set("request-id", "req_test")
		}
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv, c
}

func newTestLLM(t *testing.T, baseURL string, mutate func(*LLMOptions)) *LLM {
	t.Helper()
	opts := LLMOptions{APIKey: testKey, BaseURL: baseURL}
	if mutate != nil {
		mutate(&opts)
	}
	l, err := NewLLM(opts)
	if err != nil {
		t.Fatalf("NewLLM: %v", err)
	}
	return l
}

// chat sends req to a server answering okStream and returns the request the
// server saw.
func chat(t *testing.T, req llm.Request, mutate func(*LLMOptions)) (http.Header, string, map[string]any) {
	t.Helper()
	srv, c := newServer(t, http.StatusOK, okStream)
	l := newTestLLM(t, srv.URL, mutate)
	st, err := l.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	defer st.Close()
	drain(t, st)
	return c.get()
}

// drain reads st to io.EOF and returns the chunks.
func drain(t *testing.T, st llm.Stream) []llm.Chunk {
	t.Helper()
	chunks, err := collect(st)
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	return chunks
}

// collect reads st until an error; io.EOF is reported as nil.
func collect(st llm.Stream) ([]llm.Chunk, error) {
	var chunks []llm.Chunk
	for {
		c, err := st.Recv()
		if errors.Is(err, io.EOF) {
			return chunks, nil
		}
		if err != nil {
			return chunks, err
		}
		chunks = append(chunks, c)
	}
}

// wantJSON compares got with the JSON document want.
func wantJSON(t *testing.T, name string, got any, want string) {
	t.Helper()
	var w any
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("bad expectation for %s: %v", name, err)
	}
	if !reflect.DeepEqual(got, w) {
		g, _ := json.MarshalIndent(got, "", "  ")
		t.Errorf("%s mismatch\n got: %s\nwant: %s", name, g, want)
	}
}

func user(s string) llm.Message { return llm.Message{Role: llm.RoleUser, Content: s} }

// assertNoKey checks that the key appears nowhere in err: neither in its
// message, nor in its formatting, nor in any error of its Unwrap chain.
func assertNoKey(t *testing.T, err error) {
	t.Helper()
	for e := err; e != nil; e = errors.Unwrap(e) {
		for _, s := range []string{e.Error(), fmt.Sprintf("%v", e), fmt.Sprintf("%+v", e), fmt.Sprintf("%#v", e)} {
			if strings.Contains(s, testKey) {
				t.Errorf("error leaks the API key (%T in the chain): %s", e, strings.ReplaceAll(s, testKey, "<KEY>"))
			}
		}
	}
}

// assertPlainError checks the contract of constructor errors: a plain error
// prefixed with the package name, never a *providers.Error.
func assertPlainError(t *testing.T, err error, contains ...string) {
	t.Helper()
	if err == nil {
		t.Fatalf("want an error mentioning %q, got nil", contains)
	}
	var pe *providers.Error
	if errors.As(err, &pe) {
		t.Errorf("constructor error is a *providers.Error: %v", err)
	}
	if !strings.HasPrefix(err.Error(), "anthropic: ") {
		t.Errorf("error = %q, want an \"anthropic: \" prefix", err)
	}
	for _, c := range contains {
		if !strings.Contains(err.Error(), c) {
			t.Errorf("error %q does not mention %q", err, c)
		}
	}
	assertNoKey(t, err)
}

// providerError checks that err is a *providers.Error of this package that
// does not hold the key.
func providerError(t *testing.T, err error) *providers.Error {
	t.Helper()
	var pe *providers.Error
	if !errors.As(err, &pe) {
		t.Fatalf("error %v (%T) is not a *providers.Error", err, err)
	}
	if pe.Provider != "anthropic" || pe.Kind != providers.KindLLM || pe.Op == "" {
		t.Fatalf("unexpected error identity: %+v", pe)
	}
	assertNoKey(t, err)
	return pe
}

// assertCause checks the contract of use-after-Close and context errors: a
// non-retryable *providers.Error that unwraps to cause.
func assertCause(t *testing.T, err error, cause error) {
	t.Helper()
	pe := providerError(t, err)
	if !errors.Is(err, cause) {
		t.Errorf("error %v does not unwrap to %v", err, cause)
	}
	if pe.Retryable || providers.IsRetryable(err) {
		t.Errorf("error %v must not be retryable", err)
	}
}

func TestNewLLMRequiresKey(t *testing.T) {
	t.Setenv(envAPIKey, "")
	_, err := NewLLM(LLMOptions{})
	assertPlainError(t, err, "API key is required (LLMOptions.APIKey or ANTHROPIC_API_KEY)")
	if _, err := NewLLM(LLMOptions{AllowNoAPIKey: true}); err != nil {
		t.Errorf("AllowNoAPIKey: %v", err)
	}
	t.Setenv(envAPIKey, "from-env")
	l, err := NewLLM(LLMOptions{})
	if err != nil {
		t.Fatalf("NewLLM with env key: %v", err)
	}
	if l.apiKey() != "from-env" {
		t.Errorf("key not read from the environment")
	}
	if l.model != DefaultLLMModel || l.endpoint != DefaultBaseURL+"/v1/messages" {
		t.Errorf("defaults = %q, %q", l.model, l.endpoint)
	}
	if l.client != defaultHTTPClient || defaultHTTPClient.Timeout != 0 {
		t.Error("a nil HTTPClient must select the shared client, which has no Timeout")
	}
	_, err = NewLLM(LLMOptions{APIKey: "k", MaxTokens: -1})
	assertPlainError(t, err, "MaxTokens must not be negative")
}

func TestNewLLMTemperature(t *testing.T) {
	temp := 0.3
	rejected := []struct {
		name     string
		opts     LLMOptions
		contains []string
	}{
		{"default configuration", LLMOptions{}, []string{"Temperature", `API version "` + DefaultAPIVersion + `"`, "LLMOptions.APIVersion", LegacyAPIVersion}},
		{"default version, older model", LLMOptions{Model: "claude-sonnet-4-6"}, []string{"Temperature", "LLMOptions.APIVersion", LegacyAPIVersion}},
		{"legacy version, default model", LLMOptions{APIVersion: LegacyAPIVersion}, []string{"Temperature", `model "claude-opus-5-5"`, "LLMOptions.Model"}},
		{"legacy version, dated current model", LLMOptions{APIVersion: LegacyAPIVersion, Model: "claude-haiku-5-5-20260901"}, []string{"Temperature", "LLMOptions.Model"}},
	}
	for _, tc := range rejected {
		t.Run(tc.name, func(t *testing.T) {
			tc.opts.APIKey, tc.opts.Temperature = testKey, &temp
			_, err := NewLLM(tc.opts)
			assertPlainError(t, err, tc.contains...)
			// The same options are fine without Temperature.
			tc.opts.Temperature = nil
			if _, err := NewLLM(tc.opts); err != nil {
				t.Errorf("without Temperature: %v", err)
			}
		})
	}
	// Accepted: never dropped, including for model IDs the package does not
	// know, where the API decides.
	for _, model := range []string{"claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-1", "my-gateway-alias"} {
		l, err := NewLLM(LLMOptions{APIKey: testKey, APIVersion: LegacyAPIVersion, Model: model, Temperature: &temp})
		if err != nil {
			t.Errorf("%s: %v", model, err)
			continue
		}
		if l.temperature == nil || *l.temperature != temp {
			t.Errorf("%s: temperature dropped", model)
		}
	}
}

func TestChatNoKeyHeaderWithoutKey(t *testing.T) {
	t.Setenv(envAPIKey, "")
	header, _, _ := chat(t, llm.Request{Messages: []llm.Message{user("hi")}}, func(o *LLMOptions) {
		o.APIKey, o.AllowNoAPIKey = "", true
	})
	if _, ok := header["X-Api-Key"]; ok {
		t.Errorf("x-api-key = %q, want no header", header.Get("x-api-key"))
	}

	// Without a key nothing is redacted: errors stay intact.
	srv, _ := newServer(t, http.StatusBadRequest, `{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`)
	_, err := newTestLLM(t, srv.URL, func(o *LLMOptions) { o.APIKey, o.AllowNoAPIKey = "", true }).
		Chat(context.Background(), llm.Request{Messages: []llm.Message{user("hi")}})
	if pe := providerError(t, err); !strings.HasSuffix(err.Error(), "invalid_request_error: bad request (request req_test)") {
		t.Errorf("error = %v (%+v)", err, pe)
	}

	// AllowNoAPIKey does not discard a key that is configured.
	header, _, _ = chat(t, llm.Request{Messages: []llm.Message{user("hi")}}, func(o *LLMOptions) { o.AllowNoAPIKey = true })
	if header.Get("x-api-key") != testKey {
		t.Errorf("x-api-key = %q, want the configured key", header.Get("x-api-key"))
	}
}

func TestRequestHeadersAndDefaults(t *testing.T) {
	header, path, body := chat(t, llm.Request{Messages: []llm.Message{user("hi")}}, nil)

	if path != "/v1/messages" {
		t.Errorf("path = %q", path)
	}
	for name, want := range map[string]string{
		"x-api-key":         testKey,
		"anthropic-version": DefaultAPIVersion,
		"content-type":      "application/json",
		"anthropic-beta":    fallbackBeta,
	} {
		if got := header.Get(name); got != want {
			t.Errorf("header %s = %q, want %q", name, got, want)
		}
	}
	wantJSON(t, "body", body, `{
		"model": "claude-opus-5-5",
		"max_tokens": 4096,
		"stream": true,
		"output_config": {"effort": "low"},
		"fallbacks": "default",
		"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
	}`)
}

func TestRequestOptions(t *testing.T) {
	temp := 0.3
	msgs := llm.Request{Messages: []llm.Message{user("hi")}}

	t.Run("temperature omitted when unset", func(t *testing.T) {
		_, _, body := chat(t, msgs, func(o *LLMOptions) {
			o.Model = "claude-sonnet-4-6"
			o.APIVersion = LegacyAPIVersion
		})
		if _, ok := body["temperature"]; ok {
			t.Error("temperature must not be sent when it is not set")
		}
	})
	t.Run("temperature sent where accepted", func(t *testing.T) {
		header, _, body := chat(t, msgs, func(o *LLMOptions) {
			o.Model = "claude-sonnet-4-6"
			o.APIVersion = LegacyAPIVersion
			o.Temperature = &temp
			o.MaxTokens = 256
			o.Effort = EffortMedium
		})
		if body["temperature"] != 0.3 {
			t.Errorf("temperature = %v", body["temperature"])
		}
		if body["max_tokens"] != float64(256) {
			t.Errorf("max_tokens = %v", body["max_tokens"])
		}
		wantJSON(t, "output_config", body["output_config"], `{"effort":"medium"}`)
		if header.Get("anthropic-version") != LegacyAPIVersion {
			t.Errorf("anthropic-version = %q", header.Get("anthropic-version"))
		}
		if header.Get("anthropic-beta") != "" || body["fallbacks"] != nil {
			t.Error("the refusal fallback must not be sent to a model without it")
		}
	})
	t.Run("effort omitted", func(t *testing.T) {
		for _, mutate := range []func(*LLMOptions){
			func(o *LLMOptions) { o.Effort = EffortModelDefault },
			func(o *LLMOptions) { o.Model = "claude-haiku-4-5" },
			func(o *LLMOptions) { o.Model = "claude-haiku-4-5-20251001" },
		} {
			_, _, body := chat(t, msgs, mutate)
			if _, ok := body["output_config"]; ok {
				t.Error("output_config must be omitted")
			}
		}
	})
	t.Run("refusal fallback disabled", func(t *testing.T) {
		header, _, body := chat(t, msgs, func(o *LLMOptions) { o.DisableRefusalFallback = true })
		if header.Get("anthropic-beta") != "" || body["fallbacks"] != nil {
			t.Error("fallback still sent")
		}
	})
}

func TestRequestMessageMapping(t *testing.T) {
	req := llm.Request{Messages: []llm.Message{
		{Role: llm.RoleSystem, Content: "You are a voice agent."},
		{Role: llm.RoleSystem, Content: ""},
		user("What is the weather"),
		user("in Paris and Rome?"),
		{Role: llm.RoleAssistant, Content: "Let me check.", ToolCalls: []llm.ToolCall{
			{ID: "toolu_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Paris"}`)},
			{ID: "toolu_2", Name: "get_weather", Arguments: json.RawMessage(`  {"city":"Rome"} `)},
			{ID: "toolu_3", Name: "now"},
		}},
		{Role: llm.RoleTool, ToolCallID: "toolu_1", Name: "get_weather", Content: "sunny"},
		{Role: llm.RoleTool, ToolCallID: "toolu_2", Name: "get_weather", Content: "unknown city", IsError: true},
		{Role: llm.RoleTool, ToolCallID: "toolu_3", Name: "now", Content: "noon"},
		{Role: llm.RoleSystem, Content: "Keep answers short."},
		user("and tomorrow?"),
		{Role: llm.RoleAssistant}, // empty: dropped
		{Role: llm.RoleAssistant, Content: "Sunny"},
		{Role: llm.RoleAssistant, Content: "again.", Interrupted: true},
		user("thanks"),
	}}
	_, _, body := chat(t, req, nil)

	if got, want := body["system"], "You are a voice agent.\n\nKeep answers short."; got != want {
		t.Errorf("system = %q, want %q", got, want)
	}
	wantJSON(t, "messages", body["messages"], `[
		{"role": "user", "content": [
			{"type": "text", "text": "What is the weather"},
			{"type": "text", "text": "in Paris and Rome?"}
		]},
		{"role": "assistant", "content": [
			{"type": "text", "text": "Let me check."},
			{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
			{"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": {"city": "Rome"}},
			{"type": "tool_use", "id": "toolu_3", "name": "now", "input": {}}
		]},
		{"role": "user", "content": [
			{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
			{"type": "tool_result", "tool_use_id": "toolu_2", "content": "unknown city", "is_error": true},
			{"type": "tool_result", "tool_use_id": "toolu_3", "content": "noon"},
			{"type": "text", "text": "and tomorrow?"}
		]},
		{"role": "assistant", "content": [
			{"type": "text", "text": "Sunny"},
			{"type": "text", "text": "again."}
		]},
		{"role": "user", "content": [{"type": "text", "text": "thanks"}]}
	]`)
}

func TestRequestHistoryEdges(t *testing.T) {
	t.Run("leading assistant message", func(t *testing.T) {
		_, _, body := chat(t, llm.Request{Messages: []llm.Message{
			{Role: llm.RoleAssistant, Content: "Hello, how can I help?"},
			user("hi"),
		}}, nil)
		wantJSON(t, "messages", body["messages"], fmt.Sprintf(`[
			{"role": "user", "content": [{"type": "text", "text": %q}]},
			{"role": "assistant", "content": [{"type": "text", "text": "Hello, how can I help?"}]},
			{"role": "user", "content": [{"type": "text", "text": "hi"}]}
		]`, LeadingUserPlaceholder))
	})
	t.Run("trailing assistant message", func(t *testing.T) {
		_, _, body := chat(t, llm.Request{Messages: []llm.Message{
			user("hi"),
			{Role: llm.RoleAssistant, Content: "Hello"},
		}}, nil)
		wantJSON(t, "messages", body["messages"], fmt.Sprintf(`[
			{"role": "user", "content": [{"type": "text", "text": "hi"}]},
			{"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
			{"role": "user", "content": [{"type": "text", "text": %q}]}
		]`, TrailingUserPlaceholder))
	})
	t.Run("invalid requests", func(t *testing.T) {
		l := newTestLLM(t, "http://127.0.0.1:0", nil)
		for name, req := range map[string]llm.Request{
			"no message":  {Messages: []llm.Message{{Role: llm.RoleSystem, Content: "x"}}},
			"bad role":    {Messages: []llm.Message{{Role: "robot", Content: "x"}}},
			"bad args":    {Messages: []llm.Message{user("x"), {Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "1", Name: "t", Arguments: json.RawMessage(`[1]`)}}}}},
			"bad schema":  {Messages: []llm.Message{user("x")}, Tools: []llm.ToolSpec{{Name: "t", Parameters: json.RawMessage(`{`)}}},
			"bad choice":  {Messages: []llm.Message{user("x")}, Tools: []llm.ToolSpec{{Name: "t"}}, ToolChoice: "sometimes"},
			"empty convo": {},
		} {
			_, err := l.Chat(context.Background(), req)
			var pe *providers.Error
			if !errors.As(err, &pe) || pe.Retryable || pe.Op != "chat" {
				t.Errorf("%s: err = %v, want a non-retryable chat error", name, err)
			}
		}
	})
}

func TestRequestTools(t *testing.T) {
	tools := []llm.ToolSpec{
		{Name: "get_weather", Description: "Get the weather.", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)},
		{Name: "now"},
	}
	const wantTools = `[
		{"name": "get_weather", "description": "Get the weather.", "input_schema": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},
		{"name": "now", "input_schema": {"type":"object","properties":{}}}
	]`
	req := func(choice llm.ToolChoice) llm.Request {
		return llm.Request{
			Messages:   []llm.Message{{Role: llm.RoleSystem, Content: "Be brief."}, user("hi")},
			Tools:      tools,
			ToolChoice: choice,
		}
	}

	t.Run("auto", func(t *testing.T) {
		_, _, body := chat(t, req(llm.ToolChoiceAuto), nil)
		wantJSON(t, "tools", body["tools"], wantTools)
		if _, ok := body["tool_choice"]; ok {
			t.Error("tool_choice must be omitted for auto")
		}
	})
	t.Run("none", func(t *testing.T) {
		_, _, body := chat(t, req(llm.ToolChoiceNone), nil)
		wantJSON(t, "tools", body["tools"], wantTools)
		wantJSON(t, "tool_choice", body["tool_choice"], `{"type":"none"}`)
	})
	t.Run("required where forced tool use exists", func(t *testing.T) {
		_, _, body := chat(t, req(llm.ToolChoiceRequired), func(o *LLMOptions) { o.Model = "claude-opus-4-8" })
		wantJSON(t, "tool_choice", body["tool_choice"], `{"type":"any"}`)
		if body["system"] != "Be brief." {
			t.Errorf("system = %q", body["system"])
		}
	})
	t.Run("required on the default model", func(t *testing.T) {
		_, _, body := chat(t, req(llm.ToolChoiceRequired), nil)
		if _, ok := body["tool_choice"]; ok {
			t.Error("forced tool_choice must not be sent to the default model")
		}
		if want := "Be brief.\n\n" + requiredToolSuffix; body["system"] != want {
			t.Errorf("system = %q, want %q", body["system"], want)
		}
	})
	t.Run("no tools", func(t *testing.T) {
		_, _, body := chat(t, llm.Request{Messages: []llm.Message{user("hi")}, ToolChoice: llm.ToolChoiceRequired}, nil)
		if body["tools"] != nil || body["tool_choice"] != nil || body["system"] != nil {
			t.Errorf("unexpected tool fields: %v", body)
		}
	})
}

func TestHTTPErrors(t *testing.T) {
	cases := []struct {
		status    int
		errType   string
		retryable bool
	}{
		{http.StatusTooManyRequests, "rate_limit_error", true},
		{529, "overloaded_error", true},
		{http.StatusInternalServerError, "api_error", true},
		{http.StatusBadRequest, "invalid_request_error", false},
		{http.StatusUnauthorized, "authentication_error", false},
		{http.StatusForbidden, "permission_error", false},
		{http.StatusNotFound, "not_found_error", false},
	}
	for _, tc := range cases {
		t.Run(tc.errType, func(t *testing.T) {
			// The message echoes the key to prove it is redacted.
			body := fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":"it failed for %s"}}`, tc.errType, testKey)
			srv, _ := newServer(t, tc.status, body)
			_, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), llm.Request{Messages: []llm.Message{user("hi")}})

			pe := providerError(t, err)
			if pe.Op != "chat" {
				t.Errorf("unexpected error identity: %+v", pe)
			}
			if pe.StatusCode != tc.status || pe.Retryable != tc.retryable || providers.IsRetryable(err) != tc.retryable {
				t.Errorf("status %d retryable %v, want %d %v", pe.StatusCode, pe.Retryable, tc.status, tc.retryable)
			}
			var apiErr *APIError
			if !errors.As(err, &apiErr) || apiErr.Type != tc.errType || apiErr.RequestID != "req_test" {
				t.Errorf("APIError = %+v", apiErr)
			}
			if msg := err.Error(); !strings.Contains(msg, "it failed for") || strings.Contains(msg, testKey) {
				t.Errorf("error message %q", msg)
			}
		})
	}

	t.Run("non JSON body", func(t *testing.T) {
		srv, _ := newServer(t, http.StatusBadGateway, "<html>bad gateway</html>")
		_, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), llm.Request{Messages: []llm.Message{user("hi")}})
		if !providers.IsRetryable(err) || !strings.Contains(err.Error(), "bad gateway") {
			t.Errorf("err = %v", err)
		}
	})

	t.Run("key echoed everywhere", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			key := r.Header.Get("x-api-key")
			w.Header().Set("request-id", "req_"+key)
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprintf(w, `{"type":"error","error":{"type":"auth_%s","message":"bad key %s"},"request_id":"req_%s"}`, key, key, key)
		}))
		defer srv.Close()
		_, err := newTestLLM(t, srv.URL, nil).Chat(context.Background(), llm.Request{Messages: []llm.Message{user("hi")}})
		providerError(t, err)
		var apiErr *APIError
		if !errors.As(err, &apiErr) || apiErr.Message != "bad key [REDACTED]" {
			t.Errorf("APIError = %+v", apiErr)
		}
	})
}

func TestTransportError(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()

	_, err := newTestLLM(t, url, nil).Chat(context.Background(), llm.Request{Messages: []llm.Message{user("hi")}})
	if pe := providerError(t, err); !pe.Retryable || pe.StatusCode != 0 || pe.Op != "chat" {
		t.Fatalf("err = %v, want a retryable transport error", err)
	}
	// A clean error keeps its chain.
	var ue *neturl.Error
	if !errors.As(err, &ue) {
		t.Errorf("transport error lost its chain: %v", err)
	}

	// The *url.Error of the transport quotes the URL, which holds the key
	// here. Scrubbing the message is not enough: the original error must
	// not stay reachable through Unwrap.
	_, err = newTestLLM(t, url+"/"+testKey, nil).Chat(context.Background(), llm.Request{Messages: []llm.Message{user("hi")}})
	if pe := providerError(t, err); !pe.Retryable || !strings.Contains(err.Error(), "/[REDACTED]/v1/messages") {
		t.Errorf("err = %v, want a retryable scrubbed transport error", err)
	}
}

func TestChatContextCancelled(t *testing.T) {
	srv, _ := newServer(t, http.StatusOK, okStream)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := newTestLLM(t, srv.URL, nil).Chat(ctx, llm.Request{Messages: []llm.Message{user("hi")}})
	assertCause(t, err, context.Canceled)

	hang := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
	}))
	defer hang.Close()
	ctx, cancel = context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	_, err = newTestLLM(t, hang.URL, nil).Chat(ctx, llm.Request{Messages: []llm.Message{user("hi")}})
	assertCause(t, err, context.DeadlineExceeded)
}

func TestSecretNotPrinted(t *testing.T) {
	l := newTestLLM(t, "http://example.invalid", nil)
	for _, s := range []string{fmt.Sprintf("%v", l), fmt.Sprintf("%+v", l), fmt.Sprintf("%#v", l)} {
		if strings.Contains(s, testKey) {
			t.Errorf("key leaked through fmt: %s", s)
		}
	}
}

// waitErr fails the test when ch does not deliver within a few seconds.
func waitErr(t *testing.T, ch <-chan error) error {
	t.Helper()
	select {
	case err := <-ch:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("Recv did not return")
		return nil
	}
}
