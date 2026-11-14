//go:build e2e

package anthropic

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/maktoufzied/livekit-agents-go/llm"
)

// The tests in this file call the real API. Run them with:
//
//	ANTHROPIC_API_KEY=... go test -tags e2e -run E2E ./providers/anthropic/
//
// ANTHROPIC_E2E_MODEL overrides the model.

func e2eLLM(t *testing.T) *LLM {
	t.Helper()
	if os.Getenv(envAPIKey) == "" {
		t.Skip("ANTHROPIC_API_KEY is not set")
	}
	l, err := NewLLM(LLMOptions{Model: os.Getenv("ANTHROPIC_E2E_MODEL")})
	if err != nil {
		t.Fatalf("NewLLM: %v", err)
	}
	return l
}

// e2eRun performs one generation and returns its text and tool calls.
func e2eRun(t *testing.T, l *LLM, req llm.Request) (string, []llm.ToolCall, *llm.Usage) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	st, err := l.Chat(ctx, req)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	defer st.Close()

	var (
		text  strings.Builder
		calls []llm.ToolCall
		usage *llm.Usage
	)
	chunks, err := collect(st)
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	for _, c := range chunks {
		text.WriteString(c.Text)
		calls = append(calls, c.ToolCalls...)
		if c.Usage != nil {
			usage = c.Usage
		}
	}
	return text.String(), calls, usage
}

func TestE2EStreamedReply(t *testing.T) {
	l := e2eLLM(t)
	text, calls, usage := e2eRun(t, l, llm.Request{Messages: []llm.Message{
		{Role: llm.RoleSystem, Content: "You are a voice assistant. Answer in one short sentence."},
		{Role: llm.RoleAssistant, Content: "Hello, how can I help?"},
		user("What is the capital of France?"),
	}})
	if !strings.Contains(text, "Paris") {
		t.Errorf("reply = %q, want it to mention Paris", text)
	}
	if len(calls) != 0 {
		t.Errorf("unexpected tool calls: %v", calls)
	}
	if usage == nil || usage.InputTokens == 0 || usage.OutputTokens == 0 {
		t.Errorf("usage = %+v", usage)
	}
}

func TestE2EToolRoundTrip(t *testing.T) {
	l := e2eLLM(t)
	tools := []llm.ToolSpec{{
		Name:        "get_weather",
		Description: "Get the current weather of a city. Call this whenever the user asks about the weather.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}`),
	}}
	history := []llm.Message{
		{Role: llm.RoleSystem, Content: "You are a voice assistant. Use the get_weather tool for weather questions and answer in one short sentence."},
		user("What is the weather in Paris right now?"),
	}

	text, calls, _ := e2eRun(t, l, llm.Request{Messages: history, Tools: tools})
	if len(calls) != 1 || calls[0].Name != "get_weather" || calls[0].ID == "" {
		t.Fatalf("tool calls = %+v (text %q), want one get_weather call", calls, text)
	}
	var args struct {
		City string `json:"city"`
	}
	if err := json.Unmarshal(calls[0].Arguments, &args); err != nil || !strings.Contains(args.City, "Paris") {
		t.Fatalf("arguments = %s (%v)", calls[0].Arguments, err)
	}

	history = append(history,
		llm.Message{Role: llm.RoleAssistant, Content: text, ToolCalls: calls},
		llm.Message{Role: llm.RoleTool, ToolCallID: calls[0].ID, Name: calls[0].Name, Content: `{"temperature_celsius":21,"conditions":"sunny"}`},
	)
	text, calls, _ = e2eRun(t, l, llm.Request{Messages: history, Tools: tools})
	if len(calls) != 0 {
		t.Errorf("unexpected second tool call: %+v", calls)
	}
	if !strings.Contains(text, "21") && !strings.Contains(strings.ToLower(text), "sunny") {
		t.Errorf("reply = %q, want it to use the tool result", text)
	}
}
