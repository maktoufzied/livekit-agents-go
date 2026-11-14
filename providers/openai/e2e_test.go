//go:build e2e

package openai

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/maktoufzied/livekit-agents-go/llm"
)

// These tests call the real API. Run them with:
//
//	OPENAI_API_KEY=... go test -tags e2e -run E2E -count=1 ./providers/openai/
//
// OPENAI_E2E_LLM_MODEL and OPENAI_E2E_TTS_MODEL override the default models.

func e2eContext(t *testing.T) context.Context {
	t.Helper()
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func e2eLLM(t *testing.T) *LLM {
	t.Helper()
	l, err := NewLLM(LLMOptions{
		Model:           os.Getenv("OPENAI_E2E_LLM_MODEL"),
		ReasoningEffort: os.Getenv("OPENAI_E2E_REASONING_EFFORT"),
	})
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func TestE2EChatText(t *testing.T) {
	ctx := e2eContext(t)
	s, err := e2eLLM(t).Chat(ctx, llm.Request{Messages: []llm.Message{
		{Role: llm.RoleSystem, Content: "Answer with a single word."},
		{Role: llm.RoleUser, Content: "What is the capital of France?"},
	}})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	defer s.Close()
	var text strings.Builder
	var usage *llm.Usage
	for {
		c, err := s.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		text.WriteString(c.Text)
		if c.Usage != nil {
			usage = c.Usage
		}
	}
	if !strings.Contains(text.String(), "Paris") {
		t.Errorf("text = %q", text.String())
	}
	if usage == nil || usage.InputTokens == 0 || usage.OutputTokens == 0 {
		t.Errorf("usage = %+v", usage)
	}
}

func TestE2EChatToolCall(t *testing.T) {
	ctx := e2eContext(t)
	s, err := e2eLLM(t).Chat(ctx, llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "What is the weather in Paris?"}},
		Tools: []llm.ToolSpec{{
			Name:        "get_weather",
			Description: "Returns the current weather in a city.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		}},
		ToolChoice: llm.ToolChoiceRequired,
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	defer s.Close()
	var calls []llm.ToolCall
	for {
		c, err := s.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		calls = append(calls, c.ToolCalls...)
	}
	if len(calls) != 1 || calls[0].Name != "get_weather" || calls[0].ID == "" {
		t.Fatalf("calls = %+v", calls)
	}
	var args struct{ City string }
	if err := json.Unmarshal(calls[0].Arguments, &args); err != nil || !strings.Contains(args.City, "Paris") {
		t.Errorf("arguments = %s (%v)", calls[0].Arguments, err)
	}
}

func TestE2ESynthesize(t *testing.T) {
	ctx := e2eContext(t)
	speech, err := NewTTS(TTSOptions{Model: os.Getenv("OPENAI_E2E_TTS_MODEL")})
	if err != nil {
		t.Fatal(err)
	}
	s, err := speech.Synthesize(ctx, "Hello from the end to end test.")
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	defer s.Close()
	var total time.Duration
	for {
		f, err := s.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if f.Format() != speech.Format() {
			t.Fatalf("frame format = %+v", f.Format())
		}
		total += f.Duration()
	}
	if total < 500*time.Millisecond || total > 15*time.Second {
		t.Errorf("audio duration = %v", total)
	}
}
