package agents_test

import (
	"context"
	"fmt"
	"io"
	"log/slog"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/agentstest"
	"github.com/maktoufzied/livekit-agents-go/llm"
)

var quiet = slog.New(slog.NewTextHandler(io.Discard, nil))

// A session can run without a room: here it is driven by GenerateReply and
// plays into an in-memory output. Fake providers keep the example
// deterministic; swap them for real ones in an agent.
func ExampleSession() {
	voice := agentstest.NewTTS()
	out := agentstest.NewOutput(voice.Format())
	out.Speed = 100 // play faster than real time

	session, err := agents.NewSession(agents.SessionOptions{
		Agent:  &agents.Agent{Instructions: "You are a helpful assistant."},
		LLM:    agentstest.NewLLM(agentstest.Text("Paris is the capital of France.")),
		TTS:    voice,
		Logger: quiet,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := session.Start(context.Background(), agents.IO{Output: out}); err != nil {
		fmt.Println(err)
		return
	}
	defer session.Close()

	speech, err := session.GenerateReply(agents.ReplyOptions{UserInput: "What is the capital of France?"})
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := speech.Wait(context.Background()); err != nil {
		fmt.Println(err)
		return
	}
	for _, m := range session.History().Messages() {
		fmt.Printf("%s: %s\n", m.Role, m.Content)
	}
	// Output:
	// user: What is the capital of France?
	// assistant: Paris is the capital of France.
}

// Events are delivered in order to handlers registered with On. Use a type
// switch to pick the ones you need.
func ExampleSession_On() {
	voice := agentstest.NewTTS()
	out := agentstest.NewOutput(voice.Format())
	out.Speed = 100
	session, _ := agents.NewSession(agents.SessionOptions{LLM: agentstest.NewLLM(), TTS: voice, Logger: quiet})

	done := make(chan struct{})
	session.On(func(ev agents.Event) {
		switch e := ev.(type) {
		case agents.AgentStateChanged:
			fmt.Println("agent is", e.New)
		case agents.ConversationItemAdded:
			fmt.Printf("%s said %q\n", e.Message.Role, e.Message.Content)
		case agents.SessionClosed:
			close(done)
		}
	})

	_ = session.Start(context.Background(), agents.IO{Output: out})
	speech, _ := session.Say("Welcome aboard.", agents.SayOptions{})
	_ = speech.Wait(context.Background())
	session.Close()
	<-done
	// Output:
	// agent is listening
	// agent is thinking
	// agent is speaking
	// assistant said "Welcome aboard."
	// agent is listening
}

// Tools are plain functions with a typed argument struct. The model sees a
// JSON Schema derived from the struct, and arguments are validated against
// it before the function runs.
func ExampleSession_tools() {
	type args struct {
		City string `json:"city" desc:"City name"`
	}
	weather := llm.MustTool("get_weather", "Get the weather for a city.",
		func(_ context.Context, a args) (any, error) {
			return map[string]any{"city": a.City, "celsius": 21}, nil
		})

	voice := agentstest.NewTTS()
	out := agentstest.NewOutput(voice.Format())
	out.Speed = 100
	// The fake model first asks for the tool, then answers.
	model := agentstest.NewLLM(
		agentstest.ToolCalls(llm.ToolCall{ID: "call_1", Name: "get_weather", Arguments: []byte(`{"city":"Lyon"}`)}),
		agentstest.Text("It is 21 degrees in Lyon."),
	)
	session, _ := agents.NewSession(agents.SessionOptions{
		Agent: &agents.Agent{Tools: []llm.Tool{weather}},
		LLM:   model, TTS: voice, Logger: quiet,
	})
	_ = session.Start(context.Background(), agents.IO{Output: out})
	defer session.Close()

	speech, _ := session.GenerateReply(agents.ReplyOptions{UserInput: "Weather in Lyon?"})
	_ = speech.Wait(context.Background())
	for _, m := range session.History().Messages() {
		switch {
		case len(m.ToolCalls) > 0:
			fmt.Printf("%s: calls %s(%s)\n", m.Role, m.ToolCalls[0].Name, m.ToolCalls[0].Arguments)
		default:
			fmt.Printf("%s: %s\n", m.Role, m.Content)
		}
	}
	// Output:
	// user: Weather in Lyon?
	// assistant: calls get_weather({"city":"Lyon"})
	// tool: {"celsius":21,"city":"Lyon"}
	// assistant: It is 21 degrees in Lyon.
}
