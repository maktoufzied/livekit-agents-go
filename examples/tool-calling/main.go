// Command tool-calling shows typed tools: schema generation, argument
// validation, timeouts, model-visible errors, and a tool with side effects
// that must not be abandoned when the user interrupts.
//
//	go run ./examples/tool-calling
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/examples/internal/setup"
	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/roomio"
)

// The JSON Schema sent to the model is derived from these structs. Fields
// are required unless tagged omitempty.
type weatherArgs struct {
	City string `json:"city" desc:"City name, for example Paris"`
	Unit string `json:"unit,omitempty" enum:"celsius,fahrenheit" desc:"Temperature unit, celsius by default"`
}

type bookingArgs struct {
	Name   string `json:"name" desc:"Name the table is booked under"`
	Guests int    `json:"guests" desc:"Number of guests"`
	Time   string `json:"time" desc:"Time of the booking, 24h format HH:MM"`
}

// bookings is a stand-in for an external system with side effects.
type bookings struct {
	mu   sync.Mutex
	next int
}

func (b *bookings) book(ctx context.Context, a bookingArgs) (any, error) {
	if a.Guests < 1 || a.Guests > 12 {
		// A ToolError message is shown to the model, which can then
		// explain the problem to the user.
		return nil, llm.NewToolError("bookings are possible for 1 to 12 guests, got %d", a.Guests)
	}
	select {
	case <-time.After(300 * time.Millisecond): // pretend to call a booking API
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.next++
	return map[string]any{"confirmation": fmt.Sprintf("BK-%04d", b.next), "name": a.Name, "guests": a.Guests, "time": a.Time}, nil
}

func tools() []llm.Tool {
	store := &bookings{}
	return []llm.Tool{
		// Read-only and quick: cancelled if the user interrupts.
		llm.MustTool("get_weather", "Get the current weather for a city.",
			func(_ context.Context, a weatherArgs) (any, error) {
				temp, unit := 21.0, "celsius"
				if a.Unit == "fahrenheit" {
					temp, unit = 69.8, "fahrenheit"
				}
				return map[string]any{"city": strings.TrimSpace(a.City), "temperature": temp, "unit": unit, "conditions": "sunny"}, nil
			},
			llm.WithTimeout(5*time.Second)),

		// Has a side effect: it runs to completion even if the user starts
		// talking, and it is never retried by the SDK.
		llm.MustTool("book_table", "Book a restaurant table. Only call it once the user confirmed name, guests and time.",
			store.book,
			llm.WithTimeout(10*time.Second), llm.WithUninterruptible()),
	}
}

func entrypoint(ctx context.Context, job *agents.Job) error {
	recognizer, model, voice, err := setup.Providers()
	if err != nil {
		return err
	}
	detector, err := setup.VAD(job.Logger())
	if err != nil {
		return err
	}
	endOfTurn, err := setup.TurnDetector(job.Logger())
	if err != nil {
		return err
	}
	session, err := agents.NewSession(agents.SessionOptions{
		Agent: &agents.Agent{
			Instructions: "You are the voice assistant of a restaurant. You can check the weather for the terrace " +
				"and book tables. Keep answers short and spoken-style. Confirm details before booking.",
			Tools: tools(),
		},
		STT: recognizer, LLM: model, TTS: voice,
		VAD:          detector,
		TurnDetector: endOfTurn,
		Tools:        agents.ToolOptions{MaxSteps: 4},
		Logger:       job.Logger(),
	})
	if err != nil {
		return err
	}
	session.On(func(ev agents.Event) {
		switch e := ev.(type) {
		case agents.ToolCallStarted:
			job.Logger().Info("tool call", "tool", e.Call.Name, "arguments", string(e.Call.Arguments))
		case agents.ToolCallFinished:
			job.Logger().Info("tool result", "tool", e.Call.Name, "output", e.Output, "is_error", e.IsError, "duration", e.Duration)
		}
	})
	return roomio.Run(ctx, job, session, roomio.Options{
		OnParticipant: func(_ context.Context, s *agents.Session, _ string) {
			_, _ = s.GenerateReply(agents.ReplyOptions{Instructions: "Greet the caller and say what you can do."})
		},
	})
}

func main() {
	logger := setup.Logger()
	slog.SetDefault(logger)

	// Load the models before the first job needs them.
	if _, err := setup.VAD(logger); err != nil {
		logger.Error("cannot load the voice activity detector", "error", err)
		os.Exit(1)
	}
	if _, err := setup.TurnDetector(logger); err != nil {
		logger.Error("cannot load the end-of-turn model", "error", err)
		os.Exit(1)
	}
	worker, err := agents.NewWorker(agents.WorkerOptions{Entrypoint: entrypoint, Logger: logger})
	if err != nil {
		logger.Error("cannot create worker", "error", err)
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := worker.Run(ctx); err != nil {
		logger.Error("worker stopped", "error", err)
		os.Exit(1)
	}
}
