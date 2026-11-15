// Command explicit-dispatch runs a named agent. A named worker is never
// dispatched automatically: it only receives the jobs created for its name,
// either with the agent dispatch API (see ./dispatch) or through the room
// configuration embedded in a participant token.
//
//	go run ./examples/explicit-dispatch                       # the worker
//	go run ./examples/explicit-dispatch/dispatch -room demo   # send it to a room
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/examples/internal/setup"
	"github.com/maktoufzied/livekit-agents-go/roomio"
)

// AgentName must match the name used when creating the dispatch.
const AgentName = "support-agent"

// dispatchMetadata is the JSON passed along with the dispatch.
type dispatchMetadata struct {
	Customer string `json:"customer"`
	Language string `json:"language"`
}

func entrypoint(ctx context.Context, job *agents.Job) error {
	var meta dispatchMetadata
	if raw := job.Metadata(); raw != "" {
		if err := json.Unmarshal([]byte(raw), &meta); err != nil {
			job.Logger().Warn("ignoring invalid dispatch metadata", "error", err)
		}
	}
	if meta.Customer == "" {
		meta.Customer = "there"
	}

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
			Instructions: "You are a support agent on a phone-like call with a customer named " + meta.Customer +
				". Be concise and helpful; your answers are spoken aloud.",
		},
		STT: recognizer, LLM: model, TTS: voice,
		VAD:          detector,
		TurnDetector: endOfTurn,
		Logger:       job.Logger(),
	})
	if err != nil {
		return err
	}
	setup.LogMetrics(session, job.Logger())
	return roomio.Run(ctx, job, session, roomio.Options{
		OnParticipant: func(_ context.Context, s *agents.Session, _ string) {
			_, _ = s.Say("Hello "+meta.Customer+", thanks for calling support. What can I do for you?", agents.SayOptions{})
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
	worker, err := agents.NewWorker(agents.WorkerOptions{
		AgentName:         AgentName,
		Entrypoint:        entrypoint,
		MaxConcurrentJobs: 8,
		DrainTimeout:      5 * time.Minute,
		Logger:            logger,
		// Decide per request. Here: accept everything, with a readable
		// participant name; call req.Reject() to pass on a job.
		OnJobRequest: func(ctx context.Context, req *agents.JobRequest) error {
			return req.Accept(ctx, agents.AcceptOptions{Name: "Support agent"})
		},
		OnStats: func(st agents.WorkerStats) {
			logger.Debug("worker stats", "active_jobs", st.ActiveJobs, "load", st.Load, "goroutines", st.Goroutines, "heap_bytes", st.HeapBytes)
		},
	})
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
