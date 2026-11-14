// Command voice-assistant is the minimal voice agent: it joins every room
// it is dispatched to, listens to the first participant and talks back.
//
//	export LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
//	export DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY
//	go run ./examples/voice-assistant
package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/examples/internal/setup"
	"github.com/maktoufzied/livekit-agents-go/roomio"
)

func entrypoint(ctx context.Context, job *agents.Job) error {
	recognizer, model, voice, err := setup.Providers()
	if err != nil {
		return err
	}
	detector, err := setup.VAD(job.Logger())
	if err != nil {
		return err
	}
	session, err := agents.NewSession(agents.SessionOptions{
		Agent: &agents.Agent{
			Instructions: "You are a friendly voice assistant. Your answers are spoken aloud: " +
				"keep them short, conversational and free of markdown or emoji.",
		},
		STT:    recognizer,
		LLM:    model,
		TTS:    voice,
		VAD:    detector,
		Logger: job.Logger(),
	})
	if err != nil {
		return err
	}

	// Log what happens in the conversation.
	session.On(func(ev agents.Event) {
		switch e := ev.(type) {
		case agents.ConversationItemAdded:
			job.Logger().Info("conversation", "role", e.Message.Role, "text", e.Message.Content, "interrupted", e.Message.Interrupted)
		case agents.ErrorEvent:
			job.Logger().Warn("pipeline error", "source", e.Source, "recoverable", e.Recoverable, "error", e.Err)
		}
	})

	// Join the room, wire the audio and run until the participant leaves.
	return roomio.Run(ctx, job, session, roomio.Options{
		OnParticipant: func(_ context.Context, s *agents.Session, _ string) {
			_, _ = s.Say("Hi! How can I help you today?", agents.SayOptions{})
		},
	})
}

func main() {
	logger := setup.Logger()
	slog.SetDefault(logger)

	// Load the voice activity model before the first job needs it.
	if _, err := setup.VAD(logger); err != nil {
		logger.Error("cannot load the voice activity detector", "error", err)
		os.Exit(1)
	}

	// URL and credentials come from LIVEKIT_URL, LIVEKIT_API_KEY and
	// LIVEKIT_API_SECRET. No AgentName: the server dispatches this worker
	// to every new room automatically.
	worker, err := agents.NewWorker(agents.WorkerOptions{Entrypoint: entrypoint, Logger: logger})
	if err != nil {
		logger.Error("cannot create worker", "error", err)
		os.Exit(1)
	}

	// SIGINT/SIGTERM drain the worker: running conversations get
	// WorkerOptions.DrainTimeout to finish.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := worker.Run(ctx); err != nil {
		logger.Error("worker stopped", "error", err)
		os.Exit(1)
	}
}
