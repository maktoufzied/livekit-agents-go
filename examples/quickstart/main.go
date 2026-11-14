package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/providers/anthropic"
	"github.com/maktoufzied/livekit-agents-go/providers/deepgram"
	"github.com/maktoufzied/livekit-agents-go/roomio"
	"github.com/maktoufzied/livekit-agents-go/vad"
)

// One call per job: a room the agent was dispatched to.
func entrypoint(ctx context.Context, job *agents.Job) error {
	stt, err := deepgram.NewSTT(deepgram.STTOptions{}) // DEEPGRAM_API_KEY
	if err != nil {
		return err
	}
	tts, err := deepgram.NewTTS(deepgram.TTSOptions{})
	if err != nil {
		return err
	}
	llm, err := anthropic.NewLLM(anthropic.LLMOptions{}) // ANTHROPIC_API_KEY
	if err != nil {
		return err
	}

	session, err := agents.NewSession(agents.SessionOptions{
		Agent: &agents.Agent{Instructions: "You are a helpful voice assistant. Keep answers short."},
		STT:   stt,
		LLM:   llm,
		TTS:   tts,
		VAD:   vad.NewEnergy(vad.EnergyOptions{}),
	})
	if err != nil {
		return err
	}

	// Join the room, listen to the participant, speak, until they leave.
	return roomio.Run(ctx, job, session, roomio.Options{})
}

func main() {
	// LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
	worker, err := agents.NewWorker(agents.WorkerOptions{Entrypoint: entrypoint})
	if err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := worker.Run(ctx); err != nil { // SIGTERM drains running jobs
		log.Fatal(err)
	}
}
