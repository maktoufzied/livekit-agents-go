// Package setup builds the providers shared by the examples from
// environment variables.
package setup

import (
	"fmt"
	"log/slog"
	"os"
	"time"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/agentstest"
	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers/anthropic"
	"github.com/maktoufzied/livekit-agents-go/providers/deepgram"
	"github.com/maktoufzied/livekit-agents-go/providers/openai"
	"github.com/maktoufzied/livekit-agents-go/stt"
	"github.com/maktoufzied/livekit-agents-go/tts"
)

// Logger returns a structured logger that redacts credential-like fields.
// Set LOG_LEVEL=debug for more detail.
func Logger() *slog.Logger {
	level := slog.LevelInfo
	if os.Getenv("LOG_LEVEL") == "debug" {
		level = slog.LevelDebug
	}
	return slog.New(agents.NewRedactingHandler(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level})))
}

// Providers selects the speech and language providers:
//
//   - AGENT_FAKE_PROVIDERS=1: no credentials needed. The agent answers any
//     burst of sound with a scripted reply rendered as a tone. Useful to
//     check connectivity and interruptions.
//   - otherwise Deepgram for STT and TTS (DEEPGRAM_API_KEY), and for the LLM
//     Anthropic when ANTHROPIC_API_KEY is set, else OpenAI (OPENAI_API_KEY).
//     LLM_MODEL overrides the model.
func Providers() (stt.STT, llm.LLM, tts.TTS, error) {
	if os.Getenv("AGENT_FAKE_PROVIDERS") == "1" {
		voice := agentstest.NewTTS()
		voice.Tone = true
		voice.PerChar = 60 * time.Millisecond
		model := agentstest.NewLLM()
		model.Default = agentstest.Text("This is a scripted reply from the fake model. Speak while you hear the tone to interrupt it.")
		return &agentstest.EnergySTT{}, model, voice, nil
	}

	recognizer, err := deepgram.NewSTT(deepgram.STTOptions{})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("stt: %w", err)
	}
	voice, err := deepgram.NewTTS(deepgram.TTSOptions{})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("tts: %w", err)
	}
	var model llm.LLM
	if os.Getenv("ANTHROPIC_API_KEY") != "" {
		model, err = anthropic.NewLLM(anthropic.LLMOptions{Model: os.Getenv("LLM_MODEL")})
	} else {
		model, err = openai.NewLLM(openai.LLMOptions{Model: os.Getenv("LLM_MODEL")})
	}
	if err != nil {
		return nil, nil, nil, fmt.Errorf("llm: %w", err)
	}
	return recognizer, model, voice, nil
}
