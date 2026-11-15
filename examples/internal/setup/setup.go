// Package setup builds the providers shared by the examples from
// environment variables.
package setup

import (
	"fmt"
	"log/slog"
	"os"
	"sync"
	"time"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/agentstest"
	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers/anthropic"
	"github.com/maktoufzied/livekit-agents-go/providers/deepgram"
	"github.com/maktoufzied/livekit-agents-go/providers/openai"
	"github.com/maktoufzied/livekit-agents-go/stt"
	"github.com/maktoufzied/livekit-agents-go/tts"
	"github.com/maktoufzied/livekit-agents-go/turn/smartturn"
	"github.com/maktoufzied/livekit-agents-go/vad"
	"github.com/maktoufzied/livekit-agents-go/vad/silero"
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

var (
	vadOnce   sync.Once
	sharedVAD vad.VAD
	vadErr    error
)

// VAD returns the voice activity detector shared by every session of the
// process. The VAD environment variable selects it:
//
//   - silero: the Silero model. Needs the ONNX Runtime library, see
//     scripts/install-onnxruntime.sh.
//   - energy: the dependency-free energy detector.
//   - unset: Silero when ONNX Runtime is installed, the energy detector
//     otherwise. With AGENT_FAKE_PROVIDERS=1 always the energy detector,
//     because the fake recognizer reacts to any sound, not only to speech.
//
// The model is loaded once: call VAD from the entrypoint of each job.
func VAD(logger *slog.Logger) (vad.VAD, error) {
	vadOnce.Do(func() {
		choice := os.Getenv("VAD")
		if choice == "" && os.Getenv("AGENT_FAKE_PROVIDERS") == "1" {
			choice = "energy"
		}
		switch choice {
		case "energy":
			sharedVAD = vad.NewEnergy(vad.EnergyOptions{})
		case "silero":
			sharedVAD, vadErr = silero.New(silero.Options{})
		case "":
			detector, err := silero.New(silero.Options{})
			if err != nil {
				logger.Warn("Silero VAD unavailable, using the energy detector", "error", err)
				sharedVAD = vad.NewEnergy(vad.EnergyOptions{})
				return
			}
			sharedVAD = detector
		default:
			vadErr = fmt.Errorf("VAD=%q: want silero or energy", choice)
		}
		if vadErr == nil {
			logger.Info("voice activity detector", "vad", fmt.Sprintf("%T", sharedVAD))
		}
	})
	return sharedVAD, vadErr
}

var (
	turnOnce   sync.Once
	sharedTurn agents.TurnDetector
	turnErr    error
)

// TurnDetector returns the end-of-turn model shared by every session of the
// process, or nil for none. With it the agent waits when the user pauses in
// the middle of a sentence. The TURN_DETECTOR environment variable selects
// it:
//
//   - smartturn: the Smart Turn model. Needs the ONNX Runtime library, see
//     scripts/install-onnxruntime.sh.
//   - off: none, a turn ends after a fixed silence.
//   - unset: Smart Turn when ONNX Runtime is installed, none otherwise. With
//     AGENT_FAKE_PROVIDERS=1 always none: the test sounds are not speech.
//
// The model is loaded once: call TurnDetector from the entrypoint of each job.
func TurnDetector(logger *slog.Logger) (agents.TurnDetector, error) {
	turnOnce.Do(func() {
		choice := os.Getenv("TURN_DETECTOR")
		if choice == "" && os.Getenv("AGENT_FAKE_PROVIDERS") == "1" {
			choice = "off"
		}
		switch choice {
		case "off":
		case "smartturn", "":
			detector, err := smartturn.New(smartturn.Options{})
			switch {
			case err == nil:
				sharedTurn = detector
			case choice == "":
				logger.Warn("Smart Turn unavailable, turns end after a fixed silence", "error", err)
			default:
				turnErr = err
			}
		default:
			turnErr = fmt.Errorf("TURN_DETECTOR=%q: want smartturn or off", choice)
		}
		if turnErr == nil {
			logger.Info("end-of-turn model", "enabled", sharedTurn != nil)
		}
	})
	return sharedTurn, turnErr
}

// LogMetrics logs every latency and usage measurement of the session at
// debug level (LOG_LEVEL=debug), to see where the time of a reply goes.
func LogMetrics(session *agents.Session, logger *slog.Logger) {
	session.On(func(ev agents.Event) {
		if m, ok := ev.(agents.MetricsCollected); ok {
			logger.Debug("metric", "kind", fmt.Sprintf("%T", m.Metric), "value", fmt.Sprintf("%+v", m.Metric))
		}
	})
}
