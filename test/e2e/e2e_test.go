//go:build e2e

// Package e2e runs the voice pipeline against real providers. It is opt-in
// and needs credentials:
//
//	DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... (or OPENAI_API_KEY=...) \
//	  go test -tags e2e -count=1 -v ./test/e2e/
//
// A spoken question is synthesized with the real TTS, fed to the session as
// microphone audio, transcribed by the real STT, answered by the real LLM
// and spoken by the real TTS into an in-memory output. No LiveKit server is
// involved: transport is covered by test/integration.
package e2e

import (
	"context"
	"errors"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/agentstest"
	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers/anthropic"
	"github.com/maktoufzied/livekit-agents-go/providers/deepgram"
	"github.com/maktoufzied/livekit-agents-go/providers/openai"
	"github.com/maktoufzied/livekit-agents-go/vad"
)

func TestRealProvidersPipeline(t *testing.T) {
	if os.Getenv("DEEPGRAM_API_KEY") == "" {
		t.Skip("DEEPGRAM_API_KEY not set")
	}
	var model llm.LLM
	var err error
	switch {
	case os.Getenv("ANTHROPIC_API_KEY") != "":
		model, err = anthropic.NewLLM(anthropic.LLMOptions{Model: os.Getenv("LLM_MODEL")})
	case os.Getenv("OPENAI_API_KEY") != "":
		model, err = openai.NewLLM(openai.LLMOptions{Model: os.Getenv("LLM_MODEL")})
	default:
		t.Skip("neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set")
	}
	if err != nil {
		t.Fatal(err)
	}
	recognizer, err := deepgram.NewSTT(deepgram.STTOptions{})
	if err != nil {
		t.Fatal(err)
	}
	voice, err := deepgram.NewTTS(deepgram.TTSOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	// 1. Synthesize the user's question at the microphone rate.
	micFormat := audio.Format{SampleRate: 16000, Channels: 1}
	userVoice, err := deepgram.NewTTS(deepgram.TTSOptions{SampleRate: micFormat.SampleRate})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := userVoice.Synthesize(ctx, "What is the capital of France? Answer in one short sentence.")
	if err != nil {
		t.Fatalf("synthesize question: %v", err)
	}
	var question []audio.Frame
	for {
		f, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("synthesize question: %v", err)
		}
		question = append(question, f)
	}
	stream.Close()

	// 2. Run a session on in-memory audio.
	in := agentstest.NewInput(micFormat)
	out := agentstest.NewOutput(voice.Format())
	session, err := agents.NewSession(agents.SessionOptions{
		Agent: &agents.Agent{Instructions: "You are a concise voice assistant. Answer in one short sentence."},
		STT:   recognizer, LLM: model, TTS: voice,
		VAD: vad.NewEnergy(vad.EnergyOptions{}),
	})
	if err != nil {
		t.Fatal(err)
	}
	events := make(chan agents.Event, 256)
	session.On(func(ev agents.Event) {
		switch e := ev.(type) {
		case agents.MetricsCollected:
			t.Logf("metric: %T %+v", e.Metric, e.Metric)
		case agents.UserTranscript:
			t.Logf("user transcript (final=%v): %q", e.Final, e.Text)
		case agents.ErrorEvent:
			t.Logf("error event: %+v", e)
		}
		select {
		case events <- ev:
		default:
		}
	})
	if err := session.Start(ctx, agents.IO{Input: in, Output: out}); err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	// 3. "Speak" the question in real time, then stay silent.
	go func() {
		tick := time.NewTicker(20 * time.Millisecond)
		defer tick.Stop()
		send := func(f audio.Frame) bool {
			select {
			case <-tick.C:
				in.Push(f)
				return true
			case <-ctx.Done():
				return false
			}
		}
		chunker := audio.NewChunker(micFormat, 20*time.Millisecond)
		for _, f := range question {
			for _, c := range chunker.Write(f.BytesLE()) {
				if !send(c) {
					return
				}
			}
		}
		for {
			if !send(audio.Silence(micFormat, 20*time.Millisecond)) {
				return
			}
		}
	}()

	// 4. Expect a spoken answer.
	for {
		select {
		case ev := <-events:
			fin, ok := ev.(agents.SpeechFinished)
			if !ok {
				continue
			}
			if fin.Err != nil || fin.Interrupted {
				t.Fatalf("speech did not complete: %+v", fin)
			}
			msgs := session.History().Messages()
			if len(msgs) != 2 {
				t.Fatalf("history: %+v", msgs)
			}
			t.Logf("user: %q", msgs[0].Content)
			t.Logf("agent: %q (%v of audio)", msgs[1].Content, fin.PlayedDuration)
			if !strings.Contains(strings.ToLower(msgs[0].Content), "france") {
				t.Errorf("transcript does not mention France: %q", msgs[0].Content)
			}
			if !strings.Contains(strings.ToLower(msgs[1].Content), "paris") {
				t.Errorf("answer does not mention Paris: %q", msgs[1].Content)
			}
			if fin.PlayedDuration < 300*time.Millisecond {
				t.Errorf("only %v of audio was played", fin.PlayedDuration)
			}
			return
		case <-ctx.Done():
			t.Fatal("timed out waiting for the agent's answer")
		}
	}
}
