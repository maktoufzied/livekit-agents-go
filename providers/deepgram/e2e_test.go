//go:build e2e

package deepgram

import (
	"context"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/stt"
)

// TestE2ERoundTrip synthesizes a phrase with the live API and transcribes it
// back. Run with: DEEPGRAM_API_KEY=... go test -tags e2e -run E2E ./providers/deepgram/
func TestE2ERoundTrip(t *testing.T) {
	if os.Getenv(apiKeyEnv) == "" {
		t.Skip(apiKeyEnv + " not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	synth, err := NewTTS(TTSOptions{SampleRate: 16000})
	if err != nil {
		t.Fatal(err)
	}
	out, err := synth.Synthesize(ctx, "Hello, this is a test of the speech pipeline.")
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	defer out.Close()
	var frames []audio.Frame
	var loudest float64
	var total time.Duration
	for {
		f, err := out.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("TTS Recv: %v", err)
		}
		frames = append(frames, f)
		total += f.Duration()
		loudest = max(loudest, audio.RMS(f))
	}
	t.Logf("TTS: %d frames, %v, peak frame RMS %.3f", len(frames), total, loudest)
	if total < 500*time.Millisecond || loudest < 0.01 {
		t.Fatalf("audio too short or silent: %v, RMS %.4f", total, loudest)
	}

	rec, err := NewSTT(STTOptions{Language: "en"})
	if err != nil {
		t.Fatal(err)
	}
	in, err := rec.Stream(ctx, stt.StreamOptions{Format: synth.Format()})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer in.Close()
	go func() {
		for _, f := range frames {
			if in.Push(f) != nil {
				return
			}
			time.Sleep(5 * time.Millisecond) // 4x real time
		}
		in.CloseSend()
	}()
	var finals []string
	for {
		ev, err := in.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("STT Recv: %v", err)
		}
		t.Logf("event %d %q", ev.Type, ev.Text)
		if ev.Type == stt.Final {
			finals = append(finals, ev.Text)
		}
	}
	transcript := strings.Join(finals, " ")
	if strings.TrimSpace(transcript) == "" {
		t.Fatal("no final transcript")
	}
	if !strings.Contains(strings.ToLower(transcript), "test") {
		t.Errorf("transcript %q does not contain the spoken phrase", transcript)
	}
}
