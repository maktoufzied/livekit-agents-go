package smartturn_test

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/agentstest"
	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/turn/smartturn"
	"github.com/maktoufzied/livekit-agents-go/vad/silero"
)

// A session with the Silero VAD and Smart Turn, fed recorded speech: when
// the user stops in the middle of a sentence the reply waits for the maximum
// endpointing delay, when the sentence is finished it does not.
func TestSessionWaitsForUnfinishedSentence(t *testing.T) {
	required := os.Getenv("SMARTTURN_REQUIRED") == "1"
	detector, err := silero.New(silero.Options{})
	if errors.Is(err, silero.ErrRuntimeUnavailable) && !required {
		t.Skipf("ONNX Runtime not installed: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer detector.Close()
	endOfTurn, err := smartturn.New(smartturn.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer endOfTurn.Close()

	readClip := func(name string) []int16 {
		raw, err := os.ReadFile(filepath.Join("testdata", name+".pcm"))
		if err != nil {
			t.Fatal(err)
		}
		clip := make([]int16, len(raw)/2)
		for i := range clip {
			clip[i] = int16(binary.LittleEndian.Uint16(raw[2*i:]))
		}
		return clip
	}

	const minDelay, maxDelay = 200 * time.Millisecond, 2 * time.Second
	// replyDelay plays the clip, then silence, sends the transcript and
	// returns how long the reply took to start.
	replyDelay := func(clip string, transcript string) time.Duration {
		t.Helper()
		recognizer := agentstest.NewSTT()
		voice := agentstest.NewTTS()
		in := agentstest.NewInput(audio.Format{SampleRate: 16000, Channels: 1})
		session, err := agents.NewSession(agents.SessionOptions{
			Agent:        &agents.Agent{Instructions: "test"},
			STT:          recognizer,
			LLM:          agentstest.NewLLM(agentstest.Text("Reply.")),
			TTS:          voice,
			VAD:          detector,
			TurnDetector: endOfTurn,
			Turn:         agents.TurnOptions{MinEndpointingDelay: minDelay, MaxEndpointingDelay: maxDelay},
			Logger:       slog.New(slog.NewTextHandler(io.Discard, nil)),
		})
		if err != nil {
			t.Fatal(err)
		}
		started := make(chan time.Time, 1)
		listening := make(chan struct{}, 1)
		spoke := false
		session.On(func(ev agents.Event) {
			switch e := ev.(type) {
			case agents.UserStateChanged:
				if e.New == agents.UserSpeaking {
					spoke = true
				} else if spoke && e.New == agents.UserListening {
					select {
					case listening <- struct{}{}:
					default:
					}
				}
			case agents.SpeechStarted:
				select {
				case started <- time.Now():
				default:
				}
			}
		})
		if err := session.Start(context.Background(), agents.IO{Input: in, Output: agentstest.NewOutput(voice.Format())}); err != nil {
			t.Fatal(err)
		}
		defer session.Close()

		samples := readClip(clip)
		for i := 0; i < len(samples); i += 320 {
			in.Push(audio.Frame{Data: samples[i:min(i+320, len(samples))], SampleRate: 16000, Channels: 1})
		}
		in.PushSilence(1500 * time.Millisecond)
		select {
		case <-listening:
		case <-time.After(10 * time.Second):
			t.Fatal("the VAD did not report the end of speech")
		}
		sent := time.Now()
		recognizer.Final(transcript)
		select {
		case at := <-started:
			return at.Sub(sent)
		case <-time.After(10 * time.Second):
			t.Fatal("no reply")
			return 0
		}
	}

	unfinished := replyDelay("unfinished_16k", "I can't seem to, um...")
	finished := replyDelay("finished_16k", "I can't seem to, um, find the return label.")
	t.Logf("reply after an unfinished sentence: %v, after a finished one: %v", unfinished.Round(time.Millisecond), finished.Round(time.Millisecond))
	// The delays count from the end of the speech, which the VAD reports
	// about 0.6 s late; the transcript is sent at that moment.
	if unfinished < maxDelay-800*time.Millisecond {
		t.Errorf("unfinished sentence: reply after %v, want the maximum endpointing delay (%v after the speech)", unfinished, maxDelay)
	}
	if finished > 700*time.Millisecond {
		t.Errorf("finished sentence: reply after %v, want it at once", finished)
	}
}
