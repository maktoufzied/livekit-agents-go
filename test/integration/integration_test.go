//go:build integration

// Package integration runs the full vertical slice against a real LiveKit
// server with fake speech providers: worker registration, explicit
// dispatch, room join, audio in both directions over WebRTC, an
// interruption, and a clean shutdown.
//
// Start a server first (scripts/integration.sh does it for you):
//
//	livekit-server --dev
//	go test -tags integration -count=1 -v ./test/integration/
package integration

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	media "github.com/livekit/media-sdk"
	"github.com/livekit/protocol/livekit"
	protoLogger "github.com/livekit/protocol/logger"
	lksdk "github.com/livekit/server-sdk-go/v2"
	lkmedia "github.com/livekit/server-sdk-go/v2/pkg/media"
	"github.com/pion/webrtc/v4"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/agentstest"
	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/roomio"
	"github.com/maktoufzied/livekit-agents-go/vad"
)

func env(name, def string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return def
}

var (
	lkURL    = env("LIVEKIT_URL", "ws://localhost:7880")
	lkKey    = env("LIVEKIT_API_KEY", "devkey")
	lkSecret = env("LIVEKIT_API_SECRET", "secret")
)

// levelMeter records when the audio received from the agent was audible.
type levelMeter struct {
	mu        sync.Mutex
	lastLoud  time.Time
	loudTotal time.Duration
}

func (m *levelMeter) WriteSample(s media.PCM16Sample) error {
	f := audio.Frame{Data: s, SampleRate: 48000, Channels: 1}
	if audio.RMS(f) > 0.05 {
		m.mu.Lock()
		m.lastLoud = time.Now()
		m.loudTotal += f.Duration()
		m.mu.Unlock()
	}
	return nil
}
func (m *levelMeter) Close() error { return nil }

func (m *levelMeter) audible() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return time.Since(m.lastLoud) < 200*time.Millisecond
}

func (m *levelMeter) total() time.Duration {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.loudTotal
}

func waitFor(t *testing.T, what string, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Logf("ok: %s", what)
}

const longReply = "This is the first sentence of a long answer from the agent. It keeps talking for quite a while. " +
	"There is a third sentence here, and it is not short either. A fourth sentence follows to make the reply longer. " +
	"The fifth sentence is still going. And finally a sixth sentence ends this long reply."

func TestVoiceAgentVerticalSlice(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	agentName := fmt.Sprintf("it-agent-%d", time.Now().UnixNano())
	roomName := fmt.Sprintf("it-room-%d", time.Now().UnixNano())

	// --- Agent side -------------------------------------------------------
	var (
		mu       sync.Mutex
		events   []agents.Event
		session  *agents.Session
		jobMeta  string
		jobsDone atomic.Int32
	)
	record := func(ev agents.Event) {
		mu.Lock()
		events = append(events, ev)
		mu.Unlock()
	}
	finished := func() []agents.SpeechFinished {
		mu.Lock()
		defer mu.Unlock()
		var out []agents.SpeechFinished
		for _, ev := range events {
			if f, ok := ev.(agents.SpeechFinished); ok {
				out = append(out, f)
			}
		}
		return out
	}

	entrypoint := func(ctx context.Context, job *agents.Job) error {
		defer jobsDone.Add(1)
		ttsFake := agentstest.NewTTS()
		ttsFake.Tone = true
		ttsFake.PerChar = 60 * time.Millisecond // the long reply lasts ~18s
		s, err := agents.NewSession(agents.SessionOptions{
			Agent: &agents.Agent{Instructions: "integration test agent"},
			STT:   &agentstest.EnergySTT{Script: []string{"tell me something long", "stop, I have another question"}},
			LLM:   agentstest.NewLLM(agentstest.Text(longReply), agentstest.Text("Sure. Here is a short second answer.")),
			TTS:   ttsFake,
			VAD:   vad.NewEnergy(vad.EnergyOptions{}),
			// The test interrupts the agent two seconds into its first
			// reply, and its microphone has no echo.
			Turn:   agents.TurnOptions{EchoWarmup: -1},
			Logger: job.Logger(),
		})
		if err != nil {
			return err
		}
		s.On(record)
		mu.Lock()
		session, jobMeta = s, job.Metadata()
		mu.Unlock()
		return roomio.Run(ctx, job, s, roomio.Options{})
	}

	worker, err := agents.NewWorker(agents.WorkerOptions{
		URL: lkURL, APIKey: lkKey, APISecret: agents.Secret(lkSecret),
		AgentName:  agentName,
		Entrypoint: entrypoint,
		Logger:     logger,
	})
	if err != nil {
		t.Fatal(err)
	}
	workerCtx, stopWorker := context.WithCancel(context.Background())
	workerDone := make(chan error, 1)
	go func() { workerDone <- worker.Run(workerCtx) }()
	defer stopWorker()
	waitFor(t, "worker registered with the LiveKit server", 10*time.Second, func() bool { return worker.Stats().Connected })

	// --- Explicit dispatch ------------------------------------------------
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	httpURL := strings.Replace(strings.Replace(lkURL, "ws://", "http://", 1), "wss://", "https://", 1)
	if _, err := lksdk.NewRoomServiceClient(httpURL, lkKey, lkSecret).CreateRoom(ctx, &livekit.CreateRoomRequest{Name: roomName}); err != nil {
		t.Fatalf("create room (is livekit-server running at %s?): %v", lkURL, err)
	}
	dispatch, err := lksdk.NewAgentDispatchServiceClient(httpURL, lkKey, lkSecret).CreateDispatch(ctx, &livekit.CreateAgentDispatchRequest{
		Room: roomName, AgentName: agentName, Metadata: `{"customer":"42"}`,
	})
	if err != nil {
		t.Fatalf("create dispatch: %v", err)
	}
	t.Logf("dispatch created: %s", dispatch.Id)
	waitFor(t, "job assigned to the worker", 10*time.Second, func() bool { return worker.Stats().ActiveJobs == 1 })

	// --- User side ----------------------------------------------------------
	meter := &levelMeter{}
	var agentParticipant atomic.Pointer[lksdk.RemoteParticipant]
	userRoom, err := lksdk.ConnectToRoom(lkURL, lksdk.ConnectInfo{
		APIKey: lkKey, APISecret: lkSecret, RoomName: roomName, ParticipantIdentity: "test-user",
	}, &lksdk.RoomCallback{ParticipantCallback: lksdk.ParticipantCallback{
		OnTrackSubscribed: func(track *webrtc.TrackRemote, _ *lksdk.RemoteTrackPublication, rp *lksdk.RemoteParticipant) {
			if track.Kind() != webrtc.RTPCodecTypeAudio {
				return
			}
			agentParticipant.Store(rp)
			if _, err := lkmedia.NewPCMRemoteTrack(track, meter); err != nil {
				t.Errorf("decode agent audio: %v", err)
			}
		},
	}})
	if err != nil {
		t.Fatalf("user join: %v", err)
	}
	defer userRoom.Disconnect()
	mic, err := lkmedia.NewPCMLocalTrack(48000, 1, protoLogger.GetLogger())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := userRoom.LocalParticipant.PublishTrack(mic, &lksdk.TrackPublicationOptions{Name: "mic", Source: livekit.TrackSource_MICROPHONE}); err != nil {
		t.Fatalf("publish mic: %v", err)
	}
	speak := func(d time.Duration) {
		// The track plays its queue in real time and sends silence when empty.
		_ = mic.WriteSample(media.PCM16Sample(audio.Tone(audio.Format{SampleRate: 48000, Channels: 1}, 300, d, 0.5).Data))
	}

	waitFor(t, "agent joined the room and published audio", 15*time.Second, func() bool { return agentParticipant.Load() != nil })
	agent := agentParticipant.Load()
	if agent.Kind() != lksdk.ParticipantAgent {
		t.Errorf("agent participant kind = %v, want agent", agent.Kind())
	}
	if !strings.HasPrefix(agent.Identity(), "agent-") {
		t.Errorf("agent identity = %q", agent.Identity())
	}
	waitFor(t, "agent state published as participant attribute", 10*time.Second, func() bool {
		return agent.Attributes()[roomio.AttributeAgentState] == string(agents.AgentListening)
	})
	if got := agent.Attributes()["lk.agent.name"]; got != agentName {
		t.Errorf("lk.agent.name = %q, want %q", got, agentName)
	}
	mu.Lock()
	if jobMeta != `{"customer":"42"}` {
		t.Errorf("dispatch metadata not delivered to the job: %q", jobMeta)
	}
	s := session
	mu.Unlock()
	time.Sleep(time.Second) // let the agent subscribe to the microphone

	// --- Turn 1: the user speaks, the agent answers ---------------------------
	speak(800 * time.Millisecond)
	waitFor(t, "user speech transcribed and committed", 10*time.Second, func() bool {
		m := s.History().Messages()
		return len(m) >= 1 && m[0].Role == llm.RoleUser && m[0].Content == "tell me something long"
	})
	waitFor(t, "agent speech audible to the user over WebRTC", 10*time.Second, meter.audible)
	waitFor(t, "agent state attribute is speaking", 5*time.Second, func() bool {
		return agent.Attributes()[roomio.AttributeAgentState] == string(agents.AgentSpeaking)
	})
	time.Sleep(2 * time.Second) // let it talk

	// --- Interruption ----------------------------------------------------------
	if !meter.audible() {
		t.Fatal("agent stopped talking before the interruption")
	}
	interruptAt := time.Now()
	speak(1200 * time.Millisecond)
	waitFor(t, "first speech interrupted", 5*time.Second, func() bool {
		f := finished()
		return len(f) >= 1 && f[0].Interrupted
	})
	t.Logf("interruption detected %v after the user started speaking", time.Since(interruptAt).Round(10*time.Millisecond))
	waitFor(t, "agent audio stopped for the user", 3*time.Second, func() bool { return !meter.audible() })

	first := finished()[0]
	msgs := s.History().Messages()
	if len(msgs) < 2 || msgs[1].Role != llm.RoleAssistant || !msgs[1].Interrupted {
		t.Fatalf("history after interruption: %+v", msgs)
	}
	if msgs[1].Content == "" || len(msgs[1].Content) >= len(longReply) || !strings.HasPrefix(longReply, msgs[1].Content) {
		t.Fatalf("interrupted message must be a strict prefix of the reply, got %q", msgs[1].Content)
	}
	t.Logf("played %v of audio; history keeps %d of %d characters", first.PlayedDuration.Round(100*time.Millisecond), len(msgs[1].Content), len(longReply))
	// What the user heard must match what the agent thinks it played.
	if heard := meter.total(); heard < first.PlayedDuration-1500*time.Millisecond || heard > first.PlayedDuration+1500*time.Millisecond {
		t.Errorf("user heard %v of audio but the agent accounted %v", heard, first.PlayedDuration)
	}

	// --- Turn 2: the reply to the interruption ---------------------------------
	waitFor(t, "second reply audible", 10*time.Second, meter.audible)
	waitFor(t, "second reply finished", 15*time.Second, func() bool {
		f := finished()
		return len(f) == 2 && !f[1].Interrupted && f[1].Err == nil
	})
	msgs = s.History().Messages()
	if len(msgs) != 4 || msgs[2].Content != "stop, I have another question" || msgs[3].Content != "Sure. Here is a short second answer." || msgs[3].Interrupted {
		t.Fatalf("final history: %+v", msgs)
	}

	// --- Shutdown ----------------------------------------------------------------
	userRoom.Disconnect()
	waitFor(t, "job ended after the user left", 10*time.Second, func() bool {
		return jobsDone.Load() == 1 && worker.Stats().ActiveJobs == 0
	})
	select {
	case <-s.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("session not closed")
	}
	stopWorker()
	select {
	case err := <-workerDone:
		if err != nil {
			t.Fatalf("worker.Run: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("worker did not drain")
	}
	if st := s.Stats(); st.DroppedEvents != 0 {
		t.Errorf("dropped events: %+v", st)
	}
}

// TestAutomaticDispatch checks that a worker without an agent name is
// dispatched to a new room as soon as a participant joins it.
func TestAutomaticDispatch(t *testing.T) {
	started := make(chan string, 1)
	worker, err := agents.NewWorker(agents.WorkerOptions{
		URL: lkURL, APIKey: lkKey, APISecret: agents.Secret(lkSecret),
		Entrypoint: func(ctx context.Context, job *agents.Job) error {
			if _, err := job.Connect(ctx, lksdk.NewRoomCallback()); err != nil {
				return err
			}
			started <- job.RoomName()
			<-ctx.Done()
			return nil
		},
		DrainTimeout: 100 * time.Millisecond,
		Logger:       slog.New(slog.NewTextHandler(os.Stderr, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- worker.Run(ctx) }()
	defer cancel()
	waitFor(t, "worker registered", 10*time.Second, func() bool { return worker.Stats().Connected })

	roomName := fmt.Sprintf("it-auto-%d", time.Now().UnixNano())
	user, err := lksdk.ConnectToRoom(lkURL, lksdk.ConnectInfo{APIKey: lkKey, APISecret: lkSecret, RoomName: roomName, ParticipantIdentity: "auto-user"}, lksdk.NewRoomCallback())
	if err != nil {
		t.Fatal(err)
	}
	defer user.Disconnect()
	select {
	case got := <-started:
		if got != roomName {
			t.Fatalf("job for room %q, want %q", got, roomName)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("no job received for the new room")
	}
	waitFor(t, "agent visible in the room", 10*time.Second, func() bool {
		for _, p := range user.GetRemoteParticipants() {
			if p.Kind() == lksdk.ParticipantAgent {
				return true
			}
		}
		return false
	})

	// Stopping the worker cancels the job after the drain timeout and
	// removes the agent from the room.
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("worker.Run: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("worker did not stop")
	}
	waitFor(t, "agent left the room", 10*time.Second, func() bool { return len(user.GetRemoteParticipants()) == 0 })
}
