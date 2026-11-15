package agents_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/agentstest"
	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers"
	"github.com/maktoufzied/livekit-agents-go/stt"
	"github.com/maktoufzied/livekit-agents-go/vad"
)

// harness wires a session to fake providers and records its events.
type harness struct {
	t   *testing.T
	stt *agentstest.STT
	llm *agentstest.LLM
	tts *agentstest.TTS
	in  *agentstest.Input
	out *agentstest.Output
	s   *agents.Session

	mu     sync.Mutex
	events []agents.Event
}

type hopt func(*agents.SessionOptions, *harness)

func withVAD() hopt {
	return func(o *agents.SessionOptions, _ *harness) {
		o.VAD = vad.NewEnergy(vad.EnergyOptions{MinSpeechDuration: 40 * time.Millisecond, MinSilenceDuration: 100 * time.Millisecond})
	}
}

func withOpts(fn func(*agents.SessionOptions)) hopt {
	return func(o *agents.SessionOptions, _ *harness) { fn(o) }
}

func newHarness(t *testing.T, responses []agentstest.Response, opts ...hopt) *harness {
	t.Helper()
	h := &harness{t: t, stt: agentstest.NewSTT(), llm: agentstest.NewLLM(responses...), tts: agentstest.NewTTS(), in: agentstest.NewInput(audio.Format{})}
	h.out = agentstest.NewOutput(h.tts.Format())
	h.out.Speed = 20 // 20x faster than real time: 20ms of wall clock per second of audio
	so := agents.SessionOptions{
		Agent:  &agents.Agent{Instructions: "You are a test agent."},
		STT:    h.stt,
		LLM:    h.llm,
		TTS:    h.tts,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		Turn: agents.TurnOptions{
			MinEndpointingDelay:     20 * time.Millisecond,
			MaxEndpointingDelay:     150 * time.Millisecond,
			MinInterruptionDuration: 60 * time.Millisecond,
		},
		Retry: agents.RetryOptions{Backoff: time.Millisecond},
	}
	for _, o := range opts {
		o(&so, h)
	}
	s, err := agents.NewSession(so)
	if err != nil {
		t.Fatal(err)
	}
	h.s = s
	s.On(func(ev agents.Event) {
		h.mu.Lock()
		h.events = append(h.events, ev)
		h.mu.Unlock()
	})
	return h
}

func (h *harness) start() *harness {
	h.t.Helper()
	if err := h.s.Start(context.Background(), agents.IO{Input: h.in, Output: h.out}); err != nil {
		h.t.Fatal(err)
	}
	h.t.Cleanup(func() { h.s.Close() })
	return h
}

func (h *harness) snapshot() []agents.Event {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]agents.Event(nil), h.events...)
}

func eventsOf[T agents.Event](h *harness) []T {
	var out []T
	for _, ev := range h.snapshot() {
		if e, ok := ev.(T); ok {
			out = append(out, e)
		}
	}
	return out
}

func metricsOf[T agents.Metric](h *harness) []T {
	var out []T
	for _, ev := range eventsOf[agents.MetricsCollected](h) {
		if m, ok := ev.Metric.(T); ok {
			out = append(out, m)
		}
	}
	return out
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(2 * time.Millisecond)
	}
}

// userSays simulates a complete user utterance through the STT provider.
func (h *harness) userSays(text string) {
	h.stt.Final(text)
	h.stt.Send(stt.Event{Type: stt.EndOfSpeech})
}

func (h *harness) waitSpeeches(n int) []agents.SpeechFinished {
	h.t.Helper()
	waitFor(h.t, fmt.Sprintf("%d finished speeches", n), func() bool { return len(eventsOf[agents.SpeechFinished](h)) >= n })
	return eventsOf[agents.SpeechFinished](h)
}

func roles(msgs []llm.Message) string {
	var r []string
	for _, m := range msgs {
		r = append(r, string(m.Role))
	}
	return strings.Join(r, ",")
}

const longReply = "This is the first sentence of a rather long reply. Here comes the second sentence, which is also long. " +
	"The third sentence keeps going for a while longer. And a fourth sentence follows right after that one. " +
	"Finally the fifth sentence closes this very long answer for good."

// ---------------------------------------------------------------------------

func TestTurnProducesSpokenReply(t *testing.T) {
	h := newHarness(t, []agentstest.Response{agentstest.Text("Hello there, friend. How can I help you today?")}).start()
	h.userSays("hi agent")
	fin := h.waitSpeeches(1)[0]
	if fin.Interrupted || fin.Err != nil {
		t.Fatalf("unexpected finish: %+v", fin)
	}

	msgs := h.s.History().Messages()
	if roles(msgs) != "user,assistant" || msgs[0].Content != "hi agent" || msgs[1].Content != "Hello there, friend. How can I help you today?" || msgs[1].Interrupted {
		t.Fatalf("history: %+v", msgs)
	}
	// Instructions reach the model as a leading system message.
	req := h.llm.Requests()[0]
	if req.Messages[0].Role != llm.RoleSystem || req.Messages[0].Content != "You are a test agent." || roles(req.Messages) != "system,user" {
		t.Fatalf("request: %+v", req.Messages)
	}
	// Sentences are synthesized separately, in order.
	if got := h.tts.Texts(); len(got) != 2 || got[0] != "Hello there, friend." || got[1] != "How can I help you today?" {
		t.Fatalf("tts texts: %q", got)
	}
	want := h.tts.Duration("Hello there, friend.") + h.tts.Duration("How can I help you today?")
	if h.out.Played() != want || fin.PlayedDuration != want {
		t.Fatalf("played %v (event %v), want %v", h.out.Played(), fin.PlayedDuration, want)
	}

	var states []agents.AgentState
	for _, e := range eventsOf[agents.AgentStateChanged](h) {
		states = append(states, e.New)
	}
	if fmt.Sprint(states) != "[listening thinking speaking listening]" {
		t.Fatalf("agent states: %v", states)
	}
	if tr := eventsOf[agents.AgentTranscript](h); len(tr) != 2 || tr[1].Text != "How can I help you today?" {
		t.Fatalf("agent transcripts: %+v", tr)
	}
	if len(metricsOf[agents.LLMMetrics](h)) != 1 || len(metricsOf[agents.TTSMetrics](h)) != 2 || len(metricsOf[agents.ResponseMetrics](h)) != 1 {
		t.Fatalf("missing metrics: %+v", eventsOf[agents.MetricsCollected](h))
	}
}

func TestInterruptionTruncatesHistoryAndStopsAudio(t *testing.T) {
	h := newHarness(t, []agentstest.Response{agentstest.Text(longReply), agentstest.Text("Sure, new answer.")}, withVAD()).start()
	h.userSays("tell me a story")
	waitFor(t, "agent speaking", func() bool { return h.s.AgentState() == agents.AgentSpeaking })
	time.Sleep(60 * time.Millisecond) // ~1.2s of audio at 20x

	// The user barges in: speech longer than MinInterruptionDuration.
	h.in.PushTone(200 * time.Millisecond)
	first := h.waitSpeeches(1)[0]
	if !first.Interrupted || first.Err != nil {
		t.Fatalf("expected interruption, got %+v", first)
	}
	interruptedAt := time.Now()
	framesAtInterrupt := len(h.out.Frames())

	msgs := h.s.History().Messages()
	last := msgs[len(msgs)-1]
	if last.Role != llm.RoleAssistant || !last.Interrupted {
		t.Fatalf("expected an interrupted assistant message, got %+v", msgs)
	}
	if last.Content == "" || len(last.Content) >= len(longReply) || !strings.HasPrefix(longReply, last.Content) {
		t.Fatalf("truncated content must be a strict prefix of the reply, got %q", last.Content)
	}
	// The recorded text must match what was played, within one sentence
	// boundary of tolerance.
	playedChars := int(first.PlayedDuration / (20 * time.Millisecond))
	if diff := playedChars - len(last.Content); diff < -25 || diff > 25 {
		t.Fatalf("history has %d chars but %d chars of audio were played", len(last.Content), playedChars)
	}
	if im := metricsOf[agents.InterruptionMetrics](h); len(im) != 1 || im[0].Latency > 250*time.Millisecond {
		t.Fatalf("interruption metrics: %+v", im)
	}

	// Nothing may be written after the interruption until a new speech.
	time.Sleep(80 * time.Millisecond)
	if n := len(h.out.Frames()); n != framesAtInterrupt {
		t.Fatalf("%d stale frames written after the interruption", n-framesAtInterrupt)
	}
	if h.llm.OpenStreams() != 0 || h.tts.OpenStreams() != 0 {
		t.Fatalf("provider streams leaked: llm=%d tts=%d", h.llm.OpenStreams(), h.tts.OpenStreams())
	}

	// Next turn: only audio of the new reply may be played.
	staleMarkers := int16(len(h.tts.Texts()))
	h.in.PushSilence(200 * time.Millisecond)
	h.userSays("actually something else")
	second := h.waitSpeeches(2)[1]
	if second.Interrupted || second.Err != nil {
		t.Fatalf("second speech: %+v", second)
	}
	for _, f := range h.out.Frames() {
		if f.At.After(interruptedAt) && f.Marker <= staleMarkers {
			t.Fatalf("frame of an old sentence (marker %d) played in the next turn", f.Marker)
		}
	}
	msgs = h.s.History().Messages()
	if roles(msgs) != "user,assistant,user,assistant" || msgs[3].Content != "Sure, new answer." {
		t.Fatalf("history: %+v", msgs)
	}
}

func TestInterruptionBeforeAnyAudioLeavesNoAssistantMessage(t *testing.T) {
	h := newHarness(t, []agentstest.Response{{Hang: true}, agentstest.Text("Second.")}).start()
	h.userSays("first question")
	waitFor(t, "llm request", func() bool { return len(h.llm.Requests()) == 1 })
	// Without a VAD, a new transcript interrupts the pending reply.
	h.userSays("second question")
	fins := h.waitSpeeches(2)
	if !fins[0].Interrupted || fins[0].PlayedDuration != 0 || fins[1].Interrupted {
		t.Fatalf("finishes: %+v", fins)
	}
	msgs := h.s.History().Messages()
	if roles(msgs) != "user,user,assistant" || msgs[2].Content != "Second." {
		t.Fatalf("history: %+v", msgs)
	}
}

func TestDisableInterruptions(t *testing.T) {
	h := newHarness(t, nil, withVAD()).start()
	sp, err := h.s.Say("This announcement must be heard completely by everyone.", agents.SayOptions{DisableInterruptions: true})
	if err != nil {
		t.Fatal(err)
	}
	waitFor(t, "speaking", func() bool { return h.s.AgentState() == agents.AgentSpeaking })
	h.in.PushTone(200 * time.Millisecond)
	if err := sp.Wait(context.Background()); err != nil || sp.Interrupted() {
		t.Fatalf("speech should complete: err=%v interrupted=%v", err, sp.Interrupted())
	}
	// An explicit Interrupt always wins.
	sp2, _ := h.s.Say("Another announcement that is going to be cut by the application.", agents.SayOptions{DisableInterruptions: true})
	waitFor(t, "speaking", func() bool { return h.s.AgentState() == agents.AgentSpeaking })
	h.s.Interrupt()
	if err := sp2.Wait(context.Background()); !errors.Is(err, agents.ErrInterrupted) {
		t.Fatalf("want ErrInterrupted, got %v", err)
	}
}

func TestVADEndpointingAndTurnDetector(t *testing.T) {
	var asked atomic.Int32
	det := detectorFunc(func(context.Context, agents.TurnInput) (float64, error) {
		asked.Add(1)
		return 0.1, nil // "the user is not done"
	})
	h := newHarness(t, []agentstest.Response{agentstest.Text("Done.")}, withVAD(), withOpts(func(o *agents.SessionOptions) {
		o.TurnDetector = det
	})).start()

	h.in.PushTone(200 * time.Millisecond)
	waitFor(t, "user speaking", func() bool { return h.s.UserState() == agents.UserSpeaking })
	h.in.PushSilence(300 * time.Millisecond)
	waitFor(t, "user listening", func() bool { return h.s.UserState() == agents.UserListening })
	if len(h.llm.Requests()) != 0 {
		t.Fatal("no reply may start before a transcript exists")
	}
	start := time.Now()
	h.stt.Final("so I was thinking")
	h.waitSpeeches(1)
	if asked.Load() != 1 {
		t.Fatalf("turn detector asked %d times", asked.Load())
	}
	// The low probability must have postponed the turn to MaxEndpointingDelay.
	if el := time.Since(start); el < 120*time.Millisecond {
		t.Fatalf("turn committed after %v, expected the max endpointing delay", el)
	}
	if len(metricsOf[agents.EndOfTurnMetrics](h)) != 1 || len(metricsOf[agents.TranscriptionMetrics](h)) != 1 {
		t.Fatal("missing turn metrics")
	}
}

// The turn detector hears what the user said in this turn: not the silence
// before it, and not the previous turn.
func TestTurnDetectorHearsTheTurn(t *testing.T) {
	inputs := make(chan agents.TurnInput, 4)
	det := detectorFunc(func(_ context.Context, in agents.TurnInput) (float64, error) {
		inputs <- in
		return 0.9, nil
	})
	h := newHarness(t, []agentstest.Response{agentstest.Text("One."), agentstest.Text("Two.")}, withVAD(), withOpts(func(o *agents.SessionOptions) {
		o.TurnDetector = det
	})).start()

	say := func(tone time.Duration, text string) agents.TurnInput {
		t.Helper()
		h.in.PushTone(tone)
		waitFor(t, "user speaking", func() bool { return h.s.UserState() == agents.UserSpeaking })
		h.in.PushSilence(300 * time.Millisecond)
		waitFor(t, "user listening", func() bool { return h.s.UserState() == agents.UserListening })
		h.stt.Final(text)
		select {
		case in := <-inputs:
			return in
		case <-time.After(5 * time.Second):
			t.Fatal("turn detector not asked")
			return agents.TurnInput{}
		}
	}
	check := func(in agents.TurnInput, tone time.Duration, text string) {
		t.Helper()
		if last := in.History[len(in.History)-1]; last.Role != llm.RoleUser || last.Content != text {
			t.Fatalf("history ends with %+v, want the pending user message %q", last, text)
		}
		if in.Audio.SampleRate != 16000 || in.Audio.Channels != 1 {
			t.Fatalf("audio format = %v", in.Audio.Format())
		}
		// The tone, 200 ms before it and 200 ms after it.
		got := in.Audio.Duration()
		if lo, hi := tone+300*time.Millisecond, tone+500*time.Millisecond; got < lo || got > hi {
			t.Fatalf("turn audio = %v, want between %v and %v", got, lo, hi)
		}
		if audio.RMS(in.Audio) == 0 {
			t.Fatal("turn audio is silent")
		}
	}

	h.in.PushSilence(2 * time.Second)
	check(say(200*time.Millisecond, "first"), 200*time.Millisecond, "first")
	h.waitSpeeches(1)
	h.in.PushSilence(time.Second)
	check(say(400*time.Millisecond, "second"), 400*time.Millisecond, "second")
	h.waitSpeeches(2)
}

type detectorFunc func(context.Context, agents.TurnInput) (float64, error)

func (f detectorFunc) EndOfTurnProbability(ctx context.Context, in agents.TurnInput) (float64, error) {
	return f(ctx, in)
}

func TestManualTurns(t *testing.T) {
	h := newHarness(t, []agentstest.Response{agentstest.Text("Manual reply.")}, withOpts(func(o *agents.SessionOptions) {
		o.Turn.Mode = agents.TurnManual
	})).start()
	h.userSays("part one")
	h.stt.Final("part two")
	waitFor(t, "transcripts", func() bool { return len(eventsOf[agents.UserTranscript](h)) == 2 })
	time.Sleep(50 * time.Millisecond)
	if len(h.llm.Requests()) != 0 {
		t.Fatal("manual mode must not commit turns on its own")
	}
	h.s.CommitUserTurn()
	h.waitSpeeches(1)
	if msgs := h.s.History().Messages(); msgs[0].Content != "part one part two" {
		t.Fatalf("history: %+v", msgs)
	}
}

// ---------------------------------------------------------------------------
// Tools

type lookupArgs struct {
	Key string `json:"key"`
}

func call(id, name, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Name: name, Arguments: json.RawMessage(args)}
}

func TestToolCallsRunConcurrentlyAndFeedTheModel(t *testing.T) {
	var running, peak, calls atomic.Int32
	lookup := llm.MustTool("lookup", "Look a key up", func(ctx context.Context, a lookupArgs) (any, error) {
		calls.Add(1)
		n := running.Add(1)
		defer running.Add(-1)
		for {
			p := peak.Load()
			if n <= p || peak.CompareAndSwap(p, n) {
				break
			}
		}
		time.Sleep(30 * time.Millisecond)
		return map[string]string{"value": strings.ToUpper(a.Key)}, nil
	})
	h := newHarness(t, []agentstest.Response{
		{Chunks: []llm.Chunk{{Text: "Let me check that. "}, {ToolCalls: []llm.ToolCall{call("c1", "lookup", `{"key":"a"}`), call("c2", "lookup", `{"key":"b"}`)}}}},
		agentstest.Text("The values are A and B."),
	}, withOpts(func(o *agents.SessionOptions) { o.Agent.Tools = []llm.Tool{lookup} })).start()

	h.userSays("look up a and b")
	if fin := h.waitSpeeches(1)[0]; fin.Err != nil || fin.Interrupted {
		t.Fatalf("finish: %+v", fin)
	}
	if peak.Load() != 2 || calls.Load() != 2 {
		t.Fatalf("peak concurrency %d, calls %d", peak.Load(), calls.Load())
	}
	msgs := h.s.History().Messages()
	if roles(msgs) != "user,assistant,tool,tool,assistant" {
		t.Fatalf("history roles: %s", roles(msgs))
	}
	if msgs[1].Content != "Let me check that. " || len(msgs[1].ToolCalls) != 2 ||
		msgs[2].ToolCallID != "c1" || msgs[2].Content != `{"value":"A"}` || msgs[3].ToolCallID != "c2" || msgs[3].Content != `{"value":"B"}` {
		t.Fatalf("history: %+v", msgs)
	}
	// The second request carries the tool results and the tool definitions.
	reqs := h.llm.Requests()
	if len(reqs) != 2 || roles(reqs[1].Messages) != "system,user,assistant,tool,tool" || len(reqs[1].Tools) != 1 {
		t.Fatalf("second request: %+v", reqs[1])
	}
	if len(eventsOf[agents.ToolCallStarted](h)) != 2 || len(eventsOf[agents.ToolCallFinished](h)) != 2 {
		t.Fatal("missing tool events")
	}
}

func TestToolFailuresAreReportedToTheModelAndNeverRetried(t *testing.T) {
	var sideEffects atomic.Int32
	tools := []llm.Tool{
		llm.MustTool("charge", "Charge a card", func(context.Context, lookupArgs) (any, error) {
			sideEffects.Add(1)
			return nil, errors.New("db password=hunter2 rejected") // internal detail
		}),
		llm.MustTool("friendly", "Fails nicely", func(context.Context, lookupArgs) (any, error) {
			return nil, llm.NewToolError("no such key")
		}),
		llm.MustTool("slow", "Too slow", func(ctx context.Context, _ lookupArgs) (any, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}, llm.WithTimeout(20*time.Millisecond)),
		llm.MustTool("stuck", "Ignores its context", func(context.Context, lookupArgs) (any, error) {
			time.Sleep(300 * time.Millisecond)
			return "late", nil
		}, llm.WithTimeout(20*time.Millisecond)),
		llm.MustTool("boom", "Panics", func(context.Context, lookupArgs) (any, error) { panic("kaboom") }),
	}
	h := newHarness(t, []agentstest.Response{
		agentstest.ToolCalls(
			call("1", "charge", `{"key":"x"}`),
			call("2", "friendly", `{"key":"x"}`),
			call("3", "slow", `{"key":"x"}`),
			call("4", "stuck", `{"key":"x"}`),
			call("5", "boom", `{"key":"x"}`),
			call("6", "charge", `{"key":42}`),
			call("7", "missing", `{}`),
		),
		agentstest.Text("Something went wrong."),
	}, withOpts(func(o *agents.SessionOptions) { o.Agent.Tools = tools })).start()

	h.userSays("do everything")
	h.waitSpeeches(1)
	msgs := h.s.History().Messages()
	if roles(msgs) != "user,assistant,tool,tool,tool,tool,tool,tool,tool,assistant" {
		t.Fatalf("roles: %s", roles(msgs))
	}
	want := []string{"internal error", "no such key", "timed out", "timed out", "internal error", "invalid tool arguments", "unknown tool"}
	for i, w := range want {
		m := msgs[2+i]
		if !m.IsError || !strings.Contains(m.Content, w) {
			t.Errorf("result %d: got %q (error=%v), want it to contain %q", i+1, m.Content, m.IsError, w)
		}
		if strings.Contains(m.Content, "hunter2") {
			t.Errorf("internal error detail leaked to the model: %q", m.Content)
		}
	}
	if sideEffects.Load() != 1 {
		t.Fatalf("tool with side effects ran %d times; tools must never be retried", sideEffects.Load())
	}
}

func TestToolStepsAreBounded(t *testing.T) {
	var n atomic.Int32
	tool := llm.MustTool("again", "Loop", func(context.Context, lookupArgs) (any, error) { n.Add(1); return "ok", nil })
	llmFake := agentstest.NewLLM()
	llmFake.Handler = func(req llm.Request) agentstest.Response {
		if req.ToolChoice == llm.ToolChoiceNone {
			return agentstest.Text("I give up.")
		}
		return agentstest.ToolCalls(call(fmt.Sprintf("c%d", len(req.Messages)), "again", `{"key":"k"}`))
	}
	h := newHarness(t, nil, withOpts(func(o *agents.SessionOptions) {
		o.Agent.Tools = []llm.Tool{tool}
		o.LLM = llmFake
		o.Tools.MaxSteps = 2
	})).start()
	h.userSays("loop forever")
	h.waitSpeeches(1)
	if n.Load() != 2 {
		t.Fatalf("tool ran %d times, want 2", n.Load())
	}
	msgs := h.s.History().Messages()
	if last := msgs[len(msgs)-1]; last.Content != "I give up." {
		t.Fatalf("last message: %+v", last)
	}
}

func TestInterruptionCancelsToolsButNotUninterruptibleOnes(t *testing.T) {
	cancelled := make(chan struct{})
	var finished atomic.Bool
	tools := []llm.Tool{
		llm.MustTool("search", "Cancellable", func(ctx context.Context, _ lookupArgs) (any, error) {
			<-ctx.Done()
			close(cancelled)
			return nil, ctx.Err()
		}),
		llm.MustTool("transfer", "Must complete", func(ctx context.Context, _ lookupArgs) (any, error) {
			select {
			case <-time.After(80 * time.Millisecond):
				finished.Store(true)
				return "transferred", nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}, llm.WithUninterruptible()),
	}
	h := newHarness(t, []agentstest.Response{
		agentstest.ToolCalls(call("1", "search", `{"key":"x"}`), call("2", "transfer", `{"key":"x"}`)),
		agentstest.Text("Never spoken."),
		agentstest.Text("Reply to the interruption."),
	}, withOpts(func(o *agents.SessionOptions) { o.Agent.Tools = tools })).start()

	h.userSays("search and transfer")
	waitFor(t, "tools started", func() bool { return len(eventsOf[agents.ToolCallStarted](h)) == 2 })
	h.s.Interrupt()
	<-cancelled
	fin := h.waitSpeeches(1)[0]
	if !fin.Interrupted {
		t.Fatalf("finish: %+v", fin)
	}
	if !finished.Load() {
		t.Fatal("uninterruptible tool was abandoned")
	}
	// History stays valid: every tool call has a result.
	msgs := h.s.History().Messages()
	if roles(msgs) != "user,assistant,tool,tool" || !msgs[1].Interrupted ||
		!msgs[2].IsError || !strings.Contains(msgs[2].Content, "cancelled") || msgs[3].IsError || msgs[3].Content != "transferred" {
		t.Fatalf("history: %+v", msgs)
	}
}

// ---------------------------------------------------------------------------
// Provider failures

func retryableErr(kind providers.Kind) error {
	return &providers.Error{Provider: "fake", Kind: kind, Op: "test", StatusCode: 503, Retryable: true, Err: errors.New("unavailable")}
}

func TestLLMRetriesOnlyBeforeFirstOutput(t *testing.T) {
	h := newHarness(t, []agentstest.Response{
		{OpenErr: retryableErr(providers.KindLLM)},
		{Err: retryableErr(providers.KindLLM)}, // fails before any chunk
		agentstest.Text("Third time lucky."),
	}).start()
	h.userSays("hello")
	if fin := h.waitSpeeches(1)[0]; fin.Err != nil {
		t.Fatalf("finish: %+v", fin)
	}
	if n := len(h.llm.Requests()); n != 3 {
		t.Fatalf("%d requests, want 3", n)
	}
	if m := metricsOf[agents.LLMMetrics](h); len(m) != 1 || m[0].Attempts != 3 {
		t.Fatalf("llm metrics: %+v", m)
	}

	// A failure after output started must not be retried: the user may
	// already have heard part of it.
	h2 := newHarness(t, []agentstest.Response{
		{Chunks: []llm.Chunk{{Text: "Partial answer. "}}, Err: retryableErr(providers.KindLLM)},
		agentstest.Text("unused"),
	}).start()
	h2.userSays("hello")
	fin := h2.waitSpeeches(1)[0]
	var pe *providers.Error
	if !errors.As(fin.Err, &pe) || len(h2.llm.Requests()) != 1 {
		t.Fatalf("finish: %+v, requests: %d", fin, len(h2.llm.Requests()))
	}
	errs := eventsOf[agents.ErrorEvent](h2)
	if len(errs) == 0 || errs[len(errs)-1].Source != "llm" || !errs[len(errs)-1].Recoverable {
		t.Fatalf("error events: %+v", errs)
	}
	if h2.s.AgentState() != agents.AgentListening {
		t.Fatalf("agent state after failure: %s", h2.s.AgentState())
	}
}

func TestLLMNonRetryableAndTimeout(t *testing.T) {
	h := newHarness(t, []agentstest.Response{
		{OpenErr: &providers.Error{Provider: "fake", Kind: providers.KindLLM, Op: "chat", StatusCode: 401, Err: errors.New("bad key")}},
	}).start()
	h.userSays("hello")
	if fin := h.waitSpeeches(1)[0]; fin.Err == nil || len(h.llm.Requests()) != 1 {
		t.Fatalf("finish: %+v requests=%d", fin, len(h.llm.Requests()))
	}

	h2 := newHarness(t, []agentstest.Response{{Hang: true}, {Hang: true}}, withOpts(func(o *agents.SessionOptions) {
		o.Timeouts.LLMFirstToken = 30 * time.Millisecond
		o.Retry.MaxRetries = 1
	})).start()
	h2.userSays("hello")
	fin := h2.waitSpeeches(1)[0]
	var te *agents.TimeoutError
	if !errors.As(fin.Err, &te) || len(h2.llm.Requests()) != 2 || h2.llm.OpenStreams() != 0 {
		t.Fatalf("finish: %+v requests=%d open=%d", fin, len(h2.llm.Requests()), h2.llm.OpenStreams())
	}
}

func TestTTSFailures(t *testing.T) {
	h := newHarness(t, []agentstest.Response{agentstest.Text("A sentence to speak."), agentstest.Text("Another sentence here.")}).start()
	h.tts.FailNext(retryableErr(providers.KindTTS))
	h.userSays("one")
	if fin := h.waitSpeeches(1)[0]; fin.Err != nil {
		t.Fatalf("retryable tts failure should recover: %+v", fin)
	}
	h.tts.FailNext(errors.New("permanent"))
	h.userSays("two")
	fin := h.waitSpeeches(2)[1]
	if fin.Err == nil || fin.PlayedDuration != 0 {
		t.Fatalf("finish: %+v", fin)
	}
	// Nothing was played, so nothing is recorded for the assistant.
	if msgs := h.s.History().Messages(); roles(msgs) != "user,assistant,user" {
		t.Fatalf("history: %s", roles(msgs))
	}
	if h.s.AgentState() != agents.AgentListening {
		t.Fatal("session must stay usable after a speech failure")
	}
}

func TestSTTReconnectsThenGivesUp(t *testing.T) {
	h := newHarness(t, []agentstest.Response{agentstest.Text("Still here.")}).start()
	waitFor(t, "stt stream", func() bool { return h.stt.Opens() == 1 })
	h.stt.Fail(retryableErr(providers.KindSTT))
	waitFor(t, "stt reconnect", func() bool { return h.stt.Opens() == 2 })
	h.userSays("are you there")
	if fin := h.waitSpeeches(1)[0]; fin.Err != nil {
		t.Fatalf("finish: %+v", fin)
	}

	// Persistent failure: bounded reconnects, then the session closes.
	h2 := newHarness(t, nil, withOpts(func(o *agents.SessionOptions) { o.Retry.STTMaxReconnects = 2 }))
	h2.stt.OpenErrs = []error{retryableErr(providers.KindSTT), retryableErr(providers.KindSTT), retryableErr(providers.KindSTT), retryableErr(providers.KindSTT)}
	h2.start()
	select {
	case <-h2.s.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("session did not close")
	}
	if h2.stt.Opens() != 3 {
		t.Fatalf("stt opened %d times, want 3 (1 + 2 reconnects)", h2.stt.Opens())
	}
	closed := eventsOf[agents.SessionClosed](h2)
	if len(closed) != 1 || closed[0].Reason != agents.CloseError || h2.s.Err() == nil {
		t.Fatalf("closed: %+v err=%v", closed, h2.s.Err())
	}
	if _, err := h2.s.Say("x", agents.SayOptions{}); !errors.Is(err, agents.ErrSessionClosed) {
		t.Fatalf("Say after close: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Backpressure and cleanup

func TestBackpressureBoundsBuffering(t *testing.T) {
	// 40 sentences, each 0.6s of audio. Playback runs at 20x, so the whole
	// reply takes ~1.2s of wall clock; the LLM and TTS are instantaneous.
	var sb strings.Builder
	for i := 0; i < 40; i++ {
		fmt.Fprintf(&sb, "Sentence number %02d is right here. ", i)
	}
	h := newHarness(t, []agentstest.Response{agentstest.Text(sb.String())}, withOpts(func(o *agents.SessionOptions) {
		o.Buffers.Sentences = 2
		o.Buffers.SynthesisLookahead = 1
	})).start()
	h.userSays("go")
	waitFor(t, "speaking", func() bool { return h.s.AgentState() == agents.AgentSpeaking })
	time.Sleep(100 * time.Millisecond)

	// After ~2s of audio (3-4 sentences played) synthesis may only be a
	// few sentences ahead: 1 playing + lookahead 1 + 1 in synthesis.
	played := int(h.out.Played() / h.tts.Duration("Sentence number 00 is right here."))
	if ahead := len(h.tts.Texts()) - played; ahead > 4 {
		t.Fatalf("synthesized %d sentences ahead of playback; buffering is not bounded", ahead)
	}
	h.s.Interrupt()
	h.waitSpeeches(1)

	// A slow STT provider must not block the audio input: frames are dropped.
	h2 := newHarness(t, nil, withOpts(func(o *agents.SessionOptions) { o.Buffers.STTFrames = 5 }))
	h2.stt.PushDelay = 20 * time.Millisecond
	h2.start()
	h2.in.PushSilence(2 * time.Second) // 100 frames, instantly
	waitFor(t, "dropped frames", func() bool { return h2.s.Stats().DroppedInputFrames >= 80 })
}

func TestCloseReleasesEverything(t *testing.T) {
	base := runtime.NumGoroutine()
	for i := 0; i < 5; i++ {
		h := newHarness(t, []agentstest.Response{agentstest.Text(longReply)}, withVAD())
		if err := h.s.Start(context.Background(), agents.IO{Input: h.in, Output: h.out}); err != nil {
			t.Fatal(err)
		}
		h.in.PushSilence(100 * time.Millisecond)
		h.userSays("talk to me")
		waitFor(t, "speaking", func() bool { return h.s.AgentState() == agents.AgentSpeaking })
		sp, _ := h.s.Say("queued behind", agents.SayOptions{})

		// Close in the middle of a speech, with another one queued.
		if err := h.s.Close(); err != nil {
			t.Fatal(err)
		}
		if err := h.s.Close(); err != nil { // idempotent
			t.Fatal(err)
		}
		if err := sp.Wait(context.Background()); !errors.Is(err, agents.ErrSessionClosed) {
			t.Fatalf("queued speech: %v", err)
		}
		if h.stt.OpenStreams() != 0 || h.llm.OpenStreams() != 0 || h.tts.OpenStreams() != 0 {
			t.Fatalf("streams leaked: stt=%d llm=%d tts=%d", h.stt.OpenStreams(), h.llm.OpenStreams(), h.tts.OpenStreams())
		}
		n := len(h.out.Frames())
		time.Sleep(20 * time.Millisecond)
		if len(h.out.Frames()) != n {
			t.Fatal("audio written after Close")
		}
		ev := h.snapshot()
		if c, ok := ev[len(ev)-1].(agents.SessionClosed); !ok || c.Reason != agents.CloseRequested {
			t.Fatalf("last event: %#v", ev[len(ev)-1])
		}
		if _, err := h.s.GenerateReply(agents.ReplyOptions{}); !errors.Is(err, agents.ErrSessionClosed) {
			t.Fatalf("GenerateReply after close: %v", err)
		}
	}
	waitFor(t, "goroutines to exit", func() bool { return runtime.NumGoroutine() <= base+1 })
}

func TestContextCancellationClosesSession(t *testing.T) {
	h := newHarness(t, []agentstest.Response{agentstest.Text(longReply)})
	ctx, cancel := context.WithCancel(context.Background())
	if err := h.s.Start(ctx, agents.IO{Input: h.in, Output: h.out}); err != nil {
		t.Fatal(err)
	}
	h.userSays("hi")
	waitFor(t, "speaking", func() bool { return h.s.AgentState() == agents.AgentSpeaking })
	cancel()
	select {
	case <-h.s.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("session did not close on context cancellation")
	}
	if c := eventsOf[agents.SessionClosed](h); len(c) != 1 || c[0].Reason != agents.CloseContextDone {
		t.Fatalf("closed: %+v", c)
	}
}

func TestInputEOFKeepsSessionUsableAndDrain(t *testing.T) {
	h := newHarness(t, nil).start()
	h.in.Close()
	waitFor(t, "stt stream closed", func() bool { return h.stt.Opens() == 1 && h.stt.OpenStreams() == 0 })
	sp, err := h.s.GenerateReply(agents.ReplyOptions{UserInput: "typed question", Instructions: "Be brief."})
	if err != nil {
		t.Fatal(err)
	}
	if err := h.s.Drain(context.Background()); err != nil {
		t.Fatal(err)
	}
	if sp.Wait(context.Background()) != nil {
		t.Fatal("Drain must let the speech finish")
	}
	req := h.llm.Requests()[0]
	if roles(req.Messages) != "system,user,system" || req.Messages[2].Content != "Be brief." {
		t.Fatalf("request: %+v", req.Messages)
	}
}

func TestOptionValidation(t *testing.T) {
	tts := agentstest.NewTTS()
	if _, err := agents.NewSession(agents.SessionOptions{TTS: tts}); err == nil {
		t.Error("missing LLM accepted")
	}
	if _, err := agents.NewSession(agents.SessionOptions{LLM: agentstest.NewLLM()}); err == nil {
		t.Error("missing TTS accepted")
	}
	dup := llm.MustTool("t", "", func(context.Context, lookupArgs) (any, error) { return nil, nil })
	if _, err := agents.NewSession(agents.SessionOptions{LLM: agentstest.NewLLM(), TTS: tts, Agent: &agents.Agent{Tools: []llm.Tool{dup, dup}}}); err == nil {
		t.Error("duplicate tool accepted")
	}
	s, err := agents.NewSession(agents.SessionOptions{LLM: agentstest.NewLLM(), TTS: tts})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err := s.Say("hi", agents.SayOptions{}); !errors.Is(err, agents.ErrNotStarted) {
		t.Errorf("Say before Start: %v", err)
	}
	wrong := agentstest.NewOutput(audio.Format{SampleRate: 8000, Channels: 1})
	var fe *agents.FormatError
	if err := s.Start(context.Background(), agents.IO{Output: wrong}); !errors.As(err, &fe) {
		t.Errorf("format mismatch: %v", err)
	}
	if err := s.Start(context.Background(), agents.IO{Input: agentstest.NewInput(audio.Format{}), Output: agentstest.NewOutput(tts.Format())}); err == nil {
		t.Error("input without STT accepted")
	}
}

// A user turn that interrupts a running reply must be recorded after the
// part of that reply the user heard, never before it.
func TestUserTurnIsRecordedAfterTheInterruptedReply(t *testing.T) {
	h := newHarness(t, []agentstest.Response{agentstest.Text(longReply), agentstest.Text("Stopping.")}, withOpts(func(o *agents.SessionOptions) {
		// No VAD and a high word threshold: the barge-in path does not
		// fire, so the interruption comes from the committed user turn.
		o.Turn.MinInterruptionWords = 5
	})).start()
	h.userSays("hello there")
	waitFor(t, "agent speaking", func() bool { return h.s.AgentState() == agents.AgentSpeaking })
	time.Sleep(40 * time.Millisecond)
	h.userSays("stop")
	fins := h.waitSpeeches(2)
	if !fins[0].Interrupted || fins[1].Interrupted {
		t.Fatalf("finishes: %+v", fins)
	}
	msgs := h.s.History().Messages()
	if roles(msgs) != "user,assistant,user,assistant" {
		t.Fatalf("history order: %s", roles(msgs))
	}
	if !msgs[1].Interrupted || !strings.HasPrefix(longReply, msgs[1].Content) || msgs[2].Content != "stop" || msgs[3].Content != "Stopping." {
		t.Fatalf("history: %+v", msgs)
	}
	// The model must see the same order, ending with the new user turn.
	last := h.llm.Requests()[1].Messages
	if roles(last) != "system,user,assistant,user" {
		t.Fatalf("second request: %s", roles(last))
	}
}

// With sequential tools, a call that is cancelled before it starts must
// not run at all.
func TestCancelledToolIsNeverStarted(t *testing.T) {
	var secondRan atomic.Bool
	firstStarted := make(chan struct{})
	tools := []llm.Tool{
		llm.MustTool("first", "Blocks", func(ctx context.Context, _ lookupArgs) (any, error) {
			close(firstStarted)
			<-ctx.Done()
			return nil, ctx.Err()
		}),
		llm.MustTool("second", "Has a side effect", func(context.Context, lookupArgs) (any, error) {
			secondRan.Store(true)
			return "done", nil
		}),
	}
	h := newHarness(t, []agentstest.Response{
		agentstest.ToolCalls(call("1", "first", `{"key":"x"}`), call("2", "second", `{"key":"x"}`)),
	}, withOpts(func(o *agents.SessionOptions) {
		o.Agent.Tools = tools
		o.Tools.Sequential = true
	})).start()
	h.userSays("do both")
	<-firstStarted
	h.s.Interrupt()
	h.waitSpeeches(1)
	time.Sleep(20 * time.Millisecond)
	if secondRan.Load() {
		t.Fatal("a tool ran after its call was cancelled")
	}
	msgs := h.s.History().Messages()
	if roles(msgs) != "user,assistant,tool,tool" || !msgs[2].IsError || !msgs[3].IsError || !strings.Contains(msgs[3].Content, "cancelled") {
		t.Fatalf("history: %+v", msgs)
	}
}

// A stuck handler may cost metrics and interim transcripts, never
// lifecycle or history events.
func TestImportantEventsSurviveASlowHandler(t *testing.T) {
	block := make(chan struct{})
	var (
		mu   sync.Mutex
		seen []agents.Event
	)
	h := newHarness(t, []agentstest.Response{agentstest.Text("One. Two. Three.")}, withOpts(func(o *agents.SessionOptions) {
		o.Buffers.Events = 4
		o.Turn.Mode = agents.TurnManual
	}))
	h.s.On(func(ev agents.Event) {
		<-block
		mu.Lock()
		seen = append(seen, ev)
		mu.Unlock()
	})
	h.start()
	var unblock sync.Once
	release := func() { unblock.Do(func() { close(block) }) }
	t.Cleanup(release) // runs before the session is closed, even on failure
	for i := 0; i < 200; i++ {
		h.stt.Interim(fmt.Sprintf("interim %d", i))
	}
	h.stt.Final("the real thing")
	waitFor(t, "interim transcripts dropped", func() bool { return h.s.Stats().DroppedEvents > 100 })
	// The final transcript may still be on its way to the session: commit
	// until it has been picked up (committing an empty turn is a no-op).
	waitFor(t, "reply played", func() bool {
		h.s.CommitUserTurn()
		return h.out.Played() > 0 && len(h.s.History().Messages()) == 2
	})
	h.s.Shutdown()
	release()
	<-h.s.Done()

	mu.Lock()
	defer mu.Unlock()
	var finals, items, finished int
	for _, ev := range seen {
		switch e := ev.(type) {
		case agents.UserTranscript:
			if e.Final {
				finals++
			}
		case agents.ConversationItemAdded:
			items++
		case agents.SpeechFinished:
			finished++
		}
	}
	if _, ok := seen[len(seen)-1].(agents.SessionClosed); !ok || finals != 1 || items != 2 || finished != 1 {
		t.Fatalf("lost important events: finals=%d items=%d finished=%d last=%T", finals, items, finished, seen[len(seen)-1])
	}
}
