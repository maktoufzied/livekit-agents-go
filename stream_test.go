package agents_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/agentstest"
	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers"
)

// withStreamTTS gives the session a TTS that takes the text as it comes.
func withStreamTTS(st **agentstest.StreamTTS) hopt {
	return func(o *agents.SessionOptions, h *harness) {
		*st = &agentstest.StreamTTS{TTS: h.tts}
		o.TTS = *st
	}
}

// With a streaming TTS the text of a reply goes to one stream, chunk by
// chunk, and its transcript is still reported sentence by sentence.
func TestStreamingTTSSpeaksAReply(t *testing.T) {
	var st *agentstest.StreamTTS
	h := newHarness(t, []agentstest.Response{agentstest.Text(longReply)}, withStreamTTS(&st)).start()
	if st.Prewarms() != 1 {
		t.Fatalf("Prewarm called %d times at start", st.Prewarms())
	}
	h.userSays("tell me a story")
	fin := h.waitSpeeches(1)[0]
	if fin.Interrupted || fin.Err != nil {
		t.Fatalf("speech: %+v", fin)
	}
	if texts := h.tts.Texts(); len(texts) != 1 || texts[0] != longReply {
		t.Fatalf("text sent to the TTS: %q", texts)
	}
	if n := st.Streams(); n != 1 {
		t.Fatalf("%d streams opened, want 1", n)
	}
	want := time.Duration(len(longReply)) * 20 * time.Millisecond
	if d := fin.PlayedDuration - want; d < -20*time.Millisecond || d > 20*time.Millisecond {
		t.Fatalf("played %v, want %v", fin.PlayedDuration, want)
	}
	m := metricsOf[agents.TTSMetrics](h)
	if len(m) != 1 || m[0].Characters != len(longReply) || m[0].Attempts != 1 || m[0].AudioDuration != fin.PlayedDuration {
		t.Fatalf("tts metrics: %+v", m)
	}

	var said []string
	for _, e := range eventsOf[agents.AgentTranscript](h) {
		said = append(said, e.Text)
	}
	if len(said) != 5 || strings.Join(said, " ") != longReply {
		t.Fatalf("transcripts: %q", said)
	}
	msgs := h.s.History().Messages()
	if roles(msgs) != "user,assistant" || msgs[1].Content != longReply || msgs[1].Interrupted {
		t.Fatalf("history: %+v", msgs)
	}
	if h.tts.OpenStreams() != 0 {
		t.Fatalf("%d streams left open", h.tts.OpenStreams())
	}
}

// The audio starts before the model has finished its first sentence.
func TestStreamingTTSStartsBeforeTheSentenceEnds(t *testing.T) {
	var st *agentstest.StreamTTS
	h := newHarness(t, []agentstest.Response{{
		Chunks: []llm.Chunk{{Text: "This sentence has only begun "}},
		Hang:   true,
	}}, withStreamTTS(&st)).start()
	h.userSays("go on")
	waitFor(t, "audio before the end of the sentence", func() bool { return len(h.out.Frames()) > 0 })
	if h.s.AgentState() != agents.AgentSpeaking {
		t.Fatalf("agent state = %v", h.s.AgentState())
	}
}

func TestStreamingTTSSay(t *testing.T) {
	var st *agentstest.StreamTTS
	h := newHarness(t, nil, withStreamTTS(&st)).start()
	sp, err := h.s.Say("Hello there. How can I help?", agents.SayOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := sp.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if texts := h.tts.Texts(); len(texts) != 1 || texts[0] != "Hello there. How can I help?" {
		t.Fatalf("text sent to the TTS: %q", texts)
	}
	if msgs := h.s.History().Messages(); roles(msgs) != "assistant" || msgs[0].Content != "Hello there. How can I help?" {
		t.Fatalf("history: %+v", msgs)
	}
}

// An interruption keeps in the history what was heard, as with a plain TTS.
func TestStreamingTTSInterruptionTruncatesHistory(t *testing.T) {
	var st *agentstest.StreamTTS
	h := newHarness(t, []agentstest.Response{agentstest.Text(longReply), agentstest.Text("New answer.")}, withStreamTTS(&st)).start()
	h.userSays("tell me a story")
	waitFor(t, "agent speaking", func() bool { return h.s.AgentState() == agents.AgentSpeaking })
	time.Sleep(80 * time.Millisecond) // ~1.6 s of audio at 20x

	h.s.Interrupt()
	first := h.waitSpeeches(1)[0]
	if !first.Interrupted || first.Err != nil {
		t.Fatalf("speech: %+v", first)
	}
	msgs := h.s.History().Messages()
	last := msgs[len(msgs)-1]
	if last.Role != llm.RoleAssistant || !last.Interrupted {
		t.Fatalf("history: %+v", msgs)
	}
	if last.Content == "" || len(last.Content) >= len(longReply) || !strings.HasPrefix(longReply, last.Content) {
		t.Fatalf("truncated content must be a strict prefix of the reply, got %q", last.Content)
	}
	playedChars := int(first.PlayedDuration / (20 * time.Millisecond))
	if diff := playedChars - len(last.Content); diff < -25 || diff > 25 {
		t.Fatalf("history has %d chars but %d chars of audio were played", len(last.Content), playedChars)
	}
	// Only what was heard is reported as said.
	var said int
	for _, e := range eventsOf[agents.AgentTranscript](h) {
		said += len(e.Text)
	}
	if said >= len(longReply) {
		t.Fatal("the whole reply was reported as said")
	}
	if h.tts.OpenStreams() != 0 || h.llm.OpenStreams() != 0 {
		t.Fatalf("streams leaked: tts=%d llm=%d", h.tts.OpenStreams(), h.llm.OpenStreams())
	}

	h.userSays("something else")
	second := h.waitSpeeches(2)[1]
	if second.Interrupted || second.Err != nil {
		t.Fatalf("second speech: %+v", second)
	}
}

// A hold and a resume lose nothing and repeat nothing.
func TestStreamingTTSFalseInterruption(t *testing.T) {
	var st *agentstest.StreamTTS
	h := newHarness(t, []agentstest.Response{agentstest.Text(longReply)}, withVAD(), withStreamTTS(&st), withOpts(func(o *agents.SessionOptions) {
		o.Turn.FalseInterruptionTimeout = 150 * time.Millisecond
	})).start()
	h.userSays("tell me a story")
	waitFor(t, "agent speaking", func() bool { return h.s.AgentState() == agents.AgentSpeaking })
	time.Sleep(40 * time.Millisecond)
	h.in.PushTone(200 * time.Millisecond)
	waitFor(t, "speech on hold", func() bool { return h.s.AgentState() == agents.AgentListening })
	h.in.PushSilence(300 * time.Millisecond)
	fin := h.waitSpeeches(1)[0]
	if fin.Interrupted || fin.Err != nil {
		t.Fatalf("speech: %+v", fin)
	}
	want := time.Duration(len(longReply)) * 20 * time.Millisecond
	if d := fin.PlayedDuration - want; d < -20*time.Millisecond || d > 20*time.Millisecond {
		t.Fatalf("played %v, want %v", fin.PlayedDuration, want)
	}
	if len(eventsOf[agents.FalseInterruption](h)) != 1 {
		t.Fatal("no false interruption event")
	}
}

// Text spoken before a tool call and the answer after it use two streams,
// and the history keeps their order.
func TestStreamingTTSWithTools(t *testing.T) {
	lookup := llm.MustTool("lookup", "Look a key up", func(_ context.Context, a lookupArgs) (any, error) {
		return map[string]string{"value": strings.ToUpper(a.Key)}, nil
	})
	var st *agentstest.StreamTTS
	h := newHarness(t, []agentstest.Response{
		{Chunks: []llm.Chunk{{Text: "Let me check that. "}, {ToolCalls: []llm.ToolCall{call("c1", "lookup", `{"key":"a"}`)}}}},
		agentstest.Text("The value is A."),
	}, withStreamTTS(&st), withOpts(func(o *agents.SessionOptions) { o.Agent.Tools = []llm.Tool{lookup} })).start()
	h.userSays("look up a")
	if fin := h.waitSpeeches(1)[0]; fin.Err != nil || fin.Interrupted {
		t.Fatalf("speech: %+v", fin)
	}
	if texts := h.tts.Texts(); len(texts) != 2 || texts[0] != "Let me check that. " || texts[1] != "The value is A." {
		t.Fatalf("text sent to the TTS: %q", texts)
	}
	msgs := h.s.History().Messages()
	if roles(msgs) != "user,assistant,tool,assistant" || msgs[1].Content != "Let me check that. " || msgs[3].Content != "The value is A." {
		t.Fatalf("history: %+v", msgs)
	}
	// Frames of the first stream all come before those of the second.
	var last int16
	for _, f := range h.out.Frames() {
		if f.Marker < last {
			t.Fatal("audio of the two steps is interleaved")
		}
		last = f.Marker
	}
}

func TestStreamingTTSFailures(t *testing.T) {
	t.Run("open fails once", func(t *testing.T) {
		var st *agentstest.StreamTTS
		h := newHarness(t, []agentstest.Response{agentstest.Text("A sentence to speak.")}, withStreamTTS(&st))
		st.FailOpens = []error{retryableErr(providers.KindTTS)}
		h.start()
		h.userSays("one")
		if fin := h.waitSpeeches(1)[0]; fin.Err != nil {
			t.Fatalf("a retryable failure should recover: %+v", fin)
		}
	})
	t.Run("stream fails before any audio", func(t *testing.T) {
		var st *agentstest.StreamTTS
		h := newHarness(t, []agentstest.Response{agentstest.Text("A sentence to speak. And one more.")}, withStreamTTS(&st))
		st.FailRecvs = []error{retryableErr(providers.KindTTS)}
		h.start()
		h.userSays("one")
		fin := h.waitSpeeches(1)[0]
		if fin.Err != nil {
			t.Fatalf("a retryable failure should recover: %+v", fin)
		}
		// The second stream was given the whole text again.
		texts := h.tts.Texts()
		if len(texts) != 2 || texts[1] != "A sentence to speak. And one more." {
			t.Fatalf("text sent to the TTS: %q", texts)
		}
		if m := metricsOf[agents.TTSMetrics](h); len(m) != 1 || m[0].Attempts != 2 {
			t.Fatalf("tts metrics: %+v", m)
		}
		if msgs := h.s.History().Messages(); msgs[len(msgs)-1].Content != "A sentence to speak. And one more." {
			t.Fatalf("history: %+v", msgs)
		}
	})
	t.Run("permanent failure", func(t *testing.T) {
		var st *agentstest.StreamTTS
		h := newHarness(t, []agentstest.Response{agentstest.Text("A sentence to speak.")}, withStreamTTS(&st))
		st.FailOpens = []error{errors.New("permanent")}
		h.start()
		h.userSays("one")
		fin := h.waitSpeeches(1)[0]
		if fin.Err == nil || fin.PlayedDuration != 0 {
			t.Fatalf("speech: %+v", fin)
		}
		if msgs := h.s.History().Messages(); roles(msgs) != "user" {
			t.Fatalf("history: %s", roles(msgs))
		}
		if h.s.AgentState() != agents.AgentListening {
			t.Fatal("session must stay usable after a speech failure")
		}
	})
	t.Run("no audio in time", func(t *testing.T) {
		var st *agentstest.StreamTTS
		h := newHarness(t, []agentstest.Response{agentstest.Text("A sentence to speak.")}, withStreamTTS(&st), withOpts(func(o *agents.SessionOptions) {
			o.Timeouts.TTSFirstAudio = 50 * time.Millisecond
			o.Retry.MaxRetries = -1
		}))
		h.tts.FirstFrameDelay = time.Second
		h.start()
		h.userSays("one")
		fin := h.waitSpeeches(1)[0]
		var te *agents.TimeoutError
		if !errors.As(fin.Err, &te) {
			t.Fatalf("speech error = %v, want a timeout", fin.Err)
		}
		if h.tts.OpenStreams() != 0 {
			t.Fatalf("%d streams left open", h.tts.OpenStreams())
		}
	})
}

func TestStreamingTTSCloseReleasesTheStream(t *testing.T) {
	var st *agentstest.StreamTTS
	h := newHarness(t, []agentstest.Response{agentstest.Text(longReply)}, withStreamTTS(&st)).start()
	h.userSays("talk to me")
	waitFor(t, "speaking", func() bool { return h.s.AgentState() == agents.AgentSpeaking })
	if err := h.s.Close(); err != nil {
		t.Fatal(err)
	}
	if h.tts.OpenStreams() != 0 || h.llm.OpenStreams() != 0 {
		t.Fatalf("streams leaked: tts=%d llm=%d", h.tts.OpenStreams(), h.llm.OpenStreams())
	}
}
