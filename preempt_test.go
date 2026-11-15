package agents_test

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/agentstest"
	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/stt"
)

// In these tests there is no VAD: a turn ends on the recognizer's
// end-of-speech event, or 400 ms after the last final transcript.
func slowEndpointing(o *agents.SessionOptions) {
	o.Turn.MinEndpointingDelay = 400 * time.Millisecond
	o.Turn.MaxEndpointingDelay = 400 * time.Millisecond
}

// The model request is sent as soon as the user stops, and the reply that
// starts when the turn ends uses it instead of sending another one.
func TestPreemptiveGenerationIsUsedForTheReply(t *testing.T) {
	h := newHarness(t, []agentstest.Response{agentstest.Text("Here is the answer.")}, withOpts(slowEndpointing)).start()

	sent := time.Now()
	h.stt.Final("what time is it")
	waitFor(t, "the request sent ahead", func() bool { return len(h.llm.Requests()) == 1 })
	if early := time.Since(sent); early > 200*time.Millisecond {
		t.Fatalf("request sent %v after the transcript, want it right away", early)
	}
	if n := len(eventsOf[agents.SpeechStarted](h)); n != 0 {
		t.Fatalf("%d speeches started before the turn ended", n)
	}
	if n := h.s.History().Len(); n != 0 {
		t.Fatalf("%d messages recorded before the turn ended", n)
	}
	req := h.llm.Requests()[0]
	if last := req.Messages[len(req.Messages)-1]; last.Role != llm.RoleUser || last.Content != "what time is it" {
		t.Fatalf("request ends with %+v", last)
	}

	fin := h.waitSpeeches(1)[0]
	if fin.Interrupted || fin.Err != nil {
		t.Fatalf("speech: %+v", fin)
	}
	if n := len(h.llm.Requests()); n != 1 {
		t.Fatalf("%d model requests, want the one sent ahead", n)
	}
	m := metricsOf[agents.LLMMetrics](h)
	if len(m) != 1 || m[0].Attempts != 1 || m[0].Preemptive < 300*time.Millisecond {
		t.Fatalf("llm metrics: %+v", m)
	}
	msgs := h.s.History().Messages()
	if roles(msgs) != "user,assistant" || msgs[0].Content != "what time is it" || msgs[1].Content != "Here is the answer." {
		t.Fatalf("history: %+v", msgs)
	}
	if h.llm.OpenStreams() != 0 {
		t.Fatalf("%d model streams left open", h.llm.OpenStreams())
	}
}

// When the user goes on, the request sent ahead is thrown away: its tool
// call never runs, its stream is closed and the reply answers everything
// the user said.
func TestPreemptiveGenerationIsDiscardedWhenTheUserGoesOn(t *testing.T) {
	var ran atomic.Bool
	tool := llm.MustTool("book", "Has a side effect", func(context.Context, lookupArgs) (any, error) {
		ran.Store(true)
		return "booked", nil
	})
	h := newHarness(t, []agentstest.Response{
		agentstest.ToolCalls(call("1", "book", `{"key":"x"}`)), // answers "book a table"
		agentstest.Text("For how many people?"),                // answers the whole turn
	}, withOpts(slowEndpointing), withOpts(func(o *agents.SessionOptions) { o.Agent.Tools = []llm.Tool{tool} })).start()

	h.stt.Final("book a table")
	waitFor(t, "first request", func() bool { return len(h.llm.Requests()) == 1 })
	h.stt.Final("no wait, not yet")
	waitFor(t, "second request", func() bool { return len(h.llm.Requests()) == 2 })
	h.stt.Send(stt.Event{Type: stt.EndOfSpeech})

	fin := h.waitSpeeches(1)[0]
	if fin.Interrupted || fin.Err != nil {
		t.Fatalf("speech: %+v", fin)
	}
	time.Sleep(20 * time.Millisecond)
	if ran.Load() {
		t.Fatal("the tool call of a discarded request ran")
	}
	if n := len(h.llm.Requests()); n != 2 {
		t.Fatalf("%d model requests, want 2", n)
	}
	req := h.llm.Requests()[1]
	if last := req.Messages[len(req.Messages)-1]; last.Content != "book a table no wait, not yet" {
		t.Fatalf("second request ends with %+v", last)
	}
	msgs := h.s.History().Messages()
	if roles(msgs) != "user,assistant" || msgs[0].Content != "book a table no wait, not yet" || msgs[1].Content != "For how many people?" {
		t.Fatalf("history: %+v", msgs)
	}
	if h.llm.OpenStreams() != 0 {
		t.Fatalf("%d model streams left open", h.llm.OpenStreams())
	}
}

// A request sent ahead is not used when the history changed in between:
// the reply must see what was added.
func TestPreemptiveGenerationIsDiscardedWhenTheHistoryChanges(t *testing.T) {
	h := newHarness(t, []agentstest.Response{agentstest.Text("Stale."), agentstest.Text("Fresh.")}, withOpts(slowEndpointing)).start()

	h.stt.Final("where is my order")
	waitFor(t, "the request sent ahead", func() bool { return len(h.llm.Requests()) == 1 })
	h.s.History().Append(llm.Message{Role: llm.RoleSystem, Content: "The order shipped today."})
	h.stt.Send(stt.Event{Type: stt.EndOfSpeech})

	h.waitSpeeches(1)
	reqs := h.llm.Requests()
	if len(reqs) != 2 {
		t.Fatalf("%d model requests, want 2", len(reqs))
	}
	found := false
	for _, m := range reqs[1].Messages {
		found = found || m.Content == "The order shipped today."
	}
	if !found {
		t.Fatalf("the reply did not see the added message: %+v", reqs[1].Messages)
	}
	msgs := h.s.History().Messages()
	if msgs[len(msgs)-1].Content != "Fresh." {
		t.Fatalf("history: %+v", msgs)
	}
	if m := metricsOf[agents.LLMMetrics](h); len(m) != 1 || m[0].Preemptive != 0 {
		t.Fatalf("llm metrics: %+v", m)
	}
	if h.llm.OpenStreams() != 0 {
		t.Fatalf("%d model streams left open", h.llm.OpenStreams())
	}
}

func TestPreemptiveGenerationCanBeDisabledAndCleared(t *testing.T) {
	t.Run("disabled", func(t *testing.T) {
		h := newHarness(t, []agentstest.Response{agentstest.Text("Answer.")}, withOpts(slowEndpointing), withOpts(func(o *agents.SessionOptions) {
			o.Turn.DisablePreemptiveGeneration = true
		})).start()
		h.stt.Final("hello")
		time.Sleep(150 * time.Millisecond)
		if n := len(h.llm.Requests()); n != 0 {
			t.Fatalf("%d model requests before the turn ended", n)
		}
		h.waitSpeeches(1)
		if m := metricsOf[agents.LLMMetrics](h); len(m) != 1 || m[0].Preemptive != 0 {
			t.Fatalf("llm metrics: %+v", m)
		}
	})
	t.Run("turn cleared", func(t *testing.T) {
		h := newHarness(t, []agentstest.Response{{Hang: true}}, withOpts(slowEndpointing)).start()
		h.stt.Final("never mind")
		waitFor(t, "the request sent ahead", func() bool { return h.llm.OpenStreams() == 1 })
		h.s.ClearUserTurn()
		waitFor(t, "the request to be cancelled", func() bool { return h.llm.OpenStreams() == 0 })
		time.Sleep(500 * time.Millisecond)
		if n := len(eventsOf[agents.SpeechStarted](h)); n != 0 {
			t.Fatalf("%d speeches after the turn was cleared", n)
		}
	})
	t.Run("session closed", func(t *testing.T) {
		h := newHarness(t, []agentstest.Response{{Hang: true}}, withOpts(slowEndpointing)).start()
		h.stt.Final("goodbye")
		waitFor(t, "the request sent ahead", func() bool { return h.llm.OpenStreams() == 1 })
		if err := h.s.Close(); err != nil {
			t.Fatal(err)
		}
		if n := h.llm.OpenStreams(); n != 0 {
			t.Fatalf("%d model streams left open after Close", n)
		}
	})
}
