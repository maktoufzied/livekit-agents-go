package agents

import (
	"bytes"
	"context"
	"strings"
	"sync"
	"time"

	"github.com/maktoufzied/livekit-agents-go/llm"
)

// maxPreemptiveAttempts bounds the model requests started ahead of time in
// one user turn. Each time the user goes on, the request is thrown away.
const maxPreemptiveAttempts = 3

// preemptive is a model request started before the user turn is committed,
// while the endpointing delay runs. It is used for the reply only if, when
// the reply starts, the request to send is exactly this one.
type preemptive struct {
	text    string // the user text it answers
	req     llm.Request
	cancel  context.CancelFunc
	opened  chan struct{} // closed when Chat returned
	settled chan struct{} // closed when taken or discarded

	mu      sync.Mutex
	stream  llm.Stream
	err     error
	settle  bool
	dropped bool
}

// replyRequest builds the model request of a reply from the history.
func (s *Session) replyRequest(history []llm.Message, instructions string, choice llm.ToolChoice) llm.Request {
	req := llm.Request{ToolChoice: choice}
	if ins := strings.TrimSpace(s.opts.Agent.Instructions); ins != "" {
		req.Messages = append(req.Messages, llm.Message{Role: llm.RoleSystem, Content: ins})
	}
	req.Messages = append(req.Messages, history...)
	if instructions != "" {
		req.Messages = append(req.Messages, llm.Message{Role: llm.RoleSystem, Content: instructions})
	}
	// Tools are always declared, even with ToolChoiceNone: the history may
	// reference them and some providers reject that otherwise.
	req.Tools = s.specs
	return req
}

// startPreemptive sends the request that answers text, as if the turn was
// committed now. Nothing is recorded and no tool runs until it is taken.
func (s *Session) startPreemptive(text string) *preemptive {
	history := append(s.conv.Messages(), llm.Message{Role: llm.RoleUser, Content: text})
	ctx, cancel := context.WithCancel(s.ctx)
	p := &preemptive{
		text:    text,
		req:     s.replyRequest(history, "", llm.ToolChoiceAuto),
		cancel:  cancel,
		opened:  make(chan struct{}),
		settled: make(chan struct{}),
	}
	// Tracked so that Close waits for it; it stays until the request is
	// taken or discarded, so that an unused stream is always closed.
	s.goRun(func() {
		stream, err := s.opts.LLM.Chat(ctx, p.req)
		p.mu.Lock()
		p.stream, p.err = stream, err
		p.mu.Unlock()
		close(p.opened)
		select {
		case <-p.settled:
		case <-s.ctx.Done():
		}
		p.discard() // no-op when it was taken
	})
	return p
}

// discard cancels the request. It is idempotent and a no-op once taken.
func (p *preemptive) discard() {
	p.mu.Lock()
	if p.settle {
		p.mu.Unlock()
		return
	}
	p.settle, p.dropped = true, true
	p.mu.Unlock()
	close(p.settled)
	p.cancel()
	// The stream, if any, is closed once Chat has returned.
	go func() {
		<-p.opened
		p.mu.Lock()
		stream := p.stream
		p.mu.Unlock()
		if stream != nil {
			_ = stream.Close()
		}
	}()
}

// take hands the request over to a reply whose request is req and whose
// context is ctx. ok is false when the request does not match or was
// discarded; the caller then sends its own. Otherwise the result is that of
// the caller's first attempt: a stream that the caller owns and closes, and
// that ends with ctx, or the error of the request.
func (p *preemptive) take(ctx context.Context, req llm.Request) (stream llm.Stream, err error, ok bool) {
	if !sameRequest(p.req, req) {
		p.discard()
		return nil, nil, false
	}
	select {
	case <-p.opened:
	case <-ctx.Done():
		p.discard()
		return nil, nil, false
	}
	p.mu.Lock()
	if p.settle {
		p.mu.Unlock()
		return nil, nil, false
	}
	p.settle = true
	stream, err = p.stream, p.err
	p.mu.Unlock()
	close(p.settled)
	if err != nil {
		p.cancel()
		return nil, err, true
	}
	context.AfterFunc(ctx, p.cancel)
	return stream, nil, true
}

// sameRequest reports whether two requests are the same for the model:
// identifiers and timestamps of the history items do not count.
func sameRequest(a, b llm.Request) bool {
	if a.ToolChoice != b.ToolChoice || len(a.Messages) != len(b.Messages) || len(a.Tools) != len(b.Tools) {
		return false
	}
	for i := range a.Tools {
		if a.Tools[i].Name != b.Tools[i].Name {
			return false
		}
	}
	for i := range a.Messages {
		x, y := a.Messages[i], b.Messages[i]
		if x.Role != y.Role || x.Content != y.Content || x.ToolCallID != y.ToolCallID || x.Name != y.Name ||
			x.IsError != y.IsError || x.Interrupted != y.Interrupted || len(x.ToolCalls) != len(y.ToolCalls) {
			return false
		}
		for j := range x.ToolCalls {
			c, d := x.ToolCalls[j], y.ToolCalls[j]
			if c.ID != d.ID || c.Name != d.Name || !bytes.Equal(c.Arguments, d.Arguments) {
				return false
			}
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Loop side

// preempt starts the model on the pending user text, unless that is
// disabled, already done for this text, or pointless: while a speech runs
// or waits, the history is about to change and the request would not match.
func (s *Session) preempt() {
	t := &s.turn
	if s.opts.Turn.DisablePreemptiveGeneration {
		return
	}
	text := strings.TrimSpace(strings.Join(t.finals, " "))
	if text == "" || (t.preempt != nil && t.preempt.text == text) {
		return
	}
	s.dropPreemptive()
	s.mu.Lock()
	busy := len(s.speeches) > 0
	s.mu.Unlock()
	if busy || t.preemptCount >= maxPreemptiveAttempts {
		return
	}
	t.preemptCount++
	t.preempt = s.startPreemptive(text)
	t.preemptAt = time.Now()
}

func (s *Session) dropPreemptive() {
	if p := s.turn.preempt; p != nil {
		s.turn.preempt = nil
		p.discard()
	}
}
