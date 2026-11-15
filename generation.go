package agents

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/internal/segment"
	"github.com/maktoufzied/livekit-agents-go/llm"
)

// speechLoop executes speeches one at a time, in queue order. Because a
// speech only returns once its playout goroutine has exited and (when cut
// short) the output has been cleared, audio from one speech can never leak
// into the next.
func (s *Session) speechLoop() {
	for {
		select {
		case sp := <-s.speechCh:
			s.runSpeech(sp)
		case <-s.ctx.Done():
			return
		}
	}
}

func (s *Session) runSpeech(sp *SpeechHandle) {
	var (
		err    error
		played time.Duration
		ran    bool
	)
	// The user message that triggered this reply is committed here, not
	// when the turn ended: speeches run one at a time, so it always lands
	// after whatever the previous speech recorded (for example the played
	// part of the reply the user just interrupted).
	if sp.userMsg != nil && s.ctx.Err() == nil {
		s.commit(*sp.userMsg)
	}
	if !sp.Interrupted() && s.ctx.Err() == nil {
		ran = true
		s.emit(SpeechStarted{SpeechID: sp.id, Source: sp.source})
		s.setAgentState(AgentThinking)
		p := newPipeline(s, sp)
		if sp.source == SpeechFromSay {
			err = s.runSay(sp, p)
		} else {
			err = s.runReply(sp, p)
		}
		played = p.shutdown()
		s.setAgentState(AgentListening)
	}

	// A speech that ran to completion is not "interrupted", even if an
	// interruption arrived just after its last frame was played.
	interrupted := sp.Interrupted() && (!ran || err != nil)
	switch {
	case s.ctx.Err() != nil:
		err = ErrSessionClosed
	case interrupted:
		err = nil
	case err != nil:
		s.reportError(errorSource(err), err, true)
	}
	s.mu.Lock()
	for i, x := range s.speeches {
		if x == sp {
			s.speeches = append(s.speeches[:i], s.speeches[i+1:]...)
			break
		}
	}
	s.mu.Unlock()
	if sp.preempt != nil {
		sp.preempt.discard() // the speech never reached its first model step
	}
	s.emit(SpeechFinished{SpeechID: sp.id, Interrupted: interrupted, PlayedDuration: played, Err: err})
	if interrupted && err == nil {
		err = ErrInterrupted
	}
	sp.finish(err)
}

// States of the "first output" watchdog of a provider request. Exactly one
// of the stream reader and the timer wins the transition out of
// firstWaiting, so a stream is never closed after its first output arrived.
const (
	firstWaiting int32 = iota
	firstReceived
	firstTimedOut
)

var errFirstOutputTimeout = errors.New("first output arrived after the timeout")

// stageError tags an error with the pipeline stage that produced it.
type stageError struct {
	stage ErrorSource
	err   error
}

func (e *stageError) Error() string { return string(e.stage) + ": " + e.err.Error() }
func (e *stageError) Unwrap() error { return e.err }

func errorSource(err error) ErrorSource {
	var se *stageError
	if errors.As(err, &se) {
		return se.stage
	}
	return SourceSpeech
}

func (s *Session) runSay(sp *SpeechHandle, p *pipeline) error {
	sp2 := segment.New()
	for _, sentence := range append(sp2.Push(sp.text), sp2.Flush()...) {
		if err := p.push(0, sentence); err != nil {
			break
		}
	}
	err := p.finish()
	if !sp.addToHistory {
		return err
	}
	if err == nil {
		s.commit(llm.Message{Role: llm.RoleAssistant, Content: sp.text})
		return nil
	}
	p.stop()
	if text := p.playedText(0); text != "" {
		s.commit(llm.Message{Role: llm.RoleAssistant, Content: text, Interrupted: true})
	}
	return err
}

// runReply runs the model/tool loop of one reply. Whatever happens, the
// history ends up consistent: assistant text is what was actually played,
// and every recorded tool call has a result.
func (s *Session) runReply(sp *SpeechHandle, p *pipeline) error {
	for step := 0; ; step++ {
		choice := llm.ToolChoiceAuto
		if step >= s.opts.Tools.MaxSteps {
			choice = llm.ToolChoiceNone // force a spoken answer
		}
		text, calls, err := s.runLLMStep(sp, p, step, choice)
		if err != nil {
			p.stop()
			if played := p.playedText(step); played != "" {
				s.commit(llm.Message{Role: llm.RoleAssistant, Content: played, Interrupted: true})
			}
			return err
		}

		if len(calls) == 0 {
			if err := p.finish(); err != nil {
				p.stop()
				if played := p.playedText(step); played != "" {
					s.commit(llm.Message{Role: llm.RoleAssistant, Content: played, Interrupted: true})
				}
				return err
			}
			if strings.TrimSpace(text) != "" {
				s.commit(llm.Message{Role: llm.RoleAssistant, Content: text})
			}
			return nil
		}

		// Tools run while any text of this step is still being spoken.
		results := s.execTools(sp, calls)
		berr := p.barrier()
		assistant := llm.Message{Role: llm.RoleAssistant, Content: text, ToolCalls: calls}
		if berr != nil {
			p.stop()
			assistant.Content = p.playedText(step)
			assistant.Interrupted = true
		}
		s.commit(append([]llm.Message{assistant}, results...)...)
		if berr != nil {
			return berr
		}
	}
}

// runLLMStep streams one model response into the pipeline. The request is
// retried only if it failed before producing any output.
func (s *Session) runLLMStep(sp *SpeechHandle, p *pipeline, step int, choice llm.ToolChoice) (string, []llm.ToolCall, error) {
	ctx := p.ctx
	req := s.replyRequest(s.conv.Messages(), sp.instructions, choice)

	// A request sent ahead of the end of the turn replaces the first
	// attempt of the first step, if it is the request to send now.
	var (
		ahead    llm.Stream
		aheadErr error
		aheadOK  bool
		aheadBy  time.Duration
	)
	if pre := sp.preempt; pre != nil {
		sp.preempt = nil
		if ahead, aheadErr, aheadOK = pre.take(ctx, req); aheadOK {
			aheadBy = time.Since(sp.preemptAt)
		}
	}

	for attempt := 1; ; attempt++ {
		var (
			text  strings.Builder
			calls []llm.ToolCall
			usage llm.Usage
			got   bool
			state atomic.Int32 // firstWaiting → firstReceived | firstTimedOut
			split = segment.New()
			start = time.Now()
			ttft  time.Duration
		)
		var stream llm.Stream
		var err error
		if aheadOK {
			stream, err, aheadOK = ahead, aheadErr, false
		} else {
			aheadBy = 0
			stream, err = s.opts.LLM.Chat(ctx, req)
		}
		if err == nil {
			timer := time.AfterFunc(s.opts.Timeouts.LLMFirstToken, func() {
				if state.CompareAndSwap(firstWaiting, firstTimedOut) {
					_ = stream.Close()
				}
			})
			for {
				var ch llm.Chunk
				if ch, err = stream.Recv(); err != nil {
					break
				}
				if !got {
					// Lost the race against the timer: the stream is being
					// closed, so this attempt counts as timed out.
					if !state.CompareAndSwap(firstWaiting, firstReceived) {
						err = errFirstOutputTimeout
						break
					}
					got = true
					timer.Stop()
					ttft = time.Since(start)
				}
				if ch.Usage != nil {
					usage = *ch.Usage
				}
				calls = append(calls, ch.ToolCalls...)
				if ch.Text == "" {
					continue
				}
				text.WriteString(ch.Text)
				for _, sentence := range split.Push(ch.Text) {
					if err = p.push(step, sentence); err != nil {
						break
					}
				}
				if err != nil {
					break
				}
			}
			timer.Stop()
			_ = stream.Close()
			if errors.Is(err, io.EOF) {
				for _, sentence := range split.Flush() {
					if perr := p.push(step, sentence); perr != nil {
						return text.String(), nil, perr
					}
				}
				s.emit(MetricsCollected{Metric: LLMMetrics{SpeechID: sp.id, FirstToken: ttft, Duration: time.Since(start), Usage: usage, Attempts: attempt, Preemptive: aheadBy}})
				return text.String(), calls, nil
			}
			if state.Load() == firstTimedOut {
				err = &TimeoutError{Op: "llm first token", After: s.opts.Timeouts.LLMFirstToken}
			}
		}
		if ctx.Err() != nil {
			return text.String(), nil, p.ctxErr()
		}
		err = &stageError{stage: SourceLLM, err: err}
		if got || attempt > s.opts.Retry.MaxRetries || !retryable(err) {
			return text.String(), nil, err
		}
		s.reportError(SourceLLM, fmt.Errorf("attempt %d failed, retrying: %w", attempt, err), true)
		if !sleep(ctx, s.backoff(attempt)) {
			return "", nil, ctx.Err()
		}
	}
}

// ---------------------------------------------------------------------------
// Tools

// execTools runs the calls of one model step and returns one tool message
// per call, in call order. Tools are never retried by the SDK.
func (s *Session) execTools(sp *SpeechHandle, calls []llm.ToolCall) []llm.Message {
	results := make([]llm.Message, len(calls))
	if s.opts.Tools.Sequential {
		for i, c := range calls {
			results[i] = s.execTool(sp, c)
		}
		return results
	}
	var wg sync.WaitGroup
	for i, c := range calls {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i] = s.execTool(sp, c)
		}()
	}
	wg.Wait()
	return results
}

const genericToolFailure = "The tool failed with an internal error."

func (s *Session) execTool(sp *SpeechHandle, call llm.ToolCall) llm.Message {
	start := time.Now()
	s.emit(ToolCallStarted{SpeechID: sp.id, Call: call})
	out, err := s.invokeTool(sp, call)

	msg := llm.Message{Role: llm.RoleTool, ToolCallID: call.ID, Name: call.Name, Content: out}
	if err != nil {
		msg.IsError = true
		var te *llm.ToolError
		switch {
		case errors.As(err, &te):
			msg.Content = te.Message
		case errors.Is(err, llm.ErrInvalidArguments), errors.Is(err, errUnknownTool):
			msg.Content = err.Error()
		case errors.Is(err, context.DeadlineExceeded):
			msg.Content = "The tool call timed out."
		case errors.Is(err, context.Canceled):
			msg.Content = "The tool call was cancelled."
		default:
			msg.Content = genericToolFailure
		}
		s.log.Warn("tool call failed", "tool", call.Name, "call_id", call.ID, "error", err)
	}
	s.emit(ToolCallFinished{SpeechID: sp.id, Call: call, Output: msg.Content, IsError: msg.IsError, Err: err, Duration: time.Since(start)})
	return msg
}

var errUnknownTool = errors.New("unknown tool")

func (s *Session) invokeTool(sp *SpeechHandle, call llm.ToolCall) (string, error) {
	tool, ok := s.tools[call.Name]
	if !ok {
		return "", fmt.Errorf("%w %q", errUnknownTool, call.Name)
	}
	if err := llm.ValidateArguments(tool.Parameters, call.Arguments); err != nil {
		return "", err
	}
	args := call.Arguments
	if len(strings.TrimSpace(string(args))) == 0 {
		args = []byte(`{}`)
	}

	// An interruptible tool is cancelled with its speech; an
	// uninterruptible one only with the session.
	parent := sp.ctx
	if tool.Uninterruptible {
		parent = s.ctx
	}
	timeout := tool.Timeout
	if timeout <= 0 {
		timeout = s.opts.Tools.DefaultTimeout
	}
	ctx, cancel := context.WithTimeout(parent, timeout)
	defer cancel()
	// Never start a handler whose call is already cancelled: its side
	// effects would run while the history says the call was cancelled.
	if err := ctx.Err(); err != nil {
		return "", err
	}

	type result struct {
		out string
		err error
	}
	// The handler runs in its own goroutine so that a handler ignoring its
	// context cannot stall the conversation. Such a handler keeps running
	// in the background until it returns.
	ch := make(chan result, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				ch <- result{err: fmt.Errorf("tool panicked: %v", r)}
			}
		}()
		out, err := tool.Handler(ctx, args)
		ch <- result{out, err}
	}()
	select {
	case r := <-ch:
		return r.out, r.err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

// ---------------------------------------------------------------------------
// Speech pipeline: sentences → synthesis → paced playout

type seg struct {
	step   int
	text   string
	frames chan audio.Frame

	mu       sync.Mutex
	written  time.Duration // handed to the output
	total    time.Duration // synthesized so far
	complete bool          // synthesis finished successfully
}

type pipeItem struct {
	step    int
	text    string
	seg     *seg
	barrier chan struct{}
}

// pipeline turns sentences into played audio for one speech.
//
//	push → textCh → [synth] → segCh → [playout] → AudioOutput
//
// Every channel is bounded. The playout goroutine is the only writer to the
// output while the speech runs.
type pipeline struct {
	s      *Session
	sp     *SpeechHandle
	ctx    context.Context
	cancel context.CancelFunc

	textCh    chan pipeItem
	segCh     chan pipeItem
	closeText sync.Once
	synthDone chan struct{}
	playDone  chan struct{}

	mu         sync.Mutex
	segs       []*seg
	err        error
	stopped    bool
	dropped    time.Duration
	firstAudio bool

	// Owned by the playout goroutine. pending holds the frames to write
	// next, in order; recent the last frames written, to take them back
	// when playout is put on hold.
	pending []heldFrame
	recent  []heldFrame
}

// heldFrame is a frame together with the sentence it belongs to.
type heldFrame struct {
	sg *seg
	f  audio.Frame
}

// recentWindow bounds the frames kept to be played again after a hold. It
// only needs to cover what an output keeps queued.
const recentWindow = 2 * time.Second

func newPipeline(s *Session, sp *SpeechHandle) *pipeline {
	p := &pipeline{
		s:         s,
		sp:        sp,
		textCh:    make(chan pipeItem, s.opts.Buffers.Sentences),
		segCh:     make(chan pipeItem, s.opts.Buffers.SynthesisLookahead),
		synthDone: make(chan struct{}),
		playDone:  make(chan struct{}),
	}
	p.ctx, p.cancel = context.WithCancel(sp.ctx)
	go p.synth()
	go p.playout()
	return p
}

// fail records the first pipeline error and aborts the speech.
func (p *pipeline) fail(stage ErrorSource, err error) {
	p.mu.Lock()
	if p.err == nil {
		p.err = &stageError{stage: stage, err: err}
	}
	p.mu.Unlock()
	p.cancel()
}

// ctxErr returns the pipeline failure if any, else the context error.
func (p *pipeline) ctxErr() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.err != nil {
		return p.err
	}
	return p.ctx.Err()
}

// push queues a sentence. It blocks when the sentence buffer is full, which
// is how TTS and playback apply backpressure to the LLM stream.
func (p *pipeline) push(step int, text string) error {
	select {
	case p.textCh <- pipeItem{step: step, text: text}:
		return nil
	case <-p.ctx.Done():
		return p.ctxErr()
	}
}

// barrier blocks until everything pushed so far has been handed to the
// output.
func (p *pipeline) barrier() error {
	b := make(chan struct{})
	select {
	case p.textCh <- pipeItem{barrier: b}:
	case <-p.ctx.Done():
		return p.ctxErr()
	}
	select {
	case <-b:
		return nil
	case <-p.ctx.Done():
		return p.ctxErr()
	}
}

// finish ends the input and waits until all audio has been played.
func (p *pipeline) finish() error {
	p.closeText.Do(func() { close(p.textCh) })
	select {
	case <-p.playDone:
	case <-p.ctx.Done():
		return p.ctxErr()
	}
	if err := p.ctxErr(); err != nil {
		return err
	}
	if err := p.s.out.Flush(p.ctx); err != nil {
		if p.ctx.Err() == nil {
			p.fail(SourceOutput, err)
		}
		return p.ctxErr()
	}
	return nil
}

// stop aborts the speech: it joins the playout goroutine (the only writer)
// and then drops whatever is still queued in the output. It is idempotent.
func (p *pipeline) stop() {
	p.mu.Lock()
	if p.stopped {
		p.mu.Unlock()
		return
	}
	p.stopped = true
	p.mu.Unlock()

	p.cancel()
	<-p.playDone
	dropped := p.s.out.Clear()
	p.mu.Lock()
	p.dropped += dropped
	dropped = p.dropped
	p.mu.Unlock()
	if p.sp.Interrupted() {
		p.s.emit(MetricsCollected{Metric: InterruptionMetrics{SpeechID: p.sp.id, Latency: time.Since(p.sp.interruptedAt), Dropped: dropped}})
	}
}

// shutdown releases the pipeline and returns the audio actually played.
func (p *pipeline) shutdown() time.Duration {
	if p.ctx.Err() != nil {
		p.stop() // interrupted or failed outside of a stop() path
	}
	p.closeText.Do(func() { close(p.textCh) })
	p.cancel()
	<-p.playDone
	<-p.synthDone

	p.mu.Lock()
	defer p.mu.Unlock()
	var written time.Duration
	for _, sg := range p.segs {
		sg.mu.Lock()
		written += sg.written
		sg.mu.Unlock()
	}
	return max(0, written-p.dropped)
}

// playedText returns the text of a step that the user actually heard. It
// must be called after stop. Audio dropped from the output is attributed to
// the most recently written segments; a partially played sentence is cut at
// the word matching the played fraction of its audio.
func (p *pipeline) playedText(step int) string {
	p.mu.Lock()
	defer p.mu.Unlock()
	remaining := p.dropped
	played := make([]time.Duration, len(p.segs))
	for i := len(p.segs) - 1; i >= 0; i-- {
		sg := p.segs[i]
		sg.mu.Lock()
		w := sg.written
		sg.mu.Unlock()
		cut := min(remaining, w)
		remaining -= cut
		played[i] = w - cut
	}
	var parts []string
	for i, sg := range p.segs {
		if sg.step != step || played[i] <= 0 {
			continue
		}
		sg.mu.Lock()
		total, complete := sg.total, sg.complete
		sg.mu.Unlock()
		if !complete {
			// Synthesis was cut short: estimate the full length from a
			// typical speaking rate of ~15 characters per second.
			total = max(total, time.Duration(len(sg.text))*time.Second/15)
		}
		if total <= 0 || played[i] >= total {
			parts = append(parts, sg.text)
			continue
		}
		if t := segment.TruncateWords(sg.text, float64(played[i])/float64(total)); t != "" {
			parts = append(parts, t)
		}
	}
	return strings.Join(parts, " ")
}

func (p *pipeline) synth() {
	defer close(p.synthDone)
	defer close(p.segCh)
	for {
		var it pipeItem
		var ok bool
		select {
		case it, ok = <-p.textCh:
			if !ok {
				return
			}
		case <-p.ctx.Done():
			return
		}
		if it.barrier == nil {
			it.seg = &seg{step: it.step, text: it.text, frames: make(chan audio.Frame, p.s.opts.Buffers.SentenceFrames)}
			p.mu.Lock()
			p.segs = append(p.segs, it.seg)
			p.mu.Unlock()
		}
		select {
		case p.segCh <- it:
		case <-p.ctx.Done():
			if it.seg != nil {
				close(it.seg.frames)
			}
			return
		}
		if it.seg == nil {
			continue
		}
		err := p.synthesize(it.seg)
		close(it.seg.frames)
		if err != nil {
			if p.ctx.Err() == nil {
				p.fail(SourceTTS, err)
			}
			return
		}
	}
}

// synthesize streams the audio of one sentence into its segment. The
// request is retried only if it failed before producing any audio.
func (p *pipeline) synthesize(sg *seg) error {
	s := p.s
	want := s.opts.TTS.Format()
	for attempt := 1; ; attempt++ {
		var (
			got   bool
			state atomic.Int32 // firstWaiting → firstReceived | firstTimedOut
			start = time.Now()
			ttfb  time.Duration
		)
		stream, err := s.opts.TTS.Synthesize(p.ctx, sg.text)
		if err == nil {
			timer := time.AfterFunc(s.opts.Timeouts.TTSFirstAudio, func() {
				if state.CompareAndSwap(firstWaiting, firstTimedOut) {
					_ = stream.Close()
				}
			})
			for {
				var f audio.Frame
				if f, err = stream.Recv(); err != nil {
					break
				}
				if f.Format() != want {
					err = &FormatError{Stage: SourceTTS, Want: want, Got: f.Format()}
					break
				}
				if !got {
					if !state.CompareAndSwap(firstWaiting, firstReceived) {
						err = errFirstOutputTimeout
						break
					}
					got = true
					timer.Stop()
					ttfb = time.Since(start)
				}
				sg.mu.Lock()
				sg.total += f.Duration()
				sg.mu.Unlock()
				select {
				case sg.frames <- f:
				case <-p.ctx.Done():
					err = p.ctx.Err()
				}
				if err != nil {
					break
				}
			}
			timer.Stop()
			_ = stream.Close()
			if errors.Is(err, io.EOF) {
				sg.mu.Lock()
				sg.complete = true
				total := sg.total
				sg.mu.Unlock()
				s.emit(MetricsCollected{Metric: TTSMetrics{SpeechID: p.sp.id, FirstAudio: ttfb, Duration: time.Since(start), AudioDuration: total, Characters: len(sg.text), Attempts: attempt}})
				return nil
			}
			if state.Load() == firstTimedOut {
				err = &TimeoutError{Op: "tts first audio", After: s.opts.Timeouts.TTSFirstAudio}
			}
		}
		if cerr := p.ctx.Err(); cerr != nil {
			return cerr
		}
		if got || attempt > s.opts.Retry.MaxRetries || !retryable(err) {
			return err
		}
		s.reportError(SourceTTS, fmt.Errorf("attempt %d failed, retrying: %w", attempt, err), true)
		if !sleep(p.ctx, s.backoff(attempt)) {
			return p.ctx.Err()
		}
	}
}

func (p *pipeline) playout() {
	defer close(p.playDone)
	for it := range p.segCh {
		if it.barrier != nil {
			if p.ctx.Err() == nil {
				close(it.barrier)
			}
			continue
		}
		p.playSegment(it.seg)
	}
}

func (p *pipeline) playSegment(sg *seg) {
	announced := false
	for {
		held, changed := p.sp.hold.state()
		if held {
			// Frames taken back by the hold are played before anything new.
			if !p.waitReleased() || !p.flush() {
				return
			}
			continue
		}
		var f audio.Frame
		var ok bool
		select {
		case f, ok = <-sg.frames:
			if !ok {
				return
			}
		case <-changed:
			continue // put on hold while waiting for audio
		case <-p.ctx.Done():
			return
		}
		// Checked again right before writing: once the speech is
		// cancelled not a single extra frame may reach the output.
		if p.ctx.Err() != nil {
			return
		}
		if !announced {
			announced = true
			p.onFirstFrame(sg)
		}
		if !p.deliver(sg, f) {
			return
		}
	}
}

// deliver writes a frame to the output, after any frame taken back by a
// hold. It reports false when the speech is over.
func (p *pipeline) deliver(sg *seg, f audio.Frame) bool {
	p.pending = append(p.pending, heldFrame{sg, f})
	return p.flush()
}

// flush writes the pending frames. It reports false when the speech is
// over.
func (p *pipeline) flush() bool {
	for len(p.pending) > 0 {
		if held, _ := p.sp.hold.state(); held {
			if !p.waitReleased() {
				return false
			}
			continue
		}
		it := p.pending[0]
		ctx, cancel := context.WithCancel(p.ctx)
		if !p.sp.hold.beginWrite(cancel) {
			cancel()
			continue // held since the check above
		}
		err := p.s.out.WriteFrame(ctx, it.f)
		p.sp.hold.endWrite()
		cancel()
		if err != nil {
			if p.ctx.Err() != nil {
				return false
			}
			if held, _ := p.sp.hold.state(); held && errors.Is(err, context.Canceled) {
				continue // the hold aborted the write: the frame stays pending
			}
			p.fail(SourceOutput, err)
			return false
		}
		p.pending = p.pending[1:]
		it.sg.mu.Lock()
		it.sg.written += it.f.Duration()
		it.sg.mu.Unlock()

		p.recent = append(p.recent, it)
		var kept time.Duration
		for i := len(p.recent) - 1; i >= 0; i-- {
			if kept += p.recent[i].f.Duration(); kept > recentWindow {
				p.recent = append(p.recent[:0], p.recent[i+1:]...)
				break
			}
		}
	}
	return true
}

// waitReleased is called by the playout goroutine when the speech is on
// hold. It silences the output, takes back the frames that were queued but
// not played so that they are played after the hold, and blocks until the
// hold is released. It reports false when the speech is cancelled instead.
func (p *pipeline) waitReleased() bool {
	dropped := p.s.out.Clear()
	var back []heldFrame
	for len(p.recent) > 0 && dropped > 0 {
		it := p.recent[len(p.recent)-1]
		d := it.f.Duration()
		if d > dropped {
			// Partly played: only its unplayed end is played again.
			samples := int(int64(dropped) * int64(it.f.SampleRate) / int64(time.Second))
			if samples == 0 {
				break
			}
			it.f.Data = it.f.Data[len(it.f.Data)-samples*it.f.Channels:]
			d = it.f.Duration()
			dropped = 0
		} else {
			dropped -= d
		}
		p.recent = p.recent[:len(p.recent)-1]
		it.sg.mu.Lock()
		it.sg.written -= d
		it.sg.mu.Unlock()
		back = append(back, it)
	}
	for i, j := 0, len(back)-1; i < j; i, j = i+1, j-1 {
		back[i], back[j] = back[j], back[i]
	}
	p.pending = append(back, p.pending...)
	if dropped > 0 {
		// Audio that cannot be played again counts as not heard.
		p.mu.Lock()
		p.dropped += dropped
		p.mu.Unlock()
	}

	p.mu.Lock()
	speaking := p.firstAudio
	p.mu.Unlock()
	if speaking {
		p.s.setAgentState(AgentListening)
	}
	for {
		held, changed := p.sp.hold.state()
		if !held {
			break
		}
		select {
		case <-changed:
		case <-p.ctx.Done():
			return false
		}
	}
	if speaking {
		p.s.setAgentState(AgentSpeaking)
	}
	return true
}

func (p *pipeline) onFirstFrame(sg *seg) {
	p.mu.Lock()
	first := !p.firstAudio
	p.firstAudio = true
	p.mu.Unlock()
	if first {
		p.s.setAgentState(AgentSpeaking)
		p.s.emit(MetricsCollected{Metric: ResponseMetrics{SpeechID: p.sp.id, FirstAudio: time.Since(p.sp.createdAt)}})
	}
	p.s.emit(AgentTranscript{SpeechID: p.sp.id, Text: sg.text})
}
