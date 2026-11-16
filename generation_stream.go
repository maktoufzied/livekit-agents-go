package agents

import (
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/internal/segment"
	"github.com/maktoufzied/livekit-agents-go/tts"
)

// synthStream is the synthesis goroutine with a streaming TTS. The text of
// a model response goes to one stream as it arrives and is flushed at the
// end of the response; the audio comes back as one segment.
//
//	write → textCh → [synthStream → tts.InputStream → read] → segCh → [playout]
func (p *pipeline) synthStream() {
	defer close(p.synthDone)
	defer close(p.segCh)

	var cur *streamSeg
	defer func() {
		if cur != nil {
			cur.abort()
			<-cur.done
		}
	}()
	// end completes the current segment: it flushes the text and waits
	// until all its audio was received.
	end := func() bool {
		if cur == nil {
			return true
		}
		c := cur
		cur = nil
		c.flush()
		select {
		case <-c.done:
		case <-p.ctx.Done():
			c.abort()
			<-c.done
			return false
		}
		if c.err != nil {
			if p.ctx.Err() == nil {
				p.fail(SourceTTS, c.err)
			}
			return false
		}
		return true
	}

	for {
		var it pipeItem
		var ok bool
		select {
		case it, ok = <-p.textCh:
			if !ok {
				end()
				return
			}
		case <-p.ctx.Done():
			return
		}
		switch {
		case it.barrier != nil:
			if !end() {
				return
			}
			select {
			case p.segCh <- it:
			case <-p.ctx.Done():
				return
			}
		case it.flush:
			if !end() {
				return
			}
		case it.text != "":
			if cur == nil {
				if strings.TrimSpace(it.text) == "" {
					continue // nothing to say yet
				}
				c, err := p.openStreamSeg(it.step)
				if err != nil {
					if p.ctx.Err() == nil {
						p.fail(SourceTTS, err)
					}
					return
				}
				cur = c
				select {
				case p.segCh <- pipeItem{step: it.step, seg: c.sg}:
				case <-p.ctx.Done():
					return
				}
			}
			cur.push(it.text)
		}
	}
}

// streamSeg is the synthesis of one streamed segment.
type streamSeg struct {
	p    *pipeline
	sg   *seg
	done chan struct{} // closed when the reader exited; err is set before
	err  error

	mu      sync.Mutex
	st      tts.InputStream
	flushed bool
	attempt int
	start   time.Time // first text pushed
	timer   *time.Timer
	state   atomic.Int32 // firstWaiting → firstReceived | firstTimedOut
}

func (p *pipeline) openStreamSeg(step int) (*streamSeg, error) {
	c := &streamSeg{p: p, done: make(chan struct{}), attempt: 1}
	st, err := c.open()
	if err != nil {
		return nil, err
	}
	c.st = st
	c.sg = &seg{
		step:   step,
		stream: true,
		split:  segment.New(),
		frames: make(chan audio.Frame, p.s.opts.Buffers.SentenceFrames),
	}
	p.mu.Lock()
	p.segs = append(p.segs, c.sg)
	p.mu.Unlock()
	go c.read()
	return c, nil
}

// open opens a stream, retrying retryable failures.
func (c *streamSeg) open() (tts.InputStream, error) {
	s := c.p.s
	for {
		st, err := c.p.streamer.SynthesizeStream(c.p.ctx)
		if err == nil {
			return st, nil
		}
		if cerr := c.p.ctx.Err(); cerr != nil {
			return nil, cerr
		}
		if c.attempt > s.opts.Retry.MaxRetries || !retryable(err) {
			return nil, err
		}
		s.reportError(SourceTTS, fmt.Errorf("attempt %d failed, retrying: %w", c.attempt, err), true)
		if !sleep(c.p.ctx, s.backoff(c.attempt)) {
			return nil, c.p.ctx.Err()
		}
		c.attempt++
	}
}

// push sends text to the stream and records it. A failure of the stream is
// left to the reader, which sees it too and decides whether to retry.
func (c *streamSeg) push(text string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	sg := c.sg
	sg.mu.Lock()
	sg.text += text
	sg.addSentences(sg.split.Push(text))
	sg.mu.Unlock()
	if c.start.IsZero() {
		c.start = time.Now()
		c.armTimer()
	}
	_ = c.st.Push(text)
}

func (c *streamSeg) flush() {
	c.mu.Lock()
	defer c.mu.Unlock()
	sg := c.sg
	sg.mu.Lock()
	sg.addSentences(sg.split.Flush())
	sg.mu.Unlock()
	c.flushed = true
	_ = c.st.Flush()
}

// abort closes the stream, which makes the reader return.
func (c *streamSeg) abort() {
	c.mu.Lock()
	st := c.st
	c.mu.Unlock()
	_ = st.Close()
}

// armTimer starts the watchdog of the first audio. c.mu is held.
func (c *streamSeg) armTimer() {
	c.state.Store(firstWaiting)
	c.timer = time.AfterFunc(c.p.s.opts.Timeouts.TTSFirstAudio, func() {
		if c.state.CompareAndSwap(firstWaiting, firstTimedOut) {
			c.abort()
		}
	})
}

// addSentences records where the sentences cut from the text start. sg.mu
// is held.
func (sg *seg) addSentences(sentences []string) {
	for _, text := range sentences {
		at := sg.chars
		if i := strings.Index(sg.text[sg.chars:], text); i >= 0 {
			at += i
		}
		sg.sents = append(sg.sents, sentence{text: text, at: at})
		sg.chars = at + len(text)
	}
}

// read moves the audio of the stream into the segment. A stream that fails
// before producing any audio is reopened and fed the text again, within the
// retry limit; once audio was produced, a failure ends the speech.
func (c *streamSeg) read() {
	defer close(c.done)
	defer close(c.sg.frames)
	p, s, sg := c.p, c.p.s, c.sg
	want := s.opts.TTS.Format()
	var (
		got  bool
		ttfb time.Duration
	)
	for {
		c.mu.Lock()
		st := c.st
		c.mu.Unlock()

		f, err := st.Recv()
		if err == nil && f.Format() != want {
			err = &FormatError{Stage: SourceTTS, Want: want, Got: f.Format()}
		}
		if err == nil && !got {
			if c.state.CompareAndSwap(firstWaiting, firstReceived) {
				got = true
				c.mu.Lock()
				if c.timer != nil {
					c.timer.Stop()
				}
				ttfb = time.Since(c.start)
				c.mu.Unlock()
			} else {
				err = errFirstOutputTimeout
			}
		}
		if err == nil {
			sg.mu.Lock()
			sg.total += f.Duration()
			sg.mu.Unlock()
			select {
			case sg.frames <- f:
				continue
			case <-p.ctx.Done():
				err = p.ctx.Err()
			}
		}

		_ = st.Close()
		c.mu.Lock()
		if c.timer != nil {
			c.timer.Stop()
		}
		start, attempt := c.start, c.attempt
		c.mu.Unlock()
		if errors.Is(err, io.EOF) {
			sg.mu.Lock()
			sg.complete = true
			total, chars := sg.total, len(sg.text)
			sg.mu.Unlock()
			s.emit(MetricsCollected{Metric: TTSMetrics{SpeechID: p.sp.id, FirstAudio: ttfb, Duration: time.Since(start), AudioDuration: total, Characters: chars, Attempts: attempt}})
			return
		}
		if c.state.Load() == firstTimedOut {
			err = &TimeoutError{Op: "tts first audio", After: s.opts.Timeouts.TTSFirstAudio}
		}
		if cerr := p.ctx.Err(); cerr != nil {
			c.err = cerr
			return
		}
		if got || attempt > s.opts.Retry.MaxRetries || !retryable(err) {
			c.err = err
			return
		}
		s.reportError(SourceTTS, fmt.Errorf("attempt %d failed, retrying: %w", attempt, err), true)
		if !sleep(p.ctx, s.backoff(attempt)) {
			c.err = p.ctx.Err()
			return
		}

		// Reopen and say everything again: nothing was heard yet.
		c.mu.Lock()
		c.attempt++
		st, err = p.streamer.SynthesizeStream(p.ctx)
		if err != nil {
			c.mu.Unlock()
			c.err = err
			return
		}
		c.st = st
		sg.mu.Lock()
		text := sg.text
		sg.mu.Unlock()
		_ = st.Push(text)
		if c.flushed {
			_ = st.Flush()
		}
		if !c.start.IsZero() {
			c.armTimer()
		}
		c.mu.Unlock()
	}
}
