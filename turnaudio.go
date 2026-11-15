package agents

import (
	"sync"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
)

const (
	// turnAudioWindow is how much of the user's turn is kept for the
	// TurnDetector.
	turnAudioWindow = 8 * time.Second
	// turnAudioSlack is kept on top of the window, so that a whole window
	// is still there when the end of the speech is reported, late.
	turnAudioSlack = time.Second
	// turnAudioLead is kept before the detected start of speech, so that
	// the first syllable is not cut.
	turnAudioLead = 200 * time.Millisecond
	// turnAudioTail is kept after the detected end of speech. End-of-turn
	// models are trained on turns that stop shortly after the last word; a
	// long silence at the end makes any turn sound finished.
	turnAudioTail = 200 * time.Millisecond
)

// turnAudio keeps the most recent input audio and remembers where the
// current user turn started. The input goroutine writes, the loop goroutine
// marks and reads.
type turnAudio struct {
	mu     sync.Mutex
	format audio.Format
	ring   []int16 // interleaved samples; stream position p is at index p % len(ring)
	total  int64   // interleaved samples written since the format was set
	start  int64   // value of total where the turn starts; -1 when unset
	end    int64   // value of total where the speech stopped; -1 while it goes on
}

func newTurnAudio() *turnAudio { return &turnAudio{start: -1, end: -1} }

// write appends a frame, dropping the oldest audio when the window is full.
func (b *turnAudio) write(f audio.Frame) {
	if len(f.Data) == 0 || !f.Format().Valid() {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if f.Format() != b.format {
		b.format = f.Format()
		b.ring = make([]int16, b.samples(turnAudioWindow+turnAudioSlack))
		b.total, b.start, b.end = 0, -1, -1
	}
	data := f.Data
	if len(data) > len(b.ring) {
		b.total += int64(len(data) - len(b.ring))
		data = data[len(data)-len(b.ring):]
	}
	n := copy(b.ring[b.total%int64(len(b.ring)):], data)
	copy(b.ring, data[n:])
	b.total += int64(len(data))
}

// samples converts a duration to a number of interleaved samples.
func (b *turnAudio) samples(d time.Duration) int64 {
	frames := int64(d) * int64(b.format.SampleRate) / int64(time.Second)
	return frames * int64(b.format.Channels)
}

// position returns the place of the next sample, for mark.
func (b *turnAudio) position() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.total
}

// mark records that the user turn started ago before position at. It keeps
// an earlier mark unless restart is set.
func (b *turnAudio) mark(at int64, ago time.Duration, restart bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.format.Valid() || (b.start >= 0 && !restart) {
		return
	}
	// The format may have changed since at was read.
	b.start = max(0, min(at, b.total)-b.samples(ago+turnAudioLead))
}

// stopped records that the speech stopped ago before position at. The
// turn audio ends shortly after that point until resumed is called.
func (b *turnAudio) stopped(at int64, ago time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.format.Valid() {
		return
	}
	b.end = max(0, min(at, b.total)-b.samples(ago)+b.samples(turnAudioTail))
}

// resumed records that the user speaks again: the turn audio runs up to now.
func (b *turnAudio) resumed() {
	b.mu.Lock()
	b.end = -1
	b.mu.Unlock()
}

// reset forgets the turn.
func (b *turnAudio) reset() {
	b.mu.Lock()
	b.start, b.end = -1, -1
	b.mu.Unlock()
}

// snapshot returns a copy of the turn: from its start, or as far back as
// the window goes when the start is unknown or too old, to shortly after the
// end of the speech, or to now while the user speaks.
func (b *turnAudio) snapshot() audio.Frame {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.format.Valid() {
		return audio.Frame{}
	}
	oldest := b.total - min(b.total, int64(len(b.ring)))
	end := b.total
	if b.end >= 0 {
		end = max(oldest, min(end, b.end))
	}
	start := max(oldest, end-b.samples(turnAudioWindow))
	if b.start >= 0 {
		start = min(max(start, b.start), end)
	}
	out := make([]int16, end-start)
	k := copy(out, b.ring[start%int64(len(b.ring)):])
	copy(out[k:], b.ring)
	return audio.Frame{Data: out, SampleRate: b.format.SampleRate, Channels: b.format.Channels}
}
