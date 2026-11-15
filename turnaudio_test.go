package agents

import (
	"testing"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
)

// ramp returns n mono samples at 16 kHz counting up from first.
func ramp(first, n int) audio.Frame {
	data := make([]int16, n)
	for i := range data {
		data[i] = int16((first + i) % 30000)
	}
	return audio.Frame{Data: data, SampleRate: 16000, Channels: 1}
}

func TestTurnAudio(t *testing.T) {
	b := newTurnAudio()
	if f := b.snapshot(); len(f.Data) != 0 {
		t.Fatalf("snapshot before any audio: %d samples", len(f.Data))
	}
	b.mark(b.position(), 0, false) // no audio yet: nothing to mark

	// One second, then the turn starts 100 ms ago, then half a second more.
	b.write(ramp(0, 16000))
	b.mark(b.position(), 100*time.Millisecond, false)
	b.write(ramp(16000, 8000))
	lead := int(turnAudioLead * 16000 / time.Second)
	f := b.snapshot()
	if want := 1600 + lead + 8000; len(f.Data) != want {
		t.Fatalf("turn = %d samples, want %d", len(f.Data), want)
	}
	if f.SampleRate != 16000 || f.Channels != 1 {
		t.Fatalf("format = %v", f.Format())
	}
	if first, want := f.Data[0], int16(16000-1600-lead); first != want {
		t.Fatalf("first sample = %d, want %d", first, want)
	}
	if last := f.Data[len(f.Data)-1]; last != 23999 {
		t.Fatalf("last sample = %d, want 23999", last)
	}

	// A later mark does not move the start, unless asked to start over.
	b.mark(b.position(), 0, false)
	if got := len(b.snapshot().Data); got != len(f.Data) {
		t.Fatalf("second mark moved the start: %d samples", got)
	}
	b.mark(b.position(), 0, true)
	if got := len(b.snapshot().Data); got != lead {
		t.Fatalf("restart = %d samples, want %d", got, lead)
	}

	// A mark can point back to where the speech was detected.
	b.write(ramp(24000, 1600))
	b.mark(b.position()-1600, 0, true)
	if got := len(b.snapshot().Data); got != lead+1600 {
		t.Fatalf("mark in the past = %d samples, want %d", got, lead+1600)
	}

	// Once the speech stopped, the turn ends shortly after it, however
	// much silence follows; it runs to now again when the user resumes.
	tail := int(turnAudioTail * 16000 / time.Second)
	b.write(ramp(25600, 8000))
	b.stopped(b.position(), 400*time.Millisecond)
	b.write(ramp(33600, 16000))
	f = b.snapshot()
	if want := lead + 1600 + 8000 - 6400 + tail; len(f.Data) != want {
		t.Fatalf("stopped turn = %d samples, want %d", len(f.Data), want)
	}
	if last, want := f.Data[len(f.Data)-1], int16((33600-6400+tail-1)%30000); last != want {
		t.Fatalf("stopped turn ends with %d, want %d", last, want)
	}
	b.resumed()
	if got, want := len(b.snapshot().Data), lead+1600+8000+16000; got != want {
		t.Fatalf("resumed turn = %d samples, want %d", got, want)
	}

	// A turn longer than the window keeps its end.
	for i := 0; i < 20; i++ {
		b.write(ramp(49600+i*16000, 16000))
	}
	f = b.snapshot()
	window := int(turnAudioWindow * 16000 / time.Second)
	if len(f.Data) != window {
		t.Fatalf("long turn = %d samples, want %d", len(f.Data), window)
	}
	end := 49600 + 20*16000
	for i, s := range f.Data {
		if want := int16((end - window + i) % 30000); s != want {
			t.Fatalf("sample %d = %d, want %d", i, s, want)
		}
	}

	// Without a mark the whole window is returned.
	b.reset()
	if got := len(b.snapshot().Data); got != window {
		t.Fatalf("after reset = %d samples, want %d", got, window)
	}

	// A frame larger than the window, and a change of format.
	b.write(ramp(0, window+100))
	if f := b.snapshot(); len(f.Data) != window || f.Data[0] != 100 {
		t.Fatalf("oversized frame: %d samples, first %d", len(f.Data), f.Data[0])
	}
	b.write(audio.Frame{Data: make([]int16, 960), SampleRate: 48000, Channels: 2})
	if f := b.snapshot(); len(f.Data) != 960 || f.SampleRate != 48000 || f.Channels != 2 {
		t.Fatalf("after a format change: %d samples, %v", len(f.Data), f.Format())
	}
}
