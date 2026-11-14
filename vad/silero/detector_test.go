package silero

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/vad"
)

// scriptedModel returns one scripted probability per window and records the
// windows it was given.
type scriptedModel struct {
	probs   []float32
	windows [][]float32
	err     error
	closed  int
}

func (m *scriptedModel) infer(window []float32) (float32, error) {
	m.windows = append(m.windows, append([]float32(nil), window...))
	if m.err != nil {
		return 0, m.err
	}
	if len(m.windows) > len(m.probs) {
		return 0, nil
	}
	return m.probs[len(m.windows)-1], nil
}

func (m *scriptedModel) close() error { m.closed++; return nil }

func defaults(t *testing.T, o Options) Options {
	t.Helper()
	o, err := o.withDefaults()
	if err != nil {
		t.Fatal(err)
	}
	return o
}

// windows16k returns n windows of audio at 16 kHz, in 20 ms frames like the
// room input delivers them.
func feedWindows(t *testing.T, d *detector, n int) []vad.Event {
	t.Helper()
	format := audio.Format{SampleRate: 16000, Channels: 1}
	var out []vad.Event
	total := n * 512
	for total > 0 {
		samples := min(320, total)
		total -= samples
		ev, err := d.Process(audio.Frame{Data: make([]int16, samples), SampleRate: format.SampleRate, Channels: 1})
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, ev...)
	}
	return out
}

func repeat(p float32, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = p
	}
	return out
}

func TestStateMachine(t *testing.T) {
	const w = 32 * time.Millisecond
	var probs []float32
	probs = append(probs, repeat(0.02, 10)...) // silence
	probs = append(probs, 0.95)                // a click: one window
	probs = append(probs, repeat(0.02, 10)...) // silence
	probs = append(probs, repeat(0.95, 20)...) // speech
	probs = append(probs, repeat(0.02, 5)...)  // a pause shorter than MinSilenceDuration
	probs = append(probs, repeat(0.95, 10)...) // speech again
	probs = append(probs, repeat(0.02, 30)...) // silence

	m := &scriptedModel{probs: probs}
	d := newDetector(defaults(t, Options{MinSpeechDuration: 2 * w, MinSilenceDuration: 10 * w}), m)
	events := feedWindows(t, d, len(probs))

	if len(m.windows) != len(probs) {
		t.Fatalf("inferences = %d, want %d", len(m.windows), len(probs))
	}
	if len(events) != 2 {
		t.Fatalf("events = %+v, want one start and one end", events)
	}
	start, end := events[0], events[1]
	if start.Type != vad.SpeechStart || start.SpeechDuration != 2*w {
		t.Errorf("start = %+v, want SpeechStart after %v", start, 2*w)
	}
	if end.Type != vad.SpeechEnd || end.SilenceDuration != 10*w {
		t.Errorf("end = %+v, want SpeechEnd after %v of silence", end, 10*w)
	}
	// 20 + 5 + 10 windows of utterance: the pause counts, the trailing
	// silence does not.
	if got, want := end.SpeechDuration, 35*w; got != want {
		t.Errorf("speech duration = %v, want %v", got, want)
	}
}

func TestHysteresis(t *testing.T) {
	// 0.4 is below the activation threshold (0.5) and above the
	// deactivation threshold (0.35): it does not start speech but keeps it
	// going.
	t.Run("does not start", func(t *testing.T) {
		m := &scriptedModel{probs: repeat(0.4, 40)}
		d := newDetector(defaults(t, Options{}), m)
		if ev := feedWindows(t, d, 40); len(ev) != 0 {
			t.Fatalf("events = %+v", ev)
		}
	})
	t.Run("keeps going", func(t *testing.T) {
		m := &scriptedModel{probs: append(repeat(0.95, 10), repeat(0.4, 60)...)}
		d := newDetector(defaults(t, Options{}), m)
		ev := feedWindows(t, d, 70)
		if len(ev) != 1 || ev[0].Type != vad.SpeechStart {
			t.Fatalf("events = %+v, want only a start", ev)
		}
	})
}

func TestSmoothing(t *testing.T) {
	d := newDetector(defaults(t, Options{}), &scriptedModel{})
	d.step1(1)
	if d.prob != 1 {
		t.Fatalf("first value = %v, want it unfiltered", d.prob)
	}
	d.step1(0)
	if math.Abs(d.prob-0.35) > 1e-9 {
		t.Fatalf("after 1 then 0 = %v, want 0.35", d.prob)
	}
}

func TestWindowing(t *testing.T) {
	m := &scriptedModel{}
	d := newDetector(defaults(t, Options{}), m)

	// 500 samples: not enough for a window.
	frame := audio.Frame{Data: make([]int16, 500), SampleRate: 16000, Channels: 1}
	for i := range frame.Data {
		frame.Data[i] = int16(i)
	}
	if _, err := d.Process(frame); err != nil {
		t.Fatal(err)
	}
	if len(m.windows) != 0 {
		t.Fatalf("inference ran on a partial window")
	}
	// 600 more: two windows, 76 samples left over.
	frame.Data = make([]int16, 600)
	if _, err := d.Process(frame); err != nil {
		t.Fatal(err)
	}
	if len(m.windows) != 2 || len(m.windows[0]) != 512 {
		t.Fatalf("windows = %d", len(m.windows))
	}
	if got, want := m.windows[0][499], float32(499)/32767; got != want {
		t.Errorf("sample 499 = %v, want %v", got, want)
	}
	if len(d.pending) != 500+600-1024 {
		t.Errorf("pending = %d", len(d.pending))
	}
}

func TestResampling(t *testing.T) {
	cases := []struct {
		name            string
		in, model       int
		channels        int
		samples         int // per channel
		wantModelFrames int
	}{
		{"48k mono to 16k", 48000, 16000, 1, 48000, 16000},
		{"48k stereo to 16k", 48000, 16000, 2, 48000, 16000},
		{"44.1k to 16k", 44100, 16000, 1, 44100, 16000},
		{"24k to 16k", 24000, 16000, 1, 24000, 16000},
		{"8k to 16k", 8000, 16000, 1, 8000, 16000},
		{"16k to 8k", 16000, 8000, 1, 16000, 8000},
		{"8k to 8k", 8000, 8000, 1, 8000, 8000},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := newResampler(c.in, c.model)
			// A 200 Hz tone, pushed in 10 ms frames.
			var out []float32
			per := c.in / 100
			for start := 0; start < c.samples; start += per {
				data := make([]int16, per*c.channels)
				for i := 0; i < per; i++ {
					v := int16(16000 * math.Sin(2*math.Pi*200*float64(start+i)/float64(c.in)))
					for ch := 0; ch < c.channels; ch++ {
						data[i*c.channels+ch] = v
					}
				}
				out = r.push(out, data, c.channels)
			}
			if d := len(out) - c.wantModelFrames; d < -1 || d > 1 {
				t.Fatalf("output samples = %d, want %d", len(out), c.wantModelFrames)
			}
			// The tone survives: same amplitude, 200 cycles per second.
			var peak float32
			crossings := 0
			for i, v := range out {
				peak = max(peak, v)
				if i > 0 && out[i-1] < 0 && v >= 0 {
					crossings++
				}
			}
			if peak < 0.45 || peak > 0.5 {
				t.Errorf("peak = %v, want about 0.49", peak)
			}
			if crossings < 198 || crossings > 201 {
				t.Errorf("rising zero crossings = %d, want about 200", crossings)
			}
		})
	}
}

func TestProcessErrors(t *testing.T) {
	m := &scriptedModel{}
	d := newDetector(defaults(t, Options{}), m)

	if ev, err := d.Process(audio.Frame{SampleRate: 16000, Channels: 1}); err != nil || ev != nil {
		t.Fatalf("empty frame: %v %v", ev, err)
	}
	if _, err := d.Process(audio.Frame{Data: make([]int16, 320), Channels: 1}); err == nil {
		t.Fatal("no error for a frame without a sample rate")
	}
	if _, err := d.Process(audio.Frame{Data: make([]int16, 320), SampleRate: 16000, Channels: 1}); err != nil {
		t.Fatal(err)
	}
	if _, err := d.Process(audio.Frame{Data: make([]int16, 320), SampleRate: 48000, Channels: 1}); err == nil {
		t.Fatal("no error when the sample rate changes")
	}

	// A failing inference is reported and its window is dropped, so the
	// detector recovers on the next frame.
	boom := errors.New("boom")
	m.err = boom
	if _, err := d.Process(audio.Frame{Data: make([]int16, 512), SampleRate: 16000, Channels: 1}); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the inference error", err)
	}
	m.err = nil
	before := len(m.windows)
	if _, err := d.Process(audio.Frame{Data: make([]int16, 512), SampleRate: 16000, Channels: 1}); err != nil {
		t.Fatal(err)
	}
	if len(m.windows) != before+1 {
		t.Fatalf("inferences after recovery = %d, want %d", len(m.windows)-before, 1)
	}

	if err := d.Close(); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(); err != nil || m.closed != 1 {
		t.Fatalf("second Close: err=%v, model closed %d times", err, m.closed)
	}
	if _, err := d.Process(audio.Frame{Data: make([]int16, 320), SampleRate: 16000, Channels: 1}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Process after Close = %v, want ErrClosed", err)
	}
}

func TestOptions(t *testing.T) {
	o := defaults(t, Options{})
	if o.MinSpeechDuration != 50*time.Millisecond || o.MinSilenceDuration != 550*time.Millisecond ||
		o.ActivationThreshold != 0.5 || o.DeactivationThreshold != 0.35 || o.SampleRate != 16000 {
		t.Fatalf("defaults = %+v", o)
	}
	if o := defaults(t, Options{ActivationThreshold: 0.1}); o.DeactivationThreshold != 0.01 {
		t.Fatalf("deactivation floor = %v", o.DeactivationThreshold)
	}
	for name, bad := range map[string]Options{
		"sample rate":       {SampleRate: 44100},
		"activation":        {ActivationThreshold: 1.5},
		"deactivation":      {ActivationThreshold: 0.5, DeactivationThreshold: 0.6},
		"negative duration": {MinSilenceDuration: -time.Second},
	} {
		if _, err := bad.withDefaults(); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
