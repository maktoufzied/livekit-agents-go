package dsp

import (
	"math"
	"testing"
)

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
			r := NewResampler(c.in, c.model)
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
				out = r.Push(out, data, c.channels)
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
