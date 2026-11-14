package silero

import (
	"fmt"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/vad"
)

// smoothing is the weight of the previous value in the exponential filter
// applied to the model output, as in the upstream Python plugin.
const smoothing = 0.35

// model runs the network on one window of mono samples in [-1, 1] and returns
// the speech probability. It keeps the recurrent state between calls.
type model interface {
	infer(window []float32) (float32, error)
	close() error
}

// detector turns frames into windows, windows into probabilities and
// probabilities into speech boundaries. It is not safe for concurrent use.
type detector struct {
	opts   Options
	model  model
	window int           // samples per inference, at the model rate
	step   time.Duration // audio covered by one window

	inRate   int
	resample resampler
	pending  []float32 // model-rate samples not yet consumed

	prob     float64 // smoothed probability
	hasProb  bool
	speaking bool
	// speechRun and silenceRun are the current uninterrupted runs of windows
	// above and below the threshold. speech is the length of the utterance
	// being reported, including pauses shorter than MinSilenceDuration.
	speechRun  time.Duration
	silenceRun time.Duration
	speech     time.Duration

	closed bool
}

func newDetector(opts Options, m model) *detector {
	window := 512
	if opts.SampleRate == 8000 {
		window = 256
	}
	return &detector{
		opts:   opts,
		model:  m,
		window: window,
		step:   time.Duration(window) * time.Second / time.Duration(opts.SampleRate),
	}
}

// Process implements vad.Detector.
func (d *detector) Process(f audio.Frame) ([]vad.Event, error) {
	if d.closed {
		return nil, ErrClosed
	}
	if len(f.Data) == 0 {
		return nil, nil
	}
	if f.SampleRate <= 0 || f.Channels <= 0 {
		return nil, fmt.Errorf("silero: invalid frame format %dHz/%dch", f.SampleRate, f.Channels)
	}
	if d.inRate == 0 {
		d.inRate = f.SampleRate
		d.resample = newResampler(f.SampleRate, d.opts.SampleRate)
	} else if d.inRate != f.SampleRate {
		return nil, fmt.Errorf("silero: sample rate changed from %d to %d Hz", d.inRate, f.SampleRate)
	}

	d.pending = d.resample.push(d.pending, f.Data, f.Channels)

	var events []vad.Event
	consumed := 0
	for len(d.pending)-consumed >= d.window {
		p, err := d.model.infer(d.pending[consumed : consumed+d.window])
		consumed += d.window
		if err != nil {
			d.pending = d.pending[:copy(d.pending, d.pending[consumed:])]
			return events, fmt.Errorf("silero: inference: %w", err)
		}
		if ev, ok := d.step1(float64(p)); ok {
			events = append(events, ev)
		}
	}
	d.pending = d.pending[:copy(d.pending, d.pending[consumed:])]
	return events, nil
}

// step1 advances the state machine by one window with raw probability p.
func (d *detector) step1(p float64) (vad.Event, bool) {
	if d.hasProb {
		p = smoothing*d.prob + (1-smoothing)*p
	}
	d.prob, d.hasProb = p, true

	if d.speaking {
		d.speech += d.step
	}

	voiced := p >= d.opts.ActivationThreshold || (d.speaking && p > d.opts.DeactivationThreshold)
	if voiced {
		d.speechRun += d.step
		d.silenceRun = 0
		if !d.speaking && d.speechRun >= d.opts.MinSpeechDuration {
			d.speaking = true
			d.speech = d.speechRun
			return vad.Event{Type: vad.SpeechStart, SpeechDuration: d.speech}, true
		}
		return vad.Event{}, false
	}

	d.silenceRun += d.step
	d.speechRun = 0
	if d.speaking && d.silenceRun >= d.opts.MinSilenceDuration {
		ev := vad.Event{
			Type:            vad.SpeechEnd,
			SpeechDuration:  max(0, d.speech-d.silenceRun),
			SilenceDuration: d.silenceRun,
		}
		d.speaking = false
		d.speech = 0
		return ev, true
	}
	return vad.Event{}, false
}

// Close implements vad.Detector.
func (d *detector) Close() error {
	if d.closed {
		return nil
	}
	d.closed = true
	return d.model.close()
}

// resampler converts interleaved PCM16 at one rate into mono float32 samples
// in [-1, 1] at another. Quality is deliberately low: averaging when going
// down, linear interpolation when going up. That is enough for voice activity
// detection, and upstream uses its lowest quality setting here too.
type resampler struct {
	in, out int

	// Downsampling: running sum of the input samples of the current output
	// sample.
	sum   float32
	count int
	phase int

	// Upsampling: previous input sample and position between it and the
	// next one, in units of 1/out.
	prev    float32
	primed  bool
	fracPos int
}

func newResampler(in, out int) resampler { return resampler{in: in, out: out} }

// push appends the converted samples to dst and returns it.
func (r *resampler) push(dst []float32, data []int16, channels int) []float32 {
	const scale = 1.0 / 32767
	for i := 0; i+channels <= len(data); i += channels {
		var s float32
		if channels == 1 {
			s = float32(data[i]) * scale
		} else {
			var acc int32
			for c := 0; c < channels; c++ {
				acc += int32(data[i+c])
			}
			s = float32(acc) / float32(channels) * scale
		}

		switch {
		case r.in == r.out:
			dst = append(dst, s)
		case r.in > r.out:
			r.sum += s
			r.count++
			r.phase += r.out
			if r.phase >= r.in {
				r.phase -= r.in
				dst = append(dst, r.sum/float32(r.count))
				r.sum, r.count = 0, 0
			}
		default:
			if !r.primed {
				r.prev, r.primed = s, true
				dst = append(dst, s)
				r.fracPos = r.in
				continue
			}
			// Output samples fall at multiples of in/out between prev (0)
			// and s (1), measured here in units of 1/out.
			for r.fracPos <= r.out {
				t := float32(r.fracPos) / float32(r.out)
				dst = append(dst, r.prev+(s-r.prev)*t)
				r.fracPos += r.in
			}
			r.fracPos -= r.out
			r.prev = s
		}
	}
	return dst
}
