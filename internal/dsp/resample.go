// Package dsp holds the small signal-processing helpers shared by the speech
// models.
package dsp

// Resampler converts interleaved PCM16 at one rate into mono float32 samples
// in [-1, 1] at another. Quality is deliberately low: averaging when going
// down, linear interpolation when going up. That is enough for the speech
// models, which are fed 16 kHz audio in the default room setup anyway.
type Resampler struct {
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

// NewResampler returns a resampler from in to out Hz.
func NewResampler(in, out int) Resampler { return Resampler{in: in, out: out} }

// Push appends the converted samples to dst and returns it.
func (r *Resampler) Push(dst []float32, data []int16, channels int) []float32 {
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
