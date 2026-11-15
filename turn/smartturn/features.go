package smartturn

import (
	"math"
	"math/cmplx"
)

// The model input is the log-mel spectrogram Whisper uses, computed over the
// last 8 seconds of audio. The steps and constants follow the reference
// inference code of Smart Turn (WhisperFeatureExtractor with chunk_length=8
// and do_normalize=True).
const (
	sampleRate = 16000
	maxSeconds = 8
	numSamples = maxSeconds * sampleRate // 128000
	fftSize    = 400
	hopSize    = 160
	numBins    = fftSize/2 + 1 // 201
	numMels    = 80
	numFrames  = numSamples / hopSize // 800
	featureLen = numMels * numFrames
)

// melFilter is one triangular filter: weights for bins first, first+1, ...
type melFilter struct {
	first   int
	weights []float64
}

var (
	hann    = hannWindow()
	filters = melFilters()
	twiddle = twiddles()
)

// hannWindow is the periodic Hann window.
func hannWindow() []float64 {
	w := make([]float64, fftSize)
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/fftSize)
	}
	return w
}

func twiddles() []complex128 {
	t := make([]complex128, fftSize)
	for i := range t {
		t[i] = cmplx.Rect(1, -2*math.Pi*float64(i)/fftSize)
	}
	return t
}

// Slaney mel scale: linear below 1 kHz, logarithmic above.
func hzToMel(f float64) float64 {
	const minLogHz, minLogMel = 1000.0, 15.0
	if f < minLogHz {
		return 3 * f / 200
	}
	return minLogMel + math.Log(f/minLogHz)*27/math.Log(6.4)
}

func melToHz(m float64) float64 {
	const minLogHz, minLogMel = 1000.0, 15.0
	if m < minLogMel {
		return 200 * m / 3
	}
	return minLogHz * math.Exp(math.Log(6.4)/27*(m-minLogMel))
}

// melFilters builds the 80 triangular filters between 0 and 8 kHz, each
// scaled to unit area (Slaney normalization).
func melFilters() []melFilter {
	lo, hi := hzToMel(0), hzToMel(sampleRate/2)
	edges := make([]float64, numMels+2)
	for i := range edges {
		edges[i] = melToHz(lo + (hi-lo)*float64(i)/float64(numMels+1))
	}
	out := make([]melFilter, numMels)
	for m := range out {
		left, center, right := edges[m], edges[m+1], edges[m+2]
		norm := 2 / (right - left)
		f := melFilter{first: -1}
		for bin := 0; bin < numBins; bin++ {
			freq := float64(bin) * (sampleRate / 2) / float64(numBins-1)
			w := math.Min((freq-left)/(center-left), (right-freq)/(right-center))
			if w <= 0 {
				if f.first >= 0 {
					break
				}
				continue
			}
			if f.first < 0 {
				f.first = bin
			}
			f.weights = append(f.weights, w*norm)
		}
		if f.first < 0 {
			f.first = 0
		}
		out[m] = f
	}
	return out
}

// extractor computes features. It owns its scratch buffers and is not safe
// for concurrent use.
type extractor struct {
	audio   []float64 // normalized samples, numSamples long
	frame   []complex128
	scratch []complex128
	power   [numBins]float64
	mel     [numMels]float64
}

func newExtractor() *extractor {
	return &extractor{
		audio:   make([]float64, numSamples),
		frame:   make([]complex128, fftSize),
		scratch: make([]complex128, fftSize),
	}
}

// features writes the model input for samples (mono, 16 kHz, in [-1, 1])
// into dst, which must hold featureLen values laid out as [mel][frame].
// Only the last 8 seconds are used; shorter audio is preceded by silence.
func (e *extractor) features(dst []float32, samples []float32) {
	if len(samples) > numSamples {
		samples = samples[len(samples)-numSamples:]
	}
	pad := numSamples - len(samples)
	a := e.audio
	clear(a[:pad])
	for i, s := range samples {
		a[pad+i] = float64(s)
	}

	// Zero mean and unit variance over the whole window, padding included.
	var mean float64
	for _, v := range a {
		mean += v
	}
	mean /= numSamples
	var variance float64
	for _, v := range a {
		variance += (v - mean) * (v - mean)
	}
	scale := 1 / math.Sqrt(variance/numSamples+1e-7)
	for i, v := range a {
		a[i] = (v - mean) * scale
	}

	maxVal := math.Inf(-1)
	for t := 0; t < numFrames; t++ {
		// Frames that lie entirely in the leading padding are identical to
		// frame 0: keep its values instead of computing them again.
		if t == 0 || t*hopSize+fftSize/2 > pad {
			e.melFrame(t)
		}
		for m, v := range e.mel {
			v = math.Log10(math.Max(v, 1e-10))
			dst[m*numFrames+t] = float32(v)
			maxVal = math.Max(maxVal, v)
		}
	}
	// Whisper's dynamic range compression and scaling.
	floor := float32(maxVal - 8)
	for i, v := range dst[:featureLen] {
		dst[i] = (max(v, floor) + 4) / 4
	}
}

// melFrame computes the mel energies of frame t into e.mel. Frames are
// centered: frame t covers samples [t*hop - fft/2, t*hop + fft/2), with the
// signal reflected at both ends.
func (e *extractor) melFrame(t int) {
	start := t*hopSize - fftSize/2
	for i := 0; i < fftSize; i++ {
		j := start + i
		if j < 0 {
			j = -j
		} else if j >= numSamples {
			j = 2*(numSamples-1) - j
		}
		e.frame[i] = complex(e.audio[j]*hann[i], 0)
	}
	fft(e.frame, e.scratch, fftSize, 1)
	for k := range e.power {
		c := e.frame[k]
		e.power[k] = real(c)*real(c) + imag(c)*imag(c)
	}
	for m, f := range filters {
		var sum float64
		for i, w := range f.weights {
			sum += w * e.power[f.first+i]
		}
		e.mel[m] = sum
	}
}

// fft transforms x[0:n] in place. n must divide fftSize and have only the
// prime factors 2 and 5 (400 = 2^4 * 5^2); step is fftSize/n. scratch must
// hold n values.
func fft(x, scratch []complex128, n, step int) {
	if n == 1 {
		return
	}
	p := 2
	if n%2 != 0 {
		p = 5
	}
	m := n / p
	// De-interleave into p sub-sequences, transform each.
	for r := 0; r < p; r++ {
		for i := 0; i < m; i++ {
			scratch[r*m+i] = x[i*p+r]
		}
	}
	copy(x[:n], scratch[:n])
	for r := 0; r < p; r++ {
		fft(x[r*m:(r+1)*m], scratch[:m], m, step*p)
	}
	// Combine: X[k] = sum over r of w^(r*k) * Y_r[k mod m].
	for k := 0; k < n; k++ {
		km := k % m
		sum := x[km]
		for r := 1; r < p; r++ {
			sum += twiddle[(r*k*step)%fftSize] * x[r*m+km]
		}
		scratch[k] = sum
	}
	copy(x[:n], scratch[:n])
}
