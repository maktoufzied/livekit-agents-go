package smartturn

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/audio"
)

// load returns a detector, or skips the test when ONNX Runtime is not
// installed. Set SMARTTURN_REQUIRED=1 to fail instead, as CI does.
func load(t testing.TB, opts Options) *Detector {
	t.Helper()
	d, err := New(opts)
	if errors.Is(err, ErrRuntimeUnavailable) && os.Getenv("SMARTTURN_REQUIRED") != "1" {
		t.Skipf("ONNX Runtime not installed: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { d.Close() })
	return d
}

// readClip reads a test clip, mono PCM16 at 16 kHz. Two are synthesized
// sentences from the examples of the Smart Turn documentation: unfinished_16k
// says "I can't seem to, um..." and finished_16k says "I can't seem to, um,
// find the return label." The others are those of vad/silero.
func readClip(t testing.TB, name string) []int16 {
	t.Helper()
	path := filepath.Join("testdata", name+".pcm")
	if _, err := os.Stat(path); err != nil {
		path = filepath.Join("..", "..", "vad", "silero", "testdata", name+".pcm")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]int16, len(raw)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(raw[2*i:]))
	}
	return out
}

func first(clip []int16, seconds float64) audio.Frame {
	n := min(len(clip), int(math.Round(seconds*sampleRate)))
	return audio.Frame{Data: clip[:n], SampleRate: sampleRate, Channels: 1}
}

// refCase is one prediction of the Python reference code.
type refCase struct {
	clip        string
	seconds     float64
	probability float64         // embedded model
	fp32        float64         // unquantized model
	features    map[int]float64 // sampled model input values
}

func readReference(t testing.TB) []*refCase {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", "reference.txt"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []*refCase
	num := func(s string) float64 {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		switch {
		case len(fields) == 4 && fields[0] == "p":
			out = append(out, &refCase{clip: fields[1], seconds: num(fields[2]), probability: num(fields[3]), features: map[int]float64{}})
		case len(fields) == 4 && fields[0] == "q":
			out[len(out)-1].fp32 = num(fields[3])
		case len(fields) == 5 && fields[0] == "f":
			out[len(out)-1].features[int(num(fields[3]))] = num(fields[4])
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("empty reference")
	}
	return out
}

func floats(pcm []int16) []float32 {
	out := make([]float32, len(pcm))
	for i, s := range pcm {
		out[i] = float32(s) / 32767
	}
	return out
}

// The model input must be the one the reference Python code computes. This
// needs no ONNX Runtime.
func TestFeaturesMatchReference(t *testing.T) {
	e := newExtractor()
	dst := make([]float32, featureLen)
	clips := map[string][]int16{}
	var worst float64
	for _, c := range readReference(t) {
		if clips[c.clip] == nil {
			clips[c.clip] = readClip(t, c.clip)
		}
		e.features(dst, floats(first(clips[c.clip], c.seconds).Data))
		if len(c.features) < 100 {
			t.Fatalf("%s %vs: only %d reference values", c.clip, c.seconds, len(c.features))
		}
		for i, want := range c.features {
			diff := math.Abs(float64(dst[i]) - want)
			worst = max(worst, diff)
			if diff > 1e-3 {
				t.Fatalf("%s %vs: input[%d] = %v, reference %v", c.clip, c.seconds, i, dst[i], want)
			}
		}
	}
	t.Logf("largest difference in the model input: %.2g", worst)
}

// The embedded model is quantized to 8 bits, which makes its output jump:
// the Python code itself moves by up to 0.2 when its input changes in the
// sixth decimal (see testdata/reference.py), and ONNX Runtime computes
// quantized models differently on each processor family.
//
// So on every platform the confident answers of the reference must be
// decided the same way. On arm64, where the reference was made, most
// predictions must also be equal and none may be far off.
func TestMatchesReference(t *testing.T) {
	d := load(t, Options{})
	clips := map[string][]int16{}
	cases := readReference(t)
	equal := 0
	sameArch := runtime.GOARCH == "arm64"
	for _, c := range cases {
		if clips[c.clip] == nil {
			clips[c.clip] = readClip(t, c.clip)
		}
		got, err := d.Predict(first(clips[c.clip], c.seconds))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%-13s %5.2fs  go %.6f  python %.6f", c.clip, c.seconds, got, c.probability)
		diff := math.Abs(got - c.probability)
		if diff <= 0.002 {
			equal++
		}
		if confident := c.probability < 0.1 || c.probability > 0.9; confident && (got >= 0.5) != (c.probability >= 0.5) {
			t.Errorf("%s %vs: probability %v decides differently from the reference %v", c.clip, c.seconds, got, c.probability)
		}
		if sameArch && diff > 0.25 {
			t.Errorf("%s %vs: probability %v, reference %v", c.clip, c.seconds, got, c.probability)
		}
	}
	t.Logf("%d of %d predictions equal to the reference (made on arm64, this is %s)", equal, len(cases), runtime.GOARCH)
	if sameArch && equal*4 < len(cases)*3 {
		t.Errorf("only %d of %d predictions equal to the reference", equal, len(cases))
	}
}

// With the unquantized model the probabilities must be those of the Python
// code, on every platform. The model is not in the repository: set
// SMARTTURN_FP32_MODEL to smart-turn-v3.2-gpu.onnx to run this, as CI does.
func TestMatchesReferenceUnquantized(t *testing.T) {
	path := os.Getenv("SMARTTURN_FP32_MODEL")
	if path == "" {
		t.Skip("SMARTTURN_FP32_MODEL not set")
	}
	model, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	d := load(t, Options{Model: model})
	clips := map[string][]int16{}
	var worst float64
	for _, c := range readReference(t) {
		if clips[c.clip] == nil {
			clips[c.clip] = readClip(t, c.clip)
		}
		got, err := d.Predict(first(clips[c.clip], c.seconds))
		if err != nil {
			t.Fatal(err)
		}
		diff := math.Abs(got - c.fp32)
		worst = max(worst, diff)
		t.Logf("%-13s %5.2fs  go %.6f  python %.6f", c.clip, c.seconds, got, c.fp32)
		if diff > 1e-3 {
			t.Errorf("%s %vs: probability %v, reference %v", c.clip, c.seconds, got, c.fp32)
		}
	}
	t.Logf("largest difference: %.2g", worst)
}

// An unfinished sentence gets a low probability and a finished one a high
// probability, with or without silence after the last word.
func TestFinishedAndUnfinished(t *testing.T) {
	d := load(t, Options{})
	for _, c := range []struct {
		clip     string
		finished bool
	}{
		{"unfinished_16k", false},
		{"finished_16k", true},
	} {
		clip := readClip(t, c.clip)
		for _, silence := range []time.Duration{0, 200 * time.Millisecond, time.Second} {
			turn := append(append([]int16(nil), clip...), make([]int16, silence*sampleRate/time.Second)...)
			p, err := d.Predict(audio.Frame{Data: turn, SampleRate: sampleRate, Channels: 1})
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%s + %v of silence: %.3f", c.clip, silence, p)
			if c.finished && p < 0.8 || !c.finished && p > 0.2 {
				t.Errorf("%s + %v of silence: probability %.3f, want finished=%v", c.clip, silence, p, c.finished)
			}
		}
	}

	// A sentence cut in the middle of a word is unfinished too, as long as
	// no silence follows the cut.
	clip := readClip(t, "speech_16k") // "Hello. This is a short test of voice activity detection."
	for seconds, finished := range map[float64]bool{2.5: false, 3.0: false, 4.0: false, 5.0: true} {
		p, err := d.Predict(first(clip, seconds))
		if err != nil {
			t.Fatal(err)
		}
		if (p >= 0.5) != finished {
			t.Errorf("first %.1fs: probability %.3f, want finished=%v", seconds, p, finished)
		}
	}
}

func whole(clip []int16) audio.Frame {
	return audio.Frame{Data: clip, SampleRate: sampleRate, Channels: 1}
}

// Other input formats go through the resampler and must agree with 16 kHz
// mono on what is finished.
func TestOtherFormats(t *testing.T) {
	d := load(t, Options{})
	stereo48k := func(src []int16) audio.Frame {
		out := make([]int16, 0, len(src)*6)
		for i, s := range src {
			next := s
			if i+1 < len(src) {
				next = src[i+1]
			}
			for k := 0; k < 3; k++ {
				v := int16(int(s) + (int(next)-int(s))*k/3)
				out = append(out, v, v)
			}
		}
		return audio.Frame{Data: out, SampleRate: 48000, Channels: 2}
	}
	mono8k := func(src []int16) audio.Frame {
		out := make([]int16, 0, len(src)/2)
		for i := 0; i+1 < len(src); i += 2 {
			out = append(out, int16((int(src[i])+int(src[i+1]))/2))
		}
		return audio.Frame{Data: out, SampleRate: 8000, Channels: 1}
	}
	for name, finished := range map[string]bool{"unfinished_16k": false, "finished_16k": true} {
		clip := readClip(t, name)
		for format, turn := range map[string]audio.Frame{"48 kHz stereo": stereo48k(clip), "8 kHz": mono8k(clip)} {
			p, err := d.Predict(turn)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%s at %s: %.3f", name, format, p)
			if (p >= 0.5) != finished {
				t.Errorf("%s at %s: probability %.3f, want finished=%v", name, format, p, finished)
			}
		}
	}
}

func TestTurnDetector(t *testing.T) {
	d := load(t, Options{})
	unfinished, finished := whole(readClip(t, "unfinished_16k")), whole(readClip(t, "finished_16k"))
	ctx := context.Background()

	p, err := d.EndOfTurnProbability(ctx, agents.TurnInput{Audio: unfinished})
	if err != nil || p >= 0.5 {
		t.Fatalf("unfinished sentence: p=%v err=%v", p, err)
	}
	p, err = d.EndOfTurnProbability(ctx, agents.TurnInput{Audio: finished})
	if err != nil || p < 0.5 {
		t.Fatalf("finished sentence: p=%v err=%v", p, err)
	}
	// No audio: nothing to wait for.
	if p, err := d.EndOfTurnProbability(ctx, agents.TurnInput{}); err != nil || p != 1 {
		t.Fatalf("no audio: p=%v err=%v", p, err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := d.EndOfTurnProbability(canceled, agents.TurnInput{Audio: unfinished}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled context: err=%v", err)
	}
	if _, err := d.Predict(audio.Frame{Data: make([]int16, 100)}); err == nil {
		t.Fatal("no error for audio without a format")
	}
}

func TestConcurrent(t *testing.T) {
	d := load(t, Options{})
	clip := readClip(t, "speech_16k")
	want := map[float64]float64{}
	for _, s := range []float64{3.0, 4.5, 6.3} {
		p, err := d.Predict(first(clip, s))
		if err != nil {
			t.Fatal(err)
		}
		want[s] = p
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for s, w := range want {
				p, err := d.Predict(first(clip, s))
				if err != nil {
					t.Error(err)
					return
				}
				if p != w {
					t.Errorf("first %.1fs: %v under concurrency, %v alone", s, p, w)
				}
			}
		}()
	}
	wg.Wait()
}

func TestClose(t *testing.T) {
	d := load(t, Options{})
	clip := readClip(t, "speech_16k")
	if _, err := d.Predict(first(clip, 3.0)); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := d.Predict(first(clip, 3.0)); !errors.Is(err, ErrClosed) {
		t.Fatalf("Predict after Close = %v, want ErrClosed", err)
	}
	if _, err := New(Options{Threads: -1}); err == nil {
		t.Fatal("no error for a negative thread count")
	}
}

func TestFFT(t *testing.T) {
	// Against the definition of the discrete Fourier transform.
	x := make([]complex128, fftSize)
	for i := range x {
		x[i] = complex(math.Sin(0.37*float64(i))+0.25*float64(i%7), 0)
	}
	want := make([]complex128, fftSize)
	for k := range want {
		for n, v := range x {
			angle := -2 * math.Pi * float64(k*n%fftSize) / fftSize
			want[k] += v * complex(math.Cos(angle), math.Sin(angle))
		}
	}
	fft(x, make([]complex128, fftSize), fftSize, 1)
	for k := range x {
		if d := x[k] - want[k]; math.Hypot(real(d), imag(d)) > 1e-8 {
			t.Fatalf("bin %d = %v, want %v", k, x[k], want[k])
		}
	}
}

func BenchmarkPredict(b *testing.B) {
	turn := first(readClip(b, "speech_16k"), 6.3)
	for _, threads := range []int{1, 2, 4} {
		b.Run(strconv.Itoa(threads)+" threads", func(b *testing.B) {
			d := load(b, Options{Threads: threads})
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := d.Predict(turn); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkFeatures(b *testing.B) {
	e := newExtractor()
	samples := floats(readClip(b, "apollo11_16k"))
	dst := make([]float32, featureLen)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.features(dst, samples)
	}
}
