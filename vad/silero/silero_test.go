package silero

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/vad"
)

// The clip is mono PCM16 at 16 kHz: one second of digital silence, a
// synthesized sentence and 1.5 seconds of digital silence. The sentence is
// "Hello." (1.38 s to 1.85 s), a pause, then "This is a short test of voice
// activity detection." (2.24 s to 4.8 s). The times were read from the
// signal level.
const (
	clipSpeechStart = 1376 * time.Millisecond
	clipSpeechEnd   = 4800 * time.Millisecond
	// A stretch with continuous speech.
	clipVoicedFrom = 2400 * time.Millisecond
	clipVoicedTo   = 4700 * time.Millisecond
)

// load returns a VAD, or skips the test when ONNX Runtime is not installed.
// Set SILERO_REQUIRED=1 to fail instead, as CI does.
func load(t testing.TB, opts Options) *VAD {
	t.Helper()
	v, err := New(opts)
	if errors.Is(err, ErrRuntimeUnavailable) && os.Getenv("SILERO_REQUIRED") != "1" {
		t.Skipf("ONNX Runtime not installed: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { v.Close() })
	return v
}

func clip(t testing.TB) audio.Frame {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "speech_16k.pcm"))
	if err != nil {
		t.Fatal(err)
	}
	return audio.FromBytesLE(raw, audio.Format{SampleRate: 16000, Channels: 1})
}

type timedEvent struct {
	vad.Event
	at time.Duration // position in the audio when the event was emitted
}

// run feeds f to a new detector in 20 ms frames.
func run(t testing.TB, v *VAD, f audio.Frame) []timedEvent {
	t.Helper()
	d, err := v.NewDetector()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	per := f.SampleRate / 50 * f.Channels
	var out []timedEvent
	for i := 0; i < len(f.Data); i += per {
		end := min(i+per, len(f.Data))
		events, err := d.Process(audio.Frame{Data: f.Data[i:end], SampleRate: f.SampleRate, Channels: f.Channels})
		if err != nil {
			t.Fatal(err)
		}
		at := time.Duration(end/f.Channels) * time.Second / time.Duration(f.SampleRate)
		for _, ev := range events {
			out = append(out, timedEvent{ev, at})
		}
	}
	return out
}

func checkClipEvents(t *testing.T, events []timedEvent, minSilence time.Duration) {
	t.Helper()
	if len(events) == 0 || events[0].Type != vad.SpeechStart {
		t.Fatalf("events = %+v, want a SpeechStart first", events)
	}
	last := events[len(events)-1]
	if last.Type != vad.SpeechEnd {
		t.Fatalf("events = %+v, want a SpeechEnd last", events)
	}
	for i, ev := range events {
		if want := []vad.EventType{vad.SpeechStart, vad.SpeechEnd}[i%2]; ev.Type != want {
			t.Fatalf("event %d is %v: starts and ends must alternate: %+v", i, ev.Type, events)
		}
	}
	// The start is detected shortly after the first word, never before it.
	if start := events[0].at; start < clipSpeechStart || start > clipSpeechStart+150*time.Millisecond {
		t.Errorf("speech start at %v, want within 150ms after %v", start, clipSpeechStart)
	}
	// The end is confirmed minSilence after the last word.
	if end := last.at; end < clipSpeechEnd+minSilence || end > clipSpeechEnd+minSilence+150*time.Millisecond {
		t.Errorf("speech end at %v, want about %v", end, clipSpeechEnd+minSilence)
	}
	if last.SilenceDuration < minSilence {
		t.Errorf("silence duration = %v, want at least %v", last.SilenceDuration, minSilence)
	}
}

func TestDetectsSpeech(t *testing.T) {
	v := load(t, Options{})
	events := run(t, v, clip(t))
	t.Logf("events: %+v", events)
	checkClipEvents(t, events, 550*time.Millisecond)
}

// The same clip at other input formats goes through the resampler.
func TestDetectsSpeechResampled(t *testing.T) {
	src := clip(t)
	upsample := func(factor, channels int) audio.Frame {
		out := make([]int16, 0, len(src.Data)*factor*channels)
		for i, s := range src.Data {
			next := s
			if i+1 < len(src.Data) {
				next = src.Data[i+1]
			}
			for k := 0; k < factor; k++ {
				v := int16(int(s) + (int(next)-int(s))*k/factor)
				for c := 0; c < channels; c++ {
					out = append(out, v)
				}
			}
		}
		return audio.Frame{Data: out, SampleRate: src.SampleRate * factor, Channels: channels}
	}
	downsample := func() audio.Frame {
		out := make([]int16, 0, len(src.Data)/2)
		for i := 0; i+1 < len(src.Data); i += 2 {
			out = append(out, int16((int(src.Data[i])+int(src.Data[i+1]))/2))
		}
		return audio.Frame{Data: out, SampleRate: 8000, Channels: 1}
	}

	t.Run("48k stereo input", func(t *testing.T) {
		checkClipEvents(t, run(t, load(t, Options{}), upsample(3, 2)), 550*time.Millisecond)
	})
	t.Run("8k input, 16k model", func(t *testing.T) {
		checkClipEvents(t, run(t, load(t, Options{}), downsample()), 550*time.Millisecond)
	})
	t.Run("8k input, 8k model", func(t *testing.T) {
		checkClipEvents(t, run(t, load(t, Options{SampleRate: 8000}), downsample()), 550*time.Millisecond)
	})
}

// Sounds that are not speech must not be reported, however loud. The energy
// detector reports all of these.
func TestIgnoresNonSpeech(t *testing.T) {
	v := load(t, Options{})
	format := audio.Format{SampleRate: 16000, Channels: 1}
	const length = 4 * time.Second

	rng := rand.New(rand.NewSource(1))
	noise := audio.Silence(format, length)
	for i := range noise.Data {
		noise.Data[i] = int16(rng.Intn(16000) - 8000)
	}
	// Clicks: 5 ms bursts every 250 ms, like a keyboard.
	clicks := audio.Silence(format, length)
	for start := 0; start < len(clicks.Data); start += 4000 {
		for i := start; i < start+80 && i < len(clicks.Data); i++ {
			clicks.Data[i] = int16(rng.Intn(40000) - 20000)
		}
	}

	for name, f := range map[string]audio.Frame{
		"silence":     audio.Silence(format, length),
		"tone":        audio.Tone(format, 440, length, 0.5),
		"white noise": noise,
		"clicks":      clicks,
	} {
		t.Run(name, func(t *testing.T) {
			if events := run(t, v, f); len(events) != 0 {
				t.Errorf("events = %+v, want none", events)
			}
			// Sanity check of the premise for the loud ones.
			if name != "silence" && audio.DBFS(f) < -40 {
				t.Fatalf("test signal is too quiet: %v dBFS", audio.DBFS(f))
			}
		})
	}
}

// Detectors of one VAD run concurrently and do not share stream state.
func TestConcurrentDetectors(t *testing.T) {
	v := load(t, Options{})
	speech := clip(t)
	want := run(t, v, speech)

	var wg sync.WaitGroup
	results := make([][]timedEvent, 8)
	for i := range results {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if i%2 == 1 {
				results[i] = run(t, v, audio.Silence(speech.Format(), 6*time.Second))
				return
			}
			results[i] = run(t, v, speech)
		}()
	}
	wg.Wait()
	for i, got := range results {
		if i%2 == 1 {
			if len(got) != 0 {
				t.Errorf("detector %d on silence: %+v", i, got)
			}
			continue
		}
		if len(got) != len(want) {
			t.Fatalf("detector %d: %d events, want %d", i, len(got), len(want))
		}
		for j := range got {
			if got[j] != want[j] {
				t.Errorf("detector %d event %d = %+v, want %+v", i, j, got[j], want[j])
			}
		}
	}
}

// The raw model output is close to 0 on silence and close to 1 on speech.
func TestModelOutput(t *testing.T) {
	v := load(t, Options{})
	m, err := newOnnxModel(v.session, 16000)
	if err != nil {
		t.Fatal(err)
	}
	defer m.close()

	speech := clip(t)
	window := make([]float32, 512)
	var silenceMax, speechSum float64
	speechWindows := 0
	for i := 0; i+512 <= len(speech.Data); i += 512 {
		for j := range window {
			window[j] = float32(speech.Data[i+j]) / 32767
		}
		p, err := m.infer(window)
		if err != nil {
			t.Fatal(err)
		}
		if p < 0 || p > 1 || math.IsNaN(float64(p)) {
			t.Fatalf("probability %v out of range", p)
		}
		at := time.Duration(i) * time.Second / 16000
		switch {
		case at < 900*time.Millisecond:
			silenceMax = max(silenceMax, float64(p))
		case at >= clipVoicedFrom && at < clipVoicedTo:
			speechSum += float64(p)
			speechWindows++
		}
	}
	if silenceMax > 0.1 {
		t.Errorf("highest probability on silence = %.3f, want below 0.1", silenceMax)
	}
	if mean := speechSum / float64(speechWindows); mean < 0.95 {
		t.Errorf("mean probability on speech = %.3f, want above 0.95", mean)
	}
}

func TestClose(t *testing.T) {
	v := load(t, Options{})
	d, err := v.NewDetector()
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Close(); err != nil {
		t.Fatal(err)
	}
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}
	if err := v.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := v.NewDetector(); !errors.Is(err, ErrClosed) {
		t.Fatalf("NewDetector after Close = %v, want ErrClosed", err)
	}
}

func BenchmarkProcess(b *testing.B) {
	v := load(b, Options{})
	d, err := v.NewDetector()
	if err != nil {
		b.Fatal(err)
	}
	defer d.Close()
	speech := clip(b)
	const per = 320 // 20 ms
	frames := len(speech.Data) / per

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := (i % frames) * per
		if _, err := d.Process(audio.Frame{Data: speech.Data[start : start+per], SampleRate: 16000, Channels: 1}); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/20e6*100, "%realtime")
}

// refEvent is a speech boundary as the upstream plugin reports it.
type refEvent struct {
	kind            string // "start" or "end"
	samples         int    // position at the model rate
	speech, silence float64
}

// reference is the output of testdata/reference.py for one clip.
type reference struct {
	raw      []float64 // model output per window; 16 kHz clips only
	smoothed []float64 // filtered probability per window
	events   []refEvent
}

func readReference(t testing.TB, path string) reference {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var ref reference
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		fields := strings.Fields(line)
		switch fields[0] {
		case "raw":
			p, err := strconv.ParseFloat(fields[1], 64)
			if err != nil {
				t.Fatal(err)
			}
			ref.raw = append(ref.raw, p)
		case "ev":
			var ev refEvent
			var prob float64
			if _, err := fmt.Sscan(strings.Join(fields[1:], " "), &ev.kind, &ev.samples, &ev.speech, &ev.silence, &prob); err != nil {
				t.Fatal(err)
			}
			if ev.kind == "inference" {
				ref.smoothed = append(ref.smoothed, prob)
			} else {
				ref.events = append(ref.events, ev)
			}
		}
	}
	if len(ref.smoothed) == 0 || len(ref.events) == 0 {
		t.Fatalf("%s: %d windows, %d events", path, len(ref.smoothed), len(ref.events))
	}
	return ref
}

// analyze runs a detector over mono PCM16 in 20 ms frames, as the reference
// script does, and returns the same quantities.
func analyze(t testing.TB, v *VAD, pcm []int16, rate int) reference {
	t.Helper()
	m, err := newOnnxModel(v.session, v.opts.SampleRate)
	if err != nil {
		t.Fatal(err)
	}
	rec := &recordingModel{model: m}
	d := newDetector(v.opts, rec)
	defer d.Close()

	var got reference
	per := rate / 50
	for i := 0; i < len(pcm); i += per {
		before := len(rec.probs)
		events, err := d.Process(audio.Frame{Data: pcm[i:min(i+per, len(pcm))], SampleRate: rate, Channels: 1})
		if err != nil {
			t.Fatal(err)
		}
		// At most one window completes per 20 ms frame, except when
		// upsampling 8 kHz input; the smoothed value is then the last one.
		for n := before; n < len(rec.probs); n++ {
			got.smoothed = append(got.smoothed, d.prob)
		}
		for _, ev := range events {
			kind := "start"
			if ev.Type == vad.SpeechEnd {
				kind = "end"
			}
			got.events = append(got.events, refEvent{
				kind: kind, samples: len(rec.probs) * d.window,
				speech: ev.SpeechDuration.Seconds(), silence: ev.SilenceDuration.Seconds(),
			})
		}
	}
	for _, p := range rec.probs {
		got.raw = append(got.raw, float64(p))
	}
	return got
}

type recordingModel struct {
	model
	probs []float32
}

func (r *recordingModel) infer(window []float32) (float32, error) {
	p, err := r.model.infer(window)
	r.probs = append(r.probs, p)
	return p, err
}

func maxDiff(a, b []float64) float64 {
	var worst float64
	for i := range min(len(a), len(b)) {
		worst = max(worst, math.Abs(a[i]-b[i]))
	}
	return worst
}

func readPCM(t testing.TB, path string) []int16 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return audio.FromBytesLE(raw, audio.Format{SampleRate: 16000, Channels: 1}).Data
}

// The detector reproduces the upstream Python plugin: the raw model output
// of every window, the smoothed probability and the speech boundaries. The
// references were produced by testdata/reference.py with
// livekit-plugins-silero 1.8.12. Both clips are 16 kHz, so neither side
// resamples and the comparison is exact up to the 6 decimals of the files.
//
//   - speech_16k: a synthesized sentence between stretches of digital silence.
//   - apollo11_16k: Neil Armstrong on the lunar surface, 1969 (NASA, public
//     domain): a real voice over radio noise, with pauses and a beep.
func TestMatchesUpstreamPlugin(t *testing.T) {
	const tolerance = 1e-3
	for _, name := range []string{"speech_16k", "apollo11_16k"} {
		t.Run(name, func(t *testing.T) {
			want := readReference(t, filepath.Join("testdata", name+".reference.txt"))
			got := analyze(t, load(t, Options{}), readPCM(t, filepath.Join("testdata", name+".pcm")), 16000)

			if len(got.raw) != len(want.raw) || len(got.smoothed) != len(want.smoothed) {
				t.Fatalf("windows = %d, want %d", len(got.raw), len(want.raw))
			}
			worstRaw, worstSmoothed := maxDiff(got.raw, want.raw), maxDiff(got.smoothed, want.smoothed)
			t.Logf("%d windows, %d events: largest difference %.2g (model output), %.2g (smoothed)",
				len(want.raw), len(want.events), worstRaw, worstSmoothed)
			if worstRaw > tolerance {
				t.Errorf("model output differs from upstream by up to %.4f", worstRaw)
			}
			if worstSmoothed > tolerance {
				t.Errorf("smoothed probability differs from upstream by up to %.4f", worstSmoothed)
			}
			if len(got.events) != len(want.events) {
				t.Fatalf("events = %+v, want %+v", got.events, want.events)
			}
			for i, w := range want.events {
				g := got.events[i]
				if g.kind != w.kind || g.samples != w.samples ||
					math.Abs(g.speech-w.speech) > 1e-6 || math.Abs(g.silence-w.silence) > 1e-6 {
					t.Errorf("event %d = %+v, want %+v", i, g, w)
				}
			}
		})
	}
}
