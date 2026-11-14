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

func TestLibraryCandidates(t *testing.T) {
	t.Setenv(LibraryEnv, "/from/env/libonnxruntime.so")
	if got := libraryCandidates("/explicit.so"); len(got) != 1 || got[0] != "/explicit.so" {
		t.Errorf("explicit path: %v", got)
	}
	if got := libraryCandidates(""); len(got) != 1 || got[0] != "/from/env/libonnxruntime.so" {
		t.Errorf("environment: %v", got)
	}
	t.Setenv(LibraryEnv, "")
	if got := libraryCandidates(""); len(got) == 0 {
		t.Error("no default candidate")
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

// The detector reproduces the upstream Python plugin on the test clip: the
// raw model output of every window, the smoothed probability and the speech
// boundaries. testdata/reference.txt was produced by testdata/reference.py
// with livekit-plugins-silero 1.8.12.
func TestMatchesUpstreamPlugin(t *testing.T) {
	const tolerance = 1e-3

	raw, err := os.ReadFile(filepath.Join("testdata", "reference.txt"))
	if err != nil {
		t.Fatal(err)
	}
	type refEvent struct {
		kind            string
		samples         int
		speech, silence float64
		prob            float64
	}
	var wantRaw, wantSmoothed []float64
	var wantEvents []refEvent
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		fields := strings.Fields(line)
		switch fields[0] {
		case "raw":
			p, err := strconv.ParseFloat(fields[1], 64)
			if err != nil {
				t.Fatal(err)
			}
			wantRaw = append(wantRaw, p)
		case "ev":
			var ev refEvent
			if _, err := fmt.Sscan(strings.Join(fields[1:], " "), &ev.kind, &ev.samples, &ev.speech, &ev.silence, &ev.prob); err != nil {
				t.Fatal(err)
			}
			if ev.kind == "inference" {
				wantSmoothed = append(wantSmoothed, ev.prob)
			} else {
				wantEvents = append(wantEvents, ev)
			}
		}
	}
	if len(wantRaw) == 0 || len(wantSmoothed) != len(wantRaw) || len(wantEvents) == 0 {
		t.Fatalf("reference: %d raw, %d smoothed, %d events", len(wantRaw), len(wantSmoothed), len(wantEvents))
	}

	v := load(t, Options{})
	m, err := newOnnxModel(v.session, 16000)
	if err != nil {
		t.Fatal(err)
	}
	// Record what the model returns while the detector drives it.
	rec := &recordingModel{model: m}
	d := newDetector(v.opts, rec)
	defer d.Close()

	speech := clip(t)
	var gotSmoothed []float64
	var gotEvents []refEvent
	const per = 320 // 20 ms frames, as the reference script feeds them
	for i := 0; i < len(speech.Data); i += per {
		before := len(rec.probs)
		events, err := d.Process(audio.Frame{Data: speech.Data[i:min(i+per, len(speech.Data))], SampleRate: 16000, Channels: 1})
		if err != nil {
			t.Fatal(err)
		}
		if len(rec.probs) > before {
			gotSmoothed = append(gotSmoothed, d.prob)
		}
		for _, ev := range events {
			kind := "start"
			if ev.Type == vad.SpeechEnd {
				kind = "end"
			}
			gotEvents = append(gotEvents, refEvent{
				kind: kind, samples: len(rec.probs) * 512,
				speech: ev.SpeechDuration.Seconds(), silence: ev.SilenceDuration.Seconds(),
			})
		}
	}

	if len(rec.probs) != len(wantRaw) {
		t.Fatalf("windows = %d, want %d", len(rec.probs), len(wantRaw))
	}
	var worstRaw, worstSmoothed float64
	for i := range wantRaw {
		worstRaw = max(worstRaw, math.Abs(float64(rec.probs[i])-wantRaw[i]))
		worstSmoothed = max(worstSmoothed, math.Abs(gotSmoothed[i]-wantSmoothed[i]))
	}
	t.Logf("%d windows: largest difference %.2g (model output), %.2g (smoothed)", len(wantRaw), worstRaw, worstSmoothed)
	if worstRaw > tolerance {
		t.Errorf("model output differs from upstream by up to %.4f", worstRaw)
	}
	if worstSmoothed > tolerance {
		t.Errorf("smoothed probability differs from upstream by up to %.4f", worstSmoothed)
	}

	if len(gotEvents) != len(wantEvents) {
		t.Fatalf("events = %+v, want %+v", gotEvents, wantEvents)
	}
	for i, want := range wantEvents {
		got := gotEvents[i]
		if got.kind != want.kind || got.samples != want.samples ||
			math.Abs(got.speech-want.speech) > 1e-6 || math.Abs(got.silence-want.silence) > 1e-6 {
			t.Errorf("event %d = %+v, want %+v", i, got, want)
		}
	}
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
