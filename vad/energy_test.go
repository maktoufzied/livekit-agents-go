package vad

import (
	"testing"
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
)

func feed(t *testing.T, d Detector, f audio.Frame, n int) []Event {
	t.Helper()
	var out []Event
	for i := 0; i < n; i++ {
		ev, err := d.Process(f)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, ev...)
	}
	return out
}

func TestEnergyDetector(t *testing.T) {
	format := audio.Format{SampleRate: 16000, Channels: 1}
	frame := 20 * time.Millisecond
	silence := audio.Silence(format, frame)
	speech := audio.Tone(format, 300, frame, 0.3)
	noise := audio.Tone(format, 300, frame, 0.002) // about -57 dBFS

	d, _ := NewEnergy(EnergyOptions{MinSpeechDuration: 60 * time.Millisecond, MinSilenceDuration: 200 * time.Millisecond}).NewDetector()
	defer d.Close()

	if ev := feed(t, d, silence, 25); len(ev) != 0 {
		t.Fatalf("events on silence: %+v", ev)
	}
	if ev := feed(t, d, noise, 25); len(ev) != 0 {
		t.Fatalf("events on low noise: %+v", ev)
	}
	// A click shorter than MinSpeechDuration is ignored.
	if ev := append(feed(t, d, speech, 2), feed(t, d, silence, 5)...); len(ev) != 0 {
		t.Fatalf("events on a short click: %+v", ev)
	}
	ev := feed(t, d, speech, 3)
	if len(ev) != 1 || ev[0].Type != SpeechStart || ev[0].SpeechDuration != 60*time.Millisecond {
		t.Fatalf("want one SpeechStart after 60ms, got %+v", ev)
	}
	// A pause shorter than MinSilenceDuration does not end the utterance.
	if ev := append(feed(t, d, silence, 5), feed(t, d, speech, 10)...); len(ev) != 0 {
		t.Fatalf("events during a short pause: %+v", ev)
	}
	ev = feed(t, d, silence, 10)
	if len(ev) != 1 || ev[0].Type != SpeechEnd || ev[0].SilenceDuration != 200*time.Millisecond {
		t.Fatalf("want one SpeechEnd after 200ms of silence, got %+v", ev)
	}
	if ev[0].SpeechDuration != 360*time.Millisecond { // 60 + 100 pause + 200
		t.Fatalf("speech duration = %v", ev[0].SpeechDuration)
	}
	if ev := feed(t, d, silence, 25); len(ev) != 0 {
		t.Fatalf("events after the end: %+v", ev)
	}
}
