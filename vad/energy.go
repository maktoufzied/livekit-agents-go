package vad

import (
	"time"

	"github.com/maktoufzied/livekit-agents-go/audio"
)

// EnergyOptions configures the energy detector. Zero values select defaults.
type EnergyOptions struct {
	// ThresholdDB is the margin above the tracked noise floor, in dB, a
	// frame must exceed to count as speech. Default 12.
	ThresholdDB float64
	// MinLevelDBFS is the absolute level below which a frame is never
	// speech, whatever the noise floor. Default -50.
	MinLevelDBFS float64
	// MinSpeechDuration is the speech needed before SpeechStart. Default 60ms.
	MinSpeechDuration time.Duration
	// MinSilenceDuration is the silence needed before SpeechEnd. Default 500ms.
	MinSilenceDuration time.Duration
}

func (o EnergyOptions) withDefaults() EnergyOptions {
	if o.ThresholdDB == 0 {
		o.ThresholdDB = 12
	}
	if o.MinLevelDBFS == 0 {
		o.MinLevelDBFS = -50
	}
	if o.MinSpeechDuration == 0 {
		o.MinSpeechDuration = 60 * time.Millisecond
	}
	if o.MinSilenceDuration == 0 {
		o.MinSilenceDuration = 500 * time.Millisecond
	}
	return o
}

// Energy is a pure-Go voice activity detector based on frame energy over an
// adaptive noise floor. It needs no model and no native library. It is less
// robust than model-based detectors in noisy rooms: music or keyboard noise
// can register as speech.
type Energy struct{ opts EnergyOptions }

// NewEnergy returns an energy-based VAD.
func NewEnergy(opts EnergyOptions) *Energy { return &Energy{opts: opts.withDefaults()} }

// NewDetector implements VAD.
func (e *Energy) NewDetector() (Detector, error) {
	return &energyDetector{opts: e.opts, noiseDB: -70}, nil
}

type energyDetector struct {
	opts     EnergyOptions
	noiseDB  float64
	speaking bool
	speech   time.Duration // candidate or confirmed speech length
	silence  time.Duration // trailing silence while speaking
}

func (d *energyDetector) Process(f audio.Frame) ([]Event, error) {
	dur := f.Duration()
	if dur == 0 {
		return nil, nil
	}
	level := audio.DBFS(f)
	voiced := level > d.opts.MinLevelDBFS && level > d.noiseDB+d.opts.ThresholdDB

	// Track the noise floor: fall quickly, rise slowly, and never learn
	// from frames classified as speech.
	if !voiced {
		alpha := 0.05
		if level < d.noiseDB {
			alpha = 0.5
		}
		d.noiseDB += alpha * (level - d.noiseDB)
		if d.noiseDB < -90 {
			d.noiseDB = -90
		}
	}

	var events []Event
	switch {
	case !d.speaking && voiced:
		d.speech += dur
		if d.speech >= d.opts.MinSpeechDuration {
			d.speaking = true
			d.silence = 0
			events = append(events, Event{Type: SpeechStart, SpeechDuration: d.speech})
		}
	case !d.speaking:
		d.speech = 0
	case voiced:
		d.speech += dur + d.silence
		d.silence = 0
	default:
		d.silence += dur
		if d.silence >= d.opts.MinSilenceDuration {
			events = append(events, Event{Type: SpeechEnd, SpeechDuration: d.speech, SilenceDuration: d.silence})
			d.speaking = false
			d.speech = 0
			d.silence = 0
		}
	}
	return events, nil
}

func (d *energyDetector) Close() error { return nil }
