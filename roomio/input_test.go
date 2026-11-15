package roomio

import (
	"context"
	"testing"
	"time"

	"github.com/livekit/media-sdk"

	"github.com/maktoufzied/livekit-agents-go/audio"
)

// A browser stops sending audio while the user is silent. The input must
// keep the audio running at real time by inserting silence.
func TestInputFillsGapsWithSilence(t *testing.T) {
	format := audio.Format{SampleRate: 16000, Channels: 1}
	in := newInput(format, 64)
	ctx := context.Background()
	loud := make(media.PCM16Sample, 320) // 20 ms
	for i := range loud {
		loud[i] = 1000
	}

	// Nothing is inserted before the first packet.
	early, cancel := context.WithTimeout(ctx, 250*time.Millisecond)
	if _, err := in.ReadFrame(early); err == nil {
		t.Fatal("a frame was returned before any audio arrived")
	}
	cancel()

	// Ten packets, then the sender goes quiet for 600 ms.
	for i := 0; i < 10; i++ {
		if err := in.WriteSample(loud); err != nil {
			t.Fatal(err)
		}
	}
	start := time.Now()
	var speech, silence time.Duration
	for time.Since(start) < 600*time.Millisecond {
		f, err := in.ReadFrame(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if f.Format() != format {
			t.Fatalf("frame format = %v", f.Format())
		}
		if audio.RMS(f) > 0 {
			speech += f.Duration()
		} else {
			silence += f.Duration()
		}
	}
	if speech != 200*time.Millisecond {
		t.Fatalf("speech read = %v, want 200ms", speech)
	}
	// 600 ms passed and 200 ms were speech: about 400 ms of silence, less
	// the allowed lag.
	if silence < 250*time.Millisecond || silence > 450*time.Millisecond {
		t.Fatalf("silence inserted = %v, want about 400ms minus %v", silence, inputGap)
	}

	// Audio that arrives again is returned at once, not after more silence.
	if err := in.WriteSample(loud); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(200 * time.Millisecond)
	for {
		f, err := in.ReadFrame(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if audio.RMS(f) > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("resumed audio not returned")
		}
	}

	// A steady stream gets no silence inserted.
	silence = 0
	for i := 0; i < 25; i++ {
		if err := in.WriteSample(loud); err != nil {
			t.Fatal(err)
		}
		f, err := in.ReadFrame(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if audio.RMS(f) == 0 {
			silence += f.Duration()
		}
		time.Sleep(20 * time.Millisecond)
	}
	if silence > 40*time.Millisecond {
		t.Fatalf("%v of silence inserted into a steady stream", silence)
	}
}
