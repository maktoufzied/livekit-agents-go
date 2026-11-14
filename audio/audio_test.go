package audio

import (
	"testing"
	"time"
)

func TestFrameBasics(t *testing.T) {
	format := Format{SampleRate: 16000, Channels: 2}
	f := Tone(format, 440, 20*time.Millisecond, 0.5)
	if f.Samples() != 320 || len(f.Data) != 640 || f.Duration() != 20*time.Millisecond || f.Format() != format {
		t.Fatalf("unexpected frame: samples=%d len=%d dur=%v", f.Samples(), len(f.Data), f.Duration())
	}
	back := FromBytesLE(f.BytesLE(), format)
	for i := range f.Data {
		if back.Data[i] != f.Data[i] {
			t.Fatalf("byte round trip differs at %d", i)
		}
	}
	if db := DBFS(f); db > -8 || db < -10 { // 0.5 amplitude sine is about -9 dBFS
		t.Fatalf("DBFS = %v", db)
	}
	if DBFS(Silence(format, time.Second)) != -100 || RMS(Frame{}) != 0 {
		t.Fatal("silence level")
	}
	if (Format{}).Valid() || (Frame{}).Duration() != 0 || (Frame{}).Samples() != 0 {
		t.Fatal("zero values must be safe")
	}
}

func TestChunker(t *testing.T) {
	format := Format{SampleRate: 16000, Channels: 1}
	c := NewChunker(format, 20*time.Millisecond) // 640 bytes per frame
	src := Tone(format, 300, 50*time.Millisecond, 0.5).BytesLE()

	var frames []Frame
	// Feed odd-sized pieces, including one that splits a sample.
	for _, n := range []int{1, 333, 700, len(src) - 1034} {
		frames = append(frames, c.Write(src[:n])...)
		src = src[n:]
	}
	if len(frames) != 2 {
		t.Fatalf("got %d full frames, want 2", len(frames))
	}
	for _, f := range frames {
		if f.Duration() != 20*time.Millisecond {
			t.Fatalf("frame duration %v", f.Duration())
		}
	}
	rest, ok := c.Flush()
	if !ok || rest.Duration() != 10*time.Millisecond {
		t.Fatalf("flush: ok=%v dur=%v", ok, rest.Duration())
	}
	if _, ok := c.Flush(); ok {
		t.Fatal("second flush must be empty")
	}
	c.Write([]byte{1}) // a dangling half sample is dropped
	if _, ok := c.Flush(); ok {
		t.Fatal("half sample must not produce a frame")
	}
}
