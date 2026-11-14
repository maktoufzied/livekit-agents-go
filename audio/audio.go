// Package audio defines the PCM frame type shared by every stage of the
// voice pipeline.
//
// All audio inside the SDK is signed 16-bit linear PCM, interleaved when
// multi-channel. Encoding (Opus), decoding and resampling happen at the
// transport boundary (see package roomio), never in the orchestration core.
package audio

import (
	"encoding/binary"
	"math"
	"time"
)

// Format describes a PCM stream.
type Format struct {
	SampleRate int // Hz
	Channels   int
}

// Valid reports whether the format is usable.
func (f Format) Valid() bool { return f.SampleRate > 0 && f.Channels > 0 }

// Frame is a chunk of signed 16-bit PCM. Frames are immutable once handed to
// another component: producers must not reuse Data after sending a frame.
type Frame struct {
	Data       []int16 // interleaved samples
	SampleRate int
	Channels   int
}

// Format returns the frame's format.
func (f Frame) Format() Format { return Format{SampleRate: f.SampleRate, Channels: f.Channels} }

// Samples returns the number of samples per channel.
func (f Frame) Samples() int {
	if f.Channels <= 0 {
		return 0
	}
	return len(f.Data) / f.Channels
}

// Duration returns the playback duration of the frame.
func (f Frame) Duration() time.Duration {
	if f.SampleRate <= 0 {
		return 0
	}
	return time.Duration(f.Samples()) * time.Second / time.Duration(f.SampleRate)
}

// BytesLE encodes the frame as little-endian PCM16 bytes.
func (f Frame) BytesLE() []byte {
	b := make([]byte, 2*len(f.Data))
	for i, s := range f.Data {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(s))
	}
	return b
}

// FromBytesLE decodes little-endian PCM16 bytes. A trailing odd byte is
// ignored; use Chunker when reading from a byte stream.
func FromBytesLE(b []byte, format Format) Frame {
	n := len(b) / 2
	data := make([]int16, n)
	for i := range data {
		data[i] = int16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return Frame{Data: data, SampleRate: format.SampleRate, Channels: format.Channels}
}

// RMS returns the root-mean-square level of the frame in [0, 1].
func RMS(f Frame) float64 {
	if len(f.Data) == 0 {
		return 0
	}
	var sum float64
	for _, s := range f.Data {
		v := float64(s) / 32768
		sum += v * v
	}
	return math.Sqrt(sum / float64(len(f.Data)))
}

// DBFS returns the frame level in dB relative to full scale. Silence is
// reported as -100.
func DBFS(f Frame) float64 {
	rms := RMS(f)
	if rms <= 1e-5 {
		return -100
	}
	return 20 * math.Log10(rms)
}

// Silence returns a frame of zeros with the given duration.
func Silence(format Format, d time.Duration) Frame {
	n := int(int64(format.SampleRate) * int64(d) / int64(time.Second))
	return Frame{Data: make([]int16, n*format.Channels), SampleRate: format.SampleRate, Channels: format.Channels}
}

// Tone returns a sine wave. It is used by tests, examples and fake providers.
func Tone(format Format, freq float64, d time.Duration, amplitude float64) Frame {
	n := int(int64(format.SampleRate) * int64(d) / int64(time.Second))
	data := make([]int16, n*format.Channels)
	for i := 0; i < n; i++ {
		v := int16(amplitude * 32767 * math.Sin(2*math.Pi*freq*float64(i)/float64(format.SampleRate)))
		for c := 0; c < format.Channels; c++ {
			data[i*format.Channels+c] = v
		}
	}
	return Frame{Data: data, SampleRate: format.SampleRate, Channels: format.Channels}
}

// Chunker re-frames an arbitrary byte stream of little-endian PCM16 into
// fixed-duration frames. It is not safe for concurrent use.
type Chunker struct {
	format    Format
	frameSize int // bytes per output frame
	buf       []byte
}

// NewChunker returns a Chunker producing frames of the given duration.
func NewChunker(format Format, frame time.Duration) *Chunker {
	samples := int(int64(format.SampleRate) * int64(frame) / int64(time.Second))
	if samples <= 0 {
		samples = 1
	}
	return &Chunker{format: format, frameSize: samples * format.Channels * 2}
}

// Write appends bytes and returns every complete frame now available.
func (c *Chunker) Write(b []byte) []Frame {
	c.buf = append(c.buf, b...)
	var out []Frame
	for len(c.buf) >= c.frameSize {
		out = append(out, FromBytesLE(c.buf[:c.frameSize], c.format))
		c.buf = c.buf[c.frameSize:]
	}
	if len(c.buf) == 0 {
		c.buf = nil
	}
	return out
}

// Flush returns the remaining partial frame, if any. A dangling half sample
// is discarded.
func (c *Chunker) Flush() (Frame, bool) {
	align := 2 * c.format.Channels
	n := len(c.buf) / align * align
	if n == 0 {
		c.buf = nil
		return Frame{}, false
	}
	f := FromBytesLE(c.buf[:n], c.format)
	c.buf = nil
	return f, true
}
