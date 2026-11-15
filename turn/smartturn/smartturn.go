// Package smartturn predicts whether the user finished their turn, from the
// sound of their voice. It lets a session wait when the user pauses in the
// middle of a sentence ("I would like to, um...") instead of answering after
// a fixed silence.
//
// It runs Smart Turn v3.2, an open model by the Pipecat project (BSD
// 2-Clause license) that supports 23 languages. The model is embedded in the
// package. It runs on ONNX Runtime, loaded as a shared library when New is
// called, exactly like vad/silero: see Options.LibraryPath and
// scripts/install-onnxruntime.sh.
//
//	detector, err := smartturn.New(smartturn.Options{})
//	if err != nil {
//		return err // errors.Is(err, smartturn.ErrRuntimeUnavailable): library missing
//	}
//	defer detector.Close()
//	session, err := agents.NewSession(agents.SessionOptions{
//		VAD:          vad,
//		TurnDetector: detector,
//		...
//	})
//
// The session asks the detector after TurnOptions.MinEndpointingDelay of
// silence. A probability under TurnOptions.EndOfTurnThreshold makes it wait
// up to MaxEndpointingDelay for the user to continue.
package smartturn

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/audio"
	"github.com/maktoufzied/livekit-agents-go/internal/dsp"
	"github.com/maktoufzied/livekit-agents-go/internal/onnxrt"
)

//go:embed smart_turn_v3.2_cpu.onnx
var embeddedModel []byte

// LibraryEnv is the environment variable naming the ONNX Runtime shared
// library, used when Options.LibraryPath is empty.
const LibraryEnv = onnxrt.LibraryEnv

var (
	// ErrRuntimeUnavailable reports that the ONNX Runtime shared library
	// could not be loaded.
	ErrRuntimeUnavailable = onnxrt.ErrUnavailable
	// ErrClosed is returned when a closed Detector is used.
	ErrClosed = errors.New("smartturn: closed")
)

// Options configures the detector. The zero value is ready to use.
type Options struct {
	// Threads is the number of threads one prediction may use. Default 1.
	Threads int
	// LibraryPath is the ONNX Runtime shared library to load. When empty,
	// the ONNXRUNTIME_LIB environment variable is used, then the usual
	// install locations, then the system loader's search path.
	//
	// ONNX Runtime is loaded once per process: the first package that loads
	// it decides the library, and later values are ignored.
	LibraryPath string
	// Model replaces the embedded model with another Smart Turn v3 ONNX
	// file (input "input_features" [1, 80, 800], output "logits").
	Model []byte
}

// Detector is a Smart Turn end-of-turn detector. It implements
// agents.TurnDetector and is safe for concurrent use: share one between all
// sessions of the process.
type Detector struct {
	mu      sync.Mutex
	session *ort.DynamicAdvancedSession
	idle    []*worker
}

var _ agents.TurnDetector = (*Detector)(nil)

// New loads ONNX Runtime and the model. Call it once at startup.
func New(opts Options) (*Detector, error) {
	if opts.Threads < 0 {
		return nil, fmt.Errorf("smartturn: negative thread count %d", opts.Threads)
	}
	if opts.Threads == 0 {
		opts.Threads = 1
	}
	if err := onnxrt.Load(opts.LibraryPath); err != nil {
		return nil, fmt.Errorf("smartturn: %w", err)
	}

	so, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("smartturn: session options: %w", err)
	}
	defer so.Destroy()
	for _, set := range []func() error{
		func() error { return so.SetIntraOpNumThreads(opts.Threads) },
		func() error { return so.SetInterOpNumThreads(1) },
		func() error { return so.SetExecutionMode(ort.ExecutionModeSequential) },
		// Predictions are rare and short: idle threads must not spin.
		func() error { return so.AddSessionConfigEntry("session.intra_op.allow_spinning", "0") },
		func() error { return so.AddSessionConfigEntry("session.inter_op.allow_spinning", "0") },
	} {
		if err := set(); err != nil {
			return nil, fmt.Errorf("smartturn: session options: %w", err)
		}
	}

	data := opts.Model
	if len(data) == 0 {
		data = embeddedModel
	}
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(data,
		[]string{"input_features"}, []string{"logits"}, so)
	if err != nil {
		return nil, fmt.Errorf("smartturn: load model: %w", err)
	}
	return &Detector{session: session}, nil
}

// EndOfTurnProbability implements agents.TurnDetector. It uses in.Audio and
// ignores the transcript. Without audio there is nothing to wait for and it
// returns 1.
func (d *Detector) EndOfTurnProbability(ctx context.Context, in agents.TurnInput) (float64, error) {
	if len(in.Audio.Data) == 0 {
		return 1, nil
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return d.Predict(in.Audio)
}

// Predict returns the probability in [0, 1] that the speaker in turn is done
// talking. turn is the audio of one turn, from its start up to now, at any
// sample rate and channel count; trailing silence is fine. Only the last 8
// seconds are used. A prediction takes a few tens of milliseconds and
// cannot be canceled.
func (d *Detector) Predict(turn audio.Frame) (float64, error) {
	if turn.SampleRate <= 0 || turn.Channels <= 0 {
		return 0, fmt.Errorf("smartturn: invalid audio format %dHz/%dch", turn.SampleRate, turn.Channels)
	}
	w, session, err := d.acquire()
	if err != nil {
		return 0, err
	}
	defer d.release(w)

	// Only the tail matters: skip what the model would not see anyway.
	data := turn.Data
	if keep := (maxSeconds*turn.SampleRate + turn.SampleRate/10) * turn.Channels; len(data) > keep {
		data = data[len(data)-keep:]
	}
	resampler := dsp.NewResampler(turn.SampleRate, sampleRate)
	w.samples = resampler.Push(w.samples[:0], data, turn.Channels)
	w.extract.features(w.input.GetData(), w.samples)

	if err := session.Run(w.inputs, w.outputs); err != nil {
		return 0, fmt.Errorf("smartturn: inference: %w", err)
	}
	// Despite its name, the output already went through a sigmoid.
	return float64(w.output.GetData()[0]), nil
}

// Close releases the model. Predictions in flight must have returned. ONNX
// Runtime itself stays loaded for the life of the process.
func (d *Detector) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.session == nil {
		return nil
	}
	var errs []error
	for _, w := range d.idle {
		errs = append(errs, w.close())
	}
	d.idle = nil
	errs = append(errs, d.session.Destroy())
	d.session = nil
	return errors.Join(errs...)
}

// worker holds the buffers of one prediction. Workers are reused: a
// prediction needs about 1.5 MB of scratch memory.
type worker struct {
	extract *extractor
	samples []float32

	input           *ort.Tensor[float32] // [1, 80, 800]
	output          *ort.Tensor[float32] // [1, 1]
	inputs, outputs []ort.Value
}

func newWorker() (_ *worker, err error) {
	w := &worker{extract: newExtractor()}
	defer func() {
		if err != nil {
			w.close()
			err = fmt.Errorf("smartturn: allocate tensors: %w", err)
		}
	}()
	if w.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, numMels, numFrames)); err != nil {
		return nil, err
	}
	if w.output, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 1)); err != nil {
		return nil, err
	}
	w.inputs = []ort.Value{w.input}
	w.outputs = []ort.Value{w.output}
	return w, nil
}

func (w *worker) close() error {
	var errs []error
	if w.input != nil {
		errs = append(errs, w.input.Destroy())
	}
	if w.output != nil {
		errs = append(errs, w.output.Destroy())
	}
	return errors.Join(errs...)
}

func (d *Detector) acquire() (*worker, *ort.DynamicAdvancedSession, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.session == nil {
		return nil, nil, ErrClosed
	}
	if n := len(d.idle); n > 0 {
		w := d.idle[n-1]
		d.idle = d.idle[:n-1]
		return w, d.session, nil
	}
	w, err := newWorker()
	return w, d.session, err
}

func (d *Detector) release(w *worker) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.session == nil {
		w.close()
		return
	}
	d.idle = append(d.idle, w)
}
