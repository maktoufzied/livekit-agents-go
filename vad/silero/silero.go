// Package silero is a voice activity detector backed by the Silero VAD neural
// model. It is far more robust than vad.Energy against background noise,
// music and keyboard sounds.
//
// The model (Silero VAD 6.2.3, MIT license) is embedded in the package. It
// runs on ONNX Runtime, which is loaded as a shared library when New is
// called: building needs cgo but no ONNX Runtime headers or library, running
// needs the library on the machine. See Options.LibraryPath for how it is
// found and scripts/install-onnxruntime.sh for a pinned download.
//
//	detector, err := silero.New(silero.Options{})
//	if err != nil {
//		return err // errors.Is(err, silero.ErrRuntimeUnavailable): library missing
//	}
//	defer detector.Close()
//	session := agents.NewSession(agents.SessionOptions{VAD: detector, ...})
//
// Thresholds, timings and the smoothing of the model output follow the
// upstream Python plugin (livekit-plugins-silero 1.8.12).
package silero

import (
	_ "embed"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/maktoufzied/livekit-agents-go/vad"
)

//go:embed silero_vad.onnx
var embeddedModel []byte

// LibraryEnv is the environment variable naming the ONNX Runtime shared
// library, used when Options.LibraryPath is empty.
const LibraryEnv = "ONNXRUNTIME_LIB"

var (
	// ErrRuntimeUnavailable reports that the ONNX Runtime shared library
	// could not be loaded.
	ErrRuntimeUnavailable = errors.New("silero: ONNX Runtime unavailable")
	// ErrClosed is returned when a closed VAD or detector is used.
	ErrClosed = errors.New("silero: closed")
)

// Options configures the detector. Zero values select the defaults of the
// upstream Python plugin.
type Options struct {
	// MinSpeechDuration is the speech needed before SpeechStart. Default 50ms.
	MinSpeechDuration time.Duration
	// MinSilenceDuration is the silence needed before SpeechEnd. Default 550ms.
	MinSilenceDuration time.Duration
	// ActivationThreshold is the probability at or above which a window
	// counts as speech. Default 0.5. Raise it in noisy rooms.
	ActivationThreshold float64
	// DeactivationThreshold is the probability at or below which a window
	// stops counting as speech once speech has started. Default
	// ActivationThreshold - 0.15, at least 0.01.
	DeactivationThreshold float64
	// SampleRate is the rate the model runs at: 16000 (default) or 8000.
	// Audio at any other rate is resampled; use 8000 for telephony audio.
	SampleRate int

	// LibraryPath is the ONNX Runtime shared library to load. When empty,
	// the ONNXRUNTIME_LIB environment variable is used, then the usual
	// install locations, then the system loader's search path.
	//
	// ONNX Runtime is loaded once per process: the first successful New
	// decides the library, and later values are ignored.
	LibraryPath string
	// Model replaces the embedded model with another Silero VAD ONNX file of
	// the same generation (inputs "input", "state", "sr").
	Model []byte
}

func (o Options) withDefaults() (Options, error) {
	if o.MinSpeechDuration == 0 {
		o.MinSpeechDuration = 50 * time.Millisecond
	}
	if o.MinSilenceDuration == 0 {
		o.MinSilenceDuration = 550 * time.Millisecond
	}
	if o.ActivationThreshold == 0 {
		o.ActivationThreshold = 0.5
	}
	if o.DeactivationThreshold == 0 {
		o.DeactivationThreshold = max(o.ActivationThreshold-0.15, 0.01)
	}
	if o.SampleRate == 0 {
		o.SampleRate = 16000
	}
	switch {
	case o.SampleRate != 8000 && o.SampleRate != 16000:
		return o, fmt.Errorf("silero: sample rate %d not supported, use 8000 or 16000", o.SampleRate)
	case o.ActivationThreshold < 0 || o.ActivationThreshold > 1:
		return o, fmt.Errorf("silero: activation threshold %v out of range [0, 1]", o.ActivationThreshold)
	case o.DeactivationThreshold < 0 || o.DeactivationThreshold > o.ActivationThreshold:
		return o, fmt.Errorf("silero: deactivation threshold %v must be between 0 and the activation threshold", o.DeactivationThreshold)
	case o.MinSpeechDuration < 0 || o.MinSilenceDuration < 0:
		return o, errors.New("silero: negative duration")
	}
	return o, nil
}

// VAD is a Silero voice activity detector. It holds the loaded model, which
// all its detectors share. It is safe for concurrent use.
type VAD struct {
	opts Options

	mu      sync.Mutex
	session *ort.DynamicAdvancedSession
}

var _ vad.VAD = (*VAD)(nil)

// New loads ONNX Runtime and the model. It takes a few tens of milliseconds:
// call it once at startup and share the VAD between sessions.
func New(opts Options) (*VAD, error) {
	opts, err := opts.withDefaults()
	if err != nil {
		return nil, err
	}
	if err := loadRuntime(opts.LibraryPath); err != nil {
		return nil, err
	}

	so, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("silero: session options: %w", err)
	}
	defer so.Destroy()
	// One thread and no spinning: a window takes well under a millisecond,
	// and many sessions share the process.
	for _, set := range []func() error{
		func() error { return so.SetIntraOpNumThreads(1) },
		func() error { return so.SetInterOpNumThreads(1) },
		func() error { return so.SetExecutionMode(ort.ExecutionModeSequential) },
		func() error { return so.AddSessionConfigEntry("session.intra_op.allow_spinning", "0") },
		func() error { return so.AddSessionConfigEntry("session.inter_op.allow_spinning", "0") },
	} {
		if err := set(); err != nil {
			return nil, fmt.Errorf("silero: session options: %w", err)
		}
	}

	data := opts.Model
	if len(data) == 0 {
		data = embeddedModel
	}
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(data,
		[]string{"input", "state", "sr"}, []string{"output", "stateN"}, so)
	if err != nil {
		return nil, fmt.Errorf("silero: load model: %w", err)
	}
	return &VAD{opts: opts, session: session}, nil
}

// NewDetector implements vad.VAD.
func (v *VAD) NewDetector() (vad.Detector, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.session == nil {
		return nil, ErrClosed
	}
	m, err := newOnnxModel(v.session, v.opts.SampleRate)
	if err != nil {
		return nil, err
	}
	return newDetector(v.opts, m), nil
}

// Close releases the model. Close the detectors first: a detector must not
// be used after its VAD is closed. ONNX Runtime itself stays loaded for the
// life of the process.
func (v *VAD) Close() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.session == nil {
		return nil
	}
	err := v.session.Destroy()
	v.session = nil
	return err
}

// onnxModel is the per-stream state of the network: the tensors, the
// recurrent state and the context carried from one window to the next.
type onnxModel struct {
	session *ort.DynamicAdvancedSession
	context int

	input  *ort.Tensor[float32] // [1, context+window]
	state  *ort.Tensor[float32] // [2, 1, 128]
	rate   *ort.Scalar[int64]
	output *ort.Tensor[float32] // [1, 1]
	stateN *ort.Tensor[float32] // [2, 1, 128]

	inputs, outputs []ort.Value // the tensors above, in model order
}

func newOnnxModel(session *ort.DynamicAdvancedSession, sampleRate int) (_ *onnxModel, err error) {
	window, context := 512, 64
	if sampleRate == 8000 {
		window, context = 256, 32
	}
	m := &onnxModel{session: session, context: context}
	defer func() {
		if err != nil {
			m.close()
			err = fmt.Errorf("silero: allocate tensors: %w", err)
		}
	}()
	if m.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, int64(context+window))); err != nil {
		return nil, err
	}
	if m.state, err = ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 128)); err != nil {
		return nil, err
	}
	if m.rate, err = ort.NewScalar(int64(sampleRate)); err != nil {
		return nil, err
	}
	if m.output, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 1)); err != nil {
		return nil, err
	}
	if m.stateN, err = ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 128)); err != nil {
		return nil, err
	}
	m.inputs = []ort.Value{m.input, m.state, m.rate}
	m.outputs = []ort.Value{m.output, m.stateN}
	return m, nil
}

func (m *onnxModel) infer(window []float32) (float32, error) {
	in := m.input.GetData()
	// The model sees the tail of the previous window before the new one.
	copy(in[:m.context], in[len(in)-m.context:])
	copy(in[m.context:], window)

	if err := m.session.Run(m.inputs, m.outputs); err != nil {
		return 0, err
	}
	copy(m.state.GetData(), m.stateN.GetData())
	return m.output.GetData()[0], nil
}

func (m *onnxModel) close() error {
	var errs []error
	if m.input != nil {
		errs = append(errs, m.input.Destroy())
	}
	if m.state != nil {
		errs = append(errs, m.state.Destroy())
	}
	if m.rate != nil {
		errs = append(errs, m.rate.Destroy())
	}
	if m.output != nil {
		errs = append(errs, m.output.Destroy())
	}
	if m.stateN != nil {
		errs = append(errs, m.stateN.Destroy())
	}
	return errors.Join(errs...)
}

// ---------------------------------------------------------------------------
// ONNX Runtime loading

var runtimeMu sync.Mutex

// loadRuntime initializes the process-wide ONNX Runtime environment once. A
// failed attempt can be retried with another path.
func loadRuntime(explicit string) error {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()
	if ort.IsInitialized() {
		return nil // by an earlier New, or by the application itself
	}

	candidates := libraryCandidates(explicit)
	var firstErr error
	for _, path := range candidates {
		ort.SetSharedLibraryPath(path)
		err := ort.InitializeEnvironment()
		if err == nil {
			return nil
		}
		if firstErr == nil {
			firstErr = fmt.Errorf("%s: %w", path, err)
		}
	}
	return fmt.Errorf("%w: %v (set Options.LibraryPath or %s; see scripts/install-onnxruntime.sh)",
		ErrRuntimeUnavailable, firstErr, LibraryEnv)
}

// libraryCandidates lists the libraries to try, most specific first. An
// explicit path or the environment variable is the only candidate: a wrong
// value should fail loudly rather than fall back to another library.
func libraryCandidates(explicit string) []string {
	if explicit != "" {
		return []string{explicit}
	}
	if env := os.Getenv(LibraryEnv); env != "" {
		return []string{env}
	}

	var name string
	var dirs []string
	switch runtime.GOOS {
	case "darwin":
		name = "libonnxruntime.dylib"
		dirs = []string{"/opt/homebrew/lib", "/usr/local/lib"}
	case "windows":
		return []string{"onnxruntime.dll"}
	default:
		name = "libonnxruntime.so"
		dirs = []string{"/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"}
	}

	var out []string
	for _, dir := range dirs {
		path := filepath.Join(dir, name)
		if _, err := os.Stat(path); err == nil {
			out = append(out, path)
			continue
		}
		// Distribution packages ship only the versioned file.
		if matches, _ := filepath.Glob(path + ".*"); len(matches) > 0 {
			out = append(out, matches[len(matches)-1])
		}
	}
	// Last, the bare name: the system loader searches its own paths.
	return append(out, name)
}
