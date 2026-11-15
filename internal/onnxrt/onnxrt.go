// Package onnxrt loads the ONNX Runtime shared library once per process, for
// the packages that run models on it (vad/silero, turn/smartturn).
package onnxrt

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// LibraryEnv is the environment variable naming the ONNX Runtime shared
// library, used when no explicit path is given.
const LibraryEnv = "ONNXRUNTIME_LIB"

// ErrUnavailable reports that the ONNX Runtime shared library could not be
// loaded.
var ErrUnavailable = errors.New("ONNX Runtime unavailable")

var mu sync.Mutex

// Load initializes the process-wide ONNX Runtime environment once. A failed
// attempt can be retried with another path. Once it succeeded, later calls
// return nil and ignore explicit.
func Load(explicit string) error {
	mu.Lock()
	defer mu.Unlock()
	if ort.IsInitialized() {
		return nil // by an earlier Load, or by the application itself
	}

	var firstErr error
	for _, path := range Candidates(explicit) {
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
		ErrUnavailable, firstErr, LibraryEnv)
}

// Candidates lists the libraries to try, most specific first. An explicit
// path or the environment variable is the only candidate: a wrong value
// should fail loudly rather than fall back to another library.
func Candidates(explicit string) []string {
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
