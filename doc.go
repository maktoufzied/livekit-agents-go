// Package agents is a community Go SDK for building realtime voice AI
// agents on LiveKit. It is not an official LiveKit SDK.
//
// The package contains the orchestration core:
//
//   - Worker registers with a LiveKit server, receives jobs and runs one
//     entrypoint per job, in-process.
//   - Job describes one job and connects to its room.
//   - Session runs the voice pipeline: audio in → VAD and STT → turn
//     detection → LLM (with tools) → TTS → audio out, with interruptions.
//   - Agent holds the instructions and tools.
//
// Providers (speech-to-text, LLM, text-to-speech, VAD) are plain interfaces
// defined in the stt, llm, tts and vad packages; integrations live under
// providers/. Room audio I/O lives in package roomio, the only package that
// needs cgo.
//
// A minimal agent:
//
//	func entrypoint(ctx context.Context, job *agents.Job) error {
//		session, err := agents.NewSession(agents.SessionOptions{
//			Agent: &agents.Agent{Instructions: "You are a helpful voice assistant."},
//			STT:   mySTT, LLM: myLLM, TTS: myTTS,
//			VAD:   vad.NewEnergy(vad.EnergyOptions{}),
//		})
//		if err != nil {
//			return err
//		}
//		return roomio.Run(ctx, job, session, roomio.Options{})
//	}
//
//	func main() {
//		worker, err := agents.NewWorker(agents.WorkerOptions{Entrypoint: entrypoint})
//		if err != nil {
//			log.Fatal(err)
//		}
//		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
//		defer stop()
//		if err := worker.Run(ctx); err != nil {
//			log.Fatal(err)
//		}
//	}
package agents
