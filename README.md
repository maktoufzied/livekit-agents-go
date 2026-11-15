# livekit-agents-go

A community Go SDK for building realtime voice AI agents on [LiveKit](https://livekit.io).

> **Not an official LiveKit SDK.** This is an independent, community implementation
> inspired by LiveKit's [Python](https://github.com/livekit/agents) and
> [TypeScript](https://github.com/livekit/agents-js) Agents frameworks. It does not
> have feature parity with them; see [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

The orchestration is written in Go. It does not wrap a Python or Node.js process.
WebRTC, signaling, server APIs and protocol messages come from the official
[`server-sdk-go`](https://github.com/livekit/server-sdk-go) and
[`protocol`](https://github.com/livekit/protocol) modules.

## What it looks like

```go
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/providers/anthropic"
	"github.com/maktoufzied/livekit-agents-go/providers/deepgram"
	"github.com/maktoufzied/livekit-agents-go/roomio"
	"github.com/maktoufzied/livekit-agents-go/vad"
)

// One call per job: a room the agent was dispatched to.
func entrypoint(ctx context.Context, job *agents.Job) error {
	stt, err := deepgram.NewSTT(deepgram.STTOptions{}) // DEEPGRAM_API_KEY
	if err != nil {
		return err
	}
	tts, err := deepgram.NewTTS(deepgram.TTSOptions{})
	if err != nil {
		return err
	}
	llm, err := anthropic.NewLLM(anthropic.LLMOptions{}) // ANTHROPIC_API_KEY
	if err != nil {
		return err
	}

	session, err := agents.NewSession(agents.SessionOptions{
		Agent: &agents.Agent{Instructions: "You are a helpful voice assistant. Keep answers short."},
		STT:   stt,
		LLM:   llm,
		TTS:   tts,
		VAD:   vad.NewEnergy(vad.EnergyOptions{}),
	})
	if err != nil {
		return err
	}

	// Join the room, listen to the participant, speak, until they leave.
	return roomio.Run(ctx, job, session, roomio.Options{})
}

func main() {
	// LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
	worker, err := agents.NewWorker(agents.WorkerOptions{Entrypoint: entrypoint})
	if err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := worker.Run(ctx); err != nil { // SIGTERM drains running jobs
		log.Fatal(err)
	}
}
```

This exact program is [`examples/quickstart`](examples/quickstart). More:
[`examples/voice-assistant`](examples/voice-assistant),
[`examples/tool-calling`](examples/tool-calling),
[`examples/explicit-dispatch`](examples/explicit-dispatch).

## How it fits together

```
 LiveKit server                        your process
┌──────────────┐  /agent WebSocket   ┌─────────────────────────────────────────────┐
│ job dispatch │◄───────────────────►│ Worker   register · availability · status   │
│              │  protobuf messages  │   │ one goroutine per job                   │
│              │                     │   ▼                                         │
│     SFU      │◄─── WebRTC ────────►│ Job ──► roomio (Opus ⇄ PCM, cgo)     │
└──────────────┘  Opus audio         │                  │ PCM16 frames             │
                                     │                  ▼                          │
                                     │ Session: VAD ─► STT ─► turn ─► LLM ─► TTS   │
                                     │                        ▲ tools   │          │
                                     │        interruption ───┘         ▼ playout  │
                                     └─────────────────────────────────────────────┘
```

| Package | Role | cgo |
|---|---|---|
| `agents` (root) | `Worker`, `Job`, `Session`, `Agent`, events, metrics, errors | no |
| `llm`, `stt`, `tts`, `vad` | Provider interfaces, conversation, tools, energy VAD | no |
| `vad/silero` | Silero VAD model on ONNX Runtime | **yes**, library loaded at run time |
| `turn/smartturn` | Smart Turn end-of-turn model on ONNX Runtime | **yes**, library loaded at run time |
| `audio` | PCM frame type and helpers | no |
| `roomio` | LiveKit room audio in/out, agent state and transcripts | **yes** |
| `providers/deepgram` | Streaming STT (WebSocket), TTS (REST) | no |
| `providers/anthropic` | LLM (Messages API, streaming, tools) | no |
| `providers/openai` | LLM (Chat Completions, also OpenAI-compatible servers), TTS | no |
| `agentstest` | Deterministic STT, LLM, TTS and audio I/O for tests | no |
| `cmd/lkagent` (separate module) | Creates and deploys Go agents on LiveKit Cloud | no |
| `realtime` | Interfaces for speech-to-speech models (not wired yet) | no |

Design details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Requirements

- Go 1.27 or newer (see `go.mod`).
- For `roomio` (and so for any agent that joins a room): a C toolchain, `pkg-config`,
  **libopus**, **libopusfile** and **libsoxr**. They are required by the official Go
  SDK's PCM track helpers, which this SDK reuses for Opus and resampling.

| System | Command |
|---|---|
| macOS | `brew install pkg-config opus opusfile libsoxr` |
| Debian / Ubuntu | `apt-get install pkg-config libopus-dev libopusfile-dev libsoxr-dev` |

The core packages and the providers build with `CGO_ENABLED=0`.

### Voice activity detection

| Detector | Package | Needs | Use it when |
|---|---|---|---|
| Silero model | `vad/silero` | cgo, and the ONNX Runtime library at run time | Production: it ignores noise, music and keyboard sounds |
| Energy | `vad` | Nothing | Tests, quiet rooms, or builds without cgo |

```go
detector, err := silero.New(silero.Options{}) // once per process
if err != nil {
	return err // errors.Is(err, silero.ErrRuntimeUnavailable): library missing
}
session, err := agents.NewSession(agents.SessionOptions{VAD: detector /* ... */})
```

ONNX Runtime is not needed to build. Install the pinned version with
`make deps-onnxruntime` (or `scripts/install-onnxruntime.sh <dir>`), or point
`ONNXRUNTIME_LIB` at an existing library. The examples use Silero when the library is
found and fall back to the energy detector with a warning; `VAD=silero` or `VAD=energy`
forces one. On an Apple M4 the model takes about 0.06 ms per 20 ms frame.

### End-of-turn model

Without a model, a turn ends after a fixed silence, so the agent answers when the user
only pauses to think. `turn/smartturn` listens to the user's last 8 seconds and says
whether the sentence sounds finished.

```
user stops ─► VAD: silence ─► Smart Turn ─┬─ finished ────► reply after 500 ms
                                          └─ unfinished ──► wait up to 3 s for more
```

```go
endOfTurn, err := smartturn.New(smartturn.Options{}) // once per process
if err != nil {
	return err // errors.Is(err, smartturn.ErrRuntimeUnavailable): library missing
}
session, err := agents.NewSession(agents.SessionOptions{
	VAD:          detector,
	TurnDetector: endOfTurn,
	// ...
})
```

| Item | Value |
|---|---|
| Model | [Smart Turn](https://github.com/pipecat-ai/smart-turn) v3.2 by the Pipecat project, BSD 2-Clause, 8.7 MB, embedded |
| Languages | 23, see the model page |
| Needs | cgo and the ONNX Runtime library, like `vad/silero` |
| Cost | About 60 ms of one core per turn on an Apple M4. It runs during the endpointing delay, so a finished turn is not delayed |
| Delays | `TurnOptions.MinEndpointingDelay` (500 ms), `MaxEndpointingDelay` (3 s), `EndOfTurnThreshold` (0.5) |

The examples use it when ONNX Runtime is found; `TURN_DETECTOR=off` disables it. Any
other model can be plugged in through the `agents.TurnDetector` interface, which
receives the transcript and the audio of the turn. LiveKit's own turn-detector model is
not used: its license allows it only with LiveKit's Agents framework.

## Run an example

1. Start a LiveKit server: `livekit-server --dev` (or use a LiveKit Cloud project).
2. Configure the environment:

   ```sh
   cp .env.example .env      # fill in the keys
   set -a; source .env; set +a
   ```

3. Run the agent:

   ```sh
   go run ./examples/voice-assistant
   ```

4. Join a room as a user, for example with the
   [Agents Playground](https://agents-playground.livekit.io) or
   `lk room join --identity me my-room`, and talk.

No provider keys at hand? `AGENT_FAKE_PROVIDERS=1 go run ./examples/voice-assistant`
answers any sound with a scripted reply rendered as a tone. It is enough to check
connectivity, dispatch and interruptions.

With Docker:

```sh
docker build -t livekit-agents-go-example .
docker run --rm --env-file .env livekit-agents-go-example
```

The image runs `examples/explicit-dispatch` by default. Pass
`--build-arg EXAMPLE=voice-assistant` for another example.

From inside the container, `localhost` is the container: point `LIVEKIT_URL` at
`ws://host.docker.internal:7880` for a server on your machine. WebRTC from a
container to a local dev server also needs the server's UDP port to be reachable.

### Explicit dispatch

```sh
go run ./examples/explicit-dispatch                                   # worker named "support-agent"
go run ./examples/explicit-dispatch/dispatch -room demo -customer Ada  # send it to a room
```

A worker with `AgentName` set is only sent to rooms it is dispatched to, through the
agent dispatch API or the room configuration of a participant token. Without a name
the server dispatches it to every new room.

## Capabilities

| Area | What is implemented |
|---|---|
| Worker | Registration, job requests, accept/reject, assignment, capacity and load reporting, reconnection with running-job report, graceful drain, job termination, panic isolation, HTTP health endpoint, hosted mode for LiveKit Cloud |
| Audio | Subscribe to a participant, publish agent audio, Opus decode/encode and resampling (upstream helpers), paced playout with a bounded queue |
| Turns | Silero VAD (model) and energy VAD, Smart Turn end-of-turn model, endpointing delays, STT-driven turns, manual turns, pluggable end-of-turn model interface, model request sent ahead of the end of the turn |
| Interruptions | Cancels LLM, TTS and tools, drops queued audio, records only what was played, no stale audio in the next turn. A sound without words only pauses the agent, which then resumes where it stopped |
| Conversation | Role-based history, instructions, streaming replies, `Say`, `GenerateReply` |
| Tools | Typed tools with generated JSON Schema, argument validation, timeouts, concurrent or sequential execution, model-visible errors, never retried |
| Events | Agent and user state, transcripts, history items, speech, tool calls, errors, metrics |
| Operations | `slog`, latency metrics, resource stats, timeouts, bounded retries, typed errors, secret redaction |

## Semantics worth knowing

| Topic | Rule |
|---|---|
| Streams | The caller owns every provider stream and must close it. `Close` is idempotent and unblocks `Recv`. The context given when opening a stream bounds its whole life. `llm.Chunks`, `tts.Frames` and `stt.Events` adapt streams to `for … range`. |
| Ordering | Stream results and session events are delivered in order. Event handlers run one at a time on their own goroutine. |
| Backpressure | Every pipeline queue is bounded (`BufferOptions`). A slow TTS or playout blocks reading from the LLM. A slow STT drops the *oldest* audio. A slow event handler loses only metrics and interim transcripts (counted in `Session.Stats`); lifecycle, history, speech, tool and error events are always delivered. |
| Retries | LLM and TTS requests are retried only if they failed before producing output and the error is retryable. Tools are never retried. |
| Interruption | History keeps the text of fully played sentences, plus the played fraction of the sentence that was cut. |
| Lifetime | The context passed to `Session.Start` is the session's lifetime, not a start deadline. `Session.Close` returns after every goroutine exited and every provider stream is closed, with the error that closed the session if any. `Worker.Run` returns after jobs drained. |
| Jobs | Jobs run in-process as goroutines. A panic fails the job, not the worker. There is no process isolation. |

## Testing

| Command | What it runs | Needs |
|---|---|---|
| `make fmt-check vet` | `gofmt`, `go vet` (all build tags) | native libs |
| `go test -race -count=1 ./...` | Unit tests with fake providers and a fake LiveKit agent server | native libs |
| `./scripts/integration.sh` | Full vertical slice over real WebRTC against a local LiveKit server, fake speech providers | `livekit-server` or Docker |
| `go test -tags e2e -count=1 -v ./test/e2e/ ./providers/...` | Real STT, LLM and TTS | provider API keys |

CI (`.github/workflows/ci.yml`) runs formatting, vet, tests, the race detector, the
integration tests and the Docker build. The end-to-end job is manual.

## Verified and not verified

| Behaviour | Status |
|---|---|
| Session pipeline, interruptions, tools, retries, backpressure, cleanup | Verified by unit tests with the race detector |
| Worker protocol: register, availability, assignment, status, termination, reconnect, drain | Verified by unit tests against a fake server **and** against `livekit-server` 1.13.11 |
| Explicit and automatic dispatch, room join, audio both ways, interruption, shutdown | Verified against `livekit-server` 1.13.11 with fake speech providers |
| Agent state attribute (`lk.agent.state`) visible to clients | Verified against `livekit-server` 1.13.11 |
| Transcripts published as `lk.transcription` text streams | Verified manually on 2026-11-13: shown in the Agents Playground chat panel |
| Live voice call (browser, LiveKit Cloud, Deepgram + Anthropic) | Verified manually on 2026-11-13, one call: conversation, interruption by talking over the agent, no false interruption observed. Not an automated test |
| Deepgram, Anthropic and OpenAI providers | Verified against the live APIs on 2026-11-13 with the opt-in `e2e` tests: streaming STT, both LLMs with a tool round trip, both TTS, and the full STT → LLM → TTS pipeline (Deepgram + Anthropic) |
| LiveKit Cloud connectivity | Verified on 2026-11-13: the explicit-dispatch vertical slice (fake speech providers) passed against a Cloud project, server 1.13.10. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| LiveKit Cloud managed agent deployment | Implemented (hosted-mode worker, health endpoint, `cmd/lkagent`), tested locally. **Not yet run on LiveKit Cloud**, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| Linux arm64 (Docker) | Image builds and the example worker registers with `livekit-server`. The image loads the Silero model at startup |
| Linux amd64 (GitHub Actions, Ubuntu 26.04) | Unit tests with `-race`, including the Silero VAD and the Smart Turn model, and integration tests on every push |
| False-interruption pause and resume | Verified by unit tests with a simulated output: nothing lost, nothing played twice. The hold followed by a confirmed interruption also ran over WebRTC in the integration tests. **Not verified:** the resume over WebRTC, and a live call |
| Smart Turn end-of-turn model | Same model input and same probabilities as the reference Python code on 28 cases; a session waits on a synthesized unfinished sentence and answers a finished one at once; the pipeline test with live providers passed with it on 2026-11-14. **Not verified:** a live call with a person pausing mid-sentence. See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) |

## Limitations and follow-up work

Not implemented in this version:

- Speech-to-speech realtime models (the `realtime` interfaces exist, nothing uses them).
- Video and vision input, avatars.
- LiveKit's turn-detector model (its license excludes other frameworks; `turn/smartturn`
  is the alternative).
- Agent handoffs, multi-agent workflows, task groups.
- Telephony helpers (SIP, DTMF, voicemail detection), background audio.
- Word-level transcript alignment: history is cut per sentence and by played fraction.
- Process isolation and prewarming of jobs.
- More providers; noise cancellation.

See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for the comparison with the
upstream frameworks and for the roadmap.

## Stability

Version 0. There is no compatibility promise yet: exported names and behaviour can
change between minor versions until 1.0. Runnable examples for the main types are in
the package documentation (`go doc`, or the `Example` functions in `example_test.go`).

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). LiveKit is a trademark of
LiveKit, Inc. This project is not affiliated with or endorsed by LiveKit, Inc.
