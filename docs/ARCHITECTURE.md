# Architecture

This document explains how the SDK is built and why. It is written for people who
want to extend it, add a provider, or judge whether it fits their use.

## Layers

```
┌──────────────────────────────────────────────────────────────────────────┐
│ your code: entrypoint, Agent (instructions + tools), event handlers      │
├──────────────────────────────────────────────────────────────────────────┤
│ agents            Worker ─ Job ─ Session                          │
│                   (protocol)  (job)     (voice pipeline)                 │
├───────────────┬──────────────────────────────┬───────────────────────────┤
│ roomio (cgo)  │ stt · llm · tts · vad · audio │ providers/* · agentstest  │
│ room ⇄ PCM    │ small interfaces              │ implementations           │
├───────────────┴──────────────────────────────┴───────────────────────────┤
│ livekit/server-sdk-go · livekit/protocol · livekit/media-sdk · pion      │
└──────────────────────────────────────────────────────────────────────────┘
```

Dependency rules:

- `stt`, `llm`, `tts`, `vad`, `audio` depend on nothing but the standard library.
- `vad/silero` is the one model-backed detector. It uses cgo to call ONNX Runtime and
  loads that library when `silero.New` runs, so nothing else pays for it.
- Providers depend on those interface packages and on `providers` (errors, `Secret`).
  They never import the root package.
- The root package depends on the interfaces and on the pure-Go part of the official
  Go SDK. It has no cgo.
- `roomio` is the only package with cgo. A session can run without it (tests do).

## Worker

`Worker` speaks the agent worker protocol defined in
`livekit/protocol/protobufs/livekit_agent.proto`, the same one the Python and
TypeScript workers speak. Messages are binary protobuf over a WebSocket.

```
Worker                                              LiveKit server
  │ GET {url}/agent  Authorization: Bearer <JWT, video.agent=true>
  │────────────────────────────────────────────────────────►│
  │ WorkerMessage.register {type, agent_name, version,      │
  │                         allowed_permissions(agent=true)}│
  │────────────────────────────────────────────────────────►│
  │◄──────────────── ServerMessage.register {worker_id, server_info}
  │ update_worker {status, load, job_count}   every 1.5 s   │
  │────────────────────────────────────────────────────────►│
  │◄──────────────── availability {job}                     │
  │ availability {job_id, available, participant_identity,  │
  │               name, metadata, attributes}               │
  │────────────────────────────────────────────────────────►│
  │◄──────────────── assignment {job, url?, token}          │
  │ update_job {job_id, JS_RUNNING}                         │
  │────────────────────────────────────────────────────────►│
  │   ... the entrypoint joins the room with `token` ...    │
  │◄──────────────── termination {job_id}   (optional)      │
  │ update_job {job_id, JS_SUCCESS | JS_FAILED, error}      │
  │────────────────────────────────────────────────────────►│
```

| Concern | Behaviour |
|---|---|
| Capacity | A slot is reserved when a request arrives, before the accept decision. At `MaxConcurrentJobs` the worker reports `WS_FULL` and declines. Load is `active / max`, or `LoadFunc`; with `LoadThreshold` the worker is also full at that load. |
| Hosted mode | With a worker token (`LIVEKIT_WORKER_TOKEN`) the worker sends it as `worker_token`, sends the deployment name, serves the health check on `:8081` and reports CPU load with a 0.7 threshold, as the upstream workers do on LiveKit Cloud. |
| Health | Optional HTTP server: `GET /` is 200 while registered (503 before, and after 30 s disconnected), `GET /worker` returns worker info. |
| Assignment | After accepting, the worker waits 7.5 s for the assignment (the upstream value), then gives the slot back. |
| Reconnection | Delays of 0, 2, 4 ... capped at 10 s, up to `MaxRetries` consecutive failures (16 by default), as upstream. The count starts over only after a connection stayed up for 10 s, so a server that registers and drops the worker is not redialled in a loop. Running jobs keep running. After re-registering the worker sends `migrate_job` with the ids of the jobs it still runs. A job status that could not be written is sent first on the new connection; answers to job requests of the old connection are discarded. |
| Keep-alive | WebSocket ping every 30 s; the connection is dropped if nothing is read for two intervals. |
| Drain | When `Run`'s context is cancelled the worker reports `WS_FULL`, declines new requests and waits up to `DrainTimeout`. Remaining jobs are then cancelled and get `JobShutdownTimeout` to return. |
| Jobs | One goroutine per job, with its own context. A panic is recovered and reported as `JS_FAILED`. The room is disconnected and shutdown hooks run when the entrypoint returns. |
| Credentials | The API secret is a `Secret`. The room token of an assignment is never logged. |

Differences from the upstream workers are listed in
[COMPATIBILITY.md](COMPATIBILITY.md).

## Session

A `Session` is a set of goroutines around one event loop.

```
AudioInput ─► inputLoop ──VAD events──────────────► loop (turn state)
                 │                                    ▲      │
                 └─► sttFrames (bounded, drop oldest) │      │ enqueue reply
                         │                            │      ▼
                         ▼                        STT events  speechLoop (one speech at a time)
                     sttLoop ─────────────────────────┘      │
                     (reconnects)                            ▼
                                                    pipeline of the speech
```

- **`loop`** owns the turn state. VAD events, STT events and timers are posted to it as
  closures, so turn detection needs no locks.
- **`speechLoop`** runs speeches (replies and `Say`) strictly one after the other.
- **`inputLoop`** reads frames, runs the VAD synchronously and hands frames to the STT
  stream through a bounded queue.
- **`sttLoop`** keeps a recognition stream open and reopens it with bounded retries.

### One speech

```
LLM stream ─► sentence splitter ─► textCh ─► synth ─► segCh ─► playout ─► AudioOutput
  (step n)        (bounded: Sentences)        │ (bounded: SynthesisLookahead)
                                              └─ one TTS request per sentence,
                                                 frames buffered per sentence (bounded)
```

- The model's text is split into sentences. Each sentence is one TTS request.
- `synth` may run a few sentences ahead of `playout`, never more.
- `playout` is the only goroutine writing to the output. `AudioOutput.WriteFrame`
  blocks while more than a small look-ahead is queued, which paces everything behind it.
- When the model asks for tools, they run while the text of that step is still playing.
  The next model step starts once the tools returned and that text was handed to the
  output.

Why one TTS request per sentence: the session then knows exactly which audio belongs to
which text. That is what makes the history accurate after an interruption without
word-level timestamps from the TTS provider. The cost is that prosody cannot span
sentences and that incremental-input TTS APIs are not used to their full extent.

### Turn detection

| Mode | End of turn |
|---|---|
| `TurnVAD` (default with a VAD) | VAD reports silence **and** a final transcript exists, then `MinEndpointingDelay` (500 ms) without new speech. |
| `TurnSTT` (default without a VAD) | The provider's end-of-speech event, or `MaxEndpointingDelay` after the last final transcript. |
| `TurnManual` | Only `Session.CommitUserTurn`. |

An optional `TurnDetector` is asked when the delay elapsed. A probability under the
threshold postpones the turn end to `MaxEndpointingDelay` (3 s).

### Interruption

The user interrupts when the VAD hears `MinInterruptionDuration` (500 ms) of speech
while a speech is running or queued, optionally only after `MinInterruptionWords`
transcribed words. Without a VAD, any transcribed word interrupts. A committed user turn
also cancels speeches that are still pending for an older turn.

```
1. speech.Interrupt()      cancel the speech context
                           → LLM stream, TTS streams and interruptible tools stop
2. join playout            the only writer has exited: no frame can follow
3. AudioOutput.Clear()     drop queued audio, learn how much was dropped
4. played = written − dropped
5. history                 keep fully played sentences, plus the played fraction
                           (in words) of the sentence that was cut;
                           mark the assistant message Interrupted
6. tools                   every recorded tool call gets a result
                           (cancelled ones are recorded as errors)
```

Step 2 before step 3 is what prevents stale audio. A speech only returns after both, and
the next speech starts after that, so audio of two speeches cannot interleave.

If nothing was played, no assistant message is recorded.

The user message of a turn travels with the reply it triggers and is recorded when that
reply reaches the head of the speech queue. Speeches run one at a time, so the order in
the history is always: what the agent played, then what the user said to interrupt it,
then the new reply.

### Tools

| Aspect | Behaviour |
|---|---|
| Definition | `llm.NewTool` derives a JSON Schema from a Go struct. Raw schemas are accepted too. |
| Validation | Arguments are checked against the schema before the handler runs. A failure is reported to the model so it can correct itself. |
| Execution | The calls of one model step run concurrently (or in order with `ToolOptions.Sequential`). Results keep call order. |
| Timeout | Per tool, default 30 s. The session stops waiting even if the handler ignores its context. |
| Errors | `llm.ToolError` text goes to the model. Any other error becomes a generic message, so internals do not leak. |
| Interruption | The handler's context is cancelled, unless the tool is `Uninterruptible`: then it runs to completion and its result is recorded. |
| Retries | None. The model may decide to call again, bounded by `ToolOptions.MaxSteps`. |

### Failure handling

| Failure | Result |
|---|---|
| LLM or TTS fails before any output | Retried up to `Retry.MaxRetries` with backoff if the error is retryable (`providers.Error.Retryable`) or a timeout. |
| LLM or TTS fails after output started | The speech stops, what was played is recorded, an `ErrorEvent` is emitted. The session stays usable. |
| STT stream fails | Reopened up to `Retry.STTMaxReconnects` consecutive times, then the session closes with the error. |
| Tool fails | Reported to the model as an error result. |
| Audio input fails | The session closes with the error. End of input (`io.EOF`) is not an error. |

## Stream contracts

All provider streams follow the same rules, written on the interfaces:

| Rule | Detail |
|---|---|
| Ownership | The caller that opened a stream must close it. `Close` is idempotent. |
| Cancellation | The context passed at open bounds the whole stream. `Close` also unblocks `Recv`. |
| Concurrency | One goroutine receives. For STT, a second goroutine may push. |
| End | `Recv` returns `io.EOF` at the normal end and a typed error otherwise. |
| Backpressure | `Push` and `WriteFrame` may block; they must return when cancelled. |

## Audio

| Stage | Format | Who converts |
|---|---|---|
| Room → agent | Opus 48 kHz | `server-sdk-go/pkg/media.PCMRemoteTrack` decodes and resamples |
| Into VAD and STT | PCM16 mono, 16 kHz by default | — |
| Out of TTS | PCM16, the provider's rate (24 kHz for the bundled ones) | — |
| Agent → room | Opus 48 kHz | `server-sdk-go/pkg/media.PCMLocalTrack` resamples and encodes |

`roomio` paces writes itself. The upstream track helper accepts unlimited audio and
plays it in real time; `roomio` keeps at most `MaxQueuedAudio` (100 ms) queued, so an
interruption drops at most that much and the played duration is known within that bound.
The TTS format must match the output format; the session checks it at `Start`.

## Observability

| Signal | Where |
|---|---|
| Logs | `slog`, with job and room attributes on job loggers |
| Transcription latency | `TranscriptionMetrics.Delay`: end of speech (VAD) to final transcript |
| End-of-turn latency | `EndOfTurnMetrics.Delay` |
| Model first token | `LLMMetrics.FirstToken`, plus duration, token usage, attempts |
| TTS first audio | `TTSMetrics.FirstAudio`, plus audio duration, characters, attempts |
| Response latency | `ResponseMetrics.FirstAudio`: reply requested to first frame queued |
| Interruption latency | `InterruptionMetrics.Latency`: decision to audio dropped |
| Resource usage | `Worker.Stats` / `WorkerOptions.OnStats`: jobs, load, goroutines, heap. `Session.Stats`: dropped frames and events |

Metrics arrive as `MetricsCollected` events. Exporting them (Prometheus, OpenTelemetry)
is left to the application.

## Dependencies

| Dependency | Why | Notes |
|---|---|---|
| `livekit/server-sdk-go/v2` | Room connection, WebRTC, publishing, server API clients | Official. Not reimplemented. |
| `livekit/protocol` | Worker protocol messages, access tokens | Official. Version chosen by `server-sdk-go`. |
| `livekit/media-sdk` | Named in `roomio` for the PCM sample type | Pulled in by `server-sdk-go/pkg/media`. Needs libopus, libopusfile, libsoxr through cgo. |
| `pion/webrtc/v4` | Track types in room callbacks | Already required by `server-sdk-go`. |
| `gorilla/websocket` | Worker connection, Deepgram streaming | Already required by `server-sdk-go`. |
| `google.golang.org/protobuf` | Encoding worker messages | Already required by `protocol`. |

No other direct dependency. The JSON Schema generator and validator, the sentence
splitter and the SSE readers are small and local on purpose. The Anthropic and OpenAI
providers use `net/http` rather than the vendors' SDKs, to keep the module graph small.

## Existing Go projects

Surveyed on 2026-11-12, by reading code and metadata, without running their tests.

| Project | License | Finding |
|---|---|---|
| LiveKit (official) | — | No Go agents framework. `protocol` has the message types and token helpers but no worker client. `livekit/livekit` has the server side and a simulated worker used in its tests. |
| `cavos-io/rtp-agent` | MIT | Most complete: worker protocol, full pipeline, many adapters. Experimental, no adoption, tied to cgo audio, portaudio and ONNX Runtime. |
| `infinityscroll/livekit-agents-go` | Apache-2.0 | Worker protocol and sessions. Published in about a day, not maintained since. |
| `am-sokolov/livekit-agent-sdk-go` | Apache-2.0 | Worker protocol only. No speech pipeline. Stale. |
| `chriscow/livekit-agents-go`, `beingPro007/LK-go-agent-sdk` | none | No license file, so nothing can be reused. |
| `gojargo/jargo` | BSD-2-Clause | A Pipecat port. LiveKit is a room transport there, not a worker. |

Decision: no code was taken from any of them. None is both maintained and factored so
that a part could be imported, and the worker client is small once `protocol` provides
the messages. Two findings are worth keeping for later:

- `pion/opus` (MIT) now has a pure-Go Opus decoder and a young encoder. A cgo-free audio
  path is plausible. The encoder's quality was not evaluated, so `roomio` stays on the
  upstream helpers for now.
- Pure-Go Silero VAD ports exist (for example `zserge/govad`, MIT). They were not
  evaluated. `vad/silero` runs the official model on ONNX Runtime instead, the same
  model file and runtime as the upstream Python plugin.
