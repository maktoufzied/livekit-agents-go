# Compatibility

This is a community implementation. It is not an official LiveKit SDK and it does not
have feature parity with the Python or TypeScript Agents frameworks. This page says
what it was built against, what was checked, and what is missing.

## Baseline

Sources read on 2026-11-12:

| Source | Version or commit | Used for |
|---|---|---|
| [livekit/agents](https://github.com/livekit/agents) (Python) | `livekit-agents` 1.8.11, commit `6aa79078` | Worker protocol flow, constants, session defaults, attribute and topic names |
| [livekit/agents-js](https://github.com/livekit/agents-js) | `@livekit/agents` 1.9.5, commit `4ff47ef7` | Cross-check of names and package layout |
| [livekit/protocol](https://github.com/livekit/protocol) | `main` at `4481621c` (read); `v1.51.1-0.20260910121219-271d9cde3897` (compiled against) | `livekit_agent.proto`, access tokens |
| [livekit/server-sdk-go](https://github.com/livekit/server-sdk-go) | `v2.19.0` (compiled against); `main` at `d837caa1` (read) | Room, tracks, PCM helpers, dispatch client |
| [docs.livekit.io/agents](https://docs.livekit.io/agents/) | Agent dispatch and deployment pages | Dispatch semantics, deployment model |
| `livekit-server` | 1.13.11 | Integration tests |

The module compiles against the `protocol` version that `server-sdk-go` v2.19.0
selects. That version is older than `protocol` `main`, so messages added since are not
available (see the protocol table below).

## Tested environments

| Environment | What ran |
|---|---|
| macOS 27 arm64, Go 1.27.4, libopus 1.6.1, opusfile 0.12, libsoxr 0.1.3 | Unit tests with `-race`, integration tests against `livekit-server` 1.13.11 |
| Linux arm64 in Docker (`golang:1.27-bookworm` → `debian:bookworm-slim`) | Image build; the example worker registered with `livekit-server` 1.13.11; the Silero model loads at startup |
| Linux amd64 (GitHub Actions, Ubuntu 26.04) | Unit tests with `-race`, integration tests against `livekit-server`, image build, on every push |
| Windows | Not tested |
| LiveKit Cloud (server 1.13.10), from macOS | `TestVoiceAgentVerticalSlice`: registration, explicit dispatch, room join, audio both ways, interruption, shutdown. Automatic dispatch was not run there |
| Live provider APIs (Deepgram, Anthropic, OpenAI), 2026-11-13 | Provider `e2e` tests and `test/e2e` (pipeline with Deepgram STT and TTS and the Anthropic LLM). |
| Manual call, 2026-11-13: browser (Agents Playground) → LiveKit Cloud → `examples/explicit-dispatch` with Deepgram and `claude-haiku-5-5` | One conversation with a human voice: replies, interruption, transcripts in the chat panel. No false interruption observed. Noisy rooms and speakerphone echo were not explored |

## Worker protocol coverage

| Message | Direction | Status |
|---|---|---|
| `RegisterWorkerRequest` / `RegisterWorkerResponse` | both | Implemented, verified against `livekit-server` |
| `AvailabilityRequest` / `AvailabilityResponse` | both | Implemented, verified (accept and decline) |
| `JobAssignment` | server → worker | Implemented, verified |
| `UpdateWorkerStatus` (status, load, job count) | worker → server | Implemented, verified against a fake server; accepted by `livekit-server` |
| `UpdateJobStatus` (running, success, failed) | worker → server | Implemented, same as above |
| `JobTermination` | server → worker | Implemented, verified against a fake server only |
| `MigrateJobRequest` (job ids after reconnect) | worker → server | Implemented, verified against a fake server only |
| WebSocket ping / pong | both | Implemented |
| `WorkerPing` / `WorkerPong` | both | Not sent. The upstream Python worker does not send it either |
| `SimulateJobRequest` | worker → server | Not implemented |
| `AvailabilityResponse.terminate`, `supports_resume` | worker → server | Not set |
| `RegisterWorkerRequest.deployment` | worker → server | Set from `LIVEKIT_AGENT_DEPLOYMENT`; verified against a fake server only |
| `RegisterWorkerRequest.namespace` | worker → server | Not set |
| `JobJoinTiming`, `UpdateWorkerStatus.draining` and `seq`, `MigrateJobRequest.jobs` | worker → server | Not available in the pinned `protocol` version |
| `GoAway`, `CreateJobToken*` | both | Not available in the pinned `protocol` version |
| Agent HTTP endpoints (WebTransport data plane) | both | Not implemented |
| `worker_token` query parameter (LiveKit Cloud hosted agents) | worker → server | Implemented; verified against a fake server only |

Job types: `JT_ROOM` is the default and the only one exercised. `JT_PUBLISHER` and
`JT_PARTICIPANT` can be requested with `WorkerOptions.JobType` but are untested.

## Feature matrix

"Yes" means implemented and covered by tests in this repository. It does not mean
identical behaviour to upstream.

| Capability | Python / TypeScript | This SDK |
|---|---|---|
| Worker registration, dispatch, load reporting | Yes | Yes |
| Explicit dispatch by agent name | Yes | Yes |
| Graceful drain | Yes | Yes |
| Job isolation | Process per job, prewarmed | Goroutine per job, no isolation |
| Dev mode, hot reload, console mode, CLI | Yes | No |
| Worker HTTP health endpoint | Yes | Yes (`GET /`, `GET /worker`) |
| LiveKit Cloud managed deployment | Yes, with `lk agent` | Experimental, with `cmd/lkagent`; not yet run on Cloud |
| STT → LLM → TTS pipeline | Yes | Yes |
| Speech-to-speech realtime models | Yes | No (interfaces only) |
| VAD | Silero (model) | Silero (same model file, see below) and an energy detector |
| Turn detection | VAD, STT, realtime, LiveKit turn-detector model (text) | VAD, STT, manual, Smart Turn model (audio, see below) |
| Interruptions | Yes | Yes |
| False-interruption handling | Pause, then resume when no transcript follows (2 s) | Same idea, same default; see [ARCHITECTURE.md](ARCHITECTURE.md#interruption). Checked with fake providers only |
| Adaptive interruption detection (model that tells backchannels from interruptions) | Yes | No |
| History truncation on interruption | Aligned transcripts when the TTS provides them | Per sentence plus played fraction |
| Preemptive generation | Yes: model request, optionally synthesis too; skipped after 10 s of user speech | Model request only, on by default, at most 3 per turn; no limit on speech length. See [ARCHITECTURE.md](ARCHITECTURE.md#preemptive-generation) |
| Tool calling | Yes | Yes |
| Agent handoffs, tasks, workflows | Yes | No |
| `Say`, `GenerateReply` | Yes | Yes |
| Transcription forwarding (`lk.transcription`) | Yes, synchronized | Per sentence; checked manually in the Agents Playground |
| Agent state attribute (`lk.agent.state`) | Yes | Yes |
| Text input from the room (`lk.chat`) | Yes | No |
| Video and vision | Yes | No |
| Avatars, background audio, noise cancellation | Yes | No |
| Telephony helpers (SIP, DTMF) | Yes | No |
| Metrics | Yes, with OpenTelemetry | Events only; export is up to the application |
| Session recording, evals, testing helpers | Yes | Fake providers only |
| LiveKit Inference gateway | Yes | No |
| Providers | Dozens | Deepgram (STT, TTS), Anthropic (LLM), OpenAI (LLM, TTS) |
| Multi-participant rooms | Selectable participant | First non-agent participant, or one identity |
| End-to-end encryption | Yes | No |

## Behaviour differences to be aware of

| Topic | Upstream | This SDK |
|---|---|---|
| Load | CPU-based, threshold 0.7 in production | `active jobs / MaxConcurrentJobs`, full at the maximum. In hosted mode: CPU-based, threshold 0.7, plus the job maximum |
| Default drain timeout | 1 hour | 1 minute (`WorkerOptions.DrainTimeout`) |
| Status interval | 1.5 s | 1.5 s |
| Agent identity | `agent-<job id>` | Same |
| Participant attributes | `lk.agent.name`, `lk.agent.state` | Same |
| Endpointing defaults | 0.5 s min, 3 s max | Same |
| Interruption default | 0.5 s of speech, 0 words | Same |
| Tool steps per turn | 3 | 3 |
| TTS input | Streaming when the provider supports it | One request per sentence |

## Silero VAD

`vad/silero` follows `livekit-plugins-silero` 1.8.12.

| Item | Upstream | This SDK |
|---|---|---|
| Model file | `silero_vad.onnx`, SHA-256 `1a153a22…88e3` | The same file, embedded. It is also the file of Silero VAD release 6.2.3 |
| Runtime | ONNX Runtime, CPU, one thread, no spinning | Same settings, ONNX Runtime 1.32.0 through `yalue/onnxruntime_go` (built with the 1.31 headers; 1.31.1 was tested too) |
| Window and context | 512 + 64 samples at 16 kHz, 256 + 32 at 8 kHz | Same |
| Defaults | 50 ms speech, 550 ms silence, activation 0.5, deactivation 0.35 | Same |
| Output smoothing | Exponential filter, 0.35 | Same |
| Resampling to the model rate | libsoxr, lowest quality | Averaging or linear interpolation, in Go |
| Speech audio buffer, prefix padding, per-window inference events | Yes | No: the detector reports speech boundaries only |

Checked: on a synthesized sentence the start is reported 64 ms after the first word and
the end 576 ms after the last one; silence, a 440 Hz tone, white noise and keyboard-like
clicks produce no event; the pipeline test with live providers passes with it.

Compared with the Python plugin (`livekit-plugins-silero` 1.8.12, ONNX Runtime 1.32.0)
on 2026-11-14. Both sides were fed the same audio in 20 ms frames.

| Clip | Input rate | Windows | Largest probability difference | Speech boundaries |
|---|---|---|---|---|
| Synthesized sentence, 6.3 s | 16 kHz | 196 | 0.000001 | 2 of 2 identical |
| Apollo 11, Neil Armstrong on the Moon (NASA, public domain): real voice over radio noise, 13 s | 16 kHz | 405 | 0.000001 | 7 of 7 identical |
| Silero's own test recording, 60 s | 16 kHz | 1875 | 0.000002 | 13 of 13 identical |
| Apollo 11 | 44.1 kHz | 405 | 0.20 | 7 of 7 identical |
| Apollo 11 | 48 kHz | 405 | 0.07 | 6 of 7 identical, one end 32 ms earlier |
| Apollo 11 | 8 kHz, model at 16 kHz | 405 | 0.26 | 5 of 7 identical, two ends 32 ms earlier |

At 16 kHz neither side resamples and the results are the same to the precision of the
reference files. `roomio` delivers 16 kHz by default, so this is the path a room agent
uses. At other rates the two resamplers differ (libsoxr upstream, averaging or linear
interpolation here): probabilities move on some windows, every start of speech stays on
the same window, and an end of speech can move by one window (32 ms).

The first two rows are kept as a test, `TestMatchesUpstreamPlugin`, with references
from `vad/silero/testdata/reference.py`. The other rows were one-off runs. Not checked:
a recording from a noisy room with several speakers or music.

## End-of-turn model

Upstream uses `livekit-plugins-turn-detector`, a small language model that reads the
transcript. Its weights are under the LiveKit Model License, which allows them only
"together with the LiveKit Agents framework" and not "with any other frameworks". This
SDK is another framework, so it does not load that model.

`turn/smartturn` runs [Smart Turn](https://github.com/pipecat-ai/smart-turn) v3.2
instead (Pipecat project, BSD 2-Clause). The two models are not equivalent:

| Item | LiveKit turn detector | Smart Turn (this SDK) |
|---|---|---|
| Input | Transcript of the last turns | Audio of the current turn, last 8 s |
| Sees | Words and grammar | Tone, pace and words as they sound |
| Depends on the STT | Yes | No |
| Languages | English model, multilingual model | 23 |
| Size | 66 MB (English), 396 MB (multilingual) | 8.7 MB |

`turn/smartturn` follows the reference inference code of the Smart Turn repository
(`inference.py` at commit `4e12f2bb`): Whisper log-mel features over the last 8 seconds
with zero-mean, unit-variance input, then the ONNX model. The feature extraction is
written in Go.

Compared with that Python code (ONNX Runtime 1.32.0, `transformers` 4.48.2) on
2026-11-14, on 28 cases: four clips (two synthesized sentences of the Smart Turn
documentation, the synthesized sentence of the VAD tests and the Apollo 11 recording),
cut at different points.

| What | Result |
|---|---|
| Model input (80 × 800 values) | Largest difference 0.000006 |
| Probability, unquantized model (`smart-turn-v3.2-gpu.onnx`) | 28 of 28 equal, largest difference 0.0000015 |
| Probability, embedded model (`smart-turn-v3.2-cpu.onnx`, quantized to 8 bits) | 24 of 28 equal; the other 4 differ by 0.01, 0.03, 0.08 and 0.21 |
| Decisions at threshold 0.5, embedded model | 28 of 28 the same (one case sits exactly on 0.5 here and at 0.53 in Python) |

The embedded model is unstable by itself: the Python code moves by the same amounts
(up to 0.2 on the same cases) when random noise of 0.000005 is added to its own input.
The differences are that effect, not a difference in the computation, which the
unquantized model shows. The unquantized model is 32 MB and is not embedded; pass it as
`Options.Model` to use it.

What the model answers, and what the session does with it:

| Audio | Probability | Session (Silero VAD, delays 200 ms and 2 s) |
|---|---|---|
| "I can't seem to, um..." | 0.005, also with 1 s of silence after it | Replies after 2.0 s, the maximum delay |
| "I can't seem to, um, find the return label." | 0.98, also with 1 s of silence after it | Replies after 0.2 s, the minimum delay |
| A sentence cut in the middle of a word, no silence after the cut | 0.02 to 0.11 | — |
| The same cuts followed by silence | 0.9 and above once the silence reaches 0.2 s (two cuts) or 1 s (one cut) | — |

The last row is why the session gives the model the turn up to 200 ms after the end of
the speech and not the silence that follows: a long silence makes speech that stops
abruptly sound finished. A natural hesitation stays low whatever the silence.

The first, third and fourth rows of the comparison are kept as tests
(`TestFeaturesMatchReference`, `TestMatchesReference`); the second runs when
`SMARTTURN_FP32_MODEL` points at the unquantized model. The session behaviour is
`TestSessionWaitsForUnfinishedSentence`. References come from
`turn/smartturn/testdata/reference.py`. Not checked: recordings of real people
hesitating (the two sentences are synthesized voices), other languages and a live call.

The numbers above are from arm64 (Apple M4). ONNX Runtime computes 8-bit models
differently on x86: on the CI machine (Linux amd64) the embedded model gives the same
value as the arm64 reference on 10 of the 28 cases and decides differently on 3 of the
unclear ones (cuts in the middle of a sentence). The clear cases, the two test sentences
and the session behaviour are the same on both. The unquantized model is compared on
both platforms and CI downloads it for that.

## Provider notes

All three providers were written against the vendors' documentation as of 2026-11-12.
Their unit tests use local servers that replay documented message shapes. The opt-in
`e2e` tests passed against the live APIs on 2026-11-13 with the default models.

| Provider | API | Notes |
|---|---|---|
| Deepgram STT | `wss://api.deepgram.com/v1/listen` | Default model `nova-3`. Interim and final results, `SpeechStarted`, `speech_final` and `UtteranceEnd` mapped to end of speech, `KeepAlive`, `CloseStream`. Single channel only. |
| Deepgram TTS | `POST /v1/speak`, `linear16`, no container | Default voice `aura-2-thalia-en`, 24 kHz. Streams the response body. |
| Anthropic LLM | `POST /v1/messages`, streaming | Default model `claude-opus-5-5` with low effort. For a lower time to first token set `Model` to `claude-sonnet-5-5` or `claude-haiku-5-5`. Sends a server-side refusal fallback option with a beta header on recent models; disable it with `DisableRefusalFallback` behind gateways. Forced tool use falls back to a system instruction on models that reject it. |
| OpenAI LLM | `POST /chat/completions`, streaming | Works with OpenAI-compatible servers through `BaseURL`. Default model `gpt-6-luna` with `reasoning_effort: "none"`, which that model needs for tool calls on this endpoint. |
| OpenAI TTS | `POST /audio/speech`, `pcm` | 24 kHz mono. OpenAI lists every model of this endpoint for shutdown on 2027-01-06; prefer Deepgram TTS for new work. |

## Roadmap

| Milestone | Content |
|---|---|
| 0.2 Hardening | First managed deployment on LiveKit Cloud with `lkagent`. More real-voice testing (noise, echo, several languages). |
| 0.3 Turn quality | Silero VAD: done. End-of-turn model (Smart Turn): done. False-interruption recovery: done. Preemptive generation: done. |
| 0.4 Realtime | Wire `realtime.Model` into the session, with one provider. |
| 0.5 Audio | Streaming-input TTS with aligned transcripts. Evaluate a cgo-free Opus path with `pion/opus`. |
| Later | Handoffs and tasks, text input, telephony helpers, video, more providers, OpenTelemetry export, newer `protocol` messages (`GoAway`, join timing, job tokens). |
