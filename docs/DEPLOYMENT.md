# Deployment

There are two ways to run an agent with LiveKit Cloud. They are kept apart here
because their status differs.

| | Where the agent runs | Status with this SDK |
|---|---|---|
| **Connectivity** | Your own machine, server or container platform, connected to a self-hosted LiveKit server or a LiveKit Cloud project | Supported and tested with both |
| **Managed deployment** | LiveKit Cloud builds your Dockerfile and runs the agent for you | Implemented on the worker side and tested locally. **Not yet run on LiveKit Cloud**: see [Managed deployment](#managed-deployment-on-livekit-cloud) |

## Connecting to a LiveKit server

The worker only opens outbound connections:

```
worker ──► wss://<server>/agent        worker protocol (WebSocket)
worker ──► <server> WebRTC             one room connection per job (UDP, or TCP/TURN fallback)
worker ──► provider APIs               HTTPS and WSS
```

No inbound port is needed.

| Variable | Value |
|---|---|
| `LIVEKIT_URL` | `ws://localhost:7880` for `livekit-server --dev`, `wss://<project>.livekit.cloud` for LiveKit Cloud |
| `LIVEKIT_API_KEY` | `devkey` for the dev server, or your project's API key |
| `LIVEKIT_API_SECRET` | `secret` for the dev server, or your project's API secret |
| `LIVEKIT_AGENT_NAME` | Optional. Default agent name when `WorkerOptions.AgentName` is empty |

They can also be set in `WorkerOptions`. The secret is a `Secret` value and is never
logged.

### LiveKit Cloud

Nothing in the worker is specific to self-hosted servers: it uses the same `/agent`
endpoint, token grant and messages as the upstream workers. On 2026-11-13 the
integration test `TestVoiceAgentVerticalSlice` passed against a Cloud project (server
1.13.10): registration, explicit dispatch, room join, audio in both directions, an
interruption and shutdown. A manual voice call through the Agents Playground worked too.
Still unverified there:

- Automatic dispatch (a worker without an agent name joins every new room of the
  project, so it was not run against a shared project).
- Cloud-specific limits on concurrent agent sessions.
- Features that depend on newer protocol messages than the pinned `protocol` version,
  listed in [COMPATIBILITY.md](COMPATIBILITY.md).

### Running it yourself

| Concern | Recommendation |
|---|---|
| Image | The provided `Dockerfile` builds an example. Copy it and point `EXAMPLE` (or the build path) at your agent. Runtime needs `libopus0`, `libopusfile0`, `libsoxr0`, CA certificates and, for the Silero VAD and the Smart Turn model, the ONNX Runtime library (`scripts/install-onnxruntime.sh`). |
| Shutdown | `SIGTERM` starts a drain. Set the platform's termination grace period above `WorkerOptions.DrainTimeout`, which should exceed your longest expected conversation (default 1 minute). |
| Capacity | Set `WorkerOptions.MaxConcurrentJobs` from load tests. Each job holds one WebRTC connection, an Opus encoder and decoder, and the provider streams. Add `LoadThreshold` with a `LoadFunc` to also decline jobs on CPU or any other signal. |
| Scaling | Run several replicas. The server spreads jobs over registered workers that report themselves available. Scale on `WorkerStats.Load` (exported through `OnStats`). |
| Health | Set `WorkerOptions.HealthAddr` (for example `:8081`). `GET /` answers 200 while the worker is registered, 503 before the first registration and after 30 s of disconnection. `GET /worker` returns worker info as JSON. |
| Isolation | Jobs share one process. A panic in a job is contained, a crash (out of memory, a fault in native code) takes every job of that worker down. Keep `MaxConcurrentJobs` moderate and run more replicas. |
| Secrets | Pass keys through the environment or your secret manager. Use `agents.NewRedactingHandler` on your logger. |

## Managed deployment on LiveKit Cloud

LiveKit Cloud can build a container from your source and Dockerfile and run it, with
scaling, secrets and logs handled by the platform.

### What a hosted worker must do

These come from reading the upstream Python worker (`livekit-agents` 1.8.11) and the
LiveKit build documentation. All are implemented.

| Requirement | Source | This SDK |
|---|---|---|
| Send the `worker_token` query parameter from `LIVEKIT_WORKER_TOKEN` when connecting | Upstream worker | `WorkerOptions.WorkerToken`, read from the environment. Never logged |
| Send the deployment name from `LIVEKIT_AGENT_DEPLOYMENT` at registration | Upstream worker | `WorkerOptions.Deployment` |
| Let the platform override the agent name with `LIVEKIT_AGENT_NAME_OVERRIDE` | Upstream worker | Applied in `NewWorker` |
| Serve a health check on port 8081: `GET /` and `GET /worker` | Upstream worker (production default port) | On by default in hosted mode |
| Report CPU-based load, full at 0.7 | Upstream worker forces its defaults when hosted | Default in hosted mode: cgroup-aware CPU usage, threshold 0.7. `MaxConcurrentJobs` still applies |
| Read `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` from the environment | Build documentation: injected at runtime, must not be set | Default behaviour |
| glibc image, CA certificates, non-root user, fixed `ENTRYPOINT`, no secrets in the image | Build documentation | The provided `Dockerfile` |

A worker is in hosted mode when it has a worker token (`Worker.Hosted`).

### How to deploy: `lkagent`

The official CLI cannot create or deploy a Go agent from source. `lk agent create` and
`lk agent deploy` (CLI 2.19.2) look for `package.json` or Python project files, and then
check the version of the official Python or Node.js SDK. A Go project fails the first
check with "no agent project detected". Deploying a prebuilt image (`--image`) skips
those checks, but the documentation reserves it for the Enterprise plan.

`cmd/lkagent` fills that gap. It calls the same LiveKit Cloud API as the CLI, through
the official Go client `server-sdk-go/pkg/cloudagents` (the library the CLI itself
uses), without the language checks. It is a separate Go module so that its build-tool
dependencies stay out of the SDK's module.

```sh
export LIVEKIT_URL=wss://<project>.livekit.cloud LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...

go -C cmd/lkagent run . regions                                   # where agents can run
go -C cmd/lkagent run . create -region us-east -secrets-file ../../.env ../..
go -C cmd/lkagent run . deploy ../..                              # ship a new version
go -C cmd/lkagent run . list
go -C cmd/lkagent run . delete -id <agent id>
```

| Step | What happens |
|---|---|
| Checks | The directory must have a `Dockerfile` whose final stage is not Alpine, has a non-root `USER` and an `ENTRYPOINT` or `CMD` |
| Secrets | `-secrets-file` takes a `.env`-style file. `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` and empty values are skipped. Only names are printed |
| Upload | The directory is uploaded as the build context, minus what `.dockerignore` and `.gitignore` exclude, `.env*` files and `livekit.toml` |
| Build | LiveKit Cloud builds the `Dockerfile` (10 minute limit, 1 GB context limit) and streams the build log |
| Result | `create` prints the agent id and writes `livekit.toml`, in the format of the official CLI |

Because `livekit.toml` is compatible, the rest of the official CLI should work on the
agent: `lk agent status`, `lk agent logs`, `lk agent secrets`, `lk agent rollback`,
`lk agent delete`.

The repository's `Dockerfile` builds `examples/explicit-dispatch` by default: a named
agent, `support-agent`, that only joins rooms it is dispatched to. Cloud builds cannot
pass build arguments, so change the `EXAMPLE` default in the file to deploy another
example, or use your own Dockerfile.

### What is verified

| Item | Status |
|---|---|
| Hosted-mode worker: token, deployment, name override, health endpoints, load threshold | Unit tests against a fake server |
| Image: non-root, health check answering, CPU load reported | Run locally in Docker with a worker token set |
| `lkagent regions`, `lkagent list` | Run against a real Cloud project |
| `lkagent create` and `deploy`: upload, Cloud build of a Go Dockerfile, the agent starting on Cloud | **Not run yet** |
| Whether LiveKit Cloud accepts a worker that is not the official Python or Node.js SDK | **Unknown.** The CLI's SDK check is client-side; whether the platform checks anything else is not documented |
| `lk agent status`, `logs`, `delete` on an agent created by `lkagent` | **Not run yet** |

Until a deployment has been run, treat managed deployment as experimental. It also
depends on LiveKit keeping an undocumented-for-Go path open: the API is public and used
through the official client, but Go agents are not a documented project type.
