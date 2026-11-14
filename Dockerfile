# syntax=docker/dockerfile:1
#
# Builds one example agent. The room audio path uses cgo (libopus, libsoxr),
# so both stages need the native libraries.
#
#   docker build -t livekit-agents-go-example .
#   docker build --build-arg EXAMPLE=voice-assistant -t livekit-agents-go-voice .
#   docker run --rm --env-file .env livekit-agents-go-example
#
# The image follows the LiveKit Cloud requirements for agent images (glibc
# base, CA certificates, non-root user, fixed entrypoint, no secrets), so the
# same file is used for managed deployment: see docs/DEPLOYMENT.md.
#
# The default example is the named agent "support-agent", which only joins
# rooms it is explicitly dispatched to. LiveKit Cloud builds cannot pass
# build arguments: change the default below to deploy another example.

ARG GO_VERSION=1.27
FROM golang:${GO_VERSION}-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libopus-dev libopusfile-dev libsoxr-dev \
    && apt-get clean
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG EXAMPLE=explicit-dispatch
RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/agent ./examples/${EXAMPLE}

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libopus0 libopusfile0 libsoxr0 \
    && apt-get clean \
    && useradd --system --uid 10001 --no-create-home agent
COPY --from=build /out/agent /usr/local/bin/agent
WORKDIR /app
USER agent
# Health check server, enabled automatically when hosted by LiveKit Cloud.
EXPOSE 8081
# The worker only makes outbound connections (WebSocket to LiveKit, WebRTC,
# HTTPS to providers); no port needs to be published.
# SIGTERM drains the worker: give the container a stop timeout longer than
# WorkerOptions.DrainTimeout.
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/local/bin/agent"]
