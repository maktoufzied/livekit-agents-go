#!/usr/bin/env bash
# Runs the integration tests against a LiveKit server.
#
# Uses the server at LIVEKIT_URL if one answers; otherwise starts a local
# dev server (the `livekit-server` binary if installed, else Docker) and
# stops it afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."

export LIVEKIT_URL="${LIVEKIT_URL:-ws://localhost:7880}"
export LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
export LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
http_url="$(echo "$LIVEKIT_URL" | sed -e 's#^ws://#http://#' -e 's#^wss://#https://#')"

up() { curl -fsS -o /dev/null --max-time 2 "$http_url"; }

stop_server() { :; }
if ! up; then
  if [ "$LIVEKIT_URL" != "ws://localhost:7880" ]; then
    echo "no LiveKit server answers at $LIVEKIT_URL" >&2
    exit 1
  fi
  log="$(mktemp)"
  if command -v livekit-server >/dev/null 2>&1; then
    echo "starting livekit-server --dev (log: $log)"
    livekit-server --dev >"$log" 2>&1 &
    server_pid=$!
    stop_server() { kill "$server_pid" >/dev/null 2>&1 || true; }
  elif command -v docker >/dev/null 2>&1; then
    echo "starting livekit/livekit-server in Docker"
    container="$(docker run -d --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
      livekit/livekit-server --dev --bind 0.0.0.0 --node-ip 127.0.0.1)"
    stop_server() { docker stop "$container" >/dev/null 2>&1 || true; }
  else
    echo "install livekit-server (https://docs.livekit.io/home/self-hosting/local/) or Docker" >&2
    exit 1
  fi
  trap stop_server EXIT
  for _ in $(seq 1 50); do
    if up; then break; fi
    sleep 0.2
  done
  up || { echo "LiveKit server did not start; see $log" >&2; exit 1; }
fi

go test -tags integration -count=1 -v ./test/integration/ "$@"
