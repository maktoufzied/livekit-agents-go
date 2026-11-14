.PHONY: all fmt fmt-check vet test race integration e2e build docker deps-macos deps-debian

all: fmt-check vet race

fmt:
	gofmt -w .

fmt-check:
	@out="$$(gofmt -l .)"; if [ -n "$$out" ]; then echo "gofmt needed on:"; echo "$$out"; exit 1; fi

vet:
	go vet ./...
	go -C cmd/lkagent vet ./...
	go vet -tags "integration e2e" ./test/... ./providers/...

test:
	go test -count=1 ./...
	go -C cmd/lkagent test -count=1 ./...

race:
	go test -race -count=1 ./...
	go -C cmd/lkagent test -race -count=1 ./...

# Needs a LiveKit server; the script starts a local one if none is running.
integration:
	./scripts/integration.sh

# Opt-in: needs DEEPGRAM_API_KEY and ANTHROPIC_API_KEY or OPENAI_API_KEY.
e2e:
	go test -tags e2e -count=1 -v ./test/e2e/ ./providers/...

build:
	go build -o bin/ ./examples/...

docker:
	docker build -t livekit-agents-go-example .

deps-macos:
	brew install pkg-config opus opusfile libsoxr livekit

deps-debian:
	sudo apt-get install -y pkg-config libopus-dev libopusfile-dev libsoxr-dev
