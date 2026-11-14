package agents

import (
	"context"
	"log/slog"
	"sync"

	"github.com/livekit/protocol/livekit"
	lksdk "github.com/livekit/server-sdk-go/v2"
)

// AcceptOptions describes the participant the agent joins the room as.
type AcceptOptions struct {
	// Identity defaults to "agent-<job id>".
	Identity   string
	Name       string
	Metadata   string
	Attributes map[string]string
}

// JobRequest is an offer from the server to run a job. Exactly one of
// Accept or Reject should be called; a request left unanswered when
// WorkerOptions.OnJobRequest returns is rejected. Its methods are safe for
// concurrent use.
type JobRequest struct {
	job      *livekit.Job
	resuming bool
	worker   *Worker

	mu       sync.Mutex
	answered bool

	// slotHeld is true while the request holds a reserved capacity slot.
	// It is guarded by worker.mu.
	slotHeld bool
}

// Info returns the offered job as described by the server.
func (r *JobRequest) Info() *livekit.Job { return r.job }

// Resuming reports whether the job was previously assigned to another worker.
func (r *JobRequest) Resuming() bool { return r.resuming }

func (r *JobRequest) markAnswered() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.answered {
		return false
	}
	r.answered = true
	return true
}

// Reject declines the job; the server may offer it to another worker.
func (r *JobRequest) Reject() error {
	if !r.markAnswered() {
		return ErrRequestAnswered
	}
	return r.sendReject()
}

func (r *JobRequest) sendReject() error {
	return r.worker.send(&livekit.WorkerMessage{Message: &livekit.WorkerMessage_Availability{
		Availability: &livekit.AvailabilityResponse{JobId: r.job.Id, Available: false},
	}})
}

// Accept takes the job. It blocks until the server confirms the assignment,
// then starts the entrypoint in its own goroutine and returns. It fails
// with a *TimeoutError when the assignment does not arrive within
// WorkerOptions.AssignmentTimeout, and with ctx.Err() when ctx ends first.
// ctx only bounds the wait: it is not the context of the job.
func (r *JobRequest) Accept(ctx context.Context, opts AcceptOptions) error {
	if !r.markAnswered() {
		return ErrRequestAnswered
	}
	return r.worker.acceptJob(ctx, r, opts)
}

// Job is the handle of a running job, passed to the entrypoint. Its methods
// are safe for concurrent use.
type Job struct {
	job    *livekit.Job
	url    string
	token  string
	worker *Worker
	log    *slog.Logger
	accept AcceptOptions

	mu        sync.Mutex
	room      *lksdk.Room
	onClose   []func()
	cancelJob context.CancelFunc
}

// Info returns the job as described by the server: room, dispatch id,
// metadata, agent name. The message must not be modified.
func (j *Job) Info() *livekit.Job { return j.job }

// ID returns the job id.
func (j *Job) ID() string { return j.job.Id }

// RoomName returns the name of the job's room.
func (j *Job) RoomName() string { return j.job.GetRoom().GetName() }

// Metadata returns the dispatch metadata.
func (j *Job) Metadata() string { return j.job.Metadata }

// Identity returns the participant identity the agent joins with.
func (j *Job) Identity() string { return j.accept.Identity }

// Logger returns a logger tagged with the job and room.
func (j *Job) Logger() *slog.Logger { return j.log }

// Room returns the connected room, or nil before Connect.
func (j *Job) Room() *lksdk.Room {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.room
}

// Connect joins the job's room with the token issued by the server.
// Callbacks must be supplied here because the Go SDK only accepts them
// before joining. The room is disconnected automatically when the job ends.
func (j *Job) Connect(ctx context.Context, cb *lksdk.RoomCallback, opts ...lksdk.ConnectOption) (*lksdk.Room, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.room != nil {
		return nil, ErrAlreadyConnected
	}
	room := lksdk.NewRoom(cb)
	if err := room.JoinWithContextAndToken(ctx, j.url, j.token, opts...); err != nil {
		return nil, err
	}
	j.room = room
	return room, nil
}

// AddCleanup registers a function to run when the job ends: after the
// entrypoint returned (or panicked) and before the room is disconnected.
// Functions run in reverse registration order, like deferred calls. A
// function registered after the job ended never runs.
func (j *Job) AddCleanup(fn func()) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.onClose = append(j.onClose, fn)
}

// Shutdown asks the job to end by cancelling the entrypoint's context. It
// does not wait for the entrypoint to return and may be called repeatedly.
func (j *Job) Shutdown() { j.cancelJob() }

func (j *Job) cleanup() {
	j.mu.Lock()
	hooks := j.onClose
	room := j.room
	j.onClose = nil
	j.mu.Unlock()
	for i := len(hooks) - 1; i >= 0; i-- {
		func() {
			defer func() {
				if r := recover(); r != nil {
					j.log.Error("shutdown hook panicked", "panic", r)
				}
			}()
			hooks[i]()
		}()
	}
	if room != nil {
		room.Disconnect()
	}
}
