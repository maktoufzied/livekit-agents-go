package agents

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/livekit/protocol/auth"
	"github.com/livekit/protocol/livekit"
	"github.com/livekit/protocol/utils/hwstats"
	"google.golang.org/protobuf/proto"

	"github.com/maktoufzied/livekit-agents-go/providers"
)

// Version is the SDK version reported to the LiveKit server.
const Version = "0.1.0"

// attributeAgentName is the participant attribute upstream agents set so
// clients can tell which agent joined.
const attributeAgentName = "lk.agent.name"

// Entrypoint runs one job. ctx is cancelled when the server terminates the
// job, when Job.Shutdown is called, or when the worker stops after
// its drain timeout. The job ends when the entrypoint returns; a non-nil
// error or a panic marks the job as failed.
type Entrypoint func(ctx context.Context, job *Job) error

// WorkerOptions configures a Worker. Zero values select the defaults.
type WorkerOptions struct {
	// URL, APIKey and APISecret default to the LIVEKIT_URL,
	// LIVEKIT_API_KEY and LIVEKIT_API_SECRET environment variables.
	URL       string
	APIKey    string
	APISecret Secret

	// AgentName enables explicit dispatch: when set, the worker only
	// receives jobs dispatched to that name (agent dispatch API or room
	// configuration in the participant token). When empty, the server
	// dispatches the worker automatically to every new room.
	//
	// As in the upstream workers, the LIVEKIT_AGENT_NAME_OVERRIDE
	// environment variable (set by the platform) takes precedence over
	// this field, and LIVEKIT_AGENT_NAME is the default when it is empty.
	AgentName string

	// WorkerToken identifies a worker hosted by LiveKit Cloud (managed
	// deployment). The platform provides it in LIVEKIT_WORKER_TOKEN, which
	// is the default. A worker with a token runs in hosted mode: see
	// HealthAddr and LoadThreshold.
	WorkerToken Secret
	// Deployment is the hosted deployment name, sent at registration.
	// Defaults to LIVEKIT_AGENT_DEPLOYMENT. Empty means production.
	Deployment string

	// HealthAddr is the listen address of the HTTP health server, for
	// example ":8081". GET / answers 200 while the worker is registered
	// (503 otherwise) and GET /worker describes the worker as JSON, like
	// the upstream workers. Empty disables it, except in hosted mode
	// where it defaults to ":8081".
	HealthAddr string
	// HealthGracePeriod is how long a registered worker may stay
	// disconnected before the health check fails. Default 30s (as upstream).
	HealthGracePeriod time.Duration

	// LoadFunc returns the worker load in [0, 1], reported to the server.
	// By default the load is active jobs / MaxConcurrentJobs; in hosted
	// mode it is the CPU usage of the machine or container, which is what
	// the upstream workers report and what LiveKit Cloud scales on. It is
	// called often and must be fast; it must not call Worker methods.
	LoadFunc func() float64
	// LoadThreshold is the load at or above which the worker reports
	// itself full and declines jobs, in addition to the MaxConcurrentJobs
	// limit. Zero disables it, except in hosted mode where it defaults to
	// 0.7 (as upstream).
	LoadThreshold float64

	Entrypoint Entrypoint
	// OnJobRequest decides whether to take a job. The default accepts
	// every request with default AcceptOptions.
	OnJobRequest func(ctx context.Context, req *JobRequest) error

	// MaxConcurrentJobs bounds the jobs running at once. When reached the
	// worker reports itself full and declines requests. Default 16.
	MaxConcurrentJobs int
	// JobType defaults to one job per room (JT_ROOM).
	JobType livekit.JobType
	// Permissions are the room permissions requested for the agent.
	// The default allows publishing, subscribing and data.
	Permissions *livekit.ParticipantPermission

	// DrainTimeout is how long Run waits for running jobs after its
	// context is cancelled, before cancelling them. Default 1 minute.
	DrainTimeout time.Duration
	// JobShutdownTimeout is how long a cancelled job gets to return from
	// its entrypoint before the worker stops waiting for it. Default 10s.
	JobShutdownTimeout time.Duration
	// MaxRetries bounds consecutive failed connection attempts. Default 16.
	MaxRetries int
	// AssignmentTimeout is how long to wait for the server to confirm an
	// accepted job. Default 7.5s (as upstream).
	AssignmentTimeout time.Duration
	// StatusInterval is the load reporting period. Default 1.5s (as upstream).
	StatusInterval time.Duration
	// PingInterval is the WebSocket keep-alive period. Default 30s.
	PingInterval time.Duration

	// OnStats, when set, receives resource usage every StatsInterval
	// (default 10s).
	OnStats       func(WorkerStats)
	StatsInterval time.Duration

	// Logger defaults to slog.Default().
	Logger *slog.Logger
}

// WorkerStats is a snapshot of worker load and resource usage.
type WorkerStats struct {
	WorkerID     string
	Connected    bool
	Draining     bool
	ActiveJobs   int
	ReservedJobs int // accepted, waiting for assignment
	Load         float64
	Goroutines   int
	HeapBytes    uint64
}

// Worker registers with a LiveKit server, receives job requests and runs
// one Entrypoint per accepted job, in-process.
type Worker struct {
	opts WorkerOptions
	log  *slog.Logger

	sendCh chan outMsg

	// jobsCtx is the parent of every job context. It outlives Run's
	// context so that jobs can drain.
	jobsCtx    context.Context
	cancelJobs context.CancelFunc
	jobsWG     sync.WaitGroup

	hosted bool
	cpu    *hwstats.CPUStats

	// unsent is a message that was taken from sendCh but could not be
	// written; the next connection sends it first. Only the writer
	// goroutine of the current connection touches it.
	unsent *outMsg

	mu        sync.Mutex
	running   bool
	draining  bool
	connected bool
	workerID  string
	// registeredOnce and disconnectedAt drive the health check.
	registeredOnce bool
	disconnectedAt time.Time
	// epoch counts registrations. Answers to job requests are only valid
	// on the connection that asked.
	epoch int
	// stopping is set once Run stops waiting for jobs; no job may start
	// after that.
	stopping bool
	jobs     map[string]*Job
	reserved int
	pending  map[string]chan *livekit.JobAssignment
	idle     chan struct{} // closed while no job is active or reserved
}

// NewWorker validates the options.
func NewWorker(opts WorkerOptions) (*Worker, error) {
	if opts.URL == "" {
		opts.URL = os.Getenv("LIVEKIT_URL")
	}
	if opts.APIKey == "" {
		opts.APIKey = os.Getenv("LIVEKIT_API_KEY")
	}
	if opts.APISecret == "" {
		opts.APISecret = providers.SecretFromEnv("LIVEKIT_API_SECRET")
	}
	if v := os.Getenv("LIVEKIT_AGENT_NAME_OVERRIDE"); v != "" {
		opts.AgentName = v
	} else if opts.AgentName == "" {
		opts.AgentName = os.Getenv("LIVEKIT_AGENT_NAME")
	}
	if opts.WorkerToken == "" {
		opts.WorkerToken = providers.SecretFromEnv("LIVEKIT_WORKER_TOKEN")
	}
	if opts.Deployment == "" {
		opts.Deployment = os.Getenv("LIVEKIT_AGENT_DEPLOYMENT")
	}
	hosted := opts.WorkerToken != ""
	switch {
	case opts.URL == "":
		return nil, errors.New("agents: LiveKit URL is required (WorkerOptions.URL or LIVEKIT_URL)")
	case opts.APIKey == "" || opts.APISecret == "":
		return nil, errors.New("agents: LiveKit API key and secret are required (WorkerOptions or LIVEKIT_API_KEY / LIVEKIT_API_SECRET)")
	case opts.Entrypoint == nil:
		return nil, errors.New("agents: WorkerOptions.Entrypoint is required")
	}
	if _, err := agentURL(opts.URL); err != nil {
		return nil, err
	}
	defInt := func(v *int, d int) {
		if *v <= 0 {
			*v = d
		}
	}
	def := func(v *time.Duration, d time.Duration) {
		if *v <= 0 {
			*v = d
		}
	}
	defInt(&opts.MaxConcurrentJobs, 16)
	defInt(&opts.MaxRetries, 16)
	def(&opts.DrainTimeout, time.Minute)
	def(&opts.JobShutdownTimeout, 10*time.Second)
	def(&opts.AssignmentTimeout, 7500*time.Millisecond)
	def(&opts.StatusInterval, 1500*time.Millisecond)
	def(&opts.PingInterval, 30*time.Second)
	def(&opts.StatsInterval, 10*time.Second)
	def(&opts.HealthGracePeriod, 30*time.Second)
	if hosted {
		if opts.HealthAddr == "" {
			opts.HealthAddr = ":8081"
		}
		if opts.LoadThreshold <= 0 {
			opts.LoadThreshold = 0.7
		}
	}
	if opts.Permissions == nil {
		opts.Permissions = &livekit.ParticipantPermission{CanPublish: true, CanSubscribe: true, CanPublishData: true, CanUpdateMetadata: true}
	}
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	w := &Worker{
		opts:    opts,
		log:     opts.Logger,
		sendCh:  make(chan outMsg, 128),
		jobs:    map[string]*Job{},
		pending: map[string]chan *livekit.JobAssignment{},
		idle:    make(chan struct{}),
		hosted:  hosted,
	}
	if hosted && opts.LoadFunc == nil {
		// CPU usage, cgroup-aware in containers. Sampled in the background.
		if cpu, err := hwstats.NewCPUStats(nil); err == nil {
			w.cpu = cpu
		} else {
			w.log.Warn("cannot measure CPU usage, reporting job-based load", "error", err)
		}
	}
	close(w.idle)
	w.jobsCtx, w.cancelJobs = context.WithCancel(context.Background())
	return w, nil
}

// agentURL returns the worker WebSocket endpoint for a LiveKit URL.
func agentURL(raw string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("agents: invalid LiveKit URL: %w", err)
	}
	switch u.Scheme {
	case "http":
		u.Scheme = "ws"
	case "https":
		u.Scheme = "wss"
	case "ws", "wss":
	default:
		return "", fmt.Errorf("agents: invalid LiveKit URL scheme %q", u.Scheme)
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/agent"
	u.RawQuery = ""
	return u.String(), nil
}

// Stats returns a snapshot of load and resource usage. It is safe for
// concurrent use; it briefly stops the world to read memory statistics, so
// call it every few seconds, not in a hot path.
func (w *Worker) Stats() WorkerStats {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	w.mu.Lock()
	defer w.mu.Unlock()
	return WorkerStats{
		WorkerID:     w.workerID,
		Connected:    w.connected,
		Draining:     w.draining,
		ActiveJobs:   len(w.jobs),
		ReservedJobs: w.reserved,
		Load:         w.loadLocked(),
		Goroutines:   runtime.NumGoroutine(),
		HeapBytes:    ms.HeapAlloc,
	}
}

// Hosted reports whether the worker runs as a LiveKit Cloud hosted agent,
// that is, whether a worker token was provided.
func (w *Worker) Hosted() bool { return w.hosted }

// loadLocked returns the load reported to the server. w.mu must be held.
func (w *Worker) loadLocked() float64 {
	var load float64
	switch {
	case w.opts.LoadFunc != nil:
		load = w.opts.LoadFunc()
	case w.cpu != nil:
		load = w.cpu.GetCPULoad()
	default:
		load = float64(len(w.jobs)) / float64(w.opts.MaxConcurrentJobs)
	}
	return min(max(load, 0), 1)
}

// fullLocked reports whether the worker must decline new jobs. w.mu must
// be held.
func (w *Worker) fullLocked() bool {
	if w.draining || len(w.jobs)+w.reserved >= w.opts.MaxConcurrentJobs {
		return true
	}
	return w.opts.LoadThreshold > 0 && w.loadLocked() >= w.opts.LoadThreshold
}

// Run connects to LiveKit and serves jobs until ctx is cancelled or the
// server cannot be reached after MaxRetries attempts.
//
// When ctx is cancelled the worker drains: it reports itself full, stops
// accepting jobs and waits up to DrainTimeout for running jobs. Jobs still
// running after that are cancelled and given JobShutdownTimeout to return.
// Run returns nil after a clean drain.
func (w *Worker) Run(ctx context.Context) error {
	w.mu.Lock()
	if w.running {
		w.mu.Unlock()
		return errors.New("agents: worker already running")
	}
	w.running = true
	w.mu.Unlock()

	if w.cpu != nil {
		defer w.cpu.Stop()
	}
	if w.opts.HealthAddr != "" {
		srv, err := w.startHealthServer()
		if err != nil {
			return err
		}
		defer srv.Close()
	}
	if w.hosted {
		w.log.Info("running as a LiveKit Cloud hosted agent", "deployment", w.opts.Deployment, "agent_name", w.opts.AgentName)
	}

	connCtx, stopConn := context.WithCancel(context.Background())
	defer stopConn()
	connErr := make(chan error, 1)
	go func() { connErr <- w.connectLoop(connCtx) }()

	if w.opts.OnStats != nil {
		go func() {
			t := time.NewTicker(w.opts.StatsInterval)
			defer t.Stop()
			for {
				select {
				case <-t.C:
					w.opts.OnStats(w.Stats())
				case <-connCtx.Done():
					return
				}
			}
		}()
	}

	var runErr error
	select {
	case <-ctx.Done():
	case runErr = <-connErr:
		connErr <- runErr
	}

	// Drain.
	w.mu.Lock()
	w.draining = true
	active := len(w.jobs)
	w.mu.Unlock()
	if runErr == nil {
		w.log.Info("draining worker", "active_jobs", active, "timeout", w.opts.DrainTimeout)
		w.sendStatus()
		select {
		case <-w.waitIdle():
		case <-time.After(w.opts.DrainTimeout):
			w.log.Warn("drain timeout reached, cancelling running jobs")
		case err := <-connErr:
			runErr = err
			connErr <- err
		}
	}
	// From here on no job may start: startJob checks the flag under the
	// same lock, so jobsWG.Add never races with the Wait below.
	w.mu.Lock()
	w.stopping = true
	w.mu.Unlock()
	w.cancelJobs()
	done := make(chan struct{})
	go func() { w.jobsWG.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(w.opts.JobShutdownTimeout):
		w.log.Error("jobs did not stop in time; their entrypoints ignore context cancellation")
	}
	// Give final job status updates a moment to be written.
	w.flush(500 * time.Millisecond)
	stopConn()
	<-connErr
	return runErr
}

// waitIdle returns a channel closed once no job is active or reserved.
func (w *Worker) waitIdle() <-chan struct{} {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.idle
}

// slotsChangedLocked maintains the idle channel. w.mu must be held.
func (w *Worker) slotsChangedLocked() {
	busy := len(w.jobs)+w.reserved > 0
	select {
	case <-w.idle: // currently idle
		if busy {
			w.idle = make(chan struct{})
		}
	default:
		if !busy {
			close(w.idle)
		}
	}
}

func (w *Worker) flush(d time.Duration) {
	deadline := time.Now().Add(d)
	for len(w.sendCh) > 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
}

// outMsg is a queued message and the registration epoch it was queued in.
type outMsg struct {
	msg   *livekit.WorkerMessage
	epoch int
}

// send queues a message for the server. Job status updates are kept across
// reconnections. When the bounded queue is full the message is dropped and
// an error is returned.
func (w *Worker) send(msg *livekit.WorkerMessage) error {
	w.mu.Lock()
	out := outMsg{msg: msg, epoch: w.epoch}
	w.mu.Unlock()
	select {
	case w.sendCh <- out:
		return nil
	default:
		return errors.New("agents: worker send queue full")
	}
}

// ---------------------------------------------------------------------------
// Connection

func (w *Worker) connectLoop(ctx context.Context) error {
	failures := 0
	for {
		start := time.Now()
		registered, err := w.serveConn(ctx)
		if ctx.Err() != nil {
			return nil
		}
		// Only a connection that stayed up counts as a recovery. A server
		// that registers the worker and drops it right away must not be
		// redialled in a tight loop.
		if registered && time.Since(start) >= stableConnection {
			failures = 0
		}
		if failures >= w.opts.MaxRetries {
			return &RegistrationError{Attempts: failures, Err: err}
		}
		// Same schedule as the upstream workers: 0s, 2s, 4s ... capped at 10s.
		delay := min(time.Duration(failures)*2*time.Second, 10*time.Second)
		failures++
		w.log.Warn("worker connection lost, reconnecting", "error", err, "retry_in", delay, "attempt", failures)
		if !sleep(ctx, delay) {
			return nil
		}
	}
}

// stableConnection is how long a registered connection must last for the
// reconnection backoff to start over.
const stableConnection = 10 * time.Second

// serveConn runs one WebSocket connection: register, then exchange
// messages until the connection fails or ctx ends.
func (w *Worker) serveConn(ctx context.Context) (registered bool, err error) {
	endpoint, _ := agentURL(w.opts.URL)
	token, err := auth.NewAccessToken(w.opts.APIKey, w.opts.APISecret.Reveal()).
		SetVideoGrant(&auth.VideoGrant{Agent: true}).
		SetValidFor(time.Hour).
		ToJWT()
	if err != nil {
		return false, fmt.Errorf("sign worker token: %w", err)
	}
	// The worker token only ever appears in the dialed URL, never in
	// errors or logs, which use the bare endpoint.
	dialURL := endpoint
	if w.opts.WorkerToken != "" {
		dialURL += "?worker_token=" + url.QueryEscape(w.opts.WorkerToken.Reveal())
	}
	dialCtx, cancelDial := context.WithTimeout(ctx, 15*time.Second)
	conn, resp, err := websocket.DefaultDialer.DialContext(dialCtx, dialURL, http.Header{"Authorization": {"Bearer " + token}})
	cancelDial()
	if err != nil {
		if resp != nil {
			return false, fmt.Errorf("dial %s: %w (HTTP %d)", endpoint, err, resp.StatusCode)
		}
		return false, fmt.Errorf("dial %s: %w", endpoint, err)
	}
	defer conn.Close()
	// The handshake reads below have their own deadline; closing the
	// connection makes them return at once when ctx ends.
	stopWatch := context.AfterFunc(ctx, func() { conn.Close() })
	defer stopWatch()

	perms := proto.Clone(w.opts.Permissions).(*livekit.ParticipantPermission)
	// The field is deprecated in favour of ParticipantInfo.kind, which the
	// server sets itself. The upstream workers still send it, so older
	// servers may rely on it.
	//lint:ignore SA1019 kept for compatibility with older servers, as upstream does
	perms.Agent = true
	if err := writeProto(conn, &livekit.WorkerMessage{Message: &livekit.WorkerMessage_Register{Register: &livekit.RegisterWorkerRequest{
		Type:               w.opts.JobType,
		AgentName:          w.opts.AgentName,
		Version:            Version,
		Deployment:         w.opts.Deployment,
		AllowedPermissions: perms,
	}}}); err != nil {
		return false, fmt.Errorf("send register: %w", err)
	}
	_ = conn.SetReadDeadline(time.Now().Add(15 * time.Second))
	first, err := readProto(conn)
	if err != nil {
		return false, fmt.Errorf("read register response: %w", err)
	}
	reg := first.GetRegister()
	if reg == nil {
		return false, errors.New("expected a register response as first message")
	}

	w.mu.Lock()
	w.workerID = reg.WorkerId
	w.connected = true
	w.registeredOnce = true
	w.disconnectedAt = time.Time{}
	w.epoch++
	epoch := w.epoch
	var active []string
	for id := range w.jobs {
		active = append(active, id)
	}
	w.mu.Unlock()
	defer func() {
		w.mu.Lock()
		w.connected = false
		w.disconnectedAt = time.Now()
		w.mu.Unlock()
	}()
	w.log.Info("worker registered", "worker_id", reg.WorkerId, "agent_name", w.opts.AgentName,
		"server_version", reg.GetServerInfo().GetVersion(), "region", reg.GetServerInfo().GetRegion())

	// After a reconnection, tell the server which jobs are still running here.
	if len(active) > 0 {
		if err := writeProto(conn, &livekit.WorkerMessage{Message: &livekit.WorkerMessage_MigrateJob{MigrateJob: &livekit.MigrateJobRequest{JobIds: active}}}); err != nil {
			return true, fmt.Errorf("send migrate: %w", err)
		}
	}

	connCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Keep-alive: the read deadline is pushed forward by every pong.
	pongWait := 2*w.opts.PingInterval + 5*time.Second
	_ = conn.SetReadDeadline(time.Now().Add(pongWait))
	conn.SetPongHandler(func(string) error { return conn.SetReadDeadline(time.Now().Add(pongWait)) })

	// Writer: the only goroutine writing to the connection from here on.
	writeErr := make(chan error, 1)
	go func() {
		status := time.NewTicker(w.opts.StatusInterval)
		ping := time.NewTicker(w.opts.PingInterval)
		defer status.Stop()
		defer ping.Stop()
		write := func(out outMsg) error {
			// An answer to a job request is meaningless on another
			// connection: the server asks again there.
			if out.msg.GetAvailability() != nil && out.epoch != epoch {
				return nil
			}
			err := writeProto(conn, out.msg)
			if err != nil && out.msg.GetUpdateJob() != nil {
				// A final job status must not be lost with the connection.
				w.unsent = &out
			}
			return err
		}
		var err error
		if out := w.unsent; out != nil {
			w.unsent = nil
			err = write(*out)
		}
		for err == nil {
			select {
			case out := <-w.sendCh:
				err = write(out)
			case <-status.C:
				err = writeProto(conn, w.statusMessage())
			case <-ping.C:
				err = conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(5*time.Second))
			case <-connCtx.Done():
				_ = conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""), time.Now().Add(time.Second))
				err = connCtx.Err()
			}
		}
		writeErr <- err
		conn.Close() // unblock the reader
	}()
	w.sendStatus()

	for {
		msg, err := readProto(conn)
		if err != nil {
			cancel()
			if werr := <-writeErr; werr != nil && !errors.Is(werr, context.Canceled) {
				err = werr
			}
			return true, err
		}
		_ = conn.SetReadDeadline(time.Now().Add(pongWait))
		switch m := msg.Message.(type) {
		case *livekit.ServerMessage_Availability:
			w.handleAvailability(m.Availability)
		case *livekit.ServerMessage_Assignment:
			w.handleAssignment(m.Assignment)
		case *livekit.ServerMessage_Termination:
			w.handleTermination(m.Termination)
		case *livekit.ServerMessage_Pong:
		default:
			w.log.Debug("ignoring unknown server message")
		}
	}
}

func writeProto(conn *websocket.Conn, msg proto.Message) error {
	b, err := proto.Marshal(msg)
	if err != nil {
		return err
	}
	_ = conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
	return conn.WriteMessage(websocket.BinaryMessage, b)
}

func readProto(conn *websocket.Conn) (*livekit.ServerMessage, error) {
	for {
		typ, data, err := conn.ReadMessage()
		if err != nil {
			return nil, err
		}
		if typ != websocket.BinaryMessage {
			continue
		}
		msg := &livekit.ServerMessage{}
		if err := proto.Unmarshal(data, msg); err != nil {
			return nil, fmt.Errorf("decode server message: %w", err)
		}
		return msg, nil
	}
}

func (w *Worker) statusMessage() *livekit.WorkerMessage {
	w.mu.Lock()
	defer w.mu.Unlock()
	status := livekit.WorkerStatus_WS_AVAILABLE
	if w.fullLocked() {
		status = livekit.WorkerStatus_WS_FULL
	}
	return &livekit.WorkerMessage{Message: &livekit.WorkerMessage_UpdateWorker{UpdateWorker: &livekit.UpdateWorkerStatus{
		Status:   &status,
		Load:     float32(w.loadLocked()),
		JobCount: uint32(len(w.jobs)),
	}}}
}

func (w *Worker) sendStatus() { _ = w.send(w.statusMessage()) }

// ---------------------------------------------------------------------------
// Jobs

func (w *Worker) handleAvailability(req *livekit.AvailabilityRequest) {
	job := req.GetJob()
	if job == nil {
		return
	}
	jr := &JobRequest{job: job, resuming: req.Resuming, worker: w}

	w.mu.Lock()
	full := w.fullLocked()
	if !full {
		// Reserve the slot now so concurrent requests cannot overbook.
		w.reserved++
		jr.slotHeld = true
		w.slotsChangedLocked()
	}
	w.mu.Unlock()
	if full {
		w.log.Info("declining job request: worker is full or draining", "job_id", job.Id)
		_ = jr.Reject()
		return
	}
	w.log.Info("received job request", "job_id", job.Id, "room", job.GetRoom().GetName(), "agent_name", job.AgentName, "resuming", req.Resuming)

	go func() {
		defer func() {
			w.mu.Lock()
			if jr.slotHeld {
				jr.slotHeld = false
				w.reserved--
				w.slotsChangedLocked()
			}
			w.mu.Unlock()
			w.sendStatus()
		}()
		ctx, cancel := context.WithTimeout(w.jobsCtx, 30*time.Second)
		defer cancel()
		var err error
		func() {
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("OnJobRequest panicked: %v", r)
				}
			}()
			if w.opts.OnJobRequest != nil {
				err = w.opts.OnJobRequest(ctx, jr)
			} else {
				err = jr.Accept(ctx, AcceptOptions{})
			}
		}()
		if err != nil {
			w.log.Warn("job request failed", "job_id", job.Id, "error", err)
		}
		if jr.markAnswered() {
			w.log.Warn("job request was not answered, rejecting it", "job_id", job.Id)
			_ = jr.sendReject()
		}
	}()
}

func (w *Worker) acceptJob(ctx context.Context, jr *JobRequest, opts AcceptOptions) error {
	job := jr.job
	if opts.Identity == "" {
		opts.Identity = "agent-" + job.Id
	}
	attrs := map[string]string{attributeAgentName: w.opts.AgentName}
	for k, v := range opts.Attributes {
		attrs[k] = v
	}

	assigned := make(chan *livekit.JobAssignment, 1)
	w.mu.Lock()
	w.pending[job.Id] = assigned
	w.mu.Unlock()
	defer func() {
		w.mu.Lock()
		delete(w.pending, job.Id)
		w.mu.Unlock()
	}()

	if err := w.send(&livekit.WorkerMessage{Message: &livekit.WorkerMessage_Availability{Availability: &livekit.AvailabilityResponse{
		JobId:                 job.Id,
		Available:             true,
		ParticipantIdentity:   opts.Identity,
		ParticipantName:       opts.Name,
		ParticipantMetadata:   opts.Metadata,
		ParticipantAttributes: attrs,
	}}}); err != nil {
		return err
	}

	timer := time.NewTimer(w.opts.AssignmentTimeout)
	defer timer.Stop()
	select {
	case a := <-assigned:
		return w.startJob(jr, a, opts)
	case <-timer.C:
		return &TimeoutError{Op: "job assignment", After: w.opts.AssignmentTimeout}
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (w *Worker) handleAssignment(a *livekit.JobAssignment) {
	id := a.GetJob().GetId()
	w.mu.Lock()
	ch := w.pending[id]
	w.mu.Unlock()
	if ch == nil {
		w.log.Warn("received an assignment for an unknown job", "job_id", id)
		return
	}
	select {
	case ch <- a:
	default:
	}
}

func (w *Worker) handleTermination(t *livekit.JobTermination) {
	w.mu.Lock()
	job := w.jobs[t.JobId]
	w.mu.Unlock()
	if job != nil {
		w.log.Info("server terminated job", "job_id", t.JobId)
		job.cancelJob()
	}
}

// startJob turns the request's reserved slot into a running job.
func (w *Worker) startJob(jr *JobRequest, a *livekit.JobAssignment, accept AcceptOptions) error {
	job := a.Job
	roomURL := w.opts.URL
	if a.Url != nil && *a.Url != "" {
		roomURL = *a.Url
	}
	ctx, cancel := context.WithCancel(w.jobsCtx)
	jc := &Job{
		job:       job,
		url:       roomURL,
		token:     a.Token, // never logged
		worker:    w,
		accept:    accept,
		cancelJob: cancel,
		log:       w.log.With("job_id", job.Id, "room", job.GetRoom().GetName()),
	}
	w.mu.Lock()
	if w.stopping {
		w.mu.Unlock()
		cancel()
		return ErrWorkerStopping
	}
	w.jobsWG.Add(1)
	w.jobs[job.Id] = jc
	if jr.slotHeld {
		// The reservation becomes the job, atomically: counting both
		// would make the worker look fuller than it is.
		jr.slotHeld = false
		w.reserved--
	}
	w.slotsChangedLocked()
	w.mu.Unlock()

	go func() {
		defer w.jobsWG.Done()
		defer cancel()
		w.sendJobStatus(job.Id, livekit.JobStatus_JS_RUNNING, "")
		w.sendStatus()
		jc.log.Info("job started")
		start := time.Now()

		var err error
		func() {
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("entrypoint panicked: %v", r)
					jc.log.Error("entrypoint panicked", "panic", r, "stack", string(debug.Stack()))
				}
			}()
			err = w.opts.Entrypoint(ctx, jc)
		}()
		jc.cleanup()

		w.mu.Lock()
		delete(w.jobs, job.Id)
		w.slotsChangedLocked()
		w.mu.Unlock()

		if err != nil && !errors.Is(err, context.Canceled) {
			jc.log.Error("job failed", "error", err, "duration", time.Since(start))
			w.sendJobStatus(job.Id, livekit.JobStatus_JS_FAILED, err.Error())
		} else {
			jc.log.Info("job finished", "duration", time.Since(start))
			w.sendJobStatus(job.Id, livekit.JobStatus_JS_SUCCESS, "")
		}
		w.sendStatus()
	}()
	return nil
}

func (w *Worker) sendJobStatus(id string, status livekit.JobStatus, errMsg string) {
	if err := w.send(&livekit.WorkerMessage{Message: &livekit.WorkerMessage_UpdateJob{UpdateJob: &livekit.UpdateJobStatus{JobId: id, Status: status, Error: errMsg}}}); err != nil {
		w.log.Warn("could not report job status", "job_id", id, "error", err)
	}
}

// ---------------------------------------------------------------------------
// Health server

// workerProtocolVersion is the agent worker protocol version implemented,
// as reported by GET /worker.
const workerProtocolVersion = 1

func (w *Worker) startHealthServer() (*http.Server, error) {
	ln, err := net.Listen("tcp", w.opts.HealthAddr)
	if err != nil {
		return nil, fmt.Errorf("agents: health server: %w", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", w.handleHealth)
	mux.HandleFunc("GET /worker", w.handleWorkerInfo)
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			w.log.Error("health server stopped", "error", err)
		}
	}()
	w.log.Info("health server listening", "addr", ln.Addr().String())
	return srv, nil
}

// handleHealth mirrors the upstream health check: 503 until the first
// registration, and again once a registered worker has been disconnected
// for longer than the grace period. Short reconnections stay healthy.
func (w *Worker) handleHealth(rw http.ResponseWriter, _ *http.Request) {
	w.mu.Lock()
	registered, connected, since := w.registeredOnce, w.connected, w.disconnectedAt
	w.mu.Unlock()
	if !registered || (!connected && time.Since(since) > w.opts.HealthGracePeriod) {
		http.Error(rw, "not connected to livekit", http.StatusServiceUnavailable)
		return
	}
	_, _ = rw.Write([]byte("OK"))
}

// handleWorkerInfo answers with the fields of the protocol's WorkerInfo
// message, as JSON with proto field names.
func (w *Worker) handleWorkerInfo(rw http.ResponseWriter, _ *http.Request) {
	w.mu.Lock()
	body := fmt.Sprintf(`{"worker_type":%q,"agent_name":%q,"active_jobs":%d,"sdk_version":%q,"worker_load":%.4f,"protocol_version":%d}`,
		w.opts.JobType.String(), w.opts.AgentName, len(w.jobs), Version, w.loadLocked(), workerProtocolVersion)
	w.mu.Unlock()
	rw.Header().Set("Content-Type", "application/json")
	_, _ = rw.Write([]byte(body))
}
