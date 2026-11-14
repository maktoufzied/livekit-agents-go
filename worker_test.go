package agents_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/livekit/protocol/auth"
	"github.com/livekit/protocol/livekit"
	"google.golang.org/protobuf/proto"

	agents "github.com/maktoufzied/livekit-agents-go"
)

// fakeServer speaks the server side of the LiveKit agent worker protocol.
type fakeServer struct {
	t      *testing.T
	srv    *httptest.Server
	secret string

	mu          sync.Mutex
	conn        *websocket.Conn
	conns       int
	msgs        []*livekit.WorkerMessage
	workerToken string // worker_token query parameter of the last connection
	// rejectUpgrades makes the next N connection attempts fail with 503.
	rejectUpgrades atomic.Int32
}

func newFakeServer(t *testing.T) *fakeServer {
	f := &fakeServer{t: t, secret: "test-secret"}
	up := websocket.Upgrader{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/agent" {
			http.NotFound(rw, r)
			return
		}
		if f.rejectUpgrades.Load() > 0 {
			f.rejectUpgrades.Add(-1)
			http.Error(rw, "unavailable", http.StatusServiceUnavailable)
			return
		}
		// The worker must authenticate with a JWT carrying the agent grant.
		tok := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		v, err := auth.ParseAPIToken(tok)
		if err != nil {
			http.Error(rw, "bad token", http.StatusUnauthorized)
			return
		}
		_, grants, err := v.Verify(f.secret)
		if err != nil || grants.Video == nil || !grants.Video.Agent {
			http.Error(rw, "missing agent grant", http.StatusUnauthorized)
			return
		}
		c, err := up.Upgrade(rw, r, nil)
		if err != nil {
			return
		}
		// First message must be the registration.
		first := f.read(c)
		if first == nil || first.GetRegister() == nil {
			c.Close()
			return
		}
		f.mu.Lock()
		f.msgs = append(f.msgs, first)
		f.workerToken = r.URL.Query().Get("worker_token")
		f.conn = c
		f.conns++
		f.mu.Unlock()
		f.write(c, &livekit.ServerMessage{Message: &livekit.ServerMessage_Register{Register: &livekit.RegisterWorkerResponse{
			WorkerId: "AW_test", ServerInfo: &livekit.ServerInfo{Version: "fake"},
		}}})
		for {
			m := f.read(c)
			if m == nil {
				return
			}
			f.mu.Lock()
			f.msgs = append(f.msgs, m)
			f.mu.Unlock()
		}
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeServer) read(c *websocket.Conn) *livekit.WorkerMessage {
	_, data, err := c.ReadMessage()
	if err != nil {
		return nil
	}
	m := &livekit.WorkerMessage{}
	if err := proto.Unmarshal(data, m); err != nil {
		f.t.Errorf("worker sent an undecodable message: %v", err)
		return nil
	}
	return m
}

func (f *fakeServer) write(c *websocket.Conn, m *livekit.ServerMessage) {
	b, _ := proto.Marshal(m)
	f.mu.Lock()
	defer f.mu.Unlock()
	_ = c.WriteMessage(websocket.BinaryMessage, b)
}

func (f *fakeServer) send(m *livekit.ServerMessage) {
	f.mu.Lock()
	c := f.conn
	f.mu.Unlock()
	f.write(c, m)
}

func (f *fakeServer) dropConn() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.conn.Close()
}

func (f *fakeServer) connections() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.conns
}

func (f *fakeServer) received() []*livekit.WorkerMessage {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]*livekit.WorkerMessage(nil), f.msgs...)
}

func find[T any](f *fakeServer, pick func(*livekit.WorkerMessage) (T, bool)) []T {
	var out []T
	for _, m := range f.received() {
		if v, ok := pick(m); ok {
			out = append(out, v)
		}
	}
	return out
}

func (f *fakeServer) availabilities() []*livekit.AvailabilityResponse {
	return find(f, func(m *livekit.WorkerMessage) (*livekit.AvailabilityResponse, bool) {
		return m.GetAvailability(), m.GetAvailability() != nil
	})
}

func (f *fakeServer) jobUpdates() []*livekit.UpdateJobStatus {
	return find(f, func(m *livekit.WorkerMessage) (*livekit.UpdateJobStatus, bool) {
		return m.GetUpdateJob(), m.GetUpdateJob() != nil
	})
}

func (f *fakeServer) lastStatus() *livekit.UpdateWorkerStatus {
	all := find(f, func(m *livekit.WorkerMessage) (*livekit.UpdateWorkerStatus, bool) {
		return m.GetUpdateWorker(), m.GetUpdateWorker() != nil
	})
	if len(all) == 0 {
		return nil
	}
	return all[len(all)-1]
}

// offer sends an availability request and, once accepted, the assignment.
func (f *fakeServer) offer(t *testing.T, id string) *livekit.AvailabilityResponse {
	t.Helper()
	job := &livekit.Job{Id: id, Room: &livekit.Room{Name: "room-" + id}, AgentName: "tester"}
	before := len(f.availabilities())
	f.send(&livekit.ServerMessage{Message: &livekit.ServerMessage_Availability{Availability: &livekit.AvailabilityRequest{Job: job}}})
	waitFor(t, "availability response for "+id, func() bool { return len(f.availabilities()) > before })
	resp := f.availabilities()[before]
	if resp.Available {
		f.send(&livekit.ServerMessage{Message: &livekit.ServerMessage_Assignment{Assignment: &livekit.JobAssignment{Job: job, Token: "room-token-" + id}}})
	}
	return resp
}

type workerRun struct {
	w      *agents.Worker
	cancel context.CancelFunc
	done   chan error
}

func startWorker(t *testing.T, f *fakeServer, mod func(*agents.WorkerOptions)) *workerRun {
	t.Helper()
	opts := agents.WorkerOptions{
		URL:            strings.Replace(f.srv.URL, "http://", "ws://", 1),
		APIKey:         "test-key",
		APISecret:      "test-secret",
		AgentName:      "tester",
		StatusInterval: 20 * time.Millisecond,
		Logger:         slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	mod(&opts)
	w, err := agents.NewWorker(opts)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	r := &workerRun{w: w, cancel: cancel, done: make(chan error, 1)}
	go func() { r.done <- w.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-r.done:
		case <-time.After(5 * time.Second):
			t.Error("worker did not stop")
		}
	})
	waitFor(t, "worker registration", func() bool { return w.Stats().Connected })
	return r
}

func (r *workerRun) stop(t *testing.T) error {
	t.Helper()
	r.cancel()
	select {
	case err := <-r.done:
		r.done <- err
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not stop")
		return nil
	}
}

func TestWorkerRegistersAndRunsJobs(t *testing.T) {
	f := newFakeServer(t)
	release := make(chan struct{})
	var got atomic.Pointer[agents.Job]
	r := startWorker(t, f, func(o *agents.WorkerOptions) {
		o.Entrypoint = func(ctx context.Context, job *agents.Job) error {
			got.Store(job)
			<-release
			return nil
		}
	})

	reg := f.received()[0].GetRegister()
	//lint:ignore SA1019 the worker still sends the deprecated flag, as upstream does
	agentFlag := reg.AllowedPermissions.Agent
	if reg.AgentName != "tester" || reg.Type != livekit.JobType_JT_ROOM || reg.Version != agents.Version ||
		!agentFlag || !reg.AllowedPermissions.CanPublish || !reg.AllowedPermissions.CanSubscribe {
		t.Fatalf("register request: %+v", reg)
	}
	if st := r.w.Stats(); st.WorkerID != "AW_test" {
		t.Fatalf("stats: %+v", st)
	}

	resp := f.offer(t, "J1")
	if !resp.Available || resp.ParticipantIdentity != "agent-J1" || resp.ParticipantAttributes["lk.agent.name"] != "tester" {
		t.Fatalf("availability response: %+v", resp)
	}
	waitFor(t, "job running", func() bool { return got.Load() != nil })
	if jc := got.Load(); jc.ID() != "J1" || jc.RoomName() != "room-J1" || jc.Identity() != "agent-J1" {
		t.Fatalf("job context: id=%s room=%s", jc.ID(), jc.RoomName())
	}
	waitFor(t, "running status", func() bool {
		u := f.jobUpdates()
		return len(u) == 1 && u[0].Status == livekit.JobStatus_JS_RUNNING
	})
	waitFor(t, "load report", func() bool { s := f.lastStatus(); return s != nil && s.JobCount == 1 && s.Load > 0 })

	close(release)
	waitFor(t, "success status", func() bool {
		u := f.jobUpdates()
		return len(u) == 2 && u[1].Status == livekit.JobStatus_JS_SUCCESS
	})
	waitFor(t, "slot released", func() bool { return r.w.Stats().ActiveJobs == 0 && f.lastStatus().JobCount == 0 })
	if err := r.stop(t); err != nil {
		t.Fatalf("Run: %v", err)
	}
}

func TestWorkerBoundsConcurrencyAndReportsFull(t *testing.T) {
	f := newFakeServer(t)
	release := make(chan struct{})
	var started atomic.Int32
	r := startWorker(t, f, func(o *agents.WorkerOptions) {
		o.MaxConcurrentJobs = 2
		o.Entrypoint = func(ctx context.Context, _ *agents.Job) error {
			started.Add(1)
			<-release
			return nil
		}
	})
	if !f.offer(t, "A").Available || !f.offer(t, "B").Available {
		t.Fatal("first two jobs must be accepted")
	}
	waitFor(t, "two jobs", func() bool { return started.Load() == 2 })
	waitFor(t, "full status", func() bool { return f.lastStatus().GetStatus() == livekit.WorkerStatus_WS_FULL })
	if f.offer(t, "C").Available {
		t.Fatal("third job must be declined at capacity")
	}
	close(release)
	waitFor(t, "available again", func() bool {
		return r.w.Stats().ActiveJobs == 0 && f.lastStatus().GetStatus() == livekit.WorkerStatus_WS_AVAILABLE
	})
	if !f.offer(t, "D").Available {
		t.Fatal("slot was not released")
	}
	waitFor(t, "fourth job done", func() bool { return started.Load() == 3 && r.w.Stats().ActiveJobs == 0 })
}

func TestWorkerJobFailureTerminationAndPanic(t *testing.T) {
	f := newFakeServer(t)
	cancelled := make(chan struct{})
	var hookRan atomic.Bool
	startWorker(t, f, func(o *agents.WorkerOptions) {
		o.Entrypoint = func(ctx context.Context, job *agents.Job) error {
			switch job.ID() {
			case "FAIL":
				return errors.New("provider exploded")
			case "PANIC":
				panic("boom")
			default:
				job.AddCleanup(func() { hookRan.Store(true) })
				<-ctx.Done()
				close(cancelled)
				return ctx.Err()
			}
		}
	})
	statusOf := func(id string) (livekit.JobStatus, string) {
		var st livekit.JobStatus = -1
		var msg string
		for _, u := range f.jobUpdates() {
			if u.JobId == id {
				st, msg = u.Status, u.Error
			}
		}
		return st, msg
	}

	f.offer(t, "FAIL")
	waitFor(t, "failed status", func() bool { st, _ := statusOf("FAIL"); return st == livekit.JobStatus_JS_FAILED })
	if _, msg := statusOf("FAIL"); msg != "provider exploded" {
		t.Fatalf("error message: %q", msg)
	}
	f.offer(t, "PANIC")
	waitFor(t, "panic reported as failure", func() bool { st, _ := statusOf("PANIC"); return st == livekit.JobStatus_JS_FAILED })

	f.offer(t, "TERM")
	waitFor(t, "running", func() bool { st, _ := statusOf("TERM"); return st == livekit.JobStatus_JS_RUNNING })
	f.send(&livekit.ServerMessage{Message: &livekit.ServerMessage_Termination{Termination: &livekit.JobTermination{JobId: "TERM"}}})
	<-cancelled
	waitFor(t, "terminated job reported", func() bool { st, _ := statusOf("TERM"); return st == livekit.JobStatus_JS_SUCCESS })
	if !hookRan.Load() {
		t.Fatal("shutdown hook did not run")
	}
}

func TestWorkerReconnectsAndReportsRunningJobs(t *testing.T) {
	f := newFakeServer(t)
	release := make(chan struct{})
	r := startWorker(t, f, func(o *agents.WorkerOptions) {
		o.Entrypoint = func(ctx context.Context, _ *agents.Job) error { <-release; return nil }
	})
	f.offer(t, "KEEP")
	waitFor(t, "job", func() bool { return r.w.Stats().ActiveJobs == 1 })

	f.dropConn()
	waitFor(t, "reconnection", func() bool { return f.connections() == 2 && r.w.Stats().Connected })
	// The job survived, and the server is told about it.
	waitFor(t, "migrate request", func() bool {
		m := find(f, func(m *livekit.WorkerMessage) (*livekit.MigrateJobRequest, bool) {
			return m.GetMigrateJob(), m.GetMigrateJob() != nil
		})
		return len(m) == 1 && len(m[0].JobIds) == 1 && m[0].JobIds[0] == "KEEP"
	})
	if r.w.Stats().ActiveJobs != 1 {
		t.Fatal("job was lost during reconnection")
	}
	close(release)
	// The final status is delivered over the new connection.
	waitFor(t, "success after reconnect", func() bool {
		u := f.jobUpdates()
		return len(u) > 0 && u[len(u)-1].Status == livekit.JobStatus_JS_SUCCESS
	})
}

func TestWorkerGivesUpAfterMaxRetries(t *testing.T) {
	f := newFakeServer(t)
	f.rejectUpgrades.Store(100)
	w, err := agents.NewWorker(agents.WorkerOptions{
		URL: f.srv.URL, APIKey: "test-key", APISecret: "test-secret", MaxRetries: 1,
		WorkerToken: "wt-must-not-leak",
		Entrypoint:  func(context.Context, *agents.Job) error { return nil },
		Logger:      slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	err = w.Run(context.Background())
	var re *agents.RegistrationError
	if !errors.As(err, &re) || re.Attempts != 1 {
		t.Fatalf("want RegistrationError after 1 retry, got %v", err)
	}
	if strings.Contains(err.Error(), "test-secret") || strings.Contains(err.Error(), "wt-must-not-leak") {
		t.Fatalf("secret leaked in error: %v", err)
	}
}

func TestWorkerDrain(t *testing.T) {
	f := newFakeServer(t)
	release := make(chan struct{})
	var cancelledEarly atomic.Bool
	r := startWorker(t, f, func(o *agents.WorkerOptions) {
		o.DrainTimeout = 5 * time.Second
		o.Entrypoint = func(ctx context.Context, _ *agents.Job) error {
			select {
			case <-release:
			case <-ctx.Done():
				cancelledEarly.Store(true)
			}
			return nil
		}
	})
	f.offer(t, "DRAIN")
	waitFor(t, "job", func() bool { return r.w.Stats().ActiveJobs == 1 })

	r.cancel() // SIGTERM equivalent
	waitFor(t, "draining status", func() bool {
		return r.w.Stats().Draining && f.lastStatus().GetStatus() == livekit.WorkerStatus_WS_FULL
	})
	if f.offer(t, "LATE").Available {
		t.Fatal("a draining worker must decline jobs")
	}
	select {
	case <-r.done:
		t.Fatal("Run returned while a job was still running")
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	select {
	case err := <-r.done:
		r.done <- err
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not return after the job finished")
	}
	if cancelledEarly.Load() {
		t.Fatal("job was cancelled instead of drained")
	}
	// Run returns once the status is written; the server may still be
	// reading it.
	waitFor(t, "final job status delivered", func() bool {
		u := f.jobUpdates()
		return len(u) > 0 && u[len(u)-1].Status == livekit.JobStatus_JS_SUCCESS
	})
}

func TestWorkerDrainTimeoutCancelsJobs(t *testing.T) {
	f := newFakeServer(t)
	r := startWorker(t, f, func(o *agents.WorkerOptions) {
		o.DrainTimeout = 30 * time.Millisecond
		o.Entrypoint = func(ctx context.Context, _ *agents.Job) error { <-ctx.Done(); return ctx.Err() }
	})
	f.offer(t, "STUCK")
	waitFor(t, "job", func() bool { return r.w.Stats().ActiveJobs == 1 })
	if err := r.stop(t); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if r.w.Stats().ActiveJobs != 0 {
		t.Fatal("job not released")
	}
}

func TestWorkerOptionValidation(t *testing.T) {
	t.Setenv("LIVEKIT_URL", "")
	t.Setenv("LIVEKIT_API_KEY", "")
	t.Setenv("LIVEKIT_API_SECRET", "")
	ep := func(context.Context, *agents.Job) error { return nil }
	for name, o := range map[string]agents.WorkerOptions{
		"no url":        {APIKey: "k", APISecret: "s", Entrypoint: ep},
		"no key":        {URL: "ws://x", Entrypoint: ep},
		"no entrypoint": {URL: "ws://x", APIKey: "k", APISecret: "s"},
		"bad scheme":    {URL: "ftp://x", APIKey: "k", APISecret: "s", Entrypoint: ep},
	} {
		if _, err := agents.NewWorker(o); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	t.Setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
	t.Setenv("LIVEKIT_API_KEY", "k")
	t.Setenv("LIVEKIT_API_SECRET", "s")
	if _, err := agents.NewWorker(agents.WorkerOptions{Entrypoint: ep}); err != nil {
		t.Errorf("env configuration rejected: %v", err)
	}
}

func freeAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().String()
}

func httpGet(t *testing.T, url string) (int, string) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		return 0, err.Error()
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

// A worker hosted by LiveKit Cloud receives its identity from the platform
// through environment variables, as the upstream workers do.
func TestWorkerHostedMode(t *testing.T) {
	f := newFakeServer(t)
	addr := freeAddr(t)
	t.Setenv("LIVEKIT_WORKER_TOKEN", "wt-hosted-secret")
	t.Setenv("LIVEKIT_AGENT_DEPLOYMENT", "staging")
	t.Setenv("LIVEKIT_AGENT_NAME_OVERRIDE", "platform-name")

	var load atomic.Int64 // percent
	release := make(chan struct{})
	r := startWorker(t, f, func(o *agents.WorkerOptions) {
		o.AgentName = "tester" // overridden by the platform
		o.HealthAddr = addr
		o.HealthGracePeriod = 80 * time.Millisecond
		o.LoadFunc = func() float64 { return float64(load.Load()) / 100 }
		o.Entrypoint = func(ctx context.Context, _ *agents.Job) error { <-release; return nil }
	})
	defer close(release)

	if !r.w.Hosted() {
		t.Fatal("worker with a token must be in hosted mode")
	}
	f.mu.Lock()
	token := f.workerToken
	f.mu.Unlock()
	if token != "wt-hosted-secret" {
		t.Fatalf("worker_token query parameter = %q", token)
	}
	reg := f.received()[0].GetRegister()
	if reg.Deployment != "staging" || reg.AgentName != "platform-name" {
		t.Fatalf("register: deployment=%q agent_name=%q", reg.Deployment, reg.AgentName)
	}

	// Health and worker info endpoints.
	if code, body := httpGet(t, "http://"+addr+"/"); code != 200 || body != "OK" {
		t.Fatalf("health: %d %q", code, body)
	}
	code, body := httpGet(t, "http://"+addr+"/worker")
	var info struct {
		WorkerType      string  `json:"worker_type"`
		AgentName       string  `json:"agent_name"`
		ActiveJobs      int     `json:"active_jobs"`
		SDKVersion      string  `json:"sdk_version"`
		WorkerLoad      float64 `json:"worker_load"`
		ProtocolVersion int     `json:"protocol_version"`
	}
	if err := json.Unmarshal([]byte(body), &info); err != nil || code != 200 {
		t.Fatalf("worker info: %d %q %v", code, body, err)
	}
	if info.WorkerType != "JT_ROOM" || info.AgentName != "platform-name" || info.SDKVersion != agents.Version || info.ProtocolVersion != 1 {
		t.Fatalf("worker info: %+v", info)
	}
	if code, _ := httpGet(t, "http://"+addr+"/nope"); code != 404 {
		t.Fatalf("unknown path: %d", code)
	}

	// Load-based capacity: hosted mode defaults the threshold to 0.7.
	load.Store(50)
	waitFor(t, "load reported", func() bool { s := f.lastStatus(); return s != nil && s.Load > 0.49 && s.Load < 0.51 })
	if f.lastStatus().GetStatus() != livekit.WorkerStatus_WS_AVAILABLE || !f.offer(t, "OK").Available {
		t.Fatal("worker under the load threshold must accept jobs")
	}
	load.Store(85)
	waitFor(t, "full on high load", func() bool { return f.lastStatus().GetStatus() == livekit.WorkerStatus_WS_FULL })
	if f.offer(t, "BUSY").Available {
		t.Fatal("worker over the load threshold must decline jobs")
	}
	load.Store(10)

	// A short disconnection stays healthy; a long one does not.
	f.rejectUpgrades.Store(1000)
	f.dropConn()
	waitFor(t, "disconnected", func() bool { return !r.w.Stats().Connected })
	if code, _ := httpGet(t, "http://"+addr+"/"); code != 200 {
		t.Fatalf("health during the grace period: %d", code)
	}
	waitFor(t, "unhealthy after the grace period", func() bool {
		code, _ := httpGet(t, "http://"+addr+"/")
		return code == 503
	})
	f.rejectUpgrades.Store(0)
	waitFor(t, "healthy again after reconnecting", func() bool {
		code, _ := httpGet(t, "http://"+addr+"/")
		return code == 200
	})
}

func TestWorkerHealthBeforeRegistrationAndDefaults(t *testing.T) {
	f := newFakeServer(t)
	f.rejectUpgrades.Store(1000)
	addr := freeAddr(t)
	t.Setenv("LIVEKIT_WORKER_TOKEN", "")
	t.Setenv("LIVEKIT_AGENT_NAME", "from-env")
	w, err := agents.NewWorker(agents.WorkerOptions{
		URL: f.srv.URL, APIKey: "test-key", APISecret: "test-secret", HealthAddr: addr,
		Entrypoint: func(context.Context, *agents.Job) error { return nil },
		Logger:     slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	if w.Hosted() {
		t.Fatal("no token: not hosted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()
	waitFor(t, "health server up", func() bool { code, _ := httpGet(t, "http://"+addr+"/"); return code != 0 })
	if code, body := httpGet(t, "http://"+addr+"/"); code != 503 || !strings.Contains(body, "not connected") {
		t.Fatalf("health before registration: %d %q", code, body)
	}
	_, body := httpGet(t, "http://"+addr+"/worker")
	if !strings.Contains(body, `"agent_name":"from-env"`) {
		t.Fatalf("LIVEKIT_AGENT_NAME default not applied: %s", body)
	}
	cancel()
	if err := <-done; err != nil {
		t.Fatalf("Run: %v", err)
	}
	if code, _ := httpGet(t, "http://"+addr+"/"); code != 0 {
		t.Fatal("health server still listening after Run returned")
	}

	// A busy health port is a startup error, not a silent miss.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	w2, _ := agents.NewWorker(agents.WorkerOptions{
		URL: f.srv.URL, APIKey: "k", APISecret: "s", HealthAddr: ln.Addr().String(),
		Entrypoint: func(context.Context, *agents.Job) error { return nil },
		Logger:     slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err := w2.Run(context.Background()); err == nil || !strings.Contains(err.Error(), "health server") {
		t.Fatalf("want a health server error, got %v", err)
	}
}

// A server that accepts the registration and drops the connection at once
// must not be redialled in a tight loop.
func TestWorkerBacksOffWhenRegisteredConnectionsDrop(t *testing.T) {
	f := newFakeServer(t)
	startWorker(t, f, func(o *agents.WorkerOptions) {
		o.Entrypoint = func(context.Context, *agents.Job) error { return nil }
	})
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		seen := 0
		for {
			select {
			case <-stop:
				return
			case <-time.After(5 * time.Millisecond):
			}
			if n := f.connections(); n > seen {
				seen = n
				f.dropConn()
			}
		}
	}()
	time.Sleep(1500 * time.Millisecond)
	// Schedule: immediate retry, then 2s, 4s ... so at most 2 reconnections
	// fit in 1.5s. A tight loop would make hundreds.
	if n := f.connections(); n > 3 {
		t.Fatalf("%d connections in 1.5s: reconnection does not back off", n)
	}
}

func TestJobRequestCanOnlyBeAnsweredOnce(t *testing.T) {
	f := newFakeServer(t)
	errs := make(chan error, 2)
	startWorker(t, f, func(o *agents.WorkerOptions) {
		o.Entrypoint = func(context.Context, *agents.Job) error { return nil }
		o.OnJobRequest = func(ctx context.Context, req *agents.JobRequest) error {
			errs <- req.Reject()
			errs <- req.Accept(ctx, agents.AcceptOptions{})
			return nil
		}
	})
	if f.offer(t, "ONCE").Available {
		t.Fatal("request should have been rejected")
	}
	if err := <-errs; err != nil {
		t.Fatalf("Reject: %v", err)
	}
	if err := <-errs; !errors.Is(err, agents.ErrRequestAnswered) {
		t.Fatalf("second answer: %v", err)
	}
}
