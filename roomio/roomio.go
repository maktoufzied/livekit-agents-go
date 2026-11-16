// Package roomio connects a Session to a LiveKit room: it subscribes to a
// participant's microphone, publishes the agent's voice, and mirrors agent
// state and transcripts to the room the way upstream LiveKit agents do.
//
// This is the only package of the SDK that needs cgo. It reuses the PCM
// track helpers of the official Go SDK (server-sdk-go/pkg/media), which
// decode and encode Opus with libopus and resample with libsoxr.
//
// Audio formats:
//
//   - Input: Opus from the room, decoded and resampled to mono PCM16 at
//     Options.InputSampleRate (16 kHz by default).
//   - Output: mono or stereo PCM16 at the TTS sample rate, resampled to
//     48 kHz and encoded to Opus.
package roomio

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	media "github.com/livekit/media-sdk"
	"github.com/livekit/protocol/livekit"
	protoLogger "github.com/livekit/protocol/logger"
	lksdk "github.com/livekit/server-sdk-go/v2"
	lkmedia "github.com/livekit/server-sdk-go/v2/pkg/media"
	"github.com/pion/webrtc/v4"

	agents "github.com/maktoufzied/livekit-agents-go"
	"github.com/maktoufzied/livekit-agents-go/audio"
)

// Participant attributes and stream topics used by upstream LiveKit agents
// and understood by LiveKit client SDKs and the Agents Playground.
const (
	AttributeAgentState     = "lk.agent.state"
	TopicTranscription      = "lk.transcription"
	attrTranscribedTrackID  = "lk.transcribed_track_id"
	attrTranscriptionFinal  = "lk.transcription_final"
	attrTranscriptSegmentID = "lk.segment_id"
)

// Options configures room I/O. Zero values select the defaults.
type Options struct {
	// ParticipantIdentity selects the participant to listen to. By default
	// the first non-agent participant is used.
	ParticipantIdentity string
	// InputSampleRate is the rate audio is delivered to VAD and STT.
	// Default 16000.
	InputSampleRate int
	// InputBufferFrames bounds the decoded audio waiting for the session.
	// When full, the oldest frame is dropped. Default 100 (about 2s).
	InputBufferFrames int
	// MaxQueuedAudio is how much agent audio may be queued ahead of
	// playback. It bounds how much is dropped on interruption and how
	// stale a cancelled reply can be. Default 100ms.
	MaxQueuedAudio time.Duration
	// TrackName is the name of the published audio track. Default "agent-voice".
	TrackName string
	// DisableTranscription stops publishing transcripts to the room.
	DisableTranscription bool
	// KeepOpenOnDisconnect keeps Run going after the participant leaves.
	KeepOpenOnDisconnect bool
	// OnParticipant, when set, is called by Run once the participant the
	// agent listens to has joined. Use it to greet the user.
	OnParticipant func(ctx context.Context, session *agents.Session, identity string)
	// InputTap, when set, is given every frame of the participant's audio
	// as the session reads it, inserted silence included. It is meant for
	// recording and debugging, runs on the audio path and must not block.
	InputTap func(audio.Frame)
	// ConnectOptions are passed to the Go SDK when joining the room.
	ConnectOptions []lksdk.ConnectOption
	// Logger defaults to slog.Default().
	Logger *slog.Logger
}

// Bridge is the audio bridge between one room and one session. Its methods
// are safe for concurrent use; the room callbacks it installs may run at any
// time, including during Close.
type Bridge struct {
	opts Options
	log  *slog.Logger

	in  *input
	out *output

	mu        sync.Mutex
	room      *lksdk.Room
	linked    string // identity of the participant we listen to
	remote    *lkmedia.PCMRemoteTrack
	remoteSID string
	localSID  string
	closed    bool

	linkedCh chan struct{} // closed once a participant is linked
	doneOnce sync.Once
	done     chan struct{} // closed when the participant or the room is gone
}

// New returns a Bridge. Pass Callback to Job.Connect, then call Start.
func New(opts Options) *Bridge {
	if opts.InputSampleRate <= 0 {
		opts.InputSampleRate = 16000
	}
	if opts.InputBufferFrames <= 0 {
		opts.InputBufferFrames = 100
	}
	if opts.MaxQueuedAudio <= 0 {
		opts.MaxQueuedAudio = 100 * time.Millisecond
	}
	if opts.TrackName == "" {
		opts.TrackName = "agent-voice"
	}
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	return &Bridge{
		opts:     opts,
		log:      opts.Logger,
		in:       newInput(audio.Format{SampleRate: opts.InputSampleRate, Channels: 1}, opts.InputBufferFrames),
		linkedCh: make(chan struct{}),
		done:     make(chan struct{}),
	}
}

// Callback returns the room callbacks the bridge needs. They must be passed to
// the room when joining.
func (r *Bridge) Callback() *lksdk.RoomCallback {
	return &lksdk.RoomCallback{
		OnParticipantConnected: func(rp *lksdk.RemoteParticipant) { r.tryLink(rp) },
		OnParticipantDisconnected: func(rp *lksdk.RemoteParticipant) {
			r.mu.Lock()
			gone := r.linked != "" && rp.Identity() == r.linked
			r.mu.Unlock()
			if gone {
				r.log.Info("participant left", "identity", rp.Identity())
				r.finish()
			}
		},
		OnDisconnected: func() { r.finish() },
		ParticipantCallback: lksdk.ParticipantCallback{
			OnTrackSubscribed:   r.onTrackSubscribed,
			OnTrackUnsubscribed: r.onTrackUnsubscribed,
		},
	}
}

func (r *Bridge) finish() { r.doneOnce.Do(func() { close(r.done) }) }

// Done is closed when the linked participant left or the room disconnected.
func (r *Bridge) Done() <-chan struct{} { return r.done }

// tryLink links to rp if nobody is linked yet and rp is eligible. It
// reports whether rp is the linked participant.
func (r *Bridge) tryLink(rp *lksdk.RemoteParticipant) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.linked != "" {
		return r.linked == rp.Identity()
	}
	if r.opts.ParticipantIdentity != "" {
		if rp.Identity() != r.opts.ParticipantIdentity {
			return false
		}
	} else if rp.Kind() == lksdk.ParticipantAgent {
		return false
	}
	r.linked = rp.Identity()
	close(r.linkedCh)
	r.log.Info("listening to participant", "identity", r.linked)
	return true
}

func (r *Bridge) onTrackSubscribed(track *webrtc.TrackRemote, pub *lksdk.RemoteTrackPublication, rp *lksdk.RemoteParticipant) {
	if track.Kind() != webrtc.RTPCodecTypeAudio {
		return
	}
	// Listen to the microphone only. Screen-share audio from the same
	// participant must not replace it.
	if src := pub.Source(); src != livekit.TrackSource_MICROPHONE && src != livekit.TrackSource_UNKNOWN {
		return
	}
	if !r.tryLink(rp) {
		return
	}
	remote, err := lkmedia.NewPCMRemoteTrack(track, r.in,
		lkmedia.WithTargetSampleRate(r.opts.InputSampleRate),
		lkmedia.WithTargetChannels(1))
	if err != nil {
		r.log.Error("cannot decode participant audio", "codec", track.Codec().MimeType, "error", err)
		return
	}
	r.mu.Lock()
	old := r.remote
	if r.closed {
		r.mu.Unlock()
		remote.Close()
		return
	}
	r.remote, r.remoteSID = remote, pub.SID()
	r.mu.Unlock()
	if old != nil {
		old.Close()
	}
	r.log.Debug("subscribed to participant audio", "track", pub.SID())
}

func (r *Bridge) onTrackUnsubscribed(_ *webrtc.TrackRemote, pub *lksdk.RemoteTrackPublication, _ *lksdk.RemoteParticipant) {
	r.mu.Lock()
	var old *lkmedia.PCMRemoteTrack
	if r.remoteSID == pub.SID() {
		old, r.remote, r.remoteSID = r.remote, nil, ""
	}
	r.mu.Unlock()
	if old != nil {
		old.Close()
	}
}

// Start publishes the agent's audio track to the joined room. format is the
// PCM format the session will write (Session.OutputFormat).
func (r *Bridge) Start(room *lksdk.Room, format audio.Format) error {
	if !format.Valid() || format.Channels > 2 {
		return fmt.Errorf("roomio: unsupported output format %dHz/%dch", format.SampleRate, format.Channels)
	}
	track, err := lkmedia.NewPCMLocalTrack(format.SampleRate, format.Channels, protoLogger.GetLogger())
	if err != nil {
		return fmt.Errorf("roomio: create audio track: %w", err)
	}
	pub, err := room.LocalParticipant.PublishTrack(track, &lksdk.TrackPublicationOptions{
		Name:   r.opts.TrackName,
		Source: livekit.TrackSource_MICROPHONE,
		Stereo: format.Channels == 2,
	})
	if err != nil {
		_ = track.Close()
		return fmt.Errorf("roomio: publish audio track: %w", err)
	}
	r.mu.Lock()
	r.room = room
	r.localSID = pub.SID()
	r.out = &output{format: format, track: track, maxQueued: r.opts.MaxQueuedAudio}
	r.mu.Unlock()

	// Participants who were in the room before we joined.
	for _, rp := range room.GetRemoteParticipants() {
		r.tryLink(rp)
	}
	return nil
}

// Input is the session's audio input.
func (r *Bridge) Input() agents.AudioInput {
	if r.opts.InputTap != nil {
		return tappedInput{r.in, r.opts.InputTap}
	}
	return r.in
}

// tappedInput shows every frame read to a tap.
type tappedInput struct {
	*input
	tap func(audio.Frame)
}

func (t tappedInput) ReadFrame(ctx context.Context) (audio.Frame, error) {
	f, err := t.input.ReadFrame(ctx)
	if err == nil {
		t.tap(f)
	}
	return f, err
}

// Output is the session's audio output. It is nil before Start.
func (r *Bridge) Output() agents.AudioOutput {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.out == nil {
		return nil
	}
	return r.out
}

// DroppedInputFrames counts decoded frames dropped because the session did
// not read them in time.
func (r *Bridge) DroppedInputFrames() int64 { return r.in.dropped.Load() }

// WaitParticipant blocks until a participant is linked and returns its
// identity.
func (r *Bridge) WaitParticipant(ctx context.Context) (string, error) {
	select {
	case <-r.linkedCh:
		r.mu.Lock()
		defer r.mu.Unlock()
		return r.linked, nil
	case <-r.done:
		return "", errors.New("roomio: room closed before a participant joined")
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

// Bind mirrors session events to the room: the agent state as the
// lk.agent.state participant attribute and transcripts as lk.transcription
// text streams. It returns a function that stops mirroring.
func (r *Bridge) Bind(session *agents.Session) (unbind func()) {
	var userSegment, agentSegment int
	return session.On(func(ev agents.Event) {
		r.mu.Lock()
		room, localSID, remoteSID, closed := r.room, r.localSID, r.remoteSID, r.closed
		r.mu.Unlock()
		if room == nil || closed {
			return
		}
		lp := room.LocalParticipant
		switch e := ev.(type) {
		case agents.AgentStateChanged:
			lp.SetAttributes(map[string]string{AttributeAgentState: string(e.New)})
		case agents.UserTranscript:
			if r.opts.DisableTranscription || remoteSID == "" {
				return
			}
			r.sendTranscript(lp, e.Text, remoteSID, "user-"+strconv.Itoa(userSegment), e.Final)
			if e.Final {
				userSegment++
			}
		case agents.AgentTranscript:
			if r.opts.DisableTranscription {
				return
			}
			r.sendTranscript(lp, e.Text, localSID, "agent-"+strconv.Itoa(agentSegment), true)
			agentSegment++
		}
	})
}

func (r *Bridge) sendTranscript(lp *lksdk.LocalParticipant, text, trackSID, segment string, final bool) {
	lp.SendText(text, lksdk.StreamTextOptions{
		Topic: TopicTranscription,
		Attributes: map[string]string{
			attrTranscribedTrackID:  trackSID,
			attrTranscriptSegmentID: segment,
			attrTranscriptionFinal:  strconv.FormatBool(final),
		},
	})
}

// Close stops decoding and closes the published track. It does not
// disconnect the room; the job does.
func (r *Bridge) Close() {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return
	}
	r.closed = true
	remote, out := r.remote, r.out
	r.remote = nil
	r.mu.Unlock()
	if remote != nil {
		remote.Close()
	}
	if out != nil {
		out.track.ClearQueue()
		_ = out.track.Close()
	}
	r.in.shutdown()
	r.finish()
}

// Run is the usual way to run a session in a job: it joins the job's room,
// wires audio both ways, starts the session and blocks until the job
// context ends, the participant leaves, or the session closes. The session
// is closed before Run returns. It returns the session's error, if any.
func Run(ctx context.Context, job *agents.Job, session *agents.Session, opts Options) error {
	if opts.Logger == nil {
		opts.Logger = job.Logger()
	}
	r := New(opts)
	room, err := job.Connect(ctx, r.Callback(), opts.ConnectOptions...)
	if err != nil {
		return fmt.Errorf("roomio: join room: %w", err)
	}
	defer r.Close()
	if err := r.Start(room, session.OutputFormat()); err != nil {
		return err
	}
	unbind := r.Bind(session)
	defer unbind()
	if err := session.Start(ctx, agents.IO{Input: r.Input(), Output: r.Output()}); err != nil {
		return err
	}
	defer session.Close()

	if opts.OnParticipant != nil {
		go func() {
			if identity, err := r.WaitParticipant(ctx); err == nil {
				opts.OnParticipant(ctx, session, identity)
			}
		}()
	}

	left := r.Done()
	if opts.KeepOpenOnDisconnect {
		left = nil
	}
	select {
	case <-ctx.Done():
	case <-session.Done():
	case <-left:
	}
	session.Close()
	return session.Err()
}

// ---------------------------------------------------------------------------
// Input

type input struct {
	format  audio.Format
	ch      chan audio.Frame
	once    sync.Once
	closed  chan struct{}
	dropped atomic.Int64

	// Owned by the reader. started is when the first frame was read and
	// emitted the audio returned since: together they tell whether the
	// audio keeps up with real time.
	started time.Time
	emitted time.Duration
}

const (
	// inputGap is how far the received audio may fall behind real time
	// before silence is inserted. It absorbs network jitter.
	inputGap = 100 * time.Millisecond
	// inputFill is the length of an inserted silence frame.
	inputFill = 20 * time.Millisecond
)

func newInput(format audio.Format, size int) *input {
	return &input{format: format, ch: make(chan audio.Frame, size), closed: make(chan struct{})}
}

func (in *input) Format() audio.Format { return in.format }

// WriteSample receives decoded audio from the RTP goroutine. It never
// blocks: when the session is behind, the oldest frame is dropped.
func (in *input) WriteSample(sample media.PCM16Sample) error {
	if len(sample) == 0 {
		return nil
	}
	f := audio.Frame{Data: append([]int16(nil), sample...), SampleRate: in.format.SampleRate, Channels: in.format.Channels}
	select {
	case in.ch <- f:
		return nil
	default:
	}
	select {
	case <-in.ch:
		in.dropped.Add(1)
	default:
	}
	select {
	case in.ch <- f:
	default:
	}
	return nil
}

// Close is called by the decoder when a track ends. The input outlives
// individual tracks (the participant may republish), so it is a no-op.
func (in *input) Close() error { return nil }

func (in *input) shutdown() { in.once.Do(func() { close(in.closed) }) }

// ReadFrame returns the next frame of the participant's audio, or a frame of
// silence when none arrived in time.
//
// Browsers stop sending audio packets while the user is silent (Opus DTX),
// and the decoder does not fill the holes. Without the inserted silence the
// audio would run slower than real time: a voice activity detector waiting
// for half a second of silence would get it seconds later.
func (in *input) ReadFrame(ctx context.Context) (audio.Frame, error) {
	var late <-chan time.Time
	if !in.started.IsZero() {
		t := time.NewTimer(time.Until(in.started.Add(in.emitted + inputGap)))
		defer t.Stop()
		late = t.C
	}
	select {
	case f := <-in.ch:
		if in.started.IsZero() {
			in.started = time.Now()
		}
		in.emitted += f.Duration()
		return f, nil
	case <-late:
		f := audio.Silence(in.format, inputFill)
		in.emitted += f.Duration()
		return f, nil
	case <-in.closed:
		return audio.Frame{}, io.EOF
	case <-ctx.Done():
		return audio.Frame{}, ctx.Err()
	}
}

// ---------------------------------------------------------------------------
// Output

// output paces writes to the published track. The track itself plays its
// queue in real time, so the output keeps a playhead: the wall-clock time
// at which everything accepted so far will have been sent.
type output struct {
	format    audio.Format
	track     *lkmedia.PCMLocalTrack
	maxQueued time.Duration

	mu       sync.Mutex
	playhead time.Time
}

func (o *output) Format() audio.Format { return o.format }

func (o *output) WriteFrame(ctx context.Context, f audio.Frame) error {
	if f.Format() != o.format {
		return fmt.Errorf("roomio: frame format %dHz/%dch does not match track %dHz/%dch", f.SampleRate, f.Channels, o.format.SampleRate, o.format.Channels)
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		o.mu.Lock()
		now := time.Now()
		if o.playhead.Before(now) {
			o.playhead = now
		}
		ahead := o.playhead.Sub(now)
		if ahead <= o.maxQueued {
			err := o.track.WriteSample(media.PCM16Sample(f.Data))
			if err == nil {
				o.playhead = o.playhead.Add(f.Duration())
			}
			o.mu.Unlock()
			return err
		}
		o.mu.Unlock()
		t := time.NewTimer(ahead - o.maxQueued)
		select {
		case <-t.C:
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		}
	}
}

func (o *output) Flush(ctx context.Context) error {
	o.mu.Lock()
	wait := time.Until(o.playhead)
	o.mu.Unlock()
	if wait <= 0 {
		return nil
	}
	t := time.NewTimer(wait)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (o *output) Clear() time.Duration {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.track.ClearQueue()
	now := time.Now()
	if !o.playhead.After(now) {
		return 0
	}
	dropped := o.playhead.Sub(now)
	o.playhead = now
	return dropped
}
