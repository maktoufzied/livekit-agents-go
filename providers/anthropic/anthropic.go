// Package anthropic implements llm.LLM on top of the Anthropic Messages API
// (POST /v1/messages with "stream": true), using only the standard library.
//
// # Model selection
//
// The default model is Claude Opus 5.5 ("claude-opus-5-5"), the current
// general-purpose default of the Claude API. On that model thinking cannot be
// switched off, so latency is controlled with the effort level instead: this
// package sends output_config.effort "low" unless LLMOptions.Effort says
// otherwise, which is the recommended setting for conversational turns.
//
// A voice agent that needs a lower time to first token should pick a smaller
// model with LLMOptions.Model, in this order: "claude-sonnet-5-5", then
// "claude-haiku-5-5" (the fastest current model). Both accept the same request
// shape, so nothing else has to change.
//
// # History mapping
//
// llm.Message values are mapped to the Messages API as follows:
//
//   - RoleSystem messages, wherever they appear, are joined with a blank line
//     into the top-level "system" field.
//   - RoleUser becomes a user text block, RoleAssistant a text block followed
//     by one tool_use block per ToolCall, RoleTool a tool_result block in a
//     user message.
//   - Consecutive messages of the same API role are merged into one message,
//     so parallel tool results always travel in a single user message.
//   - The API requires the first message to be a user message. A history that
//     starts with an assistant message (for example a greeting) gets a short
//     synthetic user message in front of it, see LeadingUserPlaceholder.
//   - The current models reject a request whose last message is an assistant
//     message. Such a history gets a short synthetic user message appended,
//     see TrailingUserPlaceholder.
//
// Thinking blocks are neither exposed nor replayed: llm.Message cannot carry
// them, and the API accepts a history from which all of them were removed.
//
// # Errors
//
// NewLLM returns plain errors that describe the invalid option. Every other
// error is a *providers.Error, except the io.EOF that Recv returns at the
// normal end of a stream. A stream used after Close fails with an error
// matching providers.ErrStreamClosed, a cancelled or expired context with
// one matching the context error; neither is retryable. No error ever
// contains the API key.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/maktoufzied/livekit-agents-go/llm"
	"github.com/maktoufzied/livekit-agents-go/providers"
)

const (
	// DefaultLLMModel is the model used when LLMOptions.Model is empty.
	DefaultLLMModel = "claude-opus-5-5"
	// DefaultMaxTokens is the output cap used when LLMOptions.MaxTokens is
	// zero. Thinking counts toward the cap, so it leaves room for it on top
	// of a spoken reply; steer the reply length from the instructions.
	DefaultMaxTokens = 4096
	// DefaultBaseURL is the Claude API origin used when no BaseURL is set.
	DefaultBaseURL = "https://api.anthropic.com"
	// DefaultAPIVersion is the anthropic-version header sent by default.
	DefaultAPIVersion = "2026-10-21"
	// LegacyAPIVersion is the previous API version. It is the only one on
	// which sampling parameters such as temperature are still accepted.
	LegacyAPIVersion = "2023-06-01"

	// LeadingUserPlaceholder is the user message inserted in front of a
	// history that starts with an assistant message.
	LeadingUserPlaceholder = "(The conversation starts with the assistant speaking first.)"
	// TrailingUserPlaceholder is the user message appended to a history that
	// ends with an assistant message.
	TrailingUserPlaceholder = "(The user has not said anything else. Continue the conversation.)"

	providerName        = "anthropic"
	envAPIKey           = "ANTHROPIC_API_KEY"
	fallbackBeta        = "server-side-fallback-2026-07-01"
	requiredToolSuffix  = "You must call one of the provided tools in this response. Do not answer with text only."
	maxErrorBodyBytes   = 1 << 20
	emptyObjectSchema   = `{"type":"object","properties":{}}`
	emptyObjectArgument = `{}`
)

// Effort is the output_config.effort level of a request. It trades depth of
// reasoning, and therefore latency and tokens, against quality.
type Effort string

const (
	// EffortLow is the fastest level and the default of this package.
	EffortLow Effort = "low"
	// EffortMedium is the level between low and high.
	EffortMedium Effort = "medium"
	// EffortHigh favours quality over latency.
	EffortHigh Effort = "high"
	// EffortModelDefault omits output_config.effort, so the API applies the
	// default of the model.
	EffortModelDefault Effort = "default"
)

// LLMOptions configures an LLM.
type LLMOptions struct {
	// APIKey authenticates the requests. It defaults to the
	// ANTHROPIC_API_KEY environment variable.
	APIKey providers.Secret
	// AllowNoAPIKey permits an empty key, for gateways that inject the
	// credentials. No x-api-key header is sent then. It defaults to false.
	AllowNoAPIKey bool
	// Model is the model ID. It defaults to DefaultLLMModel; see the
	// package documentation for how to pick a faster model.
	Model string
	// MaxTokens caps the output of one generation, thinking included. The
	// API requires it. It defaults to DefaultMaxTokens.
	MaxTokens int
	// Effort is the effort level. It defaults to the zero value, which
	// sends EffortLow, except on Claude Haiku 4.5 and Claude Sonnet 4.5,
	// which reject the parameter. Any other value is sent as is;
	// EffortModelDefault omits the parameter.
	Effort Effort
	// Temperature is the sampling temperature. It defaults to nil, which
	// leaves the provider default. A value that is set is always sent,
	// never dropped; NewLLM fails instead when the API is known to answer
	// it with a 400, as it does in the default configuration:
	//
	//   - APIVersion must be LegacyAPIVersion, the only version on which
	//     sampling parameters are still accepted;
	//   - Model must not be one of the current models, which take no
	//     sampling parameters on any version (Claude Opus 5.5 and 5, Sonnet
	//     5.5, Haiku 5.5, Fable and Mythos 5.1 and 5.2). Claude Opus 4.6,
	//     Sonnet 4.6, Haiku 4.5 and older accept it. For a model ID this
	//     package does not know, the value is sent and the API decides.
	Temperature *float64
	// DisableRefusalFallback turns off the server-side refusal fallback. By
	// default, on the models that have safety classifiers able to decline a
	// request (Claude Opus 5.5, Opus 5, Sonnet 5.5, Fable 5.1 and 5.2), the
	// request carries "fallbacks": "default" and the matching anthropic-beta
	// header, so a declined request is answered by another model in the same
	// call. Disable it when BaseURL points at a gateway or platform that
	// does not support the parameter. It defaults to false.
	DisableRefusalFallback bool
	// BaseURL is the bare origin of the API (scheme and host, plus an
	// optional path prefix of a gateway), without the /v1/messages path. A
	// trailing slash is ignored. It defaults to DefaultBaseURL.
	BaseURL string
	// APIVersion is the anthropic-version header. It defaults to
	// DefaultAPIVersion.
	APIVersion string
	// HTTPClient sends the requests. It defaults to nil, which means a
	// client shared by the package with no overall Timeout, because
	// streams are long-lived: bound each stream with its context instead.
	HTTPClient *http.Client
}

// defaultHTTPClient is used when no HTTPClient is configured. It has no
// overall Timeout: responses are streams bounded by the caller's context.
var defaultHTTPClient = &http.Client{}

// LLM is a streaming chat model backed by the Anthropic Messages API. It is
// safe for concurrent use.
type LLM struct {
	// apiKey returns the key, empty with AllowNoAPIKey. A closure keeps it
	// out of fmt and log output of the LLM value itself.
	apiKey      func() string
	model       string
	maxTokens   int
	effort      Effort
	temperature *float64
	fallback    bool
	endpoint    string
	apiVersion  string
	client      *http.Client
}

var _ llm.LLM = (*LLM)(nil)

// NewLLM validates opts and returns an LLM. It fails when no API key is
// configured, unless opts.AllowNoAPIKey is set, or an option is invalid.
func NewLLM(opts LLMOptions) (*LLM, error) {
	key := opts.APIKey
	if key == "" {
		key = providers.SecretFromEnv(envAPIKey)
	}
	if key == "" && !opts.AllowNoAPIKey {
		return nil, errors.New("anthropic: API key is required (LLMOptions.APIKey or " + envAPIKey + ")")
	}
	if opts.MaxTokens < 0 {
		return nil, errors.New("anthropic: MaxTokens must not be negative")
	}

	l := &LLM{
		apiKey:     key.Reveal,
		model:      opts.Model,
		maxTokens:  opts.MaxTokens,
		apiVersion: opts.APIVersion,
		client:     opts.HTTPClient,
	}
	if l.model == "" {
		l.model = DefaultLLMModel
	}
	if l.maxTokens == 0 {
		l.maxTokens = DefaultMaxTokens
	}
	if l.apiVersion == "" {
		l.apiVersion = DefaultAPIVersion
	}
	if l.client == nil {
		l.client = defaultHTTPClient
	}
	base := opts.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	l.endpoint = strings.TrimRight(base, "/") + "/v1/messages"

	switch opts.Effort {
	case "":
		if !modelIn(l.model, modelsWithoutEffort) {
			l.effort = EffortLow
		}
	case EffortModelDefault:
	default:
		l.effort = opts.Effort
	}
	if opts.Temperature != nil {
		switch {
		case l.apiVersion != LegacyAPIVersion:
			return nil, fmt.Errorf("anthropic: Temperature is set but API version %q rejects sampling parameters: "+
				"set LLMOptions.APIVersion to LegacyAPIVersion (%q) and LLMOptions.Model to a model that accepts them (for example %q), or leave Temperature nil",
				l.apiVersion, LegacyAPIVersion, modelsWithSampling[1])
		case modelIn(l.model, modelsWithoutSampling):
			return nil, fmt.Errorf("anthropic: Temperature is set but model %q rejects sampling parameters: "+
				"set LLMOptions.Model to a model that accepts them (for example %q), or leave Temperature nil",
				l.model, modelsWithSampling[1])
		}
		t := *opts.Temperature
		l.temperature = &t
	}
	l.fallback = !opts.DisableRefusalFallback && modelIn(l.model, modelsWithFallback)
	return l, nil
}

// Model lists, taken from the Claude API model reference. A model matches an
// entry when its id equals the entry or extends it with a dated suffix.
var (
	// modelsWithSampling accept temperature on LegacyAPIVersion. Older
	// models do too; the list only feeds the advice of NewLLM's errors.
	modelsWithSampling = []string{
		"claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5",
		"claude-sonnet-4-5", "claude-opus-4-5",
	}
	// modelsWithoutSampling reject temperature on every API version.
	modelsWithoutSampling = []string{
		"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-haiku-5-5",
		"claude-fable-5-1", "claude-fable-5-2",
		"claude-mythos-5-1", "claude-mythos-5-2",
	}
	// modelsWithoutEffort reject output_config.effort.
	modelsWithoutEffort = []string{"claude-haiku-4-5", "claude-sonnet-4-5"}
	// modelsWithoutForcedTools reject tool_choice "any".
	modelsWithoutForcedTools = []string{
		"claude-opus-5-5", "claude-sonnet-5-5",
		"claude-fable-5-1", "claude-fable-5-2",
		"claude-mythos-5-1", "claude-mythos-5-2",
	}
	// modelsWithFallback take the server-side refusal fallback.
	modelsWithFallback = []string{
		"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5",
		"claude-fable-5-1", "claude-fable-5-2",
	}
)

func modelIn(model string, ids []string) bool {
	for _, id := range ids {
		if model == id || strings.HasPrefix(model, id+"-") {
			return true
		}
	}
	return false
}

// Chat starts a generation and returns its stream once the API has accepted
// the request. The context bounds the whole stream.
//
// llm.ToolChoiceNone keeps the tool definitions in the request, which the API
// needs to read earlier tool calls in the history, and sends tool_choice
// "none". llm.ToolChoiceRequired sends tool_choice "any"; on the models that
// reject forced tool use (Claude Opus 5.5, Sonnet 5.5, Fable 5.1 and 5.2) the
// choice stays automatic and an instruction to call a tool is added to the
// system prompt instead, so on those models a tool call is very likely but
// not guaranteed.
func (l *LLM) Chat(ctx context.Context, req llm.Request) (llm.Stream, error) {
	body, err := l.buildRequest(req)
	if err != nil {
		return nil, l.newError("chat", 0, false, err)
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, l.newError("chat", 0, false, fmt.Errorf("encode request: %w", err))
	}

	reqCtx, cancel := context.WithCancel(ctx)
	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, l.endpoint, bytes.NewReader(payload))
	if err != nil {
		cancel()
		return nil, l.newError("chat", 0, false, fmt.Errorf("build request: %w", err))
	}
	if key := l.apiKey(); key != "" {
		httpReq.Header.Set("x-api-key", key)
	}
	httpReq.Header.Set("anthropic-version", l.apiVersion)
	httpReq.Header.Set("content-type", "application/json")
	httpReq.Header.Set("accept", "text/event-stream")
	if l.fallback {
		httpReq.Header.Set("anthropic-beta", fallbackBeta)
	}

	resp, err := l.client.Do(httpReq)
	if err != nil {
		cancel()
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, l.newError("chat", 0, false, ctxErr)
		}
		return nil, l.newError("chat", 0, true, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		defer cancel()
		defer resp.Body.Close()
		return nil, l.newError("chat", resp.StatusCode, providers.RetryableStatus(resp.StatusCode), l.readAPIError(resp))
	}
	return newStream(ctx, l, resp.Body, cancel), nil
}

// The types below are the subset of the Messages API request this package
// sends.

type messagesRequest struct {
	Model        string        `json:"model"`
	MaxTokens    int           `json:"max_tokens"`
	Stream       bool          `json:"stream"`
	System       string        `json:"system,omitempty"`
	Messages     []apiMessage  `json:"messages"`
	Tools        []apiTool     `json:"tools,omitempty"`
	ToolChoice   *toolChoice   `json:"tool_choice,omitempty"`
	OutputConfig *outputConfig `json:"output_config,omitempty"`
	Temperature  *float64      `json:"temperature,omitempty"`
	Fallbacks    string        `json:"fallbacks,omitempty"`
}

type apiMessage struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
}

// contentBlock is a text, tool_use or tool_result block.
type contentBlock struct {
	Type string `json:"type"`
	// text
	Text string `json:"text,omitempty"`
	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// tool_result
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
	IsError   bool   `json:"is_error,omitempty"`
}

type apiTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type toolChoice struct {
	Type string `json:"type"`
}

type outputConfig struct {
	Effort Effort `json:"effort"`
}

const (
	roleUser      = "user"
	roleAssistant = "assistant"
)

func (l *LLM) buildRequest(req llm.Request) (*messagesRequest, error) {
	system, msgs, err := convertMessages(req.Messages)
	if err != nil {
		return nil, err
	}
	if len(msgs) == 0 {
		return nil, errors.New("request has no user, assistant or tool message")
	}

	body := &messagesRequest{
		Model:       l.model,
		MaxTokens:   l.maxTokens,
		Stream:      true,
		Messages:    msgs,
		Temperature: l.temperature,
	}
	if l.effort != "" {
		body.OutputConfig = &outputConfig{Effort: l.effort}
	}
	if l.fallback {
		body.Fallbacks = "default"
	}

	for _, t := range req.Tools {
		schema := t.Parameters
		if len(bytes.TrimSpace(schema)) == 0 {
			schema = json.RawMessage(emptyObjectSchema)
		}
		if !json.Valid(schema) {
			return nil, fmt.Errorf("tool %q: parameters are not valid JSON", t.Name)
		}
		body.Tools = append(body.Tools, apiTool{Name: t.Name, Description: t.Description, InputSchema: schema})
	}
	if len(body.Tools) > 0 {
		switch req.ToolChoice {
		case llm.ToolChoiceAuto:
		case llm.ToolChoiceNone:
			body.ToolChoice = &toolChoice{Type: "none"}
		case llm.ToolChoiceRequired:
			if modelIn(l.model, modelsWithoutForcedTools) {
				system = joinNonEmpty(system, requiredToolSuffix)
			} else {
				body.ToolChoice = &toolChoice{Type: "any"}
			}
		default:
			return nil, fmt.Errorf("unsupported tool choice %q", req.ToolChoice)
		}
	}
	body.System = system
	return body, nil
}

func joinNonEmpty(a, b string) string {
	if a == "" {
		return b
	}
	return a + "\n\n" + b
}

// convertMessages maps the history to the system prompt and the alternating
// user/assistant messages the API expects.
func convertMessages(in []llm.Message) (string, []apiMessage, error) {
	var (
		system []string
		out    []apiMessage
	)
	add := func(role string, blocks ...contentBlock) {
		if len(blocks) == 0 {
			return
		}
		if n := len(out); n > 0 && out[n-1].Role == role {
			out[n-1].Content = append(out[n-1].Content, blocks...)
			return
		}
		out = append(out, apiMessage{Role: role, Content: blocks})
	}

	for i, m := range in {
		switch m.Role {
		case llm.RoleSystem:
			if m.Content != "" {
				system = append(system, m.Content)
			}
		case llm.RoleUser:
			if m.Content != "" {
				add(roleUser, contentBlock{Type: "text", Text: m.Content})
			}
		case llm.RoleAssistant:
			var blocks []contentBlock
			if m.Content != "" {
				blocks = append(blocks, contentBlock{Type: "text", Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				input, err := toolInput(tc.Arguments)
				if err != nil {
					return "", nil, fmt.Errorf("message %d: tool call %q: %w", i, tc.Name, err)
				}
				blocks = append(blocks, contentBlock{Type: "tool_use", ID: tc.ID, Name: tc.Name, Input: input})
			}
			add(roleAssistant, blocks...)
		case llm.RoleTool:
			add(roleUser, contentBlock{Type: "tool_result", ToolUseID: m.ToolCallID, Content: m.Content, IsError: m.IsError})
		default:
			return "", nil, fmt.Errorf("message %d: unsupported role %q", i, m.Role)
		}
	}

	if len(out) > 0 && out[0].Role == roleAssistant {
		lead := apiMessage{Role: roleUser, Content: []contentBlock{{Type: "text", Text: LeadingUserPlaceholder}}}
		out = append([]apiMessage{lead}, out...)
	}
	if n := len(out); n > 0 && out[n-1].Role == roleAssistant {
		add(roleUser, contentBlock{Type: "text", Text: TrailingUserPlaceholder})
	}
	return strings.Join(system, "\n\n"), out, nil
}

// toolInput returns the arguments of a tool call as a JSON object.
func toolInput(args json.RawMessage) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(args)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return json.RawMessage(emptyObjectArgument), nil
	}
	if trimmed[0] != '{' || !json.Valid(trimmed) {
		return nil, errors.New("arguments are not a JSON object")
	}
	return json.RawMessage(trimmed), nil
}
