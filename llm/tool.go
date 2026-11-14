package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"time"
)

// ToolHandler runs a tool. args has already been validated against the
// tool's schema. The context is cancelled when the tool times out, when the
// session closes, or (unless the tool is Uninterruptible) when the user
// interrupts the turn. The returned string is sent to the model.
type ToolHandler func(ctx context.Context, args json.RawMessage) (string, error)

// Tool is a function the model can call.
type Tool struct {
	Name        string
	Description string
	// Parameters is a JSON Schema object. Nil means "no arguments".
	Parameters json.RawMessage
	// Timeout bounds one execution. Zero selects the session default.
	Timeout time.Duration
	// Uninterruptible keeps the tool running when the user interrupts the
	// turn. Use it for tools with external side effects that must not be
	// abandoned half-way. The result is still recorded in the history.
	Uninterruptible bool
	Handler         ToolHandler
}

// Spec returns the model-facing description.
func (t Tool) Spec() ToolSpec {
	params := t.Parameters
	if len(params) == 0 {
		params = json.RawMessage(`{"type":"object","properties":{}}`)
	}
	return ToolSpec{Name: t.Name, Description: t.Description, Parameters: params}
}

var toolNameRe = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// Validate checks the tool definition.
func (t Tool) Validate() error {
	if !toolNameRe.MatchString(t.Name) {
		return fmt.Errorf("llm: invalid tool name %q: must match %s", t.Name, toolNameRe)
	}
	if t.Handler == nil {
		return fmt.Errorf("llm: tool %q has no handler", t.Name)
	}
	if len(t.Parameters) > 0 {
		var schema map[string]any
		if err := json.Unmarshal(t.Parameters, &schema); err != nil {
			return fmt.Errorf("llm: tool %q: parameters are not a JSON object: %w", t.Name, err)
		}
		if typ, _ := schema["type"].(string); typ != "object" {
			return fmt.Errorf("llm: tool %q: parameters schema must have type \"object\"", t.Name)
		}
	}
	return nil
}

// ToolError is an error whose message is safe and useful to show to the
// model. Any other error returned by a handler is reported to the model as a
// generic failure so internal details do not leak into the conversation.
type ToolError struct{ Message string }

func (e *ToolError) Error() string { return e.Message }

// NewToolError returns a ToolError.
func NewToolError(format string, args ...any) error {
	return &ToolError{Message: fmt.Sprintf(format, args...)}
}

// ErrInvalidArguments wraps argument validation failures.
var ErrInvalidArguments = errors.New("llm: invalid tool arguments")

// ToolOption customizes NewTool.
type ToolOption func(*Tool)

// WithTimeout sets the per-call timeout.
func WithTimeout(d time.Duration) ToolOption { return func(t *Tool) { t.Timeout = d } }

// WithUninterruptible marks the tool as not cancellable by interruptions.
func WithUninterruptible() ToolOption { return func(t *Tool) { t.Uninterruptible = true } }

// NewTool builds a Tool from a typed function. The JSON Schema is derived
// from A, which must be a struct (or pointer to struct). Field names come
// from `json` tags; `desc:"..."` adds a description; `enum:"a,b"` restricts
// values; fields are required unless tagged `omitempty` or of pointer type.
//
// The result may be a string or any JSON-encodable value.
func NewTool[A any](name, description string, fn func(ctx context.Context, args A) (any, error), opts ...ToolOption) (Tool, error) {
	schema, err := SchemaFor[A]()
	if err != nil {
		return Tool{}, fmt.Errorf("llm: tool %q: %w", name, err)
	}
	t := Tool{
		Name:        name,
		Description: description,
		Parameters:  schema,
		Handler: func(ctx context.Context, raw json.RawMessage) (string, error) {
			var args A
			dec := json.NewDecoder(bytes.NewReader(raw))
			dec.DisallowUnknownFields()
			if err := dec.Decode(&args); err != nil {
				return "", fmt.Errorf("%w: %v", ErrInvalidArguments, err)
			}
			out, err := fn(ctx, args)
			if err != nil {
				return "", err
			}
			if s, ok := out.(string); ok {
				return s, nil
			}
			b, err := json.Marshal(out)
			if err != nil {
				return "", fmt.Errorf("encode tool result: %w", err)
			}
			return string(b), nil
		},
	}
	for _, o := range opts {
		o(&t)
	}
	if err := t.Validate(); err != nil {
		return Tool{}, err
	}
	return t, nil
}

// MustTool is NewTool that panics on error. Use it for package-level tools.
func MustTool[A any](name, description string, fn func(ctx context.Context, args A) (any, error), opts ...ToolOption) Tool {
	t, err := NewTool(name, description, fn, opts...)
	if err != nil {
		panic(err)
	}
	return t
}
