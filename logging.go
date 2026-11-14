package agents

import (
	"context"
	"log/slog"
	"strings"
)

// sensitiveKeyParts are substrings of attribute keys whose values are
// redacted by NewRedactingHandler.
var sensitiveKeyParts = []string{"token", "secret", "password", "authorization", "api_key", "apikey", "api-key", "credential"}

// NewRedactingHandler wraps a slog.Handler and replaces the value of any
// attribute whose key looks like a credential (token, secret, password,
// authorization, api_key, credential) with "[REDACTED]", including inside
// groups.
//
// The SDK itself never logs credentials: API keys are carried as Secret
// values, which redact themselves. This handler is a second line of defence
// for application code logging through the same logger.
func NewRedactingHandler(h slog.Handler) slog.Handler { return &redactingHandler{h} }

type redactingHandler struct{ inner slog.Handler }

func (h *redactingHandler) Enabled(ctx context.Context, l slog.Level) bool {
	return h.inner.Enabled(ctx, l)
}

func (h *redactingHandler) Handle(ctx context.Context, r slog.Record) error {
	out := slog.NewRecord(r.Time, r.Level, r.Message, r.PC)
	r.Attrs(func(a slog.Attr) bool {
		out.AddAttrs(redactAttr(a))
		return true
	})
	return h.inner.Handle(ctx, out)
}

func (h *redactingHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	red := make([]slog.Attr, len(attrs))
	for i, a := range attrs {
		red[i] = redactAttr(a)
	}
	return &redactingHandler{h.inner.WithAttrs(red)}
}

func (h *redactingHandler) WithGroup(name string) slog.Handler {
	return &redactingHandler{h.inner.WithGroup(name)}
}

func redactAttr(a slog.Attr) slog.Attr {
	a.Value = a.Value.Resolve()
	if a.Value.Kind() == slog.KindGroup {
		group := a.Value.Group()
		red := make([]slog.Attr, len(group))
		for i, g := range group {
			red[i] = redactAttr(g)
		}
		return slog.Attr{Key: a.Key, Value: slog.GroupValue(red...)}
	}
	key := strings.ToLower(a.Key)
	for _, part := range sensitiveKeyParts {
		if strings.Contains(key, part) {
			return slog.String(a.Key, "[REDACTED]")
		}
	}
	return a
}
