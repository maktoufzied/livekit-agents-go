package agents_test

import (
	"bytes"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	agents "github.com/maktoufzied/livekit-agents-go"
)

func TestSecretsNeverReachLogs(t *testing.T) {
	var buf bytes.Buffer
	log := slog.New(agents.NewRedactingHandler(slog.NewJSONHandler(&buf, nil)))
	secret := agents.Secret("sk-very-secret")

	log.Info("connecting",
		"api_key", "sk-plain-string", // redacted by key name
		"key", secret, // redacted by type
		slog.Group("auth", "Authorization", "Bearer abc", "user", "zoe"),
		"room", "lobby")
	log.With("job_token", "eyJhbGciOi").Info(fmt.Sprintf("%v %s %+v %#v", secret, secret, struct{ K agents.Secret }{secret}, secret))

	out := buf.String()
	for _, leak := range []string{"sk-very-secret", "sk-plain-string", "Bearer abc", "eyJhbGciOi"} {
		if strings.Contains(out, leak) {
			t.Errorf("%q leaked into logs: %s", leak, out)
		}
	}
	for _, keep := range []string{"lobby", "zoe", "[REDACTED]"} {
		if !strings.Contains(out, keep) {
			t.Errorf("%q missing from logs: %s", keep, out)
		}
	}
	if secret.Reveal() != "sk-very-secret" {
		t.Error("Reveal must return the value")
	}
}
