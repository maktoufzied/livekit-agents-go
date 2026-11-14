package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, dir, name, body string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadSecrets(t *testing.T) {
	p := write(t, t.TempDir(), ".env", `# comment
LIVEKIT_URL=wss://x.livekit.cloud
LIVEKIT_API_KEY=key
LIVEKIT_API_SECRET=secret

DEEPGRAM_API_KEY=dg-123
export ANTHROPIC_API_KEY="sk-ant # not a comment"
OPENAI_API_KEY=
LLM_MODEL=claude-haiku-5-5 # fastest
QUOTED='single quoted'
`)
	secrets, skipped, err := loadSecrets(p)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, s := range secrets {
		got[s.Name] = string(s.Value)
	}
	want := map[string]string{
		"DEEPGRAM_API_KEY":  "dg-123",
		"ANTHROPIC_API_KEY": "sk-ant # not a comment",
		"LLM_MODEL":         "claude-haiku-5-5",
		"QUOTED":            "single quoted",
	}
	if len(got) != len(want) {
		t.Fatalf("secrets = %v", got)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s = %q, want %q", k, got[k], v)
		}
	}
	// Reserved and empty variables are skipped, and reported by name only.
	joined := strings.Join(skipped, "|")
	for _, name := range []string{"LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "OPENAI_API_KEY"} {
		if !strings.Contains(joined, name) {
			t.Errorf("%s not reported as skipped: %v", name, skipped)
		}
	}
	if strings.Contains(joined, "secret") && strings.Contains(joined, "=") {
		t.Errorf("skipped report leaks values: %v", skipped)
	}

	if _, _, err := loadSecrets(write(t, t.TempDir(), "bad", "NOT A PAIR\n")); err == nil {
		t.Error("malformed line accepted")
	}
	if s, sk, err := loadSecrets(""); err != nil || s != nil || sk != nil {
		t.Error("empty path must be a no-op")
	}
	if _, _, err := loadSecrets(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Error("missing file accepted")
	}
}

func TestConfigRoundTrip(t *testing.T) {
	dir := t.TempDir()
	if _, err := readConfig(dir); err == nil {
		t.Fatal("missing file must be an error")
	}
	if err := writeConfig(dir, config{Subdomain: "my-project", AgentID: "CA_abc123"}); err != nil {
		t.Fatal(err)
	}
	cfg, err := readConfig(dir)
	if err != nil || cfg.Subdomain != "my-project" || cfg.AgentID != "CA_abc123" {
		t.Fatalf("round trip: %+v %v", cfg, err)
	}
	// The layout written by the official CLI.
	write(t, dir, configFile, "[project]\n  subdomain = \"other\"\n\n[agent]\n  id = \"CA_zzz\" # prod\n  name = \"x\"\n")
	cfg, err = readConfig(dir)
	if err != nil || cfg.Subdomain != "other" || cfg.AgentID != "CA_zzz" {
		t.Fatalf("official layout: %+v %v", cfg, err)
	}
}

func TestSubdomainOf(t *testing.T) {
	if s, err := subdomainOf("wss://my-project-abc123.livekit.cloud"); err != nil || s != "my-project-abc123" {
		t.Fatalf("got %q %v", s, err)
	}
	for _, bad := range []string{"", "ws://localhost:7880", "wss://example.com", "not a url"} {
		if _, err := subdomainOf(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestPreflight(t *testing.T) {
	good := "FROM golang:1.27-bookworm AS build\nRUN go build -o /agent .\n\nFROM debian:bookworm-slim\nUSER agent\nENTRYPOINT [\"/agent\"]\n"
	cases := []struct {
		name, dockerfile, wantErr string
	}{
		{"good", good, ""},
		{"alpine", strings.Replace(good, "debian:bookworm-slim", "alpine:3.20", 1), "Alpine"},
		{"root", strings.Replace(good, "USER agent\n", "", 1), "non-root USER"},
		{"explicit root", strings.Replace(good, "USER agent", "USER root", 1), "non-root USER"},
		{"no start", strings.Replace(good, "ENTRYPOINT [\"/agent\"]\n", "", 1), "ENTRYPOINT or CMD"},
		// Rules apply to the final stage only.
		{"user only in build stage", "FROM golang:1.27 AS build\nUSER app\nCMD [\"x\"]\nFROM debian:bookworm-slim\nCOPY --from=build /a /a\n", "non-root USER"},
	}
	for _, c := range cases {
		dir := t.TempDir()
		write(t, dir, "Dockerfile", c.dockerfile)
		err := preflight(dir)
		switch {
		case c.wantErr == "" && err != nil:
			t.Errorf("%s: unexpected error: %v", c.name, err)
		case c.wantErr != "" && (err == nil || !strings.Contains(err.Error(), c.wantErr)):
			t.Errorf("%s: want error containing %q, got %v", c.name, c.wantErr, err)
		}
	}
	if err := preflight(t.TempDir()); err == nil || !strings.Contains(err.Error(), "needs a Dockerfile") {
		t.Errorf("missing Dockerfile: %v", err)
	}
	// This repository's own Dockerfile must pass.
	if err := preflight("../.."); err != nil {
		t.Errorf("repository Dockerfile: %v", err)
	}
}
