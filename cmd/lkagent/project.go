package main

import (
	"bufio"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/livekit/protocol/livekit"
)

// configFile is the project file shared with the official LiveKit CLI.
const configFile = "livekit.toml"

// config is the part of livekit.toml this command reads and writes:
//
//	[project]
//	  subdomain = "my-project"
//
//	[agent]
//	  id = "CA_..."
type config struct {
	Subdomain string
	AgentID   string
}

var (
	tomlSection = regexp.MustCompile(`^\s*\[([A-Za-z0-9_.-]+)\]\s*$`)
	tomlString  = regexp.MustCompile(`^\s*([A-Za-z0-9_-]+)\s*=\s*"([^"]*)"\s*(#.*)?$`)
)

func readConfig(dir string) (config, error) {
	f, err := os.Open(filepath.Join(dir, configFile))
	if err != nil {
		return config{}, err
	}
	defer f.Close()
	var cfg config
	section := ""
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if m := tomlSection.FindStringSubmatch(line); m != nil {
			section = m[1]
			continue
		}
		m := tomlString.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		switch section + "." + m[1] {
		case "project.subdomain":
			cfg.Subdomain = m[2]
		case "agent.id":
			cfg.AgentID = m[2]
		}
	}
	return cfg, sc.Err()
}

func writeConfig(dir string, cfg config) error {
	body := fmt.Sprintf("[project]\n  subdomain = %q\n\n[agent]\n  id = %q\n", cfg.Subdomain, cfg.AgentID)
	return os.WriteFile(filepath.Join(dir, configFile), []byte(body), 0o644)
}

// subdomainOf returns the project subdomain of a LiveKit Cloud URL such as
// wss://my-project.livekit.cloud.
func subdomainOf(raw string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil || u.Hostname() == "" {
		return "", fmt.Errorf("invalid LIVEKIT_URL %q", raw)
	}
	host := u.Hostname()
	if !strings.HasSuffix(host, ".livekit.cloud") {
		return "", fmt.Errorf("LIVEKIT_URL host %q is not a LiveKit Cloud project (expected <project>.livekit.cloud)", host)
	}
	return strings.SplitN(host, ".", 2)[0], nil
}

// reservedSecrets are injected by LiveKit Cloud at runtime and must not be
// set as secrets.
var reservedSecrets = map[string]bool{"LIVEKIT_URL": true, "LIVEKIT_API_KEY": true, "LIVEKIT_API_SECRET": true}

// loadSecrets parses a KEY=VALUE file (the .env format: comments, blank
// lines, optional `export`, optional quotes). Reserved LiveKit variables and
// empty values are skipped and reported by name. Values are never printed.
func loadSecrets(path string) (secrets []*livekit.AgentSecret, skipped []string, err error) {
	if path == "" {
		return nil, nil, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("secrets file: %w", err)
	}
	defer f.Close()
	values := map[string]string{}
	sc := bufio.NewScanner(f)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		key, value, ok := strings.Cut(line, "=")
		key = strings.TrimSpace(key)
		if !ok || key == "" || strings.ContainsAny(key, " \t") {
			return nil, nil, fmt.Errorf("secrets file %s:%d: expected KEY=VALUE", path, n)
		}
		values[key] = unquote(strings.TrimSpace(value))
	}
	if err := sc.Err(); err != nil {
		return nil, nil, fmt.Errorf("secrets file: %w", err)
	}
	keys := make([]string, 0, len(values))
	for k := range values {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		switch {
		case reservedSecrets[k]:
			skipped = append(skipped, k+" (set by LiveKit Cloud)")
		case values[k] == "":
			skipped = append(skipped, k+" (empty)")
		default:
			secrets = append(secrets, &livekit.AgentSecret{Name: k, Value: []byte(values[k])})
		}
	}
	return secrets, skipped, nil
}

func unquote(v string) string {
	if len(v) >= 2 && (v[0] == '"' || v[0] == '\'') && v[len(v)-1] == v[0] {
		return v[1 : len(v)-1]
	}
	// An unquoted value ends at an inline comment.
	if i := strings.Index(v, " #"); i >= 0 {
		v = strings.TrimSpace(v[:i])
	}
	return v
}

// preflight checks what LiveKit Cloud documents for agent images, before
// uploading anything.
func preflight(dir string) error {
	data, err := os.ReadFile(filepath.Join(dir, "Dockerfile"))
	if err != nil {
		return fmt.Errorf("%s needs a Dockerfile: LiveKit Cloud builds the agent image from it", dir)
	}
	var problems []string
	finalFrom, hasUser, hasStart := "", false, false
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		switch strings.ToUpper(fields[0]) {
		case "FROM":
			if len(fields) > 1 {
				finalFrom = fields[1]
				hasUser, hasStart = false, false // rules apply to the final stage
			}
		case "USER":
			hasUser = len(fields) > 1 && fields[1] != "root" && fields[1] != "0"
		case "ENTRYPOINT", "CMD":
			hasStart = true
		}
	}
	if strings.Contains(strings.ToLower(finalFrom), "alpine") {
		problems = append(problems, "the final image is based on Alpine (musl), which LiveKit Cloud does not support; use a glibc image such as debian:bookworm-slim")
	}
	if !hasUser {
		problems = append(problems, "the final stage has no non-root USER; LiveKit Cloud requires agents not to run as root")
	}
	if !hasStart {
		problems = append(problems, "the final stage has no ENTRYPOINT or CMD that starts the agent")
	}
	if len(problems) > 0 {
		return errors.New("Dockerfile does not meet LiveKit Cloud requirements:\n  - " + strings.Join(problems, "\n  - "))
	}
	return nil
}
