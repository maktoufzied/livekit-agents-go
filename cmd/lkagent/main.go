// Command lkagent creates and deploys Go agents on LiveKit Cloud (managed
// deployment).
//
// The official CLI's `lk agent create` and `lk agent deploy` only accept
// Python and Node.js projects: they look for package.json or Python project
// files and check the version of the official SDK. This command calls the
// same LiveKit Cloud API through the official Go client
// (server-sdk-go/pkg/cloudagents) without those language checks: it uploads
// the working directory, and LiveKit Cloud builds its Dockerfile and runs
// the image.
//
// It writes the same livekit.toml as the official CLI, so that the other
// `lk agent` commands (status, logs, secrets, rollback, delete ...) can be
// used afterwards.
//
//	export LIVEKIT_URL=wss://<project>.livekit.cloud LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
//	go run ./cmd/lkagent regions
//	go run ./cmd/lkagent create -region <region> -secrets-file .env .
//	go run ./cmd/lkagent deploy .
//	go run ./cmd/lkagent list
//	go run ./cmd/lkagent delete -id <agent id>
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/livekit/protocol/livekit"
	"github.com/livekit/server-sdk-go/v2/pkg/cloudagents"
)

// buildTimeout is a little over the 10 minute limit LiveKit Cloud puts on builds.
const buildTimeout = 12 * time.Minute

func usage() {
	fmt.Fprint(os.Stderr, `lkagent: create and deploy Go agents on LiveKit Cloud

Usage:
  lkagent regions                                   list the regions agents can be deployed to
  lkagent create -region R [-secrets-file F] [dir]  create an agent from dir and deploy it
  lkagent deploy [-id ID] [-secrets-file F] [-deployment D] [dir]
                                                    build and deploy a new version
  lkagent list                                      list the agents of the project
  lkagent delete -id ID                             delete an agent

The project comes from LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET.
dir defaults to the current directory and must contain a Dockerfile.
`)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var err error
	switch cmd, args := os.Args[1], os.Args[2:]; cmd {
	case "regions":
		err = runRegions(ctx)
	case "create":
		err = runCreate(ctx, args)
	case "deploy":
		err = runDeploy(ctx, args)
	case "list":
		err = runList(ctx)
	case "delete":
		err = runDelete(ctx, args)
	case "help", "-h", "--help":
		usage()
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "lkagent:", err)
		os.Exit(1)
	}
}

type project struct {
	url, key, secret string
}

func projectFromEnv() (project, error) {
	p := project{os.Getenv("LIVEKIT_URL"), os.Getenv("LIVEKIT_API_KEY"), os.Getenv("LIVEKIT_API_SECRET")}
	if p.url == "" || p.key == "" || p.secret == "" {
		return p, errors.New("LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set")
	}
	if _, err := subdomainOf(p.url); err != nil {
		return p, err
	}
	return p, nil
}

func newClient() (*cloudagents.Client, project, error) {
	p, err := projectFromEnv()
	if err != nil {
		return nil, p, err
	}
	c, err := cloudagents.New(cloudagents.WithProject(p.url, p.key, p.secret))
	return c, p, err
}

func settings(ctx context.Context, c *cloudagents.Client) (map[string]string, error) {
	resp, err := c.GetClientSettings(ctx, &livekit.ClientSettingsRequest{})
	if err != nil {
		return nil, fmt.Errorf("get client settings: %w", err)
	}
	out := map[string]string{}
	for _, p := range resp.GetParams() {
		out[p.GetName()] = p.GetValue()
	}
	return out, nil
}

func availableRegions(s map[string]string) []string {
	var regions []string
	for _, r := range strings.Split(s["available_regions"], ",") {
		if r = strings.Trim(strings.TrimSpace(r), `"`); r != "" {
			regions = append(regions, r)
		}
	}
	sort.Strings(regions)
	return regions
}

func runRegions(ctx context.Context) error {
	c, _, err := newClient()
	if err != nil {
		return err
	}
	s, err := settings(ctx, c)
	if err != nil {
		return err
	}
	regions := availableRegions(s)
	if len(regions) == 0 {
		return errors.New("the project reports no region available for agent deployment")
	}
	for _, r := range regions {
		fmt.Println(r)
	}
	return nil
}

func runCreate(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("create", flag.ExitOnError)
	region := fs.String("region", "", "region to deploy to (see `lkagent regions`)")
	secretsFile := fs.String("secrets-file", "", "file of KEY=VALUE secrets injected as environment variables")
	_ = fs.Parse(args)
	dir := dirArg(fs)

	if err := preflight(dir); err != nil {
		return err
	}
	if existing, _ := readConfig(dir); existing.AgentID != "" {
		return fmt.Errorf("%s already names agent %s: use `lkagent deploy`, or remove the file to create another agent", configFile, existing.AgentID)
	}
	secrets, skipped, err := loadSecrets(*secretsFile)
	if err != nil {
		return err
	}
	reportSecrets(secrets, skipped)

	c, p, err := newClient()
	if err != nil {
		return err
	}
	s, err := settings(ctx, c)
	if err != nil {
		return err
	}
	regions := availableRegions(s)
	if *region == "" {
		return fmt.Errorf("-region is required; available regions: %s", strings.Join(regions, ", "))
	}
	if len(regions) > 0 && !contains(regions, *region) {
		return fmt.Errorf("region %q is not available; available regions: %s", *region, strings.Join(regions, ", "))
	}

	buildCtx, cancel := context.WithTimeout(ctx, buildTimeout)
	defer cancel()
	fmt.Fprintf(os.Stderr, "creating agent in %s from %s (LiveKit Cloud builds the Dockerfile)...\n", *region, dir)
	resp, err := c.CreateAgent(buildCtx, os.DirFS(dir), secrets, []string{*region}, nil, []string{"**/" + configFile}, os.Stderr)
	if resp != nil && resp.AgentId != "" {
		// Save the id even when the build failed: the agent exists and
		// can be redeployed or deleted.
		sub, _ := subdomainOf(p.url)
		if werr := writeConfig(dir, config{Subdomain: sub, AgentID: resp.AgentId}); werr != nil {
			fmt.Fprintf(os.Stderr, "warning: could not write %s: %v\n", configFile, werr)
		}
	}
	if err != nil {
		return fmt.Errorf("create agent: %w", err)
	}
	fmt.Fprintf(os.Stderr, "created agent %s; wrote %s\n", resp.AgentId, configFile)
	fmt.Println(resp.AgentId)
	return nil
}

func runDeploy(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("deploy", flag.ExitOnError)
	id := fs.String("id", "", "agent id (default: the one in livekit.toml)")
	secretsFile := fs.String("secrets-file", "", "file of KEY=VALUE secrets injected as environment variables")
	deployment := fs.String("deployment", "", "deployment name (default: production)")
	_ = fs.Parse(args)
	dir := dirArg(fs)

	if err := preflight(dir); err != nil {
		return err
	}
	if *id == "" {
		cfg, err := readConfig(dir)
		if err != nil || cfg.AgentID == "" {
			return fmt.Errorf("no agent id: pass -id or run `lkagent create` first (%s not found or without an agent id)", configFile)
		}
		*id = cfg.AgentID
	}
	secrets, skipped, err := loadSecrets(*secretsFile)
	if err != nil {
		return err
	}
	reportSecrets(secrets, skipped)

	c, _, err := newClient()
	if err != nil {
		return err
	}
	buildCtx, cancel := context.WithTimeout(ctx, buildTimeout)
	defer cancel()
	fmt.Fprintf(os.Stderr, "deploying a new version of agent %s from %s...\n", *id, dir)
	if err := c.DeployAgentV2(buildCtx, *id, os.DirFS(dir), secrets, nil, *deployment, []string{"**/" + configFile}, os.Stderr); err != nil {
		return fmt.Errorf("deploy agent: %w", err)
	}
	fmt.Fprintf(os.Stderr, "deployed agent %s\n", *id)
	return nil
}

func runList(ctx context.Context) error {
	c, _, err := newClient()
	if err != nil {
		return err
	}
	resp, err := c.ListAgents(ctx, &livekit.ListAgentsRequest{})
	if err != nil {
		return fmt.Errorf("list agents: %w", err)
	}
	if len(resp.GetAgents()) == 0 {
		fmt.Fprintln(os.Stderr, "no agents in this project")
		return nil
	}
	fmt.Printf("%-24s %-16s %-14s %-12s %s\n", "ID", "VERSION", "REGION", "STATUS", "REPLICAS")
	for _, a := range resp.GetAgents() {
		if len(a.GetAgentDeployments()) == 0 {
			fmt.Printf("%-24s %-16s %-14s %-12s %s\n", a.GetAgentId(), a.GetVersion(), "-", "-", "-")
		}
		for _, d := range a.GetAgentDeployments() {
			fmt.Printf("%-24s %-16s %-14s %-12s %d\n", a.GetAgentId(), a.GetVersion(), d.GetRegion(), d.GetStatus(), d.GetReplicas())
		}
	}
	return nil
}

func runDelete(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("delete", flag.ExitOnError)
	id := fs.String("id", "", "agent id (required)")
	_ = fs.Parse(args)
	if *id == "" {
		return errors.New("-id is required")
	}
	c, _, err := newClient()
	if err != nil {
		return err
	}
	if _, err := c.DeleteAgent(ctx, &livekit.DeleteAgentRequest{AgentId: *id}); err != nil {
		return fmt.Errorf("delete agent: %w", err)
	}
	fmt.Fprintf(os.Stderr, "deleted agent %s\n", *id)
	return nil
}

func dirArg(fs *flag.FlagSet) string {
	if fs.NArg() > 0 {
		return fs.Arg(0)
	}
	return "."
}

func contains(list []string, v string) bool {
	for _, x := range list {
		if x == v {
			return true
		}
	}
	return false
}

func reportSecrets(secrets []*livekit.AgentSecret, skipped []string) {
	names := make([]string, len(secrets))
	for i, s := range secrets {
		names[i] = s.Name
	}
	if len(names) > 0 {
		fmt.Fprintf(os.Stderr, "secrets: %s\n", strings.Join(names, ", "))
	}
	for _, s := range skipped {
		fmt.Fprintf(os.Stderr, "secrets: skipped %s\n", s)
	}
}
