// Command dispatch sends the "support-agent" agent to a room with the
// LiveKit agent dispatch API, and prints a token a user can join with.
//
//	go run ./examples/explicit-dispatch/dispatch -room demo -customer Ada
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/livekit/protocol/auth"
	"github.com/livekit/protocol/livekit"
	lksdk "github.com/livekit/server-sdk-go/v2"
)

func main() {
	room := flag.String("room", "demo", "room to dispatch the agent to")
	agent := flag.String("agent", "support-agent", "agent name, as registered by the worker")
	customer := flag.String("customer", "Ada", "customer name passed as dispatch metadata")
	flag.Parse()

	url, key, secret := os.Getenv("LIVEKIT_URL"), os.Getenv("LIVEKIT_API_KEY"), os.Getenv("LIVEKIT_API_SECRET")
	if url == "" || key == "" || secret == "" {
		fmt.Fprintln(os.Stderr, "LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set")
		os.Exit(1)
	}
	// The server APIs are HTTP: accept the ws:// form of the URL too.
	httpURL := strings.NewReplacer("wss://", "https://", "ws://", "http://").Replace(url)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	metadata, _ := json.Marshal(map[string]string{"customer": *customer})
	dispatch, err := lksdk.NewAgentDispatchServiceClient(httpURL, key, secret).CreateDispatch(ctx, &livekit.CreateAgentDispatchRequest{
		Room:      *room,
		AgentName: *agent,
		Metadata:  string(metadata),
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "create dispatch:", err)
		os.Exit(1)
	}
	fmt.Printf("dispatch %s created: agent %q is joining room %q\n", dispatch.Id, *agent, *room)

	token, err := auth.NewAccessToken(key, secret).
		SetIdentity("user-" + *customer).
		SetValidFor(time.Hour).
		SetVideoGrant(&auth.VideoGrant{RoomJoin: true, Room: *room}).
		ToJWT()
	if err != nil {
		fmt.Fprintln(os.Stderr, "sign token:", err)
		os.Exit(1)
	}
	fmt.Printf("\njoin as a user with this token (valid 1h), e.g. in https://agents-playground.livekit.io or `lk room join`:\n\n%s\n", token)
}
