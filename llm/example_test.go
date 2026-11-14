package llm_test

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/maktoufzied/livekit-agents-go/llm"
)

func ExampleNewTool() {
	type args struct {
		Query string `json:"query" desc:"What to search for"`
		Limit int    `json:"limit,omitempty" desc:"Maximum number of results"`
	}
	tool, err := llm.NewTool("search", "Search the knowledge base.",
		func(_ context.Context, a args) (any, error) {
			return []string{"result for " + a.Query}, nil
		})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(tool.Parameters))

	out, _ := tool.Handler(context.Background(), json.RawMessage(`{"query":"opening hours"}`))
	fmt.Println(out)
	// Output:
	// {"additionalProperties":false,"properties":{"limit":{"description":"Maximum number of results","type":"integer"},"query":{"description":"What to search for","type":"string"}},"required":["query"],"type":"object"}
	// ["result for opening hours"]
}

func ExampleValidateArguments() {
	schema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`)
	fmt.Println(llm.ValidateArguments(schema, json.RawMessage(`{"city":"Lyon"}`)))
	fmt.Println(llm.ValidateArguments(schema, json.RawMessage(`{"town":"Lyon"}`)))
	// Output:
	// <nil>
	// llm: invalid tool arguments: $: missing required property "city"
}
