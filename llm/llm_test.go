package llm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

type weatherArgs struct {
	City  string   `json:"city" desc:"City name"`
	Unit  string   `json:"unit,omitempty" enum:"celsius,fahrenheit"`
	Days  int      `json:"days"`
	Tags  []string `json:"tags,omitempty"`
	Limit *float64 `json:"limit"`
}

func TestSchemaFor(t *testing.T) {
	raw, err := SchemaFor[weatherArgs]()
	if err != nil {
		t.Fatal(err)
	}
	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatal(err)
	}
	if s["type"] != "object" || s["additionalProperties"] != false {
		t.Fatalf("unexpected schema: %s", raw)
	}
	req, _ := json.Marshal(s["required"])
	if string(req) != `["city","days"]` {
		t.Fatalf("required = %s", req)
	}
	props := s["properties"].(map[string]any)
	if props["city"].(map[string]any)["description"] != "City name" {
		t.Fatalf("missing description: %s", raw)
	}
	if len(props["unit"].(map[string]any)["enum"].([]any)) != 2 {
		t.Fatalf("missing enum: %s", raw)
	}
	if _, err := SchemaFor[string](); err == nil {
		t.Fatal("expected error for non-struct")
	}
}

func TestValidateArguments(t *testing.T) {
	schema, _ := SchemaFor[weatherArgs]()
	cases := []struct {
		args string
		ok   bool
	}{
		{`{"city":"Paris","days":2}`, true},
		{`{"city":"Paris","days":2,"unit":"celsius","tags":["a"],"limit":1.5}`, true},
		{`{"city":"Paris"}`, false},                          // missing required
		{`{"city":"Paris","days":"2"}`, false},               // wrong type
		{`{"city":"Paris","days":2.5}`, false},               // not an integer
		{`{"city":"Paris","days":2,"unit":"kelvin"}`, false}, // enum
		{`{"city":"Paris","days":2,"x":1}`, false},           // unknown property
		{`{"city":"Paris","days":2,"tags":[1]}`, false},      // items
		{`[1]`, false},
		{`{"city":`, false},
		{`{"city":"Paris","days":2} {}`, false},
	}
	for _, c := range cases {
		err := ValidateArguments(schema, json.RawMessage(c.args))
		if (err == nil) != c.ok {
			t.Errorf("%s: ok=%v err=%v", c.args, c.ok, err)
		}
		if err != nil && !errors.Is(err, ErrInvalidArguments) {
			t.Errorf("%s: error does not wrap ErrInvalidArguments: %v", c.args, err)
		}
	}
	if err := ValidateArguments(nil, nil); err != nil {
		t.Errorf("empty args against empty schema: %v", err)
	}
}

func TestNewTool(t *testing.T) {
	tool, err := NewTool("get_weather", "Weather lookup", func(_ context.Context, a weatherArgs) (any, error) {
		return map[string]any{"city": a.City, "temp": 21}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := tool.Handler(context.Background(), json.RawMessage(`{"city":"Paris","days":1}`))
	if err != nil || !strings.Contains(out, `"temp":21`) {
		t.Fatalf("out=%q err=%v", out, err)
	}
	if _, err := tool.Handler(context.Background(), json.RawMessage(`{"nope":1}`)); !errors.Is(err, ErrInvalidArguments) {
		t.Fatalf("expected ErrInvalidArguments, got %v", err)
	}
	if _, err := NewTool("bad name!", "", func(context.Context, weatherArgs) (any, error) { return nil, nil }); err == nil {
		t.Fatal("expected invalid name error")
	}
	if (Tool{Name: "x"}).Validate() == nil {
		t.Fatal("expected missing handler error")
	}
}

func TestConversationCopies(t *testing.T) {
	c := NewConversation(Message{Role: RoleUser, Content: "hi"})
	msgs := c.Messages()
	msgs[0].Content = "changed"
	if c.Messages()[0].Content != "hi" || c.Len() != 1 {
		t.Fatal("Messages must return a copy")
	}
}
