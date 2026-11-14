package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
)

// SchemaFor derives a JSON Schema object from struct type A.
func SchemaFor[A any]() (json.RawMessage, error) {
	t := reflect.TypeOf((*A)(nil)).Elem()
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("tool arguments must be a struct, got %s", t)
	}
	s, err := schemaOf(t, 0)
	if err != nil {
		return nil, err
	}
	return json.Marshal(s)
}

func schemaOf(t reflect.Type, depth int) (map[string]any, error) {
	if depth > 16 {
		return nil, fmt.Errorf("schema too deep (recursive type %s?)", t)
	}
	switch t.Kind() {
	case reflect.Pointer:
		return schemaOf(t.Elem(), depth+1)
	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	case reflect.Slice, reflect.Array:
		items, err := schemaOf(t.Elem(), depth+1)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "array", "items": items}, nil
	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("unsupported map key type %s", t.Key())
		}
		vals, err := schemaOf(t.Elem(), depth+1)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "object", "additionalProperties": vals}, nil
	case reflect.Struct:
		props := map[string]any{}
		required := []string{}
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if !f.IsExported() {
				continue
			}
			tag := f.Tag.Get("json")
			if tag == "-" {
				continue
			}
			name, opts, _ := strings.Cut(tag, ",")
			if name == "" {
				name = f.Name
			}
			fs, err := schemaOf(f.Type, depth+1)
			if err != nil {
				return nil, fmt.Errorf("field %s: %w", f.Name, err)
			}
			if d := f.Tag.Get("desc"); d != "" {
				fs["description"] = d
			}
			if e := f.Tag.Get("enum"); e != "" {
				vals := []any{}
				for _, v := range strings.Split(e, ",") {
					vals = append(vals, strings.TrimSpace(v))
				}
				fs["enum"] = vals
			}
			props[name] = fs
			if !strings.Contains(opts, "omitempty") && f.Type.Kind() != reflect.Pointer {
				required = append(required, name)
			}
		}
		sort.Strings(required)
		s := map[string]any{"type": "object", "properties": props, "additionalProperties": false}
		if len(required) > 0 {
			s["required"] = required
		}
		return s, nil
	}
	return nil, fmt.Errorf("unsupported type %s", t)
}

// ValidateArguments checks args against a JSON Schema. It supports the
// subset of JSON Schema used for tool parameters: type, properties,
// required, additionalProperties, items, enum, minimum and maximum. Unknown
// keywords are ignored. Errors wrap ErrInvalidArguments.
func ValidateArguments(schema, args json.RawMessage) error {
	if len(bytes.TrimSpace(args)) == 0 {
		args = json.RawMessage(`{}`)
	}
	var s any
	if len(schema) > 0 {
		if err := json.Unmarshal(schema, &s); err != nil {
			return fmt.Errorf("invalid tool schema: %w", err)
		}
	}
	dec := json.NewDecoder(bytes.NewReader(args))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return fmt.Errorf("%w: not valid JSON: %v", ErrInvalidArguments, err)
	}
	if dec.More() {
		return fmt.Errorf("%w: trailing data after JSON value", ErrInvalidArguments)
	}
	if s == nil {
		if _, ok := v.(map[string]any); !ok {
			return fmt.Errorf("%w: expected an object", ErrInvalidArguments)
		}
		return nil
	}
	if err := validateValue(s, v, "$"); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidArguments, err)
	}
	return nil
}

func validateValue(schema, v any, path string) error {
	s, ok := schema.(map[string]any)
	if !ok {
		return nil // boolean or unknown schema: accept
	}
	if types := schemaTypes(s["type"]); len(types) > 0 {
		matched := false
		for _, t := range types {
			if matchesType(t, v) {
				matched = true
				break
			}
		}
		if !matched {
			return fmt.Errorf("%s: expected %s, got %s", path, strings.Join(types, " or "), jsonType(v))
		}
	}
	if enum, ok := s["enum"].([]any); ok {
		found := false
		for _, e := range enum {
			if fmt.Sprint(e) == fmt.Sprint(v) && jsonType(e) == jsonType(v) {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("%s: value is not one of the allowed values", path)
		}
	}
	switch val := v.(type) {
	case json.Number:
		f, err := val.Float64()
		if err != nil {
			return fmt.Errorf("%s: invalid number", path)
		}
		if min, ok := s["minimum"].(float64); ok && f < min {
			return fmt.Errorf("%s: must be >= %v", path, min)
		}
		if max, ok := s["maximum"].(float64); ok && f > max {
			return fmt.Errorf("%s: must be <= %v", path, max)
		}
	case map[string]any:
		props, _ := s["properties"].(map[string]any)
		if req, ok := s["required"].([]any); ok {
			for _, r := range req {
				name, _ := r.(string)
				if _, present := val[name]; !present {
					return fmt.Errorf("%s: missing required property %q", path, name)
				}
			}
		}
		for name, pv := range val {
			if ps, ok := props[name]; ok {
				if err := validateValue(ps, pv, path+"."+name); err != nil {
					return err
				}
				continue
			}
			switch ap := s["additionalProperties"].(type) {
			case bool:
				if !ap {
					return fmt.Errorf("%s: unknown property %q", path, name)
				}
			case map[string]any:
				if err := validateValue(ap, pv, path+"."+name); err != nil {
					return err
				}
			}
		}
	case []any:
		if items, ok := s["items"]; ok {
			for i, item := range val {
				if err := validateValue(items, item, fmt.Sprintf("%s[%d]", path, i)); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func schemaTypes(t any) []string {
	switch tv := t.(type) {
	case string:
		return []string{tv}
	case []any:
		var out []string
		for _, x := range tv {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

func matchesType(t string, v any) bool {
	switch t {
	case "integer":
		n, ok := v.(json.Number)
		if !ok {
			return false
		}
		f, err := n.Float64()
		return err == nil && f == math.Trunc(f)
	case "number":
		return jsonType(v) == "number"
	default:
		return jsonType(v) == t
	}
}

func jsonType(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case json.Number, float64:
		return "number"
	case string:
		return "string"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	}
	return "unknown"
}
