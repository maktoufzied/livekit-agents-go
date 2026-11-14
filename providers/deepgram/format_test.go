package deepgram

import (
	"fmt"
	"strings"
	"testing"
)

func TestClientsNeverPrintTheKey(t *testing.T) {
	const key = "dg-secret-key-123"
	s, err := NewSTT(STTOptions{APIKey: key})
	if err != nil {
		t.Fatal(err)
	}
	v, err := NewTTS(TTSOptions{APIKey: key})
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range []any{s, *s, v, *v} {
		for _, f := range []string{"%v", "%+v", "%#v", "%s"} {
			if out := fmt.Sprintf(f, x); strings.Contains(out, key) {
				t.Errorf("%s of %T leaks the API key: %s", f, x, out)
			}
		}
	}
}
