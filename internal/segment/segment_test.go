package segment

import (
	"reflect"
	"strings"
	"testing"
)

func collect(chunks ...string) []string {
	s := New()
	var out []string
	for _, c := range chunks {
		out = append(out, s.Push(c)...)
	}
	return append(out, s.Flush()...)
}

func TestSplitter(t *testing.T) {
	cases := []struct {
		in   []string
		want []string
	}{
		{[]string{"Hello there, friend. How are you today? ", "I am fine."}, []string{"Hello there, friend.", "How are you today?", "I am fine."}},
		{[]string{"Pi is 3.14 roughly, see example.com for more. Done now."}, []string{"Pi is 3.14 roughly, see example.com for more.", "Done now."}},
		{[]string{"Hi. ", "Ok. ", "This is a longer sentence. "}, []string{"Hi. Ok. This is a longer sentence."}},
		{[]string{"Ask Dr. Smith about the results. Thanks a lot!"}, []string{"Ask Dr. Smith about the results.", "Thanks a lot!"}},
		{[]string{"Hel", "lo wor", "ld, how is it going", "? Fine"}, []string{"Hello world, how is it going?", "Fine"}},
		{[]string{"He said \"stop right now.\" Then he left."}, []string{"He said \"stop right now.\"", "Then he left."}},
		{[]string{"First line here\nSecond line here"}, []string{"First line here", "Second line here"}},
		{[]string{"", "   "}, nil},
	}
	for _, c := range cases {
		if got := collect(c.in...); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%q:\n got  %q\n want %q", c.in, got, c.want)
		}
	}
}

func TestSplitterMaxLen(t *testing.T) {
	long := strings.Repeat("word ", 200)
	got := collect(long)
	if len(got) < 3 {
		t.Fatalf("expected forced splits, got %d chunks", len(got))
	}
	for _, g := range got {
		if len(g) > 280 {
			t.Fatalf("chunk too long: %d", len(g))
		}
	}
	if strings.Join(got, " ") != strings.TrimSpace(long) {
		t.Fatal("forced splitting lost text")
	}
	// Multi-byte text without any break point must still split on rune boundaries.
	for _, g := range collect(strings.Repeat("é", 400)) {
		if strings.ContainsRune(g, '\uFFFD') {
			t.Fatal("split inside a rune")
		}
	}
}

func TestTruncateWords(t *testing.T) {
	text := "one two three four"
	for f, want := range map[float64]string{0: "", 0.5: "one two", 1: text, 0.26: "one", 2: text} {
		if got := TruncateWords(text, f); got != want {
			t.Errorf("fraction %v: got %q want %q", f, got, want)
		}
	}
}
