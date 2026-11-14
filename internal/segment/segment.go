// Package segment splits streamed text into sentence-sized chunks for
// speech synthesis.
package segment

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Splitter incrementally splits text into sentences. It is not safe for
// concurrent use.
type Splitter struct {
	// MinLen is the minimum number of bytes a chunk should have before it
	// is emitted; shorter sentences are merged with the next one.
	MinLen int
	// MaxLen forces a split at a word boundary when no sentence end is
	// found, bounding synthesis latency for run-on text.
	MaxLen int
	buf    strings.Builder
}

// New returns a Splitter with defaults suited to speech synthesis.
func New() *Splitter { return &Splitter{MinLen: 12, MaxLen: 280} }

var abbreviations = map[string]bool{
	"mr": true, "mrs": true, "ms": true, "dr": true, "prof": true, "sr": true, "jr": true,
	"st": true, "vs": true, "etc": true, "e.g": true, "i.e": true, "no": true,
}

// Push appends text and returns the sentences completed by it.
func (s *Splitter) Push(text string) []string {
	s.buf.WriteString(text)
	var out []string
	for {
		cur := s.buf.String()
		cut := s.boundary(cur)
		if cut < 0 {
			return out
		}
		if chunk := strings.TrimSpace(cur[:cut]); chunk != "" {
			out = append(out, chunk)
		}
		rest := cur[cut:]
		s.buf.Reset()
		s.buf.WriteString(rest)
	}
}

// Flush returns the remaining text, if any, and resets the splitter.
func (s *Splitter) Flush() []string {
	rest := strings.TrimSpace(s.buf.String())
	s.buf.Reset()
	if rest == "" {
		return nil
	}
	return []string{rest}
}

// boundary returns the byte offset just after the first usable sentence end
// in cur, or -1.
func (s *Splitter) boundary(cur string) int {
	for i := 0; i < len(cur); {
		r, size := utf8.DecodeRuneInString(cur[i:])
		end := i + size
		switch {
		case r == '\n':
			if len(strings.TrimSpace(cur[:i])) >= s.MinLen {
				return end
			}
		case r == '。' || r == '！' || r == '？':
			if len(strings.TrimSpace(cur[:end])) >= s.MinLen {
				return end
			}
		case r == '.' || r == '!' || r == '?' || r == '…':
			// Swallow closing quotes/brackets and repeated punctuation.
			for end < len(cur) && strings.ContainsRune(`.!?…"')]`, rune(cur[end])) {
				end++
			}
			// A sentence end needs following whitespace: this keeps
			// "3.14" and "example.com" intact and waits for more text
			// when the punctuation is the last thing received so far.
			if end >= len(cur) {
				return -1
			}
			next, _ := utf8.DecodeRuneInString(cur[end:])
			if unicode.IsSpace(next) && len(strings.TrimSpace(cur[:end])) >= s.MinLen && !(r == '.' && endsWithAbbreviation(cur[:i])) {
				return end
			}
			i = end
			continue
		}
		i = end
	}
	if s.MaxLen > 0 && len(cur) > s.MaxLen {
		if j := strings.LastIndexAny(cur[:s.MaxLen], ",;: "); j > 0 {
			return j + 1
		}
		// No break point: cut on a rune boundary.
		j := s.MaxLen
		for j > 0 && !utf8.RuneStart(cur[j]) {
			j--
		}
		return j
	}
	return -1
}

func endsWithAbbreviation(before string) bool {
	j := strings.LastIndexFunc(before, unicode.IsSpace)
	word := strings.ToLower(strings.Trim(before[j+1:], `"'([`))
	return abbreviations[word] || (len(word) == 1 && unicode.IsLetter(rune(word[0])))
}

// TruncateWords returns the leading fraction of text, cut at a word
// boundary. fraction is clamped to [0, 1].
func TruncateWords(text string, fraction float64) string {
	if fraction >= 1 {
		return text
	}
	words := strings.Fields(text)
	if fraction <= 0 || len(words) == 0 {
		return ""
	}
	n := int(float64(len(words))*fraction + 0.5)
	if n >= len(words) {
		return text
	}
	return strings.Join(words[:n], " ")
}
