// Package chunk splits text bodies into heading-aware sections for indexing.
package chunk

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Section is one heading-aware Markdown chunk.
type Section struct {
	Heading string
	Body    string
}

// Options controls how Markdown sections are split.
type Options struct {
	// MaxTokens is the approximate maximum number of tokens (words) per
	// section. Sections exceeding this limit are split into sub-sections.
	// Zero disables token-budget splitting.
	MaxTokens int

	// OverlapTokens is the approximate number of tokens to overlap between
	// adjacent sub-sections when a section is split. Zero disables overlap.
	OverlapTokens int
}

type heading struct {
	Level int
	Text  string
}

// Markdown splits Markdown into heading-aware sections.
func Markdown(title, body string) []Section {
	body = strings.ReplaceAll(body, "\r\n", "\n")
	lines := strings.Split(body, "\n")

	var (
		sections      []Section
		stack         []heading
		currentLines  []string
		currentHeader string
	)

	flush := func() {
		text := strings.TrimSpace(strings.Join(currentLines, "\n"))
		currentLines = nil
		if text == "" {
			return
		}

		headingText := strings.TrimSpace(currentHeader)
		if headingText == "" {
			headingText = strings.TrimSpace(title)
		}

		sections = append(sections, Section{
			Heading: headingText,
			Body:    text,
		})
	}

	for _, line := range lines {
		level, text, ok := parseHeading(line)
		if ok {
			flush()
			for len(stack) > 0 && stack[len(stack)-1].Level >= level {
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, heading{Level: level, Text: text})
			currentHeader = joinHeadings(stack)
			continue
		}

		currentLines = append(currentLines, line)
	}

	flush()

	if len(sections) == 0 {
		text := strings.TrimSpace(body)
		if text == "" {
			return nil
		}
		return []Section{{
			Heading: strings.TrimSpace(title),
			Body:    text,
		}}
	}

	return sections
}

func parseHeading(line string) (level int, text string, ok bool) {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return 0, "", false
	}

	for level < len(trimmed) && trimmed[level] == '#' {
		level++
	}
	if level == 0 || level > 6 {
		return 0, "", false
	}
	if len(trimmed) == level || trimmed[level] != ' ' {
		return 0, "", false
	}

	text = strings.TrimSpace(trimmed[level+1:])
	if text == "" {
		return 0, "", false
	}
	return level, text, true
}

func joinHeadings(stack []heading) string {
	parts := make([]string, 0, len(stack))
	for _, item := range stack {
		if item.Text == "" {
			continue
		}
		parts = append(parts, item.Text)
	}
	return strings.Join(parts, " / ")
}

// MarkdownWithOptions splits Markdown into heading-aware sections and then
// applies token-budget splitting when sections exceed opts.MaxTokens.
// Zero-value options produce the same output as Markdown.
func MarkdownWithOptions(title, body string, opts Options) []Section {
	sections := Markdown(title, body)
	if opts.MaxTokens <= 0 {
		return sections
	}

	result := make([]Section, 0, len(sections))
	for _, s := range sections {
		if countWords(s.Body) <= opts.MaxTokens {
			result = append(result, s)
			continue
		}
		result = append(result, SplitSection(s, opts.MaxTokens, opts.OverlapTokens)...)
	}
	return result
}

func countWords(text string) int {
	return len(wordSpans(text))
}

// SplitSection splits a single section into sub-sections that each contain
// at most maxTokens words, with overlapTokens words of overlap between them.
// The original text is preserved by slicing at word boundaries rather than
// reconstructing from tokenized words.
//
// If maxTokens is <= 0 or overlapTokens >= maxTokens the section is returned
// unchanged.
func SplitSection(s Section, maxTokens, overlapTokens int) []Section {
	if maxTokens <= 0 {
		return []Section{s}
	}
	if overlapTokens < 0 {
		overlapTokens = 0
	}
	if overlapTokens >= maxTokens {
		return []Section{s}
	}

	spans := wordSpans(s.Body)
	if len(spans) <= maxTokens {
		return []Section{s}
	}

	step := maxTokens - overlapTokens

	var result []Section
	for start := 0; start < len(spans); start += step {
		end := start + maxTokens
		if end > len(spans) {
			end = len(spans)
		}
		// For continuation chunks, extend lo back to the line start to
		// preserve Markdown prefixes (- , > , indentation), but never
		// earlier than the previous word's end to avoid pulling in extra words.
		minPos := 0
		if start > 0 {
			minPos = spans[start-1].end
		}
		lo := lineStart(s.Body, spans[start].start, minPos)
		hi := spans[end-1].end
		result = append(result, Section{
			Heading: s.Heading,
			Body:    strings.TrimSpace(s.Body[lo:hi]),
		})
		if end == len(spans) {
			break
		}
	}
	return result
}

type span struct {
	start, end int
}

// wordSpans returns the [start, end) byte offsets for each word in text,
// preserving the original positions so callers can slice the source string.
// It uses unicode.IsSpace so that NBSP and other Unicode whitespace are
// treated as separators, consistent with strings.Fields.
func wordSpans(text string) []span {
	var spans []span
	i := 0
	for i < len(text) {
		// Skip whitespace.
		for i < len(text) {
			r, size := utf8.DecodeRuneInString(text[i:])
			if !unicode.IsSpace(r) {
				break
			}
			i += size
		}
		if i >= len(text) {
			break
		}
		start := i
		// Advance through non-whitespace.
		for i < len(text) {
			r, size := utf8.DecodeRuneInString(text[i:])
			if unicode.IsSpace(r) {
				break
			}
			i += size
		}
		spans = append(spans, span{start, i})
	}
	return spans
}

// lineStart returns the byte offset of the beginning of the line containing
// pos, but never earlier than minPos. This preserves leading Markdown prefixes
// (list markers, blockquote >, indentation) when a chunk starts mid-line
// without pulling in words from a previous chunk.
func lineStart(text string, pos, minPos int) int {
	for pos > minPos && text[pos-1] != '\n' {
		pos--
	}
	return pos
}
