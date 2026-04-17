// Package chunk splits text bodies into heading-aware sections for indexing.
package chunk

import (
	"errors"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// ErrTooManySections is returned by MarkdownWithOptions when the input body
// produces more sections than Options.MaxSections allows. Enclosing errors
// wrap this sentinel via fmt.Errorf with %w so callers can use errors.Is
// to distinguish a pathological-input rejection from other failures (e.g.,
// to skip the record instead of aborting the whole rebuild).
var ErrTooManySections = errors.New("chunk: section count exceeds configured limit")

// Section is one heading-aware Markdown chunk.
type Section struct {
	Heading string
	Body    string
}

// Options controls how Markdown sections are split.
type Options struct {
	// MaxTokens is the approximate maximum number of tokens (words) per
	// section. Sections exceeding this limit are split into sub-sections
	// unless OverlapTokens is an invalid value that disables splitting.
	// Zero disables token-budget splitting.
	MaxTokens int

	// OverlapTokens is the approximate number of tokens to overlap between
	// adjacent sub-sections when a section is split. Zero disables overlap.
	// Values greater than or equal to MaxTokens are treated as invalid and
	// leave oversized sections unsplit.
	OverlapTokens int

	// MaxSections caps the number of heading-aware sections
	// MarkdownWithOptions will emit for a single body. Zero means no
	// cap (backward-compatible default for direct callers); any positive
	// value causes MarkdownWithOptions to return ErrTooManySections when
	// the body would produce more sections than the cap. Stroma's index
	// layer applies a conservative default (see index.DefaultMaxChunkSections)
	// so a pathological body can't DoS the embedder or balloon the snapshot.
	MaxSections int
}

type heading struct {
	Level int
	Text  string
}

// Markdown splits Markdown into heading-aware sections. No cap on section
// count — direct callers who need DoS protection should use
// MarkdownWithOptions with a positive MaxSections.
func Markdown(title, body string) []Section {
	// Unbounded path delegates to markdownBounded; the err branch is
	// unreachable with maxSections == 0.
	sections, _ := markdownBounded(title, body, 0)
	return sections
}

// markdownBounded is the shared parser for Markdown and MarkdownWithOptions.
// maxSections == 0 means unlimited; a positive value aborts section
// emission at section N+1 so a pathological body (e.g., 10^6 headings)
// can't force allocation of millions of Section entries before the cap
// is checked. Each flush appends at most one section, then tests the
// limit, so peak memory stays O(maxSections) rather than O(body size).
func markdownBounded(title, body string, maxSections int) ([]Section, error) {
	body = strings.ReplaceAll(body, "\r\n", "\n")
	lines := strings.Split(body, "\n")

	var (
		sections      []Section
		stack         []heading
		currentLines  []string
		currentHeader string
	)

	flush := func() error {
		text := strings.TrimSpace(strings.Join(currentLines, "\n"))
		currentLines = nil
		if text == "" {
			return nil
		}

		headingText := strings.TrimSpace(currentHeader)
		if headingText == "" {
			headingText = strings.TrimSpace(title)
		}

		sections = append(sections, Section{
			Heading: headingText,
			Body:    text,
		})
		if maxSections > 0 && len(sections) > maxSections {
			return fmt.Errorf("%w: heading-aware pass exceeded %d sections",
				ErrTooManySections, maxSections)
		}
		return nil
	}

	for _, line := range lines {
		level, text, ok := parseHeading(line)
		if ok {
			if err := flush(); err != nil {
				return nil, err
			}
			for len(stack) > 0 && stack[len(stack)-1].Level >= level {
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, heading{Level: level, Text: text})
			currentHeader = joinHeadings(stack)
			continue
		}

		currentLines = append(currentLines, line)
	}

	if err := flush(); err != nil {
		return nil, err
	}

	if len(sections) == 0 {
		text := strings.TrimSpace(body)
		if text == "" {
			return nil, nil
		}
		return []Section{{
			Heading: strings.TrimSpace(title),
			Body:    text,
		}}, nil
	}

	return sections, nil
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
// applies token-budget splitting when sections exceed opts.MaxTokens, unless
// opts.OverlapTokens is invalid for splitting. Zero-value options produce the
// same output as Markdown with a nil error.
//
// If opts.MaxSections is positive and either the heading-aware parse or the
// post-split pass would exceed that many sections, returns
// ErrTooManySections (wrapped with the observed count) and a nil slice.
// MaxSections is enforced inside the parser (see markdownBounded) so a
// pathological 10^6-heading body aborts after allocating one section
// past the cap rather than after materializing the full list. The
// token-budget pass then re-checks after each SplitSection call so
// one huge section split into many sub-sections can't amplify past the
// cap either — the residual risk is bounded by the fan-out of a single
// SplitSection invocation (~body_words / step).
func MarkdownWithOptions(title, body string, opts Options) ([]Section, error) {
	sections, err := markdownBounded(title, body, opts.MaxSections)
	if err != nil {
		return nil, err
	}
	if opts.MaxTokens <= 0 {
		return sections, nil
	}

	result := make([]Section, 0, len(sections))
	for _, s := range sections {
		if countWords(s.Body) <= opts.MaxTokens {
			result = append(result, s)
			if opts.MaxSections > 0 && len(result) > opts.MaxSections {
				return nil, fmt.Errorf("%w: token-budget pass exceeded %d sections",
					ErrTooManySections, opts.MaxSections)
			}
			continue
		}
		result = append(result, SplitSection(s, opts.MaxTokens, opts.OverlapTokens)...)
		if opts.MaxSections > 0 && len(result) > opts.MaxSections {
			return nil, fmt.Errorf("%w: token-budget split exceeded %d sections",
				ErrTooManySections, opts.MaxSections)
		}
	}
	return result, nil
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
		body := strings.TrimRightFunc(s.Body[lo:hi], unicode.IsSpace)
		if start > 0 && lo == minPos {
			body = strings.TrimLeftFunc(body, unicode.IsSpace)
		}
		result = append(result, Section{
			Heading: s.Heading,
			Body:    body,
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
