package chunk

import (
	"reflect"
	"strings"
	"testing"
)

func TestMarkdownSplitsHeadingAwareSections(t *testing.T) {
	t.Parallel()

	sections := Markdown("Spec Title", `
Preamble paragraph.

## Requirements

- First requirement.

### Burst Handling

More detail.

## Rollout

Ship it.
`)

	want := []Section{
		{
			Heading: "Spec Title",
			Body:    "Preamble paragraph.",
		},
		{
			Heading: "Requirements",
			Body:    "- First requirement.",
		},
		{
			Heading: "Requirements / Burst Handling",
			Body:    "More detail.",
		},
		{
			Heading: "Rollout",
			Body:    "Ship it.",
		},
	}

	if !reflect.DeepEqual(sections, want) {
		t.Fatalf("Markdown() = %#v, want %#v", sections, want)
	}
}

func TestMarkdownUsesTitleWhenBodyHasNoHeadings(t *testing.T) {
	t.Parallel()

	sections := Markdown("Spec Title", "Body text without headings.\n")
	want := []Section{{
		Heading: "Spec Title",
		Body:    "Body text without headings.",
	}}
	if !reflect.DeepEqual(sections, want) {
		t.Fatalf("Markdown() = %#v, want %#v", sections, want)
	}
}

func TestMarkdownSkipsEmptyHeadingBlocks(t *testing.T) {
	t.Parallel()

	sections := Markdown("Doc Title", `
# Doc Title

## Empty

## Filled

Actual content.
`)

	want := []Section{{
		Heading: "Doc Title / Filled",
		Body:    "Actual content.",
	}}
	if !reflect.DeepEqual(sections, want) {
		t.Fatalf("Markdown() = %#v, want %#v", sections, want)
	}
}

func TestMarkdownWithOptionsZeroValueMatchesMarkdown(t *testing.T) {
	t.Parallel()

	body := "# Overview\n\nShort text.\n\n## Details\n\nMore detail."
	base := Markdown("Title", body)
	opts := MarkdownWithOptions("Title", body, Options{})
	if !reflect.DeepEqual(base, opts) {
		t.Fatalf("MarkdownWithOptions(zero) = %#v, want %#v", opts, base)
	}
}

func TestMarkdownWithOptionsSplitsLongSection(t *testing.T) {
	t.Parallel()

	// Build a section with 20 words under one heading.
	words := make([]string, 20)
	for i := range words {
		words[i] = "word"
	}
	body := "## Section\n\n" + joinWords(words)

	sections := MarkdownWithOptions("Title", body, Options{MaxTokens: 8})
	if len(sections) < 3 {
		t.Fatalf("expected at least 3 sub-sections, got %d: %#v", len(sections), sections)
	}
	for _, s := range sections {
		if s.Heading != "Section" {
			t.Fatalf("sub-section heading = %q, want Section", s.Heading)
		}
		wc := len(splitFields(s.Body))
		if wc > 8 {
			t.Fatalf("sub-section has %d words, want <= 8", wc)
		}
	}
}

func TestMarkdownWithOptionsOverlap(t *testing.T) {
	t.Parallel()

	words := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
	body := "## H\n\n" + joinWords(words)

	sections := MarkdownWithOptions("T", body, Options{MaxTokens: 5, OverlapTokens: 2})
	if len(sections) < 2 {
		t.Fatalf("expected at least 2 sections, got %d", len(sections))
	}
	// Second section should start with overlap from first.
	firstWords := splitFields(sections[0].Body)
	secondWords := splitFields(sections[1].Body)
	lastTwo := firstWords[len(firstWords)-2:]
	firstTwo := secondWords[:2]
	if lastTwo[0] != firstTwo[0] || lastTwo[1] != firstTwo[1] {
		t.Fatalf("overlap mismatch: first ends with %v, second starts with %v", lastTwo, firstTwo)
	}
}

func TestMarkdownWithOptionsPreservesShortSections(t *testing.T) {
	t.Parallel()

	body := "## A\n\nShort.\n\n## B\n\nAlso short."
	sections := MarkdownWithOptions("T", body, Options{MaxTokens: 100})
	if len(sections) != 2 {
		t.Fatalf("expected 2 sections, got %d", len(sections))
	}
}

func joinWords(words []string) string {
	return strings.Join(words, " ")
}

func splitFields(s string) []string {
	return strings.Fields(s)
}

func TestSplitSectionZeroMaxTokensReturnsOriginal(t *testing.T) {
	t.Parallel()

	s := Section{Heading: "H", Body: "one two three four five"}
	result := SplitSection(s, 0, 0)
	if len(result) != 1 || result[0] != s {
		t.Fatalf("SplitSection(maxTokens=0) = %#v, want original", result)
	}
}

func TestSplitSectionNegativeMaxTokensReturnsOriginal(t *testing.T) {
	t.Parallel()

	s := Section{Heading: "H", Body: "one two three"}
	result := SplitSection(s, -5, 0)
	if len(result) != 1 || result[0] != s {
		t.Fatalf("SplitSection(maxTokens=-5) = %#v, want original", result)
	}
}

func TestSplitSectionOverlapGEMaxTokensReturnsOriginal(t *testing.T) {
	t.Parallel()

	s := Section{Heading: "H", Body: "a b c d e f g h i j"}
	// overlap >= maxTokens is treated as invalid and should return the section unchanged.
	result := SplitSection(s, 3, 3)
	if len(result) != 1 || result[0] != s {
		t.Fatalf("SplitSection(maxTokens=3, overlap=3) = %#v, want original", result)
	}
}

func TestSplitSectionNegativeOverlapClampsToZero(t *testing.T) {
	t.Parallel()

	s := Section{Heading: "H", Body: "a b c d e f g h i j"}
	// Negative overlap must not cause step > maxTokens (which would skip words).
	result := SplitSection(s, 3, -5)
	// Collect all words from all chunks.
	seen := map[string]bool{}
	for _, r := range result {
		for _, w := range splitFields(r.Body) {
			seen[w] = true
		}
	}
	for _, w := range []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"} {
		if !seen[w] {
			t.Fatalf("word %q missing from chunks — negative overlap caused word drop", w)
		}
	}
}

func TestSplitSectionHandlesNBSP(t *testing.T) {
	t.Parallel()

	// NBSP (\u00a0) between words — must be treated as whitespace like strings.Fields does.
	body := "alpha\u00a0bravo\u00a0charlie\u00a0delta\u00a0echo\u00a0foxtrot"
	s := Section{Heading: "H", Body: body}
	result := SplitSection(s, 3, 0)
	if len(result) < 2 {
		t.Fatalf("expected at least 2 chunks with NBSP separators, got %d", len(result))
	}
	for _, r := range result {
		wc := len(splitFields(r.Body))
		if wc > 3 {
			t.Fatalf("chunk has %d words, want <= 3: %q", wc, r.Body)
		}
	}
}

func TestSplitSectionPreservesSmallSection(t *testing.T) {
	t.Parallel()

	s := Section{Heading: "H", Body: "one two three"}
	result := SplitSection(s, 10, 0)
	if len(result) != 1 || result[0] != s {
		t.Fatalf("SplitSection small = %#v, want original", result)
	}
}

func TestSplitSectionPreservesWhitespace(t *testing.T) {
	t.Parallel()

	body := "line one\n\n- bullet a\n- bullet b\n\n  indented text\n\nfinal word extra words padding here"
	s := Section{Heading: "H", Body: body}
	result := SplitSection(s, 5, 0)
	if len(result) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d", len(result))
	}
	// First chunk should preserve the newlines and list markers.
	if !strings.Contains(result[0].Body, "\n\n- bullet") {
		t.Fatalf("first chunk lost paragraph structure: %q", result[0].Body)
	}
}
