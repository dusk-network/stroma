package chunk

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/dusk-network/stroma/v2/corpus"
)

// TestMarkdownPolicyMatchesMarkdownWithOptions pins the default-behavior
// preservation contract: MarkdownPolicy with the same Options must
// produce exactly the same Section sequence as MarkdownWithOptions, and
// every returned section must have ParentIndex == NoParent. A snapshot
// built via the default ChunkPolicy is therefore byte-equivalent to a
// pre-1.0 snapshot built without policy plumbing, modulo the always-NULL
// parent_chunk_id column.
func TestMarkdownPolicyMatchesMarkdownWithOptions(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		title  string
		body   string
		opts   Options
		format string
	}{
		{
			name:   "plaintext_no_split",
			title:  "Plain",
			body:   "one two three",
			format: corpus.FormatPlaintext,
		},
		{
			name:   "plaintext_with_split",
			title:  "Plain",
			body:   "one two three four five six seven eight",
			opts:   Options{MaxTokens: 3, OverlapTokens: 1},
			format: corpus.FormatPlaintext,
		},
		{
			name:   "markdown_simple",
			title:  "Doc",
			body:   "# A\n\none\n\n# B\n\ntwo\n",
			format: corpus.FormatMarkdown,
		},
		{
			name:   "markdown_with_token_split",
			title:  "Doc",
			body:   "# A\n\none two three four five six seven eight nine ten\n\n# B\n\ntwo\n",
			opts:   Options{MaxTokens: 4, OverlapTokens: 1},
			format: corpus.FormatMarkdown,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rec := corpus.Record{
				Ref:        "rec",
				Kind:       "note",
				Title:      tc.title,
				BodyFormat: tc.format,
				BodyText:   tc.body,
			}
			policy := MarkdownPolicy{Options: tc.opts}
			lineaged, err := policy.Chunk(context.Background(), rec)
			if err != nil {
				t.Fatalf("MarkdownPolicy.Chunk() error = %v", err)
			}

			var want []Section
			switch tc.format {
			case corpus.FormatPlaintext:
				text := strings.TrimSpace(tc.body)
				if text == "" {
					want = nil
				} else {
					base := []Section{{Heading: tc.title, Body: text}}
					if tc.opts.MaxTokens > 0 {
						for _, s := range base {
							want = append(want, SplitSection(s, tc.opts.MaxTokens, tc.opts.OverlapTokens)...)
						}
					} else {
						want = base
					}
				}
			default:
				want, err = MarkdownWithOptions(tc.title, tc.body, tc.opts)
				if err != nil {
					t.Fatalf("MarkdownWithOptions() error = %v", err)
				}
			}

			if len(lineaged) != len(want) {
				t.Fatalf("got %d sections, want %d", len(lineaged), len(want))
			}
			for i, l := range lineaged {
				if l.ParentIndex != NoParent {
					t.Fatalf("section %d ParentIndex = %d, want NoParent (%d)", i, l.ParentIndex, NoParent)
				}
				if l.Heading != want[i].Heading {
					t.Fatalf("section %d Heading = %q, want %q", i, l.Heading, want[i].Heading)
				}
				if l.Body != want[i].Body {
					t.Fatalf("section %d Body = %q, want %q", i, l.Body, want[i].Body)
				}
			}
		})
	}
}

// recordingPolicy is a Policy that records which records it was called
// for, used to verify KindRouterPolicy dispatch semantics.
type recordingPolicy struct {
	name  string
	calls []string
}

func (r *recordingPolicy) Chunk(_ context.Context, record corpus.Record) ([]SectionWithLineage, error) {
	r.calls = append(r.calls, record.Ref)
	return []SectionWithLineage{{
		Section:     Section{Heading: r.name, Body: record.BodyText},
		ParentIndex: NoParent,
	}}, nil
}

func TestKindRouterPolicyDispatchesByKind(t *testing.T) {
	t.Parallel()

	def := &recordingPolicy{name: "default"}
	notePolicy := &recordingPolicy{name: "note"}
	specPolicy := &recordingPolicy{name: "spec"}

	router := KindRouterPolicy{
		Default: def,
		ByKind: map[string]Policy{
			"note": notePolicy,
			"spec": specPolicy,
		},
	}

	for _, rec := range []corpus.Record{
		{Ref: "n1", Kind: "note", BodyText: "n1"},
		{Ref: "s1", Kind: "spec", BodyText: "s1"},
		{Ref: "x1", Kind: "other", BodyText: "x1"},
		{Ref: "n2", Kind: "note", BodyText: "n2"},
	} {
		out, err := router.Chunk(context.Background(), rec)
		if err != nil {
			t.Fatalf("Chunk(%s) error = %v", rec.Ref, err)
		}
		if len(out) != 1 {
			t.Fatalf("Chunk(%s) returned %d sections, want 1", rec.Ref, len(out))
		}
	}

	if got := len(notePolicy.calls); got != 2 {
		t.Fatalf("note policy invoked %d times, want 2", got)
	}
	if got := len(specPolicy.calls); got != 1 {
		t.Fatalf("spec policy invoked %d times, want 1", got)
	}
	if got := len(def.calls); got != 1 || def.calls[0] != "x1" {
		t.Fatalf("default policy calls = %v, want exactly [x1]", def.calls)
	}
}

func TestKindRouterPolicyMissingDefaultErrors(t *testing.T) {
	t.Parallel()

	router := KindRouterPolicy{
		ByKind: map[string]Policy{
			"note": &recordingPolicy{name: "note"},
		},
	}
	_, err := router.Chunk(context.Background(), corpus.Record{Ref: "x", Kind: "spec"})
	if err == nil {
		t.Fatal("Chunk(unknown kind, no Default) succeeded; want error")
	}
	if !strings.Contains(err.Error(), "no policy for kind") {
		t.Fatalf("err = %v, want message about missing policy", err)
	}
}

func TestLateChunkPolicyEmitsParentWithLeaves(t *testing.T) {
	t.Parallel()

	body := "# Section A\n\n" + strings.Repeat("alpha bravo charlie delta echo foxtrot ", 10) + "\n"
	rec := corpus.Record{
		Ref:        "doc",
		Kind:       "note",
		Title:      "Doc",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   body,
	}
	policy := LateChunkPolicy{
		ChildMaxTokens:     8,
		ChildOverlapTokens: 2,
	}
	out, err := policy.Chunk(context.Background(), rec)
	if err != nil {
		t.Fatalf("LateChunkPolicy.Chunk() error = %v", err)
	}
	if len(out) < 2 {
		t.Fatalf("got %d sections, want >= 2 (parent + leaves)", len(out))
	}

	// First emission must be a parent (ParentIndex = NoParent) and the
	// remaining children point at index 0.
	if out[0].ParentIndex != NoParent {
		t.Fatalf("section 0 (parent) ParentIndex = %d, want NoParent", out[0].ParentIndex)
	}
	for i := 1; i < len(out); i++ {
		if out[i].ParentIndex != 0 {
			t.Fatalf("section %d ParentIndex = %d, want 0 (parent)", i, out[i].ParentIndex)
		}
	}
}

func TestLateChunkPolicyChildMaxTokensRequired(t *testing.T) {
	t.Parallel()
	policy := LateChunkPolicy{ChildMaxTokens: 0}
	_, err := policy.Chunk(context.Background(), corpus.Record{
		Ref:        "doc",
		Kind:       "note",
		Title:      "Doc",
		BodyFormat: corpus.FormatPlaintext,
		BodyText:   "anything",
	})
	if err == nil {
		t.Fatal("LateChunkPolicy with ChildMaxTokens=0 succeeded; want config error")
	}
	if !errors.Is(err, errLateChunkChildMaxTokens) {
		t.Fatalf("err = %v, want errors.Is errLateChunkChildMaxTokens", err)
	}
}

func TestLateChunkPolicyMaxSectionsCap(t *testing.T) {
	t.Parallel()

	// Manufacture a body with lots of headings + lots of words per
	// heading so parents+leaves easily exceed a low MaxSections cap.
	var b strings.Builder
	for i := 0; i < 30; i++ {
		b.WriteString("# H")
		b.WriteString(strings.Repeat("x", i+1))
		b.WriteString("\n\n")
		b.WriteString(strings.Repeat("alpha bravo ", 20))
		b.WriteString("\n\n")
	}
	rec := corpus.Record{
		Ref:        "doc",
		Kind:       "note",
		Title:      "Doc",
		BodyFormat: corpus.FormatMarkdown,
		BodyText:   b.String(),
	}
	policy := LateChunkPolicy{
		ChildMaxTokens: 5,
		MaxSections:    20,
	}
	_, err := policy.Chunk(context.Background(), rec)
	if err == nil {
		t.Fatal("LateChunkPolicy with low MaxSections succeeded on large body; want ErrTooManySections")
	}
	if !errors.Is(err, ErrTooManySections) {
		t.Fatalf("err = %v, want errors.Is ErrTooManySections", err)
	}
}
