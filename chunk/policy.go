package chunk

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/dusk-network/stroma/v2/corpus"
)

// NoParent is the sentinel used by SectionWithLineage.ParentIndex to
// mark a chunk as a root (a flat top-level chunk or the top of a
// hierarchy). Any non-negative value is interpreted as the slice index
// of another SectionWithLineage in the same Policy result.
const NoParent = -1

// SectionWithLineage decorates a Section with optional parent linkage
// inside a Policy's returned slice. ParentIndex is either NoParent
// (root) or the slice index of another SectionWithLineage. Forward
// references — a chunk pointing at an index later in the slice — are
// rejected when the index session validates topology, so the resulting
// FK chain is always acyclic and parents always precede their leaves
// at insert time.
type SectionWithLineage struct {
	Section
	ParentIndex int
}

// Policy decides how a record's body becomes chunks. The default
// MarkdownPolicy reproduces the pre-1.0 chunking pipeline exactly
// (heading-aware Markdown sectioning followed by optional token-budget
// splitting), so callers that do not opt in see no behavior change.
//
// A Policy may emit hierarchical chunks (parent + leaves with
// ParentIndex set) so consumers like Hippocampus and Pituitary can
// retrieve a small leaf and walk back to a broader parent span via
// Snapshot.ExpandContext (#16). Substrate-neutrality is preserved by
// keeping the contract narrow: in goes a corpus.Record, out comes a
// flat slice of (Section, ParentIndex) pairs. Policies have no
// awareness of indexing, embeddings, or storage — those concerns
// remain in the index package.
//
// Implementations must be safe for concurrent use: the index session
// invokes Chunk on one record at a time today, but library callers
// that drive chunking themselves (tests, offline pipelines) should be
// able to fan out across goroutines without hidden shared state. The
// shipped Policy types (MarkdownPolicy, KindRouterPolicy,
// LateChunkPolicy) are immutable post-construction and derive all
// mutable state from the per-call record.
type Policy interface {
	Chunk(ctx context.Context, record corpus.Record) ([]SectionWithLineage, error)
}

// MarkdownPolicy is the default Policy. It reproduces the pre-1.0
// chunking pipeline exactly: heading-aware Markdown sectioning followed
// by optional token-budget splitting. Every returned section has
// ParentIndex == NoParent — the policy emits flat chunks only — so a
// snapshot built with a MarkdownPolicy is byte-equivalent to today's
// output, with parent_chunk_id NULL on every row.
//
// Plaintext records (BodyFormat corpus.FormatPlaintext) bypass Markdown
// parsing and become a single section keyed on the record title;
// MaxTokens still drives token-budget splitting if positive.
type MarkdownPolicy struct {
	Options Options
}

// Chunk implements Policy.
func (p MarkdownPolicy) Chunk(_ context.Context, record corpus.Record) ([]SectionWithLineage, error) {
	sections, err := chunkRecord(record, p.Options)
	if err != nil {
		return nil, err
	}
	out := make([]SectionWithLineage, len(sections))
	for i, s := range sections {
		out[i] = SectionWithLineage{Section: s, ParentIndex: NoParent}
	}
	return out, nil
}

// KindRouterPolicy dispatches chunking to a per-Kind Policy, falling
// back to Default when no specific policy is registered for the
// record's Kind. Use it when one corpus mixes record kinds with
// different optimal chunking shapes (e.g., reference docs with
// LateChunkPolicy, raw notes with MarkdownPolicy).
//
// A nil Default with a Kind miss returns an error rather than silently
// producing zero sections; the caller almost always wants to know.
type KindRouterPolicy struct {
	Default Policy
	ByKind  map[string]Policy
}

// Chunk implements Policy.
func (k KindRouterPolicy) Chunk(ctx context.Context, record corpus.Record) ([]SectionWithLineage, error) {
	if p, ok := k.ByKind[record.Kind]; ok && p != nil {
		return p.Chunk(ctx, record)
	}
	if k.Default == nil {
		return nil, fmt.Errorf("chunk: KindRouterPolicy has no policy for kind %q and no Default", record.Kind)
	}
	return k.Default.Chunk(ctx, record)
}

// LateChunkPolicy is a reference implementation of the late-chunking
// pattern from the research signal in #16 (Late Chunking, hierarchical
// text segmentation). It produces a parent span per heading-aware
// section, then token-budget-split children that point at the parent.
//
// Concretely, for each heading-aware section the policy emits:
//
//   - one parent SectionWithLineage holding the full section body, with
//     ParentIndex == NoParent. Parents are not embedded by the index
//     layer (Q decision in the design spec); they live as
//     storage-only context that ExpandContext can surface on
//     IncludeParent.
//   - N child SectionWithLineage rows, produced by SplitSection on the
//     parent body with (ChildMaxTokens, ChildOverlapTokens), each
//     carrying ParentIndex pointing at the parent's slice index.
//     Children are embedded and participate in retrieval as usual.
//
// ParentMaxTokens caps the parent span itself: if a heading-aware
// section exceeds it, that section is split into multiple parents
// (each with its own children). Zero means "one parent per
// heading-aware section, regardless of size."
//
// MaxSections caps the total chunk count (parents + leaves combined)
// per record, mirroring chunk.Options.MaxSections semantics. Zero
// means no cap.
type LateChunkPolicy struct {
	ParentMaxTokens    int
	ChildMaxTokens     int
	ChildOverlapTokens int
	MaxSections        int
}

// errLateChunkChildMaxTokens is returned when LateChunkPolicy is used
// without ChildMaxTokens; without it the policy degenerates to
// "everything is a parent and there are no leaves," which silently
// disables retrieval and is almost certainly a configuration bug.
var errLateChunkChildMaxTokens = errors.New("chunk: LateChunkPolicy.ChildMaxTokens must be > 0")

// Chunk implements Policy.
func (p LateChunkPolicy) Chunk(_ context.Context, record corpus.Record) ([]SectionWithLineage, error) {
	if p.ChildMaxTokens <= 0 {
		return nil, errLateChunkChildMaxTokens
	}
	parents, err := chunkRecord(record, Options{
		MaxTokens:     p.ParentMaxTokens,
		OverlapTokens: 0,
		MaxSections:   p.MaxSections, // applied per-pass; combined cap is rechecked below
	})
	if err != nil {
		return nil, err
	}
	if len(parents) == 0 {
		return nil, nil
	}
	out := make([]SectionWithLineage, 0, len(parents)*2)
	for _, parent := range parents {
		parentIdx := len(out)
		out = append(out, SectionWithLineage{Section: parent, ParentIndex: NoParent})
		if p.MaxSections > 0 && len(out) > p.MaxSections {
			return nil, fmt.Errorf("%w: late-chunk parents exceeded %d sections on record %q",
				ErrTooManySections, p.MaxSections, record.Ref)
		}
		children := SplitSection(parent, p.ChildMaxTokens, p.ChildOverlapTokens)
		// SplitSection returns a single-element slice when the section
		// already fits ChildMaxTokens. In that case emitting a child
		// with identical body would duplicate the parent's content —
		// wasting storage and making ExpandContext(IncludeParent) echo
		// the same text twice. Skip child emission instead. The parent
		// stays in the slice with no children pointing at it, so the
		// index layer's identifyParents excludes it from the
		// parent-only set and treats it as a normal flat chunk: it
		// gets embedded and indexed in FTS, so the span remains
		// retrievable.
		if len(children) <= 1 {
			continue
		}
		for _, child := range children {
			out = append(out, SectionWithLineage{
				Section:     child,
				ParentIndex: parentIdx,
			})
			if p.MaxSections > 0 && len(out) > p.MaxSections {
				return nil, fmt.Errorf("%w: late-chunk parent+leaves exceeded %d sections on record %q",
					ErrTooManySections, p.MaxSections, record.Ref)
			}
		}
	}
	return out, nil
}

// chunkRecord runs the heading-aware (or plaintext) sectioning pass
// shared by MarkdownPolicy and LateChunkPolicy. It mirrors what the
// pre-1.0 index session did inline in sectionsForRecord, hoisted here
// so any Policy implementation can reuse it without depending on the
// index package.
func chunkRecord(record corpus.Record, opts Options) ([]Section, error) {
	switch record.BodyFormat {
	case corpus.FormatPlaintext:
		text := strings.TrimSpace(record.BodyText)
		if text == "" {
			return nil, nil
		}
		sections := []Section{{
			Heading: record.Title,
			Body:    text,
		}}
		if opts.MaxTokens > 0 {
			var split []Section
			for _, s := range sections {
				split = append(split, SplitSection(s, opts.MaxTokens, opts.OverlapTokens)...)
			}
			sections = split
		}
		if opts.MaxSections > 0 && len(sections) > opts.MaxSections {
			return nil, fmt.Errorf("record %q: %w: plaintext token-budget split produced %d sections, limit is %d",
				record.Ref, ErrTooManySections, len(sections), opts.MaxSections)
		}
		return sections, nil
	default:
		// Always route through MarkdownWithOptions so the parser-side
		// MaxSections guard fires during emission, not after the full
		// list is materialized.
		sections, err := MarkdownWithOptions(record.Title, record.BodyText, opts)
		if err != nil {
			return nil, fmt.Errorf("record %q: %w", record.Ref, err)
		}
		return sections, nil
	}
}
