// Package corpus defines the neutral record unit that Stroma indexes.
package corpus

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
)

const (
	// DefaultKind is used when callers omit a more specific record kind.
	DefaultKind = "artifact"

	// FormatMarkdown stores Markdown bodies that can be heading-chunked.
	FormatMarkdown = "markdown"

	// FormatPlaintext stores plain text bodies as a single chunk.
	FormatPlaintext = "plaintext"
)

// Record is the neutral corpus unit Stroma indexes.
type Record struct {
	Ref         string
	Kind        string
	Title       string
	SourceRef   string
	BodyFormat  string
	BodyText    string
	ContentHash string
	Metadata    map[string]string
}

// NewRecord constructs a Record for the common path: a reference, a
// human title, and a Markdown body. All other fields default through
// Normalize at the point the record enters a Rebuild / Update call.
// Callers with non-default kinds, custom metadata, or plaintext bodies
// should build the struct directly and pass it through Normalize.
func NewRecord(ref, title, body string) Record {
	return Record{
		Ref:        ref,
		Title:      title,
		BodyFormat: FormatMarkdown,
		BodyText:   body,
	}
}

// Normalize returns a trimmed, validated record with safe defaults
// applied: missing Kind defaults to DefaultKind, Title and SourceRef
// default to Ref, BodyFormat defaults to FormatMarkdown, and
// ContentHash is regenerated from HashRecord when empty. The returned
// record satisfies Validate by construction.
//
// This is the entry point callers should use before invoking Validate.
// Calling Validate on a raw Record will reject records that would have
// been accepted after defaults — for example Record{Ref:"x"} fails
// Validate (no kind) but Normalize fills in DefaultKind and succeeds.
func (r Record) Normalize() (Record, error) {
	normalized := Record{
		Ref:         strings.TrimSpace(r.Ref),
		Kind:        strings.TrimSpace(r.Kind),
		Title:       strings.TrimSpace(r.Title),
		SourceRef:   strings.TrimSpace(r.SourceRef),
		BodyFormat:  strings.TrimSpace(r.BodyFormat),
		BodyText:    strings.TrimSpace(r.BodyText),
		ContentHash: strings.TrimSpace(r.ContentHash),
		Metadata:    normalizeMetadata(r.Metadata),
	}
	if normalized.Ref == "" {
		return Record{}, fmt.Errorf("record ref is required")
	}
	if normalized.Kind == "" {
		normalized.Kind = DefaultKind
	}
	if normalized.Title == "" {
		normalized.Title = normalized.Ref
	}
	if normalized.SourceRef == "" {
		normalized.SourceRef = normalized.Ref
	}
	if normalized.BodyFormat == "" {
		normalized.BodyFormat = FormatMarkdown
	}
	if normalized.ContentHash == "" {
		normalized.ContentHash = HashRecord(normalized)
	}
	if err := normalized.Validate(); err != nil {
		return Record{}, err
	}
	return normalized, nil
}

// Validate reports whether the record is complete enough to persist.
// Callers constructing a Record by hand should prefer Normalize, which
// applies safe defaults before validation — calling Validate directly
// on a raw struct will reject records that Normalize would have
// accepted (e.g. a missing Kind).
func (r Record) Validate() error {
	switch {
	case strings.TrimSpace(r.Ref) == "":
		return fmt.Errorf("record ref is required")
	case strings.TrimSpace(r.Kind) == "":
		return fmt.Errorf("record %s kind is required", r.Ref)
	case strings.TrimSpace(r.SourceRef) == "":
		return fmt.Errorf("record %s source_ref is required", r.Ref)
	case strings.TrimSpace(r.ContentHash) == "":
		return fmt.Errorf("record %s content_hash is required", r.Ref)
	}
	switch strings.TrimSpace(r.BodyFormat) {
	case FormatMarkdown, FormatPlaintext:
		return nil
	default:
		return fmt.Errorf("record %s body_format %q is not supported", r.Ref, r.BodyFormat)
	}
}

// HashRecord returns a deterministic content hash for a normalized record.
//
// Each field contributes a `%q=%q` pair so the encoding is injective: no
// combination of key/value strings (including ones that contain `=` or
// newline characters) can produce the same serialized prefix as a different
// field. Serialized parts are then sorted and SHA-256-joined in fingerprint.
// Changing this encoding requires a schema_version bump and a migration
// that rewrites persisted content_hash values (see migrateV3ToV4).
func HashRecord(r Record) string {
	parts := make([]string, 0, 4+len(r.Metadata))
	parts = append(parts,
		fmt.Sprintf("%q=%q", "kind", strings.TrimSpace(r.Kind)),
		fmt.Sprintf("%q=%q", "title", strings.TrimSpace(r.Title)),
		fmt.Sprintf("%q=%q", "body_format", strings.TrimSpace(r.BodyFormat)),
		fmt.Sprintf("%q=%q", "body_text", strings.TrimSpace(r.BodyText)),
	)
	keys := make([]string, 0, len(r.Metadata))
	for key := range r.Metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%q=%q", "metadata."+key, r.Metadata[key]))
	}
	return fingerprint(parts)
}

// Fingerprint summarizes a set of records deterministically and returns an
// error if any record fails Normalize. Prior versions silently skipped
// invalid records, which meant "corpus with an invalid record" produced the
// same digest as "same corpus without that record" — masking silent reuse of
// a snapshot that was missing data the caller thought they had. The loud
// failure forces callers to surface the problem instead.
func Fingerprint(records []Record) (string, error) {
	parts := make([]string, 0, len(records))
	for _, record := range records {
		normalized, err := record.Normalize()
		if err != nil {
			return "", fmt.Errorf("fingerprint record %q: %w", record.Ref, err)
		}
		parts = append(parts, normalized.Ref+":"+normalized.ContentHash)
	}
	return fingerprint(parts), nil
}

// RefHash is the minimal (Ref, ContentHash) pair needed to compute a corpus
// fingerprint without materializing full record bodies.
type RefHash struct {
	Ref         string
	ContentHash string
}

// FingerprintFromPairs returns the same digest as Fingerprint([]Record) for
// already-normalized (Ref, ContentHash) inputs: non-empty after trimming,
// with ContentHash already computed. Pairs with an empty Ref or ContentHash
// return an error — unlike Fingerprint([]Record), this helper cannot apply
// Normalize() defaults or regenerate ContentHash via HashRecord from other
// fields, so its output only matches Fingerprint when the inputs already
// satisfy that invariant. Callers reading persisted rows must enforce the
// invariant at read time (as index.loadCurrentRefHashes does) or use
// Fingerprint instead.
func FingerprintFromPairs(pairs []RefHash) (string, error) {
	parts := make([]string, 0, len(pairs))
	for _, p := range pairs {
		ref := strings.TrimSpace(p.Ref)
		hash := strings.TrimSpace(p.ContentHash)
		if ref == "" || hash == "" {
			return "", fmt.Errorf("fingerprint pair has empty ref or content_hash (ref=%q)", p.Ref)
		}
		parts = append(parts, ref+":"+hash)
	}
	return fingerprint(parts), nil
}

func normalizeMetadata(metadata map[string]string) map[string]string {
	if len(metadata) == 0 {
		return nil
	}
	normalized := make(map[string]string, len(metadata))
	for key, value := range metadata {
		trimmedKey := strings.TrimSpace(key)
		if trimmedKey == "" {
			continue
		}
		normalized[trimmedKey] = strings.TrimSpace(value)
	}
	if len(normalized) == 0 {
		return nil
	}
	return normalized
}

func fingerprint(parts []string) string {
	sorted := append([]string(nil), parts...)
	sort.Strings(sorted)
	sum := sha256.Sum256([]byte(strings.Join(sorted, "\n")))
	return hex.EncodeToString(sum[:])
}
