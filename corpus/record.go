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

// Normalized returns a trimmed, validated record with safe defaults applied.
func (r Record) Normalized() (Record, error) {
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
func HashRecord(r Record) string {
	parts := []string{
		"kind=" + strings.TrimSpace(r.Kind),
		"title=" + strings.TrimSpace(r.Title),
		"body_format=" + strings.TrimSpace(r.BodyFormat),
		"body_text=" + strings.TrimSpace(r.BodyText),
	}
	keys := make([]string, 0, len(r.Metadata))
	for key := range r.Metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		parts = append(parts, "metadata."+key+"="+r.Metadata[key])
	}
	return fingerprint(parts)
}

// Fingerprint summarizes a set of records deterministically.
func Fingerprint(records []Record) string {
	parts := make([]string, 0, len(records))
	for _, record := range records {
		normalized, err := record.Normalized()
		if err != nil {
			continue
		}
		parts = append(parts, normalized.Ref+":"+normalized.ContentHash)
	}
	return fingerprint(parts)
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
