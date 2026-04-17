package index

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/dusk-network/stroma/chunk"
	"github.com/dusk-network/stroma/corpus"
	"github.com/dusk-network/stroma/store"
)

// reuseState holds a live read-only handle to the prior snapshot. Record and
// chunk rows are fetched on demand by storedRecordForReuse, so resident memory
// scales with one record's chunks instead of the whole corpus.
//
// quantization is captured at open time so the per-record query knows whether
// to read embeddings from chunks_vec (float32/int8) or chunks_vec_full
// (binary, where the primary chunks_vec row is a sign-packed bit blob that
// would need rederivation on reuse).
type reuseState struct {
	db           *sql.DB
	quantization string
	// hasContextPrefix is derived from the accept-listed schema_version at
	// reuse-open time so per-record chunk queries do not reprobe PRAGMA
	// table_info(chunks) — which used to fire once per record during a
	// rebuild with reuse enabled.
	hasContextPrefix bool
}

type storedRecord struct {
	contentHash string
	title       string
	chunks      map[string][]byte
}

type recordReusePlan struct {
	sections []chunk.Section
	// prefixes[i] is the context prefix aligned with sections[i]. nil or
	// empty string means no contextualization for that section.
	prefixes []string
	// keys[i] is reuseChunkKey for sections[i]. Computed once in
	// planRecordReuse so the write path does not re-hash each section.
	keys               []string
	reusedEmbeddings   map[string][]byte
	reusedChunkCount   int
	embeddedChunkCount int
	recordUnchanged    bool
}

func loadReuseState(ctx context.Context, path, embedderFingerprint string, embedderDimension int, quantization string) *reuseState {
	empty := &reuseState{}
	path = strings.TrimSpace(path)
	if path == "" {
		return empty
	}

	info, err := os.Stat(path)
	switch {
	case os.IsNotExist(err):
		return empty
	case err != nil:
		return empty
	case info.IsDir():
		return empty
	}

	db, err := store.OpenReadOnlyContext(ctx, path)
	if err != nil {
		return empty
	}

	compatible, hasPrefix := isCompatibleReuseSnapshot(ctx, db, embedderFingerprint, embedderDimension, quantization)
	if !compatible {
		_ = db.Close()
		return empty
	}

	normQ, err := normalizeQuantization(quantization)
	if err != nil {
		_ = db.Close()
		return empty
	}
	return &reuseState{db: db, quantization: normQ, hasContextPrefix: hasPrefix}
}

// Close releases the underlying read-only connection. Nil-safe and idempotent.
func (s *reuseState) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	err := s.db.Close()
	s.db = nil
	return err
}

// isCompatibleReuseSnapshot reports whether the prior snapshot can seed the
// reuse cache, and — when compatible — whether it carries the v3
// chunks.context_prefix column. Returning the column flag here avoids a
// second schema_version round-trip in loadReuseState and lets per-record
// chunk queries skip PRAGMA table_info(chunks).
func isCompatibleReuseSnapshot(ctx context.Context, db *sql.DB, embedderFingerprint string, embedderDimension int, quantization string) (compatible, hasContextPrefix bool) {
	schema, err := readMetadataValue(ctx, db, "schema_version")
	if err != nil {
		return false, false
	}
	// v2 snapshots pre-date context_prefix. They remain reuse-compatible
	// because loadStoredChunksForRecord defaults the missing prefix to
	// "" and reuseChunkKey with prefix="" matches what the new-side key
	// produces when no Contextualizer is configured. Contextualizer-
	// enabled rebuilds produce different new-side keys, so v2 chunks
	// simply miss and get re-embedded. Record-level reuse (content_hash
	// comparison) won't fire against a real pre-1.0 v2 file because its
	// stored hashes are under the old encoding and the new-side
	// HashRecord uses the new encoding; chunk-level reuse still picks up
	// every unchanged chunk.
	trimmed := strings.TrimSpace(schema)
	if trimmed != schemaVersion && trimmed != prevSchemaVersion && trimmed != legacySchemaVersionV2 {
		return false, false
	}
	storedFingerprint, err := readMetadataValue(ctx, db, "embedder_fingerprint")
	if err != nil || strings.TrimSpace(storedFingerprint) != strings.TrimSpace(embedderFingerprint) {
		return false, false
	}
	storedDimension, err := readMetadataValue(ctx, db, "embedder_dimension")
	if err != nil || strings.TrimSpace(storedDimension) != strconv.Itoa(embedderDimension) {
		return false, false
	}
	storedQuantization, err := readMetadataValueOptional(ctx, db, "quantization", store.QuantizationFloat32)
	if err != nil {
		return false, false
	}
	normStored, err1 := normalizeQuantization(storedQuantization)
	normTarget, err2 := normalizeQuantization(quantization)
	if err1 != nil || err2 != nil || normStored != normTarget {
		return false, false
	}
	return true, schemaHasContextPrefix(trimmed)
}

// planRecordReuse builds a per-record plan given the resolved sections and
// aligned context prefixes. prefixes must be nil or the same length as
// sections; a nil slice means "no contextualizer configured" and is
// equivalent to an empty prefix per section.
func planRecordReuse(record corpus.Record, stored storedRecord, sections []chunk.Section, prefixes []string) recordReusePlan {
	keys := make([]string, len(sections))
	for i, section := range sections {
		prefix := ""
		if i < len(prefixes) {
			prefix = prefixes[i]
		}
		keys[i] = reuseChunkKey(record.Title, section.Heading, section.Body, prefix)
	}
	plan := recordReusePlan{
		sections:         sections,
		prefixes:         prefixes,
		keys:             keys,
		reusedEmbeddings: make(map[string][]byte),
	}
	if stored.contentHash == "" {
		plan.embeddedChunkCount = len(sections)
		return plan
	}

	for _, key := range keys {
		if embedding, ok := stored.chunks[key]; ok {
			plan.reusedEmbeddings[key] = embedding
			plan.reusedChunkCount++
		}
	}
	plan.embeddedChunkCount = len(sections) - plan.reusedChunkCount
	plan.recordUnchanged = stored.contentHash == record.ContentHash && len(sections) == plan.reusedChunkCount
	return plan
}

// storedRecordForReuse fetches the stored record's content hash and its chunk
// embeddings for ref, on demand. A clean "no row" fall-through still misses a
// reuse opportunity without blocking the write path; any other read error
// disables reuse for the remainder of this session so embedder cost stays
// predictable instead of producing a half-reused build under transient
// SQLITE_BUSY or corruption.
func storedRecordForReuse(ctx context.Context, state *reuseState, ref string) storedRecord {
	if state == nil || state.db == nil {
		return storedRecord{}
	}

	var (
		title       string
		contentHash string
	)
	row := state.db.QueryRowContext(ctx, `SELECT title, content_hash FROM records WHERE ref = ?`, ref)
	if err := row.Scan(&title, &contentHash); err != nil {
		if !errors.Is(err, sql.ErrNoRows) {
			state.disable()
		}
		return storedRecord{}
	}

	chunks, err := loadStoredChunksForRecord(ctx, state.db, ref, title, state.quantization, state.hasContextPrefix)
	if err != nil {
		state.disable()
		return storedRecord{}
	}
	return storedRecord{
		contentHash: contentHash,
		title:       title,
		chunks:      chunks,
	}
}

// disable closes the read-only handle and nils it out so subsequent reuse
// lookups short-circuit to an empty storedRecord. Idempotent.
func (s *reuseState) disable() {
	if s == nil || s.db == nil {
		return
	}
	_ = s.db.Close()
	s.db = nil
}

func loadStoredChunksForRecord(ctx context.Context, db *sql.DB, ref, title, quantization string, hasContextPrefix bool) (map[string][]byte, error) {
	// v2 snapshots lack context_prefix; project an empty string so the
	// scan shape is uniform and reuse keys computed with prefix="" stay
	// in sync with the new-side keys when no Contextualizer is configured.
	// hasContextPrefix comes from reuseState, which derives it from the
	// accept-listed schema_version read at open time — no PRAGMA probe.
	prefixExpr := "'' AS context_prefix"
	if hasContextPrefix {
		prefixExpr = "c.context_prefix"
	}
	// Binary mode stores the primary chunks_vec row as a sign-packed bit
	// blob and keeps the float32 source of truth in chunks_vec_full. Reuse
	// loads the full-precision blob so the write path can re-derive the
	// bit blob (via sign packing) without the lossy bit → float fallback.
	embeddingSource := "v.embedding"
	joinClause := "JOIN chunks_vec v ON v.chunk_id = c.id"
	if quantization == store.QuantizationBinary {
		embeddingSource = "vf.embedding"
		joinClause = "JOIN chunks_vec_full vf ON vf.chunk_id = c.id"
	}
	rows, err := db.QueryContext(ctx, fmt.Sprintf(`
SELECT c.heading, c.content, %s, %s
FROM chunks c
%s
WHERE c.record_ref = ?
ORDER BY c.chunk_index ASC, c.id ASC`, prefixExpr, embeddingSource, joinClause), ref)
	if err != nil {
		return nil, fmt.Errorf("query stored chunks for %s: %w", ref, err)
	}
	defer func() { _ = rows.Close() }()

	out := make(map[string][]byte)
	for rows.Next() {
		var (
			heading   string
			content   string
			prefix    string
			embedding []byte
		)
		if err := rows.Scan(&heading, &content, &prefix, &embedding); err != nil {
			return nil, fmt.Errorf("scan stored chunk for %s: %w", ref, err)
		}
		out[reuseChunkKey(title, heading, content, prefix)] = append([]byte(nil), embedding...)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate stored chunks for %s: %w", ref, err)
	}
	return out, nil
}

func reuseChunkKey(title, heading, body, prefix string) string {
	hasher := sha256.New()
	_, _ = hasher.Write([]byte(title))
	_, _ = hasher.Write([]byte{0})
	_, _ = hasher.Write([]byte(heading))
	_, _ = hasher.Write([]byte{0})
	_, _ = hasher.Write([]byte(body))
	_, _ = hasher.Write([]byte{0})
	_, _ = hasher.Write([]byte(prefix))
	return hex.EncodeToString(hasher.Sum(nil))
}
