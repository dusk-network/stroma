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
type reuseState struct {
	db *sql.DB
}

type storedRecord struct {
	contentHash string
	title       string
	chunks      map[string][]byte
}

type recordReusePlan struct {
	sections []chunk.Section
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

	if !isCompatibleReuseSnapshot(ctx, db, embedderFingerprint, embedderDimension, quantization) {
		_ = db.Close()
		return empty
	}

	return &reuseState{db: db}
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

func isCompatibleReuseSnapshot(ctx context.Context, db *sql.DB, embedderFingerprint string, embedderDimension int, quantization string) bool {
	schema, err := readMetadataValue(ctx, db, "schema_version")
	if err != nil || strings.TrimSpace(schema) != schemaVersion {
		return false
	}
	storedFingerprint, err := readMetadataValue(ctx, db, "embedder_fingerprint")
	if err != nil || strings.TrimSpace(storedFingerprint) != strings.TrimSpace(embedderFingerprint) {
		return false
	}
	storedDimension, err := readMetadataValue(ctx, db, "embedder_dimension")
	if err != nil || strings.TrimSpace(storedDimension) != strconv.Itoa(embedderDimension) {
		return false
	}
	storedQuantization, err := readMetadataValueOptional(ctx, db, "quantization", store.QuantizationFloat32)
	if err != nil {
		return false
	}
	normStored, err1 := normalizeQuantization(storedQuantization)
	normTarget, err2 := normalizeQuantization(quantization)
	if err1 != nil || err2 != nil || normStored != normTarget {
		return false
	}
	return true
}

func planRecordReuse(record corpus.Record, stored storedRecord, chunkOpts chunk.Options) recordReusePlan {
	sections := sectionsForRecord(record, chunkOpts)
	keys := make([]string, len(sections))
	for i, section := range sections {
		keys[i] = reuseChunkKey(record.Title, section.Heading, section.Body)
	}
	plan := recordReusePlan{
		sections:         sections,
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
// embeddings for ref, on demand. Lookups failures (including "no row") fall
// back to an empty storedRecord, matching the legacy best-effort reuse
// contract: a failed read misses a reuse opportunity but never blocks the
// write path.
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
			// Best-effort: swallow read errors so reuse stays non-blocking.
			_ = err
		}
		return storedRecord{}
	}

	chunks, err := loadStoredChunksForRecord(ctx, state.db, ref, title)
	if err != nil {
		return storedRecord{}
	}
	return storedRecord{
		contentHash: contentHash,
		title:       title,
		chunks:      chunks,
	}
}

func loadStoredChunksForRecord(ctx context.Context, db *sql.DB, ref, title string) (map[string][]byte, error) {
	rows, err := db.QueryContext(ctx, `
SELECT c.heading, c.content, v.embedding
FROM chunks c
JOIN chunks_vec v ON v.chunk_id = c.id
WHERE c.record_ref = ?
ORDER BY c.chunk_index ASC, c.id ASC`, ref)
	if err != nil {
		return nil, fmt.Errorf("query stored chunks for %s: %w", ref, err)
	}
	defer func() { _ = rows.Close() }()

	out := make(map[string][]byte)
	for rows.Next() {
		var (
			heading   string
			content   string
			embedding []byte
		)
		if err := rows.Scan(&heading, &content, &embedding); err != nil {
			return nil, fmt.Errorf("scan stored chunk for %s: %w", ref, err)
		}
		out[reuseChunkKey(title, heading, content)] = append([]byte(nil), embedding...)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate stored chunks for %s: %w", ref, err)
	}
	return out, nil
}

func reuseChunkKey(title, heading, body string) string {
	hasher := sha256.New()
	_, _ = hasher.Write([]byte(title))
	_, _ = hasher.Write([]byte{0})
	_, _ = hasher.Write([]byte(heading))
	_, _ = hasher.Write([]byte{0})
	_, _ = hasher.Write([]byte(body))
	return hex.EncodeToString(hasher.Sum(nil))
}
