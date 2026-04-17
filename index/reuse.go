package index

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/dusk-network/stroma/chunk"
	"github.com/dusk-network/stroma/corpus"
	"github.com/dusk-network/stroma/store"
)

type reuseState struct {
	records map[string]storedRecord
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
	empty := &reuseState{records: map[string]storedRecord{}}
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
	defer func() { _ = db.Close() }()

	if !isCompatibleReuseSnapshot(ctx, db, embedderFingerprint, embedderDimension, quantization) {
		return empty
	}

	state, err := loadStoredReuseRecords(ctx, db)
	if err != nil {
		return empty
	}
	return state
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

func loadStoredReuseRecords(ctx context.Context, db *sql.DB) (*reuseState, error) {
	rows, err := db.QueryContext(ctx, `SELECT ref, title, content_hash FROM records`)
	if err != nil {
		return nil, fmt.Errorf("query stored records: %w", err)
	}
	defer func() { _ = rows.Close() }()

	state := &reuseState{records: make(map[string]storedRecord)}
	for rows.Next() {
		var (
			ref         string
			title       string
			contentHash string
		)
		if err := rows.Scan(&ref, &title, &contentHash); err != nil {
			return nil, fmt.Errorf("scan stored record: %w", err)
		}
		state.records[ref] = storedRecord{
			contentHash: contentHash,
			title:       title,
			chunks:      map[string][]byte{},
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate stored records: %w", err)
	}

	rows, err = db.QueryContext(ctx, `
SELECT c.record_ref, c.heading, c.content, v.embedding
FROM chunks c
JOIN chunks_vec v ON v.chunk_id = c.id
ORDER BY c.record_ref, c.chunk_index ASC, c.id ASC`)
	if err != nil {
		return nil, fmt.Errorf("query stored chunks: %w", err)
	}
	defer func() { _ = rows.Close() }()

	for rows.Next() {
		var (
			ref       string
			heading   string
			content   string
			embedding []byte
		)
		if err := rows.Scan(&ref, &heading, &content, &embedding); err != nil {
			return nil, fmt.Errorf("scan stored chunk: %w", err)
		}
		record, ok := state.records[ref]
		if !ok {
			continue
		}
		record.chunks[reuseChunkKey(record.title, heading, content)] = append([]byte(nil), embedding...)
		state.records[ref] = record
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate stored chunks: %w", err)
	}

	return state, nil
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

func storedRecordForReuse(state *reuseState, ref string) storedRecord {
	if state == nil || state.records == nil {
		return storedRecord{}
	}
	return state.records[ref]
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
