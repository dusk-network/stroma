package index

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/dusk-network/stroma/chunk"
	"github.com/dusk-network/stroma/corpus"
	"github.com/dusk-network/stroma/embed"
	"github.com/dusk-network/stroma/store"
)

const schemaVersion = "1"

// BuildOptions controls how a Stroma index is rebuilt.
type BuildOptions struct {
	Path          string
	ReuseFromPath string
	Embedder      embed.Embedder
}

// BuildResult summarizes a completed rebuild.
type BuildResult struct {
	Path                string
	RecordCount         int
	ChunkCount          int
	ReusedRecordCount   int
	ReusedChunkCount    int
	EmbeddedChunkCount  int
	EmbedderDimension   int
	EmbedderFingerprint string
	ContentFingerprint  string
}

// Stats describes a built Stroma index.
type Stats struct {
	Path                string
	RecordCount         int
	ChunkCount          int
	KindCounts          map[string]int
	SchemaVersion       string
	EmbedderDimension   int
	EmbedderFingerprint string
	ContentFingerprint  string
	CreatedAt           string
}

// SearchQuery defines one semantic search.
type SearchQuery struct {
	Path     string
	Text     string
	Limit    int
	Kinds    []string
	Embedder embed.Embedder
}

// SearchHit is one retrieved section.
type SearchHit struct {
	ChunkID   int64
	Ref       string
	Kind      string
	Title     string
	SourceRef string
	Heading   string
	Content   string
	Metadata  map[string]string
	Score     float64
}

// Rebuild atomically recreates the index at the requested path.
func Rebuild(ctx context.Context, records []corpus.Record, options BuildOptions) (*BuildResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	indexPath := strings.TrimSpace(options.Path)
	if indexPath == "" {
		return nil, fmt.Errorf("build path is required")
	}
	if options.Embedder == nil {
		return nil, fmt.Errorf("build embedder is required")
	}

	normalized, err := normalizeRecords(records)
	if err != nil {
		return nil, err
	}
	dimension, err := options.Embedder.Dimension(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolve embedder dimension: %w", err)
	}
	if dimension <= 0 {
		return nil, fmt.Errorf("embedder dimension must be positive")
	}
	reuseState := loadReuseStateContext(ctx, options.ReuseFromPath, options.Embedder.Fingerprint(), dimension)

	if err := os.MkdirAll(filepath.Dir(indexPath), 0o755); err != nil {
		return nil, fmt.Errorf("create index directory: %w", err)
	}

	tmpPath := indexPath + ".new"
	_ = os.Remove(tmpPath)

	db, err := store.OpenReadWriteContext(ctx, tmpPath)
	if err != nil {
		return nil, fmt.Errorf("open staging index: %w", err)
	}
	defer func() {
		_ = db.Close()
		_ = os.Remove(tmpPath)
	}()

	if err := createSchemaContext(ctx, db, dimension); err != nil {
		return nil, fmt.Errorf("create schema: %w", err)
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("begin rebuild transaction: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()

	recordStmt, err := tx.PrepareContext(ctx, `INSERT INTO records (
ref, kind, title, source_ref, body_format, body_text, content_hash, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return nil, fmt.Errorf("prepare record insert: %w", err)
	}
	defer recordStmt.Close()

	chunkStmt, err := tx.PrepareContext(ctx, `INSERT INTO chunks (
record_ref, chunk_index, heading, content
) VALUES (?, ?, ?, ?)`)
	if err != nil {
		return nil, fmt.Errorf("prepare chunk insert: %w", err)
	}
	defer chunkStmt.Close()

	vectorStmt, err := tx.PrepareContext(ctx, `INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)`)
	if err != nil {
		return nil, fmt.Errorf("prepare vector insert: %w", err)
	}
	defer vectorStmt.Close()

	result := &BuildResult{
		Path:                indexPath,
		RecordCount:         len(normalized),
		EmbedderDimension:   dimension,
		EmbedderFingerprint: options.Embedder.Fingerprint(),
		ContentFingerprint:  corpus.Fingerprint(normalized),
	}

	for _, record := range normalized {
		if err := insertRecordContext(ctx, recordStmt, record); err != nil {
			return nil, err
		}

		plan := planRecordReuse(record, storedRecordForReuse(reuseState, record.Ref))
		sections := plan.sections
		result.ChunkCount += len(sections)
		result.ReusedChunkCount += plan.reusedChunkCount
		result.EmbeddedChunkCount += plan.embeddedChunkCount
		if plan.recordUnchanged {
			result.ReusedRecordCount++
		}
		if len(sections) == 0 {
			continue
		}

		texts := make([]string, 0, len(sections))
		for _, section := range sections {
			key := reuseChunkKey(record.Title, section.Heading, section.Body)
			if _, ok := plan.reusedEmbeddings[key]; ok {
				continue
			}
			texts = append(texts, textForEmbedding(record.Title, section))
		}
		vectors := make([][]float64, 0, len(texts))
		if len(texts) > 0 {
			vectors, err = options.Embedder.EmbedDocuments(ctx, texts)
			if err != nil {
				return nil, fmt.Errorf("embed record %s: %w", record.Ref, err)
			}
			if len(vectors) != len(texts) {
				return nil, fmt.Errorf("embedder returned %d vectors for %d new sections on %s", len(vectors), len(texts), record.Ref)
			}
		}

		newVectorIndex := 0
		for index, section := range sections {
			chunkResult, err := chunkStmt.ExecContext(ctx, record.Ref, index, section.Heading, section.Body)
			if err != nil {
				return nil, fmt.Errorf("insert record %s chunk %d: %w", record.Ref, index, err)
			}
			chunkID, err := chunkResult.LastInsertId()
			if err != nil {
				return nil, fmt.Errorf("read chunk id for %s chunk %d: %w", record.Ref, index, err)
			}
			key := reuseChunkKey(record.Title, section.Heading, section.Body)
			if blob, ok := plan.reusedEmbeddings[key]; ok {
				if _, err := vectorStmt.ExecContext(ctx, chunkID, blob); err != nil {
					return nil, fmt.Errorf("insert reused embedding for %s chunk %d: %w", record.Ref, index, err)
				}
				continue
			}
			if newVectorIndex >= len(vectors) {
				return nil, fmt.Errorf("record %s chunk %d is missing a new embedding", record.Ref, index)
			}
			if len(vectors[newVectorIndex]) != dimension {
				return nil, fmt.Errorf("record %s section %d embedding has dimension %d, want %d", record.Ref, index, len(vectors[newVectorIndex]), dimension)
			}
			blob, err := store.EncodeVectorBlob(vectors[newVectorIndex])
			if err != nil {
				return nil, fmt.Errorf("encode embedding for %s chunk %d: %w", record.Ref, index, err)
			}
			if _, err := vectorStmt.ExecContext(ctx, chunkID, blob); err != nil {
				return nil, fmt.Errorf("insert embedding for %s chunk %d: %w", record.Ref, index, err)
			}
			newVectorIndex++
		}
	}

	createdAt := time.Now().UTC().Format(time.RFC3339)
	metadata := map[string]string{
		"schema_version":       schemaVersion,
		"embedder_dimension":   strconv.Itoa(dimension),
		"embedder_fingerprint": result.EmbedderFingerprint,
		"content_fingerprint":  result.ContentFingerprint,
		"created_at":           createdAt,
	}
	for key, value := range metadata {
		if _, err := tx.ExecContext(ctx, `INSERT INTO metadata (key, value) VALUES (?, ?)`, key, value); err != nil {
			return nil, fmt.Errorf("insert metadata %s: %w", key, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("commit rebuild transaction: %w", err)
	}
	if err := runIntegrityChecksContext(ctx, db); err != nil {
		return nil, err
	}
	if err := db.Close(); err != nil {
		return nil, fmt.Errorf("close staging index: %w", err)
	}
	if err := replaceFile(tmpPath, indexPath); err != nil {
		return nil, err
	}
	return result, nil
}

// ReadStats inspects an existing index.
func ReadStats(ctx context.Context, path string) (*Stats, error) {
	snapshot, err := OpenSnapshot(ctx, path)
	if err != nil {
		return nil, err
	}
	defer snapshot.Close()
	return snapshot.Stats(ctx)
}

// Search returns semantically close sections from an existing index.
func Search(ctx context.Context, query SearchQuery) ([]SearchHit, error) {
	snapshot, err := OpenSnapshot(ctx, query.Path)
	if err != nil {
		return nil, err
	}
	defer snapshot.Close()
	return snapshot.Search(ctx, SnapshotSearchQuery{
		Text:     query.Text,
		Limit:    query.Limit,
		Kinds:    query.Kinds,
		Embedder: query.Embedder,
	})
}

func normalizeRecords(records []corpus.Record) ([]corpus.Record, error) {
	normalized := make([]corpus.Record, 0, len(records))
	for _, record := range records {
		normalizedRecord, err := record.Normalized()
		if err != nil {
			return nil, err
		}
		normalized = append(normalized, normalizedRecord)
	}
	sort.Slice(normalized, func(i, j int) bool {
		return normalized[i].Ref < normalized[j].Ref
	})
	return normalized, nil
}

func createSchemaContext(ctx context.Context, db *sql.DB, dimension int) error {
	statements := []string{
		`CREATE TABLE records (
			ref           TEXT PRIMARY KEY,
			kind          TEXT NOT NULL,
			title         TEXT NOT NULL,
			source_ref    TEXT NOT NULL,
			body_format   TEXT NOT NULL,
			body_text     TEXT NOT NULL,
			content_hash  TEXT NOT NULL,
			metadata_json TEXT NOT NULL
		)`,
		`CREATE TABLE chunks (
			id          INTEGER PRIMARY KEY AUTOINCREMENT,
			record_ref  TEXT NOT NULL,
			chunk_index INTEGER NOT NULL,
			heading     TEXT NOT NULL,
			content     TEXT NOT NULL,
			FOREIGN KEY (record_ref) REFERENCES records(ref) ON DELETE CASCADE
		)`,
		fmt.Sprintf(`CREATE VIRTUAL TABLE chunks_vec USING vec0(
			chunk_id INTEGER PRIMARY KEY,
			embedding float[%d] distance_metric=cosine
		)`, dimension),
		`CREATE TABLE metadata (
			key   TEXT PRIMARY KEY,
			value TEXT NOT NULL
		)`,
		`CREATE INDEX idx_records_kind ON records(kind)`,
		`CREATE INDEX idx_chunks_record_ref ON chunks(record_ref)`,
	}

	for _, statement := range statements {
		if _, err := db.ExecContext(ctx, statement); err != nil {
			return err
		}
	}
	return nil
}

func insertRecordContext(ctx context.Context, stmt *sql.Stmt, record corpus.Record) error {
	metadataJSON, err := json.Marshal(record.Metadata)
	if err != nil {
		return fmt.Errorf("marshal record %s metadata: %w", record.Ref, err)
	}
	if _, err := stmt.ExecContext(
		ctx,
		record.Ref,
		record.Kind,
		record.Title,
		record.SourceRef,
		record.BodyFormat,
		record.BodyText,
		record.ContentHash,
		string(metadataJSON),
	); err != nil {
		return fmt.Errorf("insert record %s: %w", record.Ref, err)
	}
	return nil
}

func sectionsForRecord(record corpus.Record) []chunk.Section {
	switch record.BodyFormat {
	case corpus.FormatPlaintext:
		text := strings.TrimSpace(record.BodyText)
		if text == "" {
			return nil
		}
		return []chunk.Section{{
			Heading: record.Title,
			Body:    text,
		}}
	default:
		return chunk.Markdown(record.Title, record.BodyText)
	}
}

func textForEmbedding(title string, section chunk.Section) string {
	parts := make([]string, 0, 3)
	if trimmed := strings.TrimSpace(title); trimmed != "" {
		parts = append(parts, trimmed)
	}
	if trimmed := strings.TrimSpace(section.Heading); trimmed != "" && trimmed != strings.TrimSpace(title) {
		parts = append(parts, trimmed)
	}
	if trimmed := strings.TrimSpace(section.Body); trimmed != "" {
		parts = append(parts, trimmed)
	}
	return strings.Join(parts, "\n\n")
}

func runIntegrityChecksContext(ctx context.Context, db *sql.DB) error {
	row := db.QueryRowContext(ctx, `PRAGMA integrity_check`)
	var result string
	if err := row.Scan(&result); err != nil {
		return fmt.Errorf("run integrity_check: %w", err)
	}
	if strings.ToLower(result) != "ok" {
		return fmt.Errorf("integrity_check failed: %s", result)
	}

	rows, err := db.QueryContext(ctx, `PRAGMA foreign_key_check`)
	if err != nil {
		return fmt.Errorf("run foreign_key_check: %w", err)
	}
	defer rows.Close()
	if rows.Next() {
		var table string
		var rowID int
		var parent string
		var foreignKeyID int
		if err := rows.Scan(&table, &rowID, &parent, &foreignKeyID); err != nil {
			return fmt.Errorf("scan foreign_key_check result: %w", err)
		}
		return fmt.Errorf("foreign_key_check failed: table=%s rowid=%d parent=%s fk=%d", table, rowID, parent, foreignKeyID)
	}
	return rows.Err()
}

func replaceFile(stagingPath, finalPath string) error {
	if err := os.Rename(stagingPath, finalPath); err == nil {
		return nil
	}
	if err := os.Remove(finalPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove existing index %s: %w", finalPath, err)
	}
	if err := os.Rename(stagingPath, finalPath); err != nil {
		return fmt.Errorf("replace %s with %s: %w", finalPath, stagingPath, err)
	}
	return nil
}

func readMetadataValue(ctx context.Context, db *sql.DB, key string) (string, error) {
	var value string
	if err := db.QueryRowContext(ctx, `SELECT value FROM metadata WHERE key = ?`, key).Scan(&value); err != nil {
		return "", fmt.Errorf("read metadata %s: %w", key, err)
	}
	return value, nil
}

func ensureCompatibleEmbedder(ctx context.Context, db *sql.DB, embedder embed.Embedder) error {
	indexFingerprint, err := readMetadataValue(ctx, db, "embedder_fingerprint")
	if err != nil {
		return err
	}
	queryFingerprint := embedder.Fingerprint()
	if indexFingerprint != queryFingerprint {
		return fmt.Errorf("embedder fingerprint mismatch: index=%q query=%q", indexFingerprint, queryFingerprint)
	}
	return nil
}

func buildSearchSQL(query SearchQuery, queryBlob []byte) (string, []any) {
	var builder strings.Builder
	args := make([]any, 0, 4+len(query.Kinds))

	builder.WriteString(`
WITH vector_hits AS (
  SELECT chunk_id, distance
  FROM chunks_vec
  WHERE embedding MATCH ? AND k = ?
  ORDER BY distance
)
SELECT
  vh.chunk_id,
  r.ref,
  r.kind,
  r.title,
  r.source_ref,
  c.heading,
  c.content,
  r.metadata_json,
  vh.distance
FROM vector_hits vh
JOIN chunks c ON c.id = vh.chunk_id
JOIN records r ON r.ref = c.record_ref
WHERE 1 = 1`)
	args = append(args, queryBlob, candidateLimit(query.Limit))

	if len(query.Kinds) > 0 {
		builder.WriteString(" AND r.kind IN (")
		for index, kind := range query.Kinds {
			if index > 0 {
				builder.WriteString(", ")
			}
			builder.WriteString("?")
			args = append(args, strings.TrimSpace(kind))
		}
		builder.WriteString(")")
	}

	builder.WriteString(`
ORDER BY vh.distance ASC, r.ref ASC, c.chunk_index ASC
LIMIT ?`)
	args = append(args, candidateLimit(query.Limit))
	return builder.String(), args
}

func candidateLimit(limit int) int {
	overfetch := limit * 5
	switch {
	case overfetch < 25:
		return 25
	case overfetch > 250:
		return 250
	default:
		return overfetch
	}
}
