// Package index orchestrates atomic Stroma index rebuilds and searches.
package index

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
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

const schemaVersion = "2"

// BuildOptions controls how a Stroma index is rebuilt.
type BuildOptions struct {
	Path          string
	ReuseFromPath string
	Embedder      embed.Embedder

	// MaxChunkTokens sets the approximate maximum number of tokens (words)
	// per chunk. Sections that exceed this limit are split into smaller
	// sub-sections. Zero disables token-budget splitting.
	MaxChunkTokens int

	// ChunkOverlapTokens sets the approximate number of overlapping tokens
	// between adjacent sub-sections when a section is split. Zero disables
	// overlap.
	ChunkOverlapTokens int

	// Quantization controls the vector storage format. Supported values
	// are "float32" (default) and "int8". Int8 reduces storage by 4x at
	// the cost of minor precision loss.
	Quantization string
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

// UpdateOptions controls how an existing Stroma index is updated in place.
type UpdateOptions struct {
	Path     string
	Embedder embed.Embedder

	// MaxChunkTokens sets the approximate maximum number of tokens (words)
	// per chunk. It should match the chunking policy used to build the current
	// index if callers want incremental updates to remain section-compatible.
	MaxChunkTokens int

	// ChunkOverlapTokens sets the approximate number of overlapping tokens
	// between adjacent sub-sections when a section is split. It should match
	// the chunking policy used to build the current index.
	ChunkOverlapTokens int

	// Quantization, when provided, must match the existing index. Leaving it
	// empty reuses the stored quantization metadata.
	Quantization string
}

// UpdateResult summarizes one incremental update.
type UpdateResult struct {
	Path                string
	UpsertedCount       int
	RemovedCount        int
	RecordCount         int
	ChunkCount          int
	ReusedRecordCount   int
	ReusedChunkCount    int
	EmbeddedChunkCount  int
	EmbedderDimension   int
	EmbedderFingerprint string
	ContentFingerprint  string
}

type updateIndexConfig struct {
	embedderFingerprint string
	dimension           int
	quantization        string
	contextualEmbedder  embed.ContextualEmbedder
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
	Reranker Reranker
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

// Reranker optionally refines one search candidate shortlist before the final
// limit truncation.
type Reranker interface {
	Rerank(ctx context.Context, query string, candidates []SearchHit) ([]SearchHit, error)
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
	quantization, err := normalizeQuantization(options.Quantization)
	if err != nil {
		return nil, err
	}
	reuseState := loadReuseStateContext(ctx, options.ReuseFromPath, options.Embedder.Fingerprint(), dimension, quantization)

	if err := os.MkdirAll(filepath.Dir(indexPath), 0o750); err != nil {
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

	if err := createSchemaContext(ctx, db, dimension, quantization); err != nil {
		return nil, fmt.Errorf("create schema: %w", err)
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("begin rebuild transaction: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()

	session, err := newIndexSession(ctx, tx, sessionConfig{
		embedder: options.Embedder,
		reuse:    reuseState,
		chunkOpts: chunk.Options{
			MaxTokens:     options.MaxChunkTokens,
			OverlapTokens: options.ChunkOverlapTokens,
		},
		dimension:    dimension,
		quantization: quantization,
	})
	if err != nil {
		return nil, err
	}
	defer session.Close()

	result := &BuildResult{
		Path:                indexPath,
		RecordCount:         len(normalized),
		EmbedderDimension:   dimension,
		EmbedderFingerprint: options.Embedder.Fingerprint(),
		ContentFingerprint:  corpus.Fingerprint(normalized),
	}

	for _, record := range normalized {
		stats, err := session.processRecord(ctx, record)
		if err != nil {
			return nil, err
		}
		result.ChunkCount += stats.sections
		result.ReusedChunkCount += stats.reusedChunks
		result.EmbeddedChunkCount += stats.embeddedChunks
		if stats.recordUnchanged {
			result.ReusedRecordCount++
		}
	}

	createdAt := time.Now().UTC().Format(time.RFC3339)
	metadata := map[string]string{
		"schema_version":       schemaVersion,
		"embedder_dimension":   strconv.Itoa(dimension),
		"embedder_fingerprint": result.EmbedderFingerprint,
		"content_fingerprint":  result.ContentFingerprint,
		"quantization":         quantization,
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

// Update applies add, replace, and remove operations to an existing Stroma
// index without rebuilding it from scratch.
func Update(ctx context.Context, added []corpus.Record, removed []string, options UpdateOptions) (*UpdateResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	indexPath := strings.TrimSpace(options.Path)
	if indexPath == "" {
		return nil, fmt.Errorf("update path is required")
	}
	if err := ensureExistingIndexContext(indexPath); err != nil {
		return nil, err
	}

	normalizedAdded, err := normalizeRecords(added)
	if err != nil {
		return nil, err
	}
	removedRefs := normalizeRemovedRefs(removed)
	if err := validateUpdateInputs(normalizedAdded, removedRefs); err != nil {
		return nil, err
	}

	db, err := store.OpenReadWriteContext(ctx, indexPath)
	if err != nil {
		return nil, fmt.Errorf("open index %s: %w", indexPath, err)
	}
	defer func() { _ = db.Close() }()

	updateConfig, err := resolveUpdateIndexContext(ctx, db, normalizedAdded, options)
	if err != nil {
		return nil, err
	}

	reuseState := &reuseState{records: map[string]storedRecord{}}
	if len(normalizedAdded) > 0 {
		reuseState = loadReuseStateContext(
			ctx,
			indexPath,
			updateConfig.embedderFingerprint,
			updateConfig.dimension,
			updateConfig.quantization,
		)
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("begin update transaction: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()

	result := &UpdateResult{
		Path:                indexPath,
		UpsertedCount:       len(normalizedAdded),
		EmbedderDimension:   updateConfig.dimension,
		EmbedderFingerprint: updateConfig.embedderFingerprint,
	}

	for _, ref := range removedRefs {
		deleted, err := deleteRecordContext(ctx, tx, ref)
		if err != nil {
			return nil, err
		}
		if deleted {
			result.RemovedCount++
		}
	}

	var (
		recordStmt *sql.Stmt
		chunkStmt  *sql.Stmt
		vectorStmt *sql.Stmt
		ftsStmt    *sql.Stmt
	)
	if len(normalizedAdded) > 0 {
		recordStmt, err = tx.PrepareContext(ctx, `INSERT INTO records (
ref, kind, title, source_ref, body_format, body_text, content_hash, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
		if err != nil {
			return nil, fmt.Errorf("prepare record insert: %w", err)
		}
		defer func() { _ = recordStmt.Close() }()

		chunkStmt, err = tx.PrepareContext(ctx, `INSERT INTO chunks (
record_ref, chunk_index, heading, content
) VALUES (?, ?, ?, ?)`)
		if err != nil {
			return nil, fmt.Errorf("prepare chunk insert: %w", err)
		}
		defer func() { _ = chunkStmt.Close() }()

		vectorInsertSQL := `INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)`
		if updateConfig.quantization == store.QuantizationInt8 {
			vectorInsertSQL = `INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, vec_int8(?))`
		}
		vectorStmt, err = tx.PrepareContext(ctx, vectorInsertSQL)
		if err != nil {
			return nil, fmt.Errorf("prepare vector insert: %w", err)
		}
		defer func() { _ = vectorStmt.Close() }()

		ftsStmt, err = tx.PrepareContext(ctx, `INSERT INTO fts_chunks(rowid, title, heading, content) VALUES (?, ?, ?, ?)`)
		if err != nil {
			return nil, fmt.Errorf("prepare fts insert: %w", err)
		}
		defer func() { _ = ftsStmt.Close() }()
	}

	chunkOpts := chunk.Options{
		MaxTokens:     options.MaxChunkTokens,
		OverlapTokens: options.ChunkOverlapTokens,
	}
	for _, record := range normalizedAdded {
		if _, err := deleteRecordContext(ctx, tx, record.Ref); err != nil {
			return nil, err
		}
		if err := insertRecordContext(ctx, recordStmt, record); err != nil {
			return nil, err
		}

		plan := planRecordReuse(record, storedRecordForReuse(reuseState, record.Ref), chunkOpts)
		sections := plan.sections
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
			if updateConfig.contextualEmbedder != nil {
				vectors, err = updateConfig.contextualEmbedder.EmbedDocumentChunks(ctx, documentTextForEmbedding(record), texts)
			} else {
				vectors, err = options.Embedder.EmbedDocuments(ctx, texts)
			}
			if err != nil {
				return nil, fmt.Errorf("embed record %s: %w", record.Ref, err)
			}
			if len(vectors) != len(texts) {
				return nil, fmt.Errorf("embedder returned %d vectors for %d new sections on %s", len(vectors), len(texts), record.Ref)
			}
		}

		if err := insertChunksContext(
			ctx,
			record,
			plan,
			sections,
			vectors,
			updateConfig.dimension,
			updateConfig.quantization,
			chunkStmt,
			ftsStmt,
			vectorStmt,
		); err != nil {
			return nil, err
		}
	}

	recordsNow, err := loadCurrentRecordsContext(ctx, tx)
	if err != nil {
		return nil, err
	}
	result.RecordCount = len(recordsNow)
	result.ContentFingerprint = corpus.Fingerprint(recordsNow)
	result.ChunkCount, err = countChunksContext(ctx, tx)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC().Format(time.RFC3339)
	createdAt, err := readMetadataValueOptional(ctx, tx, "created_at", now)
	if err != nil {
		return nil, err
	}

	metadata := map[string]string{
		"schema_version":       schemaVersion,
		"embedder_dimension":   strconv.Itoa(updateConfig.dimension),
		"embedder_fingerprint": updateConfig.embedderFingerprint,
		"content_fingerprint":  result.ContentFingerprint,
		"quantization":         updateConfig.quantization,
		"created_at":           createdAt,
		"updated_at":           now,
	}
	if err := upsertMetadataContext(ctx, tx, metadata); err != nil {
		return nil, err
	}
	if err := runIntegrityChecksContext(ctx, tx); err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("commit update transaction: %w", err)
	}
	return result, nil
}

// ReadStats inspects an existing index.
func ReadStats(ctx context.Context, path string) (*Stats, error) {
	snapshot, err := OpenSnapshot(ctx, path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = snapshot.Close() }()
	return snapshot.Stats(ctx)
}

// Search returns semantically close sections from an existing index.
func Search(ctx context.Context, query SearchQuery) ([]SearchHit, error) {
	snapshot, err := OpenSnapshot(ctx, query.Path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = snapshot.Close() }()
	return snapshot.Search(ctx, SnapshotSearchQuery{
		Text:     query.Text,
		Limit:    query.Limit,
		Kinds:    query.Kinds,
		Embedder: query.Embedder,
		Reranker: query.Reranker,
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

func createSchemaContext(ctx context.Context, db *sql.DB, dimension int, quantization string) error {
	vecType := fmt.Sprintf("float[%d]", dimension)
	if quantization == store.QuantizationInt8 {
		vecType = fmt.Sprintf("int8[%d]", dimension)
	}

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
			embedding %s distance_metric=cosine
		)`, vecType),
		`CREATE TABLE metadata (
			key   TEXT PRIMARY KEY,
			value TEXT NOT NULL
		)`,
		`CREATE VIRTUAL TABLE fts_chunks USING fts5(title, heading, content)`,
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

type sessionConfig struct {
	embedder     embed.Embedder
	reuse        *reuseState
	chunkOpts    chunk.Options
	dimension    int
	quantization string
}

type recordStats struct {
	sections        int
	reusedChunks    int
	embeddedChunks  int
	recordUnchanged bool
}

// indexSession owns the prepared statements, embedder branch, and reuse state
// for the per-record write path. Rebuild delegates to it today; Update will
// migrate in a follow-up so both write paths share the same invariants.
type indexSession struct {
	tx           *sql.Tx
	embedder     embed.Embedder
	contextual   embed.ContextualEmbedder
	reuse        *reuseState
	chunkOpts    chunk.Options
	dimension    int
	quantization string

	recordStmt *sql.Stmt
	chunkStmt  *sql.Stmt
	vectorStmt *sql.Stmt
	ftsStmt    *sql.Stmt
}

func newIndexSession(ctx context.Context, tx *sql.Tx, cfg sessionConfig) (*indexSession, error) {
	s := &indexSession{
		tx:           tx,
		embedder:     cfg.embedder,
		reuse:        cfg.reuse,
		chunkOpts:    cfg.chunkOpts,
		dimension:    cfg.dimension,
		quantization: cfg.quantization,
	}
	if contextual, ok := cfg.embedder.(embed.ContextualEmbedder); ok {
		s.contextual = contextual
	}

	var err error
	s.recordStmt, err = tx.PrepareContext(ctx, `INSERT INTO records (
ref, kind, title, source_ref, body_format, body_text, content_hash, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		s.Close()
		return nil, fmt.Errorf("prepare record insert: %w", err)
	}

	s.chunkStmt, err = tx.PrepareContext(ctx, `INSERT INTO chunks (
record_ref, chunk_index, heading, content
) VALUES (?, ?, ?, ?)`)
	if err != nil {
		s.Close()
		return nil, fmt.Errorf("prepare chunk insert: %w", err)
	}

	vectorInsertSQL := `INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)`
	if cfg.quantization == store.QuantizationInt8 {
		vectorInsertSQL = `INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, vec_int8(?))`
	}
	s.vectorStmt, err = tx.PrepareContext(ctx, vectorInsertSQL)
	if err != nil {
		s.Close()
		return nil, fmt.Errorf("prepare vector insert: %w", err)
	}

	s.ftsStmt, err = tx.PrepareContext(ctx, `INSERT INTO fts_chunks(rowid, title, heading, content) VALUES (?, ?, ?, ?)`)
	if err != nil {
		s.Close()
		return nil, fmt.Errorf("prepare fts insert: %w", err)
	}

	return s, nil
}

// Close releases prepared statements. It is nil-safe and idempotent.
func (s *indexSession) Close() {
	if s == nil {
		return
	}
	for _, stmt := range []**sql.Stmt{&s.recordStmt, &s.chunkStmt, &s.vectorStmt, &s.ftsStmt} {
		if *stmt != nil {
			_ = (*stmt).Close()
			*stmt = nil
		}
	}
}

func (s *indexSession) processRecord(ctx context.Context, record corpus.Record) (recordStats, error) {
	var stats recordStats

	if err := insertRecordContext(ctx, s.recordStmt, record); err != nil {
		return stats, err
	}

	plan := planRecordReuse(record, storedRecordForReuse(s.reuse, record.Ref), s.chunkOpts)
	sections := plan.sections
	stats.sections = len(sections)
	stats.reusedChunks = plan.reusedChunkCount
	stats.embeddedChunks = plan.embeddedChunkCount
	stats.recordUnchanged = plan.recordUnchanged

	if len(sections) == 0 {
		return stats, nil
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
		var err error
		if s.contextual != nil {
			vectors, err = s.contextual.EmbedDocumentChunks(ctx, documentTextForEmbedding(record), texts)
		} else {
			vectors, err = s.embedder.EmbedDocuments(ctx, texts)
		}
		if err != nil {
			return stats, fmt.Errorf("embed record %s: %w", record.Ref, err)
		}
		if len(vectors) != len(texts) {
			return stats, fmt.Errorf("embedder returned %d vectors for %d new sections on %s", len(vectors), len(texts), record.Ref)
		}
	}

	if err := insertChunksContext(
		ctx,
		record,
		plan,
		sections,
		vectors,
		s.dimension,
		s.quantization,
		s.chunkStmt,
		s.ftsStmt,
		s.vectorStmt,
	); err != nil {
		return stats, err
	}

	return stats, nil
}

func insertChunksContext(
	ctx context.Context,
	record corpus.Record,
	plan recordReusePlan,
	sections []chunk.Section,
	vectors [][]float64,
	dimension int,
	quantization string,
	chunkStmt, ftsStmt, vectorStmt *sql.Stmt,
) error {
	newVectorIndex := 0
	for index, section := range sections {
		chunkResult, err := chunkStmt.ExecContext(ctx, record.Ref, index, section.Heading, section.Body)
		if err != nil {
			return fmt.Errorf("insert record %s chunk %d: %w", record.Ref, index, err)
		}
		chunkID, err := chunkResult.LastInsertId()
		if err != nil {
			return fmt.Errorf("read chunk id for %s chunk %d: %w", record.Ref, index, err)
		}
		if _, err := ftsStmt.ExecContext(ctx, chunkID, record.Title, section.Heading, section.Body); err != nil {
			return fmt.Errorf("insert fts for %s chunk %d: %w", record.Ref, index, err)
		}
		key := reuseChunkKey(record.Title, section.Heading, section.Body)
		if blob, ok := plan.reusedEmbeddings[key]; ok {
			if _, err := vectorStmt.ExecContext(ctx, chunkID, blob); err != nil {
				return fmt.Errorf("insert reused embedding for %s chunk %d: %w", record.Ref, index, err)
			}
			continue
		}
		if newVectorIndex >= len(vectors) {
			return fmt.Errorf("record %s chunk %d is missing a new embedding", record.Ref, index)
		}
		if len(vectors[newVectorIndex]) != dimension {
			return fmt.Errorf("record %s section %d embedding has dimension %d, want %d", record.Ref, index, len(vectors[newVectorIndex]), dimension)
		}
		blob, err := encodeVector(vectors[newVectorIndex], quantization)
		if err != nil {
			return fmt.Errorf("encode embedding for %s chunk %d: %w", record.Ref, index, err)
		}
		if _, err := vectorStmt.ExecContext(ctx, chunkID, blob); err != nil {
			return fmt.Errorf("insert embedding for %s chunk %d: %w", record.Ref, index, err)
		}
		newVectorIndex++
	}
	return nil
}

func sectionsForRecord(record corpus.Record, opts chunk.Options) []chunk.Section {
	switch record.BodyFormat {
	case corpus.FormatPlaintext:
		text := strings.TrimSpace(record.BodyText)
		if text == "" {
			return nil
		}
		sections := []chunk.Section{{
			Heading: record.Title,
			Body:    text,
		}}
		if opts.MaxTokens > 0 {
			var result []chunk.Section
			for _, s := range sections {
				result = append(result, chunk.SplitSection(s, opts.MaxTokens, opts.OverlapTokens)...)
			}
			return result
		}
		return sections
	default:
		if opts.MaxTokens > 0 {
			return chunk.MarkdownWithOptions(record.Title, record.BodyText, opts)
		}
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

func documentTextForEmbedding(record corpus.Record) string {
	parts := make([]string, 0, 2)
	if trimmed := strings.TrimSpace(record.Title); trimmed != "" {
		parts = append(parts, trimmed)
	}
	if trimmed := strings.TrimSpace(record.BodyText); trimmed != "" {
		parts = append(parts, trimmed)
	}
	return strings.Join(parts, "\n\n")
}

func runIntegrityChecksContext(ctx context.Context, query queryContextRunner) error {
	row := query.QueryRowContext(ctx, `PRAGMA integrity_check`)
	var result string
	if err := row.Scan(&result); err != nil {
		return fmt.Errorf("run integrity_check: %w", err)
	}
	if !strings.EqualFold(result, "ok") {
		return fmt.Errorf("integrity_check failed: %s", result)
	}

	rows, err := query.QueryContext(ctx, `PRAGMA foreign_key_check`)
	if err != nil {
		return fmt.Errorf("run foreign_key_check: %w", err)
	}
	defer func() { _ = rows.Close() }()
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

type queryContextRunner interface {
	QueryContext(context.Context, string, ...any) (*sql.Rows, error)
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

func readMetadataValue(ctx context.Context, query queryContextRunner, key string) (string, error) {
	var value string
	if err := query.QueryRowContext(ctx, `SELECT value FROM metadata WHERE key = ?`, key).Scan(&value); err != nil {
		return "", fmt.Errorf("read metadata %s: %w", key, err)
	}
	return value, nil
}

//nolint:unparam // key will vary as more metadata fields are added
func readMetadataValueOptional(ctx context.Context, query queryContextRunner, key, defaultValue string) (string, error) {
	value, err := readMetadataValue(ctx, query, key)
	switch {
	case err == nil:
		return value, nil
	case errors.Is(err, sql.ErrNoRows):
		return defaultValue, nil
	default:
		return "", err
	}
}

func ensureCompatibleEmbedder(ctx context.Context, db queryContextRunner, embedder embed.Embedder) error {
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

func buildSearchSQL(query SearchQuery, queryBlob []byte, quantization string) (querySQL string, args []any) {
	var builder strings.Builder
	args = make([]any, 0, 4+len(query.Kinds))

	matchExpr := "?"
	if quantization == store.QuantizationInt8 {
		matchExpr = "vec_int8(?)"
	}

	builder.WriteString(fmt.Sprintf(`
WITH vector_hits AS (
  SELECT chunk_id, distance
  FROM chunks_vec
  WHERE embedding MATCH %s AND k = ?
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
WHERE 1 = 1`, matchExpr))
	args = append(args, queryBlob, query.Limit)

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
	args = append(args, query.Limit)
	return builder.String(), args
}

func ensureExistingIndexContext(path string) error {
	info, err := os.Stat(path)
	switch {
	case os.IsNotExist(err):
		return &store.MissingIndexError{Path: path}
	case err != nil:
		return fmt.Errorf("stat index %s: %w", path, err)
	case info.IsDir():
		return fmt.Errorf("index path %s is a directory", path)
	default:
		return nil
	}
}

func normalizeRemovedRefs(refs []string) []string {
	seen := make(map[string]struct{}, len(refs))
	result := make([]string, 0, len(refs))
	for _, ref := range refs {
		trimmed := strings.TrimSpace(ref)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		result = append(result, trimmed)
	}
	sort.Strings(result)
	return result
}

func validateUpdateInputs(added []corpus.Record, removed []string) error {
	addedRefs := make(map[string]struct{}, len(added))
	for _, record := range added {
		if _, ok := addedRefs[record.Ref]; ok {
			return fmt.Errorf("duplicate added record ref %q", record.Ref)
		}
		addedRefs[record.Ref] = struct{}{}
	}
	for _, ref := range removed {
		if _, ok := addedRefs[ref]; ok {
			return fmt.Errorf("record ref %q appears in both added and removed", ref)
		}
	}
	return nil
}

func resolveUpdateIndexContext(ctx context.Context, db *sql.DB, added []corpus.Record, options UpdateOptions) (*updateIndexConfig, error) {
	schema, err := readMetadataValue(ctx, db, "schema_version")
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(schema) != schemaVersion {
		return nil, fmt.Errorf("schema version mismatch: index=%q update=%q", strings.TrimSpace(schema), schemaVersion)
	}

	embedderFingerprint, err := readMetadataValue(ctx, db, "embedder_fingerprint")
	if err != nil {
		return nil, err
	}
	dimensionValue, err := readMetadataValue(ctx, db, "embedder_dimension")
	if err != nil {
		return nil, err
	}
	dimension, err := strconv.Atoi(strings.TrimSpace(dimensionValue))
	if err != nil {
		return nil, fmt.Errorf("parse embedder_dimension %q: %w", dimensionValue, err)
	}

	quantization, err := readMetadataValueOptional(ctx, db, "quantization", store.QuantizationFloat32)
	if err != nil {
		return nil, err
	}
	quantization, err = normalizeQuantization(quantization)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(options.Quantization) != "" {
		requestedQuantization, err := normalizeQuantization(options.Quantization)
		if err != nil {
			return nil, err
		}
		if requestedQuantization != quantization {
			return nil, fmt.Errorf("quantization mismatch: index=%q update=%q", quantization, requestedQuantization)
		}
	}

	if len(added) == 0 && options.Embedder == nil {
		return &updateIndexConfig{
			embedderFingerprint: embedderFingerprint,
			dimension:           dimension,
			quantization:        quantization,
		}, nil
	}
	if options.Embedder == nil {
		return nil, fmt.Errorf("update embedder is required when adding records")
	}

	updateFingerprint := options.Embedder.Fingerprint()
	if embedderFingerprint != updateFingerprint {
		return nil, fmt.Errorf("embedder fingerprint mismatch: index=%q update=%q", embedderFingerprint, updateFingerprint)
	}
	updateDimension, err := options.Embedder.Dimension(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolve update embedder dimension: %w", err)
	}
	if updateDimension <= 0 {
		return nil, fmt.Errorf("embedder dimension must be positive")
	}
	if updateDimension != dimension {
		return nil, fmt.Errorf("embedder dimension mismatch: index=%d update=%d", dimension, updateDimension)
	}

	contextualEmbedder, _ := options.Embedder.(embed.ContextualEmbedder)
	return &updateIndexConfig{
		embedderFingerprint: embedderFingerprint,
		dimension:           dimension,
		quantization:        quantization,
		contextualEmbedder:  contextualEmbedder,
	}, nil
}

func deleteRecordContext(ctx context.Context, tx *sql.Tx, ref string) (bool, error) {
	if _, err := tx.ExecContext(ctx, `DELETE FROM chunks_vec WHERE chunk_id IN (SELECT id FROM chunks WHERE record_ref = ?)`, ref); err != nil {
		return false, fmt.Errorf("delete vectors for %s: %w", ref, err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM fts_chunks WHERE rowid IN (SELECT id FROM chunks WHERE record_ref = ?)`, ref); err != nil {
		return false, fmt.Errorf("delete fts rows for %s: %w", ref, err)
	}
	result, err := tx.ExecContext(ctx, `DELETE FROM records WHERE ref = ?`, ref)
	if err != nil {
		return false, fmt.Errorf("delete record %s: %w", ref, err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("read deleted row count for %s: %w", ref, err)
	}
	return rowsAffected > 0, nil
}

func loadCurrentRecordsContext(ctx context.Context, query queryContextRunner) ([]corpus.Record, error) {
	rows, err := query.QueryContext(ctx, `
SELECT ref, kind, title, source_ref, body_format, body_text, content_hash, metadata_json
FROM records
ORDER BY ref ASC`)
	if err != nil {
		return nil, fmt.Errorf("query current records: %w", err)
	}
	defer func() { _ = rows.Close() }()

	records := make([]corpus.Record, 0)
	for rows.Next() {
		var (
			record      corpus.Record
			metadataRaw string
		)
		if err := rows.Scan(
			&record.Ref,
			&record.Kind,
			&record.Title,
			&record.SourceRef,
			&record.BodyFormat,
			&record.BodyText,
			&record.ContentHash,
			&metadataRaw,
		); err != nil {
			return nil, fmt.Errorf("scan current record: %w", err)
		}
		record.Metadata, err = unmarshalMetadata(record.Ref, metadataRaw)
		if err != nil {
			return nil, err
		}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate current records: %w", err)
	}
	return records, nil
}

func countChunksContext(ctx context.Context, query queryContextRunner) (int, error) {
	var chunkCount int
	if err := query.QueryRowContext(ctx, `SELECT COUNT(*) FROM chunks`).Scan(&chunkCount); err != nil {
		return 0, fmt.Errorf("count chunks: %w", err)
	}
	return chunkCount, nil
}

func upsertMetadataContext(ctx context.Context, tx *sql.Tx, metadata map[string]string) error {
	for key, value := range metadata {
		if _, err := tx.ExecContext(ctx, `
INSERT INTO metadata (key, value) VALUES (?, ?)
ON CONFLICT(key) DO UPDATE SET value = excluded.value`, key, value); err != nil {
			return fmt.Errorf("upsert metadata %s: %w", key, err)
		}
	}
	return nil
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

func normalizeQuantization(q string) (string, error) {
	q = strings.TrimSpace(strings.ToLower(q))
	switch q {
	case "", store.QuantizationFloat32:
		return store.QuantizationFloat32, nil
	case store.QuantizationInt8:
		return store.QuantizationInt8, nil
	default:
		return "", fmt.Errorf("unsupported quantization mode %q (must be %q or %q)", q, store.QuantizationFloat32, store.QuantizationInt8)
	}
}

func encodeVector(vector []float64, quantization string) ([]byte, error) {
	if quantization == store.QuantizationInt8 {
		return store.EncodeVectorBlobInt8(vector)
	}
	return store.EncodeVectorBlob(vector)
}

func decodeVector(blob []byte, quantization string) ([]float64, error) {
	if quantization == store.QuantizationInt8 {
		return store.DecodeVectorBlobInt8(blob)
	}
	return store.DecodeVectorBlob(blob)
}
