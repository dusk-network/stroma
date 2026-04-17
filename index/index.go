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

const schemaVersion = "3"

// BuildOptions controls how a Stroma index is rebuilt.
type BuildOptions struct {
	Path string

	// ReuseFromPath points at an existing Stroma snapshot whose embeddings
	// should be reused at the section level: a new section reuses its
	// stored embedding whenever its title, heading, and body match a
	// section already present in the prior snapshot. Records that are
	// fully unchanged are the maximal case, but sections carried over
	// from an edited record still reuse their embeddings. The snapshot is
	// opened read-only and queried per-record during the rebuild, so
	// resident memory scales with a single record's chunks rather than
	// with the whole corpus. Leave empty to disable reuse.
	ReuseFromPath string

	Embedder embed.Embedder

	// Contextualizer optionally produces a per-chunk prefix string that
	// gets prepended before the embedding text and the FTS5 content. When
	// set, the prefix persists on the chunk and participates in reuse
	// keying so a changed contextualizer invalidates stale reuse without
	// corrupting the stored representation. Nil disables contextualization
	// and leaves the build identical to the non-contextual path.
	Contextualizer ChunkContextualizer

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

	// Contextualizer optionally produces a per-chunk prefix string. See
	// BuildOptions.Contextualizer for the contract. Leaving it nil
	// preserves the non-contextual path and produces chunks with an
	// empty persisted prefix.
	Contextualizer ChunkContextualizer

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

	// SearchDimension optionally runs a truncated-prefix vector prefilter
	// at this dimension, then rescores the shortlist with full-dim cosine.
	// Zero (default) uses the full stored dimension throughout. Positive
	// values must be <= the stored embedder dimension. Only valid when the
	// stored quantization is float32; returns an error against int8 indexes.
	// This is the shape Matryoshka Representation Learning (MRL) embeddings
	// rely on — callers who use non-MRL embeddings should leave it zero.
	//
	// The truncated path is a brute-force scan over chunks_vec, not a
	// vec0 kNN MATCH, so it is not asymptotically cheaper than the default
	// path: its win is constant-factor (fewer floats per cosine) and only
	// pays off when the truncated prefix preserves ranking. Treat this as
	// a tuning knob for MRL snapshots rather than a blanket speedup.
	SearchDimension int
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

// ChunkContextualizer produces a short explanatory prefix for each section
// of a record. The returned slice must be the same length as sections and
// aligned with it index-for-index. An empty prefix is allowed and disables
// contextual retrieval for that section. The returned prefix is prepended
// to the embedding text and to the FTS5 content column; it is persisted so
// reuse keying can detect when a changed contextualizer needs to invalidate
// the stored embedding.
type ChunkContextualizer interface {
	ContextualizeChunks(ctx context.Context, record corpus.Record, sections []chunk.Section) ([]string, error)
}

// Rebuild atomically recreates the index at the requested path.
func Rebuild(ctx context.Context, records []corpus.Record, options BuildOptions) (*BuildResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	inputs, err := prepareBuildInputs(ctx, records, options)
	if err != nil {
		return nil, err
	}
	defer func() { _ = inputs.reuseState.Close() }()

	if err := os.MkdirAll(filepath.Dir(inputs.indexPath), 0o750); err != nil {
		return nil, fmt.Errorf("create index directory: %w", err)
	}

	tmpPath := inputs.indexPath + ".new"
	_ = os.Remove(tmpPath)

	db, err := store.OpenReadWriteContext(ctx, tmpPath)
	if err != nil {
		return nil, fmt.Errorf("open staging index: %w", err)
	}
	defer func() {
		_ = db.Close()
		_ = os.Remove(tmpPath)
	}()

	if err := createSchema(ctx, db, inputs.dimension, inputs.quantization); err != nil {
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
		embedder:            options.Embedder,
		embedderFingerprint: options.Embedder.Fingerprint(),
		contextualizer:      options.Contextualizer,
		reuse:               inputs.reuseState,
		chunkOpts: chunk.Options{
			MaxTokens:     options.MaxChunkTokens,
			OverlapTokens: options.ChunkOverlapTokens,
		},
		dimension:    inputs.dimension,
		quantization: inputs.quantization,
	})
	if err != nil {
		return nil, err
	}
	defer session.Close()

	result := &BuildResult{
		Path:                inputs.indexPath,
		RecordCount:         len(inputs.normalized),
		EmbedderDimension:   inputs.dimension,
		EmbedderFingerprint: options.Embedder.Fingerprint(),
		ContentFingerprint:  corpus.Fingerprint(inputs.normalized),
	}

	for _, record := range inputs.normalized {
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

	// Release the reuse snapshot before swapping the staged index into
	// place: on Windows, replaceFile cannot rename over an open SQLite
	// handle, and for self-reuse rebuilds (Path == ReuseFromPath) the
	// open reuse connection points at the file being replaced.
	if err := inputs.reuseState.Close(); err != nil {
		return nil, fmt.Errorf("close reuse snapshot: %w", err)
	}

	metadata := map[string]string{
		"schema_version":       schemaVersion,
		"embedder_dimension":   strconv.Itoa(inputs.dimension),
		"embedder_fingerprint": result.EmbedderFingerprint,
		"content_fingerprint":  result.ContentFingerprint,
		"quantization":         inputs.quantization,
		"created_at":           time.Now().UTC().Format(time.RFC3339),
	}
	if err := finalizeRebuild(ctx, db, tx, metadata, tmpPath, inputs.indexPath); err != nil {
		return nil, err
	}
	return result, nil
}

type rebuildInputs struct {
	indexPath    string
	normalized   []corpus.Record
	dimension    int
	quantization string
	reuseState   *reuseState
}

func prepareBuildInputs(ctx context.Context, records []corpus.Record, options BuildOptions) (rebuildInputs, error) {
	indexPath := strings.TrimSpace(options.Path)
	if indexPath == "" {
		return rebuildInputs{}, fmt.Errorf("build path is required")
	}
	if options.Embedder == nil {
		return rebuildInputs{}, fmt.Errorf("build embedder is required")
	}

	normalized, err := normalizeRecords(records)
	if err != nil {
		return rebuildInputs{}, err
	}
	dimension, err := options.Embedder.Dimension(ctx)
	if err != nil {
		return rebuildInputs{}, fmt.Errorf("resolve embedder dimension: %w", err)
	}
	if dimension <= 0 {
		return rebuildInputs{}, fmt.Errorf("embedder dimension must be positive")
	}
	quantization, err := normalizeQuantization(options.Quantization)
	if err != nil {
		return rebuildInputs{}, err
	}
	reuseState := loadReuseState(ctx, options.ReuseFromPath, options.Embedder.Fingerprint(), dimension, quantization)

	return rebuildInputs{
		indexPath:    indexPath,
		normalized:   normalized,
		dimension:    dimension,
		quantization: quantization,
		reuseState:   reuseState,
	}, nil
}

func finalizeRebuild(ctx context.Context, db *sql.DB, tx *sql.Tx, metadata map[string]string, tmpPath, indexPath string) error {
	if err := upsertMetadata(ctx, tx, metadata); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit rebuild transaction: %w", err)
	}
	if err := runIntegrityChecks(ctx, db); err != nil {
		return err
	}
	if err := db.Close(); err != nil {
		return fmt.Errorf("close staging index: %w", err)
	}
	return replaceFile(tmpPath, indexPath)
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
	if err := ensureExistingIndex(indexPath); err != nil {
		return nil, err
	}

	normalizedAdded, removedRefs, err := normalizeUpdateInputs(added, removed)
	if err != nil {
		return nil, err
	}

	db, err := store.OpenReadWriteContext(ctx, indexPath)
	if err != nil {
		return nil, fmt.Errorf("open index %s: %w", indexPath, err)
	}
	defer func() { _ = db.Close() }()

	cfg, err := resolveUpdateSessionConfig(ctx, db, normalizedAdded, options)
	if err != nil {
		return nil, err
	}
	if len(normalizedAdded) > 0 {
		cfg.reuse = loadReuseState(
			ctx,
			indexPath,
			cfg.embedderFingerprint,
			cfg.dimension,
			cfg.quantization,
		)
		defer func() { _ = cfg.reuse.Close() }()
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
		EmbedderDimension:   cfg.dimension,
		EmbedderFingerprint: cfg.embedderFingerprint,
	}

	if err := deleteRemovedRecords(ctx, tx, removedRefs, result); err != nil {
		return nil, err
	}

	if len(normalizedAdded) > 0 {
		session, err := newIndexSession(ctx, tx, cfg)
		if err != nil {
			return nil, err
		}
		defer session.Close()

		for _, record := range normalizedAdded {
			if _, err := deleteRecord(ctx, tx, record.Ref); err != nil {
				return nil, err
			}
			stats, err := session.processRecord(ctx, record)
			if err != nil {
				return nil, err
			}
			result.ReusedChunkCount += stats.reusedChunks
			result.EmbeddedChunkCount += stats.embeddedChunks
			if stats.recordUnchanged {
				result.ReusedRecordCount++
			}
		}
	}

	if err := finalizeUpdate(ctx, tx, cfg, result); err != nil {
		return nil, err
	}
	return result, nil
}

func normalizeUpdateInputs(added []corpus.Record, removed []string) ([]corpus.Record, []string, error) {
	normalizedAdded, err := normalizeRecords(added)
	if err != nil {
		return nil, nil, err
	}
	removedRefs := normalizeRemovedRefs(removed)
	if err := validateUpdateInputs(normalizedAdded, removedRefs); err != nil {
		return nil, nil, err
	}
	return normalizedAdded, removedRefs, nil
}

func deleteRemovedRecords(ctx context.Context, tx *sql.Tx, refs []string, result *UpdateResult) error {
	for _, ref := range refs {
		deleted, err := deleteRecord(ctx, tx, ref)
		if err != nil {
			return err
		}
		if deleted {
			result.RemovedCount++
		}
	}
	return nil
}

func finalizeUpdate(ctx context.Context, tx *sql.Tx, cfg sessionConfig, result *UpdateResult) error {
	recordsNow, err := loadCurrentRecords(ctx, tx)
	if err != nil {
		return err
	}
	result.RecordCount = len(recordsNow)
	result.ContentFingerprint = corpus.Fingerprint(recordsNow)
	chunkCount, err := countChunks(ctx, tx)
	if err != nil {
		return err
	}
	result.ChunkCount = chunkCount

	now := time.Now().UTC().Format(time.RFC3339)
	createdAt, err := readMetadataValueOptional(ctx, tx, "created_at", now)
	if err != nil {
		return err
	}

	metadata := map[string]string{
		"schema_version":       schemaVersion,
		"embedder_dimension":   strconv.Itoa(cfg.dimension),
		"embedder_fingerprint": cfg.embedderFingerprint,
		"content_fingerprint":  result.ContentFingerprint,
		"quantization":         cfg.quantization,
		"created_at":           createdAt,
		"updated_at":           now,
	}
	if err := upsertMetadata(ctx, tx, metadata); err != nil {
		return err
	}
	if err := runIntegrityChecks(ctx, tx); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit update transaction: %w", err)
	}
	return nil
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
		Text:            query.Text,
		Limit:           query.Limit,
		Kinds:           query.Kinds,
		Embedder:        query.Embedder,
		Reranker:        query.Reranker,
		SearchDimension: query.SearchDimension,
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

func createSchema(ctx context.Context, db *sql.DB, dimension int, quantization string) error {
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
			id             INTEGER PRIMARY KEY AUTOINCREMENT,
			record_ref     TEXT NOT NULL,
			chunk_index    INTEGER NOT NULL,
			heading        TEXT NOT NULL,
			content        TEXT NOT NULL,
			context_prefix TEXT NOT NULL DEFAULT '',
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

func insertRecord(ctx context.Context, stmt *sql.Stmt, record corpus.Record) error {
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
	// embedder is required for newIndexSession. It may be nil when Update is
	// called with only removals, in which case the resolver still fills in
	// embedderFingerprint, dimension, and quantization from stored metadata.
	embedder            embed.Embedder
	embedderFingerprint string
	contextualizer      ChunkContextualizer
	reuse               *reuseState
	chunkOpts           chunk.Options
	dimension           int
	quantization        string
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
	tx             *sql.Tx
	embedder       embed.Embedder
	contextual     embed.ContextualEmbedder
	contextualizer ChunkContextualizer
	reuse          *reuseState
	chunkOpts      chunk.Options
	dimension      int
	quantization   string

	recordStmt *sql.Stmt
	chunkStmt  *sql.Stmt
	vectorStmt *sql.Stmt
	ftsStmt    *sql.Stmt
}

func newIndexSession(ctx context.Context, tx *sql.Tx, cfg sessionConfig) (*indexSession, error) {
	s := &indexSession{
		tx:             tx,
		embedder:       cfg.embedder,
		contextualizer: cfg.contextualizer,
		reuse:          cfg.reuse,
		chunkOpts:      cfg.chunkOpts,
		dimension:      cfg.dimension,
		quantization:   cfg.quantization,
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
record_ref, chunk_index, heading, content, context_prefix
) VALUES (?, ?, ?, ?, ?)`)
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

	if err := insertRecord(ctx, s.recordStmt, record); err != nil {
		return stats, err
	}

	sections := sectionsForRecord(record, s.chunkOpts)
	prefixes, err := runContextualizer(ctx, s.contextualizer, record, sections)
	if err != nil {
		return stats, fmt.Errorf("contextualize record %s: %w", record.Ref, err)
	}

	plan := planRecordReuse(record, storedRecordForReuse(ctx, s.reuse, record.Ref), sections, prefixes)
	stats.sections = len(sections)
	stats.reusedChunks = plan.reusedChunkCount
	stats.embeddedChunks = plan.embeddedChunkCount
	stats.recordUnchanged = plan.recordUnchanged

	if len(sections) == 0 {
		return stats, nil
	}

	texts := make([]string, 0, len(sections))
	for i, section := range sections {
		if _, ok := plan.reusedEmbeddings[plan.keys[i]]; ok {
			continue
		}
		texts = append(texts, contextualEmbeddingText(sectionPrefix(plan.prefixes, i), record.Title, section))
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

	if err := insertChunks(
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

func insertChunks(
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
		prefix := sectionPrefix(plan.prefixes, index)
		chunkResult, err := chunkStmt.ExecContext(ctx, record.Ref, index, section.Heading, section.Body, prefix)
		if err != nil {
			return fmt.Errorf("insert record %s chunk %d: %w", record.Ref, index, err)
		}
		chunkID, err := chunkResult.LastInsertId()
		if err != nil {
			return fmt.Errorf("read chunk id for %s chunk %d: %w", record.Ref, index, err)
		}
		ftsContent := section.Body
		if prefix != "" {
			ftsContent = prefix + "\n\n" + section.Body
		}
		if _, err := ftsStmt.ExecContext(ctx, chunkID, record.Title, section.Heading, ftsContent); err != nil {
			return fmt.Errorf("insert fts for %s chunk %d: %w", record.Ref, index, err)
		}
		if blob, ok := plan.reusedEmbeddings[plan.keys[index]]; ok {
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

// runContextualizer calls the configured contextualizer and returns a slice
// of per-section prefixes. A nil contextualizer or zero-length sections
// returns nil so downstream code can treat "no prefix" uniformly. The
// contextualizer must return exactly len(sections) prefixes; anything else
// is a bug in the implementation and is flagged so the build fails fast.
func runContextualizer(ctx context.Context, contextualizer ChunkContextualizer, record corpus.Record, sections []chunk.Section) ([]string, error) {
	if contextualizer == nil || len(sections) == 0 {
		return nil, nil
	}
	prefixes, err := contextualizer.ContextualizeChunks(ctx, record, sections)
	if err != nil {
		return nil, err
	}
	if len(prefixes) != len(sections) {
		return nil, fmt.Errorf("contextualizer returned %d prefixes for %d sections on %s",
			len(prefixes), len(sections), record.Ref)
	}
	// Normalize whitespace-only prefixes to "" at the source so the reuse
	// key, persisted column, FTS content, and embedding text all agree on
	// the empty case. Without this, a prefix of " " would skip embedding
	// augmentation (contextualEmbeddingText trims) but still persist a
	// non-empty context_prefix and flow into FTS, diverging silently.
	for i, p := range prefixes {
		if strings.TrimSpace(p) == "" {
			prefixes[i] = ""
		}
	}
	return prefixes, nil
}

// sectionPrefix returns prefixes[i] when prefixes is populated, or the empty
// string otherwise. Keeps the single "prefix || empty" branch out of every
// call site.
func sectionPrefix(prefixes []string, i int) string {
	if i < len(prefixes) {
		return prefixes[i]
	}
	return ""
}

// contextualEmbeddingText prepends the contextualizer's prefix to the
// standard embedding text so callers can feed the same string to both the
// vector and FTS arms. When prefix is empty the result matches the
// non-contextual path exactly. Whitespace normalization happens upstream
// in runContextualizer, so any non-empty prefix here is a real prefix.
func contextualEmbeddingText(prefix, title string, section chunk.Section) string {
	base := textForEmbedding(title, section)
	if prefix == "" {
		return base
	}
	return prefix + "\n\n" + base
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

func runIntegrityChecks(ctx context.Context, query queryContextRunner) error {
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

// matryoshkaPrefilterMultiplier scales the prefilter shortlist up from the
// caller-requested candidate limit so that good candidates that only the
// truncated prefix ranks lower still have a chance to survive into the
// full-dim rescore. The final LIMIT then trims back to the caller's count.
const matryoshkaPrefilterMultiplier = 3

// buildMatryoshkaSearchSQL builds the truncated-prefix prefilter + full-dim
// rescore query. The prefilter scans every chunks_vec row with cosine over
// the first searchDim floats; the rescore recomputes full-dim cosine on
// just the prefilter shortlist. When a kind filter is supplied it is
// applied inside the prefilter CTE so selective queries do not get starved
// by unrelated kinds dominating the truncated-prefix ranking.
//
// Performance note: this path does not use vec0's native MATCH kNN
// (sqlite-vec 0.1.6 does not offer an ANN index yet), so it is a linear
// scan even without Matryoshka. At the same dim, the MATCH path can still
// be faster because sqlite-vec exposes tighter SIMD for its kNN
// implementation. Setting SearchDimension is expected to pay off only when
// the truncated dim is meaningfully smaller than the stored dim AND the
// embedder was trained with MRL so the truncated prefix preserves
// ranking. For non-MRL embedders this is actively harmful.
func buildMatryoshkaSearchSQL(limit int, kinds []string, queryBlob []byte, searchDim int) (querySQL string, args []any, err error) {
	// Truncate the full-dim float32 query blob to the first searchDim floats.
	// The stored vectors are sliced in SQL via vec_slice, keeping the
	// representation on both sides identical. Guard the slice explicitly
	// so a misbehaving embedder that returns fewer dims than the snapshot
	// was built with surfaces a clear error instead of panicking on the
	// out-of-bounds reslice.
	needed := searchDim * 4
	if len(queryBlob) < needed {
		return "", nil, fmt.Errorf("query vector blob has %d bytes, need at least %d for SearchDimension=%d", len(queryBlob), needed, searchDim)
	}
	truncatedQuery := queryBlob[:needed]
	prefilterLimit := limit * matryoshkaPrefilterMultiplier

	var builder strings.Builder
	args = make([]any, 0, 5+len(kinds))

	hasKindFilter := len(kinds) > 0
	builder.WriteString(`
WITH prefilter AS (
  SELECT v.chunk_id, v.embedding
  FROM chunks_vec v`)
	if hasKindFilter {
		// Join through chunks → records inside the CTE so the kind
		// predicate prunes the prefilter search space itself. Applying
		// the kind filter after the truncated shortlist was selected
		// would silently drop candidates of the requested kind whenever
		// another kind dominated the top matryoshkaPrefilterMultiplier×
		// limit rows.
		builder.WriteString(`
  JOIN chunks c ON c.id = v.chunk_id
  JOIN records r ON r.ref = c.record_ref
  WHERE r.kind IN (`)
		for i, kind := range kinds {
			if i > 0 {
				builder.WriteString(", ")
			}
			builder.WriteString("?")
			args = append(args, strings.TrimSpace(kind))
		}
		builder.WriteString(")")
	}
	builder.WriteString(`
  ORDER BY vec_distance_cosine(vec_slice(v.embedding, 0, ?), ?)
  LIMIT ?
)
SELECT
  pf.chunk_id,
  r.ref,
  r.kind,
  r.title,
  r.source_ref,
  c.heading,
  c.content,
  r.metadata_json,
  vec_distance_cosine(pf.embedding, ?) AS distance
FROM prefilter pf
JOIN chunks c ON c.id = pf.chunk_id
JOIN records r ON r.ref = c.record_ref
ORDER BY distance ASC, r.ref ASC, c.chunk_index ASC
LIMIT ?`)
	args = append(args, searchDim, truncatedQuery, prefilterLimit, queryBlob, limit)
	return builder.String(), args, nil
}

func ensureExistingIndex(path string) error {
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

// migrateV2ToV3 adds the context_prefix column to chunks (idempotent: a
// prior run that already added it is a no-op) and bumps the stored
// schema_version metadata to "3". Safe to call on a snapshot that is
// already v3.
func migrateV2ToV3(ctx context.Context, db *sql.DB) error {
	hasPrefix, err := hasChunkColumn(ctx, db, "context_prefix")
	if err != nil {
		return fmt.Errorf("probe chunks.context_prefix: %w", err)
	}
	if !hasPrefix {
		if _, err := db.ExecContext(ctx,
			`ALTER TABLE chunks ADD COLUMN context_prefix TEXT NOT NULL DEFAULT ''`); err != nil {
			return fmt.Errorf("migrate v2 to v3: add chunks.context_prefix: %w", err)
		}
	}
	if _, err := db.ExecContext(ctx,
		`UPDATE metadata SET value = ? WHERE key = 'schema_version'`, schemaVersion); err != nil {
		return fmt.Errorf("migrate v2 to v3: bump schema_version: %w", err)
	}
	return nil
}

// hasChunkColumn probes PRAGMA table_info(chunks) for the named column.
// Used by read/reuse paths so v2 snapshots (built before context_prefix
// existed) keep working against v3-aware code.
func hasChunkColumn(ctx context.Context, q queryContextRunner, column string) (bool, error) {
	rows, err := q.QueryContext(ctx, `PRAGMA table_info(chunks)`)
	if err != nil {
		return false, err
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var (
			cid     int
			name    string
			typ     string
			notnull int
			dflt    sql.NullString
			pk      int
		)
		if err := rows.Scan(&cid, &name, &typ, &notnull, &dflt, &pk); err != nil {
			return false, err
		}
		if name == column {
			return true, rows.Err()
		}
	}
	return false, rows.Err()
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

func resolveUpdateSessionConfig(ctx context.Context, db *sql.DB, added []corpus.Record, options UpdateOptions) (sessionConfig, error) {
	schema, err := readMetadataValue(ctx, db, "schema_version")
	if err != nil {
		return sessionConfig{}, err
	}
	trimmedSchema := strings.TrimSpace(schema)
	if trimmedSchema == "2" {
		// Migrate v2 → v3 in place so existing indexes stay updatable
		// after this library upgrade. The migration is idempotent at
		// the column level (skipped if already applied) and commits
		// the schema_version bump in the same transaction as the data
		// edits the caller is about to make.
		if err := migrateV2ToV3(ctx, db); err != nil {
			return sessionConfig{}, err
		}
	} else if trimmedSchema != schemaVersion {
		return sessionConfig{}, fmt.Errorf("schema version mismatch: index=%q update=%q", trimmedSchema, schemaVersion)
	}

	embedderFingerprint, err := readMetadataValue(ctx, db, "embedder_fingerprint")
	if err != nil {
		return sessionConfig{}, err
	}
	dimensionValue, err := readMetadataValue(ctx, db, "embedder_dimension")
	if err != nil {
		return sessionConfig{}, err
	}
	dimension, err := strconv.Atoi(strings.TrimSpace(dimensionValue))
	if err != nil {
		return sessionConfig{}, fmt.Errorf("parse embedder_dimension %q: %w", dimensionValue, err)
	}

	quantization, err := readMetadataValueOptional(ctx, db, "quantization", store.QuantizationFloat32)
	if err != nil {
		return sessionConfig{}, err
	}
	quantization, err = normalizeQuantization(quantization)
	if err != nil {
		return sessionConfig{}, err
	}
	if strings.TrimSpace(options.Quantization) != "" {
		requested, err := normalizeQuantization(options.Quantization)
		if err != nil {
			return sessionConfig{}, err
		}
		if requested != quantization {
			return sessionConfig{}, fmt.Errorf("quantization mismatch: index=%q update=%q", quantization, requested)
		}
	}

	cfg := sessionConfig{
		embedderFingerprint: embedderFingerprint,
		contextualizer:      options.Contextualizer,
		dimension:           dimension,
		quantization:        quantization,
		chunkOpts: chunk.Options{
			MaxTokens:     options.MaxChunkTokens,
			OverlapTokens: options.ChunkOverlapTokens,
		},
	}

	if len(added) == 0 && options.Embedder == nil {
		return cfg, nil
	}
	if options.Embedder == nil {
		return sessionConfig{}, fmt.Errorf("update embedder is required when adding records")
	}

	updateFingerprint := options.Embedder.Fingerprint()
	if embedderFingerprint != updateFingerprint {
		return sessionConfig{}, fmt.Errorf("embedder fingerprint mismatch: index=%q update=%q", embedderFingerprint, updateFingerprint)
	}
	updateDimension, err := options.Embedder.Dimension(ctx)
	if err != nil {
		return sessionConfig{}, fmt.Errorf("resolve update embedder dimension: %w", err)
	}
	if updateDimension <= 0 {
		return sessionConfig{}, fmt.Errorf("embedder dimension must be positive")
	}
	if updateDimension != dimension {
		return sessionConfig{}, fmt.Errorf("embedder dimension mismatch: index=%d update=%d", dimension, updateDimension)
	}

	cfg.embedder = options.Embedder
	return cfg, nil
}

func deleteRecord(ctx context.Context, tx *sql.Tx, ref string) (bool, error) {
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

func loadCurrentRecords(ctx context.Context, query queryContextRunner) ([]corpus.Record, error) {
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
		record, err := scanRecord(rows)
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

func countChunks(ctx context.Context, query queryContextRunner) (int, error) {
	var chunkCount int
	if err := query.QueryRowContext(ctx, `SELECT COUNT(*) FROM chunks`).Scan(&chunkCount); err != nil {
		return 0, fmt.Errorf("count chunks: %w", err)
	}
	return chunkCount, nil
}

func upsertMetadata(ctx context.Context, tx *sql.Tx, metadata map[string]string) error {
	for key, value := range metadata {
		if _, err := tx.ExecContext(ctx, `
INSERT INTO metadata (key, value) VALUES (?, ?)
ON CONFLICT(key) DO UPDATE SET value = excluded.value`, key, value); err != nil {
			return fmt.Errorf("upsert metadata %s: %w", key, err)
		}
	}
	return nil
}

const (
	// candidateOverfetchMultiplier scales the caller's limit up before
	// merging hybrid search candidates, so rank-fusion has room to reorder.
	candidateOverfetchMultiplier = 5
	// minCandidatePool is the floor for the candidate shortlist, keeping
	// tiny limits (e.g., 1-2 hits) from starving rank fusion.
	minCandidatePool = 25
	// maxCandidatePool caps the shortlist to bound per-query cost.
	maxCandidatePool = 250
)

func candidateLimit(limit int) int {
	overfetch := limit * candidateOverfetchMultiplier
	switch {
	case overfetch < minCandidatePool:
		return minCandidatePool
	case overfetch > maxCandidatePool:
		return maxCandidatePool
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
