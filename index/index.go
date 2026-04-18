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

const (
	// schemaVersion is the current on-disk snapshot schema. Bumps require
	// a forward migration (see migrateV2ToV3, migrateV3ToV4, migrateV4ToV5)
	// and an extension of the OpenSnapshot accept-list.
	schemaVersion = "5"

	// prevSchemaVersion is the most recent prior schema that OpenSnapshot
	// still accepts read-only and that resolveUpdateSessionConfig can
	// migrate forward in place. Older snapshots are migrated through a
	// chain migration in the Update path.
	prevSchemaVersion = "4"

	// legacySchemaVersionV3 is an older schema that OpenSnapshot still
	// accepts read-only and that the Update path migrates forward via a
	// chain (v3 → v4 → v5). v3 carries chunks.context_prefix but not
	// chunks.parent_chunk_id.
	legacySchemaVersionV3 = "3"

	// legacySchemaVersionV2 is the earliest schema that Update can still
	// chain forward (v2 → v3 → v4 → v5). OpenSnapshot accepts it
	// read-only; the v2 → v3 step adds chunks.context_prefix and re-hashes
	// records under the new content_hash encoding.
	legacySchemaVersionV2 = "2"
)

// DefaultMaxChunkSections caps the number of heading-aware sections a
// single record can contribute to the index when the caller hasn't
// overridden it. 10,000 is generous for legitimate technical documents
// (few real specs exceed a few hundred headings) while still preventing
// a pathological or hostile body from expanding into millions of
// embedder calls + rows.
const DefaultMaxChunkSections = 10_000

// resolveMaxChunkSections maps a caller-provided MaxChunkSections knob
// onto the chunk.Options.MaxSections value: zero → safe default,
// negative → chunk-level "unlimited", positive → verbatim.
func resolveMaxChunkSections(user int) int {
	switch {
	case user < 0:
		return 0 // chunk.Options.MaxSections == 0 means unlimited
	case user == 0:
		return DefaultMaxChunkSections
	default:
		return user
	}
}

// schemaHasContextPrefix reports whether a snapshot at the given schema
// version carries the chunks.context_prefix column. The v2→v3 bump added
// that column; v3→v4 kept the same table shape (it re-hashed record
// content under a new encoding); v4→v5 added chunks.parent_chunk_id but
// left context_prefix in place. All of v3, v4, v5 carry the column, so
// the flag is fully determined by the accept-listed schema_version — read
// paths can cache it at handle-open time instead of probing PRAGMA
// table_info(chunks) on every query.
func schemaHasContextPrefix(schema string) bool {
	trimmed := strings.TrimSpace(schema)
	return trimmed == schemaVersion || trimmed == prevSchemaVersion || trimmed == legacySchemaVersionV3
}

// schemaHasParentChunkID reports whether a snapshot at the given schema
// version carries the chunks.parent_chunk_id column. The v4→v5 bump
// added that column; older versions lack it entirely. Read paths cache
// the flag at OpenSnapshot time so degraded behavior on legacy files
// (no parent walks) is decided once per handle.
func schemaHasParentChunkID(schema string) bool {
	return strings.TrimSpace(schema) == schemaVersion
}

// ErrUnsupportedSchemaVersion is returned when an operation encounters a
// snapshot whose schema_version is neither the current schema nor one the
// library knows how to migrate from. It is surfaced by OpenSnapshot and
// wrapped via fmt.Errorf with %w so callers can use errors.Is to detect it.
var ErrUnsupportedSchemaVersion = errors.New("unsupported snapshot schema version")

// ErrUpdateCommittedIntegrityCheckFailed signals that Update's transaction
// committed successfully — the record, chunk, and metadata changes are
// durable on disk — but the post-commit PRAGMA integrity_check /
// foreign_key_check reported corruption. The enclosing error wraps this
// sentinel via fmt.Errorf with %w so callers can use errors.Is to detect
// it. This case is non-retriable: re-running Update will not unroll the
// already-durable changes, and the underlying file likely needs operator
// inspection (see index/ARCHITECTURE.md). Contrast with plain errors
// returned by Update, which come from pre-commit failures and leave the
// file byte-identical to its pre-call state.
var ErrUpdateCommittedIntegrityCheckFailed = errors.New("update committed but post-commit integrity check failed")

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

	// MaxChunkSections caps how many sections any single record is allowed
	// to produce. A pathological Markdown body (e.g., 10^6 heading lines)
	// would otherwise translate into 10^6 embedder calls and 10^6
	// chunk/vector rows — a DoS vector for shared embedders. Zero means
	// DefaultMaxChunkSections; a negative value disables the cap for
	// callers who have their own upstream validation. When the cap is
	// exceeded, Rebuild returns an error wrapping chunk.ErrTooManySections
	// instead of silently admitting the record.
	MaxChunkSections int

	// Quantization controls the vector storage format. See the
	// store.Quantization* constants for the accept-listed values:
	// store.QuantizationFloat32 (default), store.QuantizationInt8 (4x
	// smaller, minor precision loss), and store.QuantizationBinary
	// (32x smaller via 1-bit sign packing, full-precision rescore on a
	// companion table preserves ranking).
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

	// MaxChunkSections mirrors BuildOptions.MaxChunkSections for the
	// incremental-update path. Zero → DefaultMaxChunkSections; negative
	// → no cap.
	MaxChunkSections int

	// Quantization, when provided, must match the existing index — see
	// the store.Quantization* constants (float32, int8, binary) for the
	// accept-listed values. Leaving it empty reuses the stored
	// quantization metadata.
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
			MaxSections:   resolveMaxChunkSections(options.MaxChunkSections),
		},
		dimension:    inputs.dimension,
		quantization: inputs.quantization,
	})
	if err != nil {
		return nil, err
	}
	defer session.Close()

	contentFingerprint, err := corpus.Fingerprint(inputs.normalized)
	if err != nil {
		return nil, fmt.Errorf("compute content fingerprint: %w", err)
	}
	result := &BuildResult{
		Path:                inputs.indexPath,
		RecordCount:         len(inputs.normalized),
		EmbedderDimension:   inputs.dimension,
		EmbedderFingerprint: options.Embedder.Fingerprint(),
		ContentFingerprint:  contentFingerprint,
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

	// Open the main update transaction before resolving session config so
	// the v2→v3 migration — run from inside resolveUpdateSessionConfig —
	// commits or rolls back atomically with the rest of the update. A
	// failure at any later step (embedder error, ctx cancellation, insert
	// failure) thus cannot leave the file permanently bumped to v3 with
	// no corresponding update applied.
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("begin update transaction: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()

	cfg, err := resolveUpdateSessionConfig(ctx, tx, normalizedAdded, options)
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

	if err := finalizeUpdate(ctx, db, tx, cfg, result); err != nil {
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

func finalizeUpdate(ctx context.Context, db *sql.DB, tx *sql.Tx, cfg sessionConfig, result *UpdateResult) error {
	// Read only the persisted digest inputs (ref, content_hash) that
	// corpus.FingerprintFromPairs consumes in this path. Loading full rows
	// here scales as O(N * row_size) per Update and dominates latency on
	// large corpora (see #37).
	pairs, err := loadCurrentRefHashes(ctx, tx)
	if err != nil {
		return err
	}
	result.RecordCount = len(pairs)
	result.ContentFingerprint, err = corpus.FingerprintFromPairs(pairs)
	if err != nil {
		return fmt.Errorf("compute content fingerprint: %w", err)
	}
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
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit update transaction: %w", err)
	}
	// PRAGMA integrity_check inside an uncommitted write transaction sees
	// staged dirty pages instead of the durable state, so commit-time
	// failure modes (WAL flush errors, partial writes, etc.) slip through.
	// Run the check on the db handle after Commit returns so the result
	// reflects what actually landed on disk. Unlike finalizeRebuild (which
	// validates post-commit before a final replaceFile, so a failure
	// leaves the original file untouched), finalizeUpdate has no staging
	// layer — the durable file is already mutated by the time we reach
	// this check. Wrap any failure in ErrUpdateCommittedIntegrityCheckFailed
	// so callers can distinguish "update rolled back cleanly" from
	// "update committed but the resulting file looks corrupt" and route
	// the latter to operator inspection instead of a naive retry.
	if err := runIntegrityChecks(ctx, db); err != nil {
		return fmt.Errorf("%w: %w", ErrUpdateCommittedIntegrityCheckFailed, err)
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
	if quantization == store.QuantizationBinary && dimension%8 != 0 {
		return fmt.Errorf("binary quantization requires embedder dimension divisible by 8, got %d", dimension)
	}

	vecTableStmt := buildChunksVecDDL(dimension, quantization)

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
			id              INTEGER PRIMARY KEY AUTOINCREMENT,
			record_ref      TEXT NOT NULL,
			chunk_index     INTEGER NOT NULL,
			heading         TEXT NOT NULL,
			content         TEXT NOT NULL,
			context_prefix  TEXT NOT NULL DEFAULT '',
			parent_chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
			FOREIGN KEY (record_ref) REFERENCES records(ref) ON DELETE CASCADE
		)`,
		vecTableStmt,
		`CREATE TABLE metadata (
			key   TEXT PRIMARY KEY,
			value TEXT NOT NULL
		)`,
		`CREATE VIRTUAL TABLE fts_chunks USING fts5(title, heading, content)`,
		`CREATE INDEX idx_records_kind ON records(kind)`,
		`CREATE INDEX idx_chunks_record_ref ON chunks(record_ref)`,
		`CREATE INDEX idx_chunks_parent ON chunks(parent_chunk_id)`,
		// Same-record lineage invariant for parent_chunk_id (#16). SQLite
		// CHECK constraints cannot reference other rows, so the
		// invariant is enforced via BEFORE INSERT / BEFORE UPDATE
		// triggers that RAISE(ABORT) when a chunk's parent points at a
		// row in a different record. This blocks cross-record lineage
		// at the storage layer so ExpandContext's parent walks cannot
		// leak content across records, and so DELETE FROM records
		// cascades cannot orphan chunks_vec / fts_chunks rows in an
		// unrelated record.
		chunksParentSameRecordInsertTriggerSQL,
		chunksParentSameRecordUpdateTriggerSQL,
	}
	if quantization == store.QuantizationBinary {
		// Companion full-precision column for the rescore pass. The
		// primary chunks_vec holds sign-packed bits for the hamming
		// prefilter, and chunks_vec_full holds the float32 blob
		// sqlite-vec consumes as a cosine-distance argument.
		statements = append(statements, `CREATE TABLE chunks_vec_full (
			chunk_id  INTEGER PRIMARY KEY,
			embedding BLOB NOT NULL,
			FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
		)`)
	}

	for _, statement := range statements {
		if _, err := db.ExecContext(ctx, statement); err != nil {
			return err
		}
	}
	return nil
}

func buildChunksVecDDL(dimension int, quantization string) string {
	switch quantization {
	case store.QuantizationInt8:
		return fmt.Sprintf(`CREATE VIRTUAL TABLE chunks_vec USING vec0(
			chunk_id INTEGER PRIMARY KEY,
			embedding int8[%d] distance_metric=cosine
		)`, dimension)
	case store.QuantizationBinary:
		// sqlite-vec infers hamming distance for bit[N] columns.
		return fmt.Sprintf(`CREATE VIRTUAL TABLE chunks_vec USING vec0(
			chunk_id INTEGER PRIMARY KEY,
			embedding bit[%d]
		)`, dimension)
	default:
		return fmt.Sprintf(`CREATE VIRTUAL TABLE chunks_vec USING vec0(
			chunk_id INTEGER PRIMARY KEY,
			embedding float[%d] distance_metric=cosine
		)`, dimension)
	}
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
// for the per-record write path. Both Rebuild and Update delegate to it so
// the two write paths share the same invariants.
type indexSession struct {
	tx             *sql.Tx
	embedder       embed.Embedder
	contextual     embed.ContextualEmbedder
	contextualizer ChunkContextualizer
	reuse          *reuseState
	chunkOpts      chunk.Options
	dimension      int
	quantization   string

	recordStmt     *sql.Stmt
	chunkStmt      *sql.Stmt
	vectorStmt     *sql.Stmt
	vectorFullStmt *sql.Stmt // nil unless quantization == binary
	ftsStmt        *sql.Stmt
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
	switch cfg.quantization {
	case store.QuantizationInt8:
		vectorInsertSQL = `INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, vec_int8(?))`
	case store.QuantizationBinary:
		vectorInsertSQL = `INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, vec_bit(?))`
	}
	s.vectorStmt, err = tx.PrepareContext(ctx, vectorInsertSQL)
	if err != nil {
		s.Close()
		return nil, fmt.Errorf("prepare vector insert: %w", err)
	}

	if cfg.quantization == store.QuantizationBinary {
		s.vectorFullStmt, err = tx.PrepareContext(ctx,
			`INSERT INTO chunks_vec_full (chunk_id, embedding) VALUES (?, ?)`)
		if err != nil {
			s.Close()
			return nil, fmt.Errorf("prepare full-precision vector insert: %w", err)
		}
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
	for _, stmt := range []**sql.Stmt{&s.recordStmt, &s.chunkStmt, &s.vectorStmt, &s.vectorFullStmt, &s.ftsStmt} {
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

	sections, err := sectionsForRecord(record, s.chunkOpts)
	if err != nil {
		return stats, err
	}
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

	if err := s.insertChunks(ctx, record, plan, sections, vectors); err != nil {
		return stats, err
	}

	return stats, nil
}

func (s *indexSession) insertChunks(
	ctx context.Context,
	record corpus.Record,
	plan recordReusePlan,
	sections []chunk.Section,
	vectors [][]float64,
) error {
	newVectorIndex := 0
	for index, section := range sections {
		prefix := sectionPrefix(plan.prefixes, index)
		chunkResult, err := s.chunkStmt.ExecContext(ctx, record.Ref, index, section.Heading, section.Body, prefix)
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
		if _, err := s.ftsStmt.ExecContext(ctx, chunkID, record.Title, section.Heading, ftsContent); err != nil {
			return fmt.Errorf("insert fts for %s chunk %d: %w", record.Ref, index, err)
		}
		if reused, ok := plan.reusedEmbeddings[plan.keys[index]]; ok {
			primary, full, err := reusedBlobsFor(s.quantization, reused)
			if err != nil {
				return fmt.Errorf("prepare reused embedding for %s chunk %d: %w", record.Ref, index, err)
			}
			if err := insertVectorBlobs(ctx, chunkID, primary, full, s.vectorStmt, s.vectorFullStmt); err != nil {
				return fmt.Errorf("insert reused embedding for %s chunk %d: %w", record.Ref, index, err)
			}
			continue
		}
		if newVectorIndex >= len(vectors) {
			return fmt.Errorf("record %s chunk %d is missing a new embedding", record.Ref, index)
		}
		if len(vectors[newVectorIndex]) != s.dimension {
			return fmt.Errorf("record %s section %d embedding has dimension %d, want %d", record.Ref, index, len(vectors[newVectorIndex]), s.dimension)
		}
		primary, full, err := newBlobsFor(s.quantization, vectors[newVectorIndex])
		if err != nil {
			return fmt.Errorf("encode embedding for %s chunk %d: %w", record.Ref, index, err)
		}
		if err := insertVectorBlobs(ctx, chunkID, primary, full, s.vectorStmt, s.vectorFullStmt); err != nil {
			return fmt.Errorf("insert embedding for %s chunk %d: %w", record.Ref, index, err)
		}
		newVectorIndex++
	}
	return nil
}

// newBlobsFor encodes a newly-embedded vector into the primary (vec0) blob
// and, when binary quantization is active, the full-precision companion
// blob consumed by the rescore pass.
func newBlobsFor(quantization string, vector []float64) (primary, full []byte, err error) {
	primary, err = encodeVector(vector, quantization)
	if err != nil {
		return nil, nil, err
	}
	if quantization == store.QuantizationBinary {
		full, err = store.EncodeVectorBlob(vector)
		if err != nil {
			return nil, nil, err
		}
	}
	return primary, full, nil
}

// reusedBlobsFor splits a reused embedding blob into the primary and full
// columns. For float32 and int8 modes the reused blob is already the
// primary blob. For binary mode the reused blob is the full-precision
// float32 companion, and the primary bit blob is re-derived from it.
func reusedBlobsFor(quantization string, reused []byte) (primary, full []byte, err error) {
	if quantization != store.QuantizationBinary {
		return reused, nil, nil
	}
	vector, err := store.DecodeVectorBlob(reused)
	if err != nil {
		return nil, nil, err
	}
	primary, err = store.EncodeVectorBlobBinary(vector)
	if err != nil {
		return nil, nil, err
	}
	return primary, reused, nil
}

func insertVectorBlobs(ctx context.Context, chunkID int64, primary, full []byte, vectorStmt, vectorFullStmt *sql.Stmt) error {
	if _, err := vectorStmt.ExecContext(ctx, chunkID, primary); err != nil {
		return err
	}
	if full != nil {
		if vectorFullStmt == nil {
			return fmt.Errorf("missing prepared statement for full-precision companion insert")
		}
		if _, err := vectorFullStmt.ExecContext(ctx, chunkID, full); err != nil {
			return err
		}
	}
	return nil
}

func sectionsForRecord(record corpus.Record, opts chunk.Options) ([]chunk.Section, error) {
	switch record.BodyFormat {
	case corpus.FormatPlaintext:
		text := strings.TrimSpace(record.BodyText)
		if text == "" {
			return nil, nil
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
			sections = result
		}
		if opts.MaxSections > 0 && len(sections) > opts.MaxSections {
			return nil, fmt.Errorf("record %q: %w: plaintext token-budget split produced %d sections, limit is %d",
				record.Ref, chunk.ErrTooManySections, len(sections), opts.MaxSections)
		}
		return sections, nil
	default:
		// Always route through MarkdownWithOptions so the parser-side
		// MaxSections guard fires during section emission, not after
		// the full list is materialized. MarkdownWithOptions with
		// MaxTokens == 0 skips the token-budget pass automatically.
		sections, err := chunk.MarkdownWithOptions(record.Title, record.BodyText, opts)
		if err != nil {
			return nil, fmt.Errorf("record %q: %w", record.Ref, err)
		}
		return sections, nil
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

// runIntegrityChecksInjectFault is a test-only seam that runs before the
// real integrity checks. Production leaves it nil. Tests set it to
// simulate post-commit corruption and drive the
// ErrUpdateCommittedIntegrityCheckFailed error path without needing
// filesystem-level fault injection.
var runIntegrityChecksInjectFault func() error

func runIntegrityChecks(ctx context.Context, query queryContextRunner) error {
	if runIntegrityChecksInjectFault != nil {
		if err := runIntegrityChecksInjectFault(); err != nil {
			return err
		}
	}
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
	if err := rows.Err(); err != nil {
		return err
	}
	return checkChunksVecFullCompleteness(ctx, query)
}

// checkChunksVecFullCompleteness is a no-op for non-binary snapshots; for
// binary snapshots it verifies that every chunk has a companion
// chunks_vec_full row and vice versa. Without this check a missing
// companion row would silently drop that chunk from binary search (the
// rescore join is an inner join) and from Sections(IncludeEmbeddings).
func checkChunksVecFullCompleteness(ctx context.Context, query queryContextRunner) error {
	has, err := hasTable(ctx, query, "chunks_vec_full")
	if err != nil {
		return fmt.Errorf("probe chunks_vec_full: %w", err)
	}
	if !has {
		return nil
	}
	var missingCompanion int
	row := query.QueryRowContext(ctx, `
SELECT COUNT(*)
FROM chunks c
LEFT JOIN chunks_vec_full v ON v.chunk_id = c.id
WHERE v.chunk_id IS NULL`)
	if err := row.Scan(&missingCompanion); err != nil {
		return fmt.Errorf("count missing chunks_vec_full rows: %w", err)
	}
	if missingCompanion != 0 {
		return fmt.Errorf("%d chunk(s) missing from chunks_vec_full; binary search would silently drop them", missingCompanion)
	}
	var orphanCompanion int
	row = query.QueryRowContext(ctx, `
SELECT COUNT(*)
FROM chunks_vec_full v
LEFT JOIN chunks c ON c.id = v.chunk_id
WHERE c.id IS NULL`)
	if err := row.Scan(&orphanCompanion); err != nil {
		return fmt.Errorf("count orphan chunks_vec_full rows: %w", err)
	}
	if orphanCompanion != 0 {
		return fmt.Errorf("%d orphan row(s) in chunks_vec_full have no matching chunk", orphanCompanion)
	}
	return nil
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

// Search-path overfetch multipliers. Each search stage that has a "this
// pre-pass might miss the right hit" property scales the caller's limit up
// before doing its work, then trims back down at the final stage. Tune them
// here, in one place, so future calibration touches a single block instead
// of three scattered consts:
//
//   - matryoshkaPrefilterMultiplier: the truncated-prefix prefilter ranks
//     by a degraded representation, so candidates that would survive the
//     full-dim rescore can sit just outside the top-N. Overfetch by 3× and
//     let the rescore reorder.
//   - binaryRescoreMultiplier: the 1-bit hamming prefilter is even coarser
//     than the truncated-prefix prefilter, so the full-precision cosine
//     rescore needs the same kind of slack. Overfetch by 3×.
//   - candidateOverfetchMultiplier: hybrid search merges two independent
//     candidate sets (vector + FTS) and then rank-fuses them. Each arm
//     needs more candidates than the caller asked for so the fusion stage
//     has room to reorder before the final LIMIT. Overfetch by 5× (with a
//     [minCandidatePool, maxCandidatePool] floor/ceiling so tiny limits
//     don't starve fusion and huge limits don't blow up per-query cost).
const (
	matryoshkaPrefilterMultiplier = 3
	binaryRescoreMultiplier       = 3
	candidateOverfetchMultiplier  = 5
)

func buildSearchSQL(limit int, kinds []string, queryBlob []byte, quantization string) (querySQL string, args []any) {
	var builder strings.Builder
	args = make([]any, 0, 4+len(kinds))

	hasKindFilter := len(kinds) > 0
	matchExpr := "?"
	sliceExpr := "?"
	if quantization == store.QuantizationInt8 {
		matchExpr = "vec_int8(?)"
		sliceExpr = "vec_int8(?)"
	}

	if hasKindFilter {
		// sqlite-vec's vec0 MATCH + k = N fixes the candidate pool at N
		// *before* any joined predicate can reject rows, so filtering the
		// shortlist by an externally-joined r.kind after the fact silently
		// starves the search whenever the requested kinds are a minority
		// of the corpus. Mirror buildMatryoshkaSearchSQL: switch to a
		// brute-force vec_distance_cosine scan constrained inside the CTE
		// by the kind predicate so only matching chunks are ever ranked.
		// sqlite-vec 0.1.6 does not ship a true ANN index, so the default
		// MATCH path is also a linear scan under the hood — the cost
		// delta here is a smaller constant, not an algorithmic change.
		builder.WriteString(`
WITH vector_hits AS (
  SELECT v.chunk_id,
         vec_distance_cosine(v.embedding, `)
		builder.WriteString(sliceExpr)
		builder.WriteString(`) AS distance
  FROM chunks_vec v
  JOIN chunks c ON c.id = v.chunk_id
  JOIN records r ON r.ref = c.record_ref
  WHERE r.kind IN (`)
		// Arg order must match placeholder order in the SQL: first the
		// query blob consumed by vec_distance_cosine, then the kinds in
		// the WHERE r.kind IN (...) list, then the inner LIMIT.
		args = append(args, queryBlob)
		for i, kind := range kinds {
			if i > 0 {
				builder.WriteString(", ")
			}
			builder.WriteString("?")
			args = append(args, strings.TrimSpace(kind))
		}
		builder.WriteString(`)
  ORDER BY distance
  LIMIT ?
)`)
		args = append(args, limit)
	} else {
		builder.WriteString(fmt.Sprintf(`
WITH vector_hits AS (
  SELECT chunk_id, distance
  FROM chunks_vec
  WHERE embedding MATCH %s AND k = ?
  ORDER BY distance
)`, matchExpr))
		args = append(args, queryBlob, limit)
	}

	builder.WriteString(`
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
ORDER BY vh.distance ASC, r.ref ASC, c.chunk_index ASC
LIMIT ?`)
	args = append(args, limit)
	return builder.String(), args
}

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

// buildBinarySearchSQL builds the two-stage binary search: a hamming-distance
// prefilter over chunks_vec's bit[N] column, followed by a full-precision
// cosine rescore against chunks_vec_full. The final rescored distance is
// cosine, so downstream score normalization and fusion stay consistent with
// the float32 and int8 paths.
func buildBinarySearchSQL(limit int, kinds []string, bitQueryBlob, floatQueryBlob []byte) (querySQL string, args []any) {
	var builder strings.Builder
	args = make([]any, 0, 4+len(kinds))

	prefilterLimit := limit * binaryRescoreMultiplier

	if len(kinds) > 0 {
		// sqlite-vec's vec0 `MATCH vec_bit(?) AND k = N` caps the hamming
		// shortlist at N *before* any externally-joined predicate runs, so
		// filtering by r.kind on the outer query silently starves the
		// rescore stage whenever the requested kinds are a minority of
		// the corpus. Keep the two-stage structure — cheap 1-bit hamming
		// prefilter, full-precision cosine rescore — but swap the vec0
		// MATCH kNN for a brute-force hamming-distance scan whose CTE
		// can join `chunks` + `records` and enforce the kind predicate
		// up front. This preserves the binary path's cost benefit for
		// broad kind filters (still only prefilterLimit rows hit the
		// float rescore) while guaranteeing correctness.
		builder.WriteString(`
WITH prefilter AS (
  SELECT v.chunk_id
  FROM chunks_vec v
  JOIN chunks c ON c.id = v.chunk_id
  JOIN records r ON r.ref = c.record_ref
  WHERE r.kind IN (`)
		// Placeholder order: kinds in WHERE, then bitQueryBlob in the
		// ORDER BY hamming, then prefilterLimit, then floatQueryBlob for
		// the rescore stage, then the outer LIMIT.
		for i, kind := range kinds {
			if i > 0 {
				builder.WriteString(", ")
			}
			builder.WriteString("?")
			args = append(args, strings.TrimSpace(kind))
		}
		builder.WriteString(`)
  ORDER BY vec_distance_hamming(v.embedding, vec_bit(?))
  LIMIT ?
),
rescored AS (
  SELECT pf.chunk_id,
         vec_distance_cosine(vf.embedding, ?) AS distance
  FROM prefilter pf
  JOIN chunks_vec_full vf ON vf.chunk_id = pf.chunk_id
)
SELECT
  rs.chunk_id,
  r.ref,
  r.kind,
  r.title,
  r.source_ref,
  c.heading,
  c.content,
  r.metadata_json,
  rs.distance
FROM rescored rs
JOIN chunks c ON c.id = rs.chunk_id
JOIN records r ON r.ref = c.record_ref
ORDER BY rs.distance ASC, r.ref ASC, c.chunk_index ASC
LIMIT ?`)
		args = append(args, bitQueryBlob, prefilterLimit, floatQueryBlob, limit)
		return builder.String(), args
	}

	builder.WriteString(`
WITH prefilter AS (
  SELECT chunk_id
  FROM chunks_vec
  WHERE embedding MATCH vec_bit(?) AND k = ?
  ORDER BY distance
),
rescored AS (
  SELECT pf.chunk_id,
         vec_distance_cosine(vf.embedding, ?) AS distance
  FROM prefilter pf
  JOIN chunks_vec_full vf ON vf.chunk_id = pf.chunk_id
)
SELECT
  rs.chunk_id,
  r.ref,
  r.kind,
  r.title,
  r.source_ref,
  c.heading,
  c.content,
  r.metadata_json,
  rs.distance
FROM rescored rs
JOIN chunks c ON c.id = rs.chunk_id
JOIN records r ON r.ref = c.record_ref
ORDER BY rs.distance ASC, r.ref ASC, c.chunk_index ASC
LIMIT ?`)
	args = append(args, bitQueryBlob, prefilterLimit, floatQueryBlob, limit)
	return builder.String(), args
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

// hasTable probes sqlite_master for a table named `name`. Used to keep
// quantization-agnostic code paths from crashing against snapshots that do
// not carry the binary-only chunks_vec_full companion table.
func hasTable(ctx context.Context, q queryContextRunner, name string) (bool, error) {
	var count int
	row := q.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?`, name)
	if err := row.Scan(&count); err != nil {
		return false, err
	}
	return count > 0, nil
}

// migrateSchemaToCurrent brings the snapshot on tx forward to schemaVersion
// using whichever chain of migrations applies. Every step runs against the
// caller's transaction so the schema bump commits atomically with the rest
// of Update — any later failure rolls everything back, preventing a
// partial upgrade from stranding the file at an intermediate version.
func migrateSchemaToCurrent(ctx context.Context, tx *sql.Tx) error {
	schema, err := readMetadataValue(ctx, tx, "schema_version")
	if err != nil {
		return err
	}
	switch strings.TrimSpace(schema) {
	case legacySchemaVersionV2:
		// v2 predates context_prefix, the new content_hash encoding, and
		// parent_chunk_id; chain v2→v3→v4→v5 in one transaction.
		if err := migrateV2ToV3(ctx, tx); err != nil {
			return err
		}
		if err := migrateV3ToV4(ctx, tx); err != nil {
			return err
		}
		return migrateV4ToV5(ctx, tx)
	case legacySchemaVersionV3:
		// v3 carries context_prefix already; chain v3→v4→v5 (re-hash
		// records under #39's encoding, then add parent_chunk_id).
		if err := migrateV3ToV4(ctx, tx); err != nil {
			return err
		}
		return migrateV4ToV5(ctx, tx)
	case prevSchemaVersion:
		// v4 carries records under the current content_hash encoding;
		// only chunks.parent_chunk_id needs adding (#16).
		return migrateV4ToV5(ctx, tx)
	case schemaVersion:
		return nil
	default:
		return fmt.Errorf("%w: index=%q update=%q", ErrUnsupportedSchemaVersion, strings.TrimSpace(schema), schemaVersion)
	}
}

// migrateV2ToV3InjectFault is a test-only seam that runs between the ALTER
// and the metadata bump inside migrateV2ToV3. Production leaves it nil.
// Tests set it to simulate a crash mid-migration and assert that the
// enclosing transaction rolls back cleanly.
var migrateV2ToV3InjectFault func() error

// migrateV2ToV3 adds the context_prefix column to chunks and bumps the
// stored schema_version metadata to "3" on the provided transaction, so
// the migration commits atomically with whatever surrounding work the
// caller is running (today: Update's main tx, chained with migrateV3ToV4
// when starting from v2). A crash between the ALTER and the UPDATE — or
// a failure anywhere in the caller's transaction afterwards — rolls back
// both statements instead of leaving the file with the new column but
// schema_version="2". The column-add is idempotent (skipped if a prior
// run already applied it), so the helper is safe if re-entered on a v3
// snapshot. It must not be called on v4 or later: the schema_version
// bump is hard-coded to prevSchemaVersion and would effectively
// downgrade metadata. migrateSchemaToCurrent enforces the v2-only
// entry point.
func migrateV2ToV3(ctx context.Context, tx *sql.Tx) error {
	hasPrefix, err := hasChunkColumn(ctx, tx, "context_prefix")
	if err != nil {
		return fmt.Errorf("probe chunks.context_prefix: %w", err)
	}
	if !hasPrefix {
		if _, err := tx.ExecContext(ctx,
			`ALTER TABLE chunks ADD COLUMN context_prefix TEXT NOT NULL DEFAULT ''`); err != nil {
			return fmt.Errorf("migrate v2 to v3: add chunks.context_prefix: %w", err)
		}
	}
	if migrateV2ToV3InjectFault != nil {
		if err := migrateV2ToV3InjectFault(); err != nil {
			return fmt.Errorf("migrate v2 to v3: injected fault: %w", err)
		}
	}
	if _, err := tx.ExecContext(ctx,
		`UPDATE metadata SET value = ? WHERE key = 'schema_version'`, legacySchemaVersionV3); err != nil {
		return fmt.Errorf("migrate v2 to v3: bump schema_version: %w", err)
	}
	return nil
}

// migrateV3ToV4InjectFault is a test-only seam that runs between the
// per-row re-hash and the metadata bump inside migrateV3ToV4. Production
// leaves it nil. Tests set it to simulate a crash mid-migration and assert
// the enclosing transaction rolls back cleanly.
var migrateV3ToV4InjectFault func() error

// migrateV3ToV4BatchSize bounds how many records migrateV3ToV4 reads into
// memory at once. Keeping it small means peak overhead is O(batchSize *
// row_size) rather than O(N * row_size) — important for multi-million-
// record snapshots where the whole corpus could otherwise blow up heap
// mid-migration. Chosen as a pragmatic balance between SELECT overhead
// and bounded memory; overridable at test time.
var migrateV3ToV4BatchSize = 1000

// migrateV3ToV4 re-hashes every record's content_hash under the new
// injective encoding (see corpus.HashRecord #39) and bumps the stored
// schema_version metadata to "4". Records are streamed in batches so
// peak memory is O(batch), not O(N): SQLite won't let us issue UPDATEs
// on the same connection while a SELECT cursor is open on it, so each
// batch reads-then-updates sequentially and advances via ref > last.
// The content_fingerprint metadata key is left for finalizeUpdate to
// rewrite at commit time — it runs off the new content_hash values via
// loadCurrentRefHashes and picks up the post-migration state
// automatically.
//
// The ALTER-free shape means this migration doesn't touch table schema;
// it only rewrites row content and bumps metadata. Idempotent against a
// snapshot already at v4 (a no-op second pass re-hashes to the same
// values, then the metadata update is a same-value re-apply).
func migrateV3ToV4(ctx context.Context, tx *sql.Tx) error {
	var lastRef string
	for {
		batch, err := migrateV3ToV4ReadBatch(ctx, tx, lastRef, migrateV3ToV4BatchSize)
		if err != nil {
			return err
		}
		if len(batch) == 0 {
			break
		}
		for _, r := range batch {
			if _, err := tx.ExecContext(ctx,
				`UPDATE records SET content_hash = ? WHERE ref = ?`, r.hash, r.ref); err != nil {
				return fmt.Errorf("migrate v3 to v4: rewrite content_hash for %q: %w", r.ref, err)
			}
		}
		lastRef = batch[len(batch)-1].ref
		if len(batch) < migrateV3ToV4BatchSize {
			break
		}
	}

	if migrateV3ToV4InjectFault != nil {
		if err := migrateV3ToV4InjectFault(); err != nil {
			return fmt.Errorf("migrate v3 to v4: injected fault: %w", err)
		}
	}

	if _, err := tx.ExecContext(ctx,
		`UPDATE metadata SET value = ? WHERE key = 'schema_version'`, prevSchemaVersion); err != nil {
		return fmt.Errorf("migrate v3 to v4: bump schema_version: %w", err)
	}
	return nil
}

// chunksParentSameRecordInsertTriggerSQL and
// chunksParentSameRecordUpdateTriggerSQL guard the v5 lineage column
// against cross-record parent assignments. SQLite CHECK constraints
// cannot reference other rows, so the same-record invariant is enforced
// via BEFORE triggers that RAISE(ABORT) when NEW.parent_chunk_id points
// at a row in a different record. The trigger is a no-op when
// parent_chunk_id IS NULL (the only shape PR-A produces), so it adds
// zero overhead until PR-B's chunk.Policy framework starts emitting
// parents.
const (
	chunksParentSameRecordInsertTriggerSQL = `
CREATE TRIGGER chunks_parent_same_record_insert
BEFORE INSERT ON chunks
WHEN NEW.parent_chunk_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'chunks.parent_chunk_id must reference a chunk in the same record_ref')
    WHERE NEW.record_ref != (SELECT record_ref FROM chunks WHERE id = NEW.parent_chunk_id);
END`

	chunksParentSameRecordUpdateTriggerSQL = `
CREATE TRIGGER chunks_parent_same_record_update
BEFORE UPDATE OF parent_chunk_id ON chunks
WHEN NEW.parent_chunk_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'chunks.parent_chunk_id must reference a chunk in the same record_ref')
    WHERE NEW.record_ref != (SELECT record_ref FROM chunks WHERE id = NEW.parent_chunk_id);
END`
)

// migrateV4ToV5InjectFault is a test-only seam that runs between the
// ALTER + index-create and the metadata bump inside migrateV4ToV5.
// Production leaves it nil. Tests set it to simulate a crash
// mid-migration and assert the enclosing transaction rolls back cleanly.
var migrateV4ToV5InjectFault func() error

// migrateV4ToV5 adds the chunks.parent_chunk_id column + idx_chunks_parent
// index and bumps the stored schema_version metadata to "5". The column
// is NULLable with a CASCADE FK back to chunks(id), matching how
// chunks_vec_full keys to chunks for binary quantization. NULL is the
// only legal value until a Policy emits parent topology (#16 PR-B), so
// existing v4 row content is byte-equivalent under v5 read paths.
//
// All work runs against the caller's transaction so the schema bump
// commits atomically with whatever surrounding work the caller is
// running (today: Update's main tx, chained with migrateV3ToV4 and
// migrateV2ToV3 when starting from v3 or v2). A crash between the
// ALTER, the CREATE INDEX, and the UPDATE metadata — or a failure
// anywhere in the caller's transaction afterwards — rolls back all
// statements instead of leaving the file partially upgraded. The
// column-add and index-create are idempotent (skipped if a prior run
// already applied them), so the helper is safe if re-entered on a v5
// snapshot.
func migrateV4ToV5(ctx context.Context, tx *sql.Tx) error {
	hasParent, err := hasChunkColumn(ctx, tx, "parent_chunk_id")
	if err != nil {
		return fmt.Errorf("probe chunks.parent_chunk_id: %w", err)
	}
	if !hasParent {
		if _, err := tx.ExecContext(ctx,
			`ALTER TABLE chunks ADD COLUMN parent_chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE`); err != nil {
			return fmt.Errorf("migrate v4 to v5: add chunks.parent_chunk_id: %w", err)
		}
	}
	if _, err := tx.ExecContext(ctx,
		`CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id)`); err != nil {
		return fmt.Errorf("migrate v4 to v5: create idx_chunks_parent: %w", err)
	}
	// Same-record lineage invariant: install the same triggers a fresh
	// v5 schema gets so upgraded snapshots cannot accept cross-record
	// parent_chunk_id values from any later writer. CREATE TRIGGER does
	// not support IF NOT EXISTS in the SQL standard but SQLite does, so
	// the migration is idempotent if re-entered on a v5 file.
	if _, err := tx.ExecContext(ctx,
		strings.Replace(chunksParentSameRecordInsertTriggerSQL,
			"CREATE TRIGGER ", "CREATE TRIGGER IF NOT EXISTS ", 1)); err != nil {
		return fmt.Errorf("migrate v4 to v5: create chunks_parent_same_record_insert trigger: %w", err)
	}
	if _, err := tx.ExecContext(ctx,
		strings.Replace(chunksParentSameRecordUpdateTriggerSQL,
			"CREATE TRIGGER ", "CREATE TRIGGER IF NOT EXISTS ", 1)); err != nil {
		return fmt.Errorf("migrate v4 to v5: create chunks_parent_same_record_update trigger: %w", err)
	}
	if migrateV4ToV5InjectFault != nil {
		if err := migrateV4ToV5InjectFault(); err != nil {
			return fmt.Errorf("migrate v4 to v5: injected fault: %w", err)
		}
	}
	if _, err := tx.ExecContext(ctx,
		`UPDATE metadata SET value = ? WHERE key = 'schema_version'`, schemaVersion); err != nil {
		return fmt.Errorf("migrate v4 to v5: bump schema_version: %w", err)
	}
	return nil
}

// migrateV3ToV4ReadBatch reads up to limit records after lastRef (exclusive)
// and returns their (ref, new-algo content_hash) pairs. Factored out so
// the SELECT cursor is fully consumed and closed before migrateV3ToV4
// starts issuing UPDATEs on the same transaction.
func migrateV3ToV4ReadBatch(ctx context.Context, tx *sql.Tx, lastRef string, limit int) ([]struct{ ref, hash string }, error) {
	rows, err := tx.QueryContext(ctx, `
SELECT ref, kind, title, source_ref, body_format, body_text, content_hash, metadata_json
FROM records
WHERE ref > ?
ORDER BY ref ASC
LIMIT ?`, lastRef, limit)
	if err != nil {
		return nil, fmt.Errorf("migrate v3 to v4: query records: %w", err)
	}
	defer func() { _ = rows.Close() }()

	batch := make([]struct{ ref, hash string }, 0, limit)
	for rows.Next() {
		record, err := scanRecord(rows)
		if err != nil {
			return nil, fmt.Errorf("migrate v3 to v4: scan record: %w", err)
		}
		batch = append(batch, struct{ ref, hash string }{
			ref:  record.Ref,
			hash: corpus.HashRecord(record),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("migrate v3 to v4: iterate records: %w", err)
	}
	return batch, nil
}

// hasChunkColumn probes PRAGMA table_info(chunks) for the named column.
// Used by read/reuse paths so v2 snapshots (built before context_prefix
// existed) keep working against v3-aware code. The column parameter stays
// generic so future schema bumps (v3→v4 probes, etc.) can reuse the helper
// without re-threading a renamed signature through every read/reuse path.
//
//nolint:unparam // column is future-proofing; see doc comment above.
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

func resolveUpdateSessionConfig(ctx context.Context, tx *sql.Tx, added []corpus.Record, options UpdateOptions) (sessionConfig, error) {
	if err := migrateSchemaToCurrent(ctx, tx); err != nil {
		return sessionConfig{}, err
	}

	embedderFingerprint, err := readMetadataValue(ctx, tx, "embedder_fingerprint")
	if err != nil {
		return sessionConfig{}, err
	}
	dimensionValue, err := readMetadataValue(ctx, tx, "embedder_dimension")
	if err != nil {
		return sessionConfig{}, err
	}
	dimension, err := strconv.Atoi(strings.TrimSpace(dimensionValue))
	if err != nil {
		return sessionConfig{}, fmt.Errorf("parse embedder_dimension %q: %w", dimensionValue, err)
	}

	quantization, err := readMetadataValueOptional(ctx, tx, "quantization", store.QuantizationFloat32)
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
			MaxSections:   resolveMaxChunkSections(options.MaxChunkSections),
		},
	}

	if len(added) == 0 && options.Embedder == nil {
		return cfg, nil
	}
	if options.Embedder == nil {
		return sessionConfig{}, fmt.Errorf("update embedder is required when adding records")
	}
	if err := validateUpdateEmbedder(ctx, options.Embedder, embedderFingerprint, dimension); err != nil {
		return sessionConfig{}, err
	}

	cfg.embedder = options.Embedder
	return cfg, nil
}

// validateUpdateEmbedder checks that the update-time embedder agrees with
// the snapshot's stored fingerprint and dimension. Update is incremental:
// new chunks must encode under the same embedder identity and into the
// same vector space as everything already in the index, otherwise hybrid
// retrieval would compare vectors from two different embedding spaces and
// silently mis-rank them.
func validateUpdateEmbedder(ctx context.Context, embedder embed.Embedder, storedFingerprint string, storedDimension int) error {
	updateFingerprint := embedder.Fingerprint()
	if storedFingerprint != updateFingerprint {
		return fmt.Errorf("embedder fingerprint mismatch: index=%q update=%q", storedFingerprint, updateFingerprint)
	}
	updateDimension, err := embedder.Dimension(ctx)
	if err != nil {
		return fmt.Errorf("resolve update embedder dimension: %w", err)
	}
	if updateDimension <= 0 {
		return fmt.Errorf("embedder dimension must be positive")
	}
	if updateDimension != storedDimension {
		return fmt.Errorf("embedder dimension mismatch: index=%d update=%d", storedDimension, updateDimension)
	}
	return nil
}

func deleteRecord(ctx context.Context, tx *sql.Tx, ref string) (bool, error) {
	// chunks_vec and fts_chunks are sqlite-vec / FTS5 virtual tables and
	// do not participate in foreign-key cascades, so we delete their rows
	// explicitly. chunks_vec_full is a regular table with
	//   FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
	// and configureDB pins PRAGMA foreign_keys = ON, so the
	// records → chunks → chunks_vec_full cascade cleans it up when the
	// record is deleted below. No explicit delete needed.
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

// loadCurrentRefHashes reads only the columns corpus.FingerprintFromPairs
// consumes. It intentionally avoids body_text / metadata_json so finalizeUpdate
// pays O(N * small) per update rather than O(N * full_row) (see #37).
func loadCurrentRefHashes(ctx context.Context, query queryContextRunner) ([]corpus.RefHash, error) {
	rows, err := query.QueryContext(ctx, `
SELECT ref, content_hash
FROM records
ORDER BY ref ASC`)
	if err != nil {
		return nil, fmt.Errorf("query current ref/content_hash pairs: %w", err)
	}
	defer func() { _ = rows.Close() }()

	pairs := make([]corpus.RefHash, 0)
	for rows.Next() {
		var pair corpus.RefHash
		if err := rows.Scan(&pair.Ref, &pair.ContentHash); err != nil {
			return nil, fmt.Errorf("scan ref/content_hash row: %w", err)
		}
		// Guard the byte-identity contract with corpus.Fingerprint([]Record).
		// That path runs each record through Normalized(), which regenerates
		// ContentHash via HashRecord when empty (an O(row_size) step we
		// deliberately skip here). If a row ever lands with an empty
		// content_hash — only possible via a writer bypass or external DB
		// tampering — FingerprintFromPairs would silently drop it while
		// Fingerprint([]Record) would include it with a regenerated hash,
		// making the persisted fingerprint diverge from what Snapshot.Records
		// callers would recompute. Fail loudly instead.
		if strings.TrimSpace(pair.ContentHash) == "" {
			return nil, fmt.Errorf("record %q has empty content_hash; refusing to recompute fingerprint over malformed state", pair.Ref)
		}
		pairs = append(pairs, pair)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate ref/content_hash rows: %w", err)
	}
	return pairs, nil
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
	// minCandidatePool is the floor for the candidate shortlist, keeping
	// tiny limits (e.g., 1-2 hits) from starving rank fusion. The matching
	// candidateOverfetchMultiplier lives in the colocated overfetch block
	// above buildSearchSQL.
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
	case store.QuantizationBinary:
		return store.QuantizationBinary, nil
	default:
		return "", fmt.Errorf("unsupported quantization mode %q (must be %q, %q, or %q)",
			q, store.QuantizationFloat32, store.QuantizationInt8, store.QuantizationBinary)
	}
}

func encodeVector(vector []float64, quantization string) ([]byte, error) {
	switch quantization {
	case store.QuantizationInt8:
		return store.EncodeVectorBlobInt8(vector)
	case store.QuantizationBinary:
		return store.EncodeVectorBlobBinary(vector)
	default:
		return store.EncodeVectorBlob(vector)
	}
}

func decodeVector(blob []byte, quantization string) ([]float64, error) {
	switch quantization {
	case store.QuantizationInt8:
		return store.DecodeVectorBlobInt8(blob)
	case store.QuantizationBinary:
		return store.DecodeVectorBlobBinary(blob)
	default:
		return store.DecodeVectorBlob(blob)
	}
}
