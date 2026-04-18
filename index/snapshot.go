package index

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/dusk-network/stroma/corpus"
	"github.com/dusk-network/stroma/store"
)

// SQL fragments for projecting the chunks.context_prefix column. v2
// snapshots (and any future schema bump that drops the column) substitute
// the missing variant so scan-row shapes stay uniform across versions.
// Centralized here so reuse.go, snapshot.go, and context.go all read from
// one source of truth — `c.` prefixed because every callsite joins
// chunks AS c.
const (
	contextPrefixSelectMissing = "'' AS context_prefix"
	contextPrefixSelectPresent = "c.context_prefix"
)

// contextPrefixSelectExpr returns the SQL fragment for the
// context_prefix projection given whether the snapshot's chunks table
// carries the column.
func contextPrefixSelectExpr(hasContextPrefix bool) string {
	if hasContextPrefix {
		return contextPrefixSelectPresent
	}
	return contextPrefixSelectMissing
}

// Snapshot is one opened Stroma index snapshot.
//
// Safe for concurrent use by multiple goroutines once returned from
// OpenSnapshot: all read methods (Stats, Records, Sections, Search,
// SearchVector, ExpandContext) share the underlying *sql.DB, which
// serializes queries internally. Cached metadata fields
// (quantization, storedDimension, hasFTS, …) are populated at open
// time and read-only thereafter.
type Snapshot struct {
	path string
	db   *sql.DB
	// hasContextPrefix is derived from the accept-listed schema_version at
	// OpenSnapshot time so read paths do not need to reprobe PRAGMA
	// table_info(chunks) on every Sections()/reuse call.
	hasContextPrefix bool
	// hasFTS records whether the snapshot carries the fts_chunks virtual
	// table. Snapshots built before hybrid search lack it; cache the
	// presence at open time so searchFTS can short-circuit cleanly
	// instead of running the query and pattern-matching on the SQLite
	// "no such table" error string.
	hasFTS bool
	// hasParentChunkID records whether the snapshot carries the
	// chunks.parent_chunk_id column added in schema v5 (#16). Read paths
	// (notably ExpandContext) cache the presence at open time so legacy
	// v2/v3/v4 snapshots degrade cleanly to flat-chunk semantics without
	// an extra PRAGMA per call.
	hasParentChunkID bool
	// quantization and storedDimension are resolved from immutable
	// metadata at OpenSnapshot time so Search/SearchVector/Sections do
	// not reread the metadata table on every call. Both values are fixed
	// by the index build and cannot change while the snapshot is open.
	//
	// Errors encountered while parsing either field are cached in
	// quantizationErr / storedDimensionErr and deferred to the first
	// vector-touching read (Search, SearchVector, Sections with
	// IncludeEmbeddings=true). Non-vector reads (Records, Stats,
	// Sections with IncludeEmbeddings=false) continue to work against a
	// snapshot whose vector metadata is malformed, so recovery and
	// inspection workflows are not bricked by corrupted quantization or
	// embedder_dimension values.
	quantization       string
	quantizationErr    error
	storedDimension    int
	storedDimensionErr error
}

// SnapshotSearchQuery defines one text search against an opened snapshot.
// Retrieval parameters live on the embedded SearchParams so the same
// value can be forwarded verbatim from SearchQuery.SearchParams without
// hand-copying fields.
type SnapshotSearchQuery struct {
	SearchParams
}

// VectorSearchQuery defines one vector search against an opened snapshot.
type VectorSearchQuery struct {
	// Embedding is the precomputed query vector. Empty rejects with
	// a "search embedding is required" error — this field has no
	// default.
	Embedding []float64
	// Limit caps the number of SearchHits returned. Zero or negative
	// selects DefaultSearchLimit (10).
	Limit int
	// Kinds filters candidate records to the supplied kind list. Nil
	// or empty means "no filter, all kinds".
	Kinds []string
}

// RecordQuery filters records from an opened snapshot.
type RecordQuery struct {
	Refs  []string
	Kinds []string
}

// SectionQuery filters sections from an opened snapshot.
type SectionQuery struct {
	Refs  []string
	Kinds []string

	// IncludeEmbeddings asks Sections() to populate Section.Embedding
	// from the stored vector column. Snapshots produced by hierarchical
	// policies (e.g., chunk.LateChunkPolicy) hold parent rows that are
	// storage-only context with no vector — those rows are filtered
	// out of an IncludeEmbeddings = true query because the underlying
	// chunks → chunks_vec join is inner. Set IncludeEmbeddings = false
	// to receive every chunk row (parents + leaves) without embeddings.
	IncludeEmbeddings bool
}

// Section is one stored section from a Stroma snapshot.
type Section struct {
	ChunkID       int64
	Ref           string
	Kind          string
	Title         string
	SourceRef     string
	Heading       string
	Content       string
	ContextPrefix string
	Metadata      map[string]string
	Embedding     []float64
}

// OpenSnapshot opens a read-only Stroma snapshot at path. The path is
// OS-native; on Windows both forward and back slashes are accepted
// (the store package normalizes drive prefixes on open). The
// snapshot's schema_version metadata must be one of the accept-listed
// versions — schemaVersion (current), prevSchemaVersion,
// legacySchemaVersionV3, or legacySchemaVersionV2 — all of which read
// paths can decode directly without forcing an Update. Anything else
// returns ErrUnsupportedSchemaVersion wrapped with the observed
// version, so callers can surface a clear upgrade/downgrade message
// instead of silently misdecoding data against a future schema.
//
// The returned *Snapshot is safe for concurrent use by multiple
// goroutines once constructed: all read methods (Stats, Records,
// Sections, Search, SearchVector, ExpandContext) share the underlying
// *sql.DB which serializes queries internally.
func OpenSnapshot(ctx context.Context, path string) (*Snapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, fmt.Errorf("snapshot path is required")
	}
	db, err := store.OpenReadOnlyContext(ctx, path)
	if err != nil {
		return nil, err
	}
	schema, err := readMetadataValue(ctx, db, "schema_version")
	if err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("open snapshot %s: %w", path, err)
	}
	trimmedSchema := strings.TrimSpace(schema)
	// Read paths don't depend on the content_hash encoding — they consume
	// stored values as-is — so legacy v2 snapshots keep opening for Stats,
	// Records, Sections, and Search against v1.0+ code. Only the Update
	// path forces the content_hash algo bump via migrateSchemaToCurrent.
	if trimmedSchema != schemaVersion && trimmedSchema != prevSchemaVersion && trimmedSchema != legacySchemaVersionV3 && trimmedSchema != legacySchemaVersionV2 {
		_ = db.Close()
		return nil, fmt.Errorf("open snapshot %s: %w: got %q, supported %q, %q, %q, or %q",
			path, ErrUnsupportedSchemaVersion, trimmedSchema, legacySchemaVersionV2, legacySchemaVersionV3, prevSchemaVersion, schemaVersion)
	}
	// For binary snapshots, verify the full-precision companion table is
	// complete at open time. Read paths rely on inner joins against
	// chunks_vec_full, so a missing companion row would silently drop
	// that chunk from search and Sections() without surfacing an error.
	if err := checkChunksVecFullCompleteness(ctx, db); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("open snapshot %s: %w", path, err)
	}
	// Verify the chunks table shape agrees with the accept-listed
	// schema_version before we cache hasContextPrefix. Any divergence —
	// metadata says v3 but the column is missing, or metadata says v2 but
	// the column is present — means read paths would silently misread
	// (empty prefixes on a v3 file, or random column data on a v2 file).
	// Fail at open time instead of deferring to a cryptic Sections() error
	// or, worse, silent empty reads. This is a one-time PRAGMA per handle,
	// not per Sections() call.
	expectContextPrefix := schemaHasContextPrefix(trimmedSchema)
	hasColumn, err := hasChunkColumn(ctx, db, "context_prefix")
	if err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("open snapshot %s: probe chunks.context_prefix: %w", path, err)
	}
	if hasColumn != expectContextPrefix {
		_ = db.Close()
		return nil, fmt.Errorf("open snapshot %s: schema_version=%q but chunks.context_prefix presence=%t (want %t); snapshot metadata and table shape disagree",
			path, trimmedSchema, hasColumn, expectContextPrefix)
	}
	hasFTS, err := hasTable(ctx, db, "fts_chunks")
	if err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("open snapshot %s: probe fts_chunks: %w", path, err)
	}
	// Same pattern as hasContextPrefix above: schema_version determines
	// whether parent_chunk_id is present, but verify the table shape
	// agrees so a malformed snapshot fails at open time instead of at
	// the first ExpandContext call.
	expectParentChunkID := schemaHasParentChunkID(trimmedSchema)
	hasParentColumn, err := hasChunkColumn(ctx, db, "parent_chunk_id")
	if err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("open snapshot %s: probe chunks.parent_chunk_id: %w", path, err)
	}
	if hasParentColumn != expectParentChunkID {
		_ = db.Close()
		return nil, fmt.Errorf("open snapshot %s: schema_version=%q but chunks.parent_chunk_id presence=%t (want %t); snapshot metadata and table shape disagree",
			path, trimmedSchema, hasParentColumn, expectParentChunkID)
	}
	// Resolve immutable vector metadata once. Quantization and the stored
	// embedder dimension are fixed by the index build, so caching them on
	// the handle lets Search/SearchVector/Sections avoid reopening the
	// metadata table on every call.
	//
	// Errors are deferred: a malformed quantization or embedder_dimension
	// must not brick non-vector reads (Records, Stats, Sections without
	// embeddings) or diagnostic workflows. The deferred error is surfaced
	// at the first vector-touching read instead.
	quantization, quantizationErr := resolveSnapshotQuantization(ctx, db)
	storedDimension, storedDimensionErr := resolveSnapshotDimension(ctx, db)
	return &Snapshot{
		path:               path,
		db:                 db,
		hasContextPrefix:   expectContextPrefix,
		hasFTS:             hasFTS,
		hasParentChunkID:   expectParentChunkID,
		quantization:       quantization,
		quantizationErr:    quantizationErr,
		storedDimension:    storedDimension,
		storedDimensionErr: storedDimensionErr,
	}, nil
}

// resolveSnapshotQuantization reads and normalizes the quantization
// metadata once at OpenSnapshot time. On any error the returned string
// is empty and the error is returned for the caller to cache; read
// paths that actually depend on quantization check the cached error
// before using the value.
func resolveSnapshotQuantization(ctx context.Context, db *sql.DB) (string, error) {
	raw, err := readMetadataValueOptional(ctx, db, "quantization", store.QuantizationFloat32)
	if err != nil {
		return "", fmt.Errorf("read quantization: %w", err)
	}
	return normalizeQuantization(raw)
}

// resolveSnapshotDimension reads and parses the embedder_dimension
// metadata once at OpenSnapshot time. On any error the returned dim is
// zero and the error is returned for the caller to cache; read paths
// that actually depend on the stored dimension check the cached error
// before using the value.
func resolveSnapshotDimension(ctx context.Context, db *sql.DB) (int, error) {
	raw, err := readMetadataValue(ctx, db, "embedder_dimension")
	if err != nil {
		return 0, fmt.Errorf("read embedder_dimension: %w", err)
	}
	dim, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return 0, fmt.Errorf("parse embedder_dimension %q: %w", raw, err)
	}
	if dim <= 0 {
		return 0, fmt.Errorf("stored embedder_dimension %d is non-positive", dim)
	}
	return dim, nil
}

// Close releases the opened snapshot handle.
func (s *Snapshot) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

// Path returns the opened snapshot path.
func (s *Snapshot) Path() string {
	if s == nil {
		return ""
	}
	return s.path
}

// Stats inspects the opened snapshot.
func (s *Snapshot) Stats(ctx context.Context) (*Stats, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("snapshot is not open")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	stats := &Stats{
		Path:       s.path,
		KindCounts: map[string]int{},
	}
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM records`).Scan(&stats.RecordCount); err != nil {
		return nil, fmt.Errorf("count records: %w", err)
	}
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM chunks`).Scan(&stats.ChunkCount); err != nil {
		return nil, fmt.Errorf("count chunks: %w", err)
	}

	rows, err := s.db.QueryContext(ctx, `SELECT kind, COUNT(*) FROM records GROUP BY kind ORDER BY kind`)
	if err != nil {
		return nil, fmt.Errorf("count records by kind: %w", err)
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var kind string
		var count int
		if err := rows.Scan(&kind, &count); err != nil {
			return nil, fmt.Errorf("scan kind count: %w", err)
		}
		stats.KindCounts[kind] = count
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate kind counts: %w", err)
	}

	if err := loadStatsMetadata(ctx, s.db, stats); err != nil {
		return nil, err
	}
	return stats, nil
}

func loadStatsMetadata(ctx context.Context, db queryContextRunner, stats *Stats) error {
	var err error
	stats.SchemaVersion, err = readMetadataValue(ctx, db, "schema_version")
	if err != nil {
		return err
	}
	dimensionValue, err := readMetadataValue(ctx, db, "embedder_dimension")
	if err != nil {
		return err
	}
	stats.EmbedderDimension, err = strconv.Atoi(dimensionValue)
	if err != nil {
		return fmt.Errorf("parse embedder_dimension %q: %w", dimensionValue, err)
	}
	stats.EmbedderFingerprint, err = readMetadataValue(ctx, db, "embedder_fingerprint")
	if err != nil {
		return err
	}
	stats.ContentFingerprint, err = readMetadataValue(ctx, db, "content_fingerprint")
	if err != nil {
		return err
	}
	stats.CreatedAt, err = readMetadataValue(ctx, db, "created_at")
	return err
}

// Records returns records from the opened snapshot.
func (s *Snapshot) Records(ctx context.Context, query RecordQuery) ([]corpus.Record, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("snapshot is not open")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	var builder strings.Builder
	args := make([]any, 0, 1+len(query.Refs)+len(query.Kinds))
	builder.WriteString(`
SELECT ref, kind, title, source_ref, body_format, body_text, content_hash, metadata_json
FROM records
WHERE 1 = 1`)
	appendRecordQueryFilter(&builder, &args, "ref", query.Refs)
	appendRecordQueryFilter(&builder, &args, "kind", query.Kinds)
	builder.WriteString(`
ORDER BY ref ASC`)

	rows, err := s.db.QueryContext(ctx, builder.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("query records: %w", err)
	}
	defer func() { _ = rows.Close() }()

	result := make([]corpus.Record, 0)
	for rows.Next() {
		record, err := scanRecord(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, record)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate records: %w", err)
	}
	return result, nil
}

// scanRecord reads one row from a query that selects the full record column
// set (ref, kind, title, source_ref, body_format, body_text, content_hash,
// metadata_json) and returns the decoded record with metadata parsed.
func scanRecord(rows *sql.Rows) (corpus.Record, error) {
	var (
		record      corpus.Record
		metadataRaw []byte
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
		return corpus.Record{}, fmt.Errorf("scan record: %w", err)
	}
	metadata, err := unmarshalMetadata(record.Ref, metadataRaw)
	if err != nil {
		return corpus.Record{}, err
	}
	record.Metadata = metadata
	return record, nil
}

// Sections returns sections from the opened snapshot.
func (s *Snapshot) Sections(ctx context.Context, query SectionQuery) ([]Section, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("snapshot is not open")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	quantization := store.QuantizationFloat32
	embeddingsFromFullTable := false
	if query.IncludeEmbeddings {
		if s.quantizationErr != nil {
			return nil, s.quantizationErr
		}
		quantization = s.quantization
		if quantization == store.QuantizationBinary {
			// Read from the full-precision companion so callers get
			// the real float32 vector, not its sign-packed stub, and
			// switch the decode path to float32 so scanSectionRow
			// interprets the blob correctly.
			embeddingsFromFullTable = true
			quantization = store.QuantizationFloat32
		}
	}

	sqlText, args := buildSectionsQuery(query, s.hasContextPrefix, embeddingsFromFullTable)
	rows, err := s.db.QueryContext(ctx, sqlText, args...)
	if err != nil {
		return nil, fmt.Errorf("query sections: %w", err)
	}
	defer func() { _ = rows.Close() }()

	result := make([]Section, 0)
	for rows.Next() {
		section, err := scanSectionRow(rows, query.IncludeEmbeddings, quantization)
		if err != nil {
			return nil, err
		}
		result = append(result, section)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate sections: %w", err)
	}
	return result, nil
}

func buildSectionsQuery(query SectionQuery, hasContextPrefix, embeddingsFromFullTable bool) (sqlText string, args []any) {
	var builder strings.Builder
	args = make([]any, 0, len(query.Refs)+len(query.Kinds))
	// v2 snapshots lack context_prefix; project an empty string so the
	// row shape scanSectionRow expects stays stable across versions.
	prefixExpr := contextPrefixSelectExpr(hasContextPrefix)
	builder.WriteString(`
SELECT
  c.id,
  r.ref,
  r.kind,
  r.title,
  r.source_ref,
  c.heading,
  c.content,
  ` + prefixExpr + `,
  r.metadata_json`)
	if query.IncludeEmbeddings {
		builder.WriteString(", v.embedding")
	}
	builder.WriteString(`
FROM chunks c
JOIN records r ON r.ref = c.record_ref`)
	if query.IncludeEmbeddings {
		// Binary snapshots surface full-precision vectors via the
		// chunks_vec_full companion so IncludeEmbeddings yields usable
		// float32 values instead of sign-expanded {-1, 1} stubs.
		if embeddingsFromFullTable {
			builder.WriteString(`
JOIN chunks_vec_full v ON v.chunk_id = c.id`)
		} else {
			builder.WriteString(`
JOIN chunks_vec v ON v.chunk_id = c.id`)
		}
	}
	builder.WriteString(`
WHERE 1 = 1`)
	appendRecordQueryFilter(&builder, &args, "r.ref", query.Refs)
	appendRecordQueryFilter(&builder, &args, "r.kind", query.Kinds)
	builder.WriteString(`
ORDER BY r.ref ASC, c.chunk_index ASC, c.id ASC`)
	sqlText = builder.String()
	return sqlText, args
}

// resolveSearchDimension validates SearchDimension against the stored
// embedder dimension and quantization. Zero means "no truncation"; any
// positive value must be <= stored dim. Matryoshka-style truncation is
// only supported for float32 indexes because the sqlite-vec int8 storage
// path packs vec_int8 bytes and vec_slice over that packed form would
// not yield a valid truncated int8 vector for cosine distance.
func (s *Snapshot) resolveSearchDimension(requested int) (int, error) {
	if requested <= 0 {
		return 0, nil
	}
	if s.storedDimensionErr != nil {
		return 0, s.storedDimensionErr
	}
	if s.quantization != store.QuantizationFloat32 {
		return 0, fmt.Errorf("SearchDimension is only supported for float32 indexes, got %q", s.quantization)
	}
	if requested > s.storedDimension {
		return 0, fmt.Errorf("SearchDimension %d exceeds stored embedder_dimension %d", requested, s.storedDimension)
	}
	if requested == s.storedDimension {
		// No-op truncation: skip the Matryoshka path and fall back to the
		// existing vec0 MATCH route so callers do not pay the scan cost
		// just because they passed the stored dim explicitly.
		return 0, nil
	}
	return requested, nil
}

func scanSectionRow(rows *sql.Rows, includeEmbeddings bool, quantization string) (Section, error) {
	var (
		section     Section
		metadataRaw []byte
		blob        []byte
	)
	scanArgs := []any{
		&section.ChunkID,
		&section.Ref,
		&section.Kind,
		&section.Title,
		&section.SourceRef,
		&section.Heading,
		&section.Content,
		&section.ContextPrefix,
		&metadataRaw,
	}
	if includeEmbeddings {
		scanArgs = append(scanArgs, &blob)
	}
	if err := rows.Scan(scanArgs...); err != nil {
		return Section{}, fmt.Errorf("scan section: %w", err)
	}
	metadata, err := unmarshalMetadata(section.Ref, metadataRaw)
	if err != nil {
		return Section{}, err
	}
	section.Metadata = metadata
	if includeEmbeddings {
		section.Embedding, err = decodeVector(blob, quantization)
		if err != nil {
			return Section{}, fmt.Errorf("decode section embedding for %s: %w", section.Ref, err)
		}
	}
	return section, nil
}

// Search runs a hybrid text search (vector + FTS5) against the opened snapshot.
func (s *Snapshot) Search(ctx context.Context, query SnapshotSearchQuery) ([]SearchHit, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("snapshot is not open")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	query.Text = strings.TrimSpace(query.Text)
	if query.Text == "" {
		return nil, fmt.Errorf("search text is required")
	}
	if query.Embedder == nil {
		return nil, fmt.Errorf("search embedder is required")
	}
	if query.Limit <= 0 {
		query.Limit = DefaultSearchLimit
	}

	if err := ensureCompatibleEmbedder(ctx, s.db, query.Embedder); err != nil {
		return nil, err
	}
	if s.quantizationErr != nil {
		return nil, s.quantizationErr
	}

	vectors, err := query.Embedder.EmbedQueries(ctx, []string{query.Text})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("embedder returned %d query vectors, want 1", len(vectors))
	}

	searchDim, err := s.resolveSearchDimension(query.SearchDimension)
	if err != nil {
		return nil, err
	}

	candidateCount := candidateLimit(query.Limit)

	vectorArm := runArm(ArmVector, func() ([]SearchHit, error) {
		return s.searchVectorCandidates(ctx, vectors[0], candidateCount, query.Kinds, s.quantization, searchDim)
	})
	if vectorArm.Err != nil {
		// Pre-#17 behavior: vector-arm errors always fail the search
		// fast with the arm's own error shape, independent of the
		// configured FusionStrategy.
		return nil, vectorArm.Err
	}

	// searchFTS returns (nil, nil) on legacy snapshots without fts_chunks
	// (!s.hasFTS) and on queries whose sanitized form is empty. The first
	// case is surfaced to FusionStrategy as Available=false; the second as
	// Available=true with empty Hits ("arm ran, zero matches").
	ftsArm := runArm(ArmFTS, func() ([]SearchHit, error) {
		return s.searchFTS(ctx, query.Text, candidateCount, query.Kinds)
	})
	if !s.hasFTS {
		// Legacy snapshot without the fts_chunks virtual table. searchFTS
		// already short-circuits to (nil, nil); reflect that to the
		// FusionStrategy as an unavailable arm so custom strategies can
		// see the explicit signal rather than an empty available arm.
		ftsArm.Available = false
	}

	strategy := query.Fusion
	if strategy == nil {
		strategy = DefaultFusion()
	}
	fused, err := strategy.Fuse([]RetrievalArm{vectorArm, ftsArm}, candidateCount)
	if err != nil {
		return nil, fmt.Errorf("fuse candidates: %w", err)
	}

	return rerankCandidates(ctx, query.Text, query.Limit, fused, query.Reranker)
}

// runArm invokes fn and packs the result into a RetrievalArm. A non-nil
// error produces an Available=false arm with empty Hits; a nil error
// produces an Available=true arm with the returned hits. FusionStrategy
// implementations that want partial-arm tolerance can inspect Err; the
// default RRFFusion fails closed on any non-nil Err via validateArms.
func runArm(name string, fn func() ([]SearchHit, error)) RetrievalArm {
	hits, err := fn()
	if err != nil {
		return RetrievalArm{Name: name, Available: false, Err: err}
	}
	return RetrievalArm{Name: name, Hits: hits, Available: true}
}

// SearchVector runs a vector search against the opened snapshot.
func (s *Snapshot) SearchVector(ctx context.Context, query VectorSearchQuery) ([]SearchHit, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("snapshot is not open")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if len(query.Embedding) == 0 {
		return nil, fmt.Errorf("search embedding is required")
	}
	if query.Limit <= 0 {
		query.Limit = DefaultSearchLimit
	}
	if s.quantizationErr != nil {
		return nil, s.quantizationErr
	}

	hits, err := s.searchVectorCandidates(ctx, query.Embedding, candidateLimit(query.Limit), query.Kinds, s.quantization, 0)
	if err != nil {
		return nil, err
	}
	if len(hits) > query.Limit {
		hits = hits[:query.Limit]
	}
	return hits, nil
}

func (s *Snapshot) searchVectorCandidates(ctx context.Context, embedding []float64, limit int, kinds []string, quantization string, searchDimension int) ([]SearchHit, error) {
	queryBlob, err := encodeVector(embedding, quantization)
	if err != nil {
		return nil, fmt.Errorf("encode query vector: %w", err)
	}
	var sqlText string
	var args []any
	switch {
	case searchDimension > 0:
		// Matryoshka is already validated as float32-only upstream in
		// resolveSearchDimension, so this and the binary branch below
		// are mutually exclusive by contract.
		sqlText, args, err = buildMatryoshkaSearchSQL(limit, kinds, queryBlob, searchDimension)
		if err != nil {
			return nil, err
		}
	case quantization == store.QuantizationBinary:
		fullQueryBlob, err := store.EncodeVectorBlob(embedding)
		if err != nil {
			return nil, fmt.Errorf("encode full-precision query vector: %w", err)
		}
		sqlText, args = buildBinarySearchSQL(limit, kinds, queryBlob, fullQueryBlob)
	default:
		sqlText, args = buildSearchSQL(limit, kinds, queryBlob, quantization)
	}
	rows, err := s.db.QueryContext(ctx, sqlText, args...)
	if err != nil {
		return nil, fmt.Errorf("run search query: %w", err)
	}
	defer func() { _ = rows.Close() }()

	hits := make([]SearchHit, 0, limit)
	for rows.Next() {
		var (
			hit          SearchHit
			metadataJSON []byte
			distance     float64
		)
		if err := rows.Scan(
			&hit.ChunkID,
			&hit.Ref,
			&hit.Kind,
			&hit.Title,
			&hit.SourceRef,
			&hit.Heading,
			&hit.Content,
			&metadataJSON,
			&distance,
		); err != nil {
			return nil, fmt.Errorf("scan search hit: %w", err)
		}
		hit.Metadata, err = unmarshalMetadata(hit.Ref, metadataJSON)
		if err != nil {
			return nil, err
		}
		hit.Score = store.CosineScoreFromDistance(distance)
		hits = append(hits, hit)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate search hits: %w", err)
	}
	return hits, nil
}

func (s *Snapshot) searchFTS(ctx context.Context, text string, limit int, kinds []string) ([]SearchHit, error) {
	// fts_chunks is a virtual table that older snapshots (built before
	// hybrid search) do not carry. The presence flag is cached on the
	// snapshot at open time, so the lexical arm of hybrid search becomes
	// a no-op against legacy indexes without firing a query that would
	// fail with "no such table: fts_chunks".
	if !s.hasFTS {
		return nil, nil
	}
	ftsQuery := sanitizeFTSQuery(text)
	if ftsQuery == "" {
		return nil, nil
	}

	var builder strings.Builder
	args := make([]any, 0, 2+len(kinds))

	builder.WriteString(`
SELECT c.id, r.ref, r.kind, r.title, r.source_ref, c.heading, c.content, r.metadata_json, fts.rank
FROM fts_chunks fts
JOIN chunks c ON c.id = fts.rowid
JOIN records r ON r.ref = c.record_ref
WHERE fts_chunks MATCH ?`)
	args = append(args, ftsQuery)

	if len(kinds) > 0 {
		builder.WriteString(" AND r.kind IN (")
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
ORDER BY fts.rank, c.id ASC
LIMIT ?`)
	args = append(args, limit)

	rows, err := s.db.QueryContext(ctx, builder.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("fts query: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var hits []SearchHit
	for rows.Next() {
		var (
			hit          SearchHit
			metadataJSON []byte
			ftsRank      float64
		)
		if err := rows.Scan(
			&hit.ChunkID, &hit.Ref, &hit.Kind, &hit.Title, &hit.SourceRef,
			&hit.Heading, &hit.Content, &metadataJSON, &ftsRank,
		); err != nil {
			return nil, fmt.Errorf("scan fts hit: %w", err)
		}
		hit.Metadata, err = unmarshalMetadata(hit.Ref, metadataJSON)
		if err != nil {
			return nil, err
		}
		hit.Score = -ftsRank
		hits = append(hits, hit)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate fts hits: %w", err)
	}
	return hits, nil
}

func rerankCandidates(ctx context.Context, query string, limit int, hits []SearchHit, reranker Reranker) ([]SearchHit, error) {
	if reranker != nil {
		candidates := append([]SearchHit(nil), hits...)
		reranked, err := reranker.Rerank(ctx, query, candidates)
		if err != nil {
			return nil, fmt.Errorf("rerank candidates: %w", err)
		}
		hits = reranked
	}
	if len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, nil
}

func appendRecordQueryFilter(builder *strings.Builder, args *[]any, column string, values []string) {
	filtered := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			filtered = append(filtered, value)
		}
	}
	if len(filtered) == 0 {
		return
	}
	builder.WriteString(" AND ")
	builder.WriteString(column)
	builder.WriteString(" IN (")
	for i, value := range filtered {
		if i > 0 {
			builder.WriteString(", ")
		}
		builder.WriteString("?")
		*args = append(*args, value)
	}
	builder.WriteString(")")
}

// unmarshalMetadata decodes the per-record metadata_json column into a
// string map. The raw bytes are expected to be the exact buffer returned
// by database/sql's scan into a *[]byte destination — taking []byte keeps
// this off the string-to-[]byte copy path that json.Unmarshal([]byte(str))
// forces on every hit in the hot search loop (A2 in #57).
func unmarshalMetadata(ref string, raw []byte) (map[string]string, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("{}")) {
		return map[string]string{}, nil
	}
	metadata := map[string]string{}
	if err := json.Unmarshal(trimmed, &metadata); err != nil {
		return nil, fmt.Errorf("decode metadata for %s: %w", ref, err)
	}
	return metadata, nil
}
