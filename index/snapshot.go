package index

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/dusk-network/stroma/corpus"
	"github.com/dusk-network/stroma/embed"
	"github.com/dusk-network/stroma/store"
)

// Snapshot is one opened Stroma index snapshot.
type Snapshot struct {
	path string
	db   *sql.DB
}

// SnapshotSearchQuery defines one text search against an opened snapshot.
type SnapshotSearchQuery struct {
	Text     string
	Limit    int
	Kinds    []string
	Embedder embed.Embedder
	Reranker Reranker

	// SearchDimension optionally runs a truncated-prefix vector prefilter
	// at this dimension, then rescores the shortlist with full-dim cosine.
	// See SearchQuery.SearchDimension for the full contract.
	SearchDimension int
}

// VectorSearchQuery defines one vector search against an opened snapshot.
type VectorSearchQuery struct {
	Embedding []float64
	Limit     int
	Kinds     []string
}

// RecordQuery filters records from an opened snapshot.
type RecordQuery struct {
	Refs  []string
	Kinds []string
}

// SectionQuery filters sections from an opened snapshot.
type SectionQuery struct {
	Refs              []string
	Kinds             []string
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

// OpenSnapshot opens a read-only Stroma snapshot.
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
	return &Snapshot{path: path, db: db}, nil
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
	if query.IncludeEmbeddings {
		var err error
		quantization, err = s.resolveQuantization(ctx)
		if err != nil {
			return nil, err
		}
	}

	hasPrefix, err := hasChunkColumn(ctx, s.db, "context_prefix")
	if err != nil {
		return nil, fmt.Errorf("probe chunks.context_prefix: %w", err)
	}
	sqlText, args := buildSectionsQuery(query, hasPrefix)
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

func buildSectionsQuery(query SectionQuery, hasContextPrefix bool) (sqlText string, args []any) {
	var builder strings.Builder
	args = make([]any, 0, len(query.Refs)+len(query.Kinds))
	// v2 snapshots lack context_prefix; project an empty string so the
	// row shape scanSectionRow expects stays stable across versions.
	prefixExpr := "'' AS context_prefix"
	if hasContextPrefix {
		prefixExpr = "c.context_prefix"
	}
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
		builder.WriteString(`
JOIN chunks_vec v ON v.chunk_id = c.id`)
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

// resolveQuantization reads the snapshot's quantization metadata and
// validates it against the supported set. It fails fast on unsupported
// values so callers cannot silently misdecode vectors against stale or
// malformed metadata. An empty / missing key defaults to float32 via
// readMetadataValueOptional, which then passes normalizeQuantization.
func (s *Snapshot) resolveQuantization(ctx context.Context) (string, error) {
	raw, err := readMetadataValueOptional(ctx, s.db, "quantization", store.QuantizationFloat32)
	if err != nil {
		return "", err
	}
	return normalizeQuantization(raw)
}

// resolveSearchDimension validates SearchDimension against the stored
// embedder dimension and quantization. Zero means "no truncation"; any
// positive value must be <= stored dim. Matryoshka-style truncation is
// only supported for float32 indexes because the sqlite-vec int8 storage
// path packs vec_int8 bytes and vec_slice over that packed form would
// not yield a valid truncated int8 vector for cosine distance.
func (s *Snapshot) resolveSearchDimension(ctx context.Context, requested int, quantization string) (int, error) {
	if requested <= 0 {
		return 0, nil
	}
	if quantization != store.QuantizationFloat32 {
		return 0, fmt.Errorf("SearchDimension is only supported for float32 indexes, got %q", quantization)
	}
	storedDim, err := s.resolveDimension(ctx)
	if err != nil {
		return 0, fmt.Errorf("resolve search dimension: %w", err)
	}
	if requested > storedDim {
		return 0, fmt.Errorf("SearchDimension %d exceeds stored embedder_dimension %d", requested, storedDim)
	}
	if requested == storedDim {
		// No-op truncation: skip the Matryoshka path and fall back to the
		// existing vec0 MATCH route so callers do not pay the scan cost
		// just because they passed the stored dim explicitly.
		return 0, nil
	}
	return requested, nil
}

// resolveDimension reads the snapshot's stored embedder dimension.
func (s *Snapshot) resolveDimension(ctx context.Context) (int, error) {
	raw, err := readMetadataValue(ctx, s.db, "embedder_dimension")
	if err != nil {
		return 0, err
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

func scanSectionRow(rows *sql.Rows, includeEmbeddings bool, quantization string) (Section, error) {
	var (
		section     Section
		metadataRaw string
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
		query.Limit = 10
	}

	if err := ensureCompatibleEmbedder(ctx, s.db, query.Embedder); err != nil {
		return nil, err
	}

	quantization, err := s.resolveQuantization(ctx)
	if err != nil {
		return nil, err
	}
	vectors, err := query.Embedder.EmbedQueries(ctx, []string{query.Text})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("embedder returned %d query vectors, want 1", len(vectors))
	}

	searchDim, err := s.resolveSearchDimension(ctx, query.SearchDimension, quantization)
	if err != nil {
		return nil, err
	}

	candidateCount := candidateLimit(query.Limit)

	vectorHits, err := s.searchVectorCandidates(ctx, vectors[0], candidateCount, query.Kinds, quantization, searchDim)
	if err != nil {
		return nil, err
	}

	ftsHits, ftsErr := s.searchFTS(ctx, query.Text, candidateCount, query.Kinds)
	if ftsErr != nil {
		return nil, fmt.Errorf("fts search: %w", ftsErr)
	}

	if len(ftsHits) == 0 {
		return rerankCandidates(ctx, query.Text, query.Limit, vectorHits, query.Reranker)
	}

	return rerankCandidates(ctx, query.Text, query.Limit, mergeRRF(vectorHits, ftsHits, candidateCount), query.Reranker)
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
		query.Limit = 10
	}

	quantization, err := s.resolveQuantization(ctx)
	if err != nil {
		return nil, err
	}
	hits, err := s.searchVectorCandidates(ctx, query.Embedding, candidateLimit(query.Limit), query.Kinds, quantization, 0)
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
	if searchDimension > 0 {
		sqlText, args, err = buildMatryoshkaSearchSQL(limit, kinds, queryBlob, searchDimension)
		if err != nil {
			return nil, err
		}
	} else {
		sqlText, args = buildSearchSQL(SearchQuery{
			Limit: limit,
			Kinds: kinds,
		}, queryBlob, quantization)
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
			metadataJSON string
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
		// FTS table does not exist in indexes built before hybrid search.
		if strings.Contains(err.Error(), "no such table: fts_chunks") {
			return nil, nil
		}
		return nil, fmt.Errorf("fts query: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var hits []SearchHit
	for rows.Next() {
		var (
			hit          SearchHit
			metadataJSON string
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

func unmarshalMetadata(ref, raw string) (map[string]string, error) {
	if strings.TrimSpace(raw) == "" || raw == "{}" {
		return map[string]string{}, nil
	}
	metadata := map[string]string{}
	if err := json.Unmarshal([]byte(raw), &metadata); err != nil {
		return nil, fmt.Errorf("decode metadata for %s: %w", ref, err)
	}
	return metadata, nil
}
