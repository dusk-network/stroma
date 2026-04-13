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
	ChunkID   int64
	Ref       string
	Kind      string
	Title     string
	SourceRef string
	Heading   string
	Content   string
	Metadata  map[string]string
	Embedding []float64
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

	stats.SchemaVersion, err = readMetadataValue(ctx, s.db, "schema_version")
	if err != nil {
		return nil, err
	}
	dimensionValue, err := readMetadataValue(ctx, s.db, "embedder_dimension")
	if err != nil {
		return nil, err
	}
	stats.EmbedderDimension, err = strconv.Atoi(dimensionValue)
	if err != nil {
		return nil, fmt.Errorf("parse embedder_dimension %q: %w", dimensionValue, err)
	}
	stats.EmbedderFingerprint, err = readMetadataValue(ctx, s.db, "embedder_fingerprint")
	if err != nil {
		return nil, err
	}
	stats.ContentFingerprint, err = readMetadataValue(ctx, s.db, "content_fingerprint")
	if err != nil {
		return nil, err
	}
	stats.CreatedAt, err = readMetadataValue(ctx, s.db, "created_at")
	if err != nil {
		return nil, err
	}
	return stats, nil
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
			return nil, fmt.Errorf("scan record: %w", err)
		}
		record.Metadata, err = unmarshalMetadata(record.Ref, metadataRaw)
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

// Sections returns sections from the opened snapshot.
func (s *Snapshot) Sections(ctx context.Context, query SectionQuery) ([]Section, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("snapshot is not open")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	var builder strings.Builder
	args := make([]any, 0, len(query.Refs)+len(query.Kinds))
	builder.WriteString(`
SELECT
  c.id,
  r.ref,
  r.kind,
  r.title,
  r.source_ref,
  c.heading,
  c.content,
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

	rows, err := s.db.QueryContext(ctx, builder.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("query sections: %w", err)
	}
	defer func() { _ = rows.Close() }()

	result := make([]Section, 0)
	for rows.Next() {
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
			&metadataRaw,
		}
		if query.IncludeEmbeddings {
			scanArgs = append(scanArgs, &blob)
		}
		if err := rows.Scan(scanArgs...); err != nil {
			return nil, fmt.Errorf("scan section: %w", err)
		}
		section.Metadata, err = unmarshalMetadata(section.Ref, metadataRaw)
		if err != nil {
			return nil, err
		}
		if query.IncludeEmbeddings {
			section.Embedding, err = store.DecodeVectorBlob(blob)
			if err != nil {
				return nil, fmt.Errorf("decode section embedding for %s: %w", section.Ref, err)
			}
		}
		result = append(result, section)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate sections: %w", err)
	}
	return result, nil
}

// Search runs a text search against the opened snapshot.
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
	vectors, err := query.Embedder.EmbedQueries(ctx, []string{query.Text})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("embedder returned %d query vectors, want 1", len(vectors))
	}
	return s.SearchVector(ctx, VectorSearchQuery{
		Embedding: vectors[0],
		Limit:     query.Limit,
		Kinds:     query.Kinds,
	})
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

	queryBlob, err := store.EncodeVectorBlob(query.Embedding)
	if err != nil {
		return nil, fmt.Errorf("encode query vector: %w", err)
	}
	sqlText, args := buildSearchSQL(SearchQuery{
		Limit: query.Limit,
		Kinds: query.Kinds,
	}, queryBlob)
	rows, err := s.db.QueryContext(ctx, sqlText, args...)
	if err != nil {
		return nil, fmt.Errorf("run search query: %w", err)
	}
	defer func() { _ = rows.Close() }()

	hits := make([]SearchHit, 0, query.Limit)
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
	if len(hits) > query.Limit {
		hits = hits[:query.Limit]
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
