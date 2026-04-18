package index

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// ContextOptions controls how Snapshot.ExpandContext widens a single chunk
// hit into a local-context payload.
type ContextOptions struct {
	// IncludeParent walks the requested chunk's parent_chunk_id one level
	// up and includes the parent row in the returned slice when the chunk
	// has a parent. Multi-level ancestry walks are explicit recursion by
	// the caller.
	//
	// Against snapshots built before schema v5 (#16), there is no
	// parent_chunk_id column to walk; IncludeParent is a no-op.
	IncludeParent bool

	// NeighborWindow includes up to N sibling chunks on each side of the
	// requested chunk, ordered by chunk_index. Two chunks are siblings
	// when they share the same parent_chunk_id (NULL counts as a single
	// sibling group), so for a leaf the neighborhood stays inside the
	// same parent span and for a flat or parent chunk the neighborhood
	// is other top-level chunks under the same record. Zero means no
	// neighbors are included; the requested chunk is still returned by
	// itself.
	//
	// Against snapshots built before schema v5 (#16), the parent grouping
	// is unavailable, so neighbors degrade to "other chunks in the same
	// record_ref with chunk_index in the requested window."
	NeighborWindow int
}

// ExpandContext returns the chunk identified by chunkID together with the
// caller-requested local context, in document order:
//
//	[parent (if IncludeParent and the chunk has one), neighbors before,
//	 the chunk itself, neighbors after]
//
// The chunk itself is always included, so callers do not have to
// reconcile the original SearchHit with the expansion. Embeddings are
// never populated by ExpandContext — the API is for context retrieval,
// not for re-ranking against fresh vectors. Callers that need
// embeddings should use Sections() with IncludeEmbeddings = true.
//
// Returns an empty slice + nil error when chunkID does not exist; the
// substrate treats "no such chunk" as an empty result rather than an
// error, matching the section-read APIs.
//
// Against snapshots built before schema v5 (#16), the v5 lineage
// column is absent: IncludeParent becomes a no-op and NeighborWindow
// scopes by record_ref alone (no parent grouping). ExpandContext stays
// useful on legacy files; it just cannot surface lineage that was
// never recorded.
func (s *Snapshot) ExpandContext(ctx context.Context, chunkID int64, opts ContextOptions) ([]Section, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("snapshot is not open")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if opts.NeighborWindow < 0 {
		opts.NeighborWindow = 0
	}

	recordRef, parentChunkID, chunkIndex, found, err := s.locateChunk(ctx, chunkID)
	if err != nil {
		return nil, err
	}
	if !found {
		return []Section{}, nil
	}

	ids := map[int64]struct{}{chunkID: {}}

	includeParent := opts.IncludeParent && s.hasParentChunkID && parentChunkID.Valid
	if includeParent {
		ids[parentChunkID.Int64] = struct{}{}
	}

	if opts.NeighborWindow > 0 {
		siblings, err := s.loadSiblingIDs(ctx, recordRef, parentChunkID, chunkIndex, opts.NeighborWindow)
		if err != nil {
			return nil, err
		}
		for _, id := range siblings {
			ids[id] = struct{}{}
		}
	}

	return s.loadContextSections(ctx, ids, parentChunkID, includeParent)
}

// locateChunk reads the (record_ref, parent_chunk_id, chunk_index) of the
// requested chunk. Returns found=false (with nil error) when the chunk
// does not exist, so callers can short-circuit to an empty result.
//
// On legacy snapshots that lack the parent_chunk_id column, the parent
// projection is replaced with a NULL literal so the SELECT still parses
// and the returned NullInt64 reads as invalid.
func (s *Snapshot) locateChunk(ctx context.Context, chunkID int64) (recordRef string, parentChunkID sql.NullInt64, chunkIndex int64, found bool, err error) {
	parentExpr := "NULL"
	if s.hasParentChunkID {
		parentExpr = "parent_chunk_id"
	}
	query := fmt.Sprintf(`SELECT record_ref, %s, chunk_index FROM chunks WHERE id = ?`, parentExpr)
	scanErr := s.db.QueryRowContext(ctx, query, chunkID).Scan(&recordRef, &parentChunkID, &chunkIndex)
	if errors.Is(scanErr, sql.ErrNoRows) {
		return "", sql.NullInt64{}, 0, false, nil
	}
	if scanErr != nil {
		return "", sql.NullInt64{}, 0, false, fmt.Errorf("locate chunk %d: %w", chunkID, scanErr)
	}
	return recordRef, parentChunkID, chunkIndex, true, nil
}

// loadSiblingIDs returns the chunk IDs of siblings of the requested
// chunk within the chunk_index window [center-N, center+N]. "Sibling"
// is determined by (record_ref, parent_chunk_id) on v5 snapshots and
// by record_ref alone on legacy snapshots. The IN clause caller
// deduplicates, so loadSiblingIDs may return the requested chunk's own
// id without harm.
func (s *Snapshot) loadSiblingIDs(ctx context.Context, recordRef string, parentChunkID sql.NullInt64, center int64, window int) ([]int64, error) {
	low := center - int64(window)
	high := center + int64(window)

	var (
		query string
		args  []any
	)
	switch {
	case s.hasParentChunkID && parentChunkID.Valid:
		query = `SELECT id FROM chunks
WHERE record_ref = ? AND parent_chunk_id = ?
  AND chunk_index >= ? AND chunk_index <= ?
ORDER BY chunk_index ASC, id ASC`
		args = []any{recordRef, parentChunkID.Int64, low, high}
	case s.hasParentChunkID:
		query = `SELECT id FROM chunks
WHERE record_ref = ? AND parent_chunk_id IS NULL
  AND chunk_index >= ? AND chunk_index <= ?
ORDER BY chunk_index ASC, id ASC`
		args = []any{recordRef, low, high}
	default:
		// Legacy snapshot: no parent grouping is available, so the
		// neighborhood is scoped by record_ref + chunk_index window.
		query = `SELECT id FROM chunks
WHERE record_ref = ?
  AND chunk_index >= ? AND chunk_index <= ?
ORDER BY chunk_index ASC, id ASC`
		args = []any{recordRef, low, high}
	}

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("load sibling chunk ids: %w", err)
	}
	defer func() { _ = rows.Close() }()

	ids := make([]int64, 0, 2*window+1)
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("scan sibling id: %w", err)
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate sibling ids: %w", err)
	}
	return ids, nil
}

// loadContextSections fetches the rows for the collected chunk IDs and
// returns them in document order. The parent (when included) is placed
// first; the rest sort by (chunk_index ASC, id ASC). Embeddings are
// never populated.
func (s *Snapshot) loadContextSections(ctx context.Context, ids map[int64]struct{}, parentChunkID sql.NullInt64, includeParent bool) ([]Section, error) {
	if len(ids) == 0 {
		return []Section{}, nil
	}

	placeholders := make([]string, 0, len(ids))
	args := make([]any, 0, len(ids))
	for id := range ids {
		placeholders = append(placeholders, "?")
		args = append(args, id)
	}

	// SQL is assembled via strings.Builder rather than fmt.Sprintf so
	// gosec G201 stays clean. Every fragment written here is either a
	// hard-coded literal or a package-local constant
	// (contextPrefixSelectExpr return value, "?" placeholders); no user
	// data is concatenated into the SQL text. User data is passed
	// exclusively via the args slice to QueryContext below.
	prefixExpr := contextPrefixSelectExpr(s.hasContextPrefix)
	var qb strings.Builder
	qb.WriteString(`
SELECT
  c.id,
  r.ref,
  r.kind,
  r.title,
  r.source_ref,
  c.heading,
  c.content,
  `)
	qb.WriteString(prefixExpr)
	qb.WriteString(`,
  r.metadata_json,
  c.chunk_index
FROM chunks c
JOIN records r ON r.ref = c.record_ref
WHERE c.id IN (`)
	qb.WriteString(strings.Join(placeholders, ", "))
	qb.WriteString(`)`)
	query := qb.String()

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("load context section rows: %w", err)
	}
	defer func() { _ = rows.Close() }()

	// Pointer-valued map so neither the map iteration nor the rest-slice
	// build incurs a per-iteration copy of the ~160-byte Section struct
	// (the gocritic rangeValCopy lint catches this).
	type row struct {
		section    Section
		chunkIndex int64
	}
	loaded := make(map[int64]*row, len(ids))
	for rows.Next() {
		r := &row{}
		var metadataJSON string
		if err := rows.Scan(
			&r.section.ChunkID,
			&r.section.Ref,
			&r.section.Kind,
			&r.section.Title,
			&r.section.SourceRef,
			&r.section.Heading,
			&r.section.Content,
			&r.section.ContextPrefix,
			&metadataJSON,
			&r.chunkIndex,
		); err != nil {
			return nil, fmt.Errorf("scan context section row: %w", err)
		}
		r.section.Metadata, err = unmarshalMetadata(r.section.Ref, metadataJSON)
		if err != nil {
			return nil, err
		}
		loaded[r.section.ChunkID] = r
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate context section rows: %w", err)
	}

	out := make([]Section, 0, len(loaded))
	if includeParent && parentChunkID.Valid {
		if r, ok := loaded[parentChunkID.Int64]; ok {
			out = append(out, r.section)
			delete(loaded, parentChunkID.Int64)
		}
	}
	rest := make([]*row, 0, len(loaded))
	for _, r := range loaded {
		rest = append(rest, r)
	}
	sort.Slice(rest, func(i, j int) bool {
		if rest[i].chunkIndex != rest[j].chunkIndex {
			return rest[i].chunkIndex < rest[j].chunkIndex
		}
		return rest[i].section.ChunkID < rest[j].section.ChunkID
	})
	for _, r := range rest {
		out = append(out, r.section)
	}
	return out, nil
}
