package index

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
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
//
// Internally ExpandContext issues a small bounded number of
// parameterized reads: at most one to locate the requested chunk, one
// to fetch the parent (when IncludeParent + parent_chunk_id present),
// and one range scan over the sibling window. There is no per-result
// parameter expansion (no `WHERE id IN (?, ?, ?, ...)`), so the
// query never approaches SQLite's parameter cap regardless of
// NeighborWindow.
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

	var parentSection *Section
	if opts.IncludeParent && s.hasParentChunkID && parentChunkID.Valid {
		// Defense in depth against a malformed or tampered snapshot. The
		// v5 same-record triggers (#16) reject cross-record parent links
		// on INSERT/UPDATE, so a well-formed snapshot cannot reach this
		// branch with a mismatch. ExpandContext nonetheless requires the
		// parent row to share record_ref before surfacing it: the
		// load query bakes that constraint into its WHERE clause, so a
		// cross-record parent simply returns nil without an error.
		parentSection, err = s.loadParentSection(ctx, parentChunkID.Int64, recordRef)
		if err != nil {
			return nil, err
		}
	}

	windowSections, err := s.loadContextWindow(ctx, recordRef, parentChunkID, chunkIndex, opts.NeighborWindow)
	if err != nil {
		return nil, err
	}

	out := make([]Section, 0, len(windowSections)+1)
	if parentSection != nil {
		out = append(out, *parentSection)
	}
	out = append(out, windowSections...)
	return out, nil
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
	var qb strings.Builder
	qb.WriteString(`SELECT record_ref, `)
	qb.WriteString(parentExpr)
	qb.WriteString(`, chunk_index FROM chunks WHERE id = ?`)
	scanErr := s.db.QueryRowContext(ctx, qb.String(), chunkID).Scan(&recordRef, &parentChunkID, &chunkIndex)
	if errors.Is(scanErr, sql.ErrNoRows) {
		return "", sql.NullInt64{}, 0, false, nil
	}
	if scanErr != nil {
		return "", sql.NullInt64{}, 0, false, fmt.Errorf("locate chunk %d: %w", chunkID, scanErr)
	}
	return recordRef, parentChunkID, chunkIndex, true, nil
}

// loadParentSection fetches the parent row identified by parentID, but
// only if that row's record_ref matches expectedRecordRef. Returns
// (nil, nil) when the parent does not exist or belongs to a different
// record — both cases are non-errors from the caller's point of view.
// Embedding is never populated.
func (s *Snapshot) loadParentSection(ctx context.Context, parentID int64, expectedRecordRef string) (*Section, error) {
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
  r.metadata_json
FROM chunks c
JOIN records r ON r.ref = c.record_ref
WHERE c.id = ? AND c.record_ref = ?`)

	row := s.db.QueryRowContext(ctx, qb.String(), parentID, expectedRecordRef)
	var (
		section      Section
		metadataJSON []byte
	)
	scanErr := row.Scan(
		&section.ChunkID,
		&section.Ref,
		&section.Kind,
		&section.Title,
		&section.SourceRef,
		&section.Heading,
		&section.Content,
		&section.ContextPrefix,
		&metadataJSON,
	)
	if errors.Is(scanErr, sql.ErrNoRows) {
		return nil, nil
	}
	if scanErr != nil {
		return nil, fmt.Errorf("load parent chunk %d: %w", parentID, scanErr)
	}
	metadata, err := unmarshalMetadata(section.Ref, metadataJSON)
	if err != nil {
		return nil, err
	}
	section.Metadata = metadata
	return &section, nil
}

// loadContextWindow scans the chunks of recordRef whose chunk_index
// falls within [center-window, center+window] AND whose parent_chunk_id
// matches the same sibling group as the requested chunk. The result
// always includes the requested chunk itself (window=0 → single-row
// result). Embeddings are not populated.
//
// Sibling grouping is determined by the snapshot's lineage state:
//
//   - v5 leaf (parentChunkID.Valid): siblings share the same
//     parent_chunk_id.
//   - v5 root (parentChunkID is NULL): siblings are other rows under
//     the same record_ref with parent_chunk_id IS NULL — the bucket
//     where flat chunks and parent rows live.
//   - legacy (no parent column): siblings are any rows under the same
//     record_ref. Parent grouping is unavailable.
//
// All SQL fragments are package-local literals; user data flows only
// through the args slice, so gosec G201 does not apply.
func (s *Snapshot) loadContextWindow(ctx context.Context, recordRef string, parentChunkID sql.NullInt64, center int64, window int) ([]Section, error) {
	low := center - int64(window)
	high := center + int64(window)

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
  r.metadata_json
FROM chunks c
JOIN records r ON r.ref = c.record_ref
WHERE c.record_ref = ?
  AND c.chunk_index >= ? AND c.chunk_index <= ?`)

	args := []any{recordRef, low, high}
	switch {
	case s.hasParentChunkID && parentChunkID.Valid:
		qb.WriteString(`
  AND c.parent_chunk_id = ?`)
		args = append(args, parentChunkID.Int64)
	case s.hasParentChunkID:
		qb.WriteString(`
  AND c.parent_chunk_id IS NULL`)
		// no extra arg
	default:
		// Legacy snapshot: no parent grouping is available, so the
		// neighborhood is scoped by record_ref + chunk_index window
		// only. No additional predicate.
	}
	qb.WriteString(`
ORDER BY c.chunk_index ASC, c.id ASC`)

	rows, err := s.db.QueryContext(ctx, qb.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("load context window: %w", err)
	}
	defer func() { _ = rows.Close() }()

	out := make([]Section, 0, 2*window+1)
	for rows.Next() {
		var (
			section      Section
			metadataJSON []byte
		)
		if err := rows.Scan(
			&section.ChunkID,
			&section.Ref,
			&section.Kind,
			&section.Title,
			&section.SourceRef,
			&section.Heading,
			&section.Content,
			&section.ContextPrefix,
			&metadataJSON,
		); err != nil {
			return nil, fmt.Errorf("scan context window row: %w", err)
		}
		section.Metadata, err = unmarshalMetadata(section.Ref, metadataJSON)
		if err != nil {
			return nil, err
		}
		out = append(out, section)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate context window rows: %w", err)
	}
	return out, nil
}
