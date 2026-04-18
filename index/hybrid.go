package index

import (
	"fmt"
	"sort"
	"strings"
	"unicode"
)

const rrfK = 60

// RRFFusion is the default FusionStrategy. K controls the RRF constant;
// K<=0 is treated as K=60 for backward compatibility with the pre-#17
// mergeRRF helper.
//
// PreserveSingleArmScore controls the single-arm degenerate case. When
// true (the default used by DefaultFusion) and exactly one arm is
// available-and-non-empty, Fuse returns that arm's hits in arm order with
// arm-native Score preserved. When false, Fuse rewrites Score to the
// RRF-derived 1/(K+rank+1) on every path. Callers that want numerically
// uniform fused scores across single-arm and multi-arm paths opt in by
// setting this to false.
type RRFFusion struct {
	K                      int
	PreserveSingleArmScore bool
}

// Fuse implements FusionStrategy. See RRFFusion for the single-arm
// contract. Ties in RRF score are broken by (more contributing arms first)
// then (better cross-arm rank first).
func (r RRFFusion) Fuse(arms []RetrievalArm, limit int) ([]SearchHit, error) {
	if limit <= 0 {
		return nil, nil
	}
	if err := validateArms(arms); err != nil {
		return nil, err
	}
	k := r.K
	if k <= 0 {
		k = rrfK
	}

	nonEmpty, singleIdx := countAvailableNonEmpty(arms)
	if nonEmpty == 0 {
		return nil, nil
	}
	if nonEmpty == 1 && r.PreserveSingleArmScore {
		return fuseSingleArm(arms[singleIdx], limit), nil
	}
	return fuseRRF(arms, k, limit), nil
}

// validateArms rejects malformed RetrievalArm inputs before any scoring
// happens so custom FusionStrategy implementations (and the default
// RRFFusion) see a consistent contract. Duplicate arm names are rejected
// because HitProvenance.Arms is keyed by name; silently overwriting would
// corrupt provenance evidence for shared hits.
func validateArms(arms []RetrievalArm) error {
	seen := make(map[string]struct{}, len(arms))
	for i := range arms {
		arm := &arms[i]
		if arm.Name == "" {
			return fmt.Errorf("retrieval arm at index %d has empty Name", i)
		}
		if _, dup := seen[arm.Name]; dup {
			return fmt.Errorf("retrieval arm %q appears more than once", arm.Name)
		}
		seen[arm.Name] = struct{}{}
		if arm.Available && arm.Err != nil {
			return fmt.Errorf("retrieval arm %q: Available=true with non-nil Err", arm.Name)
		}
		if !arm.Available && len(arm.Hits) > 0 {
			return fmt.Errorf("retrieval arm %q: Available=false with non-empty Hits", arm.Name)
		}
		// RRFFusion fails closed on upstream arm errors, matching pre-#17
		// Snapshot.Search behavior. Custom strategies that want partial-arm
		// tolerance implement it themselves.
		if arm.Err != nil {
			return fmt.Errorf("retrieval arm %q failed: %w", arm.Name, arm.Err)
		}
	}
	return nil
}

func countAvailableNonEmpty(arms []RetrievalArm) (count, singleIdx int) {
	singleIdx = -1
	for i := range arms {
		if arms[i].Available && len(arms[i].Hits) > 0 {
			count++
			singleIdx = i
		}
	}
	return count, singleIdx
}

// fuseSingleArm returns the arm's hits in arm order with arm-native Score
// preserved and HitProvenance populated. Used when exactly one arm is
// available-and-non-empty and PreserveSingleArmScore is true.
func fuseSingleArm(arm RetrievalArm, limit int) []SearchHit {
	out := make([]SearchHit, 0, min(len(arm.Hits), limit))
	for rank := range arm.Hits {
		if rank >= limit {
			break
		}
		hit := arm.Hits[rank]
		hit.Provenance = &HitProvenance{Arms: map[string]ArmEvidence{
			arm.Name: {Rank: rank, Score: hit.Score},
		}}
		out = append(out, hit)
	}
	return out
}

// fuseRRF runs the full RRF scoring pass across all available arms. Ties
// are broken by source count (more contributing arms first) then by best
// cross-arm rank (lower is better).
func fuseRRF(arms []RetrievalArm, k, limit int) []SearchHit {
	type aggregate struct {
		hit      SearchHit
		rrfScore float64
		sources  int
		bestRank int
		evidence map[string]ArmEvidence
	}
	agg := make(map[int64]*aggregate)
	for i := range arms {
		arm := &arms[i]
		if !arm.Available {
			continue
		}
		for rank := range arm.Hits {
			hit := arm.Hits[rank]
			entry, ok := agg[hit.ChunkID]
			if !ok {
				entry = &aggregate{
					hit:      hit,
					bestRank: rank,
					evidence: make(map[string]ArmEvidence),
				}
				agg[hit.ChunkID] = entry
			}
			entry.rrfScore += 1.0 / float64(k+rank+1)
			entry.sources++
			if rank < entry.bestRank {
				entry.bestRank = rank
			}
			entry.evidence[arm.Name] = ArmEvidence{Rank: rank, Score: hit.Score}
		}
	}

	ranked := make([]*aggregate, 0, len(agg))
	for _, entry := range agg {
		ranked = append(ranked, entry)
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].rrfScore != ranked[j].rrfScore {
			return ranked[i].rrfScore > ranked[j].rrfScore
		}
		if ranked[i].sources != ranked[j].sources {
			return ranked[i].sources > ranked[j].sources
		}
		return ranked[i].bestRank < ranked[j].bestRank
	})
	if len(ranked) > limit {
		ranked = ranked[:limit]
	}

	result := make([]SearchHit, 0, len(ranked))
	for _, entry := range ranked {
		hit := entry.hit
		hit.Score = entry.rrfScore
		hit.Provenance = &HitProvenance{Arms: entry.evidence}
		result = append(result, hit)
	}
	return result
}

// sanitizeFTSQuery converts free text into an FTS5 AND query with quoted
// tokens. It splits on the same boundaries as the default unicode61 tokenizer
// (non-letter, non-digit characters) so identifiers like ERR_TIMEOUT_42
// produce individual tokens that match the indexed content. Using AND ensures
// all tokens must appear, preventing false positives from partial matches.
func sanitizeFTSQuery(text string) string {
	tokens := tokenizeFTS(text)
	if len(tokens) == 0 {
		return ""
	}
	parts := make([]string, 0, len(tokens))
	for _, t := range tokens {
		parts = append(parts, `"`+t+`"`)
	}
	return strings.Join(parts, " AND ")
}

// tokenizeFTS splits text into lowercase tokens on the same boundaries as the
// FTS5 unicode61 tokenizer: any character that is not a letter or digit is a
// separator.
func tokenizeFTS(text string) []string {
	var tokens []string
	var cur strings.Builder
	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur.WriteRune(r)
		} else if cur.Len() > 0 {
			tokens = append(tokens, cur.String())
			cur.Reset()
		}
	}
	if cur.Len() > 0 {
		tokens = append(tokens, cur.String())
	}
	return tokens
}
