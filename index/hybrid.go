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
	k := r.K
	if k <= 0 {
		k = rrfK
	}

	for i := range arms {
		arm := arms[i]
		if arm.Name == "" {
			return nil, fmt.Errorf("retrieval arm at index %d has empty Name", i)
		}
		if arm.Available && arm.Err != nil {
			return nil, fmt.Errorf("retrieval arm %q: Available=true with non-nil Err", arm.Name)
		}
		if !arm.Available && len(arm.Hits) > 0 {
			return nil, fmt.Errorf("retrieval arm %q: Available=false with non-empty Hits", arm.Name)
		}
		// RRFFusion fails closed on upstream arm errors, matching pre-#17
		// Snapshot.Search behavior. Custom strategies that want partial-arm
		// tolerance implement it themselves.
		if arm.Err != nil {
			return nil, fmt.Errorf("retrieval arm %q failed: %w", arm.Name, arm.Err)
		}
	}

	nonEmpty := 0
	singleIdx := -1
	for i := range arms {
		if arms[i].Available && len(arms[i].Hits) > 0 {
			nonEmpty++
			singleIdx = i
		}
	}

	if nonEmpty == 0 {
		return nil, nil
	}

	if nonEmpty == 1 && r.PreserveSingleArmScore {
		arm := arms[singleIdx]
		out := make([]SearchHit, 0, min(len(arm.Hits), limit))
		for rank, hit := range arm.Hits {
			if rank >= limit {
				break
			}
			hit.Provenance = &HitProvenance{Arms: map[string]ArmEvidence{
				arm.Name: {Rank: rank, Score: hit.Score},
			}}
			out = append(out, hit)
		}
		return out, nil
	}

	type scored struct {
		id       int64
		rrfScore float64
		sources  int
		best     int
	}
	rrfScores := make(map[int64]float64)
	sources := make(map[int64]int)
	bestRank := make(map[int64]int)
	hitMap := make(map[int64]SearchHit)
	evidence := make(map[int64]map[string]ArmEvidence)

	for _, arm := range arms {
		if !arm.Available {
			continue
		}
		for rank, hit := range arm.Hits {
			rrfScores[hit.ChunkID] += 1.0 / float64(k+rank+1)
			sources[hit.ChunkID]++
			if prev, ok := bestRank[hit.ChunkID]; !ok || rank < prev {
				bestRank[hit.ChunkID] = rank
			}
			if _, ok := hitMap[hit.ChunkID]; !ok {
				hitMap[hit.ChunkID] = hit
			}
			if _, ok := evidence[hit.ChunkID]; !ok {
				evidence[hit.ChunkID] = make(map[string]ArmEvidence)
			}
			evidence[hit.ChunkID][arm.Name] = ArmEvidence{Rank: rank, Score: hit.Score}
		}
	}

	ranked := make([]scored, 0, len(rrfScores))
	for id, rs := range rrfScores {
		ranked = append(ranked, scored{id, rs, sources[id], bestRank[id]})
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].rrfScore != ranked[j].rrfScore {
			return ranked[i].rrfScore > ranked[j].rrfScore
		}
		if ranked[i].sources != ranked[j].sources {
			return ranked[i].sources > ranked[j].sources
		}
		return ranked[i].best < ranked[j].best
	})
	if len(ranked) > limit {
		ranked = ranked[:limit]
	}

	result := make([]SearchHit, 0, len(ranked))
	for _, s := range ranked {
		hit := hitMap[s.id]
		hit.Score = s.rrfScore
		hit.Provenance = &HitProvenance{Arms: evidence[s.id]}
		result = append(result, hit)
	}
	return result, nil
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
