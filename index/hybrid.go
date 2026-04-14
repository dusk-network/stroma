package index

import (
	"sort"
	"strings"
	"unicode"
)

const rrfK = 60

// mergeRRF combines vector and FTS search hits using reciprocal rank fusion.
// The returned hits use the fused RRF score so callers can compare scores
// across vector-only, FTS-only, and blended results.
//
// Ties are broken by source count (hits found by both retrieval paths beat
// single-source hits) and then by best rank across sources, so FTS-only exact
// matches are not buried by vector-only hits at the same RRF score.
func mergeRRF(vectorHits, ftsHits []SearchHit, limit int) []SearchHit {
	rrfScores := make(map[int64]float64)
	sources := make(map[int64]int)  // number of retrieval paths that found this hit
	bestRank := make(map[int64]int) // best (lowest) rank across sources
	hitMap := make(map[int64]SearchHit)

	for rank, hit := range vectorHits {
		rrfScores[hit.ChunkID] += 1.0 / float64(rrfK+rank+1)
		sources[hit.ChunkID]++
		bestRank[hit.ChunkID] = rank
		hitMap[hit.ChunkID] = hit
	}
	for rank, hit := range ftsHits {
		rrfScores[hit.ChunkID] += 1.0 / float64(rrfK+rank+1)
		sources[hit.ChunkID]++
		if prev, ok := bestRank[hit.ChunkID]; !ok || rank < prev {
			bestRank[hit.ChunkID] = rank
		}
		if _, ok := hitMap[hit.ChunkID]; !ok {
			hitMap[hit.ChunkID] = hit
		}
	}

	type scored struct {
		id       int64
		rrfScore float64
		sources  int // how many retrieval paths found this hit
		best     int // best rank across sources (lower = better)
	}
	ranked := make([]scored, 0, len(rrfScores))
	for id, rs := range rrfScores {
		ranked = append(ranked, scored{id, rs, sources[id], bestRank[id]})
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].rrfScore != ranked[j].rrfScore {
			return ranked[i].rrfScore > ranked[j].rrfScore
		}
		// Prefer hits found by more retrieval paths.
		if ranked[i].sources != ranked[j].sources {
			return ranked[i].sources > ranked[j].sources
		}
		// Among same-source-count ties, prefer the better original rank.
		return ranked[i].best < ranked[j].best
	})
	if len(ranked) > limit {
		ranked = ranked[:limit]
	}

	result := make([]SearchHit, 0, len(ranked))
	for _, s := range ranked {
		hit := hitMap[s.id]
		hit.Score = s.rrfScore
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
