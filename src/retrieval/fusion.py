"""Candidate fusion utilities."""

from typing import Dict

from ..config import RetrievalConfig
from ..models import CodeSnippet, RetrievalCandidate


def _normalize(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high - low < 1e-12:
        return {key: 1.0 if high > 0 else 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def fuse_candidates(
    snippets: Dict[str, CodeSnippet],
    lexical_scores: Dict[str, float],
    semantic_scores: Dict[str, float],
    config: RetrievalConfig | None = None,
) -> Dict[str, RetrievalCandidate]:
    config = config or RetrievalConfig()
    lexical_norm = _normalize(lexical_scores)
    semantic_norm = _normalize(semantic_scores)
    lexical_rank = {key: rank for rank, key in enumerate(lexical_scores, 1)}
    semantic_rank = {key: rank for rank, key in enumerate(semantic_scores, 1)}
    ids = set(lexical_scores) | set(semantic_scores)
    result: Dict[str, RetrievalCandidate] = {}
    for snippet_id in ids:
        rrf = 0.0
        if snippet_id in lexical_rank:
            rrf += config.lexical_weight / (config.rrf_k + lexical_rank[snippet_id])
        if snippet_id in semantic_rank:
            rrf += config.semantic_weight / (config.rrf_k + semantic_rank[snippet_id])
        weighted = config.lexical_weight * lexical_norm.get(snippet_id, 0.0) + config.semantic_weight * semantic_norm.get(snippet_id, 0.0)
        fusion_score = rrf if config.fusion_method == "rrf" else weighted
        result[snippet_id] = RetrievalCandidate(
            snippet=snippets[snippet_id], lexical_score=lexical_norm.get(snippet_id, 0.0),
            semantic_score=semantic_norm.get(snippet_id, 0.0), lexical_rank=lexical_rank.get(snippet_id),
            semantic_rank=semantic_rank.get(snippet_id), rrf_score=rrf, fusion_score=fusion_score,
        )
    return dict(sorted(result.items(), key=lambda item: (-item[1].fusion_score, item[0])))
