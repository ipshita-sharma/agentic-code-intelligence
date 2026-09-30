"""Information retrieval metrics used by the local benchmark."""

import math
from typing import Iterable, Mapping, Sequence


def reciprocal_rank(retrieved: Sequence[str], relevant: Mapping[str, float] | set[str]) -> float:
    relevant_ids = set(relevant)
    for rank, item in enumerate(retrieved, 1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Mapping[str, float] | set[str], k: int) -> float:
    relevant_ids = set(relevant)
    return len(set(retrieved[:k]) & relevant_ids) / max(1, len(relevant_ids))


def ndcg_at_k(retrieved: Sequence[str], relevance: Mapping[str, float], k: int = 10) -> float:
    def dcg(items: Iterable[float]) -> float:
        return sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(items))
    actual = dcg([float(relevance.get(item, 0.0)) for item in retrieved[:k]])
    ideal = dcg(sorted((float(value) for value in relevance.values()), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def evaluate_run(run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, float]], k: int = 10) -> dict[str, float]:
    if not qrels:
        return {"MRR": 0.0, f"NDCG@{k}": 0.0, "Recall@5": 0.0, f"Recall@{k}": 0.0}
    reciprocal_ranks = []
    ndcgs = []
    recalls_5 = []
    recalls_k = []
    for query_id, relevance in qrels.items():
        retrieved = list(run.get(query_id, []))
        reciprocal_ranks.append(reciprocal_rank(retrieved, relevance))
        ndcgs.append(ndcg_at_k(retrieved, relevance, k))
        recalls_5.append(recall_at_k(retrieved, relevance, 5))
        recalls_k.append(recall_at_k(retrieved, relevance, k))
    return {"MRR": sum(reciprocal_ranks) / len(reciprocal_ranks), f"NDCG@{k}": sum(ndcgs) / len(ndcgs), "Recall@5": sum(recalls_5) / len(recalls_5), f"Recall@{k}": sum(recalls_k) / len(recalls_k)}
