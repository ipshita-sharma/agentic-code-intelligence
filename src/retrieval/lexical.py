"""Weighted-field BM25 retrieval with a small dependency-free implementation."""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

from ..config import RetrievalConfig
from ..models import CodeSnippet
from ..query.preprocessor import tokenize


class BM25Index:
    """BM25 over separately weighted metadata/code fields.

    ``rank-bm25`` is an excellent drop-in alternative, but keeping the core
    scorer here makes the hackathon prototype reproducible without a compiled
    dependency and allows field weighting directly.
    """

    def __init__(self, snippets: Iterable[CodeSnippet], config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()
        self.snippets = list(snippets)
        self.fields = tuple(self.config.field_weights)
        self._documents: Dict[str, Dict[str, List[str]]] = {}
        self._idf: Dict[str, Dict[str, float]] = {}
        self._avgdl: Dict[str, float] = {}
        self._build()

    def _field_text(self, snippet: CodeSnippet, field: str) -> str:
        value = getattr(snippet, field, "")
        if isinstance(value, list):
            return " ".join(value)
        return str(value)

    def _build(self) -> None:
        n = len(self.snippets)
        for field in self.fields:
            document_frequency = Counter()
            lengths = []
            for snippet in self.snippets:
                tokens = tokenize(self._field_text(snippet, field))
                self._documents.setdefault(snippet.id, {})[field] = tokens
                document_frequency.update(set(tokens))
                lengths.append(len(tokens))
            self._idf[field] = {term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in document_frequency.items()}
            self._avgdl[field] = sum(lengths) / max(1, len(lengths))

    def search(self, query: str, top_n: int = 100) -> Dict[str, float]:
        query_tokens = tokenize(query)
        raw_scores: Dict[str, float] = defaultdict(float)
        k1, b = 1.5, 0.75
        for snippet in self.snippets:
            for field, weight in self.config.field_weights.items():
                tokens = self._documents[snippet.id][field]
                if not tokens:
                    continue
                frequencies = Counter(tokens)
                dl = len(tokens)
                score = 0.0
                for term in query_tokens:
                    if term not in frequencies:
                        continue
                    tf = frequencies[term]
                    idf = self._idf[field].get(term, 0.0)
                    denominator = tf + k1 * (1 - b + b * dl / max(1.0, self._avgdl[field]))
                    score += idf * tf * (k1 + 1) / denominator
                raw_scores[snippet.id] += weight * score
        return dict(sorted(raw_scores.items(), key=lambda item: (-item[1], item[0]))[:top_n])
