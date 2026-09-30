"""Run independent retrieval passes and fuse their candidate pools."""

from ..config import RetrievalConfig
from ..models import CodeSnippet, RetrievalCandidate
from .fusion import fuse_candidates
from .lexical import BM25Index
from .semantic import SemanticIndex


class HybridRetriever:
    def __init__(self, snippets: list[CodeSnippet], config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()
        self.snippets = {snippet.id: snippet for snippet in snippets}
        self.lexical = BM25Index(self.snippets.values(), self.config)
        self.semantic = SemanticIndex(self.snippets.values(), self.config)

    def build(self) -> float:
        return self.semantic.build()

    def search(self, original_query: str, lexical_query: str | None = None) -> dict[str, RetrievalCandidate]:
        lexical_scores = self.lexical.search(lexical_query or original_query, self.config.lexical_candidates)
        semantic_scores = self.semantic.search(original_query, self.config.semantic_candidates)
        fused = fuse_candidates(self.snippets, lexical_scores, semantic_scores, self.config)
        return dict(list(fused.items())[: self.config.fusion_candidates])
