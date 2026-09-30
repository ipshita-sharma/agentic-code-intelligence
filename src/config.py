"""Configuration for the retrieval pipeline.

The defaults are intentionally laptop-friendly.  Set ``semantic_backend`` to
``sentence-transformers`` when the model is already available locally or after
installing the optional dependency; ``auto`` attempts it and falls back to the
deterministic hashing encoder.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class RetrievalConfig:
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_encoder: str = "hashing"  # hashing or minilm
    # Backward-compatible alias for older callers and persisted configs.
    semantic_backend: str | None = None
    embedding_cache: Path = Path("outputs/embeddings.npz")
    index_dir: Path = Path("outputs/index")
    lexical_candidates: int = 100
    semantic_candidates: int = 100
    fusion_candidates: int = 200
    default_top_k: int = 10
    rrf_k: int = 60
    fusion_method: str = "rrf"  # rrf or weighted
    lexical_weight: float = 0.45
    semantic_weight: float = 0.55
    field_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "function_name": 3.0,
            "class_name": 2.6,
            "docstring": 2.0,
            "imports": 1.5,
            "file_path": 1.4,
            "code": 1.0,
        }
    )
    rerank_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "semantic": 0.35,
            "lexical": 0.25,
            "identifier_overlap": 0.15,
            "intent_match": 0.10,
            "metadata_match": 0.10,
            "version_score": 0.05,
        }
    )
    # The audited feature reranker is intentionally opt-in.  Query
    # understanding must not silently re-enable it in the production path.
    enable_reranking: bool = False
    technical_aliases: Dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "preprocessing": (
                "preprocess", "preprocessing", "preprocessed", "normalize",
                "normalization", "normalized", "clean", "cleaned", "sanitize", "prepare",
                "transform", "trim", "trimmed", "standardize",
            ),
            "authentication": (
                "auth", "authenticate", "authentication", "authorization",
                "authorize", "jwt", "token", "bearer", "session", "credential",
                "validate", "validation", "expire", "expiration",
            ),
            "database": (
                "db", "database", "sql", "query", "insert", "select",
                "transaction", "connection", "jdbc", "orm", "datasource",
            ),
            "error_handling": (
                "error", "exception", "failure", "fallback", "recovery", "retry",
            ),
            "api": (
                "api", "endpoint", "request", "response", "payload", "http",
                "route", "rest", "graphql", "serialize",
            ),
        }
    )
    hash_dimensions: int = 384
    stopwords: frozenset[str] = frozenset(
        {
            "a", "an", "and", "are", "as", "at", "be", "before", "by",
            "for", "from", "how", "in", "is", "it", "of", "on", "or",
            "the", "to", "was", "were", "which", "where", "with", "this",
            "that", "does", "do", "did", "then", "into", "than", "via",
            "where", "how", "which", "who", "are", "was", "were",
        }
    )

    @property
    def selected_semantic_encoder(self) -> str:
        """Return the canonical encoder name, honoring legacy configuration."""
        selected = self.semantic_backend or self.semantic_encoder
        aliases = {
            "sentence-transformers": "minilm",
            "all-minilm-l6-v2": "minilm",
            "auto": "minilm",
        }
        return aliases.get(selected.lower(), selected.lower())

    def resolve(self, root: Path | None = None) -> "RetrievalConfig":
        """Return a copy whose relative paths are anchored at ``root``."""
        if root is None:
            return self
        return RetrievalConfig(**{
            **self.__dict__,
            "embedding_cache": root / self.embedding_cache,
            "index_dir": root / self.index_dir,
        })
