"""End-to-end retrieval-only engine."""

import json
import logging
import pickle
import shutil
import time
from pathlib import Path
from typing import Iterable, Mapping

from .config import RetrievalConfig
from .models import CodeSnippet, RetrievalCandidate, RetrievalResult
from .query.analyzer import QueryAnalyzer
from .query.preprocessor import preprocess_query
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import rerank
from .versioning.version_filter import filter_candidates, merge_query_version

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Modular multi-stage engine. It returns code snippets, never generated answers."""

    def __init__(self, snippets: Iterable[CodeSnippet], config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()
        self.snippets = list(snippets)
        if not self.snippets:
            raise ValueError("RetrievalEngine requires at least one code snippet")
        if len({snippet.id for snippet in self.snippets}) != len(self.snippets):
            raise ValueError("Snippet IDs must be unique")
        self.analyzer = QueryAnalyzer(self.config)
        self.retriever = HybridRetriever(self.snippets, self.config)
        self.last_timings: dict[str, float] = {}

    def build_index(self) -> dict[str, float | str]:
        start = time.perf_counter()
        embedding_seconds = self.retriever.build()
        elapsed = time.perf_counter() - start
        return {
            "index_build_seconds": elapsed + self.retriever.semantic.encoder_init_seconds,
            "embedding_seconds": embedding_seconds,
            "encoder_init_seconds": self.retriever.semantic.encoder_init_seconds,
            "cache_hit": self.retriever.semantic.cache_hit,
            "semantic_backend": self.retriever.semantic.backend_name,
            "semantic_model": self.retriever.semantic.model_name,
            "embedding_dimension": self.retriever.semantic.embedding_dimension,
            "snippet_count": len(self.snippets),
        }

    def save_index(self, index_dir: str | Path | None = None) -> Path:
        directory = Path(index_dir or self.config.index_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "snippets.json").open("w", encoding="utf-8") as handle:
            json.dump([snippet.to_dict() for snippet in self.snippets], handle, indent=2)
        with (directory / "config.json").open("w", encoding="utf-8") as handle:
            json.dump({key: str(value) if isinstance(value, Path) else value for key, value in self.config.__dict__.items() if key not in {"stopwords", "field_weights", "rerank_weights", "technical_aliases"}}, handle, indent=2)
        with (directory / "lexical.pkl").open("wb") as handle:
            pickle.dump(self.retriever.lexical, handle)
        cache_path = Path(self.config.embedding_cache)
        self.retriever.semantic.save(cache_path)
        # Keep the previous index-local artifact for older tooling, while all
        # validation and loads now use the configured authoritative cache path.
        legacy_cache = directory / "semantic.npz"
        if cache_path.resolve() != legacy_cache.resolve():
            shutil.copyfile(cache_path, legacy_cache)
        return directory

    @classmethod
    def load_index(cls, index_dir: str | Path, config: RetrievalConfig | None = None) -> "RetrievalEngine":
        directory = Path(index_dir)
        with (directory / "snippets.json").open(encoding="utf-8") as handle:
            snippets = [CodeSnippet.from_dict(item) for item in json.load(handle)]
        saved_config_path = directory / "config.json"
        saved_config = json.loads(saved_config_path.read_text(encoding="utf-8")) if saved_config_path.exists() else {}
        base_config = config or RetrievalConfig()
        # The query encoder must match the backend used to create the cached
        # vectors. Persisted index metadata takes precedence for that pair.
        saved_embedding_cache = saved_config.get("embedding_cache")
        config = RetrievalConfig(**{
            **base_config.__dict__,
            "semantic_encoder": saved_config.get("semantic_encoder", saved_config.get("semantic_backend", base_config.semantic_encoder)),
            "semantic_backend": saved_config.get("semantic_backend", base_config.semantic_backend),
            "semantic_model": saved_config.get("semantic_model", base_config.semantic_model),
            "embedding_cache": Path(saved_embedding_cache) if saved_embedding_cache else base_config.embedding_cache,
        })
        engine = cls(snippets, config)
        try:
            engine.retriever.semantic.load()
        except (FileNotFoundError, KeyError, ValueError):
            # Rebuild old/pre-metadata caches instead of silently mixing them
            # with a different encoder or corpus.
            engine.retriever.semantic.build()
            engine.save_index(directory)
        return engine

    def search(self, query: str, top_k: int | None = None, filters: Mapping[str, object] | None = None, version: str = "all") -> list[dict]:
        top_k = top_k or self.config.default_top_k
        start_total = time.perf_counter()
        analysis = self.analyzer.analyze(query)
        processed = preprocess_query(query, self.config)
        after_understanding = time.perf_counter()
        candidates = self.retriever.search(analysis.original_query, processed["lexical_query"])
        after_retrieval = time.perf_counter()
        merged_filters = merge_query_version(
            filters,
            explicit_version=version,
            version_intent=analysis.version_intent,
            requested_version=analysis.version,
        )
        filtered = filter_candidates(candidates.values(), merged_filters)
        after_filter = time.perf_counter()
        ranked = rerank(filtered, analysis, self.config) if self.config.enable_reranking else sorted(filtered, key=lambda item: -item.fusion_score)
        after_rerank = time.perf_counter()
        ranked = ranked[:top_k]
        results = []
        for rank, candidate in enumerate(ranked, 1):
            score = candidate.final_score if self.config.enable_reranking else candidate.fusion_score
            results.append(RetrievalResult(
                rank=rank, snippet_id=candidate.snippet.id, score=round(float(score), 6),
                file_path=candidate.snippet.file_path, function_name=candidate.snippet.function_name,
                class_name=candidate.snippet.class_name, language=candidate.snippet.language,
                version=candidate.snippet.version, repository=candidate.snippet.repository,
                code=candidate.snippet.code, retrieval_signals={
                    "semantic": round(candidate.semantic_score, 6), "lexical": round(candidate.lexical_score, 6),
                    "rrf": round(candidate.rrf_score, 6), "fusion": round(candidate.fusion_score, 6),
                    "rerank": round(candidate.final_score, 6), "features": candidate.rerank_features,
                    "intent": analysis.intent,
                    "primary_intent": analysis.primary_intent,
                    "intent_confidence": round(analysis.confidence, 6),
                    "intent_scores": analysis.intent_scores,
                    "version_intent": analysis.version_intent,
                    "requested_version": analysis.version,
                },
            ).to_dict())
        self.last_timings = {
            "query_understanding_seconds": after_understanding - start_total,
            "retrieval_seconds": after_retrieval - after_understanding,
            "filtering_seconds": after_filter - after_retrieval,
            "reranking_seconds": after_rerank - after_filter,
            "total_seconds": time.perf_counter() - start_total,
            "candidate_count": len(candidates), "filtered_count": len(filtered),
        }
        return results
