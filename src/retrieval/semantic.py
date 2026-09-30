"""Configurable CPU semantic encoders and validated embedding persistence."""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from ..config import RetrievalConfig
from ..models import CodeSnippet
from ..query.preprocessor import tokenize


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


class BaseEncoder(ABC):
    """Small encoder contract shared by hashing and Sentence Transformers."""

    name: str
    model_name: str
    dimension: int
    device: str

    @abstractmethod
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Encode document texts into row-normalized vectors."""

    @abstractmethod
    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        """Encode query texts into row-normalized vectors."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Compatibility alias retained for existing callers."""
        return self.encode_documents(texts)


class HashingEncoder(BaseEncoder):
    """Deterministic, zero-download semantic baseline for offline execution."""

    name = "hashing"
    device = "cpu"

    def __init__(self, dimensions: int = 384):
        self.dimension = dimensions
        self.dimensions = dimensions
        self.model_name = f"hashing-{dimensions}"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            # Add word and identifier-prefix/suffix features; sign hashing
            # reduces collisions while remaining deterministic.
            for token in tokens:
                for feature in (token, token[:4], token[-4:]):
                    digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                    index = int.from_bytes(digest[:4], "little") % self.dimension
                    sign = 1.0 if digest[4] & 1 else -1.0
                    matrix[row, index] += sign
        return _l2_normalize(matrix)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


class MiniLMEncoder(BaseEncoder):
    """CPU Sentence Transformers encoder using IR-specific query/document APIs."""

    name = "minilm"
    device = "cpu"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model=None, model_factory=None):
        self.model_name = model_name
        self.device = "cpu"
        if model is None:
            from sentence_transformers import SentenceTransformer
            factory = model_factory or SentenceTransformer
            model = factory(model_name, device="cpu")
        self.model = model
        self.device = str(getattr(model, "device", "cpu"))
        dimension_getter = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension")
        dimension = dimension_getter()
        if dimension is None:
            raise ValueError("Sentence Transformer did not expose an embedding dimension")
        self.dimension = int(dimension)

    def _encode(self, texts: Sequence[str], method_name: str) -> np.ndarray:
        method = getattr(self.model, method_name, None)
        if method is None:
            method = getattr(self.model, "encode")
        kwargs = {
            "batch_size": 32,
            "show_progress_bar": False,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        try:
            vectors = method(list(texts), **kwargs)
        except TypeError:
            # Supports lightweight test doubles and older compatible versions.
            vectors = method(list(texts))
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "encode_document")

    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "encode_query")


class SemanticIndex:
    """Semantic index with encoder/corpus-aware cache validation."""

    def __init__(self, snippets: Iterable[CodeSnippet], config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()
        self.snippets = list(snippets)
        self.ids = [snippet.id for snippet in self.snippets]
        started = time.perf_counter()
        self.encoder = self._make_encoder()
        self.encoder_init_seconds = time.perf_counter() - started
        self.embeddings: np.ndarray | None = None
        self.backend_name = self.encoder.name
        self.model_name = self.encoder.model_name
        self.embedding_dimension = self.encoder.dimension
        self.corpus_hash = self._compute_corpus_hash()
        self.cache_hit = False

    def _make_encoder(self) -> BaseEncoder:
        selected = self.config.selected_semantic_encoder
        if selected == "hashing":
            return HashingEncoder(self.config.hash_dimensions)
        if selected == "minilm":
            return MiniLMEncoder(self.config.semantic_model)
        raise ValueError(f"Unknown semantic encoder '{selected}'. Choose 'hashing' or 'minilm'.")

    @property
    def texts(self) -> List[str]:
        return [
            "\n".join(filter(None, [
                s.symbol_name,
                s.symbol_type,
                s.parent_symbol,
                s.function_name,
                s.class_name,
                s.docstring,
                " ".join(s.decorators),
                " ".join(s.comments),
                " ".join(s.imports),
                s.file_path,
                s.code,
            ]))
            for s in self.snippets
        ]

    def _compute_corpus_hash(self) -> str:
        digest = hashlib.sha256()
        for snippet_id, text in zip(self.ids, self.texts):
            digest.update(snippet_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(text.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _metadata_value(data, key: str):
        value = data[key]
        return value.item() if np.asarray(value).shape == () else value.tolist()

    def _cache_matches(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            data = np.load(path, allow_pickle=False)
            saved_ids = [str(item) for item in data["ids"].tolist()]
            saved_dimension = int(self._metadata_value(data, "embedding_dimension"))
            return (
                str(self._metadata_value(data, "encoder_name")) == self.backend_name
                and str(self._metadata_value(data, "model_name")) == self.model_name
                and saved_dimension == self.embedding_dimension
                and str(self._metadata_value(data, "corpus_hash")) == self.corpus_hash
                and int(self._metadata_value(data, "snippet_count")) == len(self.ids)
                and saved_ids == self.ids
            )
        except (OSError, KeyError, TypeError, ValueError):
            return False

    def build(self) -> float:
        cache_path = Path(self.config.embedding_cache)
        if self._cache_matches(cache_path):
            self.load(cache_path)
            self.cache_hit = True
            return 0.0
        started = time.perf_counter()
        self.embeddings = self.encoder.encode_documents(self.texts)
        if self.embeddings.shape != (len(self.ids), self.embedding_dimension):
            raise ValueError(f"Encoder returned shape {self.embeddings.shape}; expected {(len(self.ids), self.embedding_dimension)}")
        self.save(cache_path)
        self.cache_hit = False
        return time.perf_counter() - started

    def save(self, path: str | Path | None = None) -> None:
        if self.embeddings is None:
            raise RuntimeError("Semantic index has not been built")
        cache_path = Path(path or self.config.embedding_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=self.embeddings,
            ids=np.asarray(self.ids),
            encoder_name=np.asarray(self.backend_name),
            model_name=np.asarray(self.model_name),
            embedding_dimension=np.asarray(self.embedding_dimension),
            corpus_hash=np.asarray(self.corpus_hash),
            snippet_count=np.asarray(len(self.ids)),
        )

    def load(self, path: str | Path | None = None) -> None:
        cache_path = Path(path or self.config.embedding_cache)
        data = np.load(cache_path, allow_pickle=False)
        if not self._cache_matches(cache_path):
            raise ValueError("Embedding cache metadata does not match the current encoder, model, dimension, or snippet corpus")
        self.embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        self.cache_hit = True

    def search(self, query: str, top_n: int = 100) -> dict[str, float]:
        if self.embeddings is None:
            self.build()
        query_vector = self.encoder.encode_query([query])[0]
        scores = self.embeddings @ query_vector
        order = np.argsort(-scores)[:top_n]
        return {self.ids[int(index)]: float(scores[int(index)]) for index in order}
