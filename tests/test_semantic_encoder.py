import numpy as np
import pytest

from src.config import RetrievalConfig
from src.models import CodeSnippet
from src.retrieval import semantic as semantic_module
from src.retrieval.semantic import HashingEncoder, MiniLMEncoder, SemanticIndex


class FakeSentenceTransformer:
    device = "cpu"

    def get_sentence_embedding_dimension(self):
        return 3

    def encode_document(self, texts, **kwargs):
        return np.tile(np.array([[3.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))

    def encode_query(self, texts, **kwargs):
        return np.tile(np.array([[0.0, 4.0, 0.0]], dtype=np.float32), (len(texts), 1))


class FakeMiniLM:
    name = "minilm"
    dimension = 3
    device = "cpu"

    def __init__(self, model_name):
        self.model_name = model_name

    def encode_documents(self, texts):
        return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))

    def encode_query(self, texts):
        return np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def make_snippets(extra=False):
    values = [CodeSnippet(id="one", repository="r", file_path="one.py", language="python", version="v1", commit_id="1", code="def one(): return 1")]
    if extra:
        values.append(CodeSnippet(id="two", repository="r", file_path="two.py", language="python", version="v1", commit_id="2", code="def two(): return 2"))
    return values


def test_encoder_selection_and_hashing_baseline(tmp_path):
    config = RetrievalConfig(semantic_encoder="hashing", embedding_cache=tmp_path / "hashing.npz")
    index = SemanticIndex(make_snippets(), config)
    assert isinstance(index.encoder, HashingEncoder)
    assert index.encoder.device == "cpu"
    index.build()
    assert index.embeddings.shape == (1, 384)
    assert np.isclose(np.linalg.norm(index.embeddings[0]), 1.0)


def test_minilm_initialization_shape_normalization_and_cpu():
    encoder = MiniLMEncoder(model=FakeSentenceTransformer())
    documents = encoder.encode_documents(["document"])
    queries = encoder.encode_query(["query"])
    assert encoder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert encoder.device == "cpu"
    assert documents.shape == (1, 3)
    assert queries.shape == (1, 3)
    assert np.isclose(np.linalg.norm(documents[0]), 1.0)
    assert np.isclose(np.linalg.norm(queries[0]), 1.0)


def test_cache_metadata_and_cache_hit(tmp_path):
    cache = tmp_path / "embeddings.npz"
    config = RetrievalConfig(semantic_encoder="hashing", embedding_cache=cache)
    first = SemanticIndex(make_snippets(), config)
    first.build()
    metadata = np.load(cache, allow_pickle=False)
    assert str(metadata["encoder_name"].item()) == "hashing"
    assert str(metadata["model_name"].item()) == "hashing-384"
    assert int(metadata["embedding_dimension"].item()) == 384
    assert int(metadata["snippet_count"].item()) == 1
    assert len(str(metadata["corpus_hash"].item())) == 64

    second = SemanticIndex(make_snippets(), config)
    second.build()
    assert second.cache_hit is True
    np.testing.assert_array_equal(first.embeddings, second.embeddings)


def test_cache_invalidates_when_model_or_corpus_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic_module, "MiniLMEncoder", FakeMiniLM)
    cache = tmp_path / "embeddings.npz"
    first_config = RetrievalConfig(semantic_encoder="minilm", semantic_model="model-a", embedding_cache=cache)
    first = SemanticIndex(make_snippets(), first_config)
    first.build()

    changed_model = SemanticIndex(make_snippets(), RetrievalConfig(semantic_encoder="minilm", semantic_model="model-b", embedding_cache=cache))
    changed_model.build()
    assert changed_model.cache_hit is False

    changed_corpus = SemanticIndex(make_snippets(extra=True), first_config)
    changed_corpus.build()
    assert changed_corpus.cache_hit is False
    assert changed_corpus.embeddings.shape == (2, 3)


def test_cache_load_rejects_metadata_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic_module, "MiniLMEncoder", FakeMiniLM)
    cache = tmp_path / "embeddings.npz"
    SemanticIndex(make_snippets(), RetrievalConfig(semantic_encoder="minilm", semantic_model="model-a", embedding_cache=cache)).build()
    different = SemanticIndex(make_snippets(), RetrievalConfig(semantic_encoder="minilm", semantic_model="model-b", embedding_cache=cache))
    with pytest.raises(ValueError, match="metadata"):
        different.load()
