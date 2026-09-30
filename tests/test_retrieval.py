from src.config import RetrievalConfig
from src.models import CodeSnippet
from src.retrieval.fusion import fuse_candidates
from src.retrieval.lexical import BM25Index
from src.retrieval.reranker import rerank
from src.retrieval.semantic import SemanticIndex
from src.query.analyzer import QueryAnalyzer
from src.versioning.version_filter import filter_candidates


def snippets():
    return [
        CodeSnippet(id="a", repository="r", file_path="src/auth.py", language="python", version="v1", commit_id="1", function_name="validate_token", code="def validate_token(token): return jwt.decode(token)", snippet_type="function"),
        CodeSnippet(id="b", repository="r", file_path="src/db.py", language="python", version="v2", commit_id="2", function_name="connect_with_retry", code="def connect_with_retry(): retry database", snippet_type="function"),
    ]


def test_bm25_retrieval_and_semantic_retrieval():
    config = RetrievalConfig(semantic_backend="hashing")
    data = snippets()
    lexical = BM25Index(data, config).search("validate token", 2)
    assert list(lexical)[0] == "a"
    semantic_index = SemanticIndex(data, config)
    semantic_index.build()
    assert list(semantic_index.search("database retry", 2))[0] == "b"


def test_fusion_filtering_and_reranking():
    data = snippets()
    candidates = fuse_candidates({item.id: item for item in data}, {"a": 2.0, "b": 1.0}, {"b": 0.9, "a": 0.2}, RetrievalConfig())
    filtered = filter_candidates(candidates.values(), {"version": "v2"})
    assert [item.snippet.id for item in filtered] == ["b"]
    ranked = rerank(filtered, QueryAnalyzer().analyze("database retry"))
    assert ranked[0].snippet.id == "b"
