from src.config import RetrievalConfig
from src.models import CodeSnippet
from src.pipeline import RetrievalEngine


def test_complete_pipeline_returns_ranked_metadata():
    snippets = [
        CodeSnippet(id="pre", repository="r", file_path="src/pre.py", language="python", version="v2", commit_id="2", function_name="normalize_input", code="def normalize_input(payload): return payload.strip()", docstring="normalize input before main", snippet_type="function"),
        CodeSnippet(id="other", repository="r", file_path="src/other.py", language="python", version="v2", commit_id="2", function_name="render", code="def render(): return html", snippet_type="function"),
    ]
    engine = RetrievalEngine(snippets, RetrievalConfig(semantic_backend="hashing"))
    engine.build_index()
    results = engine.search("How is input normalized before the main function?", top_k=1)
    assert results[0]["snippet_id"] == "pre"
    assert results[0]["rank"] == 1
    assert "retrieval_signals" in results[0]
    assert engine.last_timings["total_seconds"] >= 0
