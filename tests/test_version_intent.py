from src.models import CodeSnippet, RetrievalCandidate
from src.query.analyzer import QueryAnalyzer
from src.versioning.version_filter import filter_candidates, merge_query_version, version_order_key


def _candidate(snippet_id: str, version: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        snippet=CodeSnippet(
            id=snippet_id,
            repository="repo",
            file_path="src/normalize.py",
            language="python",
            version=version,
            commit_id=snippet_id,
            function_name="normalize",
        )
    )


def test_numeric_version_order_is_semantic_not_lexical():
    assert version_order_key("v2") > version_order_key("v1")
    assert version_order_key("v10") > version_order_key("v2")
    assert version_order_key("v2.1") > version_order_key("v2.0")


def test_latest_query_selects_latest_numeric_version():
    selected = filter_candidates([_candidate("v1", "v1"), _candidate("v2", "v2")], {"version": "latest"})
    assert [item.snippet.id for item in selected] == ["v2"]


def test_query_without_version_does_not_add_version_filter():
    analysis = QueryAnalyzer().analyze("Where are browser form values trimmed and normalized?")
    filters = merge_query_version({}, explicit_version="all", version_intent=analysis.version_intent, requested_version=analysis.version)
    assert filters["version"] == "all"


def test_specific_query_adds_version_filter():
    analysis = QueryAnalyzer().analyze("show the v2 implementation")
    filters = merge_query_version({}, explicit_version="all", version_intent=analysis.version_intent, requested_version=analysis.version)
    assert filters["version"] == "v2"
