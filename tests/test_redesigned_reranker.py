from src.config import RetrievalConfig
from src.models import CodeSnippet, RetrievalCandidate
from src.query.analyzer import QueryAnalyzer
from src.retrieval.reranker import MAX_ADJUSTMENT, rerank


def candidate(
    snippet_id: str,
    *,
    version: str = "v1",
    file_path: str = "src/example.py",
    function_name: str = "example",
    docstring: str = "",
    code: str = "def example(): pass",
    fusion_score: float = 0.015,
) -> RetrievalCandidate:
    snippet = CodeSnippet(
        id=snippet_id,
        repository="repo",
        file_path=file_path,
        language="python",
        version=version,
        commit_id=version,
        function_name=function_name,
        code=code,
        docstring=docstring,
        snippet_type="function",
    )
    return RetrievalCandidate(
        snippet=snippet,
        semantic_score=0.8,
        lexical_score=0.7,
        fusion_score=fusion_score,
    )


def rerank_config(alpha: float = 1.0) -> RetrievalConfig:
    config = RetrievalConfig()
    config.rerank_alpha = alpha
    return config


def test_no_version_intent_produces_zero_version_adjustment():
    analysis = QueryAnalyzer().analyze("Where are browser form values trimmed and normalized?")
    ranked = rerank(
        [candidate("v1", version="v1"), candidate("v2", version="v2")],
        analysis,
        rerank_config(),
    )
    assert analysis.version_intent == "none"
    assert all(item.rerank_features["version_compatibility"] == 0.0 for item in ranked)


def test_latest_version_preference_is_conditional():
    analysis = QueryAnalyzer().analyze("find the latest version of example")
    ranked = rerank(
        [candidate("v1", version="v1", fusion_score=0.015), candidate("v2", version="v2", fusion_score=0.0145)],
        analysis,
        rerank_config(),
    )
    assert analysis.version_intent == "latest"
    assert ranked[0].snippet.id == "v2"
    assert ranked[0].rerank_features["version_compatibility"] == 1.0


def test_specific_version_preference_is_conditional():
    analysis = QueryAnalyzer().analyze("show version 2 of example")
    ranked = rerank(
        [candidate("v1", version="v1", fusion_score=0.015), candidate("v2", version="v2", fusion_score=0.0145)],
        analysis,
        rerank_config(),
    )
    assert analysis.version_intent == "specific"
    assert ranked[0].snippet.id == "v2"
    assert ranked[0].rerank_features["version_compatibility"] == 1.0


def test_exact_technical_phrase_is_recognized():
    analysis = QueryAnalyzer().analyze("parameterized database query")
    item = candidate("query", docstring="Execute a parameterized database query.")
    rerank([item], analysis, rerank_config())
    assert item.rerank_features["exact_phrase_match"] == 1.0


def test_generic_word_is_not_a_strong_identifier():
    analysis = QueryAnalyzer().analyze("function")
    item = candidate("function", function_name="function")
    rerank([item], analysis, rerank_config())
    assert analysis.identifiers == []
    assert item.rerank_features["distinctive_identifier_match"] == 0.0
    assert item.rerank_features["rerank_adjustment"] == 0.0


def test_high_confidence_intent_can_contribute():
    analysis = QueryAnalyzer().analyze("JWT expiration validation")
    item = candidate("auth", function_name="validate_token", docstring="Validate a JWT access token expiration.")
    rerank([item], analysis, rerank_config())
    assert analysis.confidence >= 0.8
    assert item.rerank_features["intent_compatibility"] == 1.0


def test_low_confidence_intent_contributes_zero():
    analysis = QueryAnalyzer().analyze("normalization")
    item = candidate("generic", function_name="normalize")
    rerank([item], analysis, rerank_config())
    assert analysis.confidence < 0.5
    assert item.rerank_features["intent_compatibility"] == 0.0


def test_adjustment_is_bounded_and_membership_is_preserved():
    analysis = QueryAnalyzer().analyze("parameterized database query version 2")
    items = [
        candidate("a", version="v1", docstring="Execute a parameterized database query.", fusion_score=0.015),
        candidate("b", version="v2", docstring="Execute a parameterized database query.", fusion_score=0.014),
    ]
    ranked = rerank(items, analysis, rerank_config())
    assert {item.snippet.id for item in ranked} == {"a", "b"}
    assert len(ranked) == len(items)
    assert all(-MAX_ADJUSTMENT <= item.rerank_features["rerank_adjustment"] <= MAX_ADJUSTMENT for item in ranked)


def test_repeated_reranking_is_deterministic():
    analysis = QueryAnalyzer().analyze("JWT expiration validation")
    first = rerank([candidate("a"), candidate("b", function_name="validate_token", docstring="Validate JWT expiration.")], analysis, rerank_config())
    second = rerank([candidate("a"), candidate("b", function_name="validate_token", docstring="Validate JWT expiration.")], analysis, rerank_config())
    assert [item.snippet.id for item in first] == [item.snippet.id for item in second]
    assert [item.final_score for item in first] == [item.final_score for item in second]
