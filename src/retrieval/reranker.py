"""Conservative, explainable reranking over an existing candidate set.

The reranker is deliberately a tie-breaking stage. It never creates or
removes candidates and it does not reuse the old weighted sum. The hybrid
score remains the base score; only small, bounded compatibility evidence is
added when the query provides strong evidence for it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, List

from ..config import RetrievalConfig
from ..models import QueryAnalysis, RetrievalCandidate
from ..query.preprocessor import split_identifier, tokenize
from ..versioning.version_filter import version_order_key


# Fusion scores in this project are RRF scores around 0.01--0.02. Keep the
# compatibility signal in the same order of magnitude so even alpha=0.10
# cannot overwhelm the retrieval score.
MAX_ADJUSTMENT = 0.01
DEFAULT_ALPHA = 0.05

# These words are useful for query understanding but are not distinctive code
# identifiers. They must not receive identifier evidence by themselves.
GENERIC_IDENTIFIER_TERMS = {
    "api", "code", "data", "function", "handler", "manager", "method",
    "normalization", "normalize", "payload", "request", "response", "service",
    "value", "values", "class", "module", "helper", "implementation",
    "where", "which", "how", "what", "when", "why",
}

INTENT_SUPPORT_TERMS = {
    "data_preprocessing": {"preprocess", "preprocessing", "normalize", "normalized", "clean", "sanitize", "transform", "trim", "standardize", "input", "form", "payload", "field", "fields", "tags"},
    "authentication": {"auth", "authenticate", "authentication", "authorization", "authorize", "jwt", "token", "bearer", "login", "permission", "credential", "session", "refresh", "expiration", "expired"},
    "database": {"database", "db", "sql", "query", "insert", "select", "transaction", "connection", "jdbc", "orm", "postgres", "mysql", "datasource", "repository", "rows"},
    "api": {"api", "endpoint", "request", "response", "payload", "http", "route", "rest", "graphql", "serialize", "controller"},
    "error_handling": {"error", "exception", "failure", "failed", "fallback", "recovery", "timeout", "retry", "retries"},
    "configuration": {"config", "configuration", "setting", "settings", "environment", "env", "yaml", "json", "constant", "constants", "datasource"},
    "algorithm": {"algorithm", "sort", "sorted", "search", "hash", "graph", "rank", "ranking", "score", "unique", "map", "deduplicate", "duplicate", "distinct"},
    "ui_frontend": {"ui", "frontend", "component", "react", "browser", "render", "button", "form", "view", "dom", "web"},
    "backend": {"backend", "server", "service", "worker", "controller", "repository", "dispatch", "entry"},
    "testing": {"test", "tests", "testing", "mock", "fixture", "assert", "assertion", "coverage", "expects", "verify", "unit"},
    "performance": {"performance", "fast", "latency", "optimize", "benchmark", "cache", "memory", "backoff", "efficient", "recompute"},
}


def _overlap(left: Iterable[str], right: Iterable[str]) -> float:
    """Retain the small public helper used by older callers/tests."""
    a, b = set(left), set(right)
    return len(a & b) / max(1, len(a))


def _candidate_text(candidate: RetrievalCandidate) -> str:
    snippet = candidate.snippet
    return " ".join(
        value
        for value in (
            snippet.file_path,
            snippet.function_name,
            snippet.class_name,
            getattr(snippet, "symbol_name", ""),
            getattr(snippet, "parent_symbol", ""),
            snippet.docstring,
            " ".join(snippet.imports),
            snippet.code,
        )
        if value
    )


def _phrase_text(text: str) -> str:
    """Make identifier separators readable while keeping word boundaries."""
    normalized = text.lower()
    normalized = re.sub(r"[_./:$-]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = _phrase_text(text)
    normalized_phrase = _phrase_text(phrase)
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])", normalized_text))


def _phrase_match(analysis: QueryAnalysis, candidate: RetrievalCandidate) -> float:
    phrases = list(dict.fromkeys(analysis.technical_phrases))
    if not phrases:
        return 0.0
    text = _candidate_text(candidate)
    return 1.0 if any(_contains_phrase(text, phrase) for phrase in phrases) else 0.0


def _distinctive_identifier(identifier: str) -> bool:
    parts = [part for part in split_identifier(identifier) if part]
    lowered = identifier.lower()
    return (
        len(lowered) >= 3
        and lowered not in GENERIC_IDENTIFIER_TERMS
        and any(part not in GENERIC_IDENTIFIER_TERMS and len(part) >= 3 for part in parts)
    )


def _identifier_match(analysis: QueryAnalysis, candidate: RetrievalCandidate) -> float:
    identifiers = [item for item in analysis.identifiers if _distinctive_identifier(item)]
    if not identifiers:
        return 0.0
    text = _phrase_text(_candidate_text(candidate))
    text_tokens = set(tokenize(text, preserve_identifiers=True))
    matches = 0.0
    for identifier in identifiers:
        lowered = identifier.lower()
        full_match = lowered in text_tokens or _contains_phrase(text, lowered)
        if full_match:
            matches += 1.0
            continue
        parts = [part for part in split_identifier(identifier) if part not in GENERIC_IDENTIFIER_TERMS]
        if parts and sum(part in text_tokens for part in parts) == len(parts):
            matches += 0.5
    return min(1.0, matches / max(1, len(identifiers)))


def _intent_match(analysis: QueryAnalysis, candidate: RetrievalCandidate) -> float:
    """Return evidence only for high-confidence, explicitly supported intent."""
    if analysis.confidence < 0.80 or analysis.primary_intent == "general_code_search":
        return 0.0
    support_terms = INTENT_SUPPORT_TERMS.get(analysis.primary_intent)
    if not support_terms:
        return 0.0
    candidate_tokens = set(tokenize(_candidate_text(candidate), preserve_identifiers=True))
    # A symbol type alone is meaningful only for function-behavior queries;
    # for domain intents require explicit words in the candidate text.
    if analysis.primary_intent == "function_behavior":
        return 1.0 if candidate.snippet.snippet_type in {"function", "method"} and candidate.snippet.function_name else 0.0
    return 1.0 if candidate_tokens.intersection(support_terms) else 0.0


def _version_groups(candidates: list[RetrievalCandidate]) -> dict[tuple[str, str, str], list[RetrievalCandidate]]:
    groups: dict[tuple[str, str, str], list[RetrievalCandidate]] = defaultdict(list)
    for candidate in candidates:
        snippet = candidate.snippet
        key = (snippet.repository, snippet.file_path, snippet.function_name or snippet.class_name)
        groups[key].append(candidate)
    return groups


def _version_match(
    analysis: QueryAnalysis,
    candidate: RetrievalCandidate,
    candidates: list[RetrievalCandidate],
) -> float:
    # Hard safety rule: ordinary queries receive no version contribution.
    if analysis.version_intent == "none":
        return 0.0
    version = candidate.snippet.version
    if analysis.version_intent == "specific":
        return 1.0 if analysis.version and version == analysis.version else -1.0
    if analysis.version_intent != "latest":
        return 0.0
    if version.lower() == "latest":
        return 1.0
    group = _version_groups(candidates)[
        (candidate.snippet.repository, candidate.snippet.file_path, candidate.snippet.function_name or candidate.snippet.class_name)
    ]
    numeric = [(item, version_order_key(item.snippet.version)) for item in group]
    numeric = [(item, key) for item, key in numeric if key is not None]
    if not numeric:
        return 0.0
    maximum = max(key for _, key in numeric)
    current = version_order_key(version)
    if current is None:
        return -1.0
    return 1.0 if current == maximum else -1.0


def _alpha(config: RetrievalConfig) -> float:
    value = getattr(config, "rerank_alpha", DEFAULT_ALPHA)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return DEFAULT_ALPHA


def _feature_enabled(config: RetrievalConfig, feature: str) -> bool:
    """Allow controlled leave-one-feature-out runs without a new config field."""
    enabled = getattr(config, "rerank_enabled_features", None)
    return enabled is None or feature in enabled


def rerank(
    candidates: List[RetrievalCandidate],
    analysis: QueryAnalysis,
    config: RetrievalConfig | None = None,
) -> List[RetrievalCandidate]:
    """Rerank in place without changing candidate membership or count."""
    config = config or RetrievalConfig()
    alpha = _alpha(config)
    original_count = len(candidates)
    for candidate in candidates:
        phrase_match = _phrase_match(analysis, candidate) if _feature_enabled(config, "phrase") else 0.0
        identifier_match = _identifier_match(analysis, candidate) if _feature_enabled(config, "identifier") else 0.0
        intent_match = _intent_match(analysis, candidate) if _feature_enabled(config, "intent") else 0.0
        version_match = _version_match(analysis, candidate, candidates) if _feature_enabled(config, "version") else 0.0

        # Phrase and identifier evidence carry the useful new information.
        # Semantic and lexical scores are diagnostics only: both already
        # influence fusion and are intentionally not double-counted here.
        compatibility = (
            0.50 * phrase_match
            + 0.30 * identifier_match
            + 0.15 * intent_match
            + 0.05 * version_match
        )
        adjustment = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, MAX_ADJUSTMENT * compatibility))
        final_score = candidate.fusion_score + alpha * adjustment
        candidate.rerank_features = {
            "base_hybrid_score": candidate.fusion_score,
            "semantic_similarity": candidate.semantic_score,
            "lexical_score": candidate.lexical_score,
            "exact_phrase_match": phrase_match,
            "distinctive_identifier_match": identifier_match,
            "intent_compatibility": intent_match,
            "version_compatibility": version_match,
            "rerank_adjustment": adjustment,
            "alpha": alpha,
            "final_score": final_score,
        }
        candidate.final_score = final_score

    ranked = sorted(candidates, key=lambda item: (-item.final_score, -item.fusion_score, item.snippet.id))
    if len(ranked) != original_count:
        raise AssertionError("Reranker changed candidate count")
    return ranked
