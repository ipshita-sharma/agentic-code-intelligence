"""Lightweight, confidence-aware query understanding.

Intent is a soft query signal.  It is never used to remove candidates from
retrieval; the pipeline still searches the full lexical and semantic pools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable

from .preprocessor import preprocess_query
from ..config import RetrievalConfig
from ..models import QueryAnalysis


LANGUAGE_TERMS = {
    "python": "python", "java": "java", "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript", "go": "go", "rust": "rust", "sql": "sql",
}
TECHNICAL_RE = re.compile(
    r"\b(?:[A-Za-z_$][A-Za-z0-9_$]*(?:[_./:$-][A-Za-z0-9_$]+)+|"
    r"[A-Z][A-Za-z0-9]+|[A-Za-z]+\d+[A-Za-z0-9]*)\b"
)
VERSION_RE = re.compile(r"\bv\s*(\d+(?:\.\d+)*)\b|\bversion\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)


@dataclass(frozen=True)
class IntentDefinition:
    positive_tokens: tuple[str, ...] = ()
    negative_tokens: tuple[str, ...] = ()
    technical_phrases: tuple[str, ...] = ()
    identifier_patterns: tuple[str, ...] = ()


# The definitions are intentionally explicit.  Ambiguous words such as
# normalization, retry, request, parse, and function are not enough by
# themselves to produce high confidence.
INTENT_DEFINITIONS: Dict[str, IntentDefinition] = {
    "data_preprocessing": IntentDefinition(
        positive_tokens=("preprocess", "preprocessing", "preprocessed", "normalize", "normalization", "normalized", "clean", "cleaned", "sanitize", "prepare", "transform", "trim", "trimmed", "standardize", "coerce"),
        negative_tokens=("assertion", "test", "testing", "sql", "transaction", "endpoint"),
        technical_phrases=("input normalization", "normalize input", "normalized input", "clean input", "preprocess input", "sanitize input", "transform input", "prepare input", "trim input", "standardize data", "form fields", "duplicate tags", "payload keys cleaned"),
    ),
    "function_behavior": IntentDefinition(
        positive_tokens=("function", "method", "helper", "routine", "behavior", "calculate", "return", "returns", "does", "handle", "parser"),
        negative_tokens=("test", "testing", "assertion", "configuration", "database", "sql"),
        technical_phrases=("function behavior", "raw json", "entry point"),
        identifier_patterns=(r"\b[a-zA-Z_$][a-zA-Z0-9_$]*\([^)]*\)",),
    ),
    "authentication": IntentDefinition(
        positive_tokens=("auth", "authenticate", "authenticated", "authentication", "authorization", "authorize", "jwt", "token", "bearer", "login", "permission", "credential", "session", "refresh", "expiration", "expired"),
        negative_tokens=("database", "sql", "yaml", "testing"),
        technical_phrases=("jwt expiration validation", "jwt expiration", "refresh token", "access token", "bearer token", "session token"),
        identifier_patterns=(r"\b(?:JWT|OAuth|SAML|Auth[A-Za-z0-9_$]*)\b",),
    ),
    "database": IntentDefinition(
        positive_tokens=("database", "db", "sql", "query", "queries", "insert", "select", "transaction", "connection", "connections", "pool", "jdbc", "orm", "postgres", "mysql", "datasource", "rows", "fetched", "retry", "retries"),
        negative_tokens=("frontend", "browser", "component", "jwt"),
        technical_phrases=("parameterized database query", "database connection", "connection pool", "bulk insert", "rows fetched", "database retry"),
        identifier_patterns=(r"\b(?:DB|DB_URL|JDBC|ORM|Postgres|MySQL)[A-Za-z0-9_$]*\b",),
    ),
    "api": IntentDefinition(
        positive_tokens=("api", "endpoint", "request", "response", "payload", "http", "route", "rest", "graphql", "serialize", "serialized", "controller"),
        negative_tokens=("database", "sql", "yaml", "unit", "error", "exception", "failure"),
        technical_phrases=("api request payload", "request payload", "request fields", "json headers", "api call", "http endpoint", "error response", "error body"),
        identifier_patterns=(r"\b(?:API|HTTP|REST|GraphQL|[A-Z]+_URL)\b",),
    ),
    "error_handling": IntentDefinition(
        positive_tokens=("error", "errors", "exception", "exceptions", "failure", "failed", "fallback", "recovery", "timeout", "handling", "log", "logging", "retry", "retries"),
        negative_tokens=("success", "configuration", "component"),
        technical_phrases=("error response", "error body", "api error body", "bad request", "bad-request", "bad request payload", "bad-request payload", "unknown exception", "request exceptions", "exponential backoff", "retry after failure"),
    ),
    "configuration": IntentDefinition(
        positive_tokens=("config", "configuration", "setting", "settings", "environment", "env", "yaml", "json", "constant", "constants", "datasource"),
        negative_tokens=("assertion", "unit", "component"),
        technical_phrases=("feature flag", "environment configuration", "default timeout", "retry constants"),
        identifier_patterns=(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", r"\b(?:DB_URL|API_BASE_URL|DEBUG)\b"),
    ),
    "algorithm": IntentDefinition(
        positive_tokens=("algorithm", "sort", "sorted", "search", "hash", "graph", "rank", "ranking", "score", "unique", "map", "deduplicate", "deduplication", "duplicate", "distinct", "descending"),
        negative_tokens=("jwt", "authentication", "yaml"),
        technical_phrases=("duplicate rows", "duplicate records", "unique by identifier", "descending order", "exponential backoff"),
    ),
    "ui_frontend": IntentDefinition(
        positive_tokens=("ui", "frontend", "component", "react", "browser", "render", "renders", "button", "form", "view", "dom", "javascript"),
        negative_tokens=("database", "sql", "server-side"),
        technical_phrases=("browser form values", "frontend environment configuration", "list item", "form fields"),
    ),
    "backend": IntentDefinition(
        positive_tokens=("backend", "server", "server-side", "service", "worker", "controller", "repository", "dispatch", "entry"),
        negative_tokens=("browser", "frontend", "component"),
        technical_phrases=("backend service", "server-side", "entry point", "dispatch payload"),
    ),
    "testing": IntentDefinition(
        positive_tokens=("test", "tests", "testing", "mock", "fixture", "assert", "assertion", "coverage", "expects", "verified", "verify", "unit"),
        negative_tokens=("production", "runtime"),
        technical_phrases=("unit tests", "verified with an assertion", "test expects", "assertion", "test preprocessing"),
    ),
    "performance": IntentDefinition(
        positive_tokens=("performance", "fast", "latency", "optimize", "benchmark", "cache", "memory", "backoff", "efficient", "recompute"),
        negative_tokens=("assertion", "frontend"),
        technical_phrases=("exponential backoff", "before recomputing", "connection pool"),
    ),
    "general_code_search": IntentDefinition(),
}

# Single tokens in this set are useful retrieval terms but weak intent
# evidence.  They only become decisive with a phrase, context, or identifier.
AMBIGUOUS_TERMS = {
    "normalization", "normalize", "normalized", "duplicate", "duplicates", "payload",
    "retry", "request", "function", "parse", "parser", "validation", "processing",
}


def _phrase_present(normalized_query: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9_]){re.escape(phrase)}(?![a-z0-9_])", normalized_query))


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def detect_version_intent(query: str) -> tuple[str, str | None]:
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    if re.search(r"\b(latest|newest|most recent)\b", normalized) or re.search(r"\bcurrent(?:\s+implementation|\s+version)?\b", normalized) or re.search(r"\blatest version\b", normalized):
        return "latest", None
    match = VERSION_RE.search(query)
    if match:
        return "specific", f"v{match.group(1) or match.group(2)}"
    return "none", None


def _score_definition(
    definition: IntentDefinition,
    tokens: set[str],
    normalized_query: str,
    identifiers: list[str],
) -> float:
    score = 0.0
    for token in definition.positive_tokens:
        if token in tokens:
            score += 1.0
    for phrase in definition.technical_phrases:
        if _phrase_present(normalized_query, phrase):
            score += 3.0
    for pattern in definition.identifier_patterns:
        if re.search(pattern, " ".join(identifiers) + " " + normalized_query, re.IGNORECASE):
            score += 2.0
    for token in definition.negative_tokens:
        if token in tokens:
            score -= 1.25
    return max(0.0, score)


def _intent_scores(processed: dict, query: str) -> tuple[str, float, dict[str, float]]:
    normalized = processed["normalized_query"]
    tokens = set(processed["tokens"])
    identifiers = processed["identifiers"]
    raw = {
        name: _score_definition(definition, tokens, normalized, identifiers)
        for name, definition in INTENT_DEFINITIONS.items()
        if name != "general_code_search"
    }

    # A lone ambiguous word must not force an intent.  Keep its score visible
    # for diagnostics, but damp it before confidence is calculated.
    meaningful_tokens = tokens - AMBIGUOUS_TERMS
    has_known_phrase = bool(processed.get("technical_phrases"))
    has_context = len(meaningful_tokens) >= 2 or bool(identifiers)
    if not has_known_phrase and not has_context:
        for name in raw:
            raw[name] *= 0.25

    best_name, best_raw = max(raw.items(), key=lambda item: (item[1], item[0]))
    other_scores = sorted((value for name, value in raw.items() if name != best_name), reverse=True)
    second_raw = other_scores[0] if other_scores else 0.0
    margin = best_raw - second_raw
    if best_raw < 1.5 or margin < 0.75:
        primary = "general_code_search"
        confidence = min(0.49, best_raw / max(1.0, best_raw + second_raw + 1.0))
    else:
        primary = best_name
        confidence = min(0.99, 0.50 + 0.08 * min(best_raw, 5.0) + 0.08 * min(margin, 4.0))
    maximum = max(raw.values(), default=0.0)
    scores = {
        name: round((value / maximum) if maximum else 0.0, 6)
        for name, value in raw.items()
    }
    scores["general_code_search"] = round(1.0 if primary == "general_code_search" else 0.0, 6)
    return primary, round(confidence, 6), scores


class QueryAnalyzer:
    """Classify queries with weighted phrase/token evidence, without an LLM."""

    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()

    def analyze(self, query: str) -> QueryAnalysis:
        processed = preprocess_query(query, self.config)
        normalized = processed["normalized_query"]
        primary, confidence, scores = _intent_scores(processed, query)
        technical_terms = _unique(item.lower() for item in TECHNICAL_RE.findall(query))
        technical_terms.extend(item for item in processed.get("alias_expansions", []) if item not in technical_terms)
        language_hints = _unique(
            LANGUAGE_TERMS[token]
            for token in processed["tokens"]
            if token in LANGUAGE_TERMS
        )
        version_intent, version = detect_version_intent(query)
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            intent=primary,
            keywords=processed["keywords"],
            technical_terms=technical_terms,
            language_hints=language_hints,
            identifiers=processed["identifiers"],
            technical_phrases=processed.get("technical_phrases", []),
            confidence=confidence,
            intent_scores=scores,
            primary_intent=primary,
            version_intent=version_intent,
            version=version,
        )
