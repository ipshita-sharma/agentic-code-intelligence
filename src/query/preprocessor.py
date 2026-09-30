"""Query normalization that preserves identifiers, phrases, and aliases.

The preprocessor has two consumers with different needs:

* MiniLM receives the original query so its semantic representation is not
  replaced by generated terms.
* BM25 receives a conservative lexical expansion that retains every original
  term and adds only context-supported technical aliases.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from ..config import RetrievalConfig


IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*(?:[._:/-][A-Za-z0-9_$]+)*")

# These are phrase anchors, not a list of every possible bigram.  Keeping the
# list small avoids turning ordinary prose into a large, noisy query.
TECHNICAL_PHRASES = (
    "input normalization", "normalize input", "normalized input", "clean input",
    "preprocess input", "sanitize input", "transform input", "prepare input",
    "trim input", "standardize data", "jwt expiration validation", "jwt expiration",
    "parameterized database query", "database connection", "connection pool",
    "browser form values", "api request payload", "request payload", "refresh token",
    "feature flag", "duplicate rows", "duplicate records", "duplicate tags",
    "raw json", "main function", "form fields", "unit tests", "error response",
    "error body", "payload keys cleaned", "exponential backoff",
)


def split_identifier(token: str) -> List[str]:
    """Split snake_case, kebab-case, dotted names, and camelCase safely."""
    parts = re.split(r"[_\-./:$]+|(?<=[a-z0-9])(?=[A-Z])", token)
    return [part.lower() for part in parts if part]


def tokenize(text: str, preserve_identifiers: bool = True) -> List[str]:
    raw = IDENTIFIER_RE.findall(text)
    tokens: List[str] = []
    for item in raw:
        lowered = item.lower()
        tokens.append(lowered)
        if preserve_identifiers:
            tokens.extend(split_identifier(item))
    return tokens


def extract_identifiers(query: str) -> List[str]:
    """Return code-like identifiers without discarding their original spelling."""
    identifiers: List[str] = []
    for item in IDENTIFIER_RE.findall(query):
        is_snake_or_delimited = bool(re.search(r"[_$./:$-]", item))
        is_camel_or_pascal = bool(re.search(r"[a-z][A-Z]", item)) or (
            len(item) > 1 and item[0].isupper()
        )
        is_acronym = item.isupper() and len(item) > 1
        if is_snake_or_delimited or is_camel_or_pascal or is_acronym:
            if item not in identifiers:
                identifiers.append(item)
    return identifiers


def extract_technical_phrases(query: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    found: List[str] = []
    for phrase in sorted(TECHNICAL_PHRASES, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9_]){re.escape(phrase)}(?![a-z0-9_])", normalized):
            found.append(phrase)
    return found


def _has_any(tokens: set[str], values: Iterable[str]) -> bool:
    return bool(tokens.intersection(values))


def alias_expansions(
    query: str,
    tokens: List[str],
    phrases: List[str],
    config: RetrievalConfig,
) -> List[str]:
    """Return context-supported aliases while retaining original query terms."""
    token_set = set(tokens)
    phrase_set = set(phrases)
    aliases = config.technical_aliases
    selected: set[str] = set()

    # A preprocessing word alone is deliberately insufficient.  It needs a
    # data/input context or an explicit technical phrase.
    preprocessing_terms = set(aliases.get("preprocessing", ()))
    preprocessing_context = {
        "input", "data", "field", "fields", "form", "forms", "payload",
        "value", "values", "record", "records", "row", "rows", "text",
        "string", "request", "keys", "tags",
    }
    has_preprocessing_phrase = any(
        phrase in phrase_set
        for phrase in (
            "input normalization", "normalize input", "normalized input",
            "clean input", "preprocess input", "sanitize input", "transform input",
            "prepare input", "trim input", "standardize data", "form fields", "duplicate tags",
        )
    )
    testing_context = _has_any(token_set, {"test", "tests", "testing", "assert", "assertion", "unit", "fixture", "mock"})
    if not testing_context and (has_preprocessing_phrase or (
        _has_any(token_set, preprocessing_terms)
        and _has_any(token_set, preprocessing_context)
    )):
        # Expand only the synonym family supported by the observed query.
        # Context is required for ambiguous terms such as normalization.
        selected.update({"preprocess", "preprocessing", "normalize", "normalized"})
        if "payload keys cleaned" in phrase_set:
            selected = {"clean", "preprocess"}
        elif "input normalization" not in phrase_set and "normalize input" not in phrase_set:
            selected = {"normalize", "normalized"}
        if _has_any(token_set, {"clean", "cleaned"}) or "payload keys cleaned" in phrase_set:
            selected.update({"clean", "cleaned"})
        if _has_any(token_set, {"sanitize", "sanitized"}):
            selected.add("sanitize")
        if _has_any(token_set, {"prepare", "prepared"}):
            selected.add("prepare")
        if _has_any(token_set, {"transform", "transformed"}):
            selected.add("transform")
        if _has_any(token_set, {"trim", "trimmed"}):
            selected.update({"trim", "trimmed"})
        if _has_any(token_set, {"standardize", "standardized"}) or "input normalization" in phrase_set:
            selected.add("standardize")

    auth_terms = set(aliases.get("authentication", ()))
    if _has_any(token_set, {"jwt", "bearer", "authentication", "authorization", "authenticate", "login", "credential", "session"}) or (
        "token" in token_set and _has_any(token_set, {"validate", "validation", "expire", "expiration", "refresh", "access"})
    ):
        selected.update({"auth", "authenticate", "authentication"})
        if _has_any(token_set, {"jwt", "token", "bearer"}):
            selected.update({"token", "jwt"})
        if _has_any(token_set, {"validate", "validation"}):
            selected.update({"validate", "validation"})
        if _has_any(token_set, {"expire", "expiration", "expired"}):
            selected.update({"expire", "expiration"})
        if "session" in token_set:
            selected.add("session")
        if "bearer" in token_set:
            selected.add("bearer")

    database_terms = set(aliases.get("database", ()))
    explicit_database = _has_any(token_set, {"database", "db", "sql", "jdbc", "orm", "datasource", "transaction"})
    database_context = _has_any(token_set, {"connection", "pool"})
    insert_context = _has_any(token_set, {"insert", "select"})
    algorithm_context = _has_any(token_set, {"duplicate", "duplicates", "deduplicate", "deduplication", "unique", "distinct"})
    if "parameterized database query" in phrase_set:
        selected.update({"database", "db", "sql", "query"})
    elif database_context and not explicit_database:
        selected.update({"database", "db", "connection"})
    elif insert_context and not explicit_database and not algorithm_context:
        selected.update({"database", "db"})

    error_terms = set(aliases.get("error_handling", ()))
    explicit_error_terms = {"error", "errors", "exception", "exceptions", "failure", "failures", "failed"}
    if _has_any(token_set, {"fallback", "recovery"}):
        selected.update({"error", "exception", "failure"})
    elif _has_any(token_set, explicit_error_terms):
        # Do not broaden an already explicit error query; doing so can drown a
        # more specific retry/request signal in BM25.
        selected.update(token_set.intersection(explicit_error_terms))
    elif "retry" in token_set and _has_any(token_set, {"network", "request", "connection", "database"}):
        selected.add("retry")

    api_terms = set(aliases.get("api", ()))
    api_evidence = token_set.intersection({"api", "endpoint", "http", "rest", "graphql", "route", "payload", "response", "request"})
    non_api_domain = _has_any(token_set, {
        "jwt", "token", "bearer", "authentication", "authorization", "session",
        "test", "tests", "testing", "assert", "assertion", "unit", "fixture", "mock",
        "preprocess", "preprocessing", "normalize", "normalization", "normalized",
        "clean", "cleaned", "sanitize", "transform", "trim", "standardize",
        "database", "db", "sql", "transaction", "connection", "query",
    })
    if len(api_evidence) == 1 and "api" not in token_set and not non_api_domain and not _has_any(token_set, explicit_error_terms) and "api request payload" not in phrase_set:
        selected.update({"api", "request", "response", "payload"})
    elif "api request payload" in phrase_set:
        selected.update({"api", "request", "payload"})

    return sorted(selected)


def preprocess_query(query: str, config: RetrievalConfig | None = None) -> dict:
    config = config or RetrievalConfig()
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    all_tokens = tokenize(query, preserve_identifiers=True)
    phrases = extract_technical_phrases(query)
    lexical_tokens = [token for token in all_tokens if token not in config.stopwords and len(token) > 1]
    # Keep programming terms even if a future stopword list is expanded.
    programming_terms = {
        "api", "sql", "http", "json", "jwt", "token", "function", "class",
        "method", "db", "database", "orm", "jdbc", "request", "response",
    }
    lexical_tokens = list(dict.fromkeys(lexical_tokens + [t for t in all_tokens if t in programming_terms]))
    expansions = alias_expansions(query, all_tokens, phrases, config)
    lexical_tokens = list(dict.fromkeys(lexical_tokens + expansions))
    return {
        "normalized_query": normalized,
        "lexical_query": " ".join(lexical_tokens),
        "tokens": all_tokens,
        "keywords": lexical_tokens,
        "identifiers": extract_identifiers(query),
        "technical_phrases": phrases,
        "alias_expansions": expansions,
    }
