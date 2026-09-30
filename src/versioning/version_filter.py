"""Cheap, deterministic version filters applied after fusion."""

from __future__ import annotations

import re
from typing import Iterable, Mapping

from ..models import RetrievalCandidate


_NUMERIC_VERSION_RE = re.compile(r"^v?(\d+(?:\.\d+)*)$", re.IGNORECASE)


def _matches(value: str, requested: str | Iterable[str] | None) -> bool:
    if requested is None:
        return True
    if isinstance(requested, str):
        return value == requested
    return value in set(requested)


def version_order_key(value: str) -> tuple[int, tuple[int, ...]] | None:
    """Return a numeric ordering key for v1, 1.2, v2.0.1, etc.

    Non-numeric labels return ``None``.  Callers then retain the historical
    lexical fallback instead of inventing an ordering for labels such as
    ``stable`` or ``legacy``.
    """
    match = _NUMERIC_VERSION_RE.fullmatch(value.strip())
    if not match:
        return None
    return 0, tuple(int(part) for part in match.group(1).split("."))


def _newer_candidate(left: RetrievalCandidate, right: RetrievalCandidate) -> RetrievalCandidate:
    left_key = version_order_key(left.snippet.version)
    right_key = version_order_key(right.snippet.version)
    if left_key is not None and right_key is not None:
        return left if left_key > right_key else right
    # Preserve the previous deterministic behavior when labels cannot be
    # ordered semantically.
    return left if left.snippet.version > right.snippet.version else right


def merge_query_version(
    filters: Mapping[str, object] | None,
    *,
    explicit_version: str = "all",
    version_intent: str = "none",
    requested_version: str | None = None,
) -> dict[str, object]:
    """Apply query-conditioned version intent without boosting ordinary queries."""
    merged = dict(filters or {})
    if "version" in merged:
        return merged
    if explicit_version != "all":
        merged["version"] = explicit_version
    elif version_intent == "specific" and requested_version:
        merged["version"] = requested_version
    elif version_intent == "latest":
        merged["version"] = "latest"
    else:
        merged["version"] = "all"
    return merged


def filter_candidates(candidates: Iterable[RetrievalCandidate], filters: Mapping[str, object] | None = None) -> list[RetrievalCandidate]:
    filters = filters or {}
    version = filters.get("version")
    version_filter = None if version in (None, "all", "latest") else version
    selected = []
    for candidate in candidates:
        snippet = candidate.snippet
        if not _matches(snippet.language, filters.get("language")):
            continue
        if not _matches(snippet.repository, filters.get("repository")):
            continue
        if not _matches(snippet.version, version_filter):
            continue
        if not _matches(snippet.snippet_type, filters.get("snippet_type")):
            continue
        file_path = filters.get("file_path")
        if file_path and str(file_path).lower() not in snippet.file_path.lower():
            continue
        selected.append(candidate)
    if version == "latest":
        latest_by_function: dict[tuple[str, str, str], RetrievalCandidate] = {}
        for candidate in selected:
            snippet = candidate.snippet
            key = (snippet.repository, snippet.file_path, snippet.function_name or snippet.class_name)
            prior = latest_by_function.get(key)
            if prior is None:
                latest_by_function[key] = candidate
            else:
                latest_by_function[key] = _newer_candidate(prior, candidate)
        selected = list(latest_by_function.values())
    return selected
