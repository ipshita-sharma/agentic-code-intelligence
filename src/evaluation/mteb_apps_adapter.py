"""Adapter for the CoIR/MTEB AppsRetrieval data and prediction format.

The adapter deliberately sits outside the retrieval implementation.  It turns
the BEIR/MTEB-style corpus, queries, and qrels files into the project's
``CodeSnippet`` contract, runs an already-configured ``RetrievalEngine``, and
serializes the native MTEB retrieval prediction structure::

    {"test": {"query-id": {"corpus-id": score}}}

The inner dictionaries retain ranking order.  The adapter never changes
candidate counts, score weights, filtering, query understanding, or reranking.
"""

from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..models import CodeSnippet
from .metrics import evaluate_run


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".parquet", ".tsv", ".txt"}
_MISSING = object()


@dataclass(frozen=True)
class CorpusDocument:
    """A corpus item with the official ID kept as a string key."""

    corpus_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class AppsRetrievalData:
    """Parsed AppsRetrieval data in stable input order."""

    split: str
    corpus: "OrderedDict[str, CorpusDocument]"
    queries: "OrderedDict[str, str]"
    qrels: "OrderedDict[str, OrderedDict[str, float]] | None" = None

    @property
    def corpus_ids(self) -> set[str]:
        return set(self.corpus)

    @property
    def query_ids(self) -> set[str]:
        return set(self.queries)

    def to_snippets(self) -> list[CodeSnippet]:
        """Map official corpus rows to the existing retrieval data contract."""
        return [
            CodeSnippet(
                id=corpus_id,
                repository="mteb-appsretrieval",
                # Do not put the official ID in a searchable metadata field:
                # IDs are evaluation keys, not document content.
                file_path="",
                language="unknown",
                version="official",
                commit_id="official",
                code=document.text,
                docstring=document.title,
                snippet_type="generic",
            )
            for corpus_id, document in self.corpus.items()
        ]


def _as_mapping(row: Any, source: Path, row_number: int) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"{source}:{row_number} must contain an object, got {type(row).__name__}")
    return row


def _json_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        for row_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{row_number}: {exc}") from exc
            rows.append(_as_mapping(value, path, row_number))
        return rows

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    if isinstance(value, list):
        return [_as_mapping(row, path, number) for number, row in enumerate(value, 1)]
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a list or object")

    # Also accept the common {"id": {"text": ...}} or {"id": "text"}
    # fixture shape, while preserving the explicit ID if one is present.
    rows = []
    for key, item in value.items():
        if isinstance(item, Mapping):
            row = dict(item)
            row.setdefault("_id", key)
        else:
            row = {"_id": key, "text": item}
        rows.append(_as_mapping(row, path, len(rows) + 1))
    return rows


def _tsv_rows(path: Path) -> list[Mapping[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _parquet_rows(path: Path) -> list[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore

        return [dict(row) for row in parquet.read_table(path).to_pylist()]
    except ImportError:
        try:
            import pandas as pd  # type: ignore

            return pd.read_parquet(path).to_dict(orient="records")
        except ImportError as exc:
            raise RuntimeError(
                "Reading AppsRetrieval parquet requires pyarrow or pandas; "
                "install an optional parquet reader or convert the files to JSONL/TSV"
            ) from exc
        except Exception as exc:  # pragma: no cover - depends on local parquet engines
            raise RuntimeError(f"Could not read parquet file {path}: {exc}") from exc
    except Exception as exc:  # pragma: no cover - depends on local parquet engines
        raise RuntimeError(f"Could not read parquet file {path}: {exc}") from exc


def _read_rows(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return _json_rows(path)
    if suffix in {".tsv", ".txt"}:
        return _tsv_rows(path)
    if suffix == ".parquet":
        return _parquet_rows(path)
    raise ValueError(f"Unsupported AppsRetrieval file type: {path}")


def _section_files(root: Path, section: str, split: str, *, optional: bool = False) -> list[Path]:
    """Find local JSONL/TSV/parquet files in common MTEB layouts."""
    direct = root / section
    if direct.is_file():
        return [direct]
    if direct.is_dir():
        files = sorted(path for path in direct.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
        if section == "qrels":
            split_files = [path for path in files if split.lower() in path.stem.lower()]
            files = split_files or files
        if files:
            return files

    candidates = []
    for path in root.iterdir() if root.exists() else []:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stem = path.stem.lower()
        if stem in {section, f"{section}_{split.lower()}", f"{section}-{split.lower()}"}:
            candidates.append(path)
    if candidates:
        return sorted(candidates)

    # Downloaded HF repositories sometimes put the section name in a nested
    # directory or file name.  Avoid cross-matching query/corpus/qrels names.
    nested = [
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and section in path.parts[:-1]
    ]
    if section == "qrels":
        split_nested = [path for path in nested if split.lower() in path.stem.lower()]
        nested = split_nested or nested
    if nested:
        return sorted(nested)
    if optional:
        return []
    raise FileNotFoundError(f"Could not find AppsRetrieval {section} files below {root}")


def _first_value(row: Mapping[str, Any], keys: Sequence[str], *, source: str) -> Any:
    values = [(key, row[key]) for key in keys if key in row and row[key] is not None]
    if not values:
        raise ValueError(f"{source} is missing one of: {', '.join(keys)}")
    nonempty = [(key, value) for key, value in values if str(value).strip()]
    if not nonempty:
        raise ValueError(f"{source} contains an empty identifier")
    if len(nonempty) > 1 and len({str(value) for _, value in nonempty}) > 1:
        raise ValueError(f"{source} contains conflicting identifier fields")
    return nonempty[0][1]


def _identifier(row: Mapping[str, Any], kind: str, source: str) -> str:
    keys = {
        "corpus": ("_id", "corpus-id", "corpus_id", "id"),
        "query": ("_id", "query-id", "query_id", "id"),
    }[kind]
    return str(_first_value(row, keys, source=source))


def _text(row: Mapping[str, Any], kind: str, source: str) -> str:
    keys = ("text", "code", "document") if kind == "corpus" else ("text", "query")
    if kind == "query" and any(key in row and row[key] is not None for key in keys):
        if not any(isinstance(row[key], str) and row[key].strip() for key in keys if key in row):
            raise ValueError(f"{source} contains an empty query")
    value = _first_value(row, keys, source=source)
    if not isinstance(value, str):
        raise ValueError(f"{source} text must be a string")
    if kind == "query" and not value.strip():
        raise ValueError(f"{source} contains an empty query")
    return value


def _parse_corpus(files: Iterable[Path]) -> "OrderedDict[str, CorpusDocument]":
    corpus: "OrderedDict[str, CorpusDocument]" = OrderedDict()
    for path in files:
        for row_number, row in enumerate(_read_rows(path), 1):
            source = f"{path}:{row_number}"
            corpus_id = _identifier(row, "corpus", source)
            if corpus_id in corpus:
                raise ValueError(f"Duplicate corpus ID {corpus_id!r} at {source}")
            corpus[corpus_id] = CorpusDocument(
                corpus_id=corpus_id,
                text=_text(row, "corpus", source),
                title=str(row.get("title") or ""),
            )
    if not corpus:
        raise ValueError("AppsRetrieval corpus is empty")
    return corpus


def _parse_queries(files: Iterable[Path]) -> "OrderedDict[str, str]":
    queries: "OrderedDict[str, str]" = OrderedDict()
    for path in files:
        for row_number, row in enumerate(_read_rows(path), 1):
            source = f"{path}:{row_number}"
            query_id = _identifier(row, "query", source)
            if query_id in queries:
                raise ValueError(f"Duplicate query ID {query_id!r} at {source}")
            queries[query_id] = _text(row, "query", source)
    if not queries:
        raise ValueError("AppsRetrieval query set is empty")
    return queries


def _parse_qrels(files: Iterable[Path], queries: Mapping[str, str], corpus: Mapping[str, CorpusDocument]) -> "OrderedDict[str, OrderedDict[str, float]]":
    qrels: "OrderedDict[str, OrderedDict[str, float]]" = OrderedDict((query_id, OrderedDict()) for query_id in queries)
    for path in files:
        rows = _read_rows(path)
        # A JSON qrels fixture may use {query: {document: score}}.
        if path.suffix.lower() == ".json" and rows and all("text" not in row and "score" not in row for row in rows):
            expanded = []
            for row in rows:
                query_id = str(row.get("_id", ""))
                for corpus_id, score in row.items():
                    if corpus_id != "_id":
                        expanded.append({"query-id": query_id, "corpus-id": corpus_id, "score": score})
            rows = expanded
        for row_number, row in enumerate(rows, 1):
            source = f"{path}:{row_number}"
            query_id = str(_first_value(row, ("query-id", "query_id", "qid", "_id"), source=source))
            corpus_id = str(_first_value(row, ("corpus-id", "corpus_id", "doc-id", "doc_id", "did", "id"), source=source))
            if query_id not in queries:
                raise ValueError(f"Qrels reference unknown query ID {query_id!r} at {source}")
            if corpus_id not in corpus:
                raise ValueError(f"Qrels reference unknown corpus ID {corpus_id!r} at {source}")
            try:
                score = float(_first_value(row, ("score", "relevance", "label"), source=source))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Qrels score must be numeric at {source}") from exc
            if not math.isfinite(score):
                raise ValueError(f"Qrels score must be finite at {source}")
            if corpus_id in qrels[query_id]:
                raise ValueError(f"Duplicate qrel ({query_id!r}, {corpus_id!r}) at {source}")
            qrels[query_id][corpus_id] = score
    return qrels


def load_apps_retrieval_data(dataset_dir: str | Path, split: str = "test", *, require_qrels: bool = False) -> AppsRetrievalData:
    """Load a local BEIR/MTEB-style AppsRetrieval directory.

    Supported layouts include ``corpus.jsonl``/``queries.jsonl`` with
    ``qrels/{split}.tsv`` and the extracted Hugging Face layout with
    ``corpus/``, ``queries/``, and ``qrels/`` parquet directories.
    """
    root = Path(dataset_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"AppsRetrieval dataset directory not found: {root}")
    if not split.strip():
        raise ValueError("split must not be empty")
    corpus = _parse_corpus(_section_files(root, "corpus", split))
    queries = _parse_queries(_section_files(root, "queries", split))
    qrel_files = _section_files(root, "qrels", split, optional=not require_qrels)
    qrels = _parse_qrels(qrel_files, queries, corpus) if qrel_files else None
    if require_qrels and qrels is None:
        raise FileNotFoundError(f"Could not find qrels for split {split!r} below {root}")
    return AppsRetrievalData(split=split, corpus=corpus, queries=queries, qrels=qrels)


def _result_value(result: Any, key: str) -> Any:
    if isinstance(result, Mapping):
        return result[key]
    return getattr(result, key)


def _validate_score(score: Any, source: str) -> float:
    if isinstance(score, bool):
        raise ValueError(f"{source} score must be numeric, not boolean")
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} score must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source} score must be finite")
    return value


def validate_predictions(payload: Mapping[str, Any], data: AppsRetrievalData, *, top_k: int = 10) -> None:
    """Validate the exact native MTEB retrieval prediction shape."""
    if not isinstance(payload, Mapping):
        raise ValueError("Prediction payload must be an object")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if set(payload) != {data.split}:
        raise ValueError(f"Prediction payload must contain exactly the {data.split!r} split")
    split_predictions = payload[data.split]
    if not isinstance(split_predictions, Mapping):
        raise ValueError("Prediction split must map query IDs to ranked corpus-score objects")
    if set(split_predictions) != data.query_ids:
        missing = sorted(data.query_ids - set(split_predictions))
        extra = sorted(set(split_predictions) - data.query_ids)
        raise ValueError(f"Prediction query IDs do not match input; missing={missing}, extra={extra}")
    for query_id in data.queries:
        ranking = split_predictions[query_id]
        if not isinstance(ranking, Mapping):
            raise ValueError(f"Prediction for query {query_id!r} must be an object")
        # An empty ranking is valid when the frozen filtering stage removes
        # every candidate.  The query still has an output entry; MTEB's
        # scorer handles the resulting zero-recall case.
        if not ranking:
            continue
        if len(ranking) > top_k:
            raise ValueError(f"Prediction for query {query_id!r} has {len(ranking)} items; top_k={top_k}")
        seen: set[str] = set()
        for corpus_id, score in ranking.items():
            corpus_id = str(corpus_id)
            if corpus_id in seen:
                raise ValueError(f"Duplicate corpus ID {corpus_id!r} in query {query_id!r}")
            seen.add(corpus_id)
            if corpus_id not in data.corpus_ids:
                raise ValueError(f"Prediction for query {query_id!r} references unknown corpus ID {corpus_id!r}")
            _validate_score(score, f"Prediction {query_id}/{corpus_id}")


def run_retrieval(data: AppsRetrievalData, engine: Any, *, top_k: int = 10) -> dict[str, dict[str, dict[str, float]]]:
    """Run an existing engine and return the native MTEB prediction payload."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    predictions: "OrderedDict[str, OrderedDict[str, float]]" = OrderedDict()
    for query_id, query in data.queries.items():
        results = engine.search(query, top_k=min(top_k, len(data.corpus)), filters=None, version="all")
        ranking: "OrderedDict[str, float]" = OrderedDict()
        for result in results:
            corpus_id = str(_result_value(result, "snippet_id"))
            if corpus_id in ranking:
                raise ValueError(f"Engine returned duplicate corpus ID {corpus_id!r} for query {query_id!r}")
            if corpus_id not in data.corpus_ids:
                raise ValueError(f"Engine returned unknown corpus ID {corpus_id!r} for query {query_id!r}")
            ranking[corpus_id] = _validate_score(_result_value(result, "score"), f"Engine {query_id}/{corpus_id}")
        predictions[query_id] = ranking
    payload = {data.split: predictions}
    validate_predictions(payload, data, top_k=top_k)
    return payload


def write_predictions(payload: Mapping[str, Any], path: str | Path, data: AppsRetrievalData, *, top_k: int = 10) -> Path:
    """Validate and write deterministic UTF-8 JSON."""
    validate_predictions(payload, data, top_k=top_k)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key {key!r}")
        result[key] = value
    return result


def load_and_validate_predictions(path: str | Path, data: AppsRetrievalData, *, top_k: int = 10) -> dict[str, Any]:
    """Read a prediction file with duplicate-key detection and validate it."""
    output = Path(path)
    try:
        payload = json.loads(output.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid prediction JSON at {output}: {exc}") from exc
    validate_predictions(payload, data, top_k=top_k)
    return payload


def metrics_from_predictions(payload: Mapping[str, Any], data: AppsRetrievalData, *, k: int = 10) -> dict[str, float] | None:
    """Compute project-compatible metrics when qrels are available.

    These are adapter-side metrics, not official MTEB output.  The report keeps
    that distinction explicit so they cannot be mistaken for an official run.
    """
    if data.qrels is None:
        return None
    validate_predictions(payload, data, top_k=max(k, 10))
    run = {query_id: list(payload[data.split][query_id]) for query_id in data.queries}
    return evaluate_run(run, data.qrels, k=k)
