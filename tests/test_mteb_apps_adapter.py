"""Contract tests for the CoIR/MTEB AppsRetrieval adapter."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pytest

from src.evaluation.mteb_apps_adapter import (
    AppsRetrievalData,
    CorpusDocument,
    load_and_validate_predictions,
    load_apps_retrieval_data,
    metrics_from_predictions,
    run_retrieval,
    validate_predictions,
    write_predictions,
)


class FakeEngine:
    def __init__(self, results=None):
        self.results = results or [
            {"snippet_id": "corpus-a", "score": 0.9},
            {"snippet_id": "corpus-b", "score": 0.2},
            {"snippet_id": "corpus-c", "score": 0.1},
        ]

    def search(self, query, top_k, filters=None, version="all"):
        del query, filters, version
        return self.results[:top_k]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _dataset(tmp_path: Path, *, empty_query: bool = False) -> Path:
    root = tmp_path / "appsretrieval"
    root.mkdir()
    _write_jsonl(root / "corpus.jsonl", [
        {"_id": "corpus-a", "title": "A", "text": "def alpha(): pass"},
        {"_id": "corpus-b", "title": "B", "text": "def beta(): pass"},
        {"_id": "corpus-c", "title": "C", "text": "def gamma(): pass"},
    ])
    _write_jsonl(root / "queries.jsonl", [
        {"_id": "query-1", "text": "" if empty_query else "find alpha"},
        {"_id": "query-2", "text": "find beta"},
    ])
    qrels = root / "qrels"
    qrels.mkdir()
    (qrels / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nquery-1\tcorpus-a\t2\nquery-2\tcorpus-b\t1\n",
        encoding="utf-8",
    )
    return root


def _data() -> AppsRetrievalData:
    return AppsRetrievalData(
        split="test",
        corpus=OrderedDict([
            ("corpus-a", CorpusDocument("corpus-a", "alpha")),
            ("corpus-b", CorpusDocument("corpus-b", "beta")),
            ("corpus-c", CorpusDocument("corpus-c", "gamma")),
        ]),
        queries=OrderedDict([( "query-1", "find alpha"), ("query-2", "find beta")]),
        qrels=OrderedDict([
            ("query-1", OrderedDict([( "corpus-a", 2.0)])),
            ("query-2", OrderedDict([( "corpus-b", 1.0)])),
        ]),
    )


def test_appsretrieval_input_parsing_and_id_preservation(tmp_path):
    data = load_apps_retrieval_data(_dataset(tmp_path), require_qrels=True)
    assert list(data.corpus) == ["corpus-a", "corpus-b", "corpus-c"]
    assert list(data.queries) == ["query-1", "query-2"]
    assert data.qrels["query-1"]["corpus-a"] == 2.0
    assert [snippet.id for snippet in data.to_snippets()] == list(data.corpus)


def test_empty_query_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty query"):
        load_apps_retrieval_data(_dataset(tmp_path, empty_query=True))


def test_missing_or_invalid_ids_are_rejected(tmp_path):
    root = _dataset(tmp_path)
    _write_jsonl(root / "corpus.jsonl", [{"text": "missing id"}])
    with pytest.raises(ValueError, match="missing one of"):
        load_apps_retrieval_data(root)


def test_duplicate_corpus_ids_are_rejected(tmp_path):
    root = _dataset(tmp_path)
    _write_jsonl(root / "corpus.jsonl", [
        {"_id": "same", "text": "one"},
        {"_id": "same", "text": "two"},
    ])
    with pytest.raises(ValueError, match="Duplicate corpus ID"):
        load_apps_retrieval_data(root)


def test_output_schema_and_top_k_truncation():
    data = _data()
    payload = run_retrieval(data, FakeEngine(), top_k=2)
    validate_predictions(payload, data, top_k=2)
    assert set(payload) == {"test"}
    assert list(payload["test"]["query-1"]) == ["corpus-a", "corpus-b"]
    assert all(isinstance(score, float) for score in payload["test"]["query-1"].values())


def test_unknown_and_duplicate_engine_ids_are_rejected():
    data = _data()
    with pytest.raises(ValueError, match="unknown corpus ID"):
        run_retrieval(data, FakeEngine([{"snippet_id": "not-in-corpus", "score": 1.0}]))
    with pytest.raises(ValueError, match="duplicate corpus ID"):
        run_retrieval(data, FakeEngine([
            {"snippet_id": "corpus-a", "score": 1.0},
            {"snippet_id": "corpus-a", "score": 0.9},
        ]))


def test_query_completeness_and_numeric_score_validation():
    data = _data()
    with pytest.raises(ValueError, match="query IDs"):
        validate_predictions({"test": {"query-1": {"corpus-a": 1.0}}}, data, top_k=1)
    with pytest.raises(ValueError, match="numeric"):
        validate_predictions({"test": {
            "query-1": {"corpus-a": "not-a-score"},
            "query-2": {"corpus-b": 1.0},
        }}, data, top_k=1)


def test_empty_ranking_is_valid_when_filtering_removes_all_candidates():
    data = _data()
    payload = {"test": {
        "query-1": {},
        "query-2": {"corpus-b": 1.0},
    }}
    validate_predictions(payload, data, top_k=1)


def test_duplicate_json_keys_are_rejected(tmp_path):
    data = _data()
    path = tmp_path / "predictions.json"
    path.write_text(
        '{"test":{"query-1":{"corpus-a":1,"corpus-a":0.5},"query-2":{"corpus-b":1}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        load_and_validate_predictions(path, data, top_k=1)


def test_repeated_evaluation_is_byte_deterministic(tmp_path):
    data = _data()
    first = write_predictions(run_retrieval(data, FakeEngine(), top_k=2), tmp_path / "one.json", data, top_k=2)
    second = write_predictions(run_retrieval(data, FakeEngine(), top_k=2), tmp_path / "two.json", data, top_k=2)
    assert first.read_bytes() == second.read_bytes()
    assert load_and_validate_predictions(first, data, top_k=2)["test"]["query-1"] == {"corpus-a": 0.9, "corpus-b": 0.2}


def test_adapter_side_qrels_metrics_are_available_but_local():
    data = _data()
    payload = run_retrieval(data, FakeEngine(), top_k=3)
    metrics = metrics_from_predictions(payload, data)
    assert metrics["MRR"] == 0.75
    assert metrics["NDCG@10"] > 0.0


def test_local_regression_benchmark_contract_is_explicit():
    from scripts.evaluate_mteb_apps import EXPECTED_LOCAL

    assert EXPECTED_LOCAL == {"MRR": 0.9722, "NDCG@10": 0.9326, "Recall@5": 0.9464, "Recall@10": 0.9821}
