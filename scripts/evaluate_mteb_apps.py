#!/usr/bin/env python3
"""Generate and validate native CoIR/MTEB AppsRetrieval predictions.

This entry point is an evaluation adapter, not a second retrieval pipeline.
The production configuration is constructed explicitly so the adapter cannot
silently fall back to hashing or disable the accepted reranker.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RetrievalConfig
from src.evaluation.metrics import evaluate_run
from src.evaluation.mteb_apps_adapter import (
    AppsRetrievalData,
    CorpusDocument,
    load_apps_retrieval_data,
    metrics_from_predictions,
    run_retrieval,
    validate_predictions,
    write_predictions,
)
from src.ingestion.loader import load_jsonl
from src.pipeline import RetrievalEngine


DEFAULT_REPORT_DIR = ROOT / "outputs/mteb-evaluation"
DEFAULT_DATASET_DIR = ROOT / "data/mteb/AppsRetrieval"
DEFAULT_AUDIT_QUERIES = ROOT.parent / "retrieval-audit/audit_queries.jsonl"
EXPECTED_LOCAL = {
    "MRR": 0.9722,
    "NDCG@10": 0.9326,
    "Recall@5": 0.9464,
    "Recall@10": 0.9821,
}


def _package_status(name: str) -> dict[str, Any]:
    available = importlib.util.find_spec(name) is not None
    version = None
    if available:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
        except Exception as exc:  # pragma: no cover - package-specific import errors
            return {"available": True, "version": None, "import_error": str(exc)}
    return {"available": available, "version": version}


def _load_audit_queries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _production_config(output_dir: Path) -> RetrievalConfig:
    config = RetrievalConfig(
        semantic_encoder="minilm",
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_cache=output_dir / "local_regression_minilm_embeddings.npz",
        index_dir=output_dir / "local_regression_index",
        enable_reranking=True,
    )
    # This is the accepted system setting.  It is deliberately an experiment
    # attribute because the frozen RetrievalConfig contract has not changed.
    config.rerank_alpha = 0.05
    return config


def run_local_regression(query_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the unchanged 42-query regression against the accepted system."""
    if not query_path.exists():
        return {"status": "unavailable", "reason": f"query file not found: {query_path}"}

    snippets = load_jsonl(ROOT / "data/sample/snippets.jsonl")
    queries = _load_audit_queries(query_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _production_config(output_dir)
    started_build = time.perf_counter()
    engine = RetrievalEngine(snippets, config)
    build_stats = dict(engine.build_index())
    build_stats["wall_seconds"] = time.perf_counter() - started_build

    run: dict[str, list[str]] = {}
    qrels: dict[str, dict[str, float]] = {}
    latencies_ms: list[float] = []
    for item in queries:
        started = time.perf_counter()
        results = engine.search(
            item["query"],
            top_k=10,
            filters=item.get("filters"),
            version="all",
        )
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        run[item["query_id"]] = [result["snippet_id"] for result in results]
        qrels[item["query_id"]] = {str(key): float(value) for key, value in item["relevance"].items()}

    metrics = evaluate_run(run, qrels, k=10)
    metrics["average_query_latency_ms"] = statistics.mean(latencies_ms) if latencies_ms else 0.0
    metrics["p95_query_latency_ms"] = sorted(latencies_ms)[max(0, int(len(latencies_ms) * 0.95) - 1)] if latencies_ms else 0.0
    metric_match = all(abs(metrics[key] - expected) < 1e-9 for key, expected in {
        "MRR": 0.9722222222222223,
        "NDCG@10": 0.932596469068554,
        "Recall@5": 0.9464285714285714,
        "Recall@10": 0.9821428571428571,
    }.items())
    return {
        "status": "passed" if metric_match else "regression",
        "query_count": len(queries),
        "configuration": {
            "semantic_encoder": "minilm",
            "semantic_model": config.semantic_model,
            "enable_reranking": True,
            "rerank_alpha": 0.05,
            "code_aware_ingestion": False,
        },
        "build": build_stats,
        "metrics": metrics,
        "expected_metrics": EXPECTED_LOCAL,
        "matches_accepted_baseline": metric_match,
        "latency_samples_count": len(latencies_ms),
    }


class _FixtureEngine:
    """Small deterministic engine used only for adapter schema validation."""

    def search(self, query: str, top_k: int, filters=None, version: str = "all") -> list[dict[str, Any]]:
        del query, filters, version
        return [
            {"snippet_id": "doc-1", "score": 0.9},
            {"snippet_id": "doc-2", "score": 0.1},
        ][:top_k]


def run_schema_fixture(output_path: Path | None = None) -> dict[str, Any]:
    data = AppsRetrievalData(
        split="test",
        corpus=OrderedDict([
            ("doc-1", CorpusDocument("doc-1", "def first(): pass")),
            ("doc-2", CorpusDocument("doc-2", "def second(): pass")),
        ]),
        queries=OrderedDict([("query-1", "find the first function")]),
    )
    payload = run_retrieval(data, _FixtureEngine(), top_k=2)
    validate_predictions(payload, data, top_k=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    result = {
        "passed": True,
        "shape": "{split: {query_id: {corpus_id: numeric_score}}}",
        "query_ids": list(payload["test"]),
        "corpus_ids": list(payload["test"]["query-1"]),
        "top_k": 2,
    }
    if output_path is not None:
        result["fixture_output"] = str(output_path)
    return result


def run_tests() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    summary = (result.stdout.strip().splitlines() or [""])[-1]
    return {"status": "passed" if result.returncode == 0 else "failed", "returncode": result.returncode, "summary": summary, "command": "pytest -q", "stderr_tail": result.stderr[-1000:]}


def _markdown_report(report: dict[str, Any]) -> str:
    official = report["official_evaluation"]
    local = report["local_regression"]
    local_metrics = local.get("metrics", {})
    lines = [
        "# MTEB / CoIR AppsRetrieval Evaluation Readiness",
        "",
        "## 1. Interface discovered",
        "",
        "AppsRetrieval uses the BEIR/MTEB retrieval layout: corpus records, query records, and relevance judgments (qrels). The native retrieval prediction shape is `split -> query_id -> corpus_id -> numeric score`; insertion order is the ranking order.",
        "",
        "The current documented MTEB evaluation entry point is `mteb.evaluate(model, tasks=tasks, prediction_folder=...)`; the retrieval prediction artifact is written below that folder using the task name. CoIR identifies the task as `AppsRetrieval` and follows the same retrieval data conventions. No installed MTEB package was available in this environment, so the exact installed package version could not be inspected.",
        "",
        "## 2. Adapter architecture",
        "",
        "- Parses JSON/JSONL, TSV, and optional parquet corpus/query/qrels files.",
        "- Preserves official query and corpus IDs as string keys and rejects duplicates or unknown references.",
        "- Converts corpus rows to the existing `CodeSnippet` contract without changing retrieval code.",
        "- Runs the accepted MiniLM + BM25 + hybrid + filtering + alpha=0.05 reranker configuration only when official data is supplied.",
        "- Validates and writes deterministic JSON with no extra metadata in the inference file.",
        "",
        "## 3. Output schema",
        "",
        "```json",
        '{"test": {"query-id": {"corpus-id": 0.123}}}',
        "```",
        "",
        "Validation covers query completeness, corpus-ID validity, numeric finite scores, duplicate prevention, top-k truncation, deterministic ordering, and strict JSON duplicate-key detection.",
        "",
        "## 4. Dependency and data status",
        "",
        f"- `mteb`: available=`{report['dependencies']['mteb']['available']}`, version=`{report['dependencies']['mteb']['version']}`",
        f"- `datasets`: available=`{report['dependencies']['datasets']['available']}`, version=`{report['dependencies']['datasets']['version']}`",
        f"- Official AppsRetrieval data present locally: `{report['dataset_availability']['present']}`",
        f"- Official evaluation executed: `{official['ran']}`",
        "",
        "The environment has a local MiniLM snapshot, but package and dataset downloads were not available. No official dataset or MTEB-generated metric is fabricated here.",
        "The optional MTEB dependency was not installed during this task: `requirements.txt` already documents it as optional, and adding/installing it without network access would not make the official evaluation runnable.",
        "",
        "## 5. Official evaluation",
        "",
        f"{official['reason']}",
        "",
        "Official metrics: **not measured**.",
        "",
        "## 6. Local regression",
        "",
        "| Metric | Result | Accepted |",
        "|---|---:|---:|",
        f"| MRR | {local_metrics.get('MRR', 'n/a')} | 0.9722 |",
        f"| NDCG@10 | {local_metrics.get('NDCG@10', 'n/a')} | 0.9326 |",
        f"| Recall@5 | {local_metrics.get('Recall@5', 'n/a')} | 0.9464 |",
        f"| Recall@10 | {local_metrics.get('Recall@10', 'n/a')} | 0.9821 |",
        f"| Average latency (ms) | {local_metrics.get('average_query_latency_ms', 'n/a')} | timing-only |",
        "",
        f"Regression status: **{local.get('status')}**; query count: `{local.get('query_count', 'n/a')}`.",
        "",
        "## 7. Remaining steps",
        "",
        "1. Make the official AppsRetrieval files available locally (or install `mteb` and its dataset dependencies in a network-enabled environment).",
        "2. Run `python scripts/evaluate_mteb_apps.py --dataset-dir <AppsRetrieval-directory>`.",
        "3. Submit the generated `AppsRetrieval_predictions.json` to the installed MTEB/CoIR evaluator or run its official task scoring path.",
        "",
        "## 8. Limitations",
        "",
        "- This run does not claim official AppsRetrieval performance because the official data and MTEB package were unavailable.",
        "- Adapter-side qrels metrics, when available, are diagnostic and are not labeled as official MTEB metrics.",
        "- The old `scripts/evaluate.py --mteb` path wraps only the semantic encoder; this new entry point is the pipeline-compatible adapter.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_DIR / "AppsRetrieval_predictions.json")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--audit-queries", type=Path, default=DEFAULT_AUDIT_QUERIES)
    parser.add_argument("--skip-local-regression", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    dataset_present = args.dataset_dir.exists() and args.dataset_dir.is_dir()
    report: dict[str, Any] = {
        "adapter": "src.evaluation.mteb_apps_adapter",
        "entry_point": str(Path(__file__).resolve()),
        "dependencies": {"mteb": _package_status("mteb"), "datasets": _package_status("datasets")},
        "dependency_installation": "mteb was not added or installed; it is already documented as optional in requirements.txt and package downloads were unavailable.",
        "mteb_interface": {
            "task_name": "AppsRetrieval",
            "evaluation_call": "mteb.evaluate(model, tasks=tasks, prediction_folder=prediction_folder)",
            "prediction_filename": "AppsRetrieval_predictions.json",
            "prediction_schema": "{split: {query_id: {corpus_id: numeric_score}}}",
            "split_used_by_adapter": args.split,
        },
        "dataset_availability": {"directory": str(args.dataset_dir), "present": dataset_present},
        "schema_validation": run_schema_fixture(args.report_dir / "schema_fixture_predictions.json"),
        "official_evaluation": {"ran": False, "reason": "Official AppsRetrieval evaluation was not run: the dataset and/or MTEB package are unavailable in this environment."},
        "official_metrics": None,
        "inference_output": None,
    }

    if dataset_present:
        try:
            data = load_apps_retrieval_data(args.dataset_dir, args.split, require_qrels=False)
            config = _production_config(args.report_dir)
            engine = RetrievalEngine(data.to_snippets(), config)
            build = engine.build_index()
            payload = run_retrieval(data, engine, top_k=args.top_k)
            output_path = write_predictions(payload, args.output, data, top_k=args.top_k)
            report["inference_output"] = {"path": str(output_path), "query_count": len(data.queries), "corpus_count": len(data.corpus), "build": build}
            report["adapter_side_metrics"] = metrics_from_predictions(payload, data, k=10)
            report["official_evaluation"] = {"ran": False, "reason": "Adapter inference was generated, but official MTEB scoring was not run because the installed MTEB package is unavailable. The JSON is ready for the official evaluator."}
        except Exception as exc:
            report["inference_output"] = {"status": "failed", "error": str(exc)}
    else:
        report["official_evaluation"]["reason"] = "Official AppsRetrieval evaluation was not run: no local AppsRetrieval corpus/query/qrels directory was found, and no network download was attempted."

    report["local_regression"] = {"status": "skipped"} if args.skip_local_regression else run_local_regression(args.audit_queries, args.report_dir)
    report["tests"] = {"status": "skipped"} if args.skip_tests else run_tests()
    report["official_interface_references"] = {
        "mteb_evaluation_api": "https://docs.mteb.org/api/evaluation/",
        "mteb_retrieval_task_api": "https://docs.mteb.org/api/task/",
        "coir_appsretrieval": "https://github.com/CoIR-team/coir",
        "appsretrieval_dataset": "https://huggingface.co/datasets/mteb/AppsRetrieval",
    }

    json_path = args.report_dir / "mteb_evaluation_report.json"
    md_path = args.report_dir / "mteb_evaluation_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps({"report_json": str(json_path), "report_markdown": str(md_path), "official_ran": report["official_evaluation"]["ran"], "local_regression": report["local_regression"].get("status")}, indent=2))
    return 0 if report["tests"].get("status") in {"passed", "skipped"} and report["local_regression"].get("status") in {"passed", "skipped", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
