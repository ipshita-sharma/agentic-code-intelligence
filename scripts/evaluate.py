#!/usr/bin/env python3
"""Run local qrels evaluation and optionally the official MTEB adapter."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RetrievalConfig
from src.evaluation.metrics import evaluate_run
from src.evaluation.mteb_evaluator import RetrievalEncoder, run_mteb
from src.ingestion.loader import load_jsonl
from src.pipeline import RetrievalEngine


def load_queries(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def local_evaluation(engine: RetrievalEngine, query_path: Path, filters=None) -> dict:
    run, qrels, latencies = {}, {}, []
    for item in load_queries(query_path):
        started = time.perf_counter()
        results = engine.search(item["query"], top_k=10, filters=filters)
        latencies.append(time.perf_counter() - started)
        run[item["query_id"]] = [result["snippet_id"] for result in results]
        qrels[item["query_id"]] = {str(key): float(value) for key, value in item["relevance"].items()}
    report = evaluate_run(run, qrels, k=10)
    report["latency_ms_mean"] = statistics.mean(latencies) * 1000
    report["latency_ms_p95"] = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] * 1000
    return report


def ablation_suite(snippets, query_path: Path) -> dict:
    """Compare the five requested retrieval configurations on identical qrels."""
    base = RetrievalConfig(semantic_backend="hashing")
    configurations = {
        "A_bm25_only": RetrievalConfig(**{**base.__dict__, "semantic_candidates": 0, "enable_reranking": False}),
        "B_semantic_only": RetrievalConfig(**{**base.__dict__, "lexical_candidates": 0, "enable_reranking": False}),
        "C_bm25_plus_semantic": RetrievalConfig(**{**base.__dict__, "enable_reranking": False}),
        "D_hybrid_plus_filtering": RetrievalConfig(**{**base.__dict__, "enable_reranking": False}),
        "E_hybrid_plus_filtering_plus_reranking": base,
    }
    reports = {}
    for name, config in configurations.items():
        engine = RetrievalEngine(snippets, config)
        engine.build_index()
        # The sample repository filter is deliberately explicit so this path
        # exercises the same metadata stage used for real multi-repo corpora.
        filters = {"repository": "sample-app"} if name.startswith(("D_", "E_")) else None
        reports[name] = local_evaluation(engine, query_path, filters=filters)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mteb", action="store_true")
    parser.add_argument("--snippets", type=Path, default=ROOT / "data/sample/snippets.jsonl")
    parser.add_argument("--queries", type=Path, default=ROOT / "data/sample/queries.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/local_evaluation.json")
    parser.add_argument("--ablations", action="store_true", help="also compare BM25, semantic, hybrid, filtered, and reranked variants")
    args = parser.parse_args()
    engine = RetrievalEngine(load_jsonl(args.snippets), RetrievalConfig(semantic_backend="hashing"))
    engine.build_index()
    report = local_evaluation(engine, args.queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.ablations:
        ablations = ablation_suite(load_jsonl(args.snippets), args.queries)
        (ROOT / "outputs/ablation_results.json").write_text(json.dumps(ablations, indent=2), encoding="utf-8")
        print(json.dumps({"ablations": ablations}, indent=2))
    if args.mteb:
        try:
            result = run_mteb(RetrievalEncoder(engine.retriever.semantic), ROOT / "outputs/mteb")
            output_path = ROOT / "outputs/appsretrieval_results.json"
            output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            print(f"MTEB results written to {output_path}")
        except RuntimeError as exc:
            print(f"MTEB unavailable: {exc}", file=sys.stderr)
            raise SystemExit(2)


if __name__ == "__main__":
    main()
