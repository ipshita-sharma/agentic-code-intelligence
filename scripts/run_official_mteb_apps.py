#!/usr/bin/env python3
"""Run the accepted retrieval pipeline through the installed MTEB task API.

This is an evaluation-only bridge.  It implements MTEB's current
``SearchProtocol`` around the unchanged ``RetrievalEngine`` and exports the
downloaded CoIR task into the existing adapter's JSONL/TSV layout for a direct
schema/ID compatibility check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RetrievalConfig
from src.evaluation.mteb_apps_adapter import load_apps_retrieval_data
from src.ingestion.loader import load_jsonl
from src.models import CodeSnippet
from src.pipeline import RetrievalEngine


EXPECTED_LOCAL = {
    "MRR": 0.9722,
    "NDCG@10": 0.9326,
    "Recall@5": 0.9464,
    "Recall@10": 0.9821,
}


def _production_config(output_dir: Path) -> RetrievalConfig:
    config = RetrievalConfig(
        semantic_encoder="minilm",
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_cache=output_dir / "official_mteb_minilm_embeddings.npz",
        index_dir=output_dir / "official_mteb_index",
        enable_reranking=True,
    )
    config.rerank_alpha = 0.05
    return config


def _rows(dataset: Any) -> list[dict[str, Any]]:
    return [dict(dataset[index]) for index in range(len(dataset))]


def _export_official_data(task: Any, output_dir: Path) -> dict[str, Any]:
    split = task.dataset["default"]["test"]
    corpus_rows = _rows(split["corpus"])
    query_rows = _rows(split["queries"])
    qrels = split["relevant_docs"]
    query_ids = set(qrels)
    query_rows = [row for row in query_rows if str(row.get("id", row.get("_id"))) in query_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    queries_path = output_dir / "queries.jsonl"
    qrels_dir = output_dir / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in corpus_rows) + "\n", encoding="utf-8")
    queries_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in query_rows) + "\n", encoding="utf-8")
    qrel_lines = ["query-id\tcorpus-id\tscore"]
    for query_id, docs in qrels.items():
        for corpus_id, score in docs.items():
            qrel_lines.append(f"{query_id}\t{corpus_id}\t{score}")
    qrels_path = qrels_dir / "test.tsv"
    qrels_path.write_text("\n".join(qrel_lines) + "\n", encoding="utf-8")

    parsed = load_apps_retrieval_data(output_dir, split="test", require_qrels=True)
    return {
        "directory": str(output_dir),
        "files": [str(corpus_path), str(queries_path), str(qrels_path)],
        "corpus_count": len(parsed.corpus),
        "query_count": len(parsed.queries),
        "qrel_count": sum(len(values) for values in (parsed.qrels or {}).values()),
        "adapter_consumed": True,
    }


def _snippets_from_corpus(corpus: Any) -> list[CodeSnippet]:
    snippets = []
    for row in _rows(corpus):
        corpus_id = str(row.get("id", row.get("_id")))
        snippets.append(CodeSnippet(
            id=corpus_id,
            repository="mteb-appsretrieval",
            file_path="",
            language=str(row.get("language") or "unknown").lower(),
            version="official",
            commit_id="official",
            code=str(row.get("text") or ""),
            docstring=str(row.get("title") or ""),
            snippet_type="generic",
        ))
    return snippets


class PipelineSearch:
    """MTEB SearchProtocol facade over the accepted retrieval engine."""

    def __init__(self, output_dir: Path, workers: int = 4):
        from mteb.models.model_meta import ModelMeta

        self.output_dir = output_dir
        self.workers = max(1, int(workers))
        self.engine: RetrievalEngine | None = None
        self.build_stats: dict[str, Any] = {}
        self.mteb_model_meta = ModelMeta.create_empty({
            "name": "local/code-retrieval-engine-hybrid",
            "revision": "accepted-system",
            "modalities": ["text"],
        })

    def index(self, corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs, num_proc):
        del task_metadata, hf_split, hf_subset, encode_kwargs, num_proc
        config = _production_config(self.output_dir)
        self.engine = RetrievalEngine(_snippets_from_corpus(corpus), config)
        started = time.perf_counter()
        self.build_stats = dict(self.engine.build_index())
        self.build_stats["wall_seconds"] = time.perf_counter() - started

    def search(self, queries, *, task_metadata, hf_split, hf_subset, top_k, encode_kwargs, top_ranked=None, num_proc=None):
        del task_metadata, hf_split, hf_subset, encode_kwargs, top_ranked, num_proc
        if self.engine is None:
            raise RuntimeError("MTEB called search before index")
        rows = _rows(queries)

        def retrieve(row: dict[str, Any]) -> tuple[str, OrderedDict[str, float]]:
            query_id = str(row.get("id", row.get("_id")))
            results = self.engine.search(str(row.get("text") or ""), top_k=top_k, filters=None, version="all")
            return query_id, OrderedDict((str(item["snippet_id"]), float(item["score"])) for item in results)

        if self.workers == 1:
            ranked = [retrieve(row) for row in rows]
        else:
            # Calls are independent; preserve the input order when rebuilding
            # the mapping so prediction JSON remains deterministic.
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                ranked = list(pool.map(retrieve, rows))
        predictions = OrderedDict(ranked)
        return predictions


def _result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    return {"value": str(result)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mteb-evaluation")
    parser.add_argument("--dataset-cache", type=Path, default=Path("/private/tmp/codex-hf-datasets"))
    parser.add_argument("--hub-cache", type=Path, default=None, help="Optional Hugging Face hub cache override; default keeps the existing user cache.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MTEB_CACHE", str(args.output_dir / "mteb-cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.dataset_cache))
    if args.hub_cache is not None:
        os.environ.setdefault("HF_HUB_CACHE", str(args.hub_cache))
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import mteb

    tasks = mteb.get_tasks(tasks=["AppsRetrieval"])
    task = tasks[0]
    # The requested official metrics are MRR/NDCG@10.  Restricting MTEB's
    # evaluation cutoffs to 10 avoids a 1,000-result search that the frozen
    # engine cannot usefully expose beyond its existing fusion candidate pool.
    # This changes only evaluation workload, not retrieval behavior or scores
    # within the requested cutoff.
    task.k_values = (1, 3, 5, 10)
    task._top_k = 10
    task.load_data()
    task_metadata = task.metadata
    split = task.dataset["default"]["test"]
    exported = _export_official_data(task, args.output_dir / "official-appsretrieval-data")

    model = PipelineSearch(args.output_dir, workers=args.workers)
    prediction_dir = args.output_dir / "official-predictions"
    started = time.perf_counter()
    result = mteb.evaluate(
        model,
        tasks=[task],
        cache=None,
        overwrite_strategy="always",
        prediction_folder=prediction_dir,
        show_progress_bar=False,
        encode_kwargs={"show_progress_bar": False},
    )
    elapsed = time.perf_counter() - started
    result_dict = _result_to_dict(result)
    report = {
        "status": "passed",
        "environment": {
            "python": sys.version,
            "mteb_version": getattr(mteb, "__version__", None),
            "datasets_version": __import__("datasets").__version__,
            "sentence_transformers_version": __import__("sentence_transformers").__version__,
            "transformers_version": __import__("transformers").__version__,
            "mteb_cache": os.environ["MTEB_CACHE"],
            "datasets_cache": os.environ["HF_DATASETS_CACHE"],
            "hub_cache": os.environ.get("HF_HUB_CACHE", "default Hugging Face cache"),
        },
        "task": {
            "name": task_metadata.name,
            "dataset_path": task_metadata.dataset["path"],
            "revision": task_metadata.dataset["revision"],
            "split": "test",
            "main_score": task_metadata.main_score,
            "evaluation_k_values": list(task.k_values),
            "query_workers": args.workers,
            "corpus_count": len(split["corpus"]),
            "query_count": len(split["queries"]),
            "qrel_query_count": len(split["relevant_docs"]),
        },
        "dataset_export": exported,
        "pipeline_configuration": {
            "semantic_encoder": "minilm",
            "semantic_model": "sentence-transformers/all-MiniLM-L6-v2",
            "bm25": True,
            "hybrid_fusion": True,
            "filtering": True,
            "query_understanding": True,
            "reranking": True,
            "rerank_alpha": 0.05,
            "code_aware_ingestion": False,
        },
        "exact_command": "MTEB_CACHE=... HF_DATASETS_CACHE=... HF_HUB_CACHE=... HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python3 scripts/run_official_mteb_apps.py",
        "official_evaluation": {
            "ran": True,
            "api": "mteb.evaluate(SearchProtocol, tasks=[AppsRetrieval], prediction_folder=...)",
            "elapsed_seconds": elapsed,
            "prediction_file": str(prediction_dir / "AppsRetrieval_predictions.json"),
            "result": result_dict,
        },
        "pipeline_build": model.build_stats,
        "local_regression": {"status": "pending; run by the calling evaluation command"},
        "limitations": [
            "MTEB evaluates this hybrid pipeline through a SearchProtocol facade; it is not a pure MiniLM encoder score.",
            "The MTEB task's official main score is ndcg_at_10; the full result object is retained in this report.",
        ],
    }
    (args.output_dir / "official_evaluation_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "prediction_file": report["official_evaluation"]["prediction_file"], "mteb_version": report["environment"]["mteb_version"], "task": report["task"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
