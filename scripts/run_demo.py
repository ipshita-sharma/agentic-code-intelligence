#!/usr/bin/env python3
"""Run the offline sample demo without requiring a pre-built index."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RetrievalConfig
from src.ingestion.loader import load_jsonl
from src.pipeline import RetrievalEngine


def main() -> None:
    snippets = load_jsonl(ROOT / "data/sample/snippets.jsonl")
    engine = RetrievalEngine(snippets, RetrievalConfig(semantic_backend="hashing"))
    build_stats = engine.build_index()
    queries = [
        "How is the input preprocessed before going to the main function?",
        "Where is authentication token validation performed?",
        "Which function handles database connection retries?",
    ]
    print("Code Retrieval Engine demo (retrieval only; no answer generation)")
    print(f"Indexed {len(snippets)} snippets with {build_stats['semantic_backend']} embeddings")
    for query in queries:
        results = engine.search(query, top_k=3)
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)
        for result in results:
            name = result["function_name"] or result["class_name"] or "-"
            print(f"#{result['rank']} score={result['score']:.4f} {result['file_path']}::{name} [{result['version']}]")
        print(f"latency={engine.last_timings['total_seconds'] * 1000:.2f} ms")


if __name__ == "__main__":
    main()
