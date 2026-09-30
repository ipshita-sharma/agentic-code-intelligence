#!/usr/bin/env python3
"""Retrieve and print ranked code snippets."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RetrievalConfig
from src.ingestion.loader import load_jsonl
from src.pipeline import RetrievalEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve relevant code snippets")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--version", default="all")
    parser.add_argument("--language")
    parser.add_argument("--repository")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "outputs/index")
    parser.add_argument("--snippets", type=Path, default=ROOT / "data/sample/snippets.jsonl")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if (args.index_dir / "snippets.json").exists():
        engine = RetrievalEngine.load_index(args.index_dir, RetrievalConfig(index_dir=args.index_dir))
    else:
        engine = RetrievalEngine(load_jsonl(args.snippets), RetrievalConfig(semantic_backend="hashing"))
        engine.build_index()
    filters = {key: value for key, value in {"language": args.language, "repository": args.repository}.items() if value}
    results = engine.search(args.query, args.top_k, filters=filters, version=args.version)
    if args.as_json:
        print(json.dumps({"results": results, "timings": engine.last_timings}, indent=2))
        return
    print("=" * 60)
    print("QUERY")
    print("=" * 60)
    print(args.query)
    print("\n" + "=" * 60)
    print("TOP RESULTS")
    print("=" * 60)
    for result in results:
        print(f"\n#{result['rank']}  Score: {result['score']:.4f}")
        print(f"File: {result['file_path']}  |  Function: {result['function_name'] or result['class_name'] or '-'}")
        print(f"Language: {result['language']}  |  Version: {result['version']}")
        print(result["code"])
        print("-" * 60)
    print(f"Latency: {engine.last_timings['total_seconds'] * 1000:.2f} ms")


if __name__ == "__main__":
    main()
