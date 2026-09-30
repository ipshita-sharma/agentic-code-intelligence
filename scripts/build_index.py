#!/usr/bin/env python3
"""Build a local index from JSONL snippets."""

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
    parser = argparse.ArgumentParser(description="Build the code retrieval index")
    parser.add_argument("--snippets", type=Path, default=ROOT / "data/sample/snippets.jsonl")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "outputs/index")
    parser.add_argument("--semantic-encoder", choices=["hashing", "minilm"], default=None, help="semantic encoder used for document/query embeddings")
    parser.add_argument("--semantic-backend", choices=["hashing", "auto", "sentence-transformers"], default=None, help="legacy alias for --semantic-encoder")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "outputs/embeddings.npz")
    args = parser.parse_args()
    snippets = load_jsonl(args.snippets)
    selected_encoder = args.semantic_encoder
    if selected_encoder is None and args.semantic_backend:
        selected_encoder = {"sentence-transformers": "minilm", "auto": "minilm"}.get(args.semantic_backend, args.semantic_backend)
    config = RetrievalConfig(
        semantic_encoder=selected_encoder or "hashing",
        semantic_model=args.model,
        index_dir=args.index_dir,
        embedding_cache=args.embedding_cache,
    )
    engine = RetrievalEngine(snippets, config)
    stats = engine.build_index()
    engine.save_index(args.index_dir)
    print(json.dumps({**stats, "index_dir": str(args.index_dir)}, indent=2))


if __name__ == "__main__":
    main()
