# Code Retrieval Engine

A CPU-friendly, modular prototype for the **Agentic Code Intelligence** hackathon problem. Given a natural-language query and a collection of versioned code snippets, it retrieves and ranks the most relevant snippets.

> **This system performs retrieval only. Answer generation is intentionally out of scope.**

## Project overview

The engine targets P0 retrieval accuracy—measured with MRR and NDCG@10—and includes P1-ready repository/version metadata. It is not a chatbot, code generator, or generic RAG answerer.

## Architecture and pipeline

```text
Natural-language query
        |
  QueryAnalyzer + conservative preprocessor
        |
  BM25 weighted-field retrieval  ----\\
                                      > RRF/weighted fusion
  normalized semantic retrieval ----/
        |
  metadata/version filtering
        |
  explainable feature reranking
        |
  top-k CodeSnippet results + intermediate scores
```

The stages are separated under `src/`, so each can be tuned or replaced:

1. Query understanding classifies authentication, API, database, preprocessing, testing, performance, and general code-search intents.
2. Query preprocessing normalizes prose while preserving identifiers, camelCase/snake_case pieces, API names, and programming vocabulary.
3. Lexical retrieval uses field-weighted BM25 over function/class names, docstrings, imports, paths, and source code.
4. Semantic retrieval supports the configurable `sentence-transformers/all-MiniLM-L6-v2` adapter. The default `hashing` backend is deterministic, CPU-only, and requires no model download.
5. Fusion supports Reciprocal Rank Fusion (`rrf`) and normalized weighted scores. Candidate streams are merged by ID rather than concatenated.
6. Filtering runs before reranking and supports language, repository, version, snippet type, and file-path constraints.
7. Reranking combines semantic score, BM25 score, identifier overlap, intent and metadata compatibility, and version preference. Weights live in `src/config.py`.

## Code chunking and ingestion

The JSONL loader accepts already chunked `CodeSnippet` records. The chunker can also ingest source files: Python uses the AST to produce functions, methods, and classes; other languages use declaration-aware patterns; unsupported files fall back to one module chunk. IDs are deterministic hashes of repository, path, version, line range, and source.

Required snippet fields are `id`, `repository`, `file_path`, `language`, `version`, `commit_id`, and `code`. The model also preserves function/class names, docstrings, imports, line range, snippet type, and tokens.

## Installation

Python 3.11+ is recommended.

```bash
cd code-retrieval-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The sample commands use the hashing semantic backend, so they run without a model download. The semantic encoder is configurable and hashing remains the default. To use CPU MiniLM:

```bash
python scripts/build_index.py --semantic-encoder minilm
```

The equivalent Python setting is `RetrievalConfig(semantic_encoder="minilm")`.
The cache path is authoritative via `--embedding-cache` / `embedding_cache`;
cache metadata records encoder, model, dimension, corpus hash, and snippet
count, so changing the encoder, model, or corpus forces regeneration.

## Build an index

```bash
python scripts/build_index.py
```

This writes a portable index under `outputs/index/`, including source metadata, the lexical index, and normalized semantic vectors. Build output reports index build time, embedding time, backend, and snippet count.

## Run retrieval

```bash
python scripts/retrieve.py \\
  --query "How is the input preprocessed before going to the main function?" \\
  --top-k 10

python scripts/retrieve.py \\
  --query "Where is user authentication handled?" \\
  --version v2 \\
  --top-k 10
```

Each result contains `rank`, `snippet_id`, final score, file/function/class, language, repository, version, source code, and `retrieval_signals` with semantic, lexical, fusion/RRF, reranking, and feature scores.

The Python API is also small:

```python
from src.config import RetrievalConfig
from src.ingestion.loader import load_jsonl
from src.pipeline import RetrievalEngine

engine = RetrievalEngine(load_jsonl("data/sample/snippets.jsonl"), RetrievalConfig(semantic_backend="hashing"))
engine.build_index()
results = engine.search("How is the input preprocessed before going to the main function?", top_k=10, version="all")
```

## Demo

```bash
python scripts/run_demo.py
```

The demo prints the top three snippets for preprocessing, authentication, and database retry queries, plus query latency. Results come from the real index; there are no hardcoded answer strings.

## Evaluation

The local qrels contain graded relevance judgments for six representative queries.

```bash
python scripts/evaluate.py
```

The report contains MRR, NDCG@10, Recall@5, Recall@10, mean latency, and p95 latency. `src/evaluation/metrics.py` is independent of the engine, making BM25-only, semantic-only, hybrid, filtered, and reranked ablations easy to add by varying `RetrievalConfig` and the enabled stages.

Run the requested five-way comparison with:

```bash
python scripts/evaluate.py --ablations
```

This writes `outputs/ablation_results.json` for BM25 only, semantic only,
BM25+semantic, hybrid+metadata filtering, and hybrid+filtering+reranking.

## MTEB / CoIR AppsRetrieval

`src/evaluation/mteb_evaluator.py` isolates the optional current MTEB runner. When a compatible MTEB release and the AppsRetrieval task are available:

```bash
pip install 'mteb>=1.30'
python scripts/evaluate.py --mteb
```

The script calls the installed MTEB API and writes the runner output to `outputs/appsretrieval_results.json`. If MTEB is not installed, it exits with a clear message rather than fabricating a non-MTEB result file.

## Tests

```bash
pytest
```

The tests cover semantic chunking, conservative preprocessing, intent classification, BM25, semantic retrieval, fusion, version filtering, reranking, and the complete pipeline.

## Performance and CPU design

- batch embeddings and normalized vectors
- disk-persisted semantic vectors
- weighted lexical search without a mandatory external service
- candidate fusion before expensive reranking
- metadata/version filters before reranking
- no GPU, API key, external LLM, or answer-generation call

The sample uses a deterministic hashing encoder by default so latency and outputs are reproducible without model downloads. MiniLM is the intended quality upgrade for real evaluation. For large corpora, optional FAISS/HNSW can replace in-memory semantic search behind the same interface.

## Version-aware retrieval and future evolution

`version="v1"` and `version="v2"` restrict candidates; `version="all"` searches across versions; `version="latest"` keeps the latest version per logical repository/path/function key. Every result preserves `commit_id` for a future change-aware layer that can diff, cluster, and retrieve code evolution.

## Future improvements: highest-impact accuracy work

1. Fine-tune or select an embedding model on CoIR AppsRetrieval positives and hard negatives.
2. Add language-aware AST/tree-sitter chunks and richer symbol/reference metadata.
3. Tune BM25 field weights, fusion weights, and reranker features on held-out qrels rather than relying on defaults.
4. Add hard-negative mining and a small cross-encoder reranker for only the filtered top 50 candidates.
5. Add version-diff features so a query can retrieve the relevant change or evolution path, not only one copy of a function.
