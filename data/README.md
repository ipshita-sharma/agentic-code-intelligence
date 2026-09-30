# Data format

`sample/snippets.jsonl` contains one `CodeSnippet` object per line. The
required identity and version fields are `id`, `repository`, `file_path`,
`language`, `version`, and `commit_id`; metadata and source fields are carried
through to every result.

`sample/queries.jsonl` stores local qrels as `{query_id, query, relevance}`.
Relevance values are graded integers, which makes the file suitable for
MRR, NDCG@10, and Recall@K experiments.
