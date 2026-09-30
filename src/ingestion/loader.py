"""JSONL loading and deterministic persistence helpers."""

import json
from pathlib import Path
from typing import Iterable, List

from ..models import CodeSnippet


def load_jsonl(path: str | Path) -> List[CodeSnippet]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Snippet JSONL not found: {path}")
    snippets: List[CodeSnippet] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                snippets.append(CodeSnippet.from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return snippets


def save_jsonl(snippets: Iterable[CodeSnippet], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for snippet in snippets:
            handle.write(json.dumps(snippet.to_dict(), sort_keys=True) + "\n")
