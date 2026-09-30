"""Language and source metadata helpers."""

from pathlib import Path
from typing import List


EXTENSION_LANGUAGE = {
    ".py": "python", ".java": "java", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".rs": "rust",
    ".rb": "ruby", ".cpp": "cpp", ".c": "c", ".h": "c", ".cs": "csharp",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".sql": "sql",
}


def infer_language(file_path: str, explicit: str = "") -> str:
    return explicit.lower() if explicit else EXTENSION_LANGUAGE.get(Path(file_path).suffix.lower(), "unknown")


def extract_imports(code: str, language: str) -> List[str]:
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if language == "python" and (stripped.startswith("import ") or stripped.startswith("from ")):
            lines.append(stripped)
        elif language in {"javascript", "typescript"} and (stripped.startswith("import ") or stripped.startswith("const ") and "require(" in stripped):
            lines.append(stripped)
        elif language == "java" and stripped.startswith("import "):
            lines.append(stripped)
    return lines
