"""Semantic source chunking with a safe text fallback."""

import ast
import hashlib
import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List

from .metadata import extract_imports, infer_language
from ..models import CodeSnippet
from ..query.preprocessor import tokenize


def deterministic_id(repository: str, file_path: str, version: str, start_line: int, end_line: int, code: str) -> str:
    raw = "|".join([repository, file_path, version, str(start_line), str(end_line), code])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _python_chunks(code: str) -> List[tuple[str, str, str, int, int, str]]:
    """Return (type, function, class, start, end, source) tuples."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines = code.splitlines()
    chunks: List[tuple[str, str, str, int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = getattr(node, "lineno", 1)
        decorator_lines = [getattr(item, "lineno", start) for item in getattr(node, "decorator_list", [])]
        if decorator_lines:
            start = min([start, *decorator_lines])
        end = getattr(node, "end_lineno", start)
        source = "\n".join(lines[start - 1:end])
        if isinstance(node, ast.ClassDef):
            chunks.append(("class", "", node.name, start, end, source))
        else:
            parent_class = ""
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef) and node in getattr(parent, "body", []):
                    parent_class = parent.name
                    break
            chunks.append(("method" if parent_class else "function", node.name, parent_class, start, end, source))
    return sorted(chunks, key=lambda item: (item[3], item[4], item[0]))


_DECLARATION_PATTERNS = [
    (re.compile(r"\b(?:public|private|protected|static|final|async|export|def|function)?\s*(?:class|interface|enum)\s+([A-Za-z_$][\w$]*)"), "class"),
    (re.compile(r"\b(?:def|function)\s+([A-Za-z_$][\w$]*)\s*\("), "function"),
    (re.compile(r"\b([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"), "function"),
]


def _generic_chunks(code: str) -> List[tuple[str, str, str, int, int, str]]:
    lines = code.splitlines() or [code]
    matches: List[tuple[str, str, str, int]] = []
    for index, line in enumerate(lines, 1):
        for pattern, kind in _DECLARATION_PATTERNS:
            match = pattern.search(line)
            if match:
                start = index
                # Keep immediately preceding annotations/decorators with the
                # declaration while leaving arbitrary comments and whitespace
                # outside the chunk. This is deliberately conservative for
                # languages without a full parser in this ingestion path.
                while start > 1 and lines[start - 2].strip().startswith("@"):
                    start -= 1
                matches.append((kind, match.group(1), "", start))
                break
    chunks: List[tuple[str, str, str, int, int, str]] = []
    if not matches:
        return [("module", "", "", 1, len(lines), code)]
    for pos, (kind, name, class_name, start) in enumerate(matches):
        end = matches[pos + 1][3] - 1 if pos + 1 < len(matches) else len(lines)
        chunks.append((kind, name, class_name, start, max(start, end), "\n".join(lines[start - 1:end])))
    return chunks


def _extract_decorators(code: str, language: str) -> List[str]:
    """Extract annotation/decorator lines without trying to parse the language."""
    if language not in {"python", "javascript", "typescript", "java", "csharp"}:
        return []
    return [
        line.strip()
        for line in code.splitlines()
        if line.strip().startswith("@") and len(line.strip()) > 1
    ]


def _extract_comments(code: str, language: str) -> List[str]:
    """Preserve explicit comment lines; docstrings remain in ``docstring``."""
    markers = {"python": ("#",), "javascript": ("//", "/*", "*", "*/"), "typescript": ("//", "/*", "*", "*/"), "java": ("//", "/*", "*", "*/"), "cpp": ("//", "/*", "*", "*/"), "c": ("//", "/*", "*", "*/"), "csharp": ("//", "/*", "*", "*/")}
    prefixes = markers.get(language, ("#", "//"))
    comments = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefixes):
            comments.append(stripped)
    return comments


def _python_docstring(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return ast.get_docstring(node) or ""
    return ast.get_docstring(tree) or ""


def extract_structural_metadata(
    code: str,
    language: str,
    snippet_type: str = "",
    function_name: str = "",
    class_name: str = "",
    docstring: str = "",
    imports: List[str] | None = None,
) -> dict:
    """Extract only metadata that can be supported by the source text."""
    symbol_type = snippet_type if snippet_type in {"function", "method", "class", "module"} else ""
    parent_symbol = class_name if symbol_type == "method" and class_name else ""
    if language == "python" and not docstring:
        docstring = _python_docstring(code)
    decorators = _extract_decorators(code, language)
    comments = _extract_comments(code, language)
    resolved_imports = list(imports or extract_imports(code, language))
    if function_name and parent_symbol:
        symbol_name = f"{parent_symbol}.{function_name}"
    elif function_name:
        symbol_name = function_name
    elif class_name:
        symbol_name = class_name
    else:
        symbol_name = ""
    return {
        "symbol_name": symbol_name,
        "symbol_type": symbol_type,
        "parent_symbol": parent_symbol,
        "decorators": decorators,
        "comments": comments,
        "docstring": docstring,
        "imports": resolved_imports,
    }


def enrich_snippet(snippet: CodeSnippet) -> CodeSnippet:
    """Enrich an existing chunk without changing its ID, source, or boundaries."""
    metadata = extract_structural_metadata(
        snippet.code,
        snippet.language,
        snippet.snippet_type,
        snippet.function_name,
        snippet.class_name,
        snippet.docstring,
        snippet.imports,
    )
    return replace(snippet, **metadata)


def chunk_source(code: str, file_path: str, repository: str = "sample-repo", version: str = "v1", commit_id: str = "local") -> List[CodeSnippet]:
    language = infer_language(file_path)
    chunks = _python_chunks(code) if language == "python" else _generic_chunks(code)
    if not chunks:
        chunks = [("module", "", "", 1, max(1, code.count("\n") + 1), code)]
    imports = extract_imports(code, language)
    results = []
    for snippet_type, function_name, class_name, start, end, source in chunks:
        docstring = ""
        if language == "python":
            try:
                node = ast.parse(source).body[0]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node) or ""
            except SyntaxError:
                pass
        structural = extract_structural_metadata(
            source, language, snippet_type, function_name, class_name, docstring, imports
        )
        snippet_id = deterministic_id(repository, file_path, version, start, end, source)
        results.append(CodeSnippet(
            id=snippet_id, repository=repository, file_path=file_path, language=language,
            version=version, commit_id=commit_id, function_name=function_name,
            class_name=class_name, code=source, docstring=structural["docstring"], imports=structural["imports"],
            start_line=start, end_line=end, snippet_type=snippet_type,
            tokens=tokenize(source), **{key: structural[key] for key in ("symbol_name", "symbol_type", "parent_symbol", "decorators", "comments")},
        ))
    return results


def chunk_file(path: str | Path, **metadata: str) -> List[CodeSnippet]:
    path = Path(path)
    return chunk_source(path.read_text(encoding="utf-8"), str(path), **metadata)
