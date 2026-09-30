from src.ingestion.chunker import chunk_source, enrich_snippet
from src.models import CodeSnippet


def test_python_structural_metadata_extracts_symbols_imports_decorators_and_lines():
    code = (
        "import functools\n\n"
        "# parser class comment\n"
        "class Parser:\n"
        "    \"\"\"Parse incoming values.\"\"\"\n\n"
        "    @classmethod\n"
        "    def parse(cls, value):\n"
        "        # normalize before parsing\n"
        "        return value.strip()\n\n"
        "@functools.lru_cache()\n"
        "def load(path):\n"
        "    \"\"\"Load a file.\"\"\"\n"
        "    return open(path).read()\n"
    )
    chunks = chunk_source(code, "parser.py", repository="repo", version="v2", commit_id="c2")
    parser = next(chunk for chunk in chunks if chunk.symbol_type == "class")
    method = next(chunk for chunk in chunks if chunk.function_name == "parse")
    function = next(chunk for chunk in chunks if chunk.function_name == "load")

    assert parser.symbol_name == "Parser"
    assert parser.symbol_type == "class"
    assert parser.docstring == "Parse incoming values."
    assert "import functools" in parser.imports
    assert parser.start_line == 4 and parser.end_line == 10

    assert method.symbol_name == "Parser.parse"
    assert method.symbol_type == "method"
    assert method.parent_symbol == "Parser"
    assert "@classmethod" in method.decorators
    assert "# normalize before parsing" in method.comments
    assert method.start_line == 7 and method.end_line == 10

    assert function.symbol_name == "load"
    assert function.symbol_type == "function"
    assert "@functools.lru_cache()" in function.decorators
    assert function.docstring == "Load a file."
    assert function.start_line == 12 and function.end_line == 15


def test_code_aware_metadata_preserves_original_code_and_serializes():
    code = "def normalize(value):\n    return value.strip()\n"
    original = chunk_source(code, "normalize.py", repository="repo", version="v1")[0]
    enriched = enrich_snippet(original)
    restored = type(original).from_dict(enriched.to_dict())
    assert enriched.code == original.code
    assert enriched.id == original.id
    assert enriched.start_line == original.start_line
    assert enriched.end_line == original.end_line
    assert restored.symbol_name == "normalize"
    assert restored.symbol_type == "function"
    assert restored.code == original.code


def test_unstructured_snippet_gets_no_fabricated_structure():
    snippet = CodeSnippet(
        id="raw",
        repository="repo",
        file_path="notes.xyz",
        language="unknown",
        version="v1",
        commit_id="c1",
        code="raw content without a recognizable declaration",
        snippet_type="generic",
    )
    enriched = enrich_snippet(snippet)
    assert enriched.code == snippet.code
    assert enriched.symbol_name == ""
    assert enriched.symbol_type == ""
    assert enriched.parent_symbol == ""
    assert enriched.decorators == []
    assert enriched.comments == []


def test_old_serialized_snippet_loads_with_empty_optional_metadata():
    snippet = CodeSnippet.from_dict({
        "id": "old",
        "repository": "repo",
        "file_path": "old.py",
        "language": "python",
        "version": "v1",
        "commit_id": "c1",
        "code": "text",
    })
    assert snippet.symbol_name == ""
    assert snippet.decorators == []
    assert snippet.comments == []


def test_generic_decorator_is_kept_with_declaration():
    chunks = chunk_source(
        "@route('/orders')\n"
        "export function listOrders() {\n"
        "  return [];\n"
        "}\n",
        "orders.js",
        repository="repo",
        version="v1",
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.start_line == 1
    assert chunk.end_line == 4
    assert chunk.symbol_name == "listOrders"
    assert chunk.symbol_type == "function"
    assert chunk.decorators == ["@route('/orders')"]
    assert chunk.code.startswith("@route('/orders')")
