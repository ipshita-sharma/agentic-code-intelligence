from src.ingestion.chunker import chunk_source, deterministic_id


def test_python_chunker_extracts_semantic_functions_and_methods():
    code = "class Parser:\n    def parse(self, value):\n        return value.strip()\n\ndef load(path):\n    return open(path).read()\n"
    chunks = chunk_source(code, "parser.py", repository="r", version="v1")
    assert {chunk.function_name for chunk in chunks} >= {"parse", "load"}
    assert any(chunk.snippet_type == "method" and chunk.class_name == "Parser" for chunk in chunks)
    assert deterministic_id("r", "parser.py", "v1", 1, 2, "x") == deterministic_id("r", "parser.py", "v1", 1, 2, "x")


def test_unsupported_language_has_safe_fallback():
    chunks = chunk_source("raw content without declarations", "notes.xyz")
    assert len(chunks) == 1
    assert chunks[0].snippet_type == "module"
