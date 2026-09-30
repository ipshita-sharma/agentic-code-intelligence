from src.query.analyzer import QueryAnalyzer
from src.query.preprocessor import preprocess_query


def test_query_preserves_programming_terms_and_identifiers():
    processed = preprocess_query("How is input preprocessed before main_function?",)
    assert "preprocessed" in processed["keywords"]
    assert "main_function" in processed["tokens"]
    assert "function" in processed["keywords"]


def test_query_classification():
    analyzer = QueryAnalyzer()
    assert analyzer.analyze("Where is JWT token validation performed?").intent == "authentication"
    assert analyzer.analyze("Which function retries database connections?").intent in {"database", "error_handling"}
