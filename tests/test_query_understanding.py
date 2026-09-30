from src.config import RetrievalConfig
from src.query.analyzer import QueryAnalyzer
from src.query.preprocessor import preprocess_query


def test_ambiguous_normalization_is_low_confidence_general_search():
    analysis = QueryAnalyzer().analyze("normalization")
    assert analysis.primary_intent == "general_code_search"
    assert analysis.confidence < 0.5


def test_input_normalization_is_data_preprocessing_and_preserves_phrase():
    analysis = QueryAnalyzer().analyze("input normalization")
    processed = preprocess_query("input normalization", RetrievalConfig())
    assert analysis.primary_intent == "data_preprocessing"
    assert "input normalization" in analysis.technical_phrases
    assert "input" in processed["lexical_query"]
    assert "normalization" in processed["lexical_query"]
    assert "preprocess" in processed["alias_expansions"]


def test_jwt_expiration_validation_is_authentication():
    analysis = QueryAnalyzer().analyze("JWT expiration validation")
    assert analysis.primary_intent == "authentication"
    assert analysis.confidence >= 0.8


def test_parameterized_database_query_is_database():
    analysis = QueryAnalyzer().analyze("parameterized database query")
    assert analysis.primary_intent == "database"
    assert "parameterized database query" in analysis.technical_phrases


def test_duplicate_removal_by_identifier_is_algorithm():
    analysis = QueryAnalyzer().analyze("remove duplicate rows by external identifier")
    assert analysis.primary_intent == "algorithm"


def test_api_payload_normalization_prefers_api_over_preprocessing():
    analysis = QueryAnalyzer().analyze("API request payload normalization")
    assert analysis.primary_intent == "api"


def test_explicit_latest_version_is_detected():
    analysis = QueryAnalyzer().analyze("find the latest version of this helper")
    assert analysis.version_intent == "latest"
    assert analysis.version is None


def test_explicit_v2_is_detected():
    analysis = QueryAnalyzer().analyze("show version 2 of normalize_input")
    assert analysis.version_intent == "specific"
    assert analysis.version == "v2"


def test_no_version_request_has_no_version_intent():
    analysis = QueryAnalyzer().analyze("Where are browser form values trimmed and normalized?")
    assert analysis.version_intent == "none"
    assert analysis.version is None


def test_low_confidence_ambiguous_payload_does_not_force_api():
    analysis = QueryAnalyzer().analyze("payload")
    assert analysis.primary_intent == "general_code_search"
    assert analysis.confidence < 0.5


def test_identifiers_and_phrases_are_preserved():
    processed = preprocess_query("Validate JWTExpirationValidator for API_BASE_URL and mainFunction")
    assert "JWTExpirationValidator" in processed["identifiers"]
    assert "API_BASE_URL" in processed["identifiers"]
    assert "mainFunction" in processed["identifiers"]
    assert "jwtexpirationvalidator" in processed["tokens"]
