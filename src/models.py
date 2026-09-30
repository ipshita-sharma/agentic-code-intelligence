"""Typed data contracts shared by ingestion, retrieval, and evaluation."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CodeSnippet:
    id: str
    repository: str
    file_path: str
    language: str
    version: str
    commit_id: str
    function_name: str = ""
    class_name: str = ""
    code: str = ""
    docstring: str = ""
    imports: List[str] = field(default_factory=list)
    start_line: int = 1
    end_line: int = 1
    snippet_type: str = "generic"
    tokens: List[str] = field(default_factory=list)
    # Optional code-aware metadata. Empty values preserve backward
    # compatibility with previously serialized snippets and indexes.
    symbol_name: str = ""
    symbol_type: str = ""
    parent_symbol: str = ""
    decorators: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CodeSnippet":
        data = dict(value)
        data.setdefault("imports", [])
        data.setdefault("tokens", [])
        data.setdefault("function_name", "")
        data.setdefault("class_name", "")
        data.setdefault("docstring", "")
        data.setdefault("start_line", 1)
        data.setdefault("end_line", max(1, data.get("code", "").count("\n") + 1))
        data.setdefault("snippet_type", "generic")
        data.setdefault("symbol_name", "")
        data.setdefault("symbol_type", "")
        data.setdefault("parent_symbol", "")
        data.setdefault("decorators", [])
        data.setdefault("comments", [])
        return cls(**data)

    @property
    def display_name(self) -> str:
        return self.function_name or self.class_name or self.file_path


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    intent: str
    keywords: List[str]
    technical_terms: List[str]
    language_hints: List[str]
    identifiers: List[str] = field(default_factory=list)
    technical_phrases: List[str] = field(default_factory=list)
    confidence: float = 0.0
    intent_scores: Dict[str, float] = field(default_factory=dict)
    primary_intent: str = "general_code_search"
    version_intent: str = "none"
    version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalCandidate:
    snippet: CodeSnippet
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    lexical_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    rrf_score: float = 0.0
    fusion_score: float = 0.0
    rerank_features: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0


@dataclass
class RetrievalResult:
    rank: int
    snippet_id: str
    score: float
    file_path: str
    function_name: str
    class_name: str
    language: str
    version: str
    repository: str
    code: str
    retrieval_signals: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
