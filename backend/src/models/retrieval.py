from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class RetrievalMode(str, Enum):
    """Enumeration of retrieval modes."""
    VECTOR_SEARCH = "vector_search"
    SELECTED_TEXT_ONLY = "selected_text_only"
    HYBRID_SEARCH = "hybrid_search"
    KEYWORD_SEARCH = "keyword_search"


class QueryNormalizationResult(BaseModel):
    """Model for the result of query normalization."""
    original_query: str
    normalized_query: str
    detected_language: Optional[str] = None
    query_type: Optional[str] = None  # factual, conceptual, procedural, etc.
    entities: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class RetrievalPlan(BaseModel):
    """Model for a structured retrieval plan."""
    query_id: str
    mode: RetrievalMode
    normalization_result: QueryNormalizationResult
    search_strategies: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5
    min_score: float = 0.3
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievalResult(BaseModel):
    """Model for individual retrieval results."""
    id: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    source_document: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_index: Optional[int] = None
    reranked_score: Optional[float] = None  # Score after reranking


class RetrievalResponse(BaseModel):
    """Model for retrieval service responses."""
    query_id: str
    results: List[RetrievalResult]
    mode_used: RetrievalMode
    retrieved_count: int
    processing_time_ms: float
    query_expansion: Optional[List[str]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class RerankingRequest(BaseModel):
    """Model for reranking requests."""
    query: str
    results: List[RetrievalResult]
    top_k: int = 5


class RerankingResponse(BaseModel):
    """Model for reranking responses."""
    results: List[RetrievalResult]
    processing_time_ms: float


class ContextFilter(BaseModel):
    """Model for context filtering parameters."""
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    max_length: int = Field(default=2000, ge=100)
    deduplicate: bool = Field(default=True)
    filter_by_source: Optional[List[str]] = Field(default_factory=list)
    filter_by_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RetrievalMetrics(BaseModel):
    """Model for retrieval metrics and statistics."""
    query_id: str
    retrieval_time_ms: float
    vector_search_time_ms: float = 0.0
    reranking_time_ms: float = 0.0
    total_results: int
    filtered_results: int
    mode: RetrievalMode
    cache_hit: bool = False
    cache_time_saved_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryExpansionResult(BaseModel):
    """Model for query expansion results."""
    original_query: str
    expanded_queries: List[str]
    expansion_method: str  # keyword_extraction, synonym_expansion, etc.
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class SimilarityThreshold(BaseModel):
    """Model for similarity threshold configuration."""
    min_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    high_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    medium_threshold: float = Field(default=0.5, ge=0.0, le=1.0)