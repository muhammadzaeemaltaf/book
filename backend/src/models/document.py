from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentChunk(BaseModel):
    """Model representing a segment of the book content that has been processed and embedded for vector storage."""

    chunk_id: str
    content: str = Field(..., min_length=1)
    source_document: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        # Allow extra fields for flexibility with metadata
        extra = "allow"


class VectorEmbedding(BaseModel):
    """Model representing the numerical representation of text content stored in the vector database."""

    vector_id: str
    vector: List[float]  # The actual embedding vector
    document_id: str  # Reference to the source document chunk
    model_version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IngestionRequest(BaseModel):
    """Model for ingestion requests."""

    source_path: str = Field(..., min_length=1)
    chunk_size: int = Field(default=512, ge=50, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)
    recursive: bool = Field(default=True)


class IngestionResponse(BaseModel):
    """Model for ingestion responses."""

    status: str
    processed_count: int
    message: str
    pipeline_id: Optional[str] = None
    duration_seconds: Optional[float] = None


class SearchRequest(BaseModel):
    """Model for search requests."""

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Model for individual search results."""

    id: str
    content: str
    score: float
    source_document: str
    metadata: Dict[str, Any]
    chunk_index: Optional[int] = None


class SearchResponse(BaseModel):
    """Model for search responses."""

    results: List[SearchResult]
    query: str
    search_time_ms: float


class IngestionPipeline(BaseModel):
    """Model representing the ingestion pipeline process."""

    id: str
    source_path: str
    status: str  # pending, processing, completed, failed
    total_documents: int
    processed_documents: int
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None

    @property
    def progress_percentage(self) -> float:
        """Calculate the progress percentage."""
        if self.total_documents == 0:
            return 0.0
        return round((self.processed_documents / self.total_documents) * 100, 2)


class DocumentProcessingConfig(BaseModel):
    """Configuration for document processing."""

    chunk_size: int = Field(default=512, ge=50, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)
    max_chunk_size: int = Field(default=1000, ge=100, le=5000)
    encoding: str = Field(default="utf-8")
    recursive: bool = Field(default=True)
    file_extensions: List[str] = Field(default=[".md", ".txt", ".pdf"])