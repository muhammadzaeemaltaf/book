from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ChatMode(str, Enum):
    """Enumeration of chat modes."""
    NORMAL_QA = "normal_qa"
    SELECTED_TEXT = "selected_text"
    VECTOR_SEARCH = "vector_search"


class ChatQuery(BaseModel):
    """Model representing a user's chat query."""
    id: Optional[str] = None
    text: str = Field(..., min_length=1, max_length=2000)
    mode: ChatMode = ChatMode.NORMAL_QA
    selected_text: Optional[str] = Field(None, max_length=5000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_context: Optional[Dict[str, Any]] = None

    class Config:
        use_enum_values = True  # Serialize enum values as strings


class ChatResponse(BaseModel):
    """Model representing the system's response to a chat query."""
    id: str
    query_id: str
    content: str
    mode_used: ChatMode
    retrieved_context: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    streaming_data: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, int]] = None  # Token usage info


class ChatRequest(BaseModel):
    """Model for incoming chat requests."""
    message: str = Field(..., min_length=1, max_length=2000)
    mode: ChatMode = ChatMode.NORMAL_QA
    selected_text: Optional[str] = Field(None, max_length=5000)
    stream: bool = True
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    class Config:
        use_enum_values = True  # Serialize enum values as strings


class ChatResponseModel(BaseModel):
    """Model for chat API responses."""
    id: str
    message: str
    mode_used: ChatMode
    retrieved_context: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    usage: Optional[Dict[str, int]] = None

    class Config:
        use_enum_values = True  # Serialize enum values as strings


class StreamingChatChunk(BaseModel):
    """Model for streaming chat response chunks."""
    type: str  # "start", "chunk", "end"
    content: Optional[str] = None
    message_id: str
    final_response: Optional[ChatResponseModel] = None


class IngestionStatus(str, Enum):
    """Enumeration of ingestion pipeline statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionPipeline(BaseModel):
    """Model representing the ingestion pipeline."""
    id: str
    source_path: str
    status: IngestionStatus
    total_documents: int = 0
    processed_documents: int = 0
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

    @property
    def progress_percentage(self) -> float:
        """Calculate the progress percentage of the ingestion."""
        if self.total_documents == 0:
            return 0.0
        return min(100.0, (self.processed_documents / self.total_documents) * 100)