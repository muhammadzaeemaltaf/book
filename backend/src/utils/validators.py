from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import re


class ChatQueryValidator(BaseModel):
    """Validator for chat query requests."""

    message: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field(default="normal_qa", pattern=r"^(normal_qa|selected_text)$")
    selected_text: Optional[str] = Field(None, max_length=5000)
    stream: bool = Field(default=True)
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator('selected_text')
    @classmethod
    def validate_selected_text(cls, v, info):
        """Validate selected text is provided when mode is selected_text."""
        # Access the mode from the validation info
        if info.data.get('mode') == 'selected_text' and not v:
            raise ValueError('selected_text is required when mode is selected_text')
        return v


class IngestRequestValidator(BaseModel):
    """Validator for document ingestion requests."""

    source_path: str = Field(..., min_length=1)
    chunk_size: int = Field(default=512, ge=50, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)
    recursive: bool = Field(default=True)


class SearchRequestValidator(BaseModel):
    """Validator for search requests."""

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentChunkValidator(BaseModel):
    """Validator for document chunk models."""

    id: str
    content: str = Field(..., min_length=1)
    source_document: str = Field(..., min_length=1)
    chunk_index: int = Field(ge=0)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


def validate_chat_mode(mode: str) -> bool:
    """Validate that the chat mode is one of the allowed values."""
    allowed_modes = {"normal_qa", "selected_text"}
    return mode in allowed_modes


def validate_text_length(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
    """Validate text length is within specified bounds."""
    return min_length <= len(text) <= max_length


def validate_top_k(top_k: int) -> bool:
    """Validate top_k parameter is within reasonable bounds."""
    return 1 <= top_k <= 50


def validate_api_key(api_key: str) -> bool:
    """Basic validation for API keys."""
    # Check if API key is not empty and has reasonable length
    return bool(api_key and len(api_key.strip()) > 10)


def validate_url(url: str) -> bool:
    """Basic URL validation."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None


def sanitize_input(text: str) -> str:
    """Basic input sanitization to prevent injection attacks."""
    # Remove potentially dangerous characters/sequences
    sanitized = text.replace('\0', '')  # Remove null bytes
    sanitized = sanitized.replace('\x00', '')  # Remove null bytes (hex)
    # Add more sanitization as needed
    return sanitized


def validate_embedding_dimensions(embedding: List[float], expected_size: int) -> bool:
    """Validate that an embedding has the expected dimensions."""
    return len(embedding) == expected_size


def validate_document_content(content: str) -> List[str]:
    """Validate document content and return a list of validation errors."""
    errors = []

    if not content or len(content.strip()) == 0:
        errors.append("Content cannot be empty")

    if len(content) > 100000:  # 100KB limit
        errors.append("Content exceeds maximum size of 100KB")

    # Add more validation rules as needed
    return errors