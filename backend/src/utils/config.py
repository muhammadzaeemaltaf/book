import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from root .env file
# This finds the .env file in the project root (parent of backend/)
root_dir = Path(__file__).resolve().parent.parent.parent.parent
env_path = root_dir / ".env"
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")

    # LLM Provider Configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")  # groq, gemini, or openai

    # Groq Configuration
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Gemini Configuration
    # Support both GOOGLE_API_KEY and GEMINI_API_KEY for compatibility
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    gemini_base_url: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "embed-multilingual-v3.0")

    # Application Settings
    backend_host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Frontend URL for CORS
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Qdrant Settings
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "documents")

    # Embedding Settings
    embedding_size: int = 1024  # Default for multilingual model
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Authentication and Personalization Settings
    database_url: str = os.getenv("DATABASE_URL", "postgresql://localhost/physical_ai_auth")
    better_auth_secret: str = os.getenv("BETTER_AUTH_SECRET", "your-secret-key-here")
    better_auth_url: str = os.getenv("BETTER_AUTH_URL", "http://localhost:8000")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    refresh_token_expire_days: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # in seconds

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"  # This allows extra environment variables
    }

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings