"""
Unified main application file for the Physical AI textbook platform backend.
This module sets up the FastAPI application with all API routes including both
chatbot functionality and authentication/personalization features.
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import asyncio

from src.api.auth import router as auth_router
from src.api.personalization import router as personalization_router
from src.api.ai_summary import router as ai_summary_router
from src.api.user import router as user_router
from src.api.routes import chat, ingest, search  # Chatbot routes
from src.api.middleware.cors import setup_cors_middleware  # Chatbot CORS middleware
from src.utils.database import init_db
from src.utils.logger import setup_logging_config
from src.utils.config import settings
from src.services.qdrant_service import qdrant_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    Handles both chatbot and auth/personalization initialization.
    """
    # Startup
    setup_logging_config()
    logging.info("Starting up unified Physical AI Textbook Platform API...")

    # Initialize Qdrant service for chatbot functionality
    try:
        await qdrant_service.initialize()
        logging.info("Qdrant service initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Qdrant service: {str(e)}")
        raise

    # Initialize database for auth/personalization functionality
    try:
        await init_db()
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.warning(f"Database initialization failed (this may be expected in development): {e}")
        # Continue startup even if database is not available (for development)

    yield  # Application runs here

    # Shutdown
    logging.info("Shutting down unified Physical AI Textbook Platform API...")


# Create FastAPI application with unified functionality
app = FastAPI(
    title="Physical AI Textbook Platform API - Unified",
    description="Backend API for the Physical AI textbook platform with authentication, personalization, AI summaries, and RAG chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# Set up CORS middleware (using chatbot's CORS setup)
setup_cors_middleware(app)

# Include authentication and personalization API routes
app.include_router(auth_router)
app.include_router(personalization_router)
app.include_router(ai_summary_router)
app.include_router(user_router)

# Include chatbot API routes
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(search.router, prefix="/search", tags=["search"])


# Health check endpoint that checks both systems
@app.get("/health")
async def health():
    """Health check endpoint for unified API."""
    qdrant_healthy = await qdrant_service.health_check()

    health_status = {
        "status": "healthy" if qdrant_healthy else "unhealthy",
        "timestamp": asyncio.get_event_loop().time(),
        "services": {
            "qdrant": "available" if qdrant_healthy else "unavailable",
            "cohere": "pending",  # We don't continuously check Cohere
            "gemini": "pending",   # We don't continuously check Gemini
            "database": "pending"  # Database health check would require additional implementation
        }
    }

    return health_status


# Root endpoint with unified API information
@app.get("/")
async def root():
    """
    Root endpoint with unified API information.
    """
    return {
        "message": "Welcome to the Physical AI Textbook Platform Unified API",
        "version": "1.0.0",
        "endpoints": [
            "/api/auth - Authentication endpoints",
            "/api/personalize - Content personalization endpoints",
            "/api/summary - AI summary endpoints",
            "/api/user - User profile endpoints",
            "/chat - RAG chatbot endpoints",
            "/ingest - Document ingestion endpoints",
            "/search - Search endpoints"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.unified_main:app",
        host=settings.backend_host if hasattr(settings, 'backend_host') else "0.0.0.0",
        port=settings.backend_port if hasattr(settings, 'backend_port') else 8000,
        reload=True  # Only for development
    )