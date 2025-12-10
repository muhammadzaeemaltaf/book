from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from ..utils.config import settings
from ..utils.logging import logger
from ..services.qdrant_service import qdrant_service
from .routes import chat, ingest, search
from .middleware.cors import setup_cors_middleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting up RAG Chatbot API...")
    try:
        # Initialize Qdrant service
        await qdrant_service.initialize()
        logger.info("Qdrant service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant service: {str(e)}")
        raise

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down RAG Chatbot API...")

# Create FastAPI app instance
app = FastAPI(
    title="RAG Chatbot API",
    description="API for RAG Chatbot integrated with Docusaurus textbook site",
    version="0.1.0",
    lifespan=lifespan
)

# Set up CORS middleware
setup_cors_middleware(app)

# Include API routes
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(search.router, prefix="/search", tags=["search"])

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "RAG Chatbot API is running", "version": "0.1.0"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    qdrant_healthy = await qdrant_service.health_check()

    health_status = {
        "status": "healthy" if qdrant_healthy else "unhealthy",
        "timestamp": asyncio.get_event_loop().time(),
        "services": {
            "qdrant": "available" if qdrant_healthy else "unavailable",
            "cohere": "pending",  # We don't continuously check Cohere
            "gemini": "pending"   # We don't continuously check Gemini
        }
    }

    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True
    )