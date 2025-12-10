# RAG Chatbot Backend

Backend service for the RAG Chatbot integrated with Docusaurus textbook site.

## Overview

This backend provides:
- Document ingestion and processing pipeline
- Vector storage and retrieval using Qdrant and Cohere embeddings
- Chat interface with normal Q&A and selected text modes
- Integration with Gemini 2.0 Flash via OpenAI Agents SDK

## Setup

1. Install dependencies:
   ```bash
   uv sync  # or pip install -r requirements.txt
   ```

2. Create environment file in the **project root** (not in backend/):
   ```bash
   cd ..  # Navigate to project root if you're in backend/
   cp .env.example .env
   ```

3. Update the `.env` file in the project root with your API keys and configuration.

4. Start the backend:
   ```bash
   cd backend  # Navigate back to backend/ directory
   uv run python -m src.api.main
   # or
   python -m src.api.main
   ```

> **Note:** The backend now uses a single `.env` file located in the project root directory instead of having separate `.env` files for backend and frontend.

## API Endpoints

- `POST /ingest` - Ingest documents into vector database
- `GET /ingest/status/{pipeline_id}` - Check ingestion status
- `POST /search` - Search documents (for debugging)
- `POST /chat` - Chat with the RAG system
- `GET /health` - Health check

## Architecture

The system follows a modular architecture with:
- Models in `src/models/`
- Services in `src/services/`
- API routes in `src/api/routes/`
- Agents (like RAGRetrievalAgent) in `src/agents/`
- Utilities in `src/utils/`