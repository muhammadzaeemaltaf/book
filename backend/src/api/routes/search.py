"""
Search API Routes

This module defines the API endpoints for document search functionality.
"""
from fastapi import APIRouter, HTTPException
import logging

from ...models.document import SearchRequest, SearchResponse
from ...services.retrieval_service import RetrievalService
from ...utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])
retrieval_service = RetrievalService()

@router.post("/", response_model=SearchResponse)
async def search_documents(request: SearchRequest) -> SearchResponse:
    """
    Search for documents in the vector database based on a query.

    This endpoint performs semantic search on the ingested documents
    and returns the most relevant results.
    """
    try:
        logger.info(f"Received search request for query: {request.query[:50]}...")

        # Perform the search using the retrieval service
        results = await retrieval_service.search_documents(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )

        logger.info(f"Search completed with {len(results)} results")

        return SearchResponse(
            results=results,
            query=request.query,
            search_time_ms=0  # Actual timing would be measured in the service
        )

    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")