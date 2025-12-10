"""
Ingest API Routes

This module defines the API endpoints for document ingestion functionality.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import asyncio
import logging

from ...models.document import IngestionRequest, IngestionResponse
from ...services.ingestion_service import IngestionService
from ...utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])
ingestion_service = IngestionService()

# In-memory storage for tracking ingestion jobs (in production, use a proper database)
ingestion_jobs: Dict[str, Dict[str, Any]] = {}

@router.post("/", response_model=IngestionResponse)
async def ingest_documents(request: IngestionRequest) -> IngestionResponse:
    """
    Ingest documents from a source path into the vector database.

    This endpoint processes documents from the specified source path,
    chunks them, generates embeddings, and stores them in the vector database.
    """
    try:
        logger.info(f"Received ingestion request for: {request.source_path}")

        # Call the ingestion service to process the documents
        response = await ingestion_service.ingest_documents(request)

        logger.info(f"Ingestion completed with status: {response.status}")
        return response

    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/async", response_model=Dict[str, str])
async def ingest_documents_async(background_tasks: BackgroundTasks, request: IngestionRequest):
    """
    Asynchronously ingest documents from a source path.

    This endpoint starts an ingestion job in the background and returns immediately
    with a job ID that can be used to check the status.
    """
    try:
        logger.info(f"Received async ingestion request for: {request.source_path}")

        # Generate a unique job ID
        import uuid
        job_id = f"ingest_job_{uuid.uuid4().hex[:8]}"

        # Store job info
        ingestion_jobs[job_id] = {
            "status": "processing",
            "request": request.dict(),
            "start_time": asyncio.get_event_loop().time(),
            "progress": 0,
            "processed_chunks": 0,
            "total_chunks": 0,
            "error": None
        }

        # Add background task to process the ingestion
        background_tasks.add_task(_process_ingestion_job, job_id, request)

        return {"job_id": job_id, "message": "Ingestion job started"}

    except Exception as e:
        logger.error(f"Error starting async ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start ingestion job: {str(e)}")


async def _process_ingestion_job(job_id: str, request: IngestionRequest):
    """
    Background task to process an ingestion job.
    """
    try:
        # Update job status
        if job_id in ingestion_jobs:
            ingestion_jobs[job_id]["status"] = "processing"

        # Perform the ingestion
        response = await ingestion_service.ingest_documents(request)

        # Update job status
        if job_id in ingestion_jobs:
            ingestion_jobs[job_id].update({
                "status": response.status,
                "processed_chunks": response.processed_count,
                "message": response.message,
                "duration_seconds": response.duration_seconds,
                "end_time": asyncio.get_event_loop().time()
            })

        logger.info(f"Async ingestion job {job_id} completed with status: {response.status}")

    except Exception as e:
        logger.error(f"Error in async ingestion job {job_id}: {str(e)}")
        if job_id in ingestion_jobs:
            ingestion_jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "end_time": asyncio.get_event_loop().time()
            })


@router.get("/status/{job_id}", response_model=Dict[str, Any])
async def get_ingestion_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of an ingestion job.
    """
    try:
        if job_id not in ingestion_jobs:
            raise HTTPException(status_code=404, detail="Ingestion job not found")

        job_info = ingestion_jobs[job_id].copy()

        # Calculate progress if possible
        if "total_chunks" in job_info and "processed_chunks" in job_info:
            total = job_info["total_chunks"]
            processed = job_info["processed_chunks"]
            job_info["progress_percentage"] = (processed / total * 100) if total > 0 else 0

        return {
            "job_id": job_id,
            **job_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ingestion status for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.delete("/clear", response_model=Dict[str, bool])
async def clear_ingested_content() -> Dict[str, bool]:
    """
    Clear all ingested content from the vector database.
    """
    try:
        logger.info("Received request to clear all ingested content")

        success = await ingestion_service.clear_ingested_content()

        if success:
            logger.info("Successfully cleared all ingested content")
            return {"success": True}
        else:
            logger.error("Failed to clear ingested content")
            raise HTTPException(status_code=500, detail="Failed to clear content")

    except Exception as e:
        logger.error(f"Error clearing ingested content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear content: {str(e)}")


# Task T048: Create ingestion status endpoint (already implemented above as /status/{job_id})
# The get_ingestion_status function above handles this requirement