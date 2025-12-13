"""
AI Summary API endpoints for the Physical AI textbook platform.
This module provides API endpoints for AI-generated chapter summaries.
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from middleware.auth import require_auth
from middleware.rate_limit import ai_summary_rate_limit
from services.ai_summary_service import get_or_generate_summary
from services.validation import validate_chapter_id
from utils.error_handler import ValidationError, handle_api_exception


router = APIRouter(prefix="/api/summary", tags=["ai_summary"])


class SummaryResponse(BaseModel):
    """
    Response model for AI summaries.
    """
    success: bool
    chapter_id: str
    summary: str
    cached: bool
    message: str


@router.get("/{chapter_id}", response_model=SummaryResponse)
async def get_chapter_summary(
    chapter_id: str,
    current_user: Dict[str, Any] = Depends(require_auth),
    rate_limited: bool = Depends(ai_summary_rate_limit())
):
    """
    Get an AI-generated summary for a chapter.

    Args:
        chapter_id: The ID of the chapter to summarize
        current_user: The currently authenticated user

    Returns:
        SummaryResponse with the chapter summary

    Raises:
        HTTPException: If summary generation fails due to validation or other errors
    """
    try:
        # Validate chapter ID
        validate_chapter_id(chapter_id)

        # Get or generate the summary
        summary, cached = await get_or_generate_summary(
            chapter_id=chapter_id,
            user_id=current_user["user_id"]
        )

        if summary:
            return SummaryResponse(
                success=True,
                chapter_id=chapter_id,
                summary=summary,
                cached=cached,
                message="Summary retrieved successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate summary"
            )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the error
        from utils.error_handler import log_api_error
        log_api_error(f"Summary generation failed for user {current_user.get('user_id')} and chapter {chapter_id}", e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during summary generation"
        )


class GenerateSummaryRequest(BaseModel):
    """
    Request model for manual summary generation.
    """
    chapter_id: str
    force_new: bool = False  # Whether to bypass cache and generate new summary


@router.post("/{chapter_id}", response_model=SummaryResponse)
async def generate_chapter_summary(
    chapter_id: str,
    request: GenerateSummaryRequest,
    current_user: Dict[str, Any] = Depends(require_auth),
    rate_limited: bool = Depends(ai_summary_rate_limit())
):
    """
    Generate an AI summary for a chapter (optionally bypassing cache).

    Args:
        chapter_id: The ID of the chapter to summarize
        request: Request with options for generation
        current_user: The currently authenticated user

    Returns:
        SummaryResponse with the chapter summary
    """
    try:
        # Validate chapter ID
        validate_chapter_id(chapter_id)

        # Verify that the chapter_id in the path matches the one in the request
        if chapter_id != request.chapter_id:
            raise ValidationError(
                message="Chapter ID in path does not match chapter ID in request body",
                field="chapter_id"
            )

        # Get or generate the summary, potentially forcing new generation
        summary, cached = await get_or_generate_summary(
            chapter_id=chapter_id,
            user_id=current_user["user_id"],
            force_new=request.force_new
        )

        if summary:
            message = "Summary retrieved successfully" if cached and not request.force_new else "New summary generated successfully"
            return SummaryResponse(
                success=True,
                chapter_id=chapter_id,
                summary=summary,
                cached=not request.force_new and cached,  # If force_new, then it's not cached even if there was a cache
                message=message
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate summary"
            )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message
        )
    except Exception as e:
        # Log the error
        from utils.error_handler import log_api_error
        log_api_error(f"Summary generation failed for user {current_user.get('user_id')} and chapter {chapter_id}", e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during summary generation"
        )


# Endpoint to get summary metadata
@router.get("/{chapter_id}/metadata")
async def get_summary_metadata(
    chapter_id: str,
    current_user: Dict[str, Any] = Depends(require_auth),
    rate_limited: bool = Depends(ai_summary_rate_limit())
):
    """
    Get metadata about a chapter's summary without generating it.

    Args:
        chapter_id: The ID of the chapter
        current_user: The currently authenticated user

    Returns:
        Summary metadata
    """
    try:
        # Validate chapter ID
        validate_chapter_id(chapter_id)

        # Import here to avoid circular imports
        from services.ai_summary_service import get_summary_metadata
        metadata = await get_summary_metadata(chapter_id)

        return {
            "chapter_id": chapter_id,
            "has_summary": metadata is not None,
            "metadata": metadata or {}
        }

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message
        )
    except Exception as e:
        # Log the error
        from utils.error_handler import log_api_error
        log_api_error(f"Failed to get summary metadata for user {current_user.get('user_id')} and chapter {chapter_id}", e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving summary metadata"
        )