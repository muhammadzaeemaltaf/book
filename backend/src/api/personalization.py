"""
Personalization API endpoints for the Physical AI textbook platform.
This module provides API endpoints for content personalization based on user profiles.
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from middleware.auth import require_auth
from services.personalization_service import personalize_content_for_user
from services.profile_service import is_profile_complete
from services.validation import validate_chapter_id
from utils.error_handler import ValidationError, handle_api_exception


router = APIRouter(prefix="/api/personalize", tags=["personalization"])


class PersonalizeRequest(BaseModel):
    """
    Request model for content personalization.
    """
    chapter_id: str
    content: str
    target_level: str = "adaptive"  # Options: beginner, intermediate, advanced, adaptive


class PersonalizeResponse(BaseModel):
    """
    Response model for content personalization.
    """
    success: bool
    chapter_id: str
    personalized_content: str
    message: str


@router.post("/{chapter_id}", response_model=PersonalizeResponse)
async def personalize_chapter(
    chapter_id: str,
    request: PersonalizeRequest,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Generate personalized content for a chapter based on the user's profile.

    Args:
        chapter_id: The ID of the chapter to personalize
        request: Personalization request with content and options
        current_user: The currently authenticated user

    Returns:
        PersonalizeResponse with personalized content

    Raises:
        HTTPException: If personalization fails due to validation or other errors
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

        # Check if user profile is complete enough for personalization
        profile_complete = await is_profile_complete(current_user["user_id"])
        if not profile_complete:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User profile is not complete enough for personalization. Please update your profile."
            )

        # Get user profile for personalization
        from services.profile_service import get_profile_summary
        user_profile = await get_profile_summary(current_user["user_id"])

        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )

        # Perform personalization
        personalized_content = await personalize_content_for_user(
            chapter_id=chapter_id,
            original_content=request.content,
            user_profile=user_profile,
            target_level=request.target_level
        )

        return PersonalizeResponse(
            success=True,
            chapter_id=chapter_id,
            personalized_content=personalized_content,
            message="Content personalized successfully"
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
        log_api_error(f"Personalization failed for user {current_user.get('user_id')} and chapter {chapter_id}", e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during content personalization"
        )


class GetPersonalizedContentRequest(BaseModel):
    """
    Request model for getting previously personalized content.
    """
    chapter_id: str


@router.post("/get", response_model=PersonalizeResponse)
async def get_personalized_content(
    request: GetPersonalizedContentRequest,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Get previously personalized content for a chapter if it exists.

    Args:
        request: Request with chapter ID
        current_user: The currently authenticated user

    Returns:
        PersonalizeResponse with personalized content if available
    """
    try:
        # Validate chapter ID
        validate_chapter_id(request.chapter_id)

        # Try to get previously personalized content
        from services.personalization_service import get_cached_personalization
        cached_content = await get_cached_personalization(
            user_id=current_user["user_id"],
            chapter_id=request.chapter_id
        )

        if cached_content:
            return PersonalizeResponse(
                success=True,
                chapter_id=request.chapter_id,
                personalized_content=cached_content,
                message="Retrieved cached personalized content"
            )
        else:
            return PersonalizeResponse(
                success=False,
                chapter_id=request.chapter_id,
                personalized_content="",
                message="No previously personalized content found for this chapter"
            )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message
        )
    except Exception as e:
        # Log the error
        from utils.error_handler import log_api_error
        log_api_error(f"Failed to retrieve personalized content for user {current_user.get('user_id')} and chapter {request.chapter_id}", e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving personalized content"
        )