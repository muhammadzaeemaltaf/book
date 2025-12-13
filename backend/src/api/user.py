"""
User API endpoints for the Physical AI textbook platform.
This module provides API endpoints for user profile management.
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from middleware.auth import require_auth
from services.profile_service import update_user_profile, get_profile_summary
from services.validation import validate_user_profile_data
from utils.error_handler import ValidationError, handle_api_exception


router = APIRouter(prefix="/api/user", tags=["user"])


class UpdateProfileRequest(BaseModel):
    """
    Request model for updating user profile.
    """
    # Technical background information
    python_experience: str = None
    cpp_experience: str = None
    js_ts_experience: str = None
    ai_ml_familiarity: str = None
    ros2_experience: str = None
    gpu_details: str = None
    ram_capacity: str = None
    operating_system: str = None
    jetson_ownership: bool = None
    realsense_lidar_availability: bool = None


class UpdateProfileResponse(BaseModel):
    """
    Response model for updating user profile.
    """
    success: bool
    message: str


class ProfileResponse(BaseModel):
    """
    Response model for getting user profile.
    """
    success: bool
    profile: Dict[str, Any]
    message: str


@router.put("/profile", response_model=UpdateProfileResponse)
async def update_profile(
    profile_data: UpdateProfileRequest,
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Update the current user's profile information.

    Args:
        profile_data: Updated profile information
        current_user: The currently authenticated user

    Returns:
        UpdateProfileResponse with success status

    Raises:
        HTTPException: If profile update fails due to validation or other errors
    """
    try:
        # Convert request data to dictionary
        profile_dict = profile_data.dict(exclude_unset=True)

        # Validate the profile data
        validate_user_profile_data(profile_dict)

        # Update the user profile
        success = await update_user_profile(
            user_id=current_user["user_id"],
            profile_data=profile_dict
        )

        if success:
            return UpdateProfileResponse(
                success=True,
                message="Profile updated successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
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
        log_api_error(f"Profile update failed for user {current_user.get('user_id')}", e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during profile update"
        )


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Get the current user's profile information.

    Args:
        current_user: The currently authenticated user

    Returns:
        ProfileResponse with user profile information
    """
    try:
        profile_summary = await get_profile_summary(current_user["user_id"])

        if profile_summary:
            return ProfileResponse(
                success=True,
                profile=profile_summary,
                message="Profile retrieved successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the error
        from utils.error_handler import log_api_error
        log_api_error(f"Profile retrieval failed for user {current_user.get('user_id')}", e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during profile retrieval"
        )


@router.get("/profile/completion")
async def get_profile_completion(
    current_user: Dict[str, Any] = Depends(require_auth)
):
    """
    Get information about profile completion status.

    Args:
        current_user: The currently authenticated user

    Returns:
        Dictionary with profile completion information
    """
    try:
        from services.profile_service import validate_profile_completion
        completion_info = await validate_profile_completion(current_user["user_id"])

        return {
            "user_id": current_user["user_id"],
            **completion_info
        }

    except Exception as e:
        # Log the error
        from utils.error_handler import log_api_error
        log_api_error(f"Profile completion check failed for user {current_user.get('user_id')}", e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during profile completion check"
        )