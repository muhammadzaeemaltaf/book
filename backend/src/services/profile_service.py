"""
Profile service for the Physical AI textbook platform.
This module provides business logic for user profile operations and validation.
"""
from typing import Dict, Any, Optional
from sqlmodel import select

from models.user import UserProfile, ExperienceLevel
from utils.database import get_db_session
from .validation import validate_user_profile_data


async def is_profile_complete(user_id: str) -> bool:
    """
    Check if a user's profile is complete (has required information).

    Args:
        user_id: The ID of the user

    Returns:
        True if profile is complete, False otherwise
    """
    profile = await get_user_profile(user_id)
    if not profile:
        return False

    # Define what constitutes a "complete" profile
    # For this implementation, we'll consider it complete if the user has
    # provided at least some technical background information
    required_fields_exist = (
        profile.python_experience != ExperienceLevel.none or
        profile.cpp_experience != ExperienceLevel.none or
        profile.js_ts_experience != ExperienceLevel.none or
        profile.ai_ml_familiarity != ExperienceLevel.none or
        profile.ros2_experience != ExperienceLevel.none
    )

    return required_fields_exist


async def get_user_profile(user_id: str) -> Optional[UserProfile]:
    """
    Get a user's profile information.

    Args:
        user_id: The ID of the user

    Returns:
        UserProfile if found, None otherwise
    """
    async with get_db_session() as session:
        statement = select(UserProfile).where(UserProfile.user_id == user_id)
        result = await session.execute(statement)
        profile_row = result.first()
        return profile_row[0] if profile_row else None


async def update_user_profile(user_id: str, profile_data: Dict[str, Any]) -> bool:
    """
    Update a user's profile information.

    Args:
        user_id: The ID of the user
        profile_data: Updated profile data

    Returns:
        True if update was successful, False otherwise
    """
    # Validate the profile data first
    try:
        validate_user_profile_data(profile_data)
    except Exception:
        return False

    async with get_db_session() as session:
        statement = select(UserProfile).where(UserProfile.user_id == user_id)
        result = await session.execute(statement)
        profile_row = result.first()

        if not profile_row:
            return False

        profile = profile_row[0]

        # Update profile fields based on provided data
        for field, value in profile_data.items():
            if hasattr(profile, field) and value is not None:
                setattr(profile, field, value)

        session.add(profile)
        # The session will be committed automatically by the context manager

        return True


async def validate_profile_completion(user_id: str) -> Dict[str, Any]:
    """
    Validate profile completion and return details about what's missing.

    Args:
        user_id: The ID of the user

    Returns:
        Dictionary with validation results and missing fields
    """
    profile = await get_user_profile(user_id)
    if not profile:
        return {
            "complete": False,
            "missing_fields": ["profile_not_found"],
            "message": "User profile not found"
        }

    missing_fields = []

    # Check if the user has provided technical experience
    if profile.python_experience == ExperienceLevel.none and \
       profile.cpp_experience == ExperienceLevel.none and \
       profile.js_ts_experience == ExperienceLevel.none and \
       profile.ai_ml_familiarity == ExperienceLevel.none and \
       profile.ros2_experience == ExperienceLevel.none:
        missing_fields.append("technical_experience")

    # Check if the user has provided hardware information
    if profile.gpu_details == "none" and not profile.jetson_ownership and not profile.realsense_lidar_availability:
        missing_fields.append("hardware_information")

    # Check if the user has provided system information
    if profile.ram_capacity == "4GB" and profile.operating_system == "linux":  # Assuming default values
        # Check if these are just defaults or actual user input
        # In a real implementation, you might track whether values are default or set by user
        pass  # For now, we'll consider these acceptable

    is_complete = len(missing_fields) == 0

    if is_complete:
        message = "Profile is complete"
    else:
        message = f"Profile missing: {', '.join(missing_fields)}"

    return {
        "complete": is_complete,
        "missing_fields": missing_fields,
        "message": message
    }


async def calculate_profile_completeness_percentage(user_id: str) -> float:
    """
    Calculate the percentage of profile completion.

    Args:
        user_id: The ID of the user

    Returns:
        Float between 0 and 100 representing completion percentage
    """
    profile = await get_user_profile(user_id)
    if not profile:
        return 0.0

    total_weight = 10  # Total possible weight
    current_weight = 0

    # Experience fields (weighted higher)
    experience_fields = [
        profile.python_experience,
        profile.cpp_experience,
        profile.js_ts_experience,
        profile.ai_ml_familiarity,
        profile.ros2_experience
    ]

    for exp in experience_fields:
        if exp != ExperienceLevel.none:
            current_weight += 1.2  # Higher weight for experience fields

    # Hardware fields
    if profile.gpu_details != "none":
        current_weight += 0.8
    if profile.ram_capacity != "4GB":
        current_weight += 0.5
    if profile.operating_system != "linux":
        current_weight += 0.3
    if profile.jetson_ownership:
        current_weight += 0.6
    if profile.realsense_lidar_availability:
        current_weight += 0.6

    # Calculate percentage
    completeness = (current_weight / total_weight) * 100
    return min(completeness, 100.0)  # Cap at 100%


async def get_profile_summary(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a summary of the user's profile for display purposes.

    Args:
        user_id: The ID of the user

    Returns:
        Dictionary with profile summary information
    """
    profile = await get_user_profile(user_id)
    if not profile:
        return None

    completeness_percentage = await calculate_profile_completeness_percentage(user_id)

    return {
        "user_id": user_id,
        "python_experience": profile.python_experience.value,
        "cpp_experience": profile.cpp_experience.value,
        "js_ts_experience": profile.js_ts_experience.value,
        "ai_ml_familiarity": profile.ai_ml_familiarity.value,
        "ros2_experience": profile.ros2_experience.value,
        "gpu_details": profile.gpu_details.value,
        "ram_capacity": profile.ram_capacity.value,
        "operating_system": profile.operating_system.value,
        "jetson_ownership": profile.jetson_ownership,
        "realsense_lidar_availability": profile.realsense_lidar_availability,
        "completeness_percentage": completeness_percentage,
        "profile_complete": await is_profile_complete(user_id)
    }