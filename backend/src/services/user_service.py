"""
User service for the Physical AI textbook platform.
This module provides business logic for user-related operations.
"""
from typing import Dict, Any, Optional
from sqlmodel import select
from uuid import UUID
import uuid

from models.user import User, UserProfile, ExperienceLevel, GPUOption, RAMCapacity, OperatingSystem
from utils.database import get_db_session
from utils.better_auth import better_auth


async def create_user_with_profile(
    email: str,
    password: str,
    name: str,
    profile_data: Dict[str, Any]
) -> tuple[str, bool]:
    """
    Create a new user with profile information.

    Args:
        email: User's email address
        password: User's password
        name: User's name
        profile_data: Technical background information

    Returns:
        Tuple of (user_id, profile_created) where profile_created indicates if profile was created
    """
    # Generate user ID
    user_id = str(uuid.uuid4())

    # Validate profile data
    validated_profile_data = validate_profile_data(profile_data)

    async with get_db_session() as session:
        # First, create the user record
        user = User(
            id=user_id,
            email=email,
            name=name,
            email_verified=None,  # Will be set when user verifies email
            image=None
        )
        session.add(user)
        await session.flush()  # Flush to ensure user exists before creating profile

        # Then create the user profile with the user_id
        profile = UserProfile(
            user_id=user_id,
            python_experience=validated_profile_data.get('python_experience', ExperienceLevel.none),
            cpp_experience=validated_profile_data.get('cpp_experience', ExperienceLevel.none),
            js_ts_experience=validated_profile_data.get('js_ts_experience', ExperienceLevel.none),
            ai_ml_familiarity=validated_profile_data.get('ai_ml_familiarity', ExperienceLevel.none),
            ros2_experience=validated_profile_data.get('ros2_experience', ExperienceLevel.none),
            gpu_details=validated_profile_data.get('gpu_details', GPUOption.none),
            ram_capacity=validated_profile_data.get('ram_capacity', RAMCapacity.gb_4),
            operating_system=validated_profile_data.get('operating_system', OperatingSystem.linux),
            jetson_ownership=validated_profile_data.get('jetson_ownership', False),
            realsense_lidar_availability=validated_profile_data.get('realsense_lidar_availability', False)
        )
        session.add(profile)
        # The session will be committed automatically by the context manager
        # when it exits successfully

    return user_id, True


def validate_profile_data(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize profile data.

    Args:
        profile_data: Raw profile data to validate

    Returns:
        Validated and normalized profile data
    """
    validated = {}

    # Validate experience levels
    valid_experience_levels = [level.value for level in ExperienceLevel]
    validated['python_experience'] = profile_data.get('python_experience', 'none')
    if validated['python_experience'] not in valid_experience_levels:
        validated['python_experience'] = 'none'

    validated['cpp_experience'] = profile_data.get('cpp_experience', 'none')
    if validated['cpp_experience'] not in valid_experience_levels:
        validated['cpp_experience'] = 'none'

    validated['js_ts_experience'] = profile_data.get('js_ts_experience', 'none')
    if validated['js_ts_experience'] not in valid_experience_levels:
        validated['js_ts_experience'] = 'none'

    validated['ai_ml_familiarity'] = profile_data.get('ai_ml_familiarity', 'none')
    if validated['ai_ml_familiarity'] not in valid_experience_levels:
        validated['ai_ml_familiarity'] = 'none'

    validated['ros2_experience'] = profile_data.get('ros2_experience', 'none')
    if validated['ros2_experience'] not in valid_experience_levels:
        validated['ros2_experience'] = 'none'

    # Validate GPU details
    valid_gpu_options = [option.value for option in GPUOption]
    validated['gpu_details'] = profile_data.get('gpu_details', 'none')
    if validated['gpu_details'] not in valid_gpu_options:
        validated['gpu_details'] = 'none'

    # Validate RAM capacity
    valid_ram_capacities = [capacity.value for capacity in RAMCapacity]
    validated['ram_capacity'] = profile_data.get('ram_capacity', '4GB')
    if validated['ram_capacity'] not in valid_ram_capacities:
        validated['ram_capacity'] = '4GB'

    # Validate operating system
    valid_os_options = [os.value for os in OperatingSystem]
    validated['operating_system'] = profile_data.get('operating_system', 'linux')
    if validated['operating_system'] not in valid_os_options:
        validated['operating_system'] = 'linux'

    # Validate boolean fields
    validated['jetson_ownership'] = bool(profile_data.get('jetson_ownership', False))
    validated['realsense_lidar_availability'] = bool(profile_data.get('realsense_lidar_availability', False))

    return validated


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
        profile = result.first()
        return profile[0] if profile else None


async def update_user_profile(user_id: str, profile_data: Dict[str, Any]) -> bool:
    """
    Update a user's profile information.

    Args:
        user_id: The ID of the user
        profile_data: Updated profile data

    Returns:
        True if update was successful, False otherwise
    """
    async with get_db_session() as session:
        statement = select(UserProfile).where(UserProfile.user_id == user_id)
        result = await session.execute(statement)
        profile_row = result.first()

        if not profile_row:
            return False

        profile = profile_row[0]

        # Validate and update profile fields
        validated_data = validate_profile_data(profile_data)

        for field, value in validated_data.items():
            if hasattr(profile, field):
                setattr(profile, field, value)

        session.add(profile)
        # The session will be committed automatically by the context manager

        return True


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

    # For now, consider profile complete if it exists
    # In a real implementation, you might have required fields
    return True