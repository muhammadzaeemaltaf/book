"""
Personalization service for the Physical AI textbook platform.
This module provides business logic for content personalization based on user profiles.
"""
from typing import Dict, Any, Optional
from sqlmodel import select
import json
from datetime import datetime

from models.user import PersonalizationRecord, UserProfile
from utils.database import get_db_session
from utils.gemini_client import get_ai_agent
from .content_service import get_chapter_content


async def personalize_content_for_user(
    chapter_id: str,
    original_content: str,
    user_profile: Dict[str, Any],
    target_level: str = "adaptive"
) -> str:
    """
    Personalize content for a specific user based on their profile.

    Args:
        chapter_id: The ID of the chapter
        original_content: The original content to personalize
        user_profile: The user's profile information
        target_level: The target level for personalization

    Returns:
        Personalized content string
    """
    # Get the AI agent for content personalization
    ai_agent = get_ai_agent()

    # Personalize the content using the AI agent
    personalized_content = await ai_agent.personalize_content(
        original_content=original_content,
        user_profile=user_profile,
        target_audience=target_level
    )

    # Save the personalization record to the database
    await save_personalization_record(
        user_id=user_profile["user_id"],
        chapter_id=chapter_id,
        original_content=original_content,
        personalized_content=personalized_content,
        user_profile=user_profile
    )

    return personalized_content


async def get_cached_personalization(user_id: str, chapter_id: str) -> Optional[str]:
    """
    Get cached personalized content for a user and chapter if it exists.

    Args:
        user_id: The ID of the user
        chapter_id: The ID of the chapter

    Returns:
        Cached personalized content if found, None otherwise
    """
    async with get_db_session() as session:
        statement = select(PersonalizationRecord).where(
            PersonalizationRecord.user_id == user_id,
            PersonalizationRecord.chapter_id == chapter_id
        )
        result = await session.execute(statement)
        record = result.first()

        if record:
            return record[0].personalized_content

        return None


async def save_personalization_record(
    user_id: str,
    chapter_id: str,
    original_content: str,
    personalized_content: str,
    user_profile: Dict[str, Any]
) -> bool:
    """
    Save a personalization record to the database.

    Args:
        user_id: The ID of the user
        chapter_id: The ID of the chapter
        original_content: The original content
        personalized_content: The personalized content
        user_profile: The user profile used for personalization

    Returns:
        True if save was successful, False otherwise
    """
    async with get_db_session() as session:
        # Check if a record already exists for this user and chapter
        statement = select(PersonalizationRecord).where(
            PersonalizationRecord.user_id == user_id,
            PersonalizationRecord.chapter_id == chapter_id
        )
        result = await session.execute(statement)
        existing_record = result.first()

        if existing_record:
            # Update existing record
            record = existing_record[0]
            record.personalized_content = personalized_content
            record.personalization_metadata = {
                "updated_at": datetime.utcnow().isoformat(),
                "user_profile_snapshot": user_profile,
                "original_content_length": len(original_content)
            }
            record.updated_at = datetime.utcnow()
        else:
            # Create new record
            record = PersonalizationRecord(
                user_id=user_id,
                chapter_id=chapter_id,
                personalized_content=personalized_content,
                personalization_metadata={
                    "created_at": datetime.utcnow().isoformat(),
                    "user_profile_snapshot": user_profile,
                    "original_content_length": len(original_content)
                }
            )
            session.add(record)

        # The session will be committed automatically by the context manager
        return True


async def get_personalization_history(user_id: str, limit: int = 10) -> list:
    """
    Get the personalization history for a user.

    Args:
        user_id: The ID of the user
        limit: Maximum number of records to return

    Returns:
        List of personalization records
    """
    async with get_db_session() as session:
        statement = select(PersonalizationRecord).where(
            PersonalizationRecord.user_id == user_id
        ).order_by(PersonalizationRecord.updated_at.desc()).limit(limit)

        result = await session.execute(statement)
        records = result.all()

        return [record[0] for record in records]


async def update_personalization_for_profile_changes(user_id: str, new_profile: Dict[str, Any]) -> int:
    """
    Update personalizations when a user's profile changes.
    This is a simplified implementation - in a real system you might want to
    regenerate personalizations when significant profile changes occur.

    Args:
        user_id: The ID of the user
        new_profile: The updated profile

    Returns:
        Number of personalizations updated
    """
    # Get all previous personalization records for this user
    history = await get_personalization_history(user_id, limit=100)  # reasonable limit

    updated_count = 0

    for record in history:
        try:
            # Get the original content for this personalization
            from .content_service import get_chapter_content
            original_content = await get_chapter_content(record.chapter_id)

            if original_content:
                # Regenerate the personalization with the new profile
                ai_agent = get_ai_agent()
                new_personalized_content = await ai_agent.personalize_content(
                    original_content=original_content,
                    user_profile=new_profile,
                    target_audience="adaptive"
                )

                # Update the record with new personalized content
                async with get_db_session() as session:
                    # Fetch the specific record to update
                    statement = select(PersonalizationRecord).where(
                        PersonalizationRecord.id == record.id
                    )
                    result = await session.execute(statement)
                    existing_record = result.first()

                    if existing_record:
                        existing_record = existing_record[0]
                        existing_record.personalized_content = new_personalized_content
                        existing_record.personalization_metadata = {
                            **existing_record.personalization_metadata,
                            "updated_for_profile_change": datetime.utcnow().isoformat(),
                            "user_profile_snapshot": new_profile
                        }
                        existing_record.updated_at = datetime.utcnow()

                        updated_count += 1

        except Exception as e:
            # Log error but continue processing other records
            from utils.error_handler import log_api_error
            log_api_error(f"Failed to update personalization for user {user_id}, chapter {record.chapter_id}", e)
            continue

    return updated_count


async def get_personalization_stats(user_id: str) -> Dict[str, Any]:
    """
    Get statistics about a user's personalizations.

    Args:
        user_id: The ID of the user

    Returns:
        Dictionary with personalization statistics
    """
    history = await get_personalization_history(user_id, limit=1000)  # large limit for stats
    total_personalizations = len(history)

    # Calculate some basic stats
    total_content_length = sum(len(record.personalized_content) for record in history)

    return {
        "total_personalizations": total_personalizations,
        "total_content_length": total_content_length,
        "average_content_length": total_content_length / total_personalizations if total_personalizations > 0 else 0
    }


async def personalize_chapter_content(
    user_id: str,
    chapter_id: str,
    target_level: str = "adaptive"
) -> str:
    """
    Personalize an entire chapter's content based on user profile.

    Args:
        user_id: The ID of the user
        chapter_id: The ID of the chapter
        target_level: The target level for personalization

    Returns:
        Personalized chapter content
    """
    # Get the original chapter content
    original_content = await get_chapter_content(chapter_id)
    if not original_content:
        raise ValueError(f"Chapter content not found for chapter_id: {chapter_id}")

    # Get user profile
    from services.profile_service import get_profile_summary
    user_profile = await get_profile_summary(user_id)
    if not user_profile:
        raise ValueError(f"User profile not found for user_id: {user_id}")

    # Personalize the content
    return await personalize_content_for_user(
        chapter_id=chapter_id,
        original_content=original_content,
        user_profile=user_profile,
        target_level=target_level
    )