"""
AI Summary service for the Physical AI textbook platform.
This module provides business logic for AI-generated chapter summaries.
"""
from typing import Dict, Any, Optional
from sqlmodel import select
import json
from datetime import datetime, timedelta
import logging

from models.user import AISummary
from utils.database import get_db_session
from utils.gemini_client import get_ai_agent
from .content_service import get_chapter_content


logger = logging.getLogger(__name__)


async def get_or_generate_summary(chapter_id: str, user_id: str, force_new: bool = False) -> tuple[Optional[str], bool]:
    """
    Get a cached summary for a chapter or generate a new one if it doesn't exist.

    Args:
        chapter_id: The ID of the chapter
        user_id: The ID of the user requesting the summary
        force_new: Whether to force generation of a new summary

    Returns:
        Tuple of (summary content, whether it was cached)
    """
    if not force_new:
        # Try to get cached summary
        cached_summary = await get_cached_summary(chapter_id)
        if cached_summary:
            # Update access count and last accessed time
            await update_summary_access_stats(chapter_id, user_id)
            return cached_summary, True

    # Generate new summary
    new_summary = await generate_new_summary(chapter_id)
    if new_summary:
        # Save the new summary to cache
        await save_summary_to_cache(chapter_id, new_summary)
        return new_summary, False

    return None, False


async def get_cached_summary(chapter_id: str) -> Optional[str]:
    """
    Get a cached summary for a chapter.

    Args:
        chapter_id: The ID of the chapter

    Returns:
        Cached summary if found, None otherwise
    """
    async with get_db_session() as session:
        statement = select(AISummary).where(AISummary.chapter_id == chapter_id)
        result = await session.execute(statement)
        summary_record = result.first()

        if summary_record:
            return summary_record[0].summary_content

        return None


async def generate_new_summary(chapter_id: str) -> Optional[str]:
    """
    Generate a new AI summary for a chapter.

    Args:
        chapter_id: The ID of the chapter

    Returns:
        Generated summary if successful, None otherwise
    """
    try:
        # Get the chapter content
        chapter_content = await get_chapter_content(chapter_id)
        if not chapter_content:
            logger.error(f"Could not retrieve content for chapter {chapter_id}")
            return None

        # Get the AI agent for summary generation
        ai_agent = get_ai_agent()

        # Generate the summary
        summary = await ai_agent.generate_summary(
            content=chapter_content,
            max_length=500,  # Limit to 500 words
            style="concise"
        )

        return summary

    except Exception as e:
        logger.error(f"Error generating summary for chapter {chapter_id}: {str(e)}")
        return None


async def save_summary_to_cache(chapter_id: str, summary_content: str) -> bool:
    """
    Save a summary to the cache.

    Args:
        chapter_id: The ID of the chapter
        summary_content: The summary content to save

    Returns:
        True if save was successful, False otherwise
    """
    try:
        async with get_db_session() as session:
            # Check if a summary already exists for this chapter
            statement = select(AISummary).where(AISummary.chapter_id == chapter_id)
            result = await session.execute(statement)
            existing_record = result.first()

            if existing_record:
                # Update existing record
                record = existing_record[0]
                record.summary_content = summary_content
                record.summary_metadata = {
                    "updated_at": datetime.utcnow().isoformat(),
                    "generated_by_ai": True,
                    "content_length": len(summary_content)
                }
                record.updated_at = datetime.utcnow()
            else:
                # Create new record
                record = AISummary(
                    chapter_id=chapter_id,
                    summary_content=summary_content,
                    summary_metadata={
                        "created_at": datetime.utcnow().isoformat(),
                        "generated_by_ai": True,
                        "content_length": len(summary_content)
                    }
                )
                session.add(record)

            # The session will be committed automatically by the context manager
            return True

    except Exception as e:
        logger.error(f"Error saving summary to cache for chapter {chapter_id}: {str(e)}")
        return False


async def update_summary_access_stats(chapter_id: str, user_id: str) -> bool:
    """
    Update access statistics for a summary.

    Args:
        chapter_id: The ID of the chapter
        user_id: The ID of the user accessing the summary

    Returns:
        True if update was successful, False otherwise
    """
    try:
        async with get_db_session() as session:
            statement = select(AISummary).where(AISummary.chapter_id == chapter_id)
            result = await session.execute(statement)
            record = result.first()

            if record:
                record = record[0]
                record.access_count = record.access_count + 1
                record.last_accessed = datetime.utcnow()
                record.updated_at = datetime.utcnow()

                # The session will be committed automatically by the context manager
                return True

            return False

    except Exception as e:
        logger.error(f"Error updating access stats for chapter {chapter_id}: {str(e)}")
        return False


async def get_summary_metadata(chapter_id: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a chapter's summary.

    Args:
        chapter_id: The ID of the chapter

    Returns:
        Summary metadata if summary exists, None otherwise
    """
    async with get_db_session() as session:
        statement = select(AISummary).where(AISummary.chapter_id == chapter_id)
        result = await session.execute(statement)
        record = result.first()

        if record:
            record = record[0]
            return {
                "chapter_id": record.chapter_id,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "updated_at": record.updated_at.isoformat() if record.updated_at else None,
                "access_count": record.access_count,
                "last_accessed": record.last_accessed.isoformat() if record.last_accessed else None,
                "content_length": len(record.summary_content),
                "metadata": record.summary_metadata
            }

        return None


async def invalidate_summary_cache(chapter_id: str) -> bool:
    """
    Invalidate the cached summary for a chapter (mark for regeneration).

    Args:
        chapter_id: The ID of the chapter

    Returns:
        True if invalidation was successful, False otherwise
    """
    # In this implementation, we'll just delete the cached summary
    # In a real implementation, you might want to mark it as stale instead
    try:
        async with get_db_session() as session:
            statement = select(AISummary).where(AISummary.chapter_id == chapter_id)
            result = await session.execute(statement)
            record = result.first()

            if record:
                await session.delete(record[0])
                # The session will be committed automatically by the context manager

            return True

    except Exception as e:
        logger.error(f"Error invalidating summary cache for chapter {chapter_id}: {str(e)}")
        return False


async def get_popular_summaries(limit: int = 10) -> list:
    """
    Get the most popular summaries based on access count.

    Args:
        limit: Maximum number of summaries to return

    Returns:
        List of popular summaries with metadata
    """
    async with get_db_session() as session:
        statement = select(AISummary).order_by(AISummary.access_count.desc()).limit(limit)
        result = await session.execute(statement)
        records = result.all()

        popular_summaries = []
        for record in records:
            record = record[0]
            popular_summaries.append({
                "chapter_id": record.chapter_id,
                "access_count": record.access_count,
                "summary_preview": record.summary_content[:100] + "..." if len(record.summary_content) > 100 else record.summary_content,
                "last_accessed": record.last_accessed.isoformat() if record.last_accessed else None
            })

        return popular_summaries


async def apply_content_safety_filters(summary_content: str) -> str:
    """
    Apply content safety filters to a summary.

    Args:
        summary_content: The summary content to filter

    Returns:
        Filtered summary content
    """
    # This is a placeholder implementation
    # In a real implementation, you would integrate with content safety APIs
    # or implement content filtering logic

    # For now, just return the content as-is
    # In a real implementation, you would check for inappropriate content
    # and either filter it or raise an exception
    return summary_content


async def get_summarization_stats() -> Dict[str, Any]:
    """
    Get statistics about the summarization system.

    Returns:
        Dictionary with summarization statistics
    """
    async with get_db_session() as session:
        # Count total summaries
        from sqlalchemy import func
        total_summaries = await session.execute(select(func.count(AISummary.id)))
        total_count = total_summaries.scalar()

        # Calculate total access count
        total_accesses_result = await session.execute(select(func.sum(AISummary.access_count)))
        total_accesses = total_accesses_result.scalar() or 0

        return {
            "total_summaries": total_count,
            "total_accesses": total_accesses,
            "average_accesses_per_summary": total_accesses / total_count if total_count > 0 else 0
        }