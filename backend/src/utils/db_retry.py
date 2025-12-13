"""
Database connection retry and error handling utilities.
"""
import asyncio
import logging
from functools import wraps
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

def retry_on_db_error(max_retries=3, delay=1.0):
    """
    Decorator to retry database operations on connection errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (DBAPIError, OperationalError) as e:
                    last_exception = e
                    error_msg = str(e)
                    
                    # Check if it's a connection error
                    if "connection" in error_msg.lower() or "closed" in error_msg.lower():
                        if attempt < max_retries - 1:
                            wait_time = delay * (attempt + 1)
                            logger.warning(
                                f"Database connection error on attempt {attempt + 1}/{max_retries}. "
                                f"Retrying in {wait_time}s... Error: {error_msg}"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                    
                    # If it's not a connection error, raise immediately
                    raise
                except Exception as e:
                    # For non-database errors, raise immediately
                    raise
            
            # If all retries failed, raise the last exception
            logger.error(f"All {max_retries} retry attempts failed. Last error: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator


async def ensure_connection_alive(session: AsyncSession) -> bool:
    """
    Check if the database connection is alive.
    
    Args:
        session: The database session to check
        
    Returns:
        True if connection is alive, False otherwise
    """
    try:
        from sqlalchemy import text
        await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Connection health check failed: {e}")
        return False
