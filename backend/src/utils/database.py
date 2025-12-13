"""
Database configuration and connection utilities for the authentication and personalization backend.
"""
import os
from typing import AsyncGenerator
from sqlmodel import create_engine, Session
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from contextlib import asynccontextmanager

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/physical_ai_auth")
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create sync and async engines with connection pool settings
sync_engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using them
    pool_recycle=3600,   # Recycle connections after 1 hour
)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=True,
    pool_pre_ping=True,      # Verify connections before using them
    pool_recycle=3600,       # Recycle connections after 1 hour
    pool_size=10,            # Maximum number of connections
    max_overflow=20,         # Maximum overflow connections
    pool_timeout=30,         # Timeout for getting connection from pool
)

@asynccontextmanager
async def get_db_session():
    """
    Get database session for dependency injection.
    Uses pool_pre_ping to ensure connection is alive before using it.
    """
    async with AsyncSession(async_engine, expire_on_commit=False) as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            # Log the error
            import logging
            logging.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()

async def init_db():
    """
    Initialize the database by creating all tables.
    """
    from models.user import User, UserProfile, PersonalizationRecord, AISummary  # Import models to register them

    # Create all tables
    async with async_engine.begin() as conn:
        await conn.run_sync(User.__table__.create)
        await conn.run_sync(UserProfile.__table__.create)
        await conn.run_sync(PersonalizationRecord.__table__.create)
        await conn.run_sync(AISummary.__table__.create)

async def check_db_connection():
    """
    Check if the database connection is alive and working.
    Returns True if connection is healthy, False otherwise.
    """
    try:
        async with get_db_session() as session:
            await session.execute("SELECT 1")
            return True
    except Exception as e:
        import logging
        logging.error(f"Database connection check failed: {e}")
        return False

async def dispose_engine():
    """
    Dispose of the async engine and close all connections.
    Useful for cleanup or when restarting the connection pool.
    """
    await async_engine.dispose()

# For Alembic migrations
def get_engine():
    """
    Get the async engine for migrations.
    """
    return async_engine