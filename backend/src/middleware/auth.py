"""
Authentication middleware for session validation in the Physical AI textbook platform.
This module provides dependency functions for validating authentication sessions.
"""
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer

from utils.better_auth import get_current_user, better_auth


async def get_current_active_user(request: Request = None) -> Optional[Dict[str, Any]]:
    """
    Get the current active user from the request, validating the session.

    Args:
        request: The FastAPI request object containing the authorization header

    Returns:
        The user information if authenticated and active

    Raises:
        HTTPException: If not authenticated or if the session is invalid
    """
    if request is None:
        raise HTTPException(status_code=401, detail="Request object is required")

    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Additional validation can be added here if needed
    # For example, checking if the user account is active, not suspended, etc.

    return user


def require_auth(current_user: Dict[str, Any] = Depends(get_current_active_user)) -> Dict[str, Any]:
    """
    Dependency to require authentication for protected endpoints.

    This function can be used as a FastAPI dependency to enforce authentication
    on specific routes/endpoints.

    Args:
        current_user: The current authenticated user (injected via dependency injection)

    Returns:
        The authenticated user information

    Raises:
        HTTPException: If authentication fails
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    return current_user


async def require_auth_optional(request: Request = None) -> Optional[Dict[str, Any]]:
    """
    Dependency to optionally check for authentication.

    This function can be used for endpoints that work differently based on
    whether the user is authenticated or not.

    Args:
        request: The FastAPI request object containing the authorization header

    Returns:
        The user information if authenticated, None otherwise
    """
    try:
        return await get_current_active_user(request)
    except HTTPException:
        # If authentication fails, return None instead of raising an exception
        return None


# Security scheme for API docs
auth_scheme = HTTPBearer()


async def validate_session_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Validate a session token and return user information if valid.

    Args:
        token: The authentication token to validate

    Returns:
        User information if token is valid, None otherwise
    """
    try:
        user_payload = await better_auth.verify_token(token)
        return user_payload
    except Exception:
        return None


async def get_user_id_from_token(token: str) -> Optional[str]:
    """
    Extract user ID from authentication token.

    Args:
        token: The authentication token

    Returns:
        The user ID if found and valid, None otherwise
    """
    user_info = await validate_session_token(token)
    if user_info and 'user_id' in user_info:
        return user_info['user_id']
    return None