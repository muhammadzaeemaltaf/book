"""
Rate limiting middleware for the Physical AI textbook platform.
This module provides rate limiting functionality for API endpoints.
"""
import time
from typing import Dict, Optional
from fastapi import Request, HTTPException, status
from collections import defaultdict, deque
import hashlib


class RateLimiter:
    """
    Simple in-memory rate limiter.
    In production, you would typically use Redis or another distributed store.
    """
    def __init__(self):
        # Store requests per identifier with timestamps
        self.requests = defaultdict(deque)
        # Default limit: 100 requests per hour per user
        self.default_limit = 100
        self.default_window = 3600  # 1 hour in seconds

    def is_allowed(
        self,
        identifier: str,
        limit: int = None,
        window: int = None
    ) -> bool:
        """
        Check if a request from the given identifier is allowed.

        Args:
            identifier: Unique identifier for the requester (e.g., user ID, IP)
            limit: Number of requests allowed per window
            window: Time window in seconds

        Returns:
            True if request is allowed, False otherwise
        """
        if limit is None:
            limit = self.default_limit
        if window is None:
            window = self.default_window

        now = time.time()
        window_start = now - window

        # Remove old requests outside the time window
        while self.requests[identifier] and self.requests[identifier][0] < window_start:
            self.requests[identifier].popleft()

        # Check if we're under the limit
        if len(self.requests[identifier]) < limit:
            self.requests[identifier].append(now)
            return True

        return False

    def get_reset_time(self, identifier: str, window: int = None) -> int:
        """
        Get the time when the rate limit will reset.

        Args:
            identifier: Unique identifier for the requester
            window: Time window in seconds

        Returns:
            Unix timestamp when the rate limit will reset
        """
        if window is None:
            window = self.default_window

        now = time.time()
        return int(now + window)


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_client_identifier(request: Request) -> str:
    """
    Get a unique identifier for the client making the request.
    This combines user ID (if authenticated) and IP address.

    Args:
        request: The FastAPI request object

    Returns:
        Unique identifier string
    """
    # Get user ID if available
    user_id = None
    if hasattr(request.state, 'user') and request.state.user:
        user_id = request.state.user.get('user_id')

    # Get IP address
    client_ip = request.client.host if request.client else "unknown"

    # Create identifier - if authenticated, use user ID; otherwise use IP
    if user_id:
        identifier = f"user:{user_id}"
    else:
        identifier = f"ip:{client_ip}"

    # Use hash to ensure consistent length
    return hashlib.sha256(identifier.encode()).hexdigest()


def rate_limit(
    limit: int = 100,
    window: int = 3600,  # 1 hour
    per_endpoint: bool = False
):
    """
    Rate limiting decorator for FastAPI routes.

    Args:
        limit: Number of requests allowed per window
        window: Time window in seconds
        per_endpoint: Whether to apply limits per endpoint as well as per user

    Returns:
        Request validation function
    """
    def rate_limit_dependency(request: Request):
        identifier = get_client_identifier(request)

        # If per_endpoint is True, include the endpoint in the identifier
        if per_endpoint:
            endpoint = f"{request.method}:{request.url.path}"
            identifier = f"{identifier}:{endpoint}"

        if not rate_limiter.is_allowed(identifier, limit, window):
            reset_time = rate_limiter.get_reset_time(identifier, window)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": limit,
                    "window": window,
                    "reset_time": reset_time
                }
            )

        # Add rate limit info to response headers
        remaining = limit - len(rate_limiter.requests[identifier])
        request.state.rate_limit_remaining = remaining
        request.state.rate_limit_reset = rate_limiter.get_reset_time(identifier, window)

        return True

    return rate_limit_dependency


# Specific rate limiters for different use cases

def ai_summary_rate_limit():
    """
    Rate limiter specifically for AI summary endpoints.
    More restrictive than default to control AI usage costs.
    """
    def rate_limit_dependency(request: Request):
        identifier = get_client_identifier(request)

        # AI summary specific rate limit: 50 requests per hour
        limit = 50
        window = 3600  # 1 hour

        # If per endpoint is True, include the endpoint in the identifier
        endpoint = f"{request.method}:{request.url.path}"
        identifier = f"{identifier}:{endpoint}"

        if not rate_limiter.is_allowed(identifier, limit, window):
            reset_time = rate_limiter.get_reset_time(identifier, window)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "AI summary rate limit exceeded",
                    "limit": limit,
                    "window": window,
                    "reset_time": reset_time
                }
            )

        # Add rate limit info to response headers
        remaining = limit - len(rate_limiter.requests[identifier])
        request.state.rate_limit_remaining = remaining
        request.state.rate_limit_reset = rate_limiter.get_reset_time(identifier, window)

        return True

    return rate_limit_dependency


def get_rate_limit_headers(request: Request) -> Dict[str, str]:
    """
    Get rate limit headers to add to response.

    Args:
        request: The FastAPI request object

    Returns:
        Dictionary of rate limit headers
    """
    headers = {}
    if hasattr(request.state, 'rate_limit_remaining'):
        headers['X-RateLimit-Remaining'] = str(request.state.rate_limit_remaining)
    if hasattr(request.state, 'rate_limit_reset'):
        headers['X-RateLimit-Reset'] = str(request.state.rate_limit_reset)
    return headers


# Middleware to add rate limit headers to all responses
async def add_rate_limit_headers(request: Request, call_next):
    """
    Middleware to add rate limit headers to responses.
    """
    response = await call_next(request)

    # Add rate limit headers if they exist
    rate_limit_headers = get_rate_limit_headers(request)
    for header, value in rate_limit_headers.items():
        response.headers[header] = value

    return response