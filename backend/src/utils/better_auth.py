"""
BetterAuth integration for the Physical AI textbook platform.
This module handles the integration with BetterAuth via Context7 MCP.
"""
import os
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import jwt

class BetterAuthIntegration:
    """
    Integration class for BetterAuth via Context7 MCP.
    """

    def __init__(self):
        self.auth_url = os.getenv("BETTER_AUTH_URL", "http://localhost:8000")
        self.secret = os.getenv("BETTER_AUTH_SECRET", "your-secret-key-here")
        self.client = httpx.AsyncClient(timeout=30.0)
        # Mock token store for development
        self.mock_token_store = {}

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify a BetterAuth token and return user information.
        """
        # Check if this is a mock token
        if token.startswith("mock_token_"):
            # Check if we have user info stored for this token
            if token in self.mock_token_store:
                return self.mock_token_store[token]
            else:
                # Token not found in mock store
                raise HTTPException(status_code=401, detail="Invalid token")

        # For real tokens, use JWT verification
        try:
            # This would typically call the BetterAuth API to verify the token
            # For now, we'll implement a basic verification
            payload = jwt.decode(token, self.secret, algorithms=["HS256"])

            # Ensure the payload has the expected structure
            if 'user_id' not in payload and 'id' in payload:
                # If 'id' exists but 'user_id' doesn't, map it
                payload['user_id'] = payload['id']

            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    async def get_user_from_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Extract and verify user from request headers.
        """
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]  # Remove "Bearer " prefix
        return await self.verify_token(token)

    async def close(self):
        """
        Close the HTTP client.
        """
        await self.client.aclose()

# Global instance
better_auth = BetterAuthIntegration()

# Security scheme for API docs
security_scheme = HTTPBearer()

async def get_current_user(request: Request = None) -> Optional[Dict[str, Any]]:
    """
    Dependency to get current user from request.
    """
    if request is None:
        return None

    user = await better_auth.get_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return user