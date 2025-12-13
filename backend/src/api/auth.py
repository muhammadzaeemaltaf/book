"""
Authentication API endpoints for the Physical AI textbook platform.
This module provides API endpoints for user authentication operations.
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, EmailStr

from middleware.auth import require_auth_optional
from services.user_service import create_user_with_profile
from services.validation import validate_email_format, validate_password_strength
from utils.error_handler import ValidationError, handle_api_exception
from models.user import UserProfile


router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignupRequest(BaseModel):
    """
    Request model for user signup.
    """
    email: EmailStr
    password: str
    name: str
    # Technical background information
    python_experience: str = "none"
    cpp_experience: str = "none"
    js_ts_experience: str = "none"
    ai_ml_familiarity: str = "none"
    ros2_experience: str = "none"
    gpu_details: str = "none"
    ram_capacity: str = "4GB"
    operating_system: str = "linux"
    jetson_ownership: bool = False
    realsense_lidar_availability: bool = False


class UserResponse(BaseModel):
    """
    Response model for user information.
    """
    id: str
    email: str
    name: str


class SignupResponse(BaseModel):
    """
    Response model for user signup.
    """
    success: bool
    user_id: str
    user: UserResponse
    access_token: str
    message: str
    profile_created: bool


class SigninRequest(BaseModel):
    """
    Request model for user signin.
    """
    email: EmailStr
    password: str


class SigninResponse(BaseModel):
    """
    Response model for user signin.
    """
    success: bool
    user_id: str
    user: UserResponse
    access_token: str
    message: str


@router.post("/signup", response_model=SignupResponse)
async def signup(signup_data: SignupRequest):
    """
    Create a new user account with technical background information.

    Args:
        signup_data: User signup information including email, password, and technical background

    Returns:
        SignupResponse with success status and user information

    Raises:
        HTTPException: If signup fails due to validation or other errors
    """
    try:
        # Validate input data
        validate_email_format(signup_data.email)
        validate_password_strength(signup_data.password)

        # Prepare profile data from signup information
        profile_data = {
            "python_experience": signup_data.python_experience,
            "cpp_experience": signup_data.cpp_experience,
            "js_ts_experience": signup_data.js_ts_experience,
            "ai_ml_familiarity": signup_data.ai_ml_familiarity,
            "ros2_experience": signup_data.ros2_experience,
            "gpu_details": signup_data.gpu_details,
            "ram_capacity": signup_data.ram_capacity,
            "operating_system": signup_data.operating_system,
            "jetson_ownership": signup_data.jetson_ownership,
            "realsense_lidar_availability": signup_data.realsense_lidar_availability
        }

        # Create user with profile
        user_id, profile_created = await create_user_with_profile(
            email=signup_data.email,
            password=signup_data.password,
            name=signup_data.name,
            profile_data=profile_data
        )

        # Create a mock token (in real implementation, this would come from BetterAuth)
        # For now, we'll create a simple mock token
        import uuid
        access_token = f"mock_token_{uuid.uuid4()}"

        # Store user info in the mock token store for later verification
        from utils.better_auth import better_auth
        better_auth.mock_token_store[access_token] = {
            "user_id": user_id,
            "email": signup_data.email,
            "name": signup_data.name
        }

        return SignupResponse(
            success=True,
            user_id=user_id,
            user=UserResponse(id=user_id, email=signup_data.email, name=signup_data.name),
            access_token=access_token,
            message="User account created successfully",
            profile_created=profile_created
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message
        )
    except Exception as e:
        # Log the error
        from utils.error_handler import log_api_error
        log_api_error(f"Signup failed for email {signup_data.email}", e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during signup"
        )


@router.post("/signin", response_model=SigninResponse)
async def signin(signin_data: SigninRequest):
    """
    Authenticate user and return access token.

    Args:
        signin_data: User signin information including email and password

    Returns:
        SigninResponse with success status and authentication token

    Raises:
        HTTPException: If signin fails due to invalid credentials or other errors
    """
    from utils.db_retry import retry_on_db_error
    
    @retry_on_db_error(max_retries=3, delay=1.0)
    async def _signin_with_retry():
        # Validate input data
        validate_email_format(signin_data.email)

        # In a real implementation, this would interact with BetterAuth
        # For now, we'll return a mock response with proper structure
        # In an actual implementation, you would validate credentials via BetterAuth

        # This is where BetterAuth integration would happen
        # For now, we'll return a proper response structure
        # In real implementation: result = await better_auth.signin(email, password)

        # Mock successful authentication response
        # In real implementation, BetterAuth would return actual user info and token
        # For now, we'll need to fetch user data from our database
        from sqlmodel import select
        from utils.database import get_db_session
        from models.user import User

        async with get_db_session() as session:
            result = await session.execute(
                select(User).where(User.email == signin_data.email)
            )
            user = result.first()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )

            user = user[0]  # Get the user object from the tuple
            import uuid
            access_token = f"mock_token_{uuid.uuid4()}"

            # Store user info in the mock token store for later verification
            from utils.better_auth import better_auth
            better_auth.mock_token_store[access_token] = {
                "user_id": user.id,
                "email": user.email,
                "name": user.name or user.email
            }

            return SigninResponse(
                success=True,
                user_id=user.id,
                user=UserResponse(id=user.id, email=user.email, name=user.name or user.email),
                access_token=access_token,
                message="Signin successful"
            )
    
    try:
        return await _signin_with_retry()
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message
        )
    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
    except Exception as e:
        # Log the error
        from utils.error_handler import log_api_error
        log_api_error(f"Signin failed for email {signin_data.email}", e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during signin"
        )


@router.post("/signout")
async def signout(current_user: Dict[str, Any] = Depends(require_auth_optional)):
    """
    Sign out the current user.

    Args:
        current_user: The currently authenticated user (optional)

    Returns:
        Success message

    Raises:
        HTTPException: If signout fails
    """
    try:
        # In a real implementation, you would invalidate the user's session/token
        # For now, we'll just return a success message
        return {"message": "Signout successful"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during signout"
        )


@router.get("/me")
async def get_current_user_info(current_user: Dict[str, Any] = Depends(require_auth_optional)):
    """
    Get information about the current authenticated user.

    Args:
        current_user: The currently authenticated user (optional)

    Returns:
        User information if authenticated, empty dict otherwise
    """
    if current_user and current_user.get("user_id"):
        return {
            "authenticated": True,
            "user_id": current_user.get("user_id"),
            "email": current_user.get("email"),
            "name": current_user.get("name")
        }
    else:
        return {"authenticated": False}