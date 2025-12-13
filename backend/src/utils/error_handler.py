"""
API error handling utilities for the Physical AI textbook platform.
This module provides standardized error handling and exception classes for the API.
"""
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
import logging


# Set up logger for error handling
logger = logging.getLogger(__name__)


class APIError(Exception):
    """
    Base exception class for API errors.
    """
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary representation.
        """
        return {
            "error": self.message,
            "status_code": self.status_code,
            "details": self.details
        }


class ValidationError(APIError):
    """
    Exception raised for validation errors.
    """
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=error_details
        )


class AuthenticationError(APIError):
    """
    Exception raised for authentication errors.
    """
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


class AuthorizationError(APIError):
    """
    Exception raised for authorization errors.
    """
    def __init__(self, message: str = "Not authorized", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )


class NotFoundError(APIError):
    """
    Exception raised when a resource is not found.
    """
    def __init__(self, resource: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"{resource} not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details or {"resource": resource}
        )


class ConflictError(APIError):
    """
    Exception raised when there's a conflict (e.g., duplicate resource).
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            details=details
        )


def create_error_response(
    message: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """
    Create a standardized error response.

    Args:
        message: The error message
        status_code: The HTTP status code
        details: Additional error details

    Returns:
        JSONResponse with standardized error format
    """
    error_payload = {
        "error": message,
        "status_code": status_code
    }

    if details:
        error_payload["details"] = details

    logger.error(f"API Error: {message} (Status: {status_code}, Details: {details})")
    return JSONResponse(
        status_code=status_code,
        content=error_payload
    )


def handle_api_exception(exc: APIError) -> JSONResponse:
    """
    Handle API exceptions and return appropriate response.

    Args:
        exc: The APIError exception

    Returns:
        JSONResponse with error details
    """
    logger.error(f"API Exception: {exc.message} (Status: {exc.status_code}, Details: {exc.details})")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


def handle_http_exception(exc: HTTPException) -> JSONResponse:
    """
    Handle FastAPI HTTP exceptions and return appropriate response.

    Args:
        exc: The HTTPException

    Returns:
        JSONResponse with error details
    """
    logger.error(f"HTTP Exception: {exc.detail} (Status: {exc.status_code})")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": str(exc.detail),
            "status_code": exc.status_code
        }
    )


def log_api_error(message: str, error: Exception, user_id: Optional[str] = None) -> None:
    """
    Log API errors with additional context.

    Args:
        message: The error message
        error: The exception object
        user_id: Optional user ID for the request
    """
    log_data = {
        "message": message,
        "error_type": type(error).__name__,
        "error_details": str(error)
    }

    if user_id:
        log_data["user_id"] = user_id

    logger.error(log_data)


def validate_required_fields(data: Dict[str, Any], required_fields: list) -> None:
    """
    Validate that required fields are present in the data.

    Args:
        data: The data dictionary to validate
        required_fields: List of required field names

    Raises:
        ValidationError: If any required field is missing
    """
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)

    if missing_fields:
        raise ValidationError(
            message=f"Missing required fields: {', '.join(missing_fields)}",
            details={"missing_fields": missing_fields}
        )


def validate_field_length(field_value: str, field_name: str, min_length: int = 1, max_length: int = None) -> None:
    """
    Validate the length of a field.

    Args:
        field_value: The field value to validate
        field_name: The name of the field
        min_length: Minimum length allowed
        max_length: Maximum length allowed

    Raises:
        ValidationError: If the field length is invalid
    """
    if field_value is None:
        raise ValidationError(
            message=f"{field_name} cannot be null",
            field=field_name
        )

    if len(field_value) < min_length:
        raise ValidationError(
            message=f"{field_name} must be at least {min_length} characters long",
            field=field_name,
            details={"min_length": min_length, "actual_length": len(field_value)}
        )

    if max_length and len(field_value) > max_length:
        raise ValidationError(
            message=f"{field_name} must be no more than {max_length} characters long",
            field=field_name,
            details={"max_length": max_length, "actual_length": len(field_value)}
        )


def validate_email_format(email: str) -> None:
    """
    Validate email format.

    Args:
        email: The email address to validate

    Raises:
        ValidationError: If the email format is invalid
    """
    import re
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_regex, email):
        raise ValidationError(
            message="Invalid email format",
            field="email",
            details={"email": email}
        )


# Custom exception handlers for FastAPI
def add_exception_handlers(app):
    """
    Add exception handlers to a FastAPI application.

    Args:
        app: The FastAPI application instance
    """
    @app.exception_handler(APIError)
    async def api_error_handler(request, exc):
        return handle_api_exception(exc)

    @app.exception_handler(HTTPException)
    async def http_error_handler(request, exc):
        return handle_http_exception(exc)