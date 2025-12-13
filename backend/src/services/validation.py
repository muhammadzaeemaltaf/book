"""
Validation service for the Physical AI textbook platform.
This module provides validation functions for user input and other data.
"""
import re
from typing import Dict, Any
from utils.error_handler import ValidationError


def validate_email_format(email: str) -> bool:
    """
    Validate email format.

    Args:
        email: The email address to validate

    Returns:
        True if email format is valid, False otherwise

    Raises:
        ValidationError: If email format is invalid
    """
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_regex, email):
        raise ValidationError(
            message="Invalid email format",
            field="email",
            details={"email": email}
        )
    return True


def validate_password_strength(password: str) -> bool:
    """
    Validate password strength.

    Args:
        password: The password to validate

    Returns:
        True if password meets strength requirements, False otherwise

    Raises:
        ValidationError: If password doesn't meet requirements
    """
    if len(password) < 8:
        raise ValidationError(
            message="Password must be at least 8 characters long",
            field="password",
            details={"min_length": 8, "actual_length": len(password)}
        )

    # Check for at least one uppercase, one lowercase, and one digit
    if not re.search(r"[A-Z]", password):
        raise ValidationError(
            message="Password must contain at least one uppercase letter",
            field="password"
        )

    if not re.search(r"[a-z]", password):
        raise ValidationError(
            message="Password must contain at least one lowercase letter",
            field="password"
        )

    if not re.search(r"\d", password):
        raise ValidationError(
            message="Password must contain at least one digit",
            field="password"
        )

    return True


def validate_required_fields(data: Dict[str, Any], required_fields: list) -> bool:
    """
    Validate that required fields are present in the data.

    Args:
        data: The data dictionary to validate
        required_fields: List of required field names

    Returns:
        True if all required fields are present, False otherwise

    Raises:
        ValidationError: If any required field is missing
    """
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            missing_fields.append(field)

    if missing_fields:
        raise ValidationError(
            message=f"Missing required fields: {', '.join(missing_fields)}",
            details={"missing_fields": missing_fields}
        )

    return True


def validate_field_length(field_value: str, field_name: str, min_length: int = 1, max_length: int = None) -> bool:
    """
    Validate the length of a field.

    Args:
        field_value: The field value to validate
        field_name: The name of the field
        min_length: Minimum length allowed
        max_length: Maximum length allowed

    Returns:
        True if field length is valid, False otherwise

    Raises:
        ValidationError: If the field length is invalid
    """
    if field_value is None:
        raise ValidationError(
            message=f"{field_name} cannot be null",
            field=field_name
        )

    if len(str(field_value)) < min_length:
        raise ValidationError(
            message=f"{field_name} must be at least {min_length} characters long",
            field=field_name,
            details={"min_length": min_length, "actual_length": len(str(field_value))}
        )

    if max_length and len(str(field_value)) > max_length:
        raise ValidationError(
            message=f"{field_name} must be no more than {max_length} characters long",
            field=field_name,
            details={"max_length": max_length, "actual_length": len(str(field_value))}
        )

    return True


def validate_user_profile_data(profile_data: Dict[str, Any]) -> bool:
    """
    Validate user profile data.

    Args:
        profile_data: The profile data to validate

    Returns:
        True if profile data is valid, False otherwise

    Raises:
        ValidationError: If profile data is invalid
    """
    # Define valid options for each field
    valid_experience_levels = ["none", "beginner", "intermediate", "advanced", "expert"]
    valid_gpu_options = ["none", "1650", "3050+", "4070+", "cloud_gpu"]
    valid_ram_capacities = ["4GB", "8GB", "16GB", "32GB", "64GB+"]
    valid_operating_systems = ["linux", "windows", "mac"]

    # Validate experience levels
    experience_fields = [
        "python_experience", "cpp_experience", "js_ts_experience",
        "ai_ml_familiarity", "ros2_experience"
    ]

    for field in experience_fields:
        value = profile_data.get(field, "none")
        if value not in valid_experience_levels:
            raise ValidationError(
                message=f"Invalid value for {field}: {value}. Must be one of {valid_experience_levels}",
                field=field
            )

    # Validate GPU details
    gpu_value = profile_data.get("gpu_details", "none")
    if gpu_value not in valid_gpu_options:
        raise ValidationError(
            message=f"Invalid value for gpu_details: {gpu_value}. Must be one of {valid_gpu_options}",
            field="gpu_details"
        )

    # Validate RAM capacity
    ram_value = profile_data.get("ram_capacity", "4GB")
    if ram_value not in valid_ram_capacities:
        raise ValidationError(
            message=f"Invalid value for ram_capacity: {ram_value}. Must be one of {valid_ram_capacities}",
            field="ram_capacity"
        )

    # Validate operating system
    os_value = profile_data.get("operating_system", "linux")
    if os_value not in valid_operating_systems:
        raise ValidationError(
            message=f"Invalid value for operating_system: {os_value}. Must be one of {valid_operating_systems}",
            field="operating_system"
        )

    # Validate boolean fields
    boolean_fields = ["jetson_ownership", "realsense_lidar_availability"]
    for field in boolean_fields:
        value = profile_data.get(field)
        if value is not None and not isinstance(value, bool):
            raise ValidationError(
                message=f"{field} must be a boolean value",
                field=field
            )

    return True


def validate_chapter_id(chapter_id: str) -> bool:
    """
    Validate chapter ID format.

    Args:
        chapter_id: The chapter ID to validate

    Returns:
        True if chapter ID is valid, False otherwise

    Raises:
        ValidationError: If chapter ID is invalid
    """
    if not chapter_id or not isinstance(chapter_id, str):
        raise ValidationError(
            message="Chapter ID must be a non-empty string",
            field="chapter_id"
        )

    # Basic validation - chapter ID should not be too long and should have valid characters
    if len(chapter_id) > 100:
        raise ValidationError(
            message="Chapter ID is too long (max 100 characters)",
            field="chapter_id",
            details={"max_length": 100, "actual_length": len(chapter_id)}
        )

    # Check for valid characters (alphanumeric, hyphens, underscores, dots)
    if not re.match(r'^[a-zA-Z0-9._-]+$', chapter_id):
        raise ValidationError(
            message="Chapter ID contains invalid characters",
            field="chapter_id",
            details={"valid_pattern": "alphanumeric, hyphens, underscores, dots"}
        )

    return True


def validate_content(content: str) -> bool:
    """
    Validate content length and format.

    Args:
        content: The content to validate

    Returns:
        True if content is valid, False otherwise

    Raises:
        ValidationError: If content is invalid
    """
    if content is None:
        raise ValidationError(
            message="Content cannot be null",
            field="content"
        )

    content_str = str(content)
    if len(content_str) == 0:
        raise ValidationError(
            message="Content cannot be empty",
            field="content"
        )

    # Check maximum length (e.g., 10MB as per model definition)
    max_length = 10 * 1024 * 1024  # 10MB
    if len(content_str) > max_length:
        raise ValidationError(
            message="Content exceeds maximum length",
            field="content",
            details={"max_length": max_length, "actual_length": len(content_str)}
        )

    return True


def sanitize_input(input_str: str) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        input_str: The input string to sanitize

    Returns:
        Sanitized string
    """
    if input_str is None:
        return None

    # Remove null bytes which can be problematic
    sanitized = input_str.replace('\x00', '')

    # Additional sanitization can be added here as needed
    # For example, you might want to strip certain characters
    # or HTML tags depending on your use case

    return sanitized


def validate_and_sanitize_text(text: str, field_name: str, max_length: int = 10000) -> str:
    """
    Validate and sanitize text input.

    Args:
        text: The text to validate and sanitize
        field_name: The name of the field for error messages
        max_length: Maximum allowed length

    Returns:
        Sanitized text if valid

    Raises:
        ValidationError: If text is invalid
    """
    if text is None:
        raise ValidationError(
            message=f"{field_name} cannot be null",
            field=field_name
        )

    # Sanitize the input
    sanitized_text = sanitize_input(text)

    # Validate length
    if len(sanitized_text) > max_length:
        raise ValidationError(
            message=f"{field_name} exceeds maximum length of {max_length}",
            field=field_name,
            details={"max_length": max_length, "actual_length": len(sanitized_text)}
        )

    # Check for potentially dangerous patterns
    dangerous_patterns = [
        r'<script',  # Potential XSS
        r'javascript:',  # Potential XSS
        r'on\w+\s*=',  # Potential event handlers
    ]

    import re
    for pattern in dangerous_patterns:
        if re.search(pattern, sanitized_text, re.IGNORECASE):
            raise ValidationError(
                message=f"{field_name} contains potentially dangerous content",
                field=field_name
            )

    return sanitized_text


def validate_user_id(user_id: str) -> bool:
    """
    Validate user ID format.

    Args:
        user_id: The user ID to validate

    Returns:
        True if user ID is valid, False otherwise

    Raises:
        ValidationError: If user ID is invalid
    """
    if not user_id or not isinstance(user_id, str):
        raise ValidationError(
            message="User ID must be a non-empty string",
            field="user_id"
        )

    # Basic validation - user ID should not be too long and should have valid characters
    if len(user_id) > 100:
        raise ValidationError(
            message="User ID is too long (max 100 characters)",
            field="user_id",
            details={"max_length": 100, "actual_length": len(user_id)}
        )

    # Check for valid characters (alphanumeric, hyphens, underscores, dots)
    if not re.match(r'^[a-zA-Z0-9._-]+$', user_id):
        raise ValidationError(
            message="User ID contains invalid characters",
            field="user_id",
            details={"valid_pattern": "alphanumeric, hyphens, underscores, dots"}
        )

    return True


def validate_api_key_format(api_key: str) -> bool:
    """
    Validate API key format.

    Args:
        api_key: The API key to validate

    Returns:
        True if API key format is valid, False otherwise

    Raises:
        ValidationError: If API key format is invalid
    """
    if not api_key or not isinstance(api_key, str):
        raise ValidationError(
            message="API key must be a non-empty string",
            field="api_key"
        )

    # Basic format check - should be at least 20 characters for security
    if len(api_key) < 20:
        raise ValidationError(
            message="API key is too short (min 20 characters)",
            field="api_key",
            details={"min_length": 20, "actual_length": len(api_key)}
        )

    # Should not contain spaces
    if ' ' in api_key:
        raise ValidationError(
            message="API key should not contain spaces",
            field="api_key"
        )

    return True