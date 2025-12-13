"""
Sanitization utilities for the Physical AI textbook platform.
This module provides functions for sanitizing user input and other data.
"""
import html
import re
from typing import Union, List, Dict, Any


def sanitize_string(value: str) -> str:
    """
    Sanitize a string by escaping HTML and removing dangerous content.

    Args:
        value: The string to sanitize

    Returns:
        Sanitized string
    """
    if value is None:
        return None

    # First, escape HTML entities
    sanitized = html.escape(value)

    # Remove null bytes which can be problematic
    sanitized = sanitized.replace('\x00', '')

    # Remove other potentially dangerous characters if needed
    # This is a basic implementation - expand based on your specific requirements

    return sanitized


def sanitize_dict(data: Dict[str, Any], fields_to_sanitize: List[str] = None) -> Dict[str, Any]:
    """
    Sanitize specific fields in a dictionary.

    Args:
        data: The dictionary to sanitize
        fields_to_sanitize: List of field names to sanitize. If None, sanitizes all string values.

    Returns:
        Dictionary with sanitized values
    """
    sanitized_data = {}

    for key, value in data.items():
        if fields_to_sanitize is None or key in fields_to_sanitize:
            if isinstance(value, str):
                sanitized_data[key] = sanitize_string(value)
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized_data[key] = sanitize_dict(value)
            elif isinstance(value, list):
                # Sanitize each item in the list
                sanitized_data[key] = sanitize_list(value)
            else:
                sanitized_data[key] = value
        else:
            sanitized_data[key] = value

    return sanitized_data


def sanitize_list(data: List[Any]) -> List[Any]:
    """
    Sanitize items in a list.

    Args:
        data: The list to sanitize

    Returns:
        List with sanitized items
    """
    sanitized_list = []

    for item in data:
        if isinstance(item, str):
            sanitized_list.append(sanitize_string(item))
        elif isinstance(item, dict):
            sanitized_list.append(sanitize_dict(item))
        elif isinstance(item, list):
            sanitized_list.append(sanitize_list(item))
        else:
            sanitized_list.append(item)

    return sanitized_list


def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from text while preserving the content.

    Args:
        text: The text to strip HTML tags from

    Returns:
        Text with HTML tags removed
    """
    if not text:
        return text

    # Use regex to remove HTML tags
    clean_text = re.sub(r'<[^>]*>', '', text)

    # Unescape any HTML entities that might have been in the original text
    clean_text = html.unescape(clean_text)

    return clean_text


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent directory traversal and other attacks.

    Args:
        filename: The filename to sanitize

    Returns:
        Sanitized filename
    """
    if not filename:
        return filename

    # Remove path traversal attempts
    filename = filename.replace('../', '').replace('..\\', '')

    # Only allow alphanumeric characters, dots, hyphens, and underscores
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

    # Limit length to prevent issues
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        if ext:
            filename = name[:254 - len(ext)] + '.' + ext
        else:
            filename = name[:255]

    return filename


def sanitize_url(url: str) -> str:
    """
    Sanitize a URL to prevent open redirect and other attacks.

    Args:
        url: The URL to sanitize

    Returns:
        Sanitized URL or None if invalid
    """
    if not url:
        return url

    # Basic URL validation to prevent open redirects
    # Only allow http, https, and relative URLs
    if not re.match(r'^(/|https?://)', url):
        return None

    # Prevent javascript: and data: URLs
    if re.match(r'^(javascript:|data:|vbscript:)', url, re.IGNORECASE):
        return None

    return url


def sanitize_for_sql(value: Union[str, int, float]) -> Union[str, int, float]:
    """
    Sanitize values for use in SQL queries to prevent injection.
    Note: This is not a complete solution - always use parameterized queries.

    Args:
        value: The value to sanitize

    Returns:
        Sanitized value
    """
    if isinstance(value, str):
        # Escape single quotes and other SQL metacharacters
        # This is a basic implementation - use parameterized queries in production
        return value.replace("'", "''")
    return value


def sanitize_json_input(data: Union[Dict, List]) -> Union[Dict, List]:
    """
    Sanitize JSON input recursively.

    Args:
        data: The JSON data to sanitize

    Returns:
        Sanitized JSON data
    """
    if isinstance(data, dict):
        return sanitize_dict(data)
    elif isinstance(data, list):
        return sanitize_list(data)
    else:
        return data