"""
Comprehensive error logging utilities for the Physical AI textbook platform.
This module provides structured logging functionality for the application.
"""
import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime
import json
import traceback
from enum import Enum
import os


class LogLevel(str, Enum):
    """
    Log levels for the application.
    """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """
    Available log formats.
    """
    JSON = "json"
    TEXT = "text"


class AppLogger:
    """
    Application logger with structured logging capabilities.
    """
    def __init__(
        self,
        name: str = "physical_ai_app",
        level: LogLevel = LogLevel.INFO,
        format_type: LogFormat = LogFormat.JSON,
        log_file: Optional[str] = None
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))

        # Avoid adding handlers multiple times
        if not self.logger.handlers:
            # Create formatter based on format type
            if format_type == LogFormat.JSON:
                formatter = JSONFormatter()
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File handler if specified
            if log_file:
                # Create directory if it doesn't exist
                log_dir = os.path.dirname(log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)

                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

        self.name = name
        self.format_type = format_type

    def _log(
        self,
        level: LogLevel,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """
        Internal method to log messages with additional context.

        Args:
            level: The log level
            message: The log message
            extra: Additional context data
            exception: Exception to log (if any)
        """
        extra_data = extra or {}

        if exception:
            extra_data["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }

        if self.format_type == LogFormat.JSON:
            # For JSON format, we'll add structured data
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": level.value,
                "logger": self.name,
                "message": message,
                **extra_data
            }
            log_message = json.dumps(log_entry)
        else:
            # For text format, combine message and extra data
            if extra_data:
                extra_str = " | ".join([f"{k}={v}" for k, v in extra_data.items()])
                log_message = f"{message} | {extra_str}"
            else:
                log_message = message

        getattr(self.logger, level.lower())(log_message)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log a debug message."""
        self._log(LogLevel.DEBUG, message, extra)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log an info message."""
        self._log(LogLevel.INFO, message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log a warning message."""
        self._log(LogLevel.WARNING, message, extra)

    def error(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """Log an error message."""
        self._log(LogLevel.ERROR, message, extra, exception)

    def critical(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """Log a critical message."""
        self._log(LogLevel.CRITICAL, message, extra, exception)


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage'
            ]:
                log_entry[key] = value

        return json.dumps(log_entry)


# Global logger instances
app_logger = AppLogger()
auth_logger = AppLogger(name="auth", log_file="logs/auth.log")
api_logger = AppLogger(name="api", log_file="logs/api.log")
ai_logger = AppLogger(name="ai", log_file="logs/ai.log")
db_logger = AppLogger(name="database", log_file="logs/database.log")


def get_logger(name: str = "default") -> AppLogger:
    """
    Get a logger instance by name.

    Args:
        name: The name of the logger

    Returns:
        AppLogger instance
    """
    if name == "auth":
        return auth_logger
    elif name == "api":
        return api_logger
    elif name == "ai":
        return ai_logger
    elif name == "database":
        return db_logger
    else:
        return AppLogger(name=name)


def log_api_request(
    endpoint: str,
    method: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    response_time: Optional[float] = None,
    status_code: Optional[int] = None
):
    """
    Log API request details.

    Args:
        endpoint: The API endpoint
        method: The HTTP method
        user_id: The user ID (if authenticated)
        ip_address: The client IP address
        response_time: The response time in seconds
        status_code: The HTTP status code
    """
    extra = {
        "endpoint": endpoint,
        "method": method,
        "user_id": user_id,
        "ip_address": ip_address,
        "response_time_ms": int(response_time * 1000) if response_time else None,
        "status_code": status_code
    }

    api_logger.info("API request", extra=extra)


def log_auth_event(
    event_type: str,
    user_id: Optional[str] = None,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None
):
    """
    Log authentication-related events.

    Args:
        event_type: Type of auth event (login, logout, signup, etc.)
        user_id: The user ID
        success: Whether the event was successful
        details: Additional event details
    """
    extra = {
        "event_type": event_type,
        "user_id": user_id,
        "success": success,
        "details": details or {}
    }

    auth_logger.info(f"Auth event: {event_type}", extra=extra)


def log_ai_operation(
    operation: str,
    user_id: Optional[str] = None,
    chapter_id: Optional[str] = None,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None
):
    """
    Log AI-related operations.

    Args:
        operation: Type of AI operation (summarization, personalization, etc.)
        user_id: The user ID
        chapter_id: The chapter ID (if applicable)
        success: Whether the operation was successful
        details: Additional operation details
    """
    extra = {
        "operation": operation,
        "user_id": user_id,
        "chapter_id": chapter_id,
        "success": success,
        "details": details or {}
    }

    ai_logger.info(f"AI operation: {operation}", extra=extra)


def log_error(
    error: Exception,
    context: str,
    user_id: Optional[str] = None,
    endpoint: Optional[str] = None
):
    """
    Log an error with context.

    Args:
        error: The exception that occurred
        context: Context where the error occurred
        user_id: The user ID (if applicable)
        endpoint: The endpoint where error occurred (if applicable)
    """
    extra = {
        "context": context,
        "user_id": user_id,
        "endpoint": endpoint,
        "error_type": type(error).__name__
    }

    app_logger.error(
        f"Error in {context}: {str(error)}",
        extra=extra,
        exception=error
    )


def setup_logging_config():
    """
    Set up logging configuration for the application.
    This function configures logging according to best practices.
    """
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not root_logger.handlers:
        # Console handler with colorized output
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Set specific levels for third-party loggers to reduce noise
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# Initialize logging configuration
setup_logging_config()