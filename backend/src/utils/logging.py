import logging
import sys
from typing import Optional
from logging import Logger
from datetime import datetime

def setup_logging(level: Optional[str] = None) -> Logger:
    """
    Set up logging configuration for the application.

    Args:
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    # Determine logging level
    log_level = getattr(logging, level or 'INFO')

    # Create logger
    logger = logging.getLogger('rag_chatbot')
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Create file handler
    file_handler = logging.FileHandler('rag_chatbot.log')
    file_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def get_logger(name: Optional[str] = None) -> Logger:
    """
    Get a logger instance, optionally with a specific name.

    Args:
        name: Optional name for the logger (will be prefixed with 'rag_chatbot.')

    Returns:
        Logger instance
    """
    if name:
        logger_name = f'rag_chatbot.{name}'
    else:
        logger_name = 'rag_chatbot'

    return logging.getLogger(logger_name)

# Initialize the main logger
logger = setup_logging()

def log_api_call(endpoint: str, method: str, duration: float, user_id: Optional[str] = None):
    """
    Log API calls with performance metrics.

    Args:
        endpoint: API endpoint that was called
        method: HTTP method used
        duration: Time taken for the call in seconds
        user_id: Optional user identifier
    """
    logger.info(f"API Call: {method} {endpoint} | Duration: {duration:.2f}s | User: {user_id or 'unknown'}")

def log_error(error: Exception, context: str = ""):
    """
    Log error with context information.

    Args:
        error: Exception that occurred
        context: Context where the error occurred
    """
    logger.error(f"Error in {context}: {str(error)}", exc_info=True)

def log_retrieval(query: str, results_count: int, duration: float):
    """
    Log retrieval operations.

    Args:
        query: Query that was processed
        results_count: Number of results returned
        duration: Time taken for retrieval in seconds
    """
    logger.info(f"Retrieval: Query '{query[:50]}...' | Results: {results_count} | Duration: {duration:.2f}s")