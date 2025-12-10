from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from ...utils.config import settings

def setup_cors_middleware(app: FastAPI, additional_origins: List[str] = None):
    """
    Set up CORS middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        additional_origins: Additional origins to allow beyond the configured frontend URL
    """
    origins = [
        settings.frontend_url,  # Main frontend URL
        "http://localhost:3000",  # Common Docusaurus dev server
        "http://localhost:3001",  # Alternative Docusaurus port
        "http://127.0.0.1:3000",  # Alternative localhost format
        "http://127.0.0.1:3001",  # Alternative localhost format
        "http://localhost:8000",  # Self-origin for testing
        "http://localhost:8080",  # Common alternative port
        "https://*.vercel.app",   # Common deployment platform
        "https://*.github.io",    # GitHub Pages
    ]

    # Add any additional origins if provided
    if additional_origins:
        origins.extend(additional_origins)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=["*"],
        # Allow credentials for sessions/cookies if needed
        allow_credentials=True,
        # Expose headers that clients can access
        expose_headers=["Access-Control-Allow-Origin"]
    )

def add_cors_headers(response):
    """
    Helper function to add CORS headers to responses if needed.
    This is typically not needed when using the middleware but can be useful for custom handling.
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response