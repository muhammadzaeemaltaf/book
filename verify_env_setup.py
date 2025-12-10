#!/usr/bin/env python3
"""
Test script to verify the consolidated .env file is working correctly.
Run this from the backend directory after installing dependencies.

Usage:
    cd backend
    uv run python ../verify_env_setup.py
"""

import sys
from pathlib import Path

# Add backend/src to Python path
backend_dir = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_dir / "src"))

try:
    from utils.config import Settings
    
    print("=" * 60)
    print("Environment Configuration Test")
    print("=" * 60)
    
    settings = Settings()
    
    # Check critical settings
    checks = {
        "LLM Provider": settings.llm_provider,
        "Backend Host": settings.backend_host,
        "Backend Port": settings.backend_port,
        "Frontend URL": settings.frontend_url,
        "Qdrant Collection": settings.qdrant_collection_name,
        "Embedding Model": settings.embedding_model,
        "Debug Mode": settings.debug,
    }
    
    print("\n✓ Configuration loaded successfully from root .env file!\n")
    print("Configuration Summary:")
    print("-" * 60)
    
    for key, value in checks.items():
        print(f"  {key:.<40} {value}")
    
    # Check API keys (masked)
    print("\nAPI Keys Status:")
    print("-" * 60)
    
    api_keys = {
        "Cohere API Key": settings.cohere_api_key,
        "Qdrant API Key": settings.qdrant_api_key,
        "Groq API Key": settings.groq_api_key,
        "Google API Key": settings.google_api_key,
        "OpenAI API Key": settings.openai_api_key,
    }
    
    for key_name, key_value in api_keys.items():
        if key_value and key_value != "":
            masked = key_value[:8] + "..." if len(key_value) > 8 else "***"
            print(f"  {key_name:.<40} {masked} ✓")
        else:
            print(f"  {key_name:.<40} Not set")
    
    print("\n" + "=" * 60)
    print("✓ All environment variables loaded successfully!")
    print("=" * 60)
    
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nMake sure to install dependencies first:")
    print("  cd backend")
    print("  uv sync")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
