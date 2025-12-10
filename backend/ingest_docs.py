"""
Script to run the ingestion process for Docusaurus documentation
"""
import asyncio
import os
from pathlib import Path
import time

# Add the src directory to the path so we can import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.document import IngestionRequest
from src.services.ingestion_service import IngestionService

async def run_ingestion():
    print("Starting Docusaurus documentation ingestion...")

    # Create ingestion service
    ingestion_service = IngestionService()

    # Define the source path - going up one directory to find the docs
    docs_path = Path(__file__).parent.parent / "docs"

    if not docs_path.exists():
        print(f"Docs directory not found at: {docs_path}")
        print("Looking for documentation directories...")

        # Look for common documentation directories
        possible_paths = [
            Path(__file__).parent.parent / "src" / "docs",
            Path(__file__).parent.parent / "documentation",
            Path(__file__).parent.parent / "website" / "docs",
            Path(__file__).parent.parent,  # Check root directory
        ]

        for path in possible_paths:
            if path.exists():
                print(f"Found documentation directory: {path}")
                docs_path = path
                break
        else:
            print("No documentation directory found. Please ensure you have a 'docs' directory with markdown files.")
            return

    print(f"Using documentation source: {docs_path}")

    # Create ingestion request
    request = IngestionRequest(
        source_path=str(docs_path),
        chunk_size=512,  # Reduced chunk size to generate fewer embeddings
        overlap=50,
        recursive=True
    )

    print("Starting ingestion process...")
    print("Note: This may take a while due to API rate limits...")

    response = await ingestion_service.ingest_documents(request)

    print(f"Ingestion completed with status: {response.status}")
    print(f"Processed {response.processed_count} chunks")
    print(f"Message: {response.message}")
    if response.duration_seconds:
        print(f"Duration: {response.duration_seconds:.2f} seconds")

    return response

if __name__ == "__main__":
    asyncio.run(run_ingestion())