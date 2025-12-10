"""
Document Ingestion Service for RAG Pipeline

This service handles the processing of documents for the RAG system,
including parsing, chunking, and storing in the vector database.
"""
import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

from ..models.document import DocumentChunk, IngestionRequest, IngestionResponse, IngestionPipeline
from ..services.qdrant_service import QdrantService
from ..services.embedding_service import EmbeddingService
from ..utils.config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

class IngestionService:
    def __init__(self):
        self.qdrant_service = QdrantService()
        self.embedding_service = EmbeddingService()
        self.pipeline_id_counter = 0

    async def ingest_documents(
        self,
        request: IngestionRequest
    ) -> IngestionResponse:
        """
        Main method to ingest documents from a source path
        """
        start_time = datetime.now()
        logger.info(f"Starting ingestion process for source: {request.source_path}")

        try:
            # Validate source path
            source_path = Path(request.source_path)
            if not source_path.exists():
                raise ValueError(f"Source path does not exist: {request.source_path}")

            # Parse documents based on source type
            if source_path.is_file():
                documents = await self._parse_single_document(source_path)
            else:
                documents = await self._parse_document_directory(
                    source_path,
                    recursive=request.recursive or True
                )

            # Process and chunk documents
            all_chunks = []
            for doc in documents:
                chunks = await self._chunk_document(
                    doc,
                    chunk_size=request.chunk_size or settings.DEFAULT_CHUNK_SIZE,
                    overlap=request.overlap or settings.DEFAULT_OVERLAP
                )
                all_chunks.extend(chunks)

            # Generate embeddings and store in vector database
            import time
            processed_count = 0
            for chunk in all_chunks:
                try:
                    # Generate embedding for the chunk using the correct method
                    embedding = await self.embedding_service.generate_embedding(chunk.content)

                    # Store in Qdrant using the correct method
                    await self.qdrant_service.store_embedding(
                        text=chunk.content,
                        embedding=embedding,
                        doc_id=chunk.chunk_id,
                        metadata=chunk.metadata
                    )
                    processed_count += 1

                    logger.debug(f"Processed chunk {chunk.chunk_id}: {len(chunk.content)} chars")

                    # Add a small delay to respect API rate limits
                    # This is especially important for Cohere's free tier
                    await asyncio.sleep(0.1)  # 100ms delay between chunks
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error processing chunk {chunk.chunk_id}: {error_msg}")

                    # If it's a rate limit error, wait longer before continuing
                    if "429" in error_msg or "rate" in error_msg.lower() or "limit" in error_msg.lower():
                        logger.info("Rate limit reached, waiting 60 seconds before continuing...")
                        await asyncio.sleep(60)  # Wait 60 seconds for rate limit to reset
                        continue  # Try to continue with next chunk after delay
                    # Continue with other chunks even if one fails

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            response = IngestionResponse(
                status="completed",
                processed_count=processed_count,
                message=f"Successfully ingested {processed_count} chunks from {len(documents)} documents",
                pipeline_id=f"ingest_{int(start_time.timestamp())}",
                duration_seconds=duration
            )

            logger.info(f"Ingestion completed: {response.message}")
            return response

        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
            return IngestionResponse(
                status="failed",
                processed_count=0,
                message=f"Ingestion failed: {str(e)}",
                pipeline_id=f"ingest_{int(start_time.timestamp())}_failed"
            )

    async def _parse_single_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a single document file
        """
        logger.info(f"Parsing single document: {file_path}")

        if file_path.suffix.lower() in ['.md', '.markdown']:
            return await self._parse_markdown(file_path)
        elif file_path.suffix.lower() in ['.txt']:
            return await self._parse_text(file_path)
        elif file_path.suffix.lower() in ['.pdf']:
            return await self._parse_pdf(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []

    async def _parse_document_directory(
        self,
        directory_path: Path,
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Parse all supported documents in a directory
        """
        logger.info(f"Parsing documents in directory: {directory_path}, recursive: {recursive}")

        documents = []
        patterns = ["**/*.md", "**/*.markdown", "**/*.txt"] if recursive else ["*.md", "*.markdown", "*.txt"]

        for pattern in patterns if recursive else ["*.md", "*.markdown", "*.txt"]:
            for file_path in directory_path.glob(pattern if recursive else pattern):
                if file_path.is_file():
                    docs = await self._parse_single_document(file_path)
                    documents.extend(docs)

        return documents

    async def _parse_markdown(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a markdown file and extract content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Extract title from markdown if present
            title = self._extract_title_from_markdown(content)

            return [{
                'content': content,
                'title': title,
                'source_file': str(file_path),
                'file_type': 'markdown',
                'metadata': {
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'file_size': file_path.stat().st_size,
                    'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }]
        except Exception as e:
            logger.error(f"Error parsing markdown file {file_path}: {str(e)}")
            return []

    async def _parse_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a text file and extract content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            return [{
                'content': content,
                'title': file_path.stem,
                'source_file': str(file_path),
                'file_type': 'text',
                'metadata': {
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'file_size': file_path.stat().st_size,
                    'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }]
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {str(e)}")
            return []

    async def _parse_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a PDF file and extract content
        """
        try:
            # For now, we'll return empty as PDF parsing requires additional dependencies
            # In a real implementation, you would use PyPDF2, pdfplumber, or similar
            logger.warning(f"PDF parsing not implemented for {file_path}, skipping")
            return []
        except Exception as e:
            logger.error(f"Error parsing PDF file {file_path}: {str(e)}")
            return []

    def _extract_title_from_markdown(self, content: str) -> str:
        """
        Extract title from markdown content (first H1 or filename)
        """
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('# '):
                return line.strip()[2:].strip()  # Remove '# ' prefix
        return "Untitled Document"

    async def _chunk_document(
        self,
        document: Dict[str, Any],
        chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces for embedding
        """
        content = document['content']
        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            # If we're near the end, make sure we include the rest
            if end > len(content):
                end = len(content)

            # Extract the chunk
            chunk_text = content[start:end]

            # Create a chunk with overlap if not at the end
            if end < len(content) and overlap > 0:
                overlap_end = min(end + overlap, len(content))
                chunk_text = content[start:overlap_end]

            chunk = DocumentChunk(
                chunk_id=f"chunk_{document['metadata']['file_name']}_{start}_{end}",
                content=chunk_text,
                source_document=document['source_file'],
                metadata={
                    **document['metadata'],
                    'chunk_start': start,
                    'chunk_end': end,
                    'chunk_index': len(chunks)
                }
            )

            chunks.append(chunk)
            start = end  # Move to the end of the current chunk (without double counting overlap)

        logger.debug(f"Document chunked into {len(chunks)} pieces")
        return chunks

    async def get_ingestion_status(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Get the status of an ingestion pipeline
        """
        # In a real implementation, this would check a database or cache
        # for the status of a specific ingestion job
        return {
            "pipeline_id": pipeline_id,
            "status": "completed",  # Simplified for now
            "progress": 100,
            "processed_chunks": 0,  # Would come from actual tracking
            "total_chunks": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "error": None
        }

    async def clear_ingested_content(self) -> bool:
        """
        Clear all ingested content from the vector database
        """
        try:
            # Clear all vectors from the collection
            await self.qdrant_service.clear_collection()
            logger.info("Successfully cleared all ingested content")
            return True
        except Exception as e:
            logger.error(f"Error clearing ingested content: {str(e)}")
            return False