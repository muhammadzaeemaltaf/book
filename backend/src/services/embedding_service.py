import asyncio
from typing import List, Optional
from cohere import Client
from ..utils.config import settings
from ..utils.logging import get_logger

logger = get_logger("embedding_service")

class EmbeddingService:
    """Service class for handling text embeddings using Cohere."""

    def __init__(self):
        """Initialize the Cohere client."""
        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")

        self.client = Client(api_key=settings.cohere_api_key)
        self.model = settings.embedding_model
        self.batch_size = 96  # Cohere's recommended batch size

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate a single embedding for the given text."""
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_document"  # Using search_document for document chunks
            )

            embedding = response.embeddings[0]
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {str(e)}")
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        all_embeddings = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type="search_document"  # Using search_document for document chunks
                )

                batch_embeddings = response.embeddings
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Generated embeddings for batch of {len(batch)} texts")
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {str(e)}")
                raise

        return all_embeddings

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate an embedding for a query (with different input type)."""
        try:
            response = self.client.embed(
                texts=[query],
                model=self.model,
                input_type="search_query"  # Using search_query for search queries
            )

            embedding = response.embeddings[0]
            logger.debug(f"Generated query embedding for: {query[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise

    async def get_embedding_dimensions(self) -> int:
        """Get the expected dimensions for embeddings from this model."""
        # Test with a short text to determine dimensions
        test_embedding = await self.generate_embedding("test")
        return len(test_embedding)

    async def validate_text_for_embedding(self, text: str) -> bool:
        """Validate if text is appropriate for embedding generation."""
        if not text or len(text.strip()) == 0:
            return False

        # Cohere has a limit of about 4096 tokens, but we'll be more conservative
        # to ensure good performance
        if len(text) > 3000:  # Rough character limit
            logger.warning(f"Text length {len(text)} may be too long for optimal embedding")

        return True

    async def embed_document_chunks(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings specifically for document chunks."""
        if not chunks:
            return []

        # Validate all chunks first
        validated_chunks = []
        for chunk in chunks:
            if await self.validate_text_for_embedding(chunk):
                validated_chunks.append(chunk)
            else:
                logger.warning(f"Skipping invalid chunk: {chunk[:100]}...")

        if not validated_chunks:
            return []

        return await self.generate_embeddings(validated_chunks)

    async def calculate_similarity(self,
                                 embedding1: List[float],
                                 embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

        # Calculate magnitudes
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)

        # Ensure the result is in [-1, 1] range (due to floating point errors)
        similarity = max(-1.0, min(1.0, similarity))

        return similarity


# Global instance
embedding_service = EmbeddingService()