import asyncio
from typing import List, Dict, Any, Optional, Union
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from ..utils.config import settings
from ..utils.logging import get_logger
from ..models.document import DocumentChunk

logger = get_logger("qdrant_service")

class QdrantService:
    """Service class for handling Qdrant vector database operations."""

    def __init__(self):
        """Initialize Qdrant client and connect to the database."""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            # Prefer GRPC for better performance if available
            prefer_grpc=True
        )
        self.collection_name = settings.qdrant_collection_name
        self._initialized = False

    async def initialize(self):
        """Initialize the Qdrant service by creating the collection if it doesn't exist."""
        if self._initialized:
            return

        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                # Create collection with appropriate vector size and distance metric
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_size,  # 1024 for multilingual model
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")

            self._initialized = True
            logger.info(f"Qdrant service initialized for collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant service: {str(e)}")
            raise

    async def store_embedding(self,
                            text: str,
                            embedding: List[float],
                            doc_id: str,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a text embedding in Qdrant."""
        if not self._initialized:
            await self.initialize()

        try:
            point_id = str(uuid4())
            payload = {
                "text": text,
                "doc_id": doc_id,
                **(metadata or {})
            }

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.debug(f"Stored embedding for doc {doc_id} with point_id {point_id}")
            return point_id
        except Exception as e:
            logger.error(f"Failed to store embedding for doc {doc_id}: {str(e)}")
            raise

    async def batch_store_embeddings(self,
                                   texts: List[str],
                                   embeddings: List[List[float]],
                                   doc_ids: List[str],
                                   metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Store multiple text embeddings in Qdrant."""
        if not self._initialized:
            await self.initialize()

        if len(texts) != len(embeddings) or len(texts) != len(doc_ids):
            raise ValueError("Texts, embeddings, and doc_ids must have the same length")

        try:
            points = []
            point_ids = []

            for i, (text, embedding, doc_id) in enumerate(zip(texts, embeddings, doc_ids)):
                point_id = str(uuid4())
                point_ids.append(point_id)

                payload = {
                    "text": text,
                    "doc_id": doc_id
                }

                if metadatas and i < len(metadatas):
                    payload.update(metadatas[i])

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.debug(f"Stored {len(points)} embeddings in batch")
            return point_ids
        except Exception as e:
            logger.error(f"Failed to batch store embeddings: {str(e)}")
            raise

    async def search_similar(self,
                           query_embedding: List[float],
                           top_k: int = 5,
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Qdrant."""
        if not self._initialized:
            await self.initialize()

        try:
            # Convert filters to Qdrant filter format if provided
            qdrant_filters = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
                    elif isinstance(value, list):
                        # Handle list of values (OR condition)
                        should_conditions = [
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=v)
                            )
                            for v in value
                        ]
                        filter_conditions.append(
                            models.Filter(
                                should=should_conditions
                            )
                        )

                if filter_conditions:
                    qdrant_filters = models.Filter(
                        must=filter_conditions
                    )

            # For Qdrant client, use the search method
            # The QdrantClient uses 'search' for versions >= 1.7.0
            # For older versions, you may need to use 'query_points' or 'search_batch'
            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    query_filter=qdrant_filters
                )
            except AttributeError:
                # Fallback for older versions or different client configurations
                logger.warning("'search' method not found, trying 'query_points'")
                try:
                    from qdrant_client.http.models import SearchRequest
                    results = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_embedding,
                        limit=top_k,
                        query_filter=qdrant_filters
                    ).points
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {str(fallback_error)}")
                    return []

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "content": result.payload.get("text", ""),
                    "score": result.score,
                    "source_document": result.payload.get("doc_id", ""),
                    "metadata": {k: v for k, v in result.payload.items()
                               if k not in ["text", "doc_id"]}
                })

            logger.debug(f"Found {len(formatted_results)} similar results")
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {str(e)}")
            raise

    async def get_by_id(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific point by its ID."""
        if not self._initialized:
            await self.initialize()

        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )

            if points and len(points) > 0:
                point = points[0]
                return {
                    "id": point.id,
                    "content": point.payload.get("text", ""),
                    "source_document": point.payload.get("doc_id", ""),
                    "metadata": {k: v for k, v in point.payload.items()
                               if k not in ["text", "doc_id"]}
                }
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve point {point_id}: {str(e)}")
            raise

    async def delete_by_id(self, point_id: str) -> bool:
        """Delete a specific point by its ID."""
        if not self._initialized:
            await self.initialize()

        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id]
                )
            )
            logger.debug(f"Deleted point {point_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete point {point_id}: {str(e)}")
            return False

    async def delete_by_document_id(self, doc_id: str) -> int:
        """Delete all points associated with a specific document ID."""
        if not self._initialized:
            await self.initialize()

        try:
            # Find all points with the given document ID
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id)
                        )
                    ]
                ),
                limit=10000  # Adjust as needed
            )

            point_ids = [point.id for point in scroll_result[0]]

            if point_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )
                logger.debug(f"Deleted {len(point_ids)} points for document {doc_id}")
                return len(point_ids)

            return 0
        except Exception as e:
            logger.error(f"Failed to delete points for document {doc_id}: {str(e)}")
            raise

    async def get_total_count(self) -> int:
        """Get the total number of points in the collection."""
        if not self._initialized:
            await self.initialize()

        try:
            count = self.client.count(
                collection_name=self.collection_name
            )
            return count.count
        except Exception as e:
            logger.error(f"Failed to get total count: {str(e)}")
            raise

    async def health_check(self) -> bool:
        """Check if Qdrant service is healthy."""
        try:
            # Try to get collection info as a basic health check
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False


# Global instance
qdrant_service = QdrantService()