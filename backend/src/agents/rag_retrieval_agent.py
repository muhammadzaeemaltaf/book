from typing import List, Dict, Any, Optional
from enum import Enum
from ..models.retrieval import (
    RetrievalMode, QueryNormalizationResult, RetrievalPlan,
    RetrievalResult, RetrievalResponse, ContextFilter
)
from ..models.chat import ChatMode
from ..services.qdrant_service import qdrant_service
from ..services.embedding_service import embedding_service
from ..utils.logging import get_logger
from ..utils.validators import validate_text_length
import re
import asyncio
from datetime import datetime

logger = get_logger("rag_retrieval_agent")

class RAGRetrievalAgent:
    """
    RAGRetrievalAgent: A dedicated subagent for handling all retrieval operations
    including query normalization, mode selection, ranking, filtering, and
    structured retrieval planning.
    """

    def __init__(self):
        """Initialize the RAG Retrieval Agent."""
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service

    async def normalize_query(self, query: str) -> QueryNormalizationResult:
        """
        Normalize the user query by cleaning, detecting language, and extracting entities.

        Args:
            query: The raw user query

        Returns:
            QueryNormalizationResult with normalized query and metadata
        """
        # Basic cleaning
        normalized_query = query.strip()

        # Detect language (simplified - in practice, use a proper language detection library)
        # For now, we'll just return None, but this could be enhanced
        detected_language = None

        # Extract entities using regex patterns (simplified)
        entities = self._extract_entities(normalized_query)

        # Extract keywords (simplified)
        keywords = self._extract_keywords(normalized_query)

        # Determine query type (simplified)
        query_type = self._determine_query_type(normalized_query)

        return QueryNormalizationResult(
            original_query=query,
            normalized_query=normalized_query,
            detected_language=detected_language,
            query_type=query_type,
            entities=entities,
            keywords=keywords
        )

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using regex patterns."""
        # This is a simplified entity extraction - in practice, use NER libraries
        patterns = [
            r'\b[A-Z][a-z]+\b',  # Proper nouns
            r'\b\d+\w*\b',       # Numbers with units
            r'#\w+',             # Hashtags
            r'@\w+',             # Mentions
        ]

        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(entities))

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction - remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'may', 'might', 'must', 'can', 'could', 'this', 'that', 'these', 'those'
        }

        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Remove duplicates while preserving order
        return list(dict.fromkeys(keywords))

    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query."""
        query_lower = query.lower().strip()

        # Check for different question types
        if any(word in query_lower for word in ['what', 'define', 'explain', 'describe']):
            return 'factual'
        elif any(word in query_lower for word in ['how', 'process', 'steps', 'method']):
            return 'procedural'
        elif any(word in query_lower for word in ['why', 'reason', 'because']):
            return 'conceptual'
        elif any(word in query_lower for word in ['when', 'date', 'time']):
            return 'temporal'
        elif any(word in query_lower for word in ['where', 'location', 'place']):
            return 'spatial'
        elif any(word in query_lower for word in ['compare', 'contrast', 'difference']):
            return 'comparative'
        else:
            return 'general'

    async def determine_retrieval_mode(self,
                                     query: str,
                                     selected_text: Optional[str] = None,
                                     chat_mode: ChatMode = ChatMode.NORMAL_QA) -> RetrievalMode:
        """
        Determine the appropriate retrieval mode based on query and context.

        Args:
            query: The user query
            selected_text: Text selected by the user (if any)
            chat_mode: The chat mode selected by the user

        Returns:
            RetrievalMode indicating the appropriate retrieval strategy
        """
        if chat_mode == ChatMode.SELECTED_TEXT or selected_text is not None:
            if selected_text and len(selected_text.strip()) > 0:
                return RetrievalMode.SELECTED_TEXT_ONLY

        # For now, default to vector search for normal Q&A
        # In a more complex system, you might implement logic to choose hybrid or keyword search
        return RetrievalMode.VECTOR_SEARCH

    async def create_retrieval_plan(self,
                                  query: str,
                                  selected_text: Optional[str] = None,
                                  chat_mode: ChatMode = ChatMode.NORMAL_QA,
                                  top_k: int = 5) -> RetrievalPlan:
        """
        Create a structured retrieval plan based on the query and context.

        Args:
            query: The user query
            selected_text: Text selected by the user (if any)
            chat_mode: The chat mode selected by the user
            top_k: Number of results to retrieve

        Returns:
            RetrievalPlan with the strategy for retrieval
        """
        # Normalize the query
        normalization_result = await self.normalize_query(query)

        # Determine the retrieval mode
        mode = await self.determine_retrieval_mode(query, selected_text, chat_mode)

        # Create search strategies based on mode
        search_strategies = []
        if mode == RetrievalMode.VECTOR_SEARCH:
            search_strategies = ["semantic_search", "vector_similarity"]
        elif mode == RetrievalMode.SELECTED_TEXT_ONLY:
            search_strategies = ["selected_text_processing"]
        elif mode == RetrievalMode.HYBRID_SEARCH:
            search_strategies = ["semantic_search", "keyword_search", "vector_similarity"]

        # Create filters (empty for now, but could be expanded)
        filters = {}

        return RetrievalPlan(
            query_id=f"query_{int(datetime.utcnow().timestamp() * 1000)}",
            mode=mode,
            normalization_result=normalization_result,
            search_strategies=search_strategies,
            filters=filters,
            top_k=top_k,
            min_score=0.3,
            created_at=datetime.utcnow()
        )

    async def execute_retrieval(self,
                              retrieval_plan: RetrievalPlan,
                              selected_text: Optional[str] = None) -> RetrievalResponse:
        """
        Execute the retrieval plan and return results.

        Args:
            retrieval_plan: The structured retrieval plan to execute
            selected_text: Text selected by the user (if any)

        Returns:
            RetrievalResponse with the retrieved results
        """
        start_time = datetime.utcnow()

        if retrieval_plan.mode == RetrievalMode.SELECTED_TEXT_ONLY:
            # For selected text mode, just return the selected text as context
            results = [
                RetrievalResult(
                    id="selected_text_0",
                    content=selected_text or "",
                    score=1.0,  # Perfect score since it's the exact text
                    source_document="selected_text",
                    metadata={"type": "selected_text"}
                )
            ] if selected_text else []
        else:
            # For vector search mode, retrieve from the vector database
            # Generate embedding for the query using the appropriate method
            query_embedding = await self.embedding_service.generate_query_embedding(
                retrieval_plan.normalization_result.normalized_query
            )

            # Perform the search in the vector database with ingested content
            raw_results = await self.qdrant_service.search_similar(
                query_embedding=query_embedding,
                top_k=retrieval_plan.top_k,
                filters=retrieval_plan.filters
            )

            # Convert raw results to RetrievalResult objects
            results = [
                RetrievalResult(
                    id=result["id"],
                    content=result["content"],
                    score=result["score"],
                    source_document=result["source_document"],
                    metadata=result["metadata"],
                    chunk_index=result.get("metadata", {}).get("chunk_index")
                )
                for result in raw_results
            ]

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000  # Convert to ms

        # Calculate confidence as average of scores if there are results
        confidence = 0.0
        if results:
            confidence = sum(r.score for r in results) / len(results)

        return RetrievalResponse(
            query_id=retrieval_plan.query_id,
            results=results,
            mode_used=retrieval_plan.mode,
            retrieved_count=len(results),
            processing_time_ms=processing_time,
            confidence=confidence
        )

    async def filter_context(self,
                           results: List[RetrievalResult],
                           filter_params: ContextFilter) -> List[RetrievalResult]:
        """
        Filter the retrieved context based on various criteria.

        Args:
            results: List of retrieval results to filter
            filter_params: Parameters for filtering

        Returns:
            Filtered list of retrieval results
        """
        filtered_results = results.copy()

        # Filter by minimum score
        if filter_params.min_score > 0:
            filtered_results = [r for r in filtered_results if r.score >= filter_params.min_score]

        # Filter by maximum length (simplified - just truncates content)
        if filter_params.max_length > 0:
            for result in filtered_results:
                if len(result.content) > filter_params.max_length:
                    result.content = result.content[:filter_params.max_length] + "..."

        # Deduplicate based on content similarity (simplified)
        if filter_params.deduplicate:
            seen_contents = set()
            deduplicated_results = []
            for result in filtered_results:
                content_key = result.content.strip().lower()[:100]  # First 100 chars as key
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    deduplicated_results.append(result)
            filtered_results = deduplicated_results

        # Filter by source if specified
        if filter_params.filter_by_source:
            filtered_results = [
                r for r in filtered_results
                if r.source_document in filter_params.filter_by_source
            ]

        # Filter by metadata if specified
        if filter_params.filter_by_metadata:
            for key, value in filter_params.filter_by_metadata.items():
                filtered_results = [
                    r for r in filtered_results
                    if r.metadata.get(key) == value
                ]

        return filtered_results

    async def retrieve_context(self,
                             query: str,
                             selected_text: Optional[str] = None,
                             chat_mode: ChatMode = ChatMode.NORMAL_QA,
                             top_k: int = 5,
                             filter_params: Optional[ContextFilter] = None) -> RetrievalResponse:
        """
        Main method to retrieve context for a query using the RAG approach.

        Args:
            query: The user query
            selected_text: Text selected by the user (if any)
            chat_mode: The chat mode selected by the user
            top_k: Number of results to retrieve
            filter_params: Parameters for filtering the results

        Returns:
            RetrievalResponse with the retrieved context
        """
        logger.info(f"Starting retrieval for query: {query[:50]}...")

        # Create retrieval plan
        retrieval_plan = await self.create_retrieval_plan(
            query=query,
            selected_text=selected_text,
            chat_mode=chat_mode,
            top_k=top_k
        )

        # Execute the retrieval
        response = await self.execute_retrieval(
            retrieval_plan=retrieval_plan,
            selected_text=selected_text
        )

        # Apply filtering if specified
        if filter_params:
            filtered_results = await self.filter_context(response.results, filter_params)
            response.results = filtered_results
            response.retrieved_count = len(filtered_results)

        logger.info(f"Retrieved {response.retrieved_count} results for query: {query[:50]}...")

        return response


# Global instance of the RAG Retrieval Agent
rag_retrieval_agent = RAGRetrievalAgent()