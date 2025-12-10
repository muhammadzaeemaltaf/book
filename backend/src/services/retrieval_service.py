from typing import List, Optional, Dict, Any
from datetime import datetime
import time
from ..models.retrieval import (
    RetrievalMode, RetrievalPlan, RetrievalResult,
    RetrievalResponse, ContextFilter, RetrievalMetrics
)
from ..models.chat import ChatMode
from ..agents.rag_retrieval_agent import rag_retrieval_agent
from ..utils.logging import get_logger, log_retrieval
from ..utils.config import settings

logger = get_logger("retrieval_service")

class RetrievalService:
    """
    Service for handling retrieval operations using the RAGRetrievalAgent.
    Provides a clean interface for retrieval functionality with additional
    business logic and metrics tracking.
    """

    def __init__(self):
        """Initialize the retrieval service."""
        self.agent = rag_retrieval_agent

    async def retrieve_context(self,
                             query: str,
                             selected_text: Optional[str] = None,
                             chat_mode: ChatMode = ChatMode.NORMAL_QA,
                             top_k: int = 5,
                             filter_params: Optional[ContextFilter] = None) -> RetrievalResponse:
        """
        Retrieve context for a query using the RAG approach.

        Args:
            query: The user query
            selected_text: Text selected by the user (if any)
            chat_mode: The chat mode selected by the user
            top_k: Number of results to retrieve
            filter_params: Parameters for filtering the results

        Returns:
            RetrievalResponse with the retrieved context
        """
        start_time = time.time()

        try:
            # Use the RAGRetrievalAgent to perform the retrieval
            response = await self.agent.retrieve_context(
                query=query,
                selected_text=selected_text,
                chat_mode=chat_mode,
                top_k=top_k,
                filter_params=filter_params
            )

            # Log the retrieval operation
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds
            log_retrieval(query, len(response.results), duration)

            # Add metrics to the response
            response.metrics = RetrievalMetrics(
                query_id=response.query_id,
                retrieval_time_ms=duration,
                vector_search_time_ms=duration,  # Placeholder - actual implementation would track this separately
                total_results=len(response.results),
                filtered_results=len(response.results),
                mode=response.mode_used
            )

            logger.info(f"Retrieved {len(response.results)} results for query: {query[:50]}...")
            return response

        except Exception as e:
            logger.error(f"Error in retrieval for query '{query[:50]}...': {str(e)}")
            raise

    async def create_and_execute_plan(self,
                                    query: str,
                                    selected_text: Optional[str] = None,
                                    chat_mode: ChatMode = ChatMode.NORMAL_QA,
                                    top_k: int = 5) -> RetrievalResponse:
        """
        Create a retrieval plan and execute it.

        Args:
            query: The user query
            selected_text: Text selected by the user (if any)
            chat_mode: The chat mode selected by the user
            top_k: Number of results to retrieve

        Returns:
            RetrievalResponse with the retrieved context
        """
        start_time = time.time()

        try:
            # Create a retrieval plan
            plan = await self.agent.create_retrieval_plan(
                query=query,
                selected_text=selected_text,
                chat_mode=chat_mode,
                top_k=top_k
            )

            logger.debug(f"Created retrieval plan with mode: {plan.mode}")

            # Execute the plan
            response = await self.agent.execute_retrieval(
                retrieval_plan=plan,
                selected_text=selected_text
            )

            # Log the retrieval operation
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds
            log_retrieval(query, len(response.results), duration)

            logger.info(f"Executed retrieval plan, got {len(response.results)} results for query: {query[:50]}...")
            return response

        except Exception as e:
            logger.error(f"Error executing retrieval plan for query '{query[:50]}...': {str(e)}")
            raise

    async def normalize_and_classify_query(self, query: str) -> Dict[str, Any]:
        """
        Normalize and classify a query using the agent's capabilities.

        Args:
            query: The raw user query

        Returns:
            Dictionary with normalized query and classification information
        """
        try:
            # Use the agent's query normalization capability
            normalization_result = await self.agent.normalize_query(query)

            return {
                "original_query": normalization_result.original_query,
                "normalized_query": normalization_result.normalized_query,
                "detected_language": normalization_result.detected_language,
                "query_type": normalization_result.query_type,
                "entities": normalization_result.entities,
                "keywords": normalization_result.keywords
            }
        except Exception as e:
            logger.error(f"Error normalizing query '{query[:50]}...': {str(e)}")
            raise

    async def determine_best_retrieval_mode(self,
                                          query: str,
                                          selected_text: Optional[str] = None,
                                          chat_mode: ChatMode = ChatMode.NORMAL_QA) -> RetrievalMode:
        """
        Determine the best retrieval mode for a given query and context.

        Args:
            query: The user query
            selected_text: Text selected by the user (if any)
            chat_mode: The chat mode selected by the user

        Returns:
            The recommended RetrievalMode
        """
        try:
            mode = await self.agent.determine_retrieval_mode(
                query=query,
                selected_text=selected_text,
                chat_mode=chat_mode
            )

            logger.debug(f"Determined retrieval mode '{mode}' for query: {query[:50]}...")
            return mode
        except Exception as e:
            logger.error(f"Error determining retrieval mode for query '{query[:50]}...': {str(e)}")
            # Default to vector search if there's an error
            return RetrievalMode.VECTOR_SEARCH

    async def filter_and_rank_results(self,
                                    results: List[RetrievalResult],
                                    query: str,
                                    filter_params: Optional[ContextFilter] = None) -> List[RetrievalResult]:
        """
        Filter and rank retrieval results based on various criteria.

        Args:
            results: List of retrieval results to process
            query: The original query (for potential reranking)
            filter_params: Parameters for filtering

        Returns:
            Filtered and ranked list of retrieval results
        """
        try:
            # Apply filters if provided
            if filter_params:
                filtered_results = await self.agent.filter_context(results, filter_params)
            else:
                # Use default filtering
                default_filter = ContextFilter(min_score=0.3, max_length=2000, deduplicate=True)
                filtered_results = await self.agent.filter_context(results, default_filter)

            # In a more advanced implementation, we might add reranking here
            # For now, we'll just return the filtered results
            logger.debug(f"Filtered {len(results)} results down to {len(filtered_results)} results")
            return filtered_results

        except Exception as e:
            logger.error(f"Error filtering results: {str(e)}")
            # Return original results if filtering fails
            return results

    async def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about retrieval operations (placeholder implementation).

        Returns:
            Dictionary with retrieval statistics
        """
        # This would typically connect to a metrics database or cache
        # For now, return placeholder values
        return {
            "total_retrievals": 0,
            "avg_retrieval_time_ms": 0.0,
            "most_common_modes": [],
            "success_rate": 1.0
        }


# Global instance
retrieval_service = RetrievalService()