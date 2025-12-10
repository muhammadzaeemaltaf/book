"""
Book Agent - Handles book-related questions with RAG retrieval and fallback to general knowledge.
"""
from typing import List, Optional, Tuple
from ..models.chat import ChatMode
from ..services.qdrant_service import qdrant_service
from ..services.embedding_service import embedding_service
from .rag_retrieval_agent import rag_retrieval_agent
from ..utils.logging import get_logger

logger = get_logger("book_agent")

class BookAgent:
    """
    Agent responsible for answering book-related questions.
    Uses RAG retrieval first, falls back to general knowledge if no context found.
    """
    
    def __init__(self):
        """Initialize the Book Agent."""
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service
        self.retrieval_agent = rag_retrieval_agent
        
        # Topics covered by the book
        self.book_topics = [
            "ros2", "ros", "robot operating system", "robotics",
            "physical ai", "humanoid robotics", "humanoid robot",
            "isaac sim", "isaac", "nvidia isaac",
            "digital twin", "simulation", "gazebo", "unity",
            "vla", "vision language action", "vision-language-action",
            "urdf", "robot description", "nodes", "topics", "services",
            "launch files", "packages", "workspaces",
            "sensor", "actuator", "control", "navigation",
            "perception", "computer vision", "lidar", "camera",
            "reinforcement learning", "machine learning", "ai",
            "trajectory planning", "motion planning", "path planning",
            "kinematics", "dynamics", "manipulation"
        ]
    
    def should_handle(self, query: str) -> bool:
        """
        Determine if this agent should handle the query.
        Book agent handles all non-greeting queries.
        
        Args:
            query: The user's query
            
        Returns:
            True (book agent handles all non-greeting queries)
        """
        return True
    
    def is_book_related(self, query: str) -> bool:
        """
        Check if the query is related to book topics.
        
        Args:
            query: The user's query
            
        Returns:
            True if query is related to book topics, False otherwise
        """
        query_lower = query.lower()
        
        # Check if any book topic is mentioned in the query
        for topic in self.book_topics:
            if topic in query_lower:
                return True
        
        # Check for common robotics/AI question patterns
        robotics_patterns = [
            "robot", "ai", "artificial intelligence", "learning",
            "sensor", "actuator", "control", "program", "code",
            "model", "train", "neural", "network", "algorithm"
        ]
        
        for pattern in robotics_patterns:
            if pattern in query_lower:
                return True
        
        return False
    
    async def retrieve_context(
        self,
        query: str,
        selected_text: Optional[str] = None,
        chat_mode: ChatMode = ChatMode.NORMAL_QA,
        top_k: int = 5
    ) -> Tuple[List[str], float]:
        """
        Retrieve context from the book using RAG.
        
        Args:
            query: The user's query
            selected_text: Text selected by the user (if any)
            chat_mode: The chat mode
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (context_texts, confidence_score)
        """
        try:
            # Use the RAG retrieval agent to get context
            retrieval_response = await self.retrieval_agent.retrieve_context(
                query=query,
                selected_text=selected_text,
                chat_mode=chat_mode,
                top_k=top_k
            )
            
            # Extract context texts
            context_texts = [result.content for result in retrieval_response.results]
            
            return context_texts, retrieval_response.confidence
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return [], 0.0
    
    def get_system_prompt(
        self,
        query: str,
        context: List[str],
        chat_mode: ChatMode,
        selected_text: Optional[str] = None
    ) -> str:
        """
        Get the system prompt based on available context and mode.
        
        Args:
            query: The user's query
            context: Retrieved context from the book
            chat_mode: The chat mode
            selected_text: Text selected by the user (if any)
            
        Returns:
            System prompt for the LLM
        """
        if chat_mode == ChatMode.SELECTED_TEXT:
            # Selected text mode - use only provided context
            return (
                "You are a helpful assistant that answers questions based ONLY on the provided text "
                "from the 'Guide to Physical AI & Humanoid Robotics' book. "
                "Do not use any external knowledge or make assumptions beyond what is explicitly stated "
                "in the provided context. If the provided text does not contain information to answer "
                "the question, say so clearly and politely."
            )
        
        elif context and len(context) > 0:
            # Context found in the book
            return (
                "You are a helpful assistant that answers questions based on the provided context "
                "from the 'Guide to Physical AI & Humanoid Robotics' book. "
                "Use the information in the provided context to answer the question accurately and comprehensively. "
                "If the context doesn't fully answer the question but is related, provide what information is available "
                "and acknowledge any limitations."
            )
        
        elif self.is_book_related(query):
            # No context found, but query is book-related
            return (
                "You are an AI assistant for the 'Guide to Physical AI & Humanoid Robotics' book. "
                "No specific content from the book was found for this query, but the question appears to be related "
                "to robotics, AI, physical AI, or humanoid robotics. "
                "Provide a helpful answer using your general knowledge that would be consistent with a comprehensive "
                "textbook on physical AI and humanoid robotics. "
                "Be informative and educational in your response. "
                "If appropriate, mention that this is general knowledge since the specific topic wasn't found in the book's indexed content."
            )
        
        else:
            # Query is not book-related
            return (
                "You are an AI assistant for the 'Guide to Physical AI & Humanoid Robotics' book. "
                "The question doesn't appear to be related to the book's topics (ROS2, digital twins, "
                "Isaac Sim, vision-language-action models, robotics, or physical AI). "
                "Politely acknowledge that the question is outside the scope of the book and offer to help "
                "with questions about physical AI, humanoid robotics, or related topics."
            )
    
    def get_user_message(
        self,
        query: str,
        context: List[str],
        chat_mode: ChatMode
    ) -> str:
        """
        Get the user message formatted with context.
        
        Args:
            query: The user's query
            context: Retrieved context from the book
            chat_mode: The chat mode
            
        Returns:
            Formatted user message
        """
        if context and len(context) > 0:
            context_str = "\n\n---\n\n".join(context)
            return (
                f"Context from the book:\n\n{context_str}\n\n"
                f"---\n\n"
                f"Question: {query}\n\n"
                f"Please provide a helpful and comprehensive answer based on the context above."
            )
        else:
            if self.is_book_related(query):
                return (
                    f"Question: {query}\n\n"
                    f"Note: No specific content from the book was found for this query. "
                    f"If this question is related to robotics, AI, physical AI, or humanoid robotics, "
                    f"please provide a helpful answer based on general knowledge that would be consistent "
                    f"with a textbook on this subject."
                )
            else:
                return f"Question: {query}"


# Global instance
book_agent = BookAgent()
