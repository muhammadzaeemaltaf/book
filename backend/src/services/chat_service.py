import asyncio
from typing import List, Dict, Any, AsyncGenerator
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from datetime import datetime
import time
import json
import re
from ..models.chat import ChatRequest, ChatResponseModel, ChatMode
from ..models.retrieval import ContextFilter
from ..agents.greeting_agent import greeting_agent
from ..agents.book_agent import book_agent
from ..utils.config import settings
from ..utils.logging import get_logger, log_retrieval
from ..utils.validators import validate_text_length

logger = get_logger("chat_service")

# Disable tracing for agents
set_tracing_disabled(disabled=True)

class ChatService:
    """Service for handling chat interactions and response generation."""

    def __init__(self):
        """Initialize the chat service with necessary clients and configurations."""
        llm_provider = settings.llm_provider.lower()
        
        if llm_provider == "groq":
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable is required")
            
            # Initialize Groq client using Agents SDK
            openai_client = AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url
            )
            model_name = settings.groq_model
            logger.info(f"Initialized Groq model: {model_name}")
            
        elif llm_provider == "gemini":
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is required")
            
            # Initialize Gemini client using Agents SDK
            openai_client = AsyncOpenAI(
                api_key=settings.google_api_key,
                base_url=settings.gemini_base_url
            )
            model_name = settings.gemini_model
            logger.info(f"Initialized Gemini model: {model_name}")
            
        elif llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            # Initialize OpenAI client using Agents SDK
            openai_client = AsyncOpenAI(
                api_key=settings.openai_api_key
            )
            model_name = settings.openai_model
            logger.info(f"Initialized OpenAI model: {model_name}")
            
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Use 'groq', 'gemini', or 'openai'")
        
        # Create the model using Agents SDK
        self.model = OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=openai_client
        )
        self.llm_provider = llm_provider
        

    async def process_chat_request(self, chat_request: ChatRequest) -> ChatResponseModel:
        """
        Process a chat request and return a response using the agent system.

        Args:
            chat_request: The incoming chat request

        Returns:
            ChatResponseModel with the generated response
        """
        start_time = time.time()

        try:
            # Step 1: Check if this is a greeting - use greeting agent
            if greeting_agent.should_handle(chat_request.message):
                logger.info("Routing to greeting agent")
                
                # Get system prompt from greeting agent
                system_message = greeting_agent.get_system_prompt(chat_request.message)
                user_message = chat_request.message
                
                # Generate response
                response_content = await self._call_llm(
                    system_message=system_message,
                    user_message=user_message
                )

                # Calculate token usage (approximate)
                prompt_tokens = len(chat_request.message) // 4
                completion_tokens = len(response_content) // 4

                # Create the response model
                response = ChatResponseModel(
                    id=f"resp_{int(datetime.utcnow().timestamp() * 1000)}",
                    message=response_content,
                    mode_used=chat_request.mode,
                    retrieved_context=[],
                    confidence=1.0,  # High confidence for greetings
                    created_at=datetime.utcnow(),
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                )

                # Log the interaction
                duration = (time.time() - start_time) * 1000
                log_retrieval(chat_request.message, 0, duration)

                return response
            
            # Step 2: Not a greeting - use book agent
            logger.info("Routing to book agent")
            
            # Retrieve context from the book
            context_texts, confidence = await book_agent.retrieve_context(
                query=chat_request.message,
                selected_text=chat_request.selected_text,
                chat_mode=chat_request.mode,
                top_k=chat_request.top_k
            )

            # Get system prompt and user message from book agent
            system_message = book_agent.get_system_prompt(
                query=chat_request.message,
                context=context_texts,
                chat_mode=chat_request.mode,
                selected_text=chat_request.selected_text
            )
            user_message = book_agent.get_user_message(
                query=chat_request.message,
                context=context_texts,
                chat_mode=chat_request.mode
            )

            # Generate the response using the LLM
            response_content = await self._call_llm(
                system_message=system_message,
                user_message=user_message
            )

            # Calculate token usage (approximate)
            prompt_tokens = (len(chat_request.message) + sum(len(ctx) for ctx in context_texts)) // 4
            completion_tokens = len(response_content) // 4

            # Create the response model
            response = ChatResponseModel(
                id=f"resp_{int(datetime.utcnow().timestamp() * 1000)}",
                message=response_content,
                mode_used=chat_request.mode,
                retrieved_context=context_texts,
                confidence=confidence if confidence > 0 else 0.7,  # Use retrieval confidence or default
                created_at=datetime.utcnow(),
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )

            # Log the retrieval
            duration = (time.time() - start_time) * 1000
            log_retrieval(chat_request.message, len(context_texts), duration)

            return response

        except Exception as e:
            logger.error(f"Error processing chat request: {str(e)}")
            raise

    async def process_chat_request_streaming(self, chat_request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        Process a chat request and return a streaming response using the agent system.

        Args:
            chat_request: The incoming chat request

        Yields:
            JSON strings representing streaming response chunks
        """
        start_time = time.time()

        try:
            # Yield start event
            start_chunk = {
                "type": "start",
                "message_id": f"stream_{int(datetime.utcnow().timestamp() * 1000)}"
            }
            yield json.dumps(start_chunk)

            # Step 1: Check if this is a greeting - use greeting agent
            if greeting_agent.should_handle(chat_request.message):
                logger.info("Routing to greeting agent (streaming)")
                
                # Get system prompt from greeting agent
                system_message = greeting_agent.get_system_prompt(chat_request.message)
                user_message = chat_request.message
                
                # Stream the response
                async for chunk in self._call_llm_streaming(
                    system_message=system_message,
                    user_message=user_message,
                    context_texts=[],
                    mode=chat_request.mode
                ):
                    yield chunk
            else:
                # Step 2: Not a greeting - use book agent
                logger.info("Routing to book agent (streaming)")
                
                # Retrieve context from the book
                context_texts, confidence = await book_agent.retrieve_context(
                    query=chat_request.message,
                    selected_text=chat_request.selected_text,
                    chat_mode=chat_request.mode,
                    top_k=chat_request.top_k
                )

                # Get system prompt and user message from book agent
                system_message = book_agent.get_system_prompt(
                    query=chat_request.message,
                    context=context_texts,
                    chat_mode=chat_request.mode,
                    selected_text=chat_request.selected_text
                )
                user_message = book_agent.get_user_message(
                    query=chat_request.message,
                    context=context_texts,
                    chat_mode=chat_request.mode
                )

                # Stream the response
                async for chunk in self._call_llm_streaming(
                    system_message=system_message,
                    user_message=user_message,
                    context_texts=context_texts,
                    mode=chat_request.mode
                ):
                    yield chunk

        except Exception as e:
            logger.error(f"Error in streaming chat request: {str(e)}")
            error_chunk = {
                "type": "error",
                "message": str(e)
            }
            yield json.dumps(error_chunk)

    async def _call_llm(self, system_message: str, user_message: str, max_retries: int = 3) -> str:
        """
        Call the LLM with system and user messages using Agents SDK.
        Includes retry logic for rate limiting.

        Args:
            system_message: The system prompt
            user_message: The user message
            max_retries: Maximum number of retry attempts

        Returns:
            Generated response text
        """
        for attempt in range(max_retries):
            try:
                # Create an agent with the system message as instructions
                agent = Agent(
                    name="Assistant",
                    instructions=system_message,
                    model=self.model
                )
                
                # Run the agent and get the response
                result = await Runner.run(agent, input=user_message)
                
                return result.final_output

            except Exception as e:
                error_message = str(e)
                logger.error(f"Error calling LLM (attempt {attempt + 1}/{max_retries}): {error_message}")
                
                # Check if it's a rate limit error
                if "429" in error_message or "quota" in error_message.lower() or "RESOURCE_EXHAUSTED" in error_message:
                    # Extract retry delay from error message
                    retry_delay = self._extract_retry_delay(error_message)
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Rate limit hit. Waiting {retry_delay} seconds before retry...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        return "I'm currently experiencing high traffic. Please try again in a few moments."
                
                # For other errors, return user-friendly message
                if "404" in error_message:
                    return "The AI model is temporarily unavailable. Please try again later."
                else:
                    return "I encountered an error while processing your request. Please try again."

    def _extract_retry_delay(self, error_message: str) -> float:
        """
        Extract retry delay from error message.
        
        Args:
            error_message: The error message containing retry information
            
        Returns:
            Retry delay in seconds (defaults to exponential backoff if not found)
        """
        # Try to extract from "Please retry in X.Xs" or "retryDelay": "Xs"
        patterns = [
            r"retry in (\d+\.?\d*)s",  # e.g., "retry in 35.24s"
            r'"retryDelay":\s*"(\d+)s"',  # e.g., "retryDelay": "35s"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                return float(match.group(1))
        
        # Default to exponential backoff if no retry delay found
        return 5.0

    async def _call_llm_streaming(
        self,
        system_message: str,
        user_message: str,
        context_texts: List[str],
        mode: ChatMode,
        max_retries: int = 3
    ) -> AsyncGenerator[str, None]:
        """
        Call the LLM with streaming and yield response chunks using Agents SDK.
        Includes retry logic for rate limiting.

        Args:
            system_message: The system prompt
            user_message: The user message
            context_texts: Context texts for metadata
            mode: Chat mode for metadata
            max_retries: Maximum number of retry attempts

        Yields:
            JSON strings representing response chunks
        """
        for attempt in range(max_retries):
            try:
                # Create an agent with the system message as instructions
                agent = Agent(
                    name="Assistant",
                    instructions=system_message,
                    model=self.model
                )
                
                # Run the agent with streaming
                result = Runner.run_streamed(agent, input=user_message)
                
                full_response = ""
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        content = event.data.delta
                        if content:
                            full_response += content
                            
                            # Send the chunk
                            chunk_data = {
                                "type": "chunk",
                                "content": content
                            }
                            yield json.dumps(chunk_data)

                # Send the final response
                mode_value = mode.value if hasattr(mode, 'value') else str(mode)
                final_response = {
                    "type": "end",
                    "final_response": {
                        "id": f"resp_{int(datetime.utcnow().timestamp() * 1000)}",
                        "message": full_response,
                        "mode_used": mode_value,
                        "retrieved_context": context_texts,
                        "confidence": 0.8 if context_texts else 0.7,
                        "created_at": datetime.utcnow().isoformat()
                    }
                }
                yield json.dumps(final_response)
                return  # Success, exit retry loop

            except Exception as e:
                error_message = str(e)
                logger.error(f"Error in streaming LLM call (attempt {attempt + 1}/{max_retries}): {error_message}")
                
                # Check if it's a rate limit error
                if "429" in error_message or "quota" in error_message.lower() or "RESOURCE_EXHAUSTED" in error_message:
                    retry_delay = self._extract_retry_delay(error_message)
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Rate limit hit. Waiting {retry_delay} seconds before retry...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        user_friendly_message = "I'm currently experiencing high traffic. Please try again in a few moments."
                else:
                    # For other errors, provide user-friendly message
                    if "404" in error_message:
                        user_friendly_message = "The AI model is temporarily unavailable. Please try again later."
                    else:
                        user_friendly_message = "I encountered an error while processing your request. Please try again."
                
                # Send error chunk with user-friendly message
                error_chunk = {
                    "type": "error",
                    "message": user_friendly_message
                }
                yield json.dumps(error_chunk)
                return

    async def validate_and_clean_query(self, query: str) -> str:
        """
        Validate and clean the user query.

        Args:
            query: The raw user query

        Returns:
            Cleaned and validated query
        """
        if not query or len(query.strip()) == 0:
            raise ValueError("Query cannot be empty")

        # Basic cleaning
        cleaned_query = query.strip()

        # Validate length
        if len(cleaned_query) > 2000:  # From our model constraints
            raise ValueError("Query is too long")

        return cleaned_query


# Global instance
chat_service = ChatService()