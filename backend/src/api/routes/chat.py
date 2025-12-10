from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from typing import Optional
import asyncio
from ...models.chat import ChatRequest, ChatResponseModel
from ...models.retrieval import ContextFilter
from ...services.qdrant_service import qdrant_service
from ...services.embedding_service import embedding_service
from ...agents.rag_retrieval_agent import rag_retrieval_agent
from ...services.chat_service import chat_service
from ...utils.logging import get_logger, log_api_call, log_retrieval
from ...utils.config import settings
from ...utils.validators import ChatQueryValidator
import time

router = APIRouter()
logger = get_logger("chat_routes")

@router.post("/", response_model=ChatResponseModel)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Main chat endpoint that handles user queries and returns AI-generated responses.
    Supports both normal Q&A and selected text modes.
    """
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000000)}"

    try:
        logger.info(f"Received chat request {request_id}: {chat_request.message[:100]}...")

        # Validate the request
        validator = ChatQueryValidator(
            message=chat_request.message,
            mode=chat_request.mode,
            selected_text=chat_request.selected_text
        )

        # Process the chat request using the chat service
        response = await chat_service.process_chat_request(chat_request)

        # Log the API call
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        log_api_call("/chat", "POST", duration)

        logger.info(f"Completed chat request {request_id} in {duration:.2f}ms")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
    except Exception as e:
        logger.error(f"Error processing chat request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


@router.post("/stream")
async def chat_stream_endpoint(chat_request: ChatRequest):
    """
    Streaming chat endpoint that returns responses as Server-Sent Events.
    """
    start_time = time.time()
    request_id = f"stream_req_{int(time.time() * 1000000)}"

    try:
        logger.info(f"Received streaming chat request {request_id}: {chat_request.message[:100]}...")

        # Validate the request
        validator = ChatQueryValidator(
            message=chat_request.message,
            mode=chat_request.mode,
            selected_text=chat_request.selected_text,
            stream=True
        )

        # Process the streaming chat request
        async def generate_stream():
            try:
                async for chunk in chat_service.process_chat_request_streaming(chat_request):
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                logger.error(f"Error in streaming chat {request_id}: {str(e)}")
                yield f"data: {{\"type\":\"error\",\"message\":\"{str(e)}\"}}\n\n"

        # Log the API call
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        log_api_call("/chat/stream", "POST", duration)

        logger.info(f"Started streaming for request {request_id}")

        return StreamingResponse(generate_stream(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error starting streaming chat request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting streaming chat: {str(e)}")