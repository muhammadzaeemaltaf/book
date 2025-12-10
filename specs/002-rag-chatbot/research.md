# Research Summary: RAG Chatbot Implementation

## Technology Research Findings

### Cohere Embed-v4.0 API
- **Embedding Dimensions**: 1024 for multilingual, 1536 for English-specific
- **Input Requirements**: Max 5,120 tokens per request, supports text up to 4096 characters
- **Best Practices**: Use multilingual if content includes non-English text; otherwise use English-optimized version
- **Decision**: Using embed-multilingual-v3.0 for broader language support as textbook may have mixed content

### Qdrant Vector Database
- **Collection Schema**: Supports metadata storage alongside vectors for filtering
- **Distance Metrics**: Cosine, Euclidean, Dot product (Cosine recommended for text embeddings)
- **Performance**: Free tier supports up to 1GB storage, important to optimize chunk sizes
- **Best Practices**: Store original text content in payload for retrieval, use sparse vectors if needed

### OpenAI Agents SDK with Gemini
- **Runner API**: Supports streaming responses via async generators
- **Configuration**: Requires proper API key configuration and model specification
- **Integration**: Compatible with existing OpenAI SDK patterns
- **Rate Limits**: Important to implement proper retry logic for free-tier usage

### FastAPI Backend Architecture
- **Async Support**: Native async/await support for I/O operations
- **Dependency Injection**: Built-in DI container for service management
- **Streaming Responses**: Server-Sent Events (SSE) support for streaming chat responses
- **CORS Configuration**: Essential for Docusaurus frontend communication

### RAGRetrievalAgent Architecture
- **Subagent Pattern**: Separate agent responsible for all retrieval operations
- **Query Normalization**: Clean and standardize user queries before processing
- **Mode Selection**: Automatically determine if using selected text or vector search
- **Ranking & Filtering**: Implement relevance scoring and content filtering
- **Structured Planning**: Generate retrieval plans for complex queries

## Key Decisions & Rationale

### 1. Cohere Embedding Model Selection
- **Decision**: Use embed-multilingual-v3.0 for broader compatibility
- **Rationale**: Provides support for multiple languages while maintaining good performance
- **Alternatives**: embed-english-v3.0 (faster but limited language support)

### 2. Vector Database Schema Design
- **Decision**: Store original text in Qdrant payload with metadata for filtering
- **Rationale**: Allows retrieval of full context while maintaining search capabilities
- **Alternatives**: Store only vector IDs and reference external storage (more complex)

### 3. Retrieval Mode Selection Algorithm
- **Decision**: Implement context-aware mode selection based on user input and selected text
- **Rationale**: Provides flexibility between selected-text-only and full vector search
- **Alternatives**: Fixed mode selection (less adaptive to user needs)

### 4. Streaming Implementation
- **Decision**: Use Server-Sent Events (SSE) for streaming responses
- **Rationale**: Native FastAPI support and good browser compatibility
- **Alternatives**: WebSockets (more complex setup) or polling (higher latency)

## Risks & Mitigation Strategies

### 1. Embedding Dimension Mismatch
- **Risk**: Cohere model updates changing vector dimensions
- **Mitigation**: Implement version checking and migration scripts

### 2. Qdrant Free Tier Limitations
- **Risk**: Storage or request limits affecting performance
- **Mitigation**: Implement caching and optimize chunk sizes to reduce API calls

### 3. Text Selection Mode Complexity
- **Risk**: User confusion between different query modes
- **Mitigation**: Clear UI indicators and documentation for each mode

### 4. Latency Issues
- **Risk**: Slow response times affecting user experience
- **Mitigation**: Caching, optimized chunk sizes, and async processing

## API Integration Details

### Cohere API Configuration
- Base URL: `https://api.cohere.ai/v1/embed`
- Required headers: `Authorization: Bearer {API_KEY}`, `Content-Type: application/json`
- Parameters: `model`, `texts[]`, `input_type` ("search_document", "search_query")

### Qdrant API Configuration
- Collection creation with cosine similarity
- Payload schema including: text content, source document, chunk metadata
- Search parameters: vector, limit, filter conditions

### Gemini Integration via OpenAI SDK
- Model specification: `gemini-2.0-flash` or equivalent
- Streaming enabled via `stream=True` parameter
- Context injection through system and user message formatting

## Architecture Flow

```
Docusaurus Frontend → FastAPI Backend → RAGRetrievalAgent → Qdrant/Cohere → PromptBuilder → Gemini → Frontend
```

The RAGRetrievalAgent acts as the central hub for all retrieval operations, making decisions about query processing, mode selection, and result ranking before passing context to the LLM for response generation.