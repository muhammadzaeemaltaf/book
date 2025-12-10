# Data Model: RAG Chatbot System

## Entity Models

### ChatQuery
Represents a user's question submitted to the system, including the query text and mode (normal Q&A or selected text)

**Fields:**
- `id`: string (UUID) - Unique identifier for the query
- `text`: string - The actual question text from the user
- `mode`: enum (normal_qa | selected_text) - Query processing mode
- `selected_text`: string | null - Text selected by user (if in selected_text mode)
- `timestamp`: datetime - When the query was submitted
- `user_context`: object | null - Additional user context (if any)

**Validation:**
- `text` must be non-empty (1-2000 characters)
- `mode` must be one of the allowed values
- If `mode` is `selected_text`, `selected_text` must be provided

### DocumentChunk
Represents a segment of the book content that has been processed and embedded for vector storage

**Fields:**
- `id`: string (UUID) - Unique identifier for the chunk
- `content`: string - The text content of the chunk
- `embedding`: float[] - Vector representation of the content
- `source_document`: string - Reference to the original document
- `chunk_index`: integer - Position of chunk in original document
- `metadata`: object - Additional metadata (headers, section, etc.)
- `created_at`: datetime - When the chunk was created

**Validation:**
- `content` must be non-empty (50-1000 tokens recommended)
- `embedding` must have correct dimensions (1024 for multilingual, 1536 for English)
- `source_document` must be a valid reference

### VectorEmbedding
Represents the numerical representation of text content stored in the vector database for similarity search

**Fields:**
- `vector_id`: string - ID in the vector database
- `vector`: float[] - The actual embedding vector
- `document_id`: string - Reference to the source document chunk
- `model_version`: string - Version of the embedding model used
- `created_at`: datetime - When the embedding was generated

**Validation:**
- `vector` must match expected dimensions
- `document_id` must reference a valid document chunk

### ChatResponse
Represents the system's answer to a user's query, including streaming capability

**Fields:**
- `id`: string (UUID) - Unique identifier for the response
- `query_id`: string - Reference to the original query
- `content`: string - The response text
- `mode_used`: enum (normal_qa | selected_text | vector_search) - Mode actually used
- `retrieved_context`: string[] - Context snippets used to generate response
- `confidence_score`: float - Confidence level of the response (0-1)
- `timestamp`: datetime - When the response was generated
- `streaming_data`: object | null - Streaming response chunks (if applicable)

**Validation:**
- `content` must be non-empty
- `confidence_score` must be between 0 and 1
- `query_id` must reference a valid query

### IngestionPipeline
Represents the process that converts Docusaurus markdown content into vector embeddings stored in the database

**Fields:**
- `id`: string (UUID) - Unique identifier for the pipeline run
- `source_path`: string - Path to the source documents
- `status`: enum (pending | processing | completed | failed) - Current status
- `total_documents`: integer - Number of documents to process
- `processed_documents`: integer - Number of documents processed
- `start_time`: datetime - When the pipeline started
- `end_time`: datetime | null - When the pipeline completed
- `error_message`: string | null - Error details if failed

**Validation:**
- `status` must be one of the allowed values
- `total_documents` must be >= `processed_documents`
- `source_path` must be a valid path reference

## Relationships

```
ChatQuery (1) → (1) ChatResponse
DocumentChunk (1) → (1) VectorEmbedding
IngestionPipeline (1) → (N) DocumentChunk
ChatQuery (1) → (N) DocumentChunk (via retrieved_context)
```

## State Transitions

### IngestionPipeline States:
- `pending` → `processing` → `completed` | `failed`

## API Contract Models

### Request Models

#### IngestRequest
```json
{
  "source_path": "string",
  "chunk_size": "integer (default: 512)",
  "overlap": "integer (default: 50)"
}
```

#### ChatRequest
```json
{
  "message": "string",
  "mode": "enum: 'normal_qa' | 'selected_text'",
  "selected_text": "string (optional)",
  "stream": "boolean (default: true)"
}
```

#### SearchRequest
```json
{
  "query": "string",
  "top_k": "integer (default: 5)",
  "filters": "object (optional)"
}
```

### Response Models

#### ChatResponse
```json
{
  "id": "string",
  "message": "string",
  "mode_used": "enum: 'normal_qa' | 'selected_text' | 'vector_search'",
  "retrieved_context": ["string"],
  "confidence": "float",
  "created_at": "datetime"
}
```

#### SearchResponse
```json
{
  "results": [
    {
      "id": "string",
      "content": "string",
      "score": "float",
      "source_document": "string",
      "metadata": "object"
    }
  ],
  "total_results": "integer"
}
```

#### IngestResponse
```json
{
  "status": "string",
  "processed_count": "integer",
  "message": "string"
}
```