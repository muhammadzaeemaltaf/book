// API Client Service for RAG Chatbot
// Handles all communication with the backend API

interface ChatRequest {
  message: string;
  mode: 'normal_qa' | 'selected_text';
  selected_text?: string;
  stream?: boolean;
  top_k?: number;
  temperature?: number;
}

interface ChatResponse {
  id: string;
  message: string;
  mode_used: 'normal_qa' | 'selected_text' | 'vector_search';
  retrieved_context: string[];
  confidence: number;
  created_at: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface IngestRequest {
  source_path: string;
  chunk_size?: number;
  overlap?: number;
  recursive?: boolean;
}

interface IngestResponse {
  status: string;
  processed_count: number;
  message: string;
  pipeline_id?: string;
}

interface SearchRequest {
  query: string;
  top_k?: number;
  filters?: Record<string, any>;
}

interface SearchResponse {
  results: Array<{
    id: string;
    content: string;
    score: number;
    source_document: string;
    metadata: Record<string, any>;
    chunk_index?: number;
  }>;
  query: string;
  search_time_ms: number;
}

class ApiClient {
  private baseUrl: string;
  private defaultHeaders: HeadersInit;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl || process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  /**
   * Send a chat message to the backend
   */
  async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/chat/`, {
        method: 'POST',
        headers: this.defaultHeaders,
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`Chat request failed: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error sending chat message:', error);
      throw error;
    }
  }

  /**
   * Send a chat message with streaming support
   */
  async sendChatMessageStream(
    request: ChatRequest,
    onChunk: (chunk: string) => void,
    onEnd?: () => void
  ): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/chat/stream`, {
        method: 'POST',
        headers: this.defaultHeaders,
        body: JSON.stringify({
          ...request,
          stream: true
        }),
      });

      if (!response.ok || !response.body) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`Streaming chat request failed: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let done = false;
      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;

        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          // Process Server-Sent Events format
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6); // Remove 'data: ' prefix
              if (data && data.trim() !== '[DONE]') {
                try {
                  const parsed = JSON.parse(data);
                  onChunk(parsed);
                } catch (e) {
                  console.error('Error parsing SSE data:', e);
                }
              }
            }
          }
        }
      }

      if (onEnd) {
        onEnd();
      }
    } catch (error) {
      console.error('Error in streaming chat:', error);
      throw error;
    }
  }

  /**
   * Ingest documents into the vector database
   */
  async ingestDocuments(request: IngestRequest): Promise<IngestResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/ingest/`, {
        method: 'POST',
        headers: this.defaultHeaders,
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`Ingest request failed: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error ingesting documents:', error);
      throw error;
    }
  }

  /**
   * Search for documents in the vector database
   */
  async searchDocuments(request: SearchRequest): Promise<SearchResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/search/`, {
        method: 'POST',
        headers: this.defaultHeaders,
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`Search request failed: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error searching documents:', error);
      throw error;
    }
  }

  /**
   * Check the health of the API
   */
  async healthCheck(): Promise<{ status: string; timestamp: number; services: Record<string, string> }> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: this.defaultHeaders,
      });

      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status} - ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error checking health:', error);
      throw error;
    }
  }
}

// Create a singleton instance
export const apiClient = new ApiClient();