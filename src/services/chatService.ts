import { apiClient } from './apiClient';

// Define types for chat messages and state
export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  context?: string[]; // Retrieved context for this message
}

export interface ChatHistory {
  messages: ChatMessage[];
  sessionId?: string;
}

export interface ChatOptions {
  topK?: number;
  temperature?: number;
  stream?: boolean;
}

export class ChatService {
  private history: ChatHistory;
  private options: ChatOptions;

  constructor(initialOptions?: ChatOptions) {
    this.history = { messages: [] };
    this.options = {
      topK: 5,
      temperature: 0.7,
      stream: true,
      ...initialOptions
    };
  }

  /**
   * Send a message and get a response from the backend
   */
  async sendMessage(
    message: string,
    mode: 'normal_qa' | 'selected_text' = 'normal_qa',
    selectedText?: string,
    options?: Partial<ChatOptions>
  ): Promise<ChatMessage> {
    const mergedOptions = { ...this.options, ...options };

    try {
      // Prepare the request to the backend
      const request = {
        message,
        mode,
        selected_text: selectedText,
        stream: mergedOptions.stream,
        top_k: mergedOptions.topK,
        temperature: mergedOptions.temperature
      };

      // Send the request to the backend
      const response = await apiClient.sendChatMessage(request);

      // Validate response
      if (!response || !response.message) {
        throw new Error('Invalid response from server');
      }

      // Create the assistant message from the response
      const assistantMessage: ChatMessage = {
        id: response.id || `resp-${Date.now()}`,
        content: response.message.trim() || 'I apologize, but I was unable to generate a response. Please try rephrasing your question.',
        role: 'assistant',
        timestamp: new Date(response.created_at || new Date().toISOString()),
        context: response.retrieved_context || []
      };

      // Add both user and assistant messages to history
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        content: message,
        role: 'user',
        timestamp: new Date()
      };

      this.history.messages.push(userMessage, assistantMessage);

      return assistantMessage;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  }

  /**
   * Send a message with streaming support
   */
  async sendMessageStream(
    message: string,
    mode: 'normal_qa' | 'selected_text' = 'normal_qa',
    selectedText?: string,
    onChunk?: (chunk: string) => void,
    onEnd?: () => void,
    options?: Partial<ChatOptions>
  ): Promise<ChatMessage> {
    const mergedOptions = { ...this.options, ...options };

    return new Promise((resolve, reject) => {
      let fullResponse = '';
      let responseId = '';
      let responseCreatedAt = new Date().toISOString();
      let retrievedContext: string[] = [];

      // Prepare the request to the backend
      const request = {
        message,
        mode,
        selected_text: selectedText,
        stream: true, // Force streaming for this method
        top_k: mergedOptions.topK,
        temperature: mergedOptions.temperature
      };

      // Send the request with streaming
      apiClient.sendChatMessageStream(
        request,
        (chunk: any) => {
          if (chunk.type === 'chunk' && chunk.content) {
            fullResponse += chunk.content;
            onChunk?.(chunk.content);
          } else if (chunk.type === 'end' && chunk.final_response) {
            // Process the final response
            responseId = chunk.final_response.id;
            responseCreatedAt = chunk.final_response.created_at;
            retrievedContext = chunk.final_response.retrieved_context;
          }
        },
        () => {
          // When streaming is complete
          // Validate response
          if (!fullResponse.trim()) {
            fullResponse = 'I apologize, but I was unable to generate a response. Please try rephrasing your question.';
          }

          const assistantMessage: ChatMessage = {
            id: responseId || `resp-${Date.now()}`,
            content: fullResponse.trim(),
            role: 'assistant',
            timestamp: new Date(responseCreatedAt),
            context: retrievedContext
          };

          // Add both user and assistant messages to history
          const userMessage: ChatMessage = {
            id: `user-${Date.now()}`,
            content: message,
            role: 'user',
            timestamp: new Date()
          };

          this.history.messages.push(userMessage, assistantMessage);

          onEnd?.();
          resolve(assistantMessage);
        }
      ).catch(reject);
    });
  }

  /**
   * Get the current chat history
   */
  getHistory(): ChatMessage[] {
    return [...this.history.messages];
  }

  /**
   * Clear the chat history
   */
  clearHistory(): void {
    this.history.messages = [];
  }

  /**
   * Update chat options
   */
  updateOptions(options: Partial<ChatOptions>): void {
    this.options = { ...this.options, ...options };
  }

  /**
   * Get current chat options
   */
  getOptions(): ChatOptions {
    return { ...this.options };
  }

  /**
   * Search for relevant documents
   */
  async searchDocuments(query: string, topK?: number): Promise<any[]> {
    try {
      const response = await apiClient.searchDocuments({
        query,
        top_k: topK || this.options.topK
      });

      return response.results;
    } catch (error) {
      console.error('Error searching documents:', error);
      throw error;
    }
  }
}

// Create a singleton instance
export const chatService = new ChatService();