import { useState, useCallback, useEffect } from 'react';
import { ChatMessage, ChatService } from '../services/chatService';

interface UseChatOptions {
  initialMessages?: ChatMessage[];
  topK?: number;
  temperature?: number;
}

export const useChat = (options: UseChatOptions = {}) => {
  // Use session storage instead of local storage for session-only persistence
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    // Don't load from storage - start fresh each session
    return options.initialMessages || [];
  });

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);

  // Create a chat service instance with provided options
  const chatService = new ChatService({
    topK: options.topK,
    temperature: options.temperature
  });

  // Load initial messages if provided
  useEffect(() => {
    if (options.initialMessages && options.initialMessages.length > 0) {
      setMessages([...options.initialMessages]);
    }
  }, [options.initialMessages]);

  // Helper function to update messages (no persistence)
  const updateMessages = useCallback((updater: (prevMessages: ChatMessage[]) => ChatMessage[]) => {
    setMessages(prev => {
      const newMessages = updater(prev);
      return newMessages;
    });
  }, []);

  // Function to send a message
  const sendMessage = useCallback(async (
    message: string,
    mode: 'normal_qa' | 'selected_text' = 'normal_qa',
    selectedText?: string
  ) => {
    if (isLoading) return;

    setIsLoading(true);
    setError(null);

    try {
      // Add user message to UI immediately
      const userMessage: ChatMessage = {
        id: `temp-${Date.now()}`,
        content: message,
        role: 'user',
        timestamp: new Date()
      };

      updateMessages(prev => [...prev, userMessage]);

      // Send message to backend and get response
      const response = await chatService.sendMessage(message, mode, selectedText);

      // Update messages with the response
      updateMessages(prev => {
        // Remove the temporary user message if it exists and add both messages
        const filtered = prev.filter(msg => msg.id !== userMessage.id);
        return [...filtered, userMessage, response];
      });
    } catch (err) {
      console.error('Error sending message:', err);
      setError(err instanceof Error ? err : new Error('Failed to send message'));

      // Add error message to the chat with more helpful fallback
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        content: `I apologize, but I encountered an error while processing your request. This could be due to:\n\n* Network connectivity issues\n* Server temporarily unavailable\n* Request timeout\n\nPlease try again in a moment. If the problem persists, try refreshing the page.`,
        role: 'assistant',
        timestamp: new Date()
      };

      updateMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, updateMessages]);

  // Function to send a message with streaming
  const sendStreamMessage = useCallback(async (
    message: string,
    mode: 'normal_qa' | 'selected_text' = 'normal_qa',
    selectedText?: string
  ) => {
    if (isLoading) return;

    setIsLoading(true);
    setError(null);

    try {
      // Add user message to UI immediately
      const userMessage: ChatMessage = {
        id: `temp-${Date.now()}`,
        content: message,
        role: 'user',
        timestamp: new Date()
      };

      updateMessages(prev => [...prev, userMessage]);

      // Create a temporary assistant message for streaming content
      const streamingMessageId = `streaming-${Date.now()}`;
      const streamingMessage: ChatMessage = {
        id: streamingMessageId,
        content: '',
        role: 'assistant',
        timestamp: new Date()
      };

      updateMessages(prev => [...prev, streamingMessage]);

      // Send message with streaming
      let fullResponse = '';
      const response = await chatService.sendMessageStream(
        message,
        mode,
        selectedText,
        (chunk) => {
          fullResponse += chunk;

          // Update the streaming message with new content
          updateMessages(prev =>
            prev.map(msg =>
              msg.id === streamingMessageId
                ? { ...msg, content: fullResponse }
                : msg
            )
          );
        }
      );

      // Replace the streaming message with the final response
      updateMessages(prev =>
        prev.map(msg =>
          msg.id === streamingMessageId
            ? response
            : msg
        )
      );
    } catch (err) {
      console.error('Error sending stream message:', err);
      setError(err instanceof Error ? err : new Error('Failed to send stream message'));

      // Add error message to the chat with more helpful fallback
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        content: `I apologize, but I encountered an error while processing your request. This could be due to:\n\n* Network connectivity issues\n* Server temporarily unavailable\n* Request timeout\n\nPlease try again in a moment. If the problem persists, try refreshing the page.`,
        role: 'assistant',
        timestamp: new Date()
      };

      updateMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, updateMessages]);

  // Function to clear chat
  const clearChat = useCallback(() => {
    setMessages([]);
    setError(null);
    chatService.clearHistory();
  }, []);

  // Function to add a message manually (useful for system messages)
  const addMessage = useCallback((message: ChatMessage) => {
    updateMessages(prev => [...prev, message]);
  }, [updateMessages]);

  return {
    messages,
    sendMessage,
    sendStreamMessage,
    isLoading,
    error,
    clearChat,
    addMessage,
    // Additional utilities
    hasMessages: messages.length > 0,
    lastMessage: messages.length > 0 ? messages[messages.length - 1] : null
  };
};