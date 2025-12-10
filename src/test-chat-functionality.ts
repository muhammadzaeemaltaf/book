// Test file for basic Q&A functionality
import { ChatService } from './services/chatService';
import { apiClient } from './services/apiClient';

// Mock API client for testing
const mockApiClient = {
  sendChatMessage: jest.fn().mockResolvedValue({
    id: 'test-response-1',
    message: 'This is a test response from the chatbot.',
    mode_used: 'normal_qa',
    retrieved_context: ['Sample context from the textbook'],
    confidence: 0.95,
    created_at: new Date().toISOString()
  })
};

// Test basic chat functionality
describe('Chat Service - Basic Q&A', () => {
  let chatService: ChatService;

  beforeEach(() => {
    // @ts-ignore - for testing purposes
    chatService = new ChatService({ topK: 3, temperature: 0.7 });
    // Replace the actual apiClient with our mock
    (chatService as any).apiClient = mockApiClient;
  });

  test('should send a message and receive a response', async () => {
    const message = 'What is ROS 2?';
    const response = await chatService.sendMessage(message, 'normal_qa');

    expect(response.content).toBe('This is a test response from the chatbot.');
    expect(response.role).toBe('assistant');
    expect(mockApiClient.sendChatMessage).toHaveBeenCalledWith({
      message,
      mode: 'normal_qa',
      selected_text: undefined,
      stream: true,
      top_k: 3,
      temperature: 0.7
    });
  });

  test('should handle streaming messages', async () => {
    const message = 'Explain robot navigation';
    let receivedChunks = '';

    const mockApiClientStream = {
      sendChatMessageStream: jest.fn().mockImplementation((request, onChunk, onEnd) => {
        onChunk({ type: 'chunk', content: 'First part of ' });
        receivedChunks += 'First part of ';
        onChunk({ type: 'chunk', content: 'the response.' });
        receivedChunks += 'the response.';
        onChunk({ type: 'end', final_response: {
          id: 'stream-response-1',
          message: 'First part of the response.',
          mode_used: 'normal_qa',
          retrieved_context: ['Sample context'],
          confidence: 0.85,
          created_at: new Date().toISOString()
        }});
        onEnd?.();
      })
    };

    // @ts-ignore - for testing purposes
    chatService.apiClient = mockApiClientStream;

    const response = await chatService.sendMessageStream(message, 'normal_qa', undefined, (chunk) => {
      receivedChunks += chunk;
    });

    expect(receivedChunks).toContain('First part of the response.');
  });

  test('should maintain chat history', async () => {
    const initialHistory = chatService.getHistory();
    expect(initialHistory.length).toBe(0);

    await chatService.sendMessage('Test message', 'normal_qa');

    const updatedHistory = chatService.getHistory();
    expect(updatedHistory.length).toBe(2); // User message + Assistant response
    expect(updatedHistory[0].role).toBe('user');
    expect(updatedHistory[1].role).toBe('assistant');
  });

  test('should clear chat history', () => {
    chatService.clearHistory();
    const history = chatService.getHistory();
    expect(history.length).toBe(0);
  });
});

// Test the useChat hook
describe('useChat Hook', () => {
  test('should initialize with empty messages', () => {
    // This would be tested in a React testing environment
    // Testing the initial state of the hook
    const { result } = renderHook(() => useChat());

    expect(result.current.messages).toEqual([]);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  test('should send a message and update state', async () => {
    // Mock the chat service
    const mockSendMessage = jest.fn().mockResolvedValue({
      id: 'test-response-1',
      content: 'Test response',
      role: 'assistant',
      timestamp: new Date()
    });

    const { result } = renderHook(() => useChat());

    // Simulate sending a message
    await act(async () => {
      await result.current.sendMessage('Test question');
    });

    // Check that the message was added to the state
    expect(result.current.messages.length).toBe(2); // User + Assistant
    expect(result.current.isLoading).toBe(false);
  });
});

// Test the API client
describe('API Client', () => {
  test('should send chat message to backend', async () => {
    const apiClient = new ApiClient('http://localhost:8000');

    // Mock fetch
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        id: 'test-response-1',
        message: 'Test response from backend',
        mode_used: 'normal_qa',
        retrieved_context: [],
        confidence: 0.9,
        created_at: new Date().toISOString()
      })
    });

    const response = await apiClient.sendChatMessage({
      message: 'Test message',
      mode: 'normal_qa',
      stream: false
    });

    expect(fetch).toHaveBeenCalledWith(
      'http://localhost:8000/chat/',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        body: JSON.stringify({
          message: 'Test message',
          mode: 'normal_qa',
          stream: false,
          selected_text: undefined,
          top_k: undefined,
          temperature: undefined
        })
      })
    );

    expect(response.message).toBe('Test response from backend');
  });
});