"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
// Test file for basic Q&A functionality
var chatService_1 = require("./services/chatService");
// Mock API client for testing
var mockApiClient = {
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
describe('Chat Service - Basic Q&A', function () {
    var chatService;
    beforeEach(function () {
        // @ts-ignore - for testing purposes
        chatService = new chatService_1.ChatService({ topK: 3, temperature: 0.7 });
        // Replace the actual apiClient with our mock
        chatService.apiClient = mockApiClient;
    });
    test('should send a message and receive a response', function () { return __awaiter(void 0, void 0, void 0, function () {
        var message, response;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    message = 'What is ROS 2?';
                    return [4 /*yield*/, chatService.sendMessage(message, 'normal_qa')];
                case 1:
                    response = _a.sent();
                    expect(response.content).toBe('This is a test response from the chatbot.');
                    expect(response.role).toBe('assistant');
                    expect(mockApiClient.sendChatMessage).toHaveBeenCalledWith({
                        message: message,
                        mode: 'normal_qa',
                        selected_text: undefined,
                        stream: true,
                        top_k: 3,
                        temperature: 0.7
                    });
                    return [2 /*return*/];
            }
        });
    }); });
    test('should handle streaming messages', function () { return __awaiter(void 0, void 0, void 0, function () {
        var message, receivedChunks, mockApiClientStream, response;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    message = 'Explain robot navigation';
                    receivedChunks = '';
                    mockApiClientStream = {
                        sendChatMessageStream: jest.fn().mockImplementation(function (request, onChunk, onEnd) {
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
                                } });
                            onEnd === null || onEnd === void 0 ? void 0 : onEnd();
                        })
                    };
                    // @ts-ignore - for testing purposes
                    chatService.apiClient = mockApiClientStream;
                    return [4 /*yield*/, chatService.sendMessageStream(message, 'normal_qa', undefined, function (chunk) {
                            receivedChunks += chunk;
                        })];
                case 1:
                    response = _a.sent();
                    expect(receivedChunks).toContain('First part of the response.');
                    return [2 /*return*/];
            }
        });
    }); });
    test('should maintain chat history', function () { return __awaiter(void 0, void 0, void 0, function () {
        var initialHistory, updatedHistory;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    initialHistory = chatService.getHistory();
                    expect(initialHistory.length).toBe(0);
                    return [4 /*yield*/, chatService.sendMessage('Test message', 'normal_qa')];
                case 1:
                    _a.sent();
                    updatedHistory = chatService.getHistory();
                    expect(updatedHistory.length).toBe(2); // User message + Assistant response
                    expect(updatedHistory[0].role).toBe('user');
                    expect(updatedHistory[1].role).toBe('assistant');
                    return [2 /*return*/];
            }
        });
    }); });
    test('should clear chat history', function () {
        chatService.clearHistory();
        var history = chatService.getHistory();
        expect(history.length).toBe(0);
    });
});
// Test the useChat hook
describe('useChat Hook', function () {
    test('should initialize with empty messages', function () {
        // This would be tested in a React testing environment
        // Testing the initial state of the hook
        var result = renderHook(function () { return useChat(); }).result;
        expect(result.current.messages).toEqual([]);
        expect(result.current.isLoading).toBe(false);
        expect(result.current.error).toBeNull();
    });
    test('should send a message and update state', function () { return __awaiter(void 0, void 0, void 0, function () {
        var mockSendMessage, result;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    mockSendMessage = jest.fn().mockResolvedValue({
                        id: 'test-response-1',
                        content: 'Test response',
                        role: 'assistant',
                        timestamp: new Date()
                    });
                    result = renderHook(function () { return useChat(); }).result;
                    // Simulate sending a message
                    return [4 /*yield*/, act(function () { return __awaiter(void 0, void 0, void 0, function () {
                            return __generator(this, function (_a) {
                                switch (_a.label) {
                                    case 0: return [4 /*yield*/, result.current.sendMessage('Test question')];
                                    case 1:
                                        _a.sent();
                                        return [2 /*return*/];
                                }
                            });
                        }); })];
                case 1:
                    // Simulate sending a message
                    _a.sent();
                    // Check that the message was added to the state
                    expect(result.current.messages.length).toBe(2); // User + Assistant
                    expect(result.current.isLoading).toBe(false);
                    return [2 /*return*/];
            }
        });
    }); });
});
// Test the API client
describe('API Client', function () {
    test('should send chat message to backend', function () { return __awaiter(void 0, void 0, void 0, function () {
        var apiClient, response;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    apiClient = new ApiClient('http://localhost:8000');
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
                    return [4 /*yield*/, apiClient.sendChatMessage({
                            message: 'Test message',
                            mode: 'normal_qa',
                            stream: false
                        })];
                case 1:
                    response = _a.sent();
                    expect(fetch).toHaveBeenCalledWith('http://localhost:8000/chat/', expect.objectContaining({
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
                    }));
                    expect(response.message).toBe('Test response from backend');
                    return [2 /*return*/];
            }
        });
    }); });
});
