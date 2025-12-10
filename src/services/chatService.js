"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
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
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.chatService = exports.ChatService = void 0;
var apiClient_1 = require("./apiClient");
var ChatService = /** @class */ (function () {
    function ChatService(initialOptions) {
        this.history = { messages: [] };
        this.options = __assign({ topK: 5, temperature: 0.7, stream: true }, initialOptions);
    }
    /**
     * Send a message and get a response from the backend
     */
    ChatService.prototype.sendMessage = function (message_1) {
        return __awaiter(this, arguments, void 0, function (message, mode, selectedText, options) {
            var mergedOptions, request, response, assistantMessage, userMessage, error_1;
            if (mode === void 0) { mode = 'normal_qa'; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        mergedOptions = __assign(__assign({}, this.options), options);
                        _a.label = 1;
                    case 1:
                        _a.trys.push([1, 3, , 4]);
                        request = {
                            message: message,
                            mode: mode,
                            selected_text: selectedText,
                            stream: mergedOptions.stream,
                            top_k: mergedOptions.topK,
                            temperature: mergedOptions.temperature
                        };
                        return [4 /*yield*/, apiClient_1.apiClient.sendChatMessage(request)];
                    case 2:
                        response = _a.sent();
                        assistantMessage = {
                            id: response.id,
                            content: response.message,
                            role: 'assistant',
                            timestamp: new Date(response.created_at),
                            context: response.retrieved_context
                        };
                        userMessage = {
                            id: "user-".concat(Date.now()),
                            content: message,
                            role: 'user',
                            timestamp: new Date()
                        };
                        this.history.messages.push(userMessage, assistantMessage);
                        return [2 /*return*/, assistantMessage];
                    case 3:
                        error_1 = _a.sent();
                        console.error('Error sending message:', error_1);
                        throw error_1;
                    case 4: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * Send a message with streaming support
     */
    ChatService.prototype.sendMessageStream = function (message_1) {
        return __awaiter(this, arguments, void 0, function (message, mode, selectedText, onChunk, onEnd, options) {
            var mergedOptions;
            var _this = this;
            if (mode === void 0) { mode = 'normal_qa'; }
            return __generator(this, function (_a) {
                mergedOptions = __assign(__assign({}, this.options), options);
                return [2 /*return*/, new Promise(function (resolve, reject) {
                        var fullResponse = '';
                        var responseId = '';
                        var responseCreatedAt = new Date().toISOString();
                        var retrievedContext = [];
                        // Prepare the request to the backend
                        var request = {
                            message: message,
                            mode: mode,
                            selected_text: selectedText,
                            stream: true, // Force streaming for this method
                            top_k: mergedOptions.topK,
                            temperature: mergedOptions.temperature
                        };
                        // Send the request with streaming
                        apiClient_1.apiClient.sendChatMessageStream(request, function (chunk) {
                            if (chunk.type === 'chunk' && chunk.content) {
                                fullResponse += chunk.content;
                                onChunk === null || onChunk === void 0 ? void 0 : onChunk(chunk.content);
                            }
                            else if (chunk.type === 'end' && chunk.final_response) {
                                // Process the final response
                                responseId = chunk.final_response.id;
                                responseCreatedAt = chunk.final_response.created_at;
                                retrievedContext = chunk.final_response.retrieved_context;
                            }
                        }, function () {
                            // When streaming is complete
                            var assistantMessage = {
                                id: responseId || "resp-".concat(Date.now()),
                                content: fullResponse,
                                role: 'assistant',
                                timestamp: new Date(responseCreatedAt),
                                context: retrievedContext
                            };
                            // Add both user and assistant messages to history
                            var userMessage = {
                                id: "user-".concat(Date.now()),
                                content: message,
                                role: 'user',
                                timestamp: new Date()
                            };
                            _this.history.messages.push(userMessage, assistantMessage);
                            onEnd === null || onEnd === void 0 ? void 0 : onEnd();
                            resolve(assistantMessage);
                        }).catch(reject);
                    })];
            });
        });
    };
    /**
     * Get the current chat history
     */
    ChatService.prototype.getHistory = function () {
        return __spreadArray([], this.history.messages, true);
    };
    /**
     * Clear the chat history
     */
    ChatService.prototype.clearHistory = function () {
        this.history.messages = [];
    };
    /**
     * Update chat options
     */
    ChatService.prototype.updateOptions = function (options) {
        this.options = __assign(__assign({}, this.options), options);
    };
    /**
     * Get current chat options
     */
    ChatService.prototype.getOptions = function () {
        return __assign({}, this.options);
    };
    /**
     * Search for relevant documents
     */
    ChatService.prototype.searchDocuments = function (query, topK) {
        return __awaiter(this, void 0, void 0, function () {
            var response, error_2;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, 2, , 3]);
                        return [4 /*yield*/, apiClient_1.apiClient.searchDocuments({
                                query: query,
                                top_k: topK || this.options.topK
                            })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.results];
                    case 2:
                        error_2 = _a.sent();
                        console.error('Error searching documents:', error_2);
                        throw error_2;
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    return ChatService;
}());
exports.ChatService = ChatService;
// Create a singleton instance
exports.chatService = new ChatService();
