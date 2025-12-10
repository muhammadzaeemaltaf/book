"use strict";
// API Client Service for RAG Chatbot
// Handles all communication with the backend API
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.apiClient = void 0;
var ApiClient = /** @class */ (function () {
    function ApiClient(baseUrl) {
        this.baseUrl = baseUrl || 'http://localhost:8000';
        this.defaultHeaders = {
            'Content-Type': 'application/json',
        };
    }
    /**
     * Send a chat message to the backend
     */
    ApiClient.prototype.sendChatMessage = function (request) {
        return __awaiter(this, void 0, void 0, function () {
            var response, errorData, error_1;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, 5, , 6]);
                        return [4 /*yield*/, fetch("".concat(this.baseUrl, "/chat/"), {
                                method: 'POST',
                                headers: this.defaultHeaders,
                                body: JSON.stringify(request),
                            })];
                    case 1:
                        response = _a.sent();
                        if (!!response.ok) return [3 /*break*/, 3];
                        return [4 /*yield*/, response.json().catch(function () { return ({}); })];
                    case 2:
                        errorData = _a.sent();
                        throw new Error("Chat request failed: ".concat(response.status, " - ").concat(errorData.detail || response.statusText));
                    case 3: return [4 /*yield*/, response.json()];
                    case 4: return [2 /*return*/, _a.sent()];
                    case 5:
                        error_1 = _a.sent();
                        console.error('Error sending chat message:', error_1);
                        throw error_1;
                    case 6: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * Send a chat message with streaming support
     */
    ApiClient.prototype.sendChatMessageStream = function (request, onChunk, onEnd) {
        return __awaiter(this, void 0, void 0, function () {
            var response, errorData, reader, decoder, done, _a, value, readerDone, chunk, lines, _i, lines_1, line, data, parsed, error_2;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 7, , 8]);
                        return [4 /*yield*/, fetch("".concat(this.baseUrl, "/chat/stream"), {
                                method: 'POST',
                                headers: this.defaultHeaders,
                                body: JSON.stringify(__assign(__assign({}, request), { stream: true })),
                            })];
                    case 1:
                        response = _b.sent();
                        if (!(!response.ok || !response.body)) return [3 /*break*/, 3];
                        return [4 /*yield*/, response.json().catch(function () { return ({}); })];
                    case 2:
                        errorData = _b.sent();
                        throw new Error("Streaming chat request failed: ".concat(response.status, " - ").concat(errorData.detail || response.statusText));
                    case 3:
                        reader = response.body.getReader();
                        decoder = new TextDecoder();
                        done = false;
                        _b.label = 4;
                    case 4:
                        if (!!done) return [3 /*break*/, 6];
                        return [4 /*yield*/, reader.read()];
                    case 5:
                        _a = _b.sent(), value = _a.value, readerDone = _a.done;
                        done = readerDone;
                        if (value) {
                            chunk = decoder.decode(value, { stream: true });
                            lines = chunk.split('\n');
                            for (_i = 0, lines_1 = lines; _i < lines_1.length; _i++) {
                                line = lines_1[_i];
                                if (line.startsWith('data: ')) {
                                    data = line.slice(6);
                                    if (data && data.trim() !== '[DONE]') {
                                        try {
                                            parsed = JSON.parse(data);
                                            onChunk(parsed);
                                        }
                                        catch (e) {
                                            console.error('Error parsing SSE data:', e);
                                        }
                                    }
                                }
                            }
                        }
                        return [3 /*break*/, 4];
                    case 6:
                        if (onEnd) {
                            onEnd();
                        }
                        return [3 /*break*/, 8];
                    case 7:
                        error_2 = _b.sent();
                        console.error('Error in streaming chat:', error_2);
                        throw error_2;
                    case 8: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * Ingest documents into the vector database
     */
    ApiClient.prototype.ingestDocuments = function (request) {
        return __awaiter(this, void 0, void 0, function () {
            var response, errorData, error_3;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, 5, , 6]);
                        return [4 /*yield*/, fetch("".concat(this.baseUrl, "/ingest/"), {
                                method: 'POST',
                                headers: this.defaultHeaders,
                                body: JSON.stringify(request),
                            })];
                    case 1:
                        response = _a.sent();
                        if (!!response.ok) return [3 /*break*/, 3];
                        return [4 /*yield*/, response.json().catch(function () { return ({}); })];
                    case 2:
                        errorData = _a.sent();
                        throw new Error("Ingest request failed: ".concat(response.status, " - ").concat(errorData.detail || response.statusText));
                    case 3: return [4 /*yield*/, response.json()];
                    case 4: return [2 /*return*/, _a.sent()];
                    case 5:
                        error_3 = _a.sent();
                        console.error('Error ingesting documents:', error_3);
                        throw error_3;
                    case 6: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * Search for documents in the vector database
     */
    ApiClient.prototype.searchDocuments = function (request) {
        return __awaiter(this, void 0, void 0, function () {
            var response, errorData, error_4;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, 5, , 6]);
                        return [4 /*yield*/, fetch("".concat(this.baseUrl, "/search/"), {
                                method: 'POST',
                                headers: this.defaultHeaders,
                                body: JSON.stringify(request),
                            })];
                    case 1:
                        response = _a.sent();
                        if (!!response.ok) return [3 /*break*/, 3];
                        return [4 /*yield*/, response.json().catch(function () { return ({}); })];
                    case 2:
                        errorData = _a.sent();
                        throw new Error("Search request failed: ".concat(response.status, " - ").concat(errorData.detail || response.statusText));
                    case 3: return [4 /*yield*/, response.json()];
                    case 4: return [2 /*return*/, _a.sent()];
                    case 5:
                        error_4 = _a.sent();
                        console.error('Error searching documents:', error_4);
                        throw error_4;
                    case 6: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * Check the health of the API
     */
    ApiClient.prototype.healthCheck = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response, error_5;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, 3, , 4]);
                        return [4 /*yield*/, fetch("".concat(this.baseUrl, "/health"), {
                                method: 'GET',
                                headers: this.defaultHeaders,
                            })];
                    case 1:
                        response = _a.sent();
                        if (!response.ok) {
                            throw new Error("Health check failed: ".concat(response.status, " - ").concat(response.statusText));
                        }
                        return [4 /*yield*/, response.json()];
                    case 2: return [2 /*return*/, _a.sent()];
                    case 3:
                        error_5 = _a.sent();
                        console.error('Error checking health:', error_5);
                        throw error_5;
                    case 4: return [2 /*return*/];
                }
            });
        });
    };
    return ApiClient;
}());
// Create a singleton instance
exports.apiClient = new ApiClient();
