import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage } from './ChatMessage';
import ChatInput from './ChatInput';
import { useChat } from '../hooks/useChat';
import { apiClient } from '../services/apiClient';
import '../css/chat.css';

interface ChatWidgetProps {
  // Props for the chat widget
  initialMessages?: Array<{
    id: string;
    content: string;
    role: 'user' | 'assistant';
    timestamp: Date;
  }>;
  onMessageSend?: (message: string, mode: 'normal_qa' | 'selected_text') => void;
}

export const ChatWidget: React.FC<ChatWidgetProps> = ({
  initialMessages = [],
  onMessageSend
}) => {
  // Load initial state from sessionStorage for session-only persistence
  const [isOpen, setIsOpen] = useState(false);

  // Ref for the chat input to manage focus
  const chatInputRef = useRef<import('./ChatInput').ChatInputHandle>(null);

  const [selectedText, setSelectedText] = useState<string | null>(null);
  const [mode, setMode] = useState<'normal_qa' | 'selected_text'>('normal_qa');
  const [selectionPopup, setSelectionPopup] = useState<{top: number, left: number, text: string} | null>(null);

  const [isMinimized, setIsMinimized] = useState(() => {
    // Only access sessionStorage in browser environment
    if (typeof window !== 'undefined' && typeof sessionStorage !== 'undefined') {
      const savedMinimized = sessionStorage.getItem('chatWidgetMinimized');
      return savedMinimized ? JSON.parse(savedMinimized) : false;
    }
    return false;
  });

  const {
    messages,
    sendMessage,
    sendStreamMessage,
    isLoading,
    error,
    addMessage
  } = useChat(initialMessages);

  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Function to handle text selection with popup
  useEffect(() => {
    const handleSelection = (e: MouseEvent) => {
      // Don't handle if clicking on the popup itself
      const target = e.target as HTMLElement;
      if (target?.closest('.selection-popup')) {
        return;
      }

      const selection = window.getSelection();
      const selectedText = selection?.toString().trim();

      if (selectedText) {
        // Check if the selection is within our chat widget
        const range = selection?.getRangeAt(0);
        const selectedElement = range?.commonAncestorContainer.parentElement;

        // Only handle selection if it's not in our chat widget
        if (!selectedElement?.closest('.chat-widget')) {
          const rect = range?.getBoundingClientRect();
          if (rect) {
            setSelectionPopup({
              top: rect.top - 40, // Position above the selection
              left: rect.left + rect.width / 2, // Center horizontally
              text: selectedText
            });
          }
        }
      } else if (!target?.closest('.selection-popup')) {
        // Only clear popup if not clicking on the popup
        setSelectionPopup(null);
        setMode('normal_qa');
      }
    };

    document.addEventListener('mouseup', handleSelection as any);
    return () => {
      document.removeEventListener('mouseup', handleSelection as any);
    };
  }, []);

  // Effect to focus on input when chat opens or when selected text is set
  useEffect(() => {
    if (isOpen && !isMinimized && chatInputRef.current) {
      // Add a small delay to ensure the component is rendered
      setTimeout(() => {
        chatInputRef.current?.focus();
      }, 100);
    }
  }, [isOpen, isMinimized]);

  // Effect to focus on input after text selection
  useEffect(() => {
    if (selectedText && isOpen && !isMinimized && chatInputRef.current) {
      // Add a small delay to ensure the component is rendered
      setTimeout(() => {
        chatInputRef.current?.focus();
      }, 100);
    }
  }, [selectedText, isOpen, isMinimized]);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (message: string) => {
    if (!message.trim()) return;

    // Set the mode based on whether we have selected text
    const currentMode = selectedText ? 'selected_text' : 'normal_qa';
    setMode(currentMode);

    try {
      // If we have selected text, add it as a special context message in the chat
      if (selectedText) {
        const selectedTextContextMessage = {
          id: `context-${Date.now()}`,
          content: selectedText,
          role: 'context' as const, // Using 'context' role to distinguish from regular messages
          timestamp: new Date(),
          isContext: true // Additional flag to identify context messages
        };
        // Add the selected text as a context message to show it in the chat
        addMessage(selectedTextContextMessage);
      }

      // Use streaming method for better user experience
      // The user message will be added automatically by the chat service
      await sendStreamMessage(message, currentMode, selectedText || undefined);
      // Clear selected text after sending
      setSelectedText(null);

      // Also clear the browser's text selection
      if (window.getSelection) {
        const selection = window.getSelection();
        if (selection) {
          selection.removeAllRanges();
        }
      }

      // Call the optional callback
      if (onMessageSend) {
        onMessageSend(message, currentMode);
      }

      // Focus back on the input after sending for potential follow-up questions
      setTimeout(() => {
        if (chatInputRef.current) {
          chatInputRef.current.focus();
        }
      }, 100);
    } catch (err) {
      console.error('Error sending message:', err);
    }
  };

  const toggleChat = () => {
    const newOpenState = !isOpen;
    setIsOpen(newOpenState);
    // Use sessionStorage instead of localStorage for session-only persistence
    if (typeof window !== 'undefined' && typeof sessionStorage !== 'undefined') {
      sessionStorage.setItem('chatWidgetOpen', JSON.stringify(newOpenState));
    }
    if (newOpenState) {
      setIsMinimized(false); // When opening, ensure it's not minimized
      // Add initial greeting if no messages exist
      if (messages.length === 0) {
        const greetingMessage = {
          id: `greeting-${Date.now()}`,
          content: "Hello! I'm your AI assistant for Guide to Physical AI & Humanoid Robotics. How can I help you today?",
          role: 'assistant',
          timestamp: new Date()
        };
        addMessage(greetingMessage);
      }
    }
  };

  const toggleMinimize = () => {
    const newMinimizedState = !isMinimized;
    setIsMinimized(newMinimizedState);
    if (typeof window !== 'undefined' && typeof sessionStorage !== 'undefined') {
      sessionStorage.setItem('chatWidgetMinimized', JSON.stringify(newMinimizedState));
    }
  };

  // Function to handle selecting the text
  const handleSelectText = () => {
    if (selectionPopup) {
      // Set the selected text and switch to selected text mode
      setSelectedText(selectionPopup.text);
      setMode('selected_text');
      setSelectionPopup(null); // Hide the popup

      // Ensure chat is open and not minimized when text is selected
      if (!isOpen) {
        setIsOpen(true);
        // Store the open state in sessionStorage
        if (typeof window !== 'undefined' && typeof sessionStorage !== 'undefined') {
          sessionStorage.setItem('chatWidgetOpen', JSON.stringify(true));
        }
      }
      if (isMinimized) {
        setIsMinimized(false);
        if (typeof window !== 'undefined' && typeof sessionStorage !== 'undefined') {
          sessionStorage.setItem('chatWidgetMinimized', JSON.stringify(false));
        }
      }
    }
  };

  return (
    <div className={`chat-widget ${isOpen ? 'open' : ''} ${isMinimized ? 'minimized' : ''}`}>
      {/* Chat header */}
      <div className="chat-header" onClick={toggleChat}>
        <div className="chat-header-content">
          <h3>AI book assistant</h3>
          <div className="chat-controls">
            <button
              className="minimize-btn"
              onClick={(e) => {
                e.stopPropagation();
                toggleMinimize();
              }}
              aria-label={isMinimized ? "Maximize chat" : "Minimize chat"}
            >
              {isMinimized ? '+' : '−'}
            </button>
            <button
              className="close-btn"
              onClick={(e) => {
                e.stopPropagation();
                setIsOpen(false);
              }}
              aria-label="Close chat"
            >
              ×
            </button>
          </div>
        </div>
      </div>

      {/* Chat body - only show when open and not minimized */}
      {isOpen && !isMinimized && (
        <div className="chat-body">
          {/* Messages container */}
          <div className="messages-container">
            {messages.map((msg) => (
              <ChatMessage
                key={msg.id}
                content={msg.content}
                role={msg.role}
                timestamp={msg.timestamp}
              />
            ))}
            {isLoading && <ChatMessage content="Thinking..." role="assistant" isLoading={true} />}
            <div ref={messagesEndRef} />
          </div>

          {/* Selected text indicator above input area */}
          {selectedText && (
            <div className="selected-text-indicator">
              <p><strong>Selected text:</strong> "{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"</p>
              <button
                onClick={() => setSelectedText(null)}
                className="clear-selection-btn"
              >
                Clear selection
              </button>
            </div>
          )}

          {/* Input area */}
          <div className="input-container">
            <ChatInput
              ref={chatInputRef}
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              currentMode={mode}
              selectedText={selectedText || undefined}
            />
          </div>

          {/* Error display */}
          {error && (
            <div className="error-message">
              Error: {error.message || 'An error occurred'}
            </div>
          )}
        </div>
      )}

      {/* Floating button when closed */}
      {!isOpen && (
        <div className="chat-toggle-button" onClick={toggleChat}>
          💬
        </div>
      )}

      {/* Selection popup */}
      {selectionPopup && (
        <div
          className="selection-popup"
          style={{
            position: 'fixed',
            top: `${selectionPopup.top}px`,
            left: `${selectionPopup.left}px`,
            transform: 'translateX(-50%)',
            zIndex: 1001,
            backgroundColor: '#4f46e5',
            color: 'white',
            padding: '6px 12px',
            borderRadius: '4px',
            fontSize: '14px',
            cursor: 'pointer',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
          }}
          onClick={handleSelectText}
        >
          Select
        </div>
      )}
    </div>
  );
};