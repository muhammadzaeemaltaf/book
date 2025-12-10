import React, { useState, useRef, useEffect } from 'react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  currentMode: 'normal_qa' | 'selected_text';
  selectedText?: string;
}

export interface ChatInputHandle {
  focus: () => void;
}

export const ChatInput: React.ForwardRefRenderFunction<ChatInputHandle, ChatInputProps> = ({
  onSendMessage,
  isLoading,
  currentMode,
  selectedText
}, ref) => {
  const [inputValue, setInputValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Expose focus method to parent component
  React.useImperativeHandle(ref, () => ({
    focus: () => {
      textareaRef.current?.focus();
    }
  }));

  // Auto-resize textarea based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [inputValue]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Submit on Enter (unless Shift is pressed for new line) or Ctrl+Enter/Cmd+Enter
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit(e as any); // Type assertion to bypass event type mismatch
    }
  };

  return (
    <form className="chat-input-form" onSubmit={handleSubmit}>
      {currentMode === 'selected_text' && selectedText && (
        <div className="input-mode-indicator">
          <span className="mode-badge selected-text-mode">Selected Text Mode</span>
        </div>
      )}

      <div className="input-container">
        <textarea
          ref={textareaRef}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            currentMode === 'selected_text' && selectedText
              ? 'Ask about the selected text...'
              : 'Ask a question about the textbook...'
          }
          disabled={isLoading}
          rows={1}
          className="chat-input-textarea"
        />
        <button
          type="submit"
          disabled={isLoading || !inputValue.trim()}
          className="chat-send-button"
          aria-label="Send message"
        >
          {isLoading ? 'Sending...' : '➤'}
        </button>
      </div>

      <div className="input-hints">
        <small>Press Enter to submit, Shift+Enter for new line</small>
      </div>
    </form>
  );
};

export default React.forwardRef(ChatInput);