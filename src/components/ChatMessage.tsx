import React from 'react';
import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatMessageProps {
  content: string;
  role: 'user' | 'assistant' | 'context';
  timestamp?: Date;
  isLoading?: boolean;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  content,
  role,
  timestamp = new Date(),
  isLoading = false
}) => {
  const formattedTime = format(timestamp, 'HH:mm');

  return (
    <div className={`chat-message ${role === 'context' ? 'context' : role} ${isLoading ? 'loading' : ''}`}>
      <div className="message-content">
        {isLoading ? (
          <div className="loading-indicator">
            <div className="loading-dot"></div>
            <div className="loading-dot"></div>
            <div className="loading-dot"></div>
          </div>
        ) : (
          <>
            {role === 'context' ? (
              <div className="context-text">
                <strong>Selected text:</strong> "{content.substring(0, 100)}{content.length > 100 ? '...' : ''}"
              </div>
            ) : role === 'assistant' ? (
              <div className="message-text markdown-content">
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]}
                  components={{
                    // Custom rendering for better styling
                    p: ({node, ...props}) => <p style={{marginBottom: '0.5em', lineHeight: '1.6',}} {...props} />,
                    ul: ({node, ...props}) => <ul style={{marginLeft: '1.5em', marginBottom: '0.5em'}} {...props} />,
                    ol: ({node, ...props}) => <ol style={{marginLeft: '1.5em', marginBottom: '0.5em'}} {...props} />,
                    li: ({node, ...props}) => <li style={{marginBottom: '0.25em'}} {...props} />,
                    code: ({node, className, children, ...props}) => {
                      const isInline = !className || !className.includes('language-');
                      return isInline 
                        ? <code style={{background: '#f3f4f6', padding: '0.2em 0.4em', borderRadius: '3px', fontSize: '0.9em'}} {...props}>{children}</code>
                        : <code style={{display: 'block', background: '#f3f4f6', padding: '0.5em', borderRadius: '4px', overflowX: 'auto', fontSize: '0.9em'}} className={className} {...props}>{children}</code>;
                    },
                    blockquote: ({node, ...props}) => <blockquote style={{borderLeft: '3px solid #4f46e5', paddingLeft: '1em', margin: '0.5em 0', color: '#6b7280', fontStyle: 'italic'}} {...props} />,
                    strong: ({node, ...props}) => <strong style={{fontWeight: 600, color: '#1f2937'}} {...props} />,
                    h1: ({node, ...props}) => <h1 style={{fontSize: '1.25em', fontWeight: 600, marginTop: '0.5em', marginBottom: '0.5em'}} {...props} />,
                    h2: ({node, ...props}) => <h2 style={{fontSize: '1.15em', fontWeight: 600, marginTop: '0.5em', marginBottom: '0.5em'}} {...props} />,
                    h3: ({node, ...props}) => <h3 style={{fontSize: '1.05em', fontWeight: 600, marginTop: '0.5em', marginBottom: '0.5em'}} {...props} />,
                  }}
                >
                  {content}
                </ReactMarkdown>
              </div>
            ) : (
              <div className="message-text">{content}</div>
            )}
          </>
        )}
      </div>
      {role !== 'context' && (
        <div className="message-meta">
          <span className="message-time">{formattedTime}</span>
          <span className="message-role">{role === 'user' ? 'You' : 'Assistant'}</span>
        </div>
      )}
    </div>
  );
};