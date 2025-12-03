import React, { useRef, useEffect } from 'react';
import { Avatar, Typography, Spin } from 'antd';
import { UserOutlined, RobotOutlined } from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useTranslation } from 'react-i18next';
import './MessageList.css';

const { Text } = Typography;

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  metadata?: {
    model?: string;
    tokens?: number;
    latency?: number;
  };
}

interface MessageListProps {
  messages: Message[];
  loading?: boolean;
  autoScroll?: boolean;
}

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  loading = false,
  autoScroll = true
}) => {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, autoScroll]);

  const renderMessage = (message: Message) => {
    const isUser = message.role === 'user';
    const isSystem = message.role === 'system';

    if (isSystem) {
      return (
        <div key={message.id} className="message-system">
          <Text type="secondary">{message.content}</Text>
        </div>
      );
    }

    return (
      <div 
        key={message.id} 
        className={`message ${isUser ? 'message-user' : 'message-assistant'}`}
      >
        <Avatar
          icon={isUser ? <UserOutlined /> : <RobotOutlined />}
          className={`message-avatar ${isUser ? 'avatar-user' : 'avatar-assistant'}`}
        />
        <div className="message-content">
          <div className="message-header">
            <Text strong>
              {isUser ? t('chat.you', 'You') : t('chat.assistant', 'AI Assistant')}
            </Text>
            <Text type="secondary" className="message-time">
              {message.timestamp.toLocaleTimeString()}
            </Text>
          </div>
          <div className="message-body">
            {message.isStreaming ? (
              <>
                <ReactMarkdown
                  components={{
                    code({ node, className, children, ...props }: any) {
                      const match = /language-(\w+)/.exec(className || '');
                      const codeString = String(children).replace(/\n$/, '');
                      const isInline = !match;
                      
                      return !isInline && match ? (
                        <SyntaxHighlighter
                          style={vscDarkPlus as any}
                          language={match[1]}
                          PreTag="div"
                        >
                          {codeString}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    }
                  }}
                >
                  {message.content}
                </ReactMarkdown>
                <span className="streaming-cursor">▊</span>
              </>
            ) : (
              <ReactMarkdown
                components={{
                  code({ node, className, children, ...props }: any) {
                    const match = /language-(\w+)/.exec(className || '');
                    const codeString = String(children).replace(/\n$/, '');
                    const isInline = !match;
                    
                    return !isInline && match ? (
                      <SyntaxHighlighter
                        style={vscDarkPlus as any}
                        language={match[1]}
                        PreTag="div"
                      >
                        {codeString}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  }
                }}
              >
                {message.content}
              </ReactMarkdown>
            )}
          </div>
          {message.metadata && (
            <div className="message-metadata">
              {message.metadata.model && (
                <Text type="secondary" className="metadata-item">
                  {message.metadata.model}
                </Text>
              )}
              {message.metadata.tokens && (
                <Text type="secondary" className="metadata-item">
                  {message.metadata.tokens} tokens
                </Text>
              )}
              {message.metadata.latency && (
                <Text type="secondary" className="metadata-item">
                  {message.metadata.latency}ms
                </Text>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div ref={containerRef} className="message-list">
      {messages.length === 0 && !loading && (
        <div className="message-list-empty">
          <RobotOutlined className="empty-icon" />
          <Text type="secondary">
            {t('chat.empty', 'Start a conversation by sending a message')}
          </Text>
        </div>
      )}
      
      {messages.map(renderMessage)}
      
      {loading && (
        <div className="message message-assistant">
          <Avatar
            icon={<RobotOutlined />}
            className="message-avatar avatar-assistant"
          />
          <div className="message-content">
            <div className="message-loading">
              <Spin size="small" />
              <Text type="secondary">{t('chat.thinking', 'Thinking...')}</Text>
            </div>
          </div>
        </div>
      )}
      
      <div ref={bottomRef} />
    </div>
  );
};

export default MessageList;
