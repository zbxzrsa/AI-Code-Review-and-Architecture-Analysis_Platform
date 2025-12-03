import React, { useState, useCallback, useRef } from 'react';
import { Input, Button, Space, Tooltip, Upload, Dropdown, Typography } from 'antd';
import {
  SendOutlined,
  PaperClipOutlined,
  ClearOutlined,
  SettingOutlined,
  StopOutlined
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { MessageList, Message } from '../MessageList';
import './ChatInterface.css';

const { TextArea } = Input;
const { Text } = Typography;

interface ChatInterfaceProps {
  sessionId?: string;
  onSendMessage?: (message: string) => Promise<void>;
  onClear?: () => void;
  placeholder?: string;
  maxLength?: number;
  disabled?: boolean;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  sessionId,
  onSendMessage,
  onClear,
  placeholder,
  maxLength = 4000,
  disabled = false
}) => {
  const { t } = useTranslation();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Handle sending message
  const handleSend = useCallback(async () => {
    if (!input.trim() || isLoading || disabled) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      if (onSendMessage) {
        await onSendMessage(userMessage.content);
      } else {
        // Default streaming behavior
        setIsStreaming(true);
        abortControllerRef.current = new AbortController();

        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true
        };

        setMessages((prev) => [...prev, assistantMessage]);

        // Simulate streaming (replace with actual API call)
        const response = await fetch(`/api/chat/${sessionId || 'default'}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMessage.content }),
          signal: abortControllerRef.current.signal
        });

        if (!response.body) throw new Error('No response body');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let content = '';

        let done = false;
        while (!done) {
          const result = await reader.read();
          done = result.done;
          if (done) break;

          const chunk = decoder.decode(result.value, { stream: true });
          content += chunk;

          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessage.id
                ? { ...msg, content }
                : msg
            )
          );
        }

        // Mark as complete
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessage.id
              ? { ...msg, isStreaming: false }
              : msg
          )
        );
      }
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        const errorMessage: Message = {
          id: `system-${Date.now()}`,
          role: 'system',
          content: t('chat.error', 'An error occurred. Please try again.'),
          timestamp: new Date()
        };
        setMessages((prev) => [...prev, errorMessage]);
      }
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  }, [input, isLoading, disabled, onSendMessage, sessionId, t]);

  // Handle stop streaming
  const handleStop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  // Handle clear chat
  const handleClear = useCallback(() => {
    setMessages([]);
    onClear?.();
  }, [onClear]);

  // Handle key press
  const handleKeyPress = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  // Handle file upload
  const handleFileUpload = useCallback((file: File) => {
    // Read file content
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      setInput((prev) => prev + `\n\`\`\`\n${content}\n\`\`\``);
      inputRef.current?.focus();
    };
    reader.readAsText(file);
    return false; // Prevent default upload
  }, []);

  const settingsMenu = {
    items: [
      {
        key: 'clear',
        label: t('chat.clear_history', 'Clear History'),
        icon: <ClearOutlined />,
        onClick: handleClear
      }
    ]
  };

  return (
    <div className="chat-interface">
      <div className="chat-messages">
        <MessageList
          messages={messages}
          loading={isLoading && !isStreaming}
          autoScroll
        />
      </div>

      <div className="chat-input-container">
        <div className="chat-input-wrapper">
          <TextArea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={placeholder || t('chat.placeholder', 'Type your message...')}
            autoSize={{ minRows: 1, maxRows: 6 }}
            maxLength={maxLength}
            disabled={disabled || isLoading}
            className="chat-input"
          />
          
          <div className="chat-input-actions">
            <Space>
              <Upload
                beforeUpload={handleFileUpload}
                showUploadList={false}
                accept=".txt,.py,.js,.ts,.json,.md"
              >
                <Tooltip title={t('chat.attach_file', 'Attach file')}>
                  <Button
                    icon={<PaperClipOutlined />}
                    type="text"
                    disabled={disabled || isLoading}
                  />
                </Tooltip>
              </Upload>
              
              <Dropdown menu={settingsMenu} trigger={['click']}>
                <Button icon={<SettingOutlined />} type="text" />
              </Dropdown>
            </Space>
            
            <Space>
              {input.length > 0 && (
                <Text type="secondary" className="char-count">
                  {input.length}/{maxLength}
                </Text>
              )}
              
              {isStreaming ? (
                <Button
                  icon={<StopOutlined />}
                  onClick={handleStop}
                  danger
                >
                  {t('chat.stop', 'Stop')}
                </Button>
              ) : (
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  onClick={handleSend}
                  disabled={!input.trim() || disabled || isLoading}
                  loading={isLoading && !isStreaming}
                >
                  {t('chat.send', 'Send')}
                </Button>
              )}
            </Space>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
