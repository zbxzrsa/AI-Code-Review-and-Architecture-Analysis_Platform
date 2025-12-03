import React, { useEffect, useState, useCallback } from 'react';
import { useSSE } from '../../../hooks/useSSE';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useTranslation } from 'react-i18next';
import { Alert, Spin, Button } from 'antd';
import { ReloadOutlined, CopyOutlined, CheckOutlined } from '@ant-design/icons';
import './StreamingResponse.css';

export interface AnalysisResult {
  id: string;
  issues: Array<{
    type: string;
    severity: string;
    line_start: number;
    line_end?: number;
    description: string;
    fix?: string;
  }>;
  summary: string;
  metrics: {
    complexity: number;
    maintainability: number;
    security_score: number;
  };
  suggestions: string[];
}

interface StreamingResponseProps {
  sessionId: string;
  onComplete?: (result: AnalysisResult) => void;
  onError?: (error: Error) => void;
  autoScroll?: boolean;
}

interface SSEData {
  type: 'delta' | 'complete' | 'error' | 'thinking';
  content?: string;
  result?: AnalysisResult;
  error?: string;
}

export const StreamingResponse: React.FC<StreamingResponseProps> = ({
  sessionId,
  onComplete,
  onError,
  autoScroll = true
}) => {
  const [content, setContent] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [copied, setCopied] = useState(false);
  const { t } = useTranslation();
  const containerRef = React.useRef<HTMLDivElement>(null);

  const { data, error, isConnected, reconnect } = useSSE<SSEData>(
    `/api/analyze/${sessionId}/stream`
  );

  // Handle SSE data
  useEffect(() => {
    if (!data) return;

    switch (data.type) {
      case 'thinking':
        setIsThinking(true);
        break;
      case 'delta':
        setIsThinking(false);
        if (data.content) {
          setContent(prev => prev + data.content);
        }
        break;
      case 'complete':
        setIsComplete(true);
        setIsThinking(false);
        if (data.result && onComplete) {
          onComplete(data.result);
        }
        break;
      case 'error':
        setIsThinking(false);
        if (onError && data.error) {
          onError(new Error(data.error));
        }
        break;
    }
  }, [data, onComplete, onError]);

  // Handle connection error
  useEffect(() => {
    if (error && onError) {
      onError(error);
    }
  }, [error, onError]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [content, autoScroll]);

  // Copy content to clipboard
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, [content]);

  return (
    <div className="streaming-response" ref={containerRef}>
      {/* Toolbar */}
      <div className="streaming-toolbar">
        <Button
          size="small"
          icon={copied ? <CheckOutlined /> : <CopyOutlined />}
          onClick={handleCopy}
          disabled={!content}
        >
          {copied ? t('common.copied') : t('common.copy')}
        </Button>
        {error && (
          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={reconnect}
          >
            {t('common.retry')}
          </Button>
        )}
      </div>

      {/* Thinking indicator */}
      {isThinking && (
        <div className="thinking-indicator">
          <Spin size="small" />
          <span>{t('streaming.thinking')}</span>
        </div>
      )}

      {/* Content */}
      <div className="streaming-content">
        <ReactMarkdown
          components={{
            code({ node: _node, className, children, ...props }: any) {
              const match = /language-(\w+)/.exec(className || '');
              const codeString = String(children).replace(/\n$/, '');
              const isInline = !match;
              
              return !isInline && match ? (
                <div className="code-block-wrapper">
                  <div className="code-block-header">
                    <span className="code-language">{match[1]}</span>
                    <Button
                      size="small"
                      type="text"
                      icon={<CopyOutlined />}
                      onClick={() => navigator.clipboard.writeText(codeString)}
                    />
                  </div>
                  <SyntaxHighlighter
                    style={vscDarkPlus as any}
                    language={match[1]}
                    PreTag="div"
                  >
                    {codeString}
                  </SyntaxHighlighter>
                </div>
              ) : (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            },
            // Custom rendering for other elements
            h1: ({ children }) => <h1 className="md-h1">{children}</h1>,
            h2: ({ children }) => <h2 className="md-h2">{children}</h2>,
            h3: ({ children }) => <h3 className="md-h3">{children}</h3>,
            ul: ({ children }) => <ul className="md-ul">{children}</ul>,
            ol: ({ children }) => <ol className="md-ol">{children}</ol>,
            blockquote: ({ children }) => (
              <blockquote className="md-blockquote">{children}</blockquote>
            ),
            table: ({ children }) => (
              <div className="md-table-wrapper">
                <table className="md-table">{children}</table>
              </div>
            )
          }}
        >
          {content}
        </ReactMarkdown>
      </div>

      {/* Typing indicator */}
      {!isComplete && !isThinking && isConnected && (
        <div className="typing-indicator">
          <span></span>
          <span></span>
          <span></span>
        </div>
      )}

      {/* Error display */}
      {error && (
        <Alert
          type="error"
          message={t('streaming.error')}
          description={error.message}
          showIcon
          className="streaming-error"
        />
      )}

      {/* Connection status */}
      {!isConnected && !isComplete && !error && (
        <Alert
          type="warning"
          message={t('streaming.disconnected')}
          description={t('streaming.reconnecting')}
          showIcon
          className="streaming-warning"
        />
      )}
    </div>
  );
};

export default StreamingResponse;
