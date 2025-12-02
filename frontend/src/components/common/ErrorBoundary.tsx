import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Result, Button, Typography, Space, Card } from 'antd';
import { ReloadOutlined, HomeOutlined, BugOutlined } from '@ant-design/icons';

const { Text, Paragraph } = Typography;

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * Error Boundary Component
 * 
 * Catches JavaScript errors anywhere in the child component tree,
 * logs those errors, and displays a fallback UI.
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });
    
    // Log error to console
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    // Call optional error handler
    this.props.onError?.(error, errorInfo);
    
    // In production, send to error tracking service
    if (process.env.NODE_ENV === 'production') {
      this.reportError(error, errorInfo);
    }
  }

  private reportError(error: Error, errorInfo: ErrorInfo): void {
    // Send to error tracking service (e.g., Sentry, LogRocket)
    // This is a placeholder for actual implementation
    try {
      const errorReport = {
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        timestamp: new Date().toISOString(),
        url: window.location.href,
        userAgent: navigator.userAgent
      };
      
      // Example: Send to backend error endpoint
      fetch('/api/errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(errorReport)
      }).catch(() => {
        // Silently fail if error reporting fails
      });
    } catch {
      // Silently fail
    }
  }

  private handleReload = (): void => {
    window.location.reload();
  };

  private handleGoHome = (): void => {
    window.location.href = '/dashboard';
  };

  private handleRetry = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };

  render(): ReactNode {
    const { hasError, error, errorInfo } = this.state;
    const { children, fallback } = this.props;

    if (hasError) {
      // Return custom fallback if provided
      if (fallback) {
        return fallback;
      }

      // Default error UI
      return (
        <div style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '24px',
          background: '#f5f5f5'
        }}>
          <Card style={{ maxWidth: 600, width: '100%' }}>
            <Result
              status="error"
              title="Something went wrong"
              subTitle="We're sorry, but something unexpected happened. Please try again."
              extra={
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  <Space>
                    <Button
                      type="primary"
                      icon={<ReloadOutlined />}
                      onClick={this.handleRetry}
                    >
                      Try Again
                    </Button>
                    <Button
                      icon={<ReloadOutlined />}
                      onClick={this.handleReload}
                    >
                      Reload Page
                    </Button>
                    <Button
                      icon={<HomeOutlined />}
                      onClick={this.handleGoHome}
                    >
                      Go to Dashboard
                    </Button>
                  </Space>
                  
                  {process.env.NODE_ENV === 'development' && error && (
                    <Card
                      size="small"
                      title={
                        <Space>
                          <BugOutlined />
                          <Text strong>Error Details (Development Only)</Text>
                        </Space>
                      }
                      style={{ textAlign: 'left', marginTop: 16 }}
                    >
                      <Paragraph>
                        <Text strong>Error: </Text>
                        <Text type="danger">{error.message}</Text>
                      </Paragraph>
                      
                      {error.stack && (
                        <Paragraph>
                          <Text strong>Stack Trace:</Text>
                          <pre style={{
                            background: '#f0f0f0',
                            padding: 8,
                            borderRadius: 4,
                            overflow: 'auto',
                            fontSize: 12,
                            maxHeight: 200
                          }}>
                            {error.stack}
                          </pre>
                        </Paragraph>
                      )}
                      
                      {errorInfo?.componentStack && (
                        <Paragraph>
                          <Text strong>Component Stack:</Text>
                          <pre style={{
                            background: '#f0f0f0',
                            padding: 8,
                            borderRadius: 4,
                            overflow: 'auto',
                            fontSize: 12,
                            maxHeight: 150
                          }}>
                            {errorInfo.componentStack}
                          </pre>
                        </Paragraph>
                      )}
                    </Card>
                  )}
                </Space>
              }
            />
          </Card>
        </div>
      );
    }

    return children;
  }
}

export default ErrorBoundary;
