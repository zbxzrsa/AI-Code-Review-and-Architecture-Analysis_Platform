import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Result, Button, Typography, Space, Card } from 'antd';
import { ReloadOutlined, HomeOutlined, BugOutlined } from '@ant-design/icons';
import { errorLoggingService } from '../../services/errorLogging';

const { Text, Paragraph } = Typography;

/**
 * Error Boundary Props
 * @interface Props
 */
interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

/**
 * Error Boundary State
 * @interface State
 */
interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string | null;
}

/**
 * Error Boundary Component
 * 
 * Catches JavaScript errors anywhere in the child component tree,
 * logs those errors, and displays a fallback UI.
 * 
 * Features:
 * - Catches render errors, event handler errors
 * - Logs to error logging service
 * - User-friendly error messages
 * - Recovery options (retry, reload, go home)
 * - Development mode stack trace
 * - WCAG 2.1 AA compliant
 * 
 * @example
 * ```tsx
 * <ErrorBoundary fallback={<CustomError />}>
 *   <MyComponent />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log to error logging service
    const logEntry = errorLoggingService.logComponentError(error, errorInfo, {
      componentName: this.constructor.name,
    });

    this.setState({ 
      errorInfo,
      errorId: logEntry.id,
    });
    
    // Call optional error handler
    this.props.onError?.(error, errorInfo);
  }

  private readonly handleReload = (): void => {
    globalThis.location.reload();
  };

  private readonly handleGoHome = (): void => {
    globalThis.location.href = '/dashboard';
  };

  private readonly handleRetry = (): void => {
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
