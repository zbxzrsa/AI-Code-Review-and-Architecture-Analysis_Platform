/**
 * OAuth Callback Page
 * 
 * Handles OAuth callback from GitHub/GitLab and redirects user appropriately.
 * Shows loading state while processing and error state if authentication fails.
 */
import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate, useSearchParams, useParams } from 'react-router-dom';
import { Result, Spin, Card, Button, Typography, Space, Alert, Progress } from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
  GithubOutlined,
  GitlabOutlined,
  LinkOutlined,
} from '@ant-design/icons';
import { useAuth } from '../hooks/useAuth';

const { Title, Text, Paragraph } = Typography;

// Provider icons
const providerIcons: Record<string, React.ReactNode> = {
  github: <GithubOutlined style={{ fontSize: 48 }} />,
  gitlab: <GitlabOutlined style={{ fontSize: 48 }} />,
  bitbucket: <LinkOutlined style={{ fontSize: 48 }} />,
};

// Provider display names
const providerNames: Record<string, string> = {
  github: 'GitHub',
  gitlab: 'GitLab',
  bitbucket: 'Bitbucket',
};

type CallbackState = 'loading' | 'success' | 'error' | 'processing';

interface CallbackResult {
  success: boolean;
  message: string;
  isNewUser?: boolean;
  requiresSetup?: boolean;
}

const OAuthCallback: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { provider } = useParams<{ provider: string }>();
  const { login } = useAuth();
  
  const [state, setState] = useState<CallbackState>('loading');
  const [result, setResult] = useState<CallbackResult | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Extract parameters from URL
  const code = searchParams.get('code');
  const stateParam = searchParams.get('state');
  const errorParam = searchParams.get('error');
  const errorDescription = searchParams.get('error_description');
  const oauthSuccess = searchParams.get('oauth_success');
  const oauthError = searchParams.get('oauth_error');
  const isNewUser = searchParams.get('new_user') === 'true';

  const processCallback = useCallback(async () => {
    // Check for OAuth errors in URL
    if (errorParam || oauthError) {
      setError(errorDescription || oauthError || 'Authentication failed');
      setState('error');
      return;
    }

    // Check for success redirect from backend
    if (oauthSuccess === 'true') {
      setState('processing');
      setProgress(50);
      
      // OAuth was successful (backend handled it)
      setProgress(100);
      
      setResult({
        success: true,
        message: isNewUser 
          ? 'Account created successfully!' 
          : 'Connected successfully!',
        isNewUser,
      });
      setState('success');
      
      // Redirect after short delay
      setTimeout(() => {
        if (isNewUser) {
          navigate('/settings/profile', { replace: true });
        } else {
          navigate('/repositories', { replace: true });
        }
      }, 2000);
      
      return;
    }

    // Process authorization code
    if (code && stateParam) {
      setState('processing');
      setProgress(25);
      
      try {
        // The backend handles the code exchange via redirect
        // This code path is for direct API calls if needed
        const response = await fetch(
          `/api/auth/oauth/callback/${provider}?code=${code}&state=${stateParam}`,
          {
            method: 'GET',
            credentials: 'include',
          }
        );
        
        setProgress(75);
        
        if (response.ok) {
          const data = await response.json();
          setProgress(100);
          
          setResult({
            success: true,
            message: data.is_new_user 
              ? 'Account created successfully!' 
              : 'Connected successfully!',
            isNewUser: data.is_new_user,
          });
          setState('success');
          
          // Redirect
          setTimeout(() => {
            navigate(data.is_new_user ? '/settings/profile' : '/repositories', { 
              replace: true 
            });
          }, 2000);
          
        } else {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Authentication failed');
        }
        
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Authentication failed');
        setState('error');
      }
      return;
    }

    // No valid parameters
    setError('Invalid callback parameters');
    setState('error');
  }, [
    code, 
    stateParam, 
    errorParam, 
    errorDescription, 
    oauthSuccess, 
    oauthError, 
    isNewUser, 
    provider, 
    navigate
  ]);

  useEffect(() => {
    processCallback();
  }, [processCallback]);

  const handleRetry = () => {
    // Redirect to OAuth initiation
    window.location.href = `/api/auth/oauth/connect/${provider}?return_url=${encodeURIComponent(window.location.origin + '/repositories')}`;
  };

  const handleGoBack = () => {
    navigate('/repositories', { replace: true });
  };

  const handleGoHome = () => {
    navigate('/', { replace: true });
  };

  const providerName = providerNames[provider || ''] || provider || 'OAuth';
  const providerIcon = providerIcons[provider || ''] || <LinkOutlined style={{ fontSize: 48 }} />;

  // Loading state
  if (state === 'loading' || state === 'processing') {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      }}>
        <Card 
          style={{ 
            width: 400, 
            textAlign: 'center',
            borderRadius: 12,
            boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
          }}
        >
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <div style={{ color: '#1890ff' }}>
              {providerIcon}
            </div>
            
            <Title level={4} style={{ margin: 0 }}>
              Connecting to {providerName}
            </Title>
            
            <Spin 
              indicator={<LoadingOutlined style={{ fontSize: 32 }} spin />} 
            />
            
            {state === 'processing' && (
              <Progress 
                percent={progress} 
                status="active" 
                strokeColor={{
                  '0%': '#108ee9',
                  '100%': '#87d068',
                }}
              />
            )}
            
            <Text type="secondary">
              {state === 'loading' 
                ? 'Initializing authentication...'
                : 'Processing your request...'}
            </Text>
          </Space>
        </Card>
      </div>
    );
  }

  // Success state
  if (state === 'success' && result) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      }}>
        <Card 
          style={{ 
            width: 450, 
            textAlign: 'center',
            borderRadius: 12,
            boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
          }}
        >
          <Result
            icon={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
            status="success"
            title={result.message}
            subTitle={
              result.isNewUser
                ? "Welcome! Let's set up your profile."
                : `Your ${providerName} account has been connected.`
            }
            extra={[
              <Button 
                type="primary" 
                key="continue"
                onClick={() => navigate(
                  result.isNewUser ? '/settings/profile' : '/repositories',
                  { replace: true }
                )}
              >
                {result.isNewUser ? 'Set Up Profile' : 'Go to Repositories'}
              </Button>,
            ]}
          />
          
          <Text type="secondary">
            Redirecting automatically in 2 seconds...
          </Text>
        </Card>
      </div>
    );
  }

  // Error state
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    }}>
      <Card 
        style={{ 
          width: 500, 
          textAlign: 'center',
          borderRadius: 12,
          boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
        }}
      >
        <Result
          icon={<CloseCircleOutlined style={{ color: '#ff4d4f' }} />}
          status="error"
          title="Authentication Failed"
          subTitle={`Unable to connect to ${providerName}`}
          extra={[
            <Button type="primary" key="retry" onClick={handleRetry}>
              Try Again
            </Button>,
            <Button key="back" onClick={handleGoBack}>
              Go Back
            </Button>,
          ]}
        />
        
        {error && (
          <Alert
            message="Error Details"
            description={error}
            type="error"
            showIcon
            style={{ marginTop: 16, textAlign: 'left' }}
          />
        )}
        
        <Paragraph type="secondary" style={{ marginTop: 24 }}>
          If this problem persists, please try:
        </Paragraph>
        <ul style={{ textAlign: 'left', color: '#666' }}>
          <li>Clearing your browser cookies</li>
          <li>Using a different browser</li>
          <li>Checking your {providerName} account permissions</li>
          <li>Contacting support if the issue continues</li>
        </ul>
      </Card>
    </div>
  );
};

export default OAuthCallback;
