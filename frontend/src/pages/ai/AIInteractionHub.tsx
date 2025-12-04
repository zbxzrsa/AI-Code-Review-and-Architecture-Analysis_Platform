/**
 * AI Interaction Hub
 * Central hub for interacting with all three AI versions
 * 
 * Features:
 * - V1 Experimental AI (shadow testing, new technologies)
 * - V2 Production AI (stable, user-facing)
 * - V3 Comparison AI (baseline, archived models)
 * - Side-by-side comparison
 * - Real-time streaming responses
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Card,
  Typography,
  Space,
  Input,
  Button,
  Avatar,
  Tabs,
  Tag,
  Tooltip,
  Select,
  message,
  Spin,
  Row,
  Col,
  Switch,
  Divider,
  Badge,
  Alert,
  Progress,
  Collapse,
  List,
  Empty,
  Drawer,
  Timeline,
  Statistic,
} from 'antd';
import { useAIVersions, useAIChat, useAICompare, useCycleStatus } from '../../hooks/useAI';
import {
  RobotOutlined,
  UserOutlined,
  SendOutlined,
  CodeOutlined,
  BulbOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
  CopyOutlined,
  DeleteOutlined,
  ExperimentOutlined,
  CheckCircleOutlined,
  HistoryOutlined,
  SyncOutlined,
  CompressOutlined,
  ExpandOutlined,
  SettingOutlined,
  InfoCircleOutlined,
  RocketOutlined,
  BugOutlined,
  DiffOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const { Text, Title, Paragraph } = Typography;
const { TextArea } = Input;
const { Panel } = Collapse;

// Types
interface AIMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  version?: 'v1' | 'v2' | 'v3';
  model?: string;
  latency?: number;
  tokens?: number;
  codeBlocks?: { language: string; code: string }[];
}

interface VersionStatus {
  version: 'v1' | 'v2' | 'v3';
  status: 'online' | 'offline' | 'degraded';
  model: string;
  latency: number;
  accuracy: number;
  lastUpdated: Date;
}

interface ComparisonResult {
  query: string;
  responses: {
    version: string;
    response: string;
    latency: number;
    score: number;
  }[];
  winner: string;
  timestamp: Date;
}

// Mock data
const mockVersionStatus: VersionStatus[] = [
  {
    version: 'v1',
    status: 'online',
    model: 'claude-3.5-sonnet-experimental',
    latency: 450,
    accuracy: 0.82,
    lastUpdated: new Date(),
  },
  {
    version: 'v2',
    status: 'online',
    model: 'gpt-4-turbo',
    latency: 320,
    accuracy: 0.91,
    lastUpdated: new Date(),
  },
  {
    version: 'v3',
    status: 'online',
    model: 'gpt-3.5-turbo-archived',
    latency: 180,
    accuracy: 0.78,
    lastUpdated: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
  },
];

// Version Card Component
const VersionCard: React.FC<{
  version: VersionStatus;
  isActive: boolean;
  onSelect: () => void;
}> = ({ version, isActive, onSelect }) => {
  const { t } = useTranslation();
  
  const getVersionColor = (v: string) => {
    switch (v) {
      case 'v1': return '#722ed1';
      case 'v2': return '#52c41a';
      case 'v3': return '#faad14';
      default: return '#1890ff';
    }
  };

  const getVersionLabel = (v: string) => {
    switch (v) {
      case 'v1': return t('ai.version.experimental', 'Experimental');
      case 'v2': return t('ai.version.production', 'Production');
      case 'v3': return t('ai.version.archive', 'Archive');
      default: return v;
    }
  };

  const getVersionIcon = (v: string) => {
    switch (v) {
      case 'v1': return <ExperimentOutlined />;
      case 'v2': return <CheckCircleOutlined />;
      case 'v3': return <HistoryOutlined />;
      default: return <RobotOutlined />;
    }
  };

  return (
    <Card
      hoverable
      onClick={onSelect}
      style={{
        borderColor: isActive ? getVersionColor(version.version) : undefined,
        borderWidth: isActive ? 2 : 1,
      }}
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        <Space>
          <Avatar
            icon={getVersionIcon(version.version)}
            style={{ backgroundColor: getVersionColor(version.version) }}
          />
          <div>
            <Text strong>{version.version.toUpperCase()}</Text>
            <br />
            <Tag color={getVersionColor(version.version)}>
              {getVersionLabel(version.version)}
            </Tag>
          </div>
          <Badge
            status={version.status === 'online' ? 'success' : version.status === 'degraded' ? 'warning' : 'error'}
          />
        </Space>
        <Divider style={{ margin: '8px 0' }} />
        <Space direction="vertical" size={0} style={{ width: '100%' }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {t('ai.model', 'Model')}: {version.model}
          </Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {t('ai.latency', 'Latency')}: {version.latency}ms
          </Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {t('ai.accuracy', 'Accuracy')}: {(version.accuracy * 100).toFixed(1)}%
          </Text>
        </Space>
      </Space>
    </Card>
  );
};

// Chat Message Component
const ChatMessage: React.FC<{
  msg: AIMessage;
  showVersion?: boolean;
}> = ({ msg, showVersion = false }) => {
  const isUser = msg.role === 'user';

  const getVersionColor = (v?: string) => {
    switch (v) {
      case 'v1': return '#722ed1';
      case 'v2': return '#52c41a';
      case 'v3': return '#faad14';
      default: return '#1890ff';
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    message.success('Copied to clipboard');
  };

  const renderContent = (content: string) => {
    // Simple markdown-like rendering
    const parts = content.split(/(```[\s\S]*?```)/g);
    
    return parts.map((part, index) => {
      if (part.startsWith('```') && part.endsWith('```')) {
        const match = part.match(/```(\w+)?\n?([\s\S]*?)```/);
        if (match) {
          const [, lang = 'text', code] = match;
          return (
            <div key={index} style={{ position: 'relative', marginTop: 8 }}>
              <Button
                size="small"
                icon={<CopyOutlined />}
                style={{ position: 'absolute', right: 8, top: 8, zIndex: 1 }}
                onClick={() => copyToClipboard(code)}
              />
              <SyntaxHighlighter
                language={lang}
                style={oneDark}
                customStyle={{ borderRadius: 8, fontSize: 13 }}
              >
                {code.trim()}
              </SyntaxHighlighter>
            </div>
          );
        }
      }
      return <Paragraph key={index} style={{ marginBottom: 8 }}>{part}</Paragraph>;
    });
  };

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        marginBottom: 16,
      }}
    >
      <Space align="start" style={{ maxWidth: '80%' }}>
        {!isUser && (
          <Avatar
            icon={<RobotOutlined />}
            style={{ backgroundColor: getVersionColor(msg.version) }}
          />
        )}
        <Card
          size="small"
          style={{
            backgroundColor: isUser ? '#e6f7ff' : '#f5f5f5',
            border: 'none',
          }}
        >
          {showVersion && msg.version && (
            <Tag color={getVersionColor(msg.version)} style={{ marginBottom: 8 }}>
              {msg.version.toUpperCase()}
            </Tag>
          )}
          {renderContent(msg.content)}
          <Space style={{ marginTop: 8 }}>
            <Text type="secondary" style={{ fontSize: 11 }}>
              {msg.timestamp.toLocaleTimeString()}
            </Text>
            {msg.latency && (
              <Text type="secondary" style={{ fontSize: 11 }}>
                {msg.latency}ms
              </Text>
            )}
            {msg.tokens && (
              <Text type="secondary" style={{ fontSize: 11 }}>
                {msg.tokens} tokens
              </Text>
            )}
          </Space>
        </Card>
        {isUser && (
          <Avatar icon={<UserOutlined />} style={{ backgroundColor: '#1890ff' }} />
        )}
      </Space>
    </div>
  );
};

// Comparison View Component
const ComparisonView: React.FC<{
  query: string;
  responses: { version: string; response: string; latency: number }[];
}> = ({ query, responses }) => {
  const { t } = useTranslation();

  const getVersionColor = (v: string) => {
    switch (v) {
      case 'v1': return '#722ed1';
      case 'v2': return '#52c41a';
      case 'v3': return '#faad14';
      default: return '#1890ff';
    }
  };

  return (
    <div>
      <Card size="small" style={{ marginBottom: 16, backgroundColor: '#e6f7ff' }}>
        <Text strong>{t('ai.query', 'Query')}: </Text>
        <Text>{query}</Text>
      </Card>
      <Row gutter={16}>
        {responses.map((resp) => (
          <Col key={resp.version} xs={24} md={8}>
            <Card
              title={
                <Space>
                  <Tag color={getVersionColor(resp.version)}>{resp.version.toUpperCase()}</Tag>
                  <Text type="secondary">{resp.latency}ms</Text>
                </Space>
              }
              size="small"
              style={{ height: '100%' }}
            >
              <Paragraph ellipsis={{ rows: 10, expandable: true }}>
                {resp.response}
              </Paragraph>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
};

// Main Component
export const AIInteractionHub: React.FC = () => {
  const { t } = useTranslation();
  const [activeVersion, setActiveVersion] = useState<'v1' | 'v2' | 'v3'>('v2');
  const [messages, setMessages] = useState<AIMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [compareMode, setCompareMode] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [streamingEnabled, setStreamingEnabled] = useState(true);
  const [totalTokens, setTotalTokens] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // API Hooks
  const { data: versions, isLoading: versionsLoading } = useAIVersions();
  const { data: cycleStatus } = useCycleStatus();
  const chatMutation = useAIChat();
  const compareMutation = useAICompare();
  
  const isLoading = chatMutation.isPending || compareMutation.isPending;

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Initialize with welcome message
  useEffect(() => {
    setMessages([
      {
        id: '0',
        role: 'assistant',
        content: `# Welcome to AI Interaction Hub

I'm your AI assistant powered by the **three-version self-evolution system**.

## Available Modes:
- **V1 Experimental**: Test cutting-edge AI models with new capabilities
- **V2 Production**: Stable, reliable AI for everyday use
- **V3 Archive**: Compare against baseline models

## What I can help with:
- **Code Review** - Analyze your code for issues
- **Security Analysis** - Find vulnerabilities
- **Performance Optimization** - Improve efficiency
- **Architecture Advice** - Design better systems

Try asking me anything about your code!`,
        timestamp: new Date(),
        version: 'v2',
      },
    ]);
  }, []);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: AIMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const messageText = inputValue;
    setInputValue('');

    try {
      if (compareMode) {
        // Compare all versions
        const result = await compareMutation.mutateAsync({ message: messageText });
        
        const comparisonResponse: AIMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `## Comparison Results

### V1 (Experimental):
${result.v1.content}

### V2 (Production):
${result.v2.content}

### V3 (Archive):
${result.v3.content}

**Latencies**: V1: ${result.v1.latency}ms | V2: ${result.v2.latency}ms | V3: ${result.v3.latency}ms`,
          timestamp: new Date(),
          version: 'v2',
          latency: Math.max(result.v1.latency || 0, result.v2.latency || 0, result.v3.latency || 0),
        };

        setMessages((prev) => [...prev, comparisonResponse]);
      } else {
        // Single version chat
        const result = await chatMutation.mutateAsync({ 
          message: messageText, 
          version: activeVersion 
        });

        const response: AIMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: result.content,
          timestamp: new Date(),
          version: result.version || activeVersion,
          model: result.model,
          latency: result.latency,
          tokens: result.tokens,
        };

        setTotalTokens((prev) => prev + (result.tokens || 0));
        setMessages((prev) => [...prev, response]);
      }
    } catch (error) {
      // Fallback to mock response if API fails
      const fallbackResponse: AIMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `I apologize, but I'm currently unable to connect to the AI service. Please try again later or check if the backend services are running.

**Troubleshooting:**
1. Ensure the backend services are running (\`docker-compose up -d\`)
2. Check the API endpoint is accessible
3. Verify your authentication token is valid`,
        timestamp: new Date(),
        version: activeVersion,
        latency: 0,
      };
      setMessages((prev) => [...prev, fallbackResponse]);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setMessages([]);
    message.success(t('ai.chat_cleared', 'Chat cleared'));
  };

  const quickPrompts = [
    { icon: <CodeOutlined />, label: t('ai.quick.explain', 'Explain code'), prompt: 'Explain this code:' },
    { icon: <BugOutlined />, label: t('ai.quick.debug', 'Debug issue'), prompt: 'Help me debug:' },
    { icon: <SafetyCertificateOutlined />, label: t('ai.quick.security', 'Security check'), prompt: 'Check for security issues:' },
    { icon: <ThunderboltOutlined />, label: t('ai.quick.optimize', 'Optimize'), prompt: 'Optimize this code:' },
    { icon: <RocketOutlined />, label: t('ai.quick.refactor', 'Refactor'), prompt: 'Refactor this:' },
  ];

  return (
    <div style={{ height: 'calc(100vh - 120px)', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <RobotOutlined style={{ fontSize: 24, color: '#1890ff' }} />
              <Title level={4} style={{ margin: 0 }}>
                {t('ai.hub.title', 'AI Interaction Hub')}
              </Title>
              <Tag color="blue">{t('ai.hub.three_version', 'Three-Version System')}</Tag>
            </Space>
          </Col>
          <Col>
            <Space>
              <Tooltip title={t('ai.compare_mode', 'Compare all versions')}>
                <Switch
                  checkedChildren={<DiffOutlined />}
                  unCheckedChildren={<DiffOutlined />}
                  checked={compareMode}
                  onChange={setCompareMode}
                />
              </Tooltip>
              <Button icon={<DeleteOutlined />} onClick={clearChat}>
                {t('ai.clear', 'Clear')}
              </Button>
              <Button icon={<SettingOutlined />} onClick={() => setShowSettings(true)}>
                {t('ai.settings', 'Settings')}
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      <Row gutter={16} style={{ flex: 1, minHeight: 0 }}>
        {/* Version Selection Sidebar */}
        <Col xs={24} md={6}>
          <Card
            title={t('ai.versions', 'AI Versions')}
            size="small"
            style={{ height: '100%' }}
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              {mockVersionStatus.map((version) => (
                <VersionCard
                  key={version.version}
                  version={version}
                  isActive={activeVersion === version.version}
                  onSelect={() => setActiveVersion(version.version)}
                />
              ))}
            </Space>

            <Divider />

            <Collapse ghost>
              <Panel header={t('ai.quick_prompts', 'Quick Prompts')} key="prompts">
                <Space direction="vertical" style={{ width: '100%' }}>
                  {quickPrompts.map((prompt, index) => (
                    <Button
                      key={index}
                      icon={prompt.icon}
                      block
                      onClick={() => setInputValue(prompt.prompt + ' ')}
                    >
                      {prompt.label}
                    </Button>
                  ))}
                </Space>
              </Panel>
            </Collapse>
          </Card>
        </Col>

        {/* Chat Area */}
        <Col xs={24} md={18}>
          <Card
            style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
            bodyStyle={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 16 }}
          >
            {/* Messages */}
            <div
              style={{
                flex: 1,
                overflowY: 'auto',
                paddingRight: 8,
                marginBottom: 16,
              }}
            >
              {messages.length === 0 ? (
                <Empty description={t('ai.no_messages', 'No messages yet')} />
              ) : (
                messages.map((msg) => (
                  <ChatMessage
                    key={msg.id}
                    msg={msg}
                    showVersion={compareMode}
                  />
                ))
              )}
              {isLoading && (
                <div style={{ textAlign: 'center', padding: 16 }}>
                  <Spin tip={t('ai.thinking', 'AI is thinking...')} />
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div>
              <Space direction="vertical" style={{ width: '100%' }}>
                {compareMode && (
                  <Alert
                    message={t('ai.compare_mode_active', 'Compare mode active - responses from all versions will be shown')}
                    type="info"
                    showIcon
                    closable
                  />
                )}
                <Space.Compact style={{ width: '100%' }}>
                  <TextArea
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder={t('ai.input_placeholder', 'Type your message or paste code...')}
                    autoSize={{ minRows: 2, maxRows: 6 }}
                    disabled={isLoading}
                  />
                </Space.Compact>
                <Row justify="space-between" align="middle">
                  <Col>
                    <Space>
                      <Tag color={activeVersion === 'v1' ? 'purple' : activeVersion === 'v2' ? 'green' : 'orange'}>
                        {t('ai.using', 'Using')}: {activeVersion.toUpperCase()}
                      </Tag>
                      <Text type="secondary">
                        {mockVersionStatus.find((v) => v.version === activeVersion)?.model}
                      </Text>
                    </Space>
                  </Col>
                  <Col>
                    <Button
                      type="primary"
                      icon={<SendOutlined />}
                      onClick={handleSend}
                      loading={isLoading}
                      disabled={!inputValue.trim()}
                    >
                      {t('ai.send', 'Send')}
                    </Button>
                  </Col>
                </Row>
              </Space>
            </div>
          </Card>
        </Col>
      </Row>

      {/* Settings Drawer */}
      <Drawer
        title={t('ai.settings', 'AI Settings')}
        open={showSettings}
        onClose={() => setShowSettings(false)}
        width={400}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <div>
            <Text strong>{t('ai.settings.streaming', 'Streaming Responses')}</Text>
            <br />
            <Switch
              checked={streamingEnabled}
              onChange={setStreamingEnabled}
            />
            <Text type="secondary" style={{ marginLeft: 8 }}>
              {t('ai.settings.streaming_desc', 'Show responses as they are generated')}
            </Text>
          </div>

          <Divider />

          <div>
            <Text strong>{t('ai.settings.version_info', 'Version Information')}</Text>
            <Timeline style={{ marginTop: 16 }}>
              <Timeline.Item color="purple">
                <Text strong>V1 Experimental</Text>
                <br />
                <Text type="secondary">Testing new AI models and features</Text>
              </Timeline.Item>
              <Timeline.Item color="green">
                <Text strong>V2 Production</Text>
                <br />
                <Text type="secondary">Stable, user-facing AI assistant</Text>
              </Timeline.Item>
              <Timeline.Item color="orange">
                <Text strong>V3 Archive</Text>
                <br />
                <Text type="secondary">Baseline for comparison</Text>
              </Timeline.Item>
            </Timeline>
          </div>

          <Divider />

          <div>
            <Text strong>{t('ai.settings.stats', 'Session Statistics')}</Text>
            <Row gutter={16} style={{ marginTop: 16 }}>
              <Col span={12}>
                <Statistic title={t('ai.messages', 'Messages')} value={messages.length} />
              </Col>
              <Col span={12}>
                <Statistic title={t('ai.tokens_used', 'Tokens Used')} value={2450} />
              </Col>
            </Row>
          </div>
        </Space>
      </Drawer>
    </div>
  );
};

export default AIInteractionHub;
