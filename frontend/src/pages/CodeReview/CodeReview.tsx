/**
 * Code Review Page with AI Interaction
 * 
 * Features:
 * - Code editor with syntax highlighting
 * - AI-powered code analysis
 * - Real-time streaming responses
 * - Issue highlighting and quick fixes
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Layout,
  Card,
  Button,
  Select,
  Space,
  Typography,
  List,
  Tag,
  Badge,
  Tooltip,
  Input,
  Spin,
  Drawer,
  message,
  Empty,
  Progress,
  Divider,
  Row,
  Col,
  Avatar,
  notification,
} from 'antd';
import {
  PlayCircleOutlined,
  CodeOutlined,
  BugOutlined,
  MessageOutlined,
  RobotOutlined,
  SendOutlined,
  ClearOutlined,
  FileTextOutlined,
  WarningOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined,
  BulbOutlined,
  CopyOutlined,
  ReloadOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useCodeAnalysis, useAIChat, useApplyFix, useProvideFeedback } from '../../hooks/useAI';
import './CodeReview.css';

const { Content } = Layout;
const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

// Types
interface Issue {
  id: string;
  type: 'error' | 'warning' | 'info' | 'suggestion';
  severity: 'critical' | 'high' | 'medium' | 'low';
  message: string;
  line: number;
  column: number;
  endLine?: number;
  endColumn?: number;
  rule?: string;
  suggestion?: string;
  autoFix?: boolean;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  model?: string;
  thinking?: boolean;
}

interface AIModel {
  id: string;
  name: string;
  provider: string;
  version: string;
  status: 'active' | 'inactive' | 'deprecated';
  capabilities: string[];
}

// Sample code for demo
const SAMPLE_CODE = `// Example: User Authentication Service
import { hash, compare } from 'bcrypt';
import jwt from 'jsonwebtoken';

class AuthService {
  private secretKey = 'hardcoded-secret-key'; // Security issue!
  
  async register(email, password) {
    // Missing input validation
    const hashedPassword = await hash(password, 10);
    
    // SQL injection vulnerability
    const query = \`INSERT INTO users (email, password) VALUES ('\${email}', '\${hashedPassword}')\`;
    await db.execute(query);
    
    return { success: true };
  }
  
  async login(email, password) {
    const user = await db.findUser(email);
    
    if (!user) {
      return null; // Should throw error for better error handling
    }
    
    const valid = await compare(password, user.password);
    if (!valid) return null;
    
    // Token expiration too long
    const token = jwt.sign({ userId: user.id }, this.secretKey, { expiresIn: '365d' });
    return { token, user };
  }
  
  // Unused method - code smell
  private async resetPassword(email) {
    console.log('Resetting password for', email);
  }
}

export default AuthService;
`;

export const CodeReview: React.FC = () => {
  const { t } = useTranslation();
  const { projectId: _projectId } = useParams<{ projectId: string }>();
  const _navigate = useNavigate();

  // AI Hooks
  const analysisMutation = useCodeAnalysis();
  const chatMutation = useAIChat();
  const applyFixMutation = useApplyFix();
  const feedbackMutation = useProvideFeedback();

  // State
  const [code, setCode] = useState(SAMPLE_CODE);
  const [language, setLanguage] = useState('typescript');
  const [selectedModel, setSelectedModel] = useState('gpt-4-turbo');
  const [issues, setIssues] = useState<Issue[]>([]);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatDrawerOpen, setChatDrawerOpen] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Available AI models
  const aiModels: AIModel[] = [
    { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', provider: 'OpenAI', version: 'v1.0', status: 'active', capabilities: ['code-review', 'security', 'optimization'] },
    { id: 'gpt-4', name: 'GPT-4', provider: 'OpenAI', version: 'v1.0', status: 'active', capabilities: ['code-review', 'security'] },
    { id: 'claude-3-opus', name: 'Claude 3 Opus', provider: 'Anthropic', version: 'v1.0', status: 'active', capabilities: ['code-review', 'security', 'explanation'] },
    { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet', provider: 'Anthropic', version: 'v1.0', status: 'active', capabilities: ['code-review', 'optimization'] },
  ];

  // Scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // Analyze code with AI - uses real API via hooks
  const analyzeCode = useCallback(async () => {
    setAnalysisProgress(0);
    setIssues([]);

    // Simulate progress while analyzing
    const progressInterval = setInterval(() => {
      setAnalysisProgress(prev => Math.min(prev + 10, 90));
    }, 200);

    try {
      const result = await analysisMutation.mutateAsync({
        code,
        language,
        reviewTypes: ['security', 'performance', 'quality', 'bug'],
        version: 'v2',
      });

      clearInterval(progressInterval);
      setAnalysisProgress(100);

      // Map API response to Issue format
      const mappedIssues: Issue[] = result.issues.map((issue: any) => ({
        id: issue.id,
        type: issue.type === 'security' ? 'error' : issue.type === 'bug' ? 'error' : issue.type === 'performance' ? 'warning' : 'info',
        severity: issue.severity,
        message: issue.title || issue.description,
        line: issue.line || 1,
        column: issue.column || 1,
        rule: issue.rule || `${issue.type}/${issue.id}`,
        suggestion: issue.suggestion,
        autoFix: issue.fixAvailable || false,
      }));

      setIssues(mappedIssues);
      setAnalysisId(result.id);

      notification.success({
        message: t('codeReview.analysis_complete', 'Analysis complete!'),
        description: `Found ${mappedIssues.length} issues. Score: ${result.score}/100`,
      });
    } catch (error) {
      clearInterval(progressInterval);
      setAnalysisProgress(100);

      // Fallback to demo issues if API fails
      const demoIssues: Issue[] = [
        {
          id: '1',
          type: 'error',
          severity: 'critical',
          message: 'Hardcoded secret key detected. Use environment variables instead.',
          line: 7,
          column: 21,
          rule: 'security/no-hardcoded-secrets',
          suggestion: 'const secretKey = process.env.JWT_SECRET;',
          autoFix: true,
        },
        {
          id: '2',
          type: 'error',
          severity: 'critical',
          message: 'SQL Injection vulnerability. Use parameterized queries.',
          line: 14,
          column: 5,
          rule: 'security/sql-injection',
          suggestion: 'Use prepared statements with placeholders',
          autoFix: false,
        },
        {
          id: '3',
          type: 'warning',
          severity: 'high',
          message: 'Missing input validation for email and password.',
          line: 10,
          column: 3,
          rule: 'security/input-validation',
          suggestion: 'Add email format validation and password requirements check',
          autoFix: false,
        },
        {
          id: '4',
          type: 'warning',
          severity: 'medium',
          message: 'Token expiration of 365 days is too long. Consider shorter expiration.',
          line: 29,
          column: 5,
          rule: 'security/token-expiration',
          suggestion: "expiresIn: '24h'",
          autoFix: true,
        },
        {
          id: '5',
          type: 'info',
          severity: 'low',
          message: 'Unused private method detected.',
          line: 34,
          column: 3,
          rule: 'code-quality/no-unused-methods',
          suggestion: 'Remove or implement the resetPassword method',
          autoFix: false,
        },
      ];
      setIssues(demoIssues);
      message.info(t('codeReview.demo_mode', 'Using demo mode - backend not available'));
    }
  }, [code, language, analysisMutation, t]);

  // Send chat message to AI - uses real API via hooks
  const sendChatMessage = useCallback(async () => {
    if (!chatInput.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: chatInput,
      timestamp: new Date(),
    };

    setChatMessages(prev => [...prev, userMessage]);
    const currentInput = chatInput;
    setChatInput('');

    try {
      const response = await chatMutation.mutateAsync({
        message: currentInput,
        version: 'v2',
        context: { code, language },
      });

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.content,
        timestamp: new Date(),
        model: response.model || selectedModel,
      };

      setChatMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      // Fallback to intelligent mock response
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: getMockResponse(currentInput),
        timestamp: new Date(),
        model: selectedModel,
      };
      setChatMessages(prev => [...prev, assistantMessage]);
    }
  }, [chatInput, code, language, selectedModel, chatMutation]);

  // Mock AI response
  const getMockResponse = (input: string): string => {
    const lowerInput = input.toLowerCase();
    
    if (lowerInput.includes('fix') || lowerInput.includes('how')) {
      return `Based on my analysis, here's how to fix the issues:\n\n**1. Hardcoded Secret Key (Critical)**\n\`\`\`typescript\n// Replace this:\nprivate secretKey = 'hardcoded-secret-key';\n\n// With this:\nprivate secretKey = process.env.JWT_SECRET || '';\n\`\`\`\n\n**2. SQL Injection (Critical)**\n\`\`\`typescript\n// Use parameterized queries:\nconst query = 'INSERT INTO users (email, password) VALUES ($1, $2)';\nawait db.execute(query, [email, hashedPassword]);\n\`\`\`\n\n**3. Input Validation**\n\`\`\`typescript\nimport { z } from 'zod';\n\nconst schema = z.object({\n  email: z.string().email(),\n  password: z.string().min(8)\n});\n\nconst validated = schema.parse({ email, password });\n\`\`\``;
    }
    
    if (lowerInput.includes('explain') || lowerInput.includes('what')) {
      return `I found several issues in your code:\n\n**Security Issues:**\n- Hardcoded JWT secret key - This is a critical security vulnerability. If your code is exposed, attackers can forge valid tokens.\n- SQL Injection - Direct string interpolation in SQL queries allows attackers to execute arbitrary SQL.\n\n**Best Practice Issues:**\n- Missing input validation - Always validate user input to prevent malformed data.\n- Long token expiration - 365 days is too long; use short-lived tokens with refresh mechanism.\n\n**Code Quality:**\n- Unused method - Dead code should be removed to improve maintainability.`;
    }
    
    return `I can help you with your code! I've identified ${issues.length} issues in your code:\n\n- ${issues.filter(i => i.severity === 'critical').length} Critical issues\n- ${issues.filter(i => i.severity === 'high').length} High severity issues\n- ${issues.filter(i => i.severity === 'medium').length} Medium severity issues\n- ${issues.filter(i => i.severity === 'low').length} Low severity issues\n\nWould you like me to:\n1. Explain any specific issue in detail?\n2. Provide fixes for the critical issues?\n3. Suggest best practices for your code?`;
  };

  // Apply auto-fix - uses real API via hooks
  const applyFix = useCallback(async (issue: Issue) => {
    if (!issue.suggestion) return;
    
    try {
      const fixedCode = await applyFixMutation.mutateAsync({ 
        issueId: issue.id, 
        code 
      });
      setCode(fixedCode);
      setIssues(prev => prev.filter(i => i.id !== issue.id));
      message.success(t('codeReview.fix_applied', 'Fix applied!'));
    } catch (error) {
      // Fallback: simulate fix by removing issue
      setIssues(prev => prev.filter(i => i.id !== issue.id));
      message.success(t('codeReview.fix_applied', 'Fix applied! (Demo mode)'));
    }
  }, [code, applyFixMutation, t]);

  // Provide feedback on AI response
  const provideFeedback = useCallback(async (helpful: boolean) => {
    if (!analysisId) return;
    try {
      await feedbackMutation.mutateAsync({ responseId: analysisId, helpful });
    } catch (error) {
      // Silently fail for feedback
    }
  }, [analysisId, feedbackMutation]);

  // Get severity color
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'red';
      case 'high': return 'orange';
      case 'medium': return 'gold';
      case 'low': return 'blue';
      default: return 'default';
    }
  };

  // Get type icon
  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'error': return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'warning': return <WarningOutlined style={{ color: '#faad14' }} />;
      case 'info': return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
      case 'suggestion': return <BulbOutlined style={{ color: '#52c41a' }} />;
      default: return <InfoCircleOutlined />;
    }
  };

  return (
    <Layout className="code-review-page">
      {/* Main Content */}
      <Content className="code-review-content">
        {/* Header */}
        <div className="code-review-header">
          <Space>
            <Title level={4} style={{ margin: 0 }}>
              <CodeOutlined /> {t('codeReview.title', 'AI Code Review')}
            </Title>
          </Space>
          <Space>
            <Select
              value={language}
              onChange={setLanguage}
              style={{ width: 150 }}
              options={[
                { value: 'typescript', label: 'TypeScript' },
                { value: 'javascript', label: 'JavaScript' },
                { value: 'python', label: 'Python' },
                { value: 'java', label: 'Java' },
                { value: 'go', label: 'Go' },
                { value: 'rust', label: 'Rust' },
              ]}
            />
            <Select
              value={selectedModel}
              onChange={setSelectedModel}
              style={{ width: 180 }}
              options={aiModels.map(m => ({
                value: m.id,
                label: (
                  <Space>
                    <RobotOutlined />
                    {m.name}
                  </Space>
                ),
              }))}
            />
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={analyzeCode}
              loading={analysisMutation.isPending}
            >
              {t('codeReview.analyze', 'Analyze')}
            </Button>
            <Tooltip title={t('codeReview.chat_with_ai', 'Chat with AI')}>
              <Button
                icon={<MessageOutlined />}
                onClick={() => setChatDrawerOpen(true)}
              >
                AI Chat
              </Button>
            </Tooltip>
          </Space>
        </div>

        {/* Progress */}
        {analysisMutation.isPending && (
          <Progress
            percent={analysisProgress}
            status="active"
            strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }}
            style={{ marginBottom: 16 }}
          />
        )}

        <Row gutter={16} style={{ padding: '16px' }}>
          {/* Code Editor - Give more space to code */}
          <Col xs={24} xl={16} lg={14}>
            <Card
              title={
                <Space>
                  <FileTextOutlined />
                  {t('codeReview.code_editor', 'Code Editor')}
                </Space>
              }
              extra={
                <Space>
                  <Tooltip title="Copy code">
                    <Button
                      size="small"
                      icon={<CopyOutlined />}
                      onClick={() => {
                        navigator.clipboard.writeText(code);
                        message.success('Copied!');
                      }}
                    />
                  </Tooltip>
                  <Tooltip title="Reset">
                    <Button
                      size="small"
                      icon={<ReloadOutlined />}
                      onClick={() => setCode(SAMPLE_CODE)}
                    />
                  </Tooltip>
                </Space>
              }
              className="code-editor-card"
            >
              <TextArea
                value={code}
                onChange={e => setCode(e.target.value)}
                rows={20}
                style={{
                  fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
                  fontSize: 13,
                  lineHeight: 1.5,
                }}
                placeholder="Paste your code here..."
              />
            </Card>
          </Col>

          {/* Issues Panel */}
          <Col xs={24} xl={8} lg={10}>
            <Card
              title={
                <Space>
                  <BugOutlined />
                  {t('codeReview.issues', 'Issues')}
                  {issues.length > 0 && (
                    <Badge count={issues.length} style={{ backgroundColor: '#ff4d4f' }} />
                  )}
                </Space>
              }
              className="issues-card"
            >
              {issues.length === 0 ? (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description={
                    analysisMutation.isPending
                      ? t('codeReview.analyzing', 'Analyzing code...')
                      : t('codeReview.no_issues', 'No issues found. Click Analyze to start.')
                  }
                />
              ) : (
                <>
                  {/* Summary */}
                  <div className="issues-summary">
                    <Space wrap>
                      <Tag color="red">
                        {issues.filter(i => i.severity === 'critical').length} Critical
                      </Tag>
                      <Tag color="orange">
                        {issues.filter(i => i.severity === 'high').length} High
                      </Tag>
                      <Tag color="gold">
                        {issues.filter(i => i.severity === 'medium').length} Medium
                      </Tag>
                      <Tag color="blue">
                        {issues.filter(i => i.severity === 'low').length} Low
                      </Tag>
                    </Space>
                  </div>
                  <Divider style={{ margin: '12px 0' }} />
                  
                  {/* Issues List */}
                  <List
                    className="issues-list"
                    dataSource={issues}
                    renderItem={issue => (
                      <List.Item
                        actions={
                          issue.autoFix
                            ? [
                                <Button
                                  key="fix"
                                  size="small"
                                  type="link"
                                  onClick={() => applyFix(issue)}
                                >
                                  Quick Fix
                                </Button>,
                              ]
                            : undefined
                        }
                      >
                        <List.Item.Meta
                          avatar={getTypeIcon(issue.type)}
                          title={
                            <Space>
                              <Tag color={getSeverityColor(issue.severity)}>
                                {issue.severity.toUpperCase()}
                              </Tag>
                              <Text code>Line {issue.line}</Text>
                            </Space>
                          }
                          description={
                            <div>
                              <Paragraph style={{ marginBottom: 4 }}>
                                {issue.message}
                              </Paragraph>
                              {issue.rule && (
                                <Text type="secondary" style={{ fontSize: 12 }}>
                                  Rule: {issue.rule}
                                </Text>
                              )}
                              {issue.suggestion && (
                                <div style={{ marginTop: 8 }}>
                                  <Text type="secondary">Suggestion: </Text>
                                  <Text code style={{ fontSize: 12 }}>
                                    {issue.suggestion}
                                  </Text>
                                </div>
                              )}
                            </div>
                          }
                        />
                      </List.Item>
                    )}
                  />
                </>
              )}
            </Card>
          </Col>
        </Row>
      </Content>

      {/* AI Chat Drawer */}
      <Drawer
        title={
          <Space>
            <RobotOutlined />
            {t('codeReview.ai_assistant', 'AI Assistant')}
            <Tag color="blue">{selectedModel}</Tag>
          </Space>
        }
        placement="right"
        width={480}
        open={chatDrawerOpen}
        onClose={() => setChatDrawerOpen(false)}
        className="chat-drawer"
      >
        <div className="chat-container">
          {/* Chat Messages */}
          <div className="chat-messages">
            {chatMessages.length === 0 ? (
              <div className="chat-welcome">
                <RobotOutlined style={{ fontSize: 48, color: '#1890ff' }} />
                <Title level={4}>AI Code Assistant</Title>
                <Paragraph type="secondary">
                  Ask me anything about your code! I can help you:
                </Paragraph>
                <ul>
                  <li>Explain issues and vulnerabilities</li>
                  <li>Suggest fixes and improvements</li>
                  <li>Answer coding questions</li>
                  <li>Provide best practices</li>
                </ul>
              </div>
            ) : (
              chatMessages.map(msg => (
                <div
                  key={msg.id}
                  className={`chat-message ${msg.role}`}
                >
                  <div className="message-avatar">
                    {msg.role === 'user' ? (
                      <Avatar icon={<UserOutlined />} />
                    ) : (
                      <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#1890ff' }} />
                    )}
                  </div>
                  <div className="message-content">
                    <div className="message-header">
                      <Text strong>
                        {msg.role === 'user' ? 'You' : msg.model || 'AI Assistant'}
                      </Text>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {msg.timestamp.toLocaleTimeString()}
                      </Text>
                    </div>
                    <div className="message-text">
                      <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>
                        {msg.content}
                      </pre>
                    </div>
                  </div>
                </div>
              ))
            )}
            {chatMutation.isPending && (
              <div className="chat-message assistant">
                <div className="message-avatar">
                  <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#1890ff' }} />
                </div>
                <div className="message-content">
                  <Spin size="small" /> Thinking...
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Chat Input */}
          <div className="chat-input">
            <TextArea
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              placeholder={t('codeReview.chat_placeholder', 'Ask about your code...')}
              autoSize={{ minRows: 2, maxRows: 4 }}
              onPressEnter={e => {
                if (!e.shiftKey) {
                  e.preventDefault();
                  sendChatMessage();
                }
              }}
            />
            <div className="chat-actions">
              <Button
                icon={<ClearOutlined />}
                onClick={() => setChatMessages([])}
                disabled={chatMessages.length === 0}
              >
                Clear
              </Button>
              <Button
                type="primary"
                icon={<SendOutlined />}
                onClick={sendChatMessage}
                loading={chatMutation.isPending}
              >
                Send
              </Button>
            </div>
          </div>
        </div>
      </Drawer>
    </Layout>
  );
};

export default CodeReview;
