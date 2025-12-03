/**
 * AI Assistant Page
 * AI助手页面
 * 
 * Features:
 * - Chat interface with AI
 * - Code explanation
 * - Refactoring suggestions
 * - Security analysis
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Input,
  Button,
  Avatar,
  List,
  Tag,
  Tooltip,
  Divider,
  Select,
  message,
  Spin,
} from 'antd';
import {
  RobotOutlined,
  UserOutlined,
  SendOutlined,
  CodeOutlined,
  BulbOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
  CopyOutlined,
  ReloadOutlined,
  HistoryOutlined,
  DeleteOutlined,
  PlusOutlined,
  StarOutlined,
  BookOutlined,
  ToolOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  codeBlocks?: { language: string; code: string }[];
}

interface Conversation {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
}

const quickPrompts = [
  { icon: <CodeOutlined />, label: 'Explain this code', prompt: 'Explain what this code does:' },
  { icon: <BulbOutlined />, label: 'Suggest improvements', prompt: 'How can I improve this code?' },
  { icon: <SafetyCertificateOutlined />, label: 'Security check', prompt: 'Check this code for security vulnerabilities:' },
  { icon: <ThunderboltOutlined />, label: 'Optimize performance', prompt: 'Optimize this code for better performance:' },
  { icon: <ToolOutlined />, label: 'Refactor', prompt: 'Refactor this code to be more maintainable:' },
  { icon: <BookOutlined />, label: 'Add documentation', prompt: 'Add documentation to this code:' },
];

const mockConversations: Conversation[] = [
  { id: '1', title: 'SQL Injection Fix', lastMessage: 'The parameterized query looks good...', timestamp: new Date(Date.now() - 30 * 60 * 1000) },
  { id: '2', title: 'React Hook Optimization', lastMessage: 'Consider using useMemo for...', timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000) },
  { id: '3', title: 'API Design Review', lastMessage: 'The REST endpoints follow...', timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000) },
];

export const AIAssistant: React.FC = () => {
  const { t } = useTranslation();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hello! I'm your AI coding assistant. I can help you with:\n\n• **Code explanation** - Understand complex code\n• **Security analysis** - Find vulnerabilities\n• **Refactoring** - Improve code quality\n• **Performance optimization** - Make code faster\n\nPaste your code or ask me anything about your project!",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gpt-4-turbo');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: generateMockResponse(inputValue),
        timestamp: new Date(),
        codeBlocks: inputValue.toLowerCase().includes('fix') ? [
          {
            language: 'python',
            code: `# Secure parameterized query
def get_user(user_id: str) -> User:
    query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()`,
          },
        ] : undefined,
      };
      setMessages(prev => [...prev, aiResponse]);
      setIsLoading(false);
    }, 1500);
  };

  const generateMockResponse = (input: string): string => {
    if (input.toLowerCase().includes('security') || input.toLowerCase().includes('vulnerability')) {
      return "I've analyzed the code for security vulnerabilities. Here's what I found:\n\n**🔴 Critical: SQL Injection**\nThe query uses string interpolation which is vulnerable to SQL injection attacks.\n\n**Recommendation:**\nUse parameterized queries instead of string formatting. See the code example below:";
    }
    if (input.toLowerCase().includes('explain')) {
      return "Here's an explanation of the code:\n\n**Purpose:**\nThis function handles user authentication by validating credentials against the database.\n\n**Flow:**\n1. Receives username and password\n2. Queries the database for matching user\n3. Verifies password hash\n4. Returns JWT token on success\n\n**Key points:**\n• Uses bcrypt for password hashing\n• Implements rate limiting for security\n• Returns standardized error responses";
    }
    if (input.toLowerCase().includes('improve') || input.toLowerCase().includes('refactor')) {
      return "Here are my suggestions for improving this code:\n\n**1. Add Type Hints**\nAdding type hints improves code readability and enables better IDE support.\n\n**2. Extract Magic Numbers**\nReplace hardcoded values with named constants.\n\n**3. Add Error Handling**\nWrap database operations in try-catch blocks.\n\n**4. Use Dependency Injection**\nThis makes the code more testable.";
    }
    return "I understand your question. Based on my analysis of your code and best practices, here are my thoughts:\n\n• The code structure follows good patterns\n• Consider adding more unit tests\n• Documentation could be improved\n\nWould you like me to elaborate on any of these points?";
  };

  const handleQuickPrompt = (prompt: string) => {
    setInputValue(prompt + '\n\n```\n// Paste your code here\n```');
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    message.success('Copied to clipboard');
  };

  return (
    <div className="ai-assistant-page" style={{ height: 'calc(100vh - 120px)', display: 'flex', gap: 16 }}>
      {/* Sidebar - Conversation History */}
      <Card
        style={{ width: 280, borderRadius: 12, flexShrink: 0 }}
        bodyStyle={{ padding: 0 }}
      >
        <div style={{ padding: 16, borderBottom: '1px solid #f1f5f9' }}>
          <Button type="primary" block icon={<PlusOutlined />}>
            New Chat
          </Button>
        </div>
        <div style={{ padding: '8px 16px' }}>
          <Text type="secondary" style={{ fontSize: 12 }}>RECENT CONVERSATIONS</Text>
        </div>
        <List
          dataSource={mockConversations}
          renderItem={conv => (
            <List.Item
              style={{ padding: '12px 16px', cursor: 'pointer', borderBottom: '1px solid #f8fafc' }}
              className="conversation-item"
            >
              <div style={{ width: '100%' }}>
                <Text strong ellipsis style={{ display: 'block' }}>{conv.title}</Text>
                <Text type="secondary" style={{ fontSize: 12 }} ellipsis>
                  {conv.lastMessage}
                </Text>
              </div>
            </List.Item>
          )}
        />
      </Card>

      {/* Main Chat Area */}
      <Card
        style={{ flex: 1, borderRadius: 12, display: 'flex', flexDirection: 'column' }}
        bodyStyle={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 0 }}
      >
        {/* Header */}
        <div style={{ padding: '12px 24px', borderBottom: '1px solid #f1f5f9', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <Avatar style={{ background: '#2563eb' }} icon={<RobotOutlined />} />
            <div>
              <Text strong>AI Code Assistant</Text>
              <div>
                <Tag color="green" style={{ fontSize: 11 }}>Online</Tag>
              </div>
            </div>
          </Space>
          <Space>
            <Select
              value={selectedModel}
              onChange={setSelectedModel}
              style={{ width: 150 }}
              size="small"
              options={[
                { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
                { value: 'gpt-4', label: 'GPT-4' },
                { value: 'claude-3', label: 'Claude 3' },
              ]}
            />
            <Tooltip title="Clear conversation">
              <Button size="small" icon={<DeleteOutlined />} />
            </Tooltip>
          </Space>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
          {messages.map(msg => (
            <div
              key={msg.id}
              style={{
                display: 'flex',
                gap: 12,
                marginBottom: 24,
                flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
              }}
            >
              <Avatar
                style={{
                  background: msg.role === 'user' ? '#64748b' : '#2563eb',
                  flexShrink: 0,
                }}
                icon={msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
              />
              <div
                style={{
                  maxWidth: '70%',
                  padding: 16,
                  borderRadius: 12,
                  background: msg.role === 'user' ? '#f1f5f9' : '#eff6ff',
                }}
              >
                <Paragraph style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                  {msg.content}
                </Paragraph>
                {msg.codeBlocks?.map((block, idx) => (
                  <div key={idx} style={{ marginTop: 12 }}>
                    <div style={{
                      background: '#1e293b',
                      borderRadius: '8px 8px 0 0',
                      padding: '8px 12px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}>
                      <Text style={{ color: '#94a3b8', fontSize: 12 }}>{block.language}</Text>
                      <Button
                        type="text"
                        size="small"
                        icon={<CopyOutlined />}
                        style={{ color: '#94a3b8' }}
                        onClick={() => copyToClipboard(block.code)}
                      />
                    </div>
                    <pre style={{
                      background: '#0f172a',
                      padding: 16,
                      borderRadius: '0 0 8px 8px',
                      margin: 0,
                      overflow: 'auto',
                      color: '#e2e8f0',
                      fontSize: 13,
                    }}>
                      {block.code}
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          ))}
          {isLoading && (
            <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
              <Avatar style={{ background: '#2563eb' }} icon={<RobotOutlined />} />
              <div style={{ padding: 16 }}>
                <Spin size="small" /> <Text type="secondary">Thinking...</Text>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Prompts */}
        <div style={{ padding: '12px 24px', borderTop: '1px solid #f1f5f9' }}>
          <Space wrap>
            {quickPrompts.map((prompt, idx) => (
              <Button
                key={idx}
                size="small"
                icon={prompt.icon}
                onClick={() => handleQuickPrompt(prompt.prompt)}
              >
                {prompt.label}
              </Button>
            ))}
          </Space>
        </div>

        {/* Input Area */}
        <div style={{ padding: '16px 24px', borderTop: '1px solid #f1f5f9' }}>
          <div style={{ display: 'flex', gap: 12 }}>
            <TextArea
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              placeholder="Ask me anything about your code... (Shift+Enter for new line)"
              autoSize={{ minRows: 1, maxRows: 6 }}
              onPressEnter={e => {
                if (!e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              style={{ flex: 1 }}
            />
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={handleSend}
              loading={isLoading}
              style={{ alignSelf: 'flex-end' }}
            >
              Send
            </Button>
          </div>
        </div>
      </Card>

      <style>{`
        .conversation-item:hover {
          background: #f8fafc !important;
        }
      `}</style>
    </div>
  );
};

export default AIAssistant;
