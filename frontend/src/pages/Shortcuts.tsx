/**
 * Keyboard Shortcuts Page
 * 键盘快捷键页面
 * 
 * Features:
 * - All keyboard shortcuts
 * - Command palette instructions
 * - Customization options
 */

import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Tag,
  Input,
  Collapse,
  Divider,
  Alert,
} from 'antd';
import {
  ThunderboltOutlined,
  SearchOutlined,
  CodeOutlined,
  FileOutlined,
  SettingOutlined,
  BranchesOutlined,
  CommentOutlined,
  SaveOutlined,
  UndoOutlined,
  RedoOutlined,
  CopyOutlined,
  ScissorOutlined,
  EnterOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  HomeOutlined,
  GlobalOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text, Paragraph } = Typography;
const { Panel } = Collapse;

interface Shortcut {
  keys: string[];
  description: string;
  icon?: React.ReactNode;
}

interface ShortcutCategory {
  title: string;
  icon: React.ReactNode;
  shortcuts: Shortcut[];
}

const shortcutCategories: ShortcutCategory[] = [
  {
    title: 'General',
    icon: <GlobalOutlined />,
    shortcuts: [
      { keys: ['Ctrl/⌘', 'K'], description: 'Open command palette', icon: <SearchOutlined /> },
      { keys: ['Ctrl/⌘', 'P'], description: 'Quick file search' },
      { keys: ['Ctrl/⌘', 'Shift', 'P'], description: 'Open settings' },
      { keys: ['Ctrl/⌘', '/'], description: 'Show keyboard shortcuts' },
      { keys: ['Esc'], description: 'Close modal/panel' },
    ],
  },
  {
    title: 'Navigation',
    icon: <HomeOutlined />,
    shortcuts: [
      { keys: ['G', 'D'], description: 'Go to Dashboard' },
      { keys: ['G', 'P'], description: 'Go to Projects' },
      { keys: ['G', 'R'], description: 'Go to Code Review' },
      { keys: ['G', 'S'], description: 'Go to Security' },
      { keys: ['G', 'T'], description: 'Go to Teams' },
      { keys: ['G', 'N'], description: 'Go to Notifications' },
    ],
  },
  {
    title: 'Code Review',
    icon: <CodeOutlined />,
    shortcuts: [
      { keys: ['Ctrl/⌘', 'Enter'], description: 'Submit review comment' },
      { keys: ['Ctrl/⌘', 'Shift', 'Enter'], description: 'Approve and submit' },
      { keys: ['N'], description: 'Next issue' },
      { keys: ['P'], description: 'Previous issue' },
      { keys: ['J'], description: 'Next file' },
      { keys: ['K'], description: 'Previous file' },
      { keys: ['C'], description: 'Add comment' },
      { keys: ['A'], description: 'Toggle AI suggestions' },
    ],
  },
  {
    title: 'Editor',
    icon: <FileOutlined />,
    shortcuts: [
      { keys: ['Ctrl/⌘', 'S'], description: 'Save', icon: <SaveOutlined /> },
      { keys: ['Ctrl/⌘', 'Z'], description: 'Undo', icon: <UndoOutlined /> },
      { keys: ['Ctrl/⌘', 'Shift', 'Z'], description: 'Redo', icon: <RedoOutlined /> },
      { keys: ['Ctrl/⌘', 'C'], description: 'Copy', icon: <CopyOutlined /> },
      { keys: ['Ctrl/⌘', 'X'], description: 'Cut', icon: <ScissorOutlined /> },
      { keys: ['Ctrl/⌘', 'V'], description: 'Paste' },
      { keys: ['Ctrl/⌘', 'F'], description: 'Find in file' },
      { keys: ['Ctrl/⌘', 'H'], description: 'Find and replace' },
      { keys: ['Ctrl/⌘', 'G'], description: 'Go to line' },
    ],
  },
  {
    title: 'Pull Requests',
    icon: <BranchesOutlined />,
    shortcuts: [
      { keys: ['Ctrl/⌘', 'M'], description: 'Merge pull request' },
      { keys: ['R'], description: 'Request changes' },
      { keys: ['A'], description: 'Approve' },
      { keys: ['D'], description: 'View diff' },
      { keys: ['F'], description: 'View files changed' },
    ],
  },
  {
    title: 'AI Features',
    icon: <ThunderboltOutlined />,
    shortcuts: [
      { keys: ['Ctrl/⌘', 'Shift', 'A'], description: 'Run AI analysis' },
      { keys: ['Ctrl/⌘', 'Shift', 'F'], description: 'Apply AI auto-fix' },
      { keys: ['Ctrl/⌘', 'Shift', 'E'], description: 'Explain selected code' },
      { keys: ['Ctrl/⌘', 'Shift', 'R'], description: 'Refactor with AI' },
    ],
  },
];

const KeyboardKey: React.FC<{ children: string }> = ({ children }) => (
  <Tag
    style={{
      background: 'linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%)',
      border: '1px solid #cbd5e1',
      borderRadius: 6,
      padding: '4px 10px',
      fontSize: 13,
      fontFamily: 'monospace',
      fontWeight: 600,
      color: '#1e293b',
      boxShadow: '0 2px 0 #cbd5e1',
      minWidth: 32,
      textAlign: 'center',
    }}
  >
    {children}
  </Tag>
);

export const Shortcuts: React.FC = () => {
  const { t } = useTranslation();
  const [searchQuery, setSearchQuery] = useState('');

  const filteredCategories = shortcutCategories.map(category => ({
    ...category,
    shortcuts: category.shortcuts.filter(s =>
      s.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.keys.join(' ').toLowerCase().includes(searchQuery.toLowerCase())
    ),
  })).filter(c => c.shortcuts.length > 0);

  return (
    <div className="shortcuts-page" style={{ maxWidth: 1000, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <ThunderboltOutlined style={{ color: '#2563eb' }} /> Keyboard Shortcuts
          </Title>
          <Text type="secondary">Master the platform with keyboard commands</Text>
        </div>
      </div>

      {/* Command Palette Highlight */}
      <Alert
        type="info"
        showIcon
        icon={<SearchOutlined />}
        message={
          <Space>
            <Text strong>Pro Tip: Use the Command Palette</Text>
            <Space size={4}>
              <KeyboardKey>Ctrl/⌘</KeyboardKey>
              <span>+</span>
              <KeyboardKey>K</KeyboardKey>
            </Space>
          </Space>
        }
        description="Access any command quickly without leaving the keyboard. Just type what you want to do!"
        style={{ marginBottom: 24, borderRadius: 12 }}
      />

      {/* Search */}
      <Card style={{ marginBottom: 24, borderRadius: 12 }}>
        <Input.Search
          placeholder="Search shortcuts..."
          size="large"
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          allowClear
          prefix={<SearchOutlined style={{ color: '#94a3b8' }} />}
        />
      </Card>

      {/* Shortcuts Grid */}
      <Row gutter={[16, 16]}>
        {filteredCategories.map((category, index) => (
          <Col key={index} xs={24} md={12}>
            <Card
              title={
                <Space>
                  <span style={{ color: '#2563eb' }}>{category.icon}</span>
                  {category.title}
                </Space>
              }
              style={{ borderRadius: 12, height: '100%' }}
            >
              <Space direction="vertical" style={{ width: '100%' }} size={12}>
                {category.shortcuts.map((shortcut, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '8px 0',
                      borderBottom: idx < category.shortcuts.length - 1 ? '1px solid #f1f5f9' : 'none',
                    }}
                  >
                    <Space>
                      {shortcut.icon && <span style={{ color: '#64748b' }}>{shortcut.icon}</span>}
                      <Text>{shortcut.description}</Text>
                    </Space>
                    <Space size={4}>
                      {shortcut.keys.map((key, keyIdx) => (
                        <React.Fragment key={keyIdx}>
                          <KeyboardKey>{key}</KeyboardKey>
                          {keyIdx < shortcut.keys.length - 1 && (
                            <span style={{ color: '#94a3b8' }}>+</span>
                          )}
                        </React.Fragment>
                      ))}
                    </Space>
                  </div>
                ))}
              </Space>
            </Card>
          </Col>
        ))}
      </Row>

      {/* Tips */}
      <Card title="Tips & Tricks" style={{ marginTop: 24, borderRadius: 12 }}>
        <Row gutter={24}>
          <Col xs={24} md={8}>
            <div style={{ padding: 16, background: '#f8fafc', borderRadius: 12 }}>
              <Text strong style={{ display: 'block', marginBottom: 8 }}>
                💡 Quick Navigation
              </Text>
              <Text type="secondary">
                Use <KeyboardKey>G</KeyboardKey> followed by a letter to quickly navigate to any page.
              </Text>
            </div>
          </Col>
          <Col xs={24} md={8}>
            <div style={{ padding: 16, background: '#f8fafc', borderRadius: 12 }}>
              <Text strong style={{ display: 'block', marginBottom: 8 }}>
                🚀 AI Shortcuts
              </Text>
              <Text type="secondary">
                Hold <KeyboardKey>Ctrl/⌘</KeyboardKey> + <KeyboardKey>Shift</KeyboardKey> for AI actions.
              </Text>
            </div>
          </Col>
          <Col xs={24} md={8}>
            <div style={{ padding: 16, background: '#f8fafc', borderRadius: 12 }}>
              <Text strong style={{ display: 'block', marginBottom: 8 }}>
                ⌨️ Vim Mode
              </Text>
              <Text type="secondary">
                Enable Vim keybindings in Settings for advanced navigation.
              </Text>
            </div>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default Shortcuts;
