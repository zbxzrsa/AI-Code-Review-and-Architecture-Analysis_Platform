/**
 * Code Comparison Page
 * 代码比较页面
 * 
 * Diff viewer with:
 * - Side-by-side comparison
 * - Inline diff view
 * - Syntax highlighting
 * - AI-generated change summary
 */

import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Select,
  Button,
  Segmented,
  Tag,
  Statistic,
  Alert,
  Divider,
} from 'antd';
import {
  DiffOutlined,
  SwapOutlined,
  CodeOutlined,
  PlusOutlined,
  MinusOutlined,
  RobotOutlined,
  BranchesOutlined,
  FileTextOutlined,
  CopyOutlined,
  ExpandOutlined,
  CompressOutlined,
  CheckCircleOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text } = Typography;

interface DiffLine {
  type: 'add' | 'remove' | 'context' | 'header';
  content: string;
  oldLineNumber?: number;
  newLineNumber?: number;
}

const mockDiff: DiffLine[] = [
  { type: 'header', content: '@@ -42,15 +42,20 @@ class UserAuthentication:' },
  { type: 'context', content: '    def authenticate(self, request):', oldLineNumber: 42, newLineNumber: 42 },
  { type: 'context', content: '        """Authenticate user with credentials."""', oldLineNumber: 43, newLineNumber: 43 },
  { type: 'context', content: '        username = request.data.get("username")', oldLineNumber: 44, newLineNumber: 44 },
  { type: 'context', content: '        password = request.data.get("password")', oldLineNumber: 45, newLineNumber: 45 },
  { type: 'context', content: '', oldLineNumber: 46, newLineNumber: 46 },
  { type: 'remove', content: '        # Vulnerable: SQL injection possible', oldLineNumber: 47 },
  { type: 'remove', content: '        query = f"SELECT * FROM users WHERE username = \'{username}\'"', oldLineNumber: 48 },
  { type: 'remove', content: '        user = db.execute(query).fetchone()', oldLineNumber: 49 },
  { type: 'add', content: '        # Secure: Using parameterized queries', newLineNumber: 47 },
  { type: 'add', content: '        query = "SELECT * FROM users WHERE username = %s"', newLineNumber: 48 },
  { type: 'add', content: '        user = db.execute(query, (username,)).fetchone()', newLineNumber: 49 },
  { type: 'add', content: '', newLineNumber: 50 },
  { type: 'add', content: '        # Input validation', newLineNumber: 51 },
  { type: 'add', content: '        if not self.validate_input(username):', newLineNumber: 52 },
  { type: 'add', content: '            raise ValidationError("Invalid username format")', newLineNumber: 53 },
  { type: 'context', content: '', oldLineNumber: 50, newLineNumber: 54 },
  { type: 'context', content: '        if user and self.verify_password(password, user.password_hash):', oldLineNumber: 51, newLineNumber: 55 },
  { type: 'context', content: '            return self.generate_token(user)', oldLineNumber: 52, newLineNumber: 56 },
  { type: 'context', content: '        return None', oldLineNumber: 53, newLineNumber: 57 },
];

const aiSummary = {
  title: 'Security Fix: SQL Injection Prevention',
  description: 'This change addresses a critical SQL injection vulnerability by replacing string interpolation with parameterized queries and adding input validation.',
  changes: [
    { type: 'security', text: 'Fixed SQL injection vulnerability in user authentication' },
    { type: 'improvement', text: 'Added input validation for username field' },
    { type: 'improvement', text: 'Improved code documentation with security comments' },
  ],
  riskLevel: 'low',
  confidence: 0.96,
};

export const CodeComparison: React.FC = () => {
  const { t } = useTranslation();
  const [viewMode, setViewMode] = useState<'split' | 'unified'>('split');
  const [expandedContext, setExpandedContext] = useState(false);

  const additions = mockDiff.filter(l => l.type === 'add').length;
  const deletions = mockDiff.filter(l => l.type === 'remove').length;

  const renderDiffLine = (line: DiffLine, index: number) => {
    const bgColor = line.type === 'add' ? 'rgba(34, 197, 94, 0.1)' :
                    line.type === 'remove' ? 'rgba(239, 68, 68, 0.1)' :
                    line.type === 'header' ? 'rgba(59, 130, 246, 0.1)' : 'transparent';
    
    const borderColor = line.type === 'add' ? '#22c55e' :
                        line.type === 'remove' ? '#ef4444' : 'transparent';

    const textColor = line.type === 'header' ? '#3b82f6' : 'inherit';

    return (
      <div
        key={index}
        style={{
          display: 'flex',
          background: bgColor,
          borderLeft: `3px solid ${borderColor}`,
          fontFamily: '"Fira Code", "Monaco", monospace',
          fontSize: 13,
          lineHeight: '22px',
        }}
      >
        {/* Line Numbers */}
        {line.type !== 'header' && (
          <>
            <div style={{
              width: 50,
              textAlign: 'right',
              paddingRight: 8,
              color: '#94a3b8',
              background: 'rgba(0,0,0,0.02)',
              userSelect: 'none',
            }}>
              {line.oldLineNumber || ''}
            </div>
            <div style={{
              width: 50,
              textAlign: 'right',
              paddingRight: 8,
              color: '#94a3b8',
              background: 'rgba(0,0,0,0.02)',
              userSelect: 'none',
              borderRight: '1px solid #e2e8f0',
            }}>
              {line.newLineNumber || ''}
            </div>
          </>
        )}

        {/* Change Indicator */}
        <div style={{
          width: 24,
          textAlign: 'center',
          color: line.type === 'add' ? '#22c55e' : line.type === 'remove' ? '#ef4444' : '#94a3b8',
          fontWeight: 600,
        }}>
          {line.type === 'add' && '+'}
          {line.type === 'remove' && '-'}
          {line.type === 'header' && '@@'}
        </div>

        {/* Code Content */}
        <div style={{
          flex: 1,
          paddingLeft: 8,
          whiteSpace: 'pre',
          overflow: 'auto',
          color: textColor,
        }}>
          {line.content}
        </div>
      </div>
    );
  };

  return (
    <div className="code-comparison-page" style={{ maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <DiffOutlined style={{ color: '#2563eb' }} /> Code Comparison
          </Title>
          <Text type="secondary">Review code changes with AI-powered analysis</Text>
        </div>
        <Space>
          <Segmented
            value={viewMode}
            onChange={(v) => setViewMode(v as 'split' | 'unified')}
            options={[
              { label: 'Split', value: 'split', icon: <SwapOutlined /> },
              { label: 'Unified', value: 'unified', icon: <CodeOutlined /> },
            ]}
          />
          <Button icon={expandedContext ? <CompressOutlined /> : <ExpandOutlined />} onClick={() => setExpandedContext(!expandedContext)}>
            {expandedContext ? 'Collapse' : 'Expand'} Context
          </Button>
          <Button icon={<CopyOutlined />}>Copy Diff</Button>
        </Space>
      </div>

      {/* File Selection */}
      <Card style={{ marginBottom: 16, borderRadius: 12 }}>
        <Row gutter={24} align="middle">
          <Col flex="1">
            <Space>
              <Select
                value="main"
                style={{ width: 200 }}
                options={[
                  { value: 'main', label: <><BranchesOutlined /> main</> },
                  { value: 'develop', label: <><BranchesOutlined /> develop</> },
                  { value: 'feature/auth', label: <><BranchesOutlined /> feature/auth</> },
                ]}
              />
              <SwapOutlined style={{ color: '#64748b' }} />
              <Select
                value="feature/auth"
                style={{ width: 200 }}
                options={[
                  { value: 'main', label: <><BranchesOutlined /> main</> },
                  { value: 'develop', label: <><BranchesOutlined /> develop</> },
                  { value: 'feature/auth', label: <><BranchesOutlined /> feature/auth</> },
                ]}
              />
            </Space>
          </Col>
          <Col>
            <Space size={24}>
              <Statistic
                title="Additions"
                value={additions}
                valueStyle={{ color: '#22c55e', fontSize: 20 }}
                prefix={<PlusOutlined />}
              />
              <Statistic
                title="Deletions"
                value={deletions}
                valueStyle={{ color: '#ef4444', fontSize: 20 }}
                prefix={<MinusOutlined />}
              />
            </Space>
          </Col>
        </Row>
      </Card>

      <Row gutter={16}>
        {/* Diff Viewer */}
        <Col xs={24} lg={16}>
          <Card
            title={
              <Space>
                <FileTextOutlined />
                <Text>src/auth/authentication.py</Text>
                <Tag color="blue">{additions + deletions} changes</Tag>
              </Space>
            }
            style={{ borderRadius: 12 }}
            bodyStyle={{ padding: 0 }}
          >
            <div style={{
              maxHeight: 600,
              overflow: 'auto',
              borderRadius: '0 0 12px 12px',
            }}>
              {mockDiff.map((line, index) => renderDiffLine(line, index))}
            </div>
          </Card>
        </Col>

        {/* AI Summary Sidebar */}
        <Col xs={24} lg={8}>
          <Card
            title={<><RobotOutlined /> AI Change Analysis</>}
            style={{ borderRadius: 12, marginBottom: 16 }}
          >
            <Alert
              message={aiSummary.title}
              description={aiSummary.description}
              type="info"
              showIcon
              icon={<RobotOutlined />}
              style={{ marginBottom: 16 }}
            />

            <Divider>Changes Detected</Divider>

            <Space direction="vertical" style={{ width: '100%' }}>
              {aiSummary.changes.map((change, index) => (
                <div
                  key={index}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 8,
                    padding: '8px 12px',
                    background: change.type === 'security' ? '#fef2f2' : '#f0fdf4',
                    borderRadius: 8,
                  }}
                >
                  {change.type === 'security' ? (
                    <WarningOutlined style={{ color: '#ef4444', marginTop: 2 }} />
                  ) : (
                    <CheckCircleOutlined style={{ color: '#22c55e', marginTop: 2 }} />
                  )}
                  <Text style={{ flex: 1 }}>{change.text}</Text>
                </div>
              ))}
            </Space>

            <Divider>Risk Assessment</Divider>

            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="Risk Level"
                  value={aiSummary.riskLevel.toUpperCase()}
                  valueStyle={{ color: '#22c55e' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="AI Confidence"
                  value={Math.round(aiSummary.confidence * 100)}
                  suffix="%"
                  valueStyle={{ color: '#2563eb' }}
                />
              </Col>
            </Row>
          </Card>

          <Card title="Quick Actions" style={{ borderRadius: 12 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button block type="primary" icon={<CheckCircleOutlined />}>
                Approve Changes
              </Button>
              <Button block icon={<RobotOutlined />}>
                Request AI Review
              </Button>
              <Button block icon={<CodeOutlined />}>
                View Full File
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default CodeComparison;
