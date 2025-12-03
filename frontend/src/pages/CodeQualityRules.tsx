/**
 * Code Quality Rules Page
 * 代码质量规则页面
 * 
 * Linting and code quality configuration with:
 * - Custom rule management
 * - Severity levels
 * - Rule templates
 * - Project-specific overrides
 */

import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Tag,
  Space,
  Typography,
  Switch,
  Modal,
  Form,
  Input,
  Select,
  Tooltip,
  Collapse,
  Statistic,
  message,
} from 'antd';
import type { TableProps } from 'antd';
import {
  SettingOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  StopOutlined,
  InfoCircleOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  ImportOutlined,
  ExportOutlined,
  SafetyCertificateOutlined,
  CodeOutlined,
  ThunderboltOutlined,
  FileTextOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text } = Typography;

interface Rule {
  id: string;
  name: string;
  description: string;
  category: 'security' | 'quality' | 'performance' | 'style' | 'best-practice';
  severity: 'error' | 'warning' | 'info' | 'off';
  enabled: boolean;
  autoFix: boolean;
  source: 'builtin' | 'custom' | 'imported';
  tags: string[];
  options?: Record<string, any>;
}

const mockRules: Rule[] = [
  {
    id: 'rule_1',
    name: 'no-sql-injection',
    description: 'Prevent SQL injection by requiring parameterized queries',
    category: 'security',
    severity: 'error',
    enabled: true,
    autoFix: true,
    source: 'builtin',
    tags: ['security', 'critical', 'owasp'],
  },
  {
    id: 'rule_2',
    name: 'no-hardcoded-secrets',
    description: 'Detect hardcoded API keys, passwords, and secrets',
    category: 'security',
    severity: 'error',
    enabled: true,
    autoFix: false,
    source: 'builtin',
    tags: ['security', 'secrets'],
  },
  {
    id: 'rule_3',
    name: 'no-eval',
    description: 'Disallow use of eval() and similar functions',
    category: 'security',
    severity: 'error',
    enabled: true,
    autoFix: false,
    source: 'builtin',
    tags: ['security', 'xss'],
  },
  {
    id: 'rule_4',
    name: 'max-complexity',
    description: 'Enforce maximum cyclomatic complexity per function',
    category: 'quality',
    severity: 'warning',
    enabled: true,
    autoFix: false,
    source: 'builtin',
    tags: ['complexity', 'maintainability'],
    options: { max: 10 },
  },
  {
    id: 'rule_5',
    name: 'no-unused-variables',
    description: 'Disallow unused variables and imports',
    category: 'quality',
    severity: 'warning',
    enabled: true,
    autoFix: true,
    source: 'builtin',
    tags: ['cleanup', 'dead-code'],
  },
  {
    id: 'rule_6',
    name: 'prefer-const',
    description: 'Prefer const over let for variables never reassigned',
    category: 'style',
    severity: 'info',
    enabled: true,
    autoFix: true,
    source: 'builtin',
    tags: ['es6', 'style'],
  },
  {
    id: 'rule_7',
    name: 'no-n-plus-one',
    description: 'Detect N+1 query patterns in database access',
    category: 'performance',
    severity: 'warning',
    enabled: true,
    autoFix: false,
    source: 'builtin',
    tags: ['database', 'performance'],
  },
  {
    id: 'rule_8',
    name: 'require-error-handling',
    description: 'Require try-catch or error handling for async operations',
    category: 'best-practice',
    severity: 'warning',
    enabled: true,
    autoFix: false,
    source: 'builtin',
    tags: ['error-handling', 'async'],
  },
  {
    id: 'rule_9',
    name: 'custom-api-naming',
    description: 'Enforce custom API naming conventions',
    category: 'style',
    severity: 'info',
    enabled: false,
    autoFix: false,
    source: 'custom',
    tags: ['naming', 'api'],
  },
];

const categoryConfig = {
  security: { color: '#ef4444', icon: <SafetyCertificateOutlined />, label: 'Security' },
  quality: { color: '#3b82f6', icon: <CodeOutlined />, label: 'Quality' },
  performance: { color: '#8b5cf6', icon: <ThunderboltOutlined />, label: 'Performance' },
  style: { color: '#06b6d4', icon: <FileTextOutlined />, label: 'Style' },
  'best-practice': { color: '#22c55e', icon: <CheckCircleOutlined />, label: 'Best Practice' },
};

const severityConfig = {
  error: { color: 'red', icon: <StopOutlined />, label: 'Error' },
  warning: { color: 'orange', icon: <WarningOutlined />, label: 'Warning' },
  info: { color: 'blue', icon: <InfoCircleOutlined />, label: 'Info' },
  off: { color: 'default', icon: <StopOutlined />, label: 'Off' },
};

export const CodeQualityRules: React.FC = () => {
  const { t: _t } = useTranslation();
  const [rules, setRules] = useState<Rule[]>(mockRules);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [_editingRule, setEditingRule] = useState<Rule | null>(null);
  const [form] = Form.useForm();

  const filteredRules = rules.filter(rule => {
    if (selectedCategory !== 'all' && rule.category !== selectedCategory) return false;
    if (searchQuery && !rule.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !rule.description.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const stats = {
    total: rules.length,
    enabled: rules.filter(r => r.enabled).length,
    errors: rules.filter(r => r.severity === 'error' && r.enabled).length,
    warnings: rules.filter(r => r.severity === 'warning' && r.enabled).length,
  };

  const handleToggleRule = (ruleId: string, enabled: boolean) => {
    setRules(prev => prev.map(r =>
      r.id === ruleId ? { ...r, enabled } : r
    ));
    message.success(enabled ? 'Rule enabled' : 'Rule disabled');
  };

  const handleSeverityChange = (ruleId: string, severity: string) => {
    setRules(prev => prev.map(r =>
      r.id === ruleId ? { ...r, severity: severity as Rule['severity'] } : r
    ));
  };

  const handleCreateRule = (values: any) => {
    const newRule: Rule = {
      id: `rule_${Date.now()}`,
      ...values,
      enabled: true,
      source: 'custom',
      tags: values.tags || [],
    };
    setRules(prev => [...prev, newRule]);
    setCreateModalOpen(false);
    form.resetFields();
    message.success('Rule created successfully');
  };

  const columns: TableProps<Rule>['columns'] = [
    {
      title: 'Rule',
      key: 'rule',
      render: (_, record) => {
        const catConfig = categoryConfig[record.category];
        return (
          <div>
            <Space>
              <span style={{ color: catConfig.color }}>{catConfig.icon}</span>
              <Text strong style={{ fontFamily: 'monospace' }}>{record.name}</Text>
              {record.source === 'custom' && <Tag color="purple">Custom</Tag>}
              {record.autoFix && (
                <Tooltip title="Auto-fix available">
                  <Tag color="green" icon={<ThunderboltOutlined />}>Auto-fix</Tag>
                </Tooltip>
              )}
            </Space>
            <div>
              <Text type="secondary" style={{ fontSize: 13 }}>{record.description}</Text>
            </div>
            <div style={{ marginTop: 4 }}>
              {record.tags.map(tag => (
                <Tag key={tag} style={{ fontSize: 11 }}>{tag}</Tag>
              ))}
            </div>
          </div>
        );
      },
    },
    {
      title: 'Category',
      dataIndex: 'category',
      width: 130,
      render: (category) => {
        const config = categoryConfig[category as keyof typeof categoryConfig];
        return <Tag color={config.color}>{config.label}</Tag>;
      },
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      width: 130,
      render: (severity, record) => (
        <Select
          value={severity}
          onChange={(v) => handleSeverityChange(record.id, v)}
          style={{ width: 110 }}
          size="small"
          options={Object.entries(severityConfig).map(([key, config]) => ({
            value: key,
            label: (
              <Space>
                <span style={{ color: config.color === 'default' ? '#64748b' : config.color }}>
                  {config.icon}
                </span>
                {config.label}
              </Space>
            ),
          }))}
        />
      ),
    },
    {
      title: 'Enabled',
      dataIndex: 'enabled',
      width: 100,
      render: (enabled, record) => (
        <Switch
          checked={enabled}
          onChange={(checked) => handleToggleRule(record.id, checked)}
          size="small"
        />
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_, record) => (
        <Space>
          <Tooltip title="Edit">
            <Button size="small" icon={<EditOutlined />} onClick={() => setEditingRule(record)} />
          </Tooltip>
          {record.source === 'custom' && (
            <Tooltip title="Delete">
              <Button size="small" danger icon={<DeleteOutlined />} />
            </Tooltip>
          )}
        </Space>
      ),
    },
  ];

  return (
    <div className="code-quality-rules-page" style={{ maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <SettingOutlined style={{ color: '#2563eb' }} /> Code Quality Rules
          </Title>
          <Text type="secondary">Configure linting rules and code quality standards</Text>
        </div>
        <Space>
          <Button icon={<ImportOutlined />}>Import Rules</Button>
          <Button icon={<ExportOutlined />}>Export</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalOpen(true)}>
            Create Rule
          </Button>
        </Space>
      </div>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Total Rules" value={stats.total} prefix={<FileTextOutlined />} />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Enabled" value={stats.enabled} valueStyle={{ color: '#22c55e' }} prefix={<CheckCircleOutlined />} />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Error Rules" value={stats.errors} valueStyle={{ color: '#ef4444' }} prefix={<StopOutlined />} />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Warning Rules" value={stats.warnings} valueStyle={{ color: '#f59e0b' }} prefix={<WarningOutlined />} />
          </Card>
        </Col>
      </Row>

      {/* Filters */}
      <Card style={{ marginBottom: 16, borderRadius: 12 }}>
        <Space wrap>
          <Input.Search
            placeholder="Search rules..."
            style={{ width: 280 }}
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            allowClear
          />
          <Select
            value={selectedCategory}
            onChange={setSelectedCategory}
            style={{ width: 160 }}
            options={[
              { value: 'all', label: 'All Categories' },
              ...Object.entries(categoryConfig).map(([key, config]) => ({
                value: key,
                label: <Space>{config.icon} {config.label}</Space>,
              })),
            ]}
          />
        </Space>
      </Card>

      {/* Rules Table */}
      <Card title={<><CodeOutlined /> Rules Configuration</>} style={{ borderRadius: 12 }}>
        <Table
          columns={columns}
          dataSource={filteredRules}
          rowKey="id"
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* Create Rule Modal */}
      <Modal
        title={<><PlusOutlined /> Create Custom Rule</>}
        open={createModalOpen}
        onCancel={() => {
          setCreateModalOpen(false);
          form.resetFields();
        }}
        onOk={() => form.submit()}
        okText="Create"
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={handleCreateRule}>
          <Form.Item name="name" label="Rule Name" rules={[{ required: true }]}>
            <Input placeholder="e.g., custom-naming-convention" />
          </Form.Item>
          <Form.Item name="description" label="Description" rules={[{ required: true }]}>
            <Input.TextArea rows={2} placeholder="Describe what this rule checks" />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="category" label="Category" rules={[{ required: true }]}>
                <Select
                  options={Object.entries(categoryConfig).map(([key, config]) => ({
                    value: key,
                    label: config.label,
                  }))}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="severity" label="Severity" initialValue="warning">
                <Select
                  options={Object.entries(severityConfig).map(([key, config]) => ({
                    value: key,
                    label: config.label,
                  }))}
                />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="autoFix" label="Auto-fix Available" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item name="tags" label="Tags">
            <Select mode="tags" placeholder="Add tags" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default CodeQualityRules;
