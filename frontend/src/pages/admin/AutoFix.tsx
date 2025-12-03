/**
 * AI Auto-Fix Management Page (Admin)
 * AI自动修复管理页面
 * 
 * Features:
 * - Automatic vulnerability detection & fixing
 * - AI-powered bug resolution
 * - Version control cycle visualization
 * - Fix approval workflow
 * - Rollback capability
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Tag,
  Space,
  Typography,
  Modal,
  Progress,
  Timeline,
  Alert,
  Statistic,
  Badge,
  Tooltip,
  Switch,
  Tabs,
  List,
  Avatar,
  Descriptions,
  message,
  Popconfirm,
  Steps,
} from 'antd';
import type { TableProps } from 'antd';
import {
  RobotOutlined,
  BugOutlined,
  SafetyCertificateOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
  HistoryOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  RollbackOutlined,
  EyeOutlined,
  CodeOutlined,
  FileTextOutlined,
  ExclamationCircleOutlined,
  SettingOutlined,
  ReloadOutlined,
  CheckOutlined,
  WarningOutlined,
  ArrowRightOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { api } from '../../services/api';

const { Title, Text, Paragraph } = Typography;

interface AutoFix {
  id: string;
  type: 'vulnerability' | 'bug' | 'quality' | 'performance';
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  file: string;
  line: number;
  status: 'pending' | 'analyzing' | 'fixing' | 'review' | 'approved' | 'applied' | 'rejected' | 'failed';
  aiModel: string;
  confidence: number;
  fix?: {
    before: string;
    after: string;
    explanation: string;
  };
  createdAt: string;
  appliedAt?: string;
  appliedBy?: string;
}

interface FixCycle {
  id: string;
  startedAt: string;
  completedAt?: string;
  status: 'running' | 'completed' | 'failed';
  issuesFound: number;
  issuesFixed: number;
  issuesPending: number;
  duration?: number;
}

const mockAutoFixes: AutoFix[] = [
  {
    id: 'fix_1',
    type: 'vulnerability',
    severity: 'critical',
    title: 'SQL Injection in User Query',
    description: 'Unsanitized user input used directly in SQL query',
    file: 'src/api/users.py',
    line: 45,
    status: 'review',
    aiModel: 'GPT-4 Turbo',
    confidence: 0.95,
    fix: {
      before: 'query = f"SELECT * FROM users WHERE id = {user_id}"',
      after: 'query = "SELECT * FROM users WHERE id = %s"\ncursor.execute(query, (user_id,))',
      explanation: 'Replaced string interpolation with parameterized query to prevent SQL injection.',
    },
    createdAt: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
  },
  {
    id: 'fix_2',
    type: 'vulnerability',
    severity: 'critical',
    title: 'Hardcoded API Secret',
    description: 'API secret key exposed in source code',
    file: 'src/config/settings.py',
    line: 12,
    status: 'pending',
    aiModel: 'Claude 3 Opus',
    confidence: 0.98,
    fix: {
      before: 'API_SECRET = "sk_live_xxxxxxxxxxxx"',
      after: 'API_SECRET = os.environ.get("API_SECRET")',
      explanation: 'Moved secret to environment variable for secure handling.',
    },
    createdAt: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
  },
  {
    id: 'fix_3',
    type: 'bug',
    severity: 'high',
    title: 'Null Pointer Exception',
    description: 'Missing null check before accessing object property',
    file: 'src/services/analysis.ts',
    line: 128,
    status: 'applied',
    aiModel: 'GPT-4 Turbo',
    confidence: 0.92,
    fix: {
      before: 'const result = data.analysis.results[0]',
      after: 'const result = data?.analysis?.results?.[0] ?? null',
      explanation: 'Added optional chaining and nullish coalescing for safe access.',
    },
    createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    appliedAt: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
    appliedBy: 'AI Auto-Fix',
  },
  {
    id: 'fix_4',
    type: 'performance',
    severity: 'medium',
    title: 'N+1 Query Problem',
    description: 'Multiple database queries in loop instead of batch query',
    file: 'src/api/projects.py',
    line: 89,
    status: 'analyzing',
    aiModel: 'GPT-4 Turbo',
    confidence: 0.78,
    createdAt: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
  },
  {
    id: 'fix_5',
    type: 'quality',
    severity: 'low',
    title: 'Duplicate Code Block',
    description: 'Similar code blocks can be refactored into a function',
    file: 'src/utils/helpers.ts',
    line: 56,
    status: 'rejected',
    aiModel: 'Claude 3 Sonnet',
    confidence: 0.65,
    createdAt: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
  },
];

const mockCycles: FixCycle[] = [
  {
    id: 'cycle_1',
    startedAt: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    status: 'running',
    issuesFound: 12,
    issuesFixed: 8,
    issuesPending: 4,
  },
  {
    id: 'cycle_2',
    startedAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    completedAt: new Date(Date.now() - 23 * 60 * 60 * 1000).toISOString(),
    status: 'completed',
    issuesFound: 25,
    issuesFixed: 23,
    issuesPending: 0,
    duration: 3600000,
  },
  {
    id: 'cycle_3',
    startedAt: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
    completedAt: new Date(Date.now() - 47 * 60 * 60 * 1000).toISOString(),
    status: 'completed',
    issuesFound: 18,
    issuesFixed: 17,
    issuesPending: 0,
    duration: 2700000,
  },
];

export const AutoFix: React.FC = () => {
  const { t } = useTranslation();
  const [fixes, setFixes] = useState<AutoFix[]>(mockAutoFixes);
  const [cycles, setCycles] = useState<FixCycle[]>(mockCycles);
  const [autoFixEnabled, setAutoFixEnabled] = useState(true);
  const [selectedFix, setSelectedFix] = useState<AutoFix | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  // Statistics
  const stats = {
    totalFixes: fixes.length,
    applied: fixes.filter(f => f.status === 'applied').length,
    pending: fixes.filter(f => ['pending', 'review'].includes(f.status)).length,
    critical: fixes.filter(f => f.severity === 'critical' && f.status !== 'applied').length,
  };

  const currentCycle = cycles.find(c => c.status === 'running');

  // Approve fix
  const handleApprove = async (fix: AutoFix) => {
    try {
      await api.post(`/api/admin/auto-fix/${fix.id}/approve`);
      message.success('Fix approved and applied');
    } catch (error) {
      message.success('Fix approved and applied (demo)');
    }
    setFixes(prev => prev.map(f =>
      f.id === fix.id ? { ...f, status: 'applied' as const, appliedAt: new Date().toISOString(), appliedBy: 'Admin' } : f
    ));
  };

  // Reject fix
  const handleReject = async (fix: AutoFix) => {
    try {
      await api.post(`/api/admin/auto-fix/${fix.id}/reject`);
      message.info('Fix rejected');
    } catch (error) {
      message.info('Fix rejected (demo)');
    }
    setFixes(prev => prev.map(f =>
      f.id === fix.id ? { ...f, status: 'rejected' as const } : f
    ));
  };

  // Rollback fix
  const handleRollback = async (fix: AutoFix) => {
    try {
      await api.post(`/api/admin/auto-fix/${fix.id}/rollback`);
      message.success('Fix rolled back');
    } catch (error) {
      message.success('Fix rolled back (demo)');
    }
    setFixes(prev => prev.map(f =>
      f.id === fix.id ? { ...f, status: 'pending' as const, appliedAt: undefined, appliedBy: undefined } : f
    ));
  };

  // Start new cycle
  const handleStartCycle = async () => {
    try {
      await api.post('/api/admin/auto-fix/cycle/start');
      message.success('New fix cycle started');
    } catch (error) {
      message.success('New fix cycle started (demo)');
      setCycles(prev => [{
        id: `cycle_${Date.now()}`,
        startedAt: new Date().toISOString(),
        status: 'running',
        issuesFound: 0,
        issuesFixed: 0,
        issuesPending: 0,
      }, ...prev]);
    }
  };

  const getTypeConfig = (type: string) => {
    const configs: Record<string, { color: string; icon: React.ReactNode }> = {
      vulnerability: { color: '#ef4444', icon: <SafetyCertificateOutlined /> },
      bug: { color: '#f59e0b', icon: <BugOutlined /> },
      quality: { color: '#3b82f6', icon: <CodeOutlined /> },
      performance: { color: '#8b5cf6', icon: <ThunderboltOutlined /> },
    };
    return configs[type] || { color: '#64748b', icon: <CodeOutlined /> };
  };

  const getSeverityColor = (severity: string) => {
    const colors: Record<string, string> = {
      critical: 'red',
      high: 'orange',
      medium: 'gold',
      low: 'blue',
    };
    return colors[severity] || 'default';
  };

  const getStatusBadge = (status: string) => {
    const configs: Record<string, { color: string; icon: React.ReactNode; text: string }> = {
      pending: { color: 'default', icon: <ClockCircleOutlined />, text: 'Pending' },
      analyzing: { color: 'processing', icon: <SyncOutlined spin />, text: 'Analyzing' },
      fixing: { color: 'processing', icon: <SyncOutlined spin />, text: 'Fixing' },
      review: { color: 'warning', icon: <ExclamationCircleOutlined />, text: 'Review' },
      approved: { color: 'success', icon: <CheckCircleOutlined />, text: 'Approved' },
      applied: { color: 'success', icon: <CheckCircleOutlined />, text: 'Applied' },
      rejected: { color: 'error', icon: <CloseCircleOutlined />, text: 'Rejected' },
      failed: { color: 'error', icon: <CloseCircleOutlined />, text: 'Failed' },
    };
    const config = configs[status] || { color: 'default', icon: null, text: status };
    return <Tag color={config.color} icon={config.icon}>{config.text}</Tag>;
  };

  const columns: TableProps<AutoFix>['columns'] = [
    {
      title: 'Type',
      key: 'type',
      width: 100,
      render: (_, record) => {
        const config = getTypeConfig(record.type);
        return (
          <Tooltip title={record.type.charAt(0).toUpperCase() + record.type.slice(1)}>
            <div style={{
              width: 36,
              height: 36,
              borderRadius: 8,
              background: `${config.color}15`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: config.color,
              fontSize: 18,
            }}>
              {config.icon}
            </div>
          </Tooltip>
        );
      },
    },
    {
      title: 'Issue',
      key: 'issue',
      render: (_, record) => (
        <div>
          <Space>
            <Text strong>{record.title}</Text>
            <Tag color={getSeverityColor(record.severity)}>{record.severity.toUpperCase()}</Tag>
          </Space>
          <div>
            <Text type="secondary" style={{ fontSize: 13 }}>{record.description}</Text>
          </div>
          <div style={{ marginTop: 4 }}>
            <Text code style={{ fontSize: 12 }}>{record.file}:{record.line}</Text>
          </div>
        </div>
      ),
    },
    {
      title: 'AI Model',
      key: 'ai',
      width: 150,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Text>{record.aiModel}</Text>
          <Progress
            percent={Math.round(record.confidence * 100)}
            size="small"
            strokeColor={record.confidence >= 0.9 ? '#52c41a' : record.confidence >= 0.7 ? '#faad14' : '#ff4d4f'}
            format={p => `${p}% conf.`}
            style={{ width: 100 }}
          />
        </Space>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      width: 120,
      render: (status) => getStatusBadge(status),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 200,
      render: (_, record) => (
        <Space>
          <Tooltip title="View Details">
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedFix(record);
                setDetailModalOpen(true);
              }}
            />
          </Tooltip>
          
          {record.status === 'review' && (
            <>
              <Popconfirm
                title="Apply this fix?"
                description="The AI-generated fix will be applied to the codebase."
                onConfirm={() => handleApprove(record)}
              >
                <Button size="small" type="primary" icon={<CheckOutlined />}>
                  Approve
                </Button>
              </Popconfirm>
              <Button size="small" danger icon={<CloseCircleOutlined />} onClick={() => handleReject(record)}>
                Reject
              </Button>
            </>
          )}

          {record.status === 'applied' && (
            <Popconfirm
              title="Rollback this fix?"
              onConfirm={() => handleRollback(record)}
            >
              <Button size="small" icon={<RollbackOutlined />}>
                Rollback
              </Button>
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ];

  return (
    <div className="auto-fix-page">
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <RobotOutlined style={{ color: '#2563eb' }} /> AI Auto-Fix System
          </Title>
          <Text type="secondary">
            Automated vulnerability and bug detection with AI-powered fixes
          </Text>
        </div>
        <Space>
          <span>Auto-Fix:</span>
          <Switch
            checked={autoFixEnabled}
            onChange={setAutoFixEnabled}
            checkedChildren="ON"
            unCheckedChildren="OFF"
          />
          <Button icon={<ReloadOutlined />}>Refresh</Button>
          <Button
            type="primary"
            icon={currentCycle ? <SyncOutlined spin /> : <PlayCircleOutlined />}
            onClick={handleStartCycle}
            disabled={!!currentCycle}
          >
            {currentCycle ? 'Cycle Running' : 'Start Cycle'}
          </Button>
        </Space>
      </div>

      {/* Current Cycle Status */}
      {currentCycle && (
        <Alert
          type="info"
          showIcon
          icon={<SyncOutlined spin />}
          message="Fix Cycle in Progress"
          description={
            <Row gutter={24} align="middle">
              <Col>
                <Statistic title="Issues Found" value={currentCycle.issuesFound} />
              </Col>
              <Col>
                <Statistic title="Fixed" value={currentCycle.issuesFixed} valueStyle={{ color: '#52c41a' }} />
              </Col>
              <Col>
                <Statistic title="Pending" value={currentCycle.issuesPending} valueStyle={{ color: '#faad14' }} />
              </Col>
              <Col flex="1">
                <Progress
                  percent={Math.round((currentCycle.issuesFixed / currentCycle.issuesFound) * 100) || 0}
                  status="active"
                />
              </Col>
            </Row>
          }
          style={{ marginBottom: 24 }}
        />
      )}

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Total Fixes"
              value={stats.totalFixes}
              prefix={<RobotOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Applied"
              value={stats.applied}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Pending Review"
              value={stats.pending}
              valueStyle={{ color: '#faad14' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12, background: stats.critical > 0 ? '#fff1f0' : undefined }}>
            <Statistic
              title="Critical Issues"
              value={stats.critical}
              valueStyle={{ color: stats.critical > 0 ? '#ff4d4f' : '#52c41a' }}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Version Control Cycle Visualization */}
      <Card title={<><HistoryOutlined /> Fix Cycle Pipeline</>} style={{ marginBottom: 24, borderRadius: 12 }}>
        <Steps
          current={currentCycle ? 2 : -1}
          status={currentCycle ? 'process' : 'wait'}
          items={[
            {
              title: 'Scan',
              description: 'Detect issues',
              icon: <EyeOutlined />,
            },
            {
              title: 'Analyze',
              description: 'AI analysis',
              icon: <RobotOutlined />,
            },
            {
              title: 'Fix',
              description: 'Generate fixes',
              icon: <CodeOutlined />,
            },
            {
              title: 'Review',
              description: 'Human review',
              icon: <ExclamationCircleOutlined />,
            },
            {
              title: 'Apply',
              description: 'Deploy fixes',
              icon: <CheckCircleOutlined />,
            },
          ]}
        />
      </Card>

      {/* Fixes Table */}
      <Card title={<><BugOutlined /> Auto-Fix Queue</>} style={{ borderRadius: 12 }}>
        <Tabs
          defaultActiveKey="all"
          items={[
            { key: 'all', label: `All (${fixes.length})` },
            { key: 'review', label: `Needs Review (${fixes.filter(f => f.status === 'review').length})` },
            { key: 'applied', label: `Applied (${fixes.filter(f => f.status === 'applied').length})` },
            { key: 'rejected', label: `Rejected (${fixes.filter(f => f.status === 'rejected').length})` },
          ]}
        />
        <Table
          columns={columns}
          dataSource={fixes}
          rowKey="id"
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* Fix Detail Modal */}
      <Modal
        title={<><RobotOutlined /> Fix Details</>}
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        width={800}
        footer={selectedFix?.status === 'review' ? [
          <Button key="reject" danger onClick={() => {
            if (selectedFix) handleReject(selectedFix);
            setDetailModalOpen(false);
          }}>
            Reject
          </Button>,
          <Button key="approve" type="primary" onClick={() => {
            if (selectedFix) handleApprove(selectedFix);
            setDetailModalOpen(false);
          }}>
            Approve & Apply
          </Button>,
        ] : null}
      >
        {selectedFix && (
          <div>
            <Descriptions bordered column={2}>
              <Descriptions.Item label="Type">
                <Tag color={getTypeConfig(selectedFix.type).color}>
                  {selectedFix.type.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Severity">
                <Tag color={getSeverityColor(selectedFix.severity)}>
                  {selectedFix.severity.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="File" span={2}>
                <Text code>{selectedFix.file}:{selectedFix.line}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="AI Model">{selectedFix.aiModel}</Descriptions.Item>
              <Descriptions.Item label="Confidence">
                <Progress
                  percent={Math.round(selectedFix.confidence * 100)}
                  size="small"
                  style={{ width: 100 }}
                />
              </Descriptions.Item>
            </Descriptions>

            {selectedFix.fix && (
              <div style={{ marginTop: 24 }}>
                <Title level={5}>Proposed Fix</Title>
                
                <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
                  <div style={{ flex: 1 }}>
                    <Text type="secondary" strong>Before:</Text>
                    <pre style={{
                      background: '#fee2e2',
                      padding: 12,
                      borderRadius: 8,
                      fontSize: 13,
                      overflow: 'auto',
                    }}>
                      {selectedFix.fix.before}
                    </pre>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <ArrowRightOutlined style={{ fontSize: 20, color: '#64748b' }} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <Text type="secondary" strong>After:</Text>
                    <pre style={{
                      background: '#d1fae5',
                      padding: 12,
                      borderRadius: 8,
                      fontSize: 13,
                      overflow: 'auto',
                    }}>
                      {selectedFix.fix.after}
                    </pre>
                  </div>
                </div>

                <Alert
                  message="AI Explanation"
                  description={selectedFix.fix.explanation}
                  type="info"
                  showIcon
                  icon={<RobotOutlined />}
                />
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default AutoFix;
