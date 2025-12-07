/**
 * V1 Experimental Hub
 *
 * Admin-only dashboard for experimental features.
 * Testing ground for new AI models and configurations.
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  Layout,
  Card,
  Row,
  Col,
  Input,
  Typography,
  Space,
  Tag,
  Alert,
  Progress,
  Statistic,
  Button,
  Badge,
  Table,
  Tooltip,
  Tabs,
  Timeline,
  Empty,
} from 'antd';
import {
  ExperimentOutlined,
  SearchOutlined,
  RocketOutlined,
  BugOutlined,
  ThunderboltOutlined,
  SyncOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ArrowRightOutlined,
  LineChartOutlined,
  BarChartOutlined,
  AlertOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { AdminOnly } from '../../components/common/PermissionGate';

const { Content } = Layout;
const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

// =============================================================================
// V1 Experimental Functions
// =============================================================================

interface ExperimentFunction {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  path: string;
  status: 'running' | 'completed' | 'failed' | 'pending';
  progress?: number;
  metrics?: {
    accuracy?: number;
    latency?: number;
    errorRate?: number;
  };
}

const EXPERIMENT_FUNCTIONS: ExperimentFunction[] = [
  {
    id: 'experiments',
    name: 'Experiment Management',
    description: 'Create and manage AI experiments',
    icon: <ExperimentOutlined />,
    path: '/admin/experiments',
    status: 'running',
    progress: 67,
    metrics: { accuracy: 92.5, latency: 245, errorRate: 1.2 },
  },
  {
    id: 'ai-testing',
    name: 'AI Model Testing',
    description: 'Test models with sample data',
    icon: <BugOutlined />,
    path: '/admin/model-testing',
    status: 'completed',
    progress: 100,
    metrics: { accuracy: 94.8, latency: 189, errorRate: 0.8 },
  },
  {
    id: 'model-comparison',
    name: 'Model Comparison',
    description: 'Compare model performance',
    icon: <ThunderboltOutlined />,
    path: '/admin/model-comparison',
    status: 'running',
    progress: 45,
    metrics: { accuracy: 91.2, latency: 312, errorRate: 2.1 },
  },
  {
    id: 'learning-cycle',
    name: 'Learning Cycle',
    description: 'Continuous learning dashboard',
    icon: <RocketOutlined />,
    path: '/admin/learning',
    status: 'running',
    progress: 78,
    metrics: { accuracy: 93.1, latency: 203, errorRate: 1.5 },
  },
  {
    id: 'evolution-cycle',
    name: 'Evolution Cycle',
    description: 'AI self-evolution monitoring',
    icon: <SyncOutlined />,
    path: '/admin/evolution',
    status: 'pending',
    progress: 0,
  },
];

// =============================================================================
// Experiment Card
// =============================================================================

interface ExperimentCardProps {
  experiment: ExperimentFunction;
  onNavigate: (path: string) => void;
}

const ExperimentCard: React.FC<ExperimentCardProps> = ({ experiment, onNavigate }) => {
  const statusColors = {
    running: '#1890ff',
    completed: '#52c41a',
    failed: '#f5222d',
    pending: '#8c8c8c',
  };

  const statusIcons = {
    running: <PlayCircleOutlined spin />,
    completed: <CheckCircleOutlined />,
    failed: <AlertOutlined />,
    pending: <ClockCircleOutlined />,
  };

  return (
    <Card
      hoverable
      onClick={() => onNavigate(experiment.path)}
      style={{
        height: '100%',
        borderTop: `3px solid ${statusColors[experiment.status]}`,
      }}
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <div
            style={{
              width: 48,
              height: 48,
              borderRadius: 12,
              background: `${statusColors[experiment.status]}15`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 24,
              color: statusColors[experiment.status],
            }}
          >
            {experiment.icon}
          </div>
          <Tag
            color={statusColors[experiment.status]}
            icon={statusIcons[experiment.status]}
          >
            {experiment.status.toUpperCase()}
          </Tag>
        </Space>

        <div>
          <Text strong style={{ fontSize: 16 }}>{experiment.name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {experiment.description}
          </Text>
        </div>

        {experiment.progress !== undefined && experiment.status !== 'pending' && (
          <Progress
            percent={experiment.progress}
            size="small"
            status={experiment.status === 'failed' ? 'exception' : undefined}
            strokeColor={statusColors[experiment.status]}
          />
        )}

        {experiment.metrics && (
          <Row gutter={8}>
            <Col span={8}>
              <Tooltip title="Accuracy">
                <Text type="secondary" style={{ fontSize: 11 }}>
                  <CheckCircleOutlined /> {experiment.metrics.accuracy}%
                </Text>
              </Tooltip>
            </Col>
            <Col span={8}>
              <Tooltip title="Latency (ms)">
                <Text type="secondary" style={{ fontSize: 11 }}>
                  <ClockCircleOutlined /> {experiment.metrics.latency}ms
                </Text>
              </Tooltip>
            </Col>
            <Col span={8}>
              <Tooltip title="Error Rate">
                <Text
                  type="secondary"
                  style={{
                    fontSize: 11,
                    color: experiment.metrics.errorRate! > 2 ? '#f5222d' : undefined,
                  }}
                >
                  <WarningOutlined /> {experiment.metrics.errorRate}%
                </Text>
              </Tooltip>
            </Col>
          </Row>
        )}

        <Button
          type="link"
          size="small"
          style={{ padding: 0, color: statusColors[experiment.status] }}
        >
          Open <ArrowRightOutlined />
        </Button>
      </Space>
    </Card>
  );
};

// =============================================================================
// Mock Active Experiments Data
// =============================================================================

const ACTIVE_EXPERIMENTS = [
  {
    key: '1',
    name: 'GPT-4 Turbo Integration',
    model: 'gpt-4-turbo',
    status: 'running',
    accuracy: 94.2,
    progress: 72,
    started: '2 hours ago',
  },
  {
    key: '2',
    name: 'Claude-3 Opus Test',
    model: 'claude-3-opus',
    status: 'running',
    accuracy: 93.8,
    progress: 45,
    started: '4 hours ago',
  },
  {
    key: '3',
    name: 'Code Llama Fine-tune',
    model: 'code-llama-34b',
    status: 'pending',
    accuracy: null,
    progress: 0,
    started: 'Scheduled',
  },
];

// =============================================================================
// Main Component
// =============================================================================

const V1ExperimentalHub: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');

  const filteredExperiments = useMemo(() => {
    if (!searchQuery) return EXPERIMENT_FUNCTIONS;
    const query = searchQuery.toLowerCase();
    return EXPERIMENT_FUNCTIONS.filter(e =>
      e.name.toLowerCase().includes(query) ||
      e.description.toLowerCase().includes(query)
    );
  }, [searchQuery]);

  const handleNavigate = useCallback((path: string) => {
    navigate(path);
  }, [navigate]);

  const runningCount = EXPERIMENT_FUNCTIONS.filter(e => e.status === 'running').length;
  const completedCount = EXPERIMENT_FUNCTIONS.filter(e => e.status === 'completed').length;

  const columns = [
    {
      title: 'Experiment',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Model',
      dataIndex: 'model',
      key: 'model',
      render: (model: string) => <Tag color="purple">{model}</Tag>,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'running' ? 'blue' : 'default'}>
          {status === 'running' && <PlayCircleOutlined spin />} {status.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Accuracy',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (val: number | null) => val ? `${val}%` : '-',
    },
    {
      title: 'Progress',
      dataIndex: 'progress',
      key: 'progress',
      render: (val: number) => <Progress percent={val} size="small" />,
    },
    {
      title: 'Started',
      dataIndex: 'started',
      key: 'started',
    },
  ];

  return (
    <AdminOnly fallback={
      <Alert
        message="Admin Access Required"
        description="V1 Experimental features are only available to administrators."
        type="warning"
        showIcon
        style={{ margin: 24 }}
      />
    }>
      <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
        <Content style={{ padding: 24 }}>
          {/* Header */}
          <div style={{ marginBottom: 24 }}>
            <Space align="center" style={{ marginBottom: 8 }}>
              <Tag color="orange" style={{ fontSize: 14, padding: '4px 12px' }}>
                <ExperimentOutlined /> V1 Experimental
              </Tag>
              <Tag color="red">Admin Only</Tag>
            </Space>

            <Title level={2} style={{ margin: 0 }}>
              Experimental Hub
            </Title>
            <Paragraph type="secondary" style={{ marginBottom: 0 }}>
              Test new AI models and experimental features before production
            </Paragraph>
          </div>

          {/* Warning Banner */}
          <Alert
            message="Experimental Environment"
            description="Functions in this area are under testing. Results may vary and should not be used in production."
            type="warning"
            showIcon
            icon={<WarningOutlined />}
            style={{ marginBottom: 24 }}
          />

          {/* Stats */}
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col xs={24} sm={8}>
              <Card>
                <Statistic
                  title="Running Experiments"
                  value={runningCount}
                  prefix={<PlayCircleOutlined style={{ color: '#1890ff' }} />}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={8}>
              <Card>
                <Statistic
                  title="Completed"
                  value={completedCount}
                  prefix={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={8}>
              <Card>
                <Statistic
                  title="Avg. Accuracy"
                  value={92.4}
                  suffix="%"
                  prefix={<BarChartOutlined style={{ color: '#722ed1' }} />}
                  valueStyle={{ color: '#722ed1' }}
                />
              </Card>
            </Col>
          </Row>

          {/* Search */}
          <Card style={{ marginBottom: 24 }}>
            <Input
              placeholder="Search experimental functions..."
              prefix={<SearchOutlined />}
              size="large"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              allowClear
            />
          </Card>

          <Tabs defaultActiveKey="functions">
            <TabPane tab="Functions" key="functions">
              {/* Experiment Functions */}
              <Card
                title={
                  <Space>
                    <ExperimentOutlined style={{ color: '#faad14' }} />
                    <span>Experimental Functions</span>
                    <Badge count={runningCount} style={{ backgroundColor: '#1890ff' }} />
                  </Space>
                }
                style={{ marginBottom: 24 }}
              >
                <Row gutter={[16, 16]}>
                  {filteredExperiments.map(experiment => (
                    <Col xs={24} sm={12} md={8} key={experiment.id}>
                      <ExperimentCard
                        experiment={experiment}
                        onNavigate={handleNavigate}
                      />
                    </Col>
                  ))}
                </Row>

                {filteredExperiments.length === 0 && (
                  <Empty description="No experiments found" />
                )}
              </Card>
            </TabPane>

            <TabPane tab="Active Experiments" key="active">
              <Card title="Active Experiments">
                <Table
                  dataSource={ACTIVE_EXPERIMENTS}
                  columns={columns}
                  pagination={false}
                />
              </Card>
            </TabPane>

            <TabPane tab="Promotion Queue" key="promotion">
              <Card
                title={
                  <Space>
                    <RocketOutlined style={{ color: '#52c41a' }} />
                    <span>Ready for Promotion to V2</span>
                  </Space>
                }
              >
                <Timeline>
                  <Timeline.Item color="green">
                    <Text strong>GPT-4 Turbo Integration</Text>
                    <br />
                    <Text type="secondary">Accuracy: 94.2% | Error Rate: 0.8%</Text>
                    <br />
                    <Button type="primary" size="small" style={{ marginTop: 8 }}>
                      Promote to V2
                    </Button>
                  </Timeline.Item>
                  <Timeline.Item color="blue">
                    <Text strong>Enhanced Security Scanner</Text>
                    <br />
                    <Text type="secondary">Testing in progress - 89% complete</Text>
                  </Timeline.Item>
                  <Timeline.Item color="gray">
                    <Text strong>Code Llama Integration</Text>
                    <br />
                    <Text type="secondary">Scheduled for testing</Text>
                  </Timeline.Item>
                </Timeline>
              </Card>
            </TabPane>
          </Tabs>
        </Content>
      </Layout>
    </AdminOnly>
  );
};

export default V1ExperimentalHub;
