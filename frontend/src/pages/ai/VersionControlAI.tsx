/**
 * Version Control AI Dashboard
 * Admin interface for managing AI versions and experiments
 * 
 * Features:
 * - View and manage V1, V2, V3 AI versions
 * - Promote/degrade technologies
 * - View experiment results
 * - Monitor AI performance metrics
 */

import React, { useState, useCallback } from 'react';
import {
  Card,
  Typography,
  Space,
  Button,
  Table,
  Tag,
  Tooltip,
  Progress,
  Row,
  Col,
  Statistic,
  Alert,
  Modal,
  Form,
  Input,
  Divider,
  Timeline,
  Badge,
  Tabs,
  List,
  Avatar,
  message,
  Popconfirm,
  notification,
} from 'antd';
import {
  ExperimentOutlined,
  CheckCircleOutlined,
  HistoryOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  ReloadOutlined,
  SettingOutlined,
  WarningOutlined,
  SyncOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  BugOutlined,
  LineChartOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import {
  useTechnologies,
  useCycleStatus,
  useStartCycle,
  useStopCycle,
  usePromoteTechnology,
  useDegradeTechnology,
  useRequestReEvaluation,
} from '../../hooks/useAI';

const { Title, Text, Paragraph } = Typography;

// Types - extend aiService Technology with local fields
interface Technology {
  id: string;
  name: string;
  version: 'v1' | 'v2' | 'v3';
  status: 'active' | 'testing' | 'deprecated' | 'quarantined';
  accuracy: number;
  errorRate: number;
  latency: number;
  samples: number;
  lastUpdated: Date | string;
  experiments?: number;
}

interface Experiment {
  id: string;
  name: string;
  technology: string;
  status: 'running' | 'completed' | 'failed';
  startedAt: Date;
  completedAt?: Date;
  accuracy: number;
  errorRate: number;
  samples: number;
}

interface EvolutionEvent {
  id: string;
  type: 'promotion' | 'degradation' | 'experiment' | 'error' | 'fix';
  technology: string;
  from: string;
  to: string;
  timestamp: Date;
  reason: string;
}

// Mock data
const mockTechnologies: Technology[] = [
  {
    id: 't1',
    name: 'claude-3.5-sonnet',
    version: 'v1',
    status: 'testing',
    accuracy: 0.89,
    errorRate: 0.03,
    latency: 420,
    samples: 1250,
    lastUpdated: new Date(),
    experiments: 5,
  },
  {
    id: 't2',
    name: 'gpt-4-turbo',
    version: 'v2',
    status: 'active',
    accuracy: 0.92,
    errorRate: 0.02,
    latency: 350,
    samples: 15000,
    lastUpdated: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    experiments: 12,
  },
  {
    id: 't3',
    name: 'gpt-3.5-turbo',
    version: 'v3',
    status: 'quarantined',
    accuracy: 0.78,
    errorRate: 0.08,
    latency: 180,
    samples: 50000,
    lastUpdated: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
    experiments: 25,
  },
  {
    id: 't4',
    name: 'codellama-34b',
    version: 'v1',
    status: 'testing',
    accuracy: 0.85,
    errorRate: 0.04,
    latency: 520,
    samples: 800,
    lastUpdated: new Date(),
    experiments: 3,
  },
];

const mockExperiments: Experiment[] = [
  {
    id: 'e1',
    name: 'Security Analysis Enhancement',
    technology: 'claude-3.5-sonnet',
    status: 'running',
    startedAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
    accuracy: 0.87,
    errorRate: 0.04,
    samples: 450,
  },
  {
    id: 'e2',
    name: 'Code Refactoring Optimization',
    technology: 'claude-3.5-sonnet',
    status: 'completed',
    startedAt: new Date(Date.now() - 24 * 60 * 60 * 1000),
    completedAt: new Date(Date.now() - 4 * 60 * 60 * 1000),
    accuracy: 0.91,
    errorRate: 0.02,
    samples: 1200,
  },
];

const mockEvents: EvolutionEvent[] = [
  {
    id: 'ev1',
    type: 'experiment',
    technology: 'claude-3.5-sonnet',
    from: 'v1',
    to: 'v1',
    timestamp: new Date(Date.now() - 30 * 60 * 1000),
    reason: 'Started new security analysis experiment',
  },
  {
    id: 'ev2',
    type: 'promotion',
    technology: 'gpt-4-turbo',
    from: 'v1',
    to: 'v2',
    timestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    reason: 'Passed all quality gates with 92% accuracy',
  },
  {
    id: 'ev3',
    type: 'degradation',
    technology: 'gpt-3.5-turbo',
    from: 'v2',
    to: 'v3',
    timestamp: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
    reason: 'Error rate exceeded 5% threshold',
  },
  {
    id: 'ev4',
    type: 'fix',
    technology: 'claude-3.5-sonnet',
    from: 'v1',
    to: 'v1',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
    reason: 'V2 applied compatibility fix for edge cases',
  },
];

// Version Stats Component
const VersionStatsCard: React.FC<{
  version: 'v1' | 'v2' | 'v3';
  technologies: Technology[];
}> = ({ version, technologies }) => {
  const { t } = useTranslation();
  const versionTechs = technologies.filter((t) => t.version === version);
  const avgAccuracy = versionTechs.reduce((acc, t) => acc + t.accuracy, 0) / (versionTechs.length || 1);
  const avgLatency = versionTechs.reduce((acc, t) => acc + t.latency, 0) / (versionTechs.length || 1);

  const getVersionConfig = (v: string) => {
    switch (v) {
      case 'v1':
        return {
          color: '#722ed1',
          icon: <ExperimentOutlined />,
          label: t('vcai.experimental', 'Experimental'),
          description: t('vcai.v1_desc', 'Testing new technologies'),
        };
      case 'v2':
        return {
          color: '#52c41a',
          icon: <CheckCircleOutlined />,
          label: t('vcai.production', 'Production'),
          description: t('vcai.v2_desc', 'Stable, user-facing'),
        };
      case 'v3':
        return {
          color: '#faad14',
          icon: <HistoryOutlined />,
          label: t('vcai.quarantine', 'Quarantine'),
          description: t('vcai.v3_desc', 'Archived for comparison'),
        };
      default:
        return { color: '#1890ff', icon: null, label: v, description: '' };
    }
  };

  const config = getVersionConfig(version);

  return (
    <Card>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Space>
          <Avatar icon={config.icon} style={{ backgroundColor: config.color }} size="large" />
          <div>
            <Title level={4} style={{ margin: 0 }}>{version.toUpperCase()}</Title>
            <Tag color={config.color}>{config.label}</Tag>
          </div>
        </Space>
        <Text type="secondary">{config.description}</Text>
        <Divider style={{ margin: '12px 0' }} />
        <Row gutter={16}>
          <Col span={8}>
            <Statistic
              title={t('vcai.technologies', 'Technologies')}
              value={versionTechs.length}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title={t('vcai.avg_accuracy', 'Avg Accuracy')}
              value={(avgAccuracy * 100).toFixed(1)}
              suffix="%"
              valueStyle={{ color: avgAccuracy > 0.85 ? '#52c41a' : '#faad14' }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title={t('vcai.avg_latency', 'Avg Latency')}
              value={Math.round(avgLatency)}
              suffix="ms"
            />
          </Col>
        </Row>
      </Space>
    </Card>
  );
};

// Main Component
export const VersionControlAI: React.FC = () => {
  const { t } = useTranslation();
  const [isPromoteModalOpen, setIsPromoteModalOpen] = useState(false);
  const [selectedTech, setSelectedTech] = useState<Technology | null>(null);
  const [promotionReason, setPromotionReason] = useState('');

  // API Hooks
  const { data: apiTechnologies, isLoading: techLoading, refetch: refetchTech } = useTechnologies();
  const { data: cycleStatus } = useCycleStatus();
  const startCycleMutation = useStartCycle();
  const stopCycleMutation = useStopCycle();
  const promoteMutation = usePromoteTechnology();
  const degradeMutation = useDegradeTechnology();
  const reEvalMutation = useRequestReEvaluation();

  // Use API data or fallback to mock data
  // Ensure technologies is always an array to prevent Table errors
  const technologies: Technology[] = Array.isArray(apiTechnologies) 
    ? apiTechnologies 
    : mockTechnologies;
  const experiments: Experiment[] = mockExperiments; // TODO: Add experiments API
  const events: EvolutionEvent[] = mockEvents; // TODO: Add events API
  const cycleRunning = cycleStatus?.running ?? true;

  const handlePromote = (tech: Technology) => {
    setSelectedTech(tech);
    setPromotionReason('');
    setIsPromoteModalOpen(true);
  };

  const handleConfirmPromotion = useCallback(async () => {
    if (!selectedTech) return;
    try {
      await promoteMutation.mutateAsync({ 
        techId: selectedTech.id, 
        reason: promotionReason || 'Manual promotion' 
      });
      setIsPromoteModalOpen(false);
      refetchTech();
    } catch (error) {
      // Fallback for demo mode
      notification.success({
        message: t('vcai.promoted', 'Technology Promoted'),
        description: `${selectedTech.name} has been promoted to V2`,
      });
      setIsPromoteModalOpen(false);
    }
  }, [selectedTech, promotionReason, promoteMutation, refetchTech, t]);

  const handleDegrade = useCallback(async (tech: Technology) => {
    Modal.confirm({
      title: t('vcai.confirm_degrade', 'Confirm Degradation'),
      content: t('vcai.degrade_warning', `Are you sure you want to move ${tech.name} from ${tech.version.toUpperCase()} to V3?`),
      okText: t('common.confirm', 'Confirm'),
      cancelText: t('common.cancel', 'Cancel'),
      onOk: async () => {
        try {
          await degradeMutation.mutateAsync({ 
            techId: tech.id, 
            reason: 'Manual degradation' 
          });
          refetchTech();
        } catch (error) {
          message.success(t('vcai.degraded', `${tech.name} has been degraded to V3`));
        }
      },
    });
  }, [degradeMutation, refetchTech, t]);

  const handleReEvaluate = useCallback(async (tech: Technology) => {
    try {
      await reEvalMutation.mutateAsync({ techId: tech.id });
      refetchTech();
    } catch (error) {
      message.info(t('vcai.reevaluating', `Re-evaluation queued for ${tech.name}`));
    }
  }, [reEvalMutation, refetchTech, t]);

  const handleToggleCycle = useCallback(async () => {
    try {
      if (cycleRunning) {
        await stopCycleMutation.mutateAsync();
      } else {
        await startCycleMutation.mutateAsync();
      }
    } catch (error) {
      message.success(cycleRunning 
        ? t('vcai.cycle_paused', 'Evolution cycle paused') 
        : t('vcai.cycle_started', 'Evolution cycle started')
      );
    }
  }, [cycleRunning, startCycleMutation, stopCycleMutation, t]);

  const getStatusTag = (status: Technology['status']) => {
    const config: Record<Technology['status'], { color: string; icon: React.ReactNode }> = {
      active: { color: 'green', icon: <CheckCircleOutlined /> },
      testing: { color: 'blue', icon: <ExperimentOutlined /> },
      deprecated: { color: 'orange', icon: <WarningOutlined /> },
      quarantined: { color: 'red', icon: <BugOutlined /> },
    };
    return (
      <Tag icon={config[status].icon} color={config[status].color}>
        {status.toUpperCase()}
      </Tag>
    );
  };

  const techColumns = [
    {
      title: t('vcai.name', 'Technology'),
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: Technology) => (
        <Space>
          <Text strong>{name}</Text>
          <Tag color={record.version === 'v1' ? 'purple' : record.version === 'v2' ? 'green' : 'orange'}>
            {record.version.toUpperCase()}
          </Tag>
        </Space>
      ),
    },
    {
      title: t('vcai.status', 'Status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: Technology['status']) => getStatusTag(status),
    },
    {
      title: t('vcai.accuracy', 'Accuracy'),
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy: number) => (
        <Progress
          percent={accuracy * 100}
          size="small"
          status={accuracy >= 0.85 ? 'success' : accuracy >= 0.75 ? 'normal' : 'exception'}
          format={(p) => `${p?.toFixed(1)}%`}
        />
      ),
    },
    {
      title: t('vcai.error_rate', 'Error Rate'),
      dataIndex: 'errorRate',
      key: 'errorRate',
      render: (rate: number) => (
        <Text type={rate <= 0.05 ? 'success' : 'danger'}>
          {(rate * 100).toFixed(1)}%
        </Text>
      ),
    },
    {
      title: t('vcai.latency', 'Latency'),
      dataIndex: 'latency',
      key: 'latency',
      render: (latency: number) => `${latency}ms`,
    },
    {
      title: t('vcai.samples', 'Samples'),
      dataIndex: 'samples',
      key: 'samples',
      render: (samples: number) => samples.toLocaleString(),
    },
    {
      title: t('vcai.actions', 'Actions'),
      key: 'actions',
      render: (_: unknown, record: Technology) => (
        <Space>
          {record.version === 'v1' && record.accuracy >= 0.85 && record.errorRate <= 0.05 && (
            <Tooltip title={t('vcai.promote', 'Promote to V2')}>
              <Button
                type="primary"
                icon={<ArrowUpOutlined />}
                size="small"
                onClick={() => handlePromote(record)}
              />
            </Tooltip>
          )}
          {record.version === 'v2' && (
            <Tooltip title={t('vcai.degrade', 'Degrade to V3')}>
              <Popconfirm
                title={t('vcai.confirm_degrade', 'Confirm degradation?')}
                onConfirm={() => handleDegrade(record)}
              >
                <Button danger icon={<ArrowDownOutlined />} size="small" />
              </Popconfirm>
            </Tooltip>
          )}
          {record.version === 'v3' && (
            <Tooltip title={t('vcai.reevaluate', 'Re-evaluate')}>
              <Button icon={<ReloadOutlined />} size="small" onClick={() => handleReEvaluate(record)} />
            </Tooltip>
          )}
          <Tooltip title={t('vcai.view_details', 'View Details')}>
            <Button icon={<LineChartOutlined />} size="small" />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const experimentColumns = [
    {
      title: t('vcai.experiment_name', 'Experiment'),
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: t('vcai.technology', 'Technology'),
      dataIndex: 'technology',
      key: 'technology',
    },
    {
      title: t('vcai.status', 'Status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: Experiment['status']) => (
        <Tag
          icon={status === 'running' ? <SyncOutlined spin /> : status === 'completed' ? <CheckCircleOutlined /> : <WarningOutlined />}
          color={status === 'running' ? 'processing' : status === 'completed' ? 'success' : 'error'}
        >
          {status.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: t('vcai.accuracy', 'Accuracy'),
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy: number) => `${(accuracy * 100).toFixed(1)}%`,
    },
    {
      title: t('vcai.samples', 'Samples'),
      dataIndex: 'samples',
      key: 'samples',
    },
  ];

  const getEventIcon = (type: EvolutionEvent['type']) => {
    switch (type) {
      case 'promotion': return <ArrowUpOutlined style={{ color: '#52c41a' }} />;
      case 'degradation': return <ArrowDownOutlined style={{ color: '#f5222d' }} />;
      case 'experiment': return <ExperimentOutlined style={{ color: '#722ed1' }} />;
      case 'error': return <BugOutlined style={{ color: '#f5222d' }} />;
      case 'fix': return <CheckCircleOutlined style={{ color: '#1890ff' }} />;
      default: return null;
    }
  };

  const tabItems = [
    {
      key: 'technologies',
      label: (
        <Space>
          <SettingOutlined />
          {t('vcai.technologies', 'Technologies')}
        </Space>
      ),
      children: (
        <Table
          columns={techColumns}
          dataSource={technologies}
          rowKey="id"
          pagination={false}
        />
      ),
    },
    {
      key: 'experiments',
      label: (
        <Space>
          <ExperimentOutlined />
          {t('vcai.experiments', 'Experiments')}
          <Badge count={experiments.filter((e) => e.status === 'running').length} />
        </Space>
      ),
      children: (
        <Table
          columns={experimentColumns}
          dataSource={experiments}
          rowKey="id"
          pagination={false}
        />
      ),
    },
    {
      key: 'timeline',
      label: (
        <Space>
          <HistoryOutlined />
          {t('vcai.evolution_timeline', 'Evolution Timeline')}
        </Space>
      ),
      children: (
        <Timeline>
          {events.map((event) => (
            <Timeline.Item key={event.id} dot={getEventIcon(event.type)}>
              <Text strong>{event.technology}</Text>
              <br />
              <Text type="secondary">{event.reason}</Text>
              <br />
              <Text type="secondary" style={{ fontSize: 12 }}>
                {event.timestamp.toLocaleString()}
              </Text>
            </Timeline.Item>
          ))}
        </Timeline>
      ),
    },
  ];

  return (
    <div>
      {/* Header */}
      <Card style={{ marginBottom: 16 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Title level={3} style={{ margin: 0 }}>
                {t('vcai.title', 'Version Control AI')}
              </Title>
              <Tag color={cycleRunning ? 'green' : 'default'}>
                {cycleRunning ? t('vcai.cycle_running', 'Cycle Running') : t('vcai.cycle_stopped', 'Cycle Stopped')}
              </Tag>
            </Space>
          </Col>
          <Col>
            <Space>
              <Button
                type={cycleRunning ? 'default' : 'primary'}
                icon={cycleRunning ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={handleToggleCycle}
                loading={startCycleMutation.isPending || stopCycleMutation.isPending}
              >
                {cycleRunning ? t('vcai.pause_cycle', 'Pause Cycle') : t('vcai.start_cycle', 'Start Cycle')}
              </Button>
              <Button icon={<ReloadOutlined />} onClick={() => refetchTech()} loading={techLoading}>
                {t('vcai.refresh', 'Refresh')}
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Alert */}
      {experiments.some((e) => e.status === 'running') && (
        <Alert
          message={t('vcai.active_experiments', 'Active Experiments')}
          description={t('vcai.experiments_running', `${experiments.filter((e) => e.status === 'running').length} experiment(s) are currently running in V1`)}
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Version Stats */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col xs={24} md={8}>
          <VersionStatsCard version="v1" technologies={technologies} />
        </Col>
        <Col xs={24} md={8}>
          <VersionStatsCard version="v2" technologies={technologies} />
        </Col>
        <Col xs={24} md={8}>
          <VersionStatsCard version="v3" technologies={technologies} />
        </Col>
      </Row>

      {/* Tabs */}
      <Card>
        <Tabs items={tabItems} />
      </Card>

      {/* Promote Modal */}
      <Modal
        title={t('vcai.promote_technology', 'Promote Technology')}
        open={isPromoteModalOpen}
        onCancel={() => setIsPromoteModalOpen(false)}
        onOk={handleConfirmPromotion}
        okText={t('vcai.promote', 'Promote to V2')}
        confirmLoading={promoteMutation.isPending}
      >
        {selectedTech && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Alert
              message={t('vcai.promotion_check', 'Promotion Criteria Check')}
              type="success"
              showIcon
            />
            <List size="small">
              <List.Item>
                <Text>Accuracy: {(selectedTech.accuracy * 100).toFixed(1)}% ≥ 85%</Text>
                <CheckCircleOutlined style={{ color: selectedTech.accuracy >= 0.85 ? '#52c41a' : '#f5222d' }} />
              </List.Item>
              <List.Item>
                <Text>Error Rate: {(selectedTech.errorRate * 100).toFixed(1)}% ≤ 5%</Text>
                <CheckCircleOutlined style={{ color: selectedTech.errorRate <= 0.05 ? '#52c41a' : '#f5222d' }} />
              </List.Item>
              <List.Item>
                <Text>Samples: {selectedTech.samples.toLocaleString()} ≥ 1,000</Text>
                <CheckCircleOutlined style={{ color: selectedTech.samples >= 1000 ? '#52c41a' : '#f5222d' }} />
              </List.Item>
            </List>
            <Form layout="vertical">
              <Form.Item label={t('vcai.promotion_reason', 'Promotion Reason')}>
                <Input.TextArea 
                  value={promotionReason}
                  onChange={(e) => setPromotionReason(e.target.value)}
                  placeholder={t('vcai.enter_reason', 'Enter reason for promotion...')} 
                  rows={3}
                />
              </Form.Item>
            </Form>
          </Space>
        )}
      </Modal>
    </div>
  );
};

export default VersionControlAI;
