/**
 * Evolution Cycle Dashboard
 * 
 * Monitors the three-version self-evolution cycle:
 * - V1 Experimentation: New technology testing
 * - V2 Production: Stable user-facing AI
 * - V3 Quarantine: Failed experiment archive
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Tag,
  Button,
  Space,
  Progress,
  Tabs,
  Statistic,
  Timeline,
  Typography,
  Select,
  Spin,
  Descriptions,
  Alert,
  message,
  Divider,
  notification,
} from 'antd';
import {
  ExperimentOutlined,
  RocketOutlined,
  StopOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  ReloadOutlined,
  SyncOutlined,
  ThunderboltOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { useTranslation } from 'react-i18next';
import { aiService } from '../services/aiService';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

// Types
interface Technology {
  tech_id: string;
  name: string;
  category: string;
  description: string;
  source: string;
  version: 'v1' | 'v2' | 'v3';
  status: 'experimental' | 'promoted' | 'quarantined' | 're-evaluating';
  metrics: {
    accuracy?: number;
    error_rate?: number;
    latency_p95_ms?: number;
    sample_count?: number;
  };
  created_at: string;
}

interface Experiment {
  experiment_id: string;
  name: string;
  technology_type: string;
  category: string;
  status: 'pending' | 'running' | 'evaluating' | 'completed' | 'failed' | 'promoted' | 'quarantined';
  samples_collected: number;
  min_samples: number;
  accuracy: number;
  error_rate: number;
  latency_p95: number;
  started_at: string;
  recommendation?: string;
}

interface VersionMetrics {
  request_count: number;
  error_count: number;
  error_rate: number;
  avg_latency_ms: number;
  technology_count: number;
}

interface PromotionRecord {
  record_id: string;
  from_version: string;
  to_version: string;
  technology_name: string;
  reason: string;
  timestamp: string;
}

interface CycleStatus {
  running: boolean;
  cycle_count: number;
  last_cycle_at: string;
  promotions: number;
  degradations: number;
  v1_metrics: VersionMetrics;
  v2_metrics: VersionMetrics;
  v3_metrics: VersionMetrics;
}

// Version colors
const versionColors: Record<string, string> = {
  v1: 'orange',
  v2: 'green',
  v3: 'red',
};

const versionIcons: Record<string, React.ReactNode> = {
  v1: <ExperimentOutlined />,
  v2: <RocketOutlined />,
  v3: <StopOutlined />,
};

// Status colors
const statusColors: Record<string, string> = {
  pending: 'default',
  running: 'processing',
  evaluating: 'warning',
  completed: 'success',
  failed: 'error',
  promoted: 'green',
  quarantined: 'red',
  experimental: 'orange',
  're-evaluating': 'blue',
};

const EvolutionCycleDashboard: React.FC = () => {
  const { t: _t } = useTranslation();

  // State
  const [technologies, setTechnologies] = useState<Technology[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [promotionHistory, setPromotionHistory] = useState<PromotionRecord[]>([]);
  const [cycleStatus, setCycleStatus] = useState<CycleStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedVersion, setSelectedVersion] = useState<string>('all');

  // Fetch data from real API with fallback to mock
  const fetchData = useCallback(async () => {
    try {
      setLoading(true);

      // Try to fetch real data from API
      try {
        const [cycleResponse, techResponse] = await Promise.all([
          aiService.getCycleStatus(),
          aiService.getTechnologies(),
        ]);

        if (cycleResponse) {
          setCycleStatus(cycleResponse as unknown as CycleStatus);
        }
        if (techResponse && techResponse.length > 0) {
          // Map API response to local Technology type
          const mappedTech: Technology[] = techResponse.map((t: any) => ({
            tech_id: t.id,
            name: t.name,
            category: 'ai-model',
            description: t.name,
            source: 'API',
            version: t.version as 'v1' | 'v2' | 'v3',
            status: t.status === 'active' ? 'promoted' : t.status === 'testing' ? 'experimental' : 'quarantined',
            metrics: {
              accuracy: t.accuracy,
              error_rate: t.errorRate,
              latency_p95_ms: t.latency,
              sample_count: t.samples,
            },
            created_at: t.lastUpdated,
          }));
          setTechnologies(mappedTech);
        }
        setLoading(false);
        return;
      } catch (apiError) {
        // API not available, use mock data
        console.log('Using mock data - API not available');
      }

      // Fallback mock data for development
      const mockTechnologies: Technology[] = [
        {
          tech_id: 'tech-001',
          name: 'Grouped-Query Attention',
          category: 'attention',
          description: 'Efficient KV sharing for reduced memory',
          source: 'LLMs-from-scratch',
          version: 'v1',
          status: 'experimental',
          metrics: { accuracy: 0.87, error_rate: 0.03, latency_p95_ms: 2500, sample_count: 750 },
          created_at: new Date().toISOString(),
        },
        {
          tech_id: 'tech-002',
          name: 'Multi-Head Attention',
          category: 'attention',
          description: 'Standard transformer attention',
          source: 'LLMs-from-scratch',
          version: 'v2',
          status: 'promoted',
          metrics: { accuracy: 0.92, error_rate: 0.02, latency_p95_ms: 2000, sample_count: 5000 },
          created_at: new Date(Date.now() - 86400000).toISOString(),
        },
        {
          tech_id: 'tech-003',
          name: 'Sliding Window Attention',
          category: 'attention',
          description: 'Local attention for long sequences',
          source: 'LLMs-from-scratch',
          version: 'v3',
          status: 'quarantined',
          metrics: { accuracy: 0.68, error_rate: 0.15, latency_p95_ms: 4500, sample_count: 1200 },
          created_at: new Date(Date.now() - 172800000).toISOString(),
        },
        {
          tech_id: 'tech-004',
          name: 'KV Cache Optimization',
          category: 'optimization',
          description: 'Efficient autoregressive generation',
          source: 'LLMs-from-scratch',
          version: 'v2',
          status: 'promoted',
          metrics: { accuracy: 0.91, error_rate: 0.01, latency_p95_ms: 1500, sample_count: 8000 },
          created_at: new Date(Date.now() - 259200000).toISOString(),
        },
      ];

      const mockExperiments: Experiment[] = [
        {
          experiment_id: 'exp-001',
          name: 'GQA Code Review Test',
          technology_type: 'grouped_query_attention',
          category: 'attention',
          status: 'running',
          samples_collected: 750,
          min_samples: 1000,
          accuracy: 0.87,
          error_rate: 0.03,
          latency_p95: 2500,
          started_at: new Date().toISOString(),
        },
        {
          experiment_id: 'exp-002',
          name: 'DPO Alignment Test',
          technology_type: 'direct_preference_optimization',
          category: 'training',
          status: 'evaluating',
          samples_collected: 1200,
          min_samples: 1000,
          accuracy: 0.89,
          error_rate: 0.04,
          latency_p95: 2800,
          started_at: new Date(Date.now() - 3600000).toISOString(),
          recommendation: 'PROMOTE: All thresholds met',
        },
      ];

      const mockPromotionHistory: PromotionRecord[] = [
        {
          record_id: 'promo-001',
          from_version: 'v1',
          to_version: 'v2',
          technology_name: 'Multi-Head Attention',
          reason: 'Passed all evaluation criteria',
          timestamp: new Date(Date.now() - 86400000).toISOString(),
        },
        {
          record_id: 'promo-002',
          from_version: 'v2',
          to_version: 'v3',
          technology_name: 'Sliding Window Attention',
          reason: 'Error rate exceeded threshold (15% > 5%)',
          timestamp: new Date(Date.now() - 172800000).toISOString(),
        },
      ];

      const mockCycleStatus: CycleStatus = {
        running: true,
        cycle_count: 24,
        last_cycle_at: new Date().toISOString(),
        promotions: 5,
        degradations: 2,
        v1_metrics: {
          request_count: 2500,
          error_count: 75,
          error_rate: 0.03,
          avg_latency_ms: 2500,
          technology_count: 3,
        },
        v2_metrics: {
          request_count: 15000,
          error_count: 150,
          error_rate: 0.01,
          avg_latency_ms: 1800,
          technology_count: 5,
        },
        v3_metrics: {
          request_count: 500,
          error_count: 100,
          error_rate: 0.20,
          avg_latency_ms: 4000,
          technology_count: 2,
        },
      };

      setTechnologies(mockTechnologies);
      setExperiments(mockExperiments);
      setPromotionHistory(mockPromotionHistory);
      setCycleStatus(mockCycleStatus);
    } catch (error) {
      console.error('Failed to fetch data:', error);
      message.error('Failed to load evolution cycle data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Filter technologies
  const filteredTechnologies = technologies.filter((tech) => {
    if (selectedVersion !== 'all' && tech.version !== selectedVersion) return false;
    return true;
  });

  // Technology table columns
  const techColumns: ColumnsType<Technology> = [
    {
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
      width: 100,
      render: (version: string) => (
        <Tag color={versionColors[version]} icon={versionIcons[version]}>
          {version.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: Technology) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.category}
          </Text>
        </div>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => (
        <Tag color={statusColors[status]}>{status.toUpperCase()}</Tag>
      ),
    },
    {
      title: 'Accuracy',
      key: 'accuracy',
      width: 120,
      render: (_: unknown, record: Technology) => (
        <Progress
          percent={Math.round((record.metrics.accuracy || 0) * 100)}
          size="small"
          status={
            (record.metrics.accuracy || 0) >= 0.85 ? 'success' : 'exception'
          }
        />
      ),
    },
    {
      title: 'Error Rate',
      key: 'error_rate',
      width: 100,
      render: (_: unknown, record: Technology) => (
        <Text
          type={
            (record.metrics.error_rate || 0) <= 0.05 ? 'success' : 'danger'
          }
        >
          {((record.metrics.error_rate || 0) * 100).toFixed(1)}%
        </Text>
      ),
    },
    {
      title: 'Latency (p95)',
      key: 'latency',
      width: 120,
      render: (_: unknown, record: Technology) => (
        <Text
          type={
            (record.metrics.latency_p95_ms || 0) <= 3000 ? 'success' : 'warning'
          }
        >
          {record.metrics.latency_p95_ms || 0}ms
        </Text>
      ),
    },
    {
      title: 'Samples',
      key: 'samples',
      width: 100,
      render: (_: unknown, record: Technology) => (
        <Text>{record.metrics.sample_count || 0}</Text>
      ),
    },
  ];

  // Experiment table columns
  const expColumns: ColumnsType<Experiment> = [
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => (
        <Tag color={statusColors[status]}>
          {status === 'running' && <SyncOutlined spin />} {status.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Experiment',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: Experiment) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.technology_type}
          </Text>
        </div>
      ),
    },
    {
      title: 'Progress',
      key: 'progress',
      width: 150,
      render: (_: unknown, record: Experiment) => (
        <Progress
          percent={Math.round((record.samples_collected / record.min_samples) * 100)}
          size="small"
          format={() => `${record.samples_collected}/${record.min_samples}`}
        />
      ),
    },
    {
      title: 'Accuracy',
      dataIndex: 'accuracy',
      key: 'accuracy',
      width: 100,
      render: (accuracy: number) => (
        <Text type={accuracy >= 0.85 ? 'success' : 'warning'}>
          {(accuracy * 100).toFixed(1)}%
        </Text>
      ),
    },
    {
      title: 'Error Rate',
      dataIndex: 'error_rate',
      key: 'error_rate',
      width: 100,
      render: (rate: number) => (
        <Text type={rate <= 0.05 ? 'success' : 'danger'}>
          {(rate * 100).toFixed(1)}%
        </Text>
      ),
    },
    {
      title: 'Recommendation',
      dataIndex: 'recommendation',
      key: 'recommendation',
      render: (rec: string) =>
        rec ? (
          <Tag color={rec.includes('PROMOTE') ? 'green' : 'orange'}>{rec}</Tag>
        ) : (
          '-'
        ),
    },
  ];

  // Render version card
  const renderVersionCard = (
    version: string,
    title: string,
    icon: React.ReactNode,
    metrics: VersionMetrics,
    description: string
  ) => (
    <Card
      title={
        <Space>
          {icon}
          {title}
        </Space>
      }
      extra={<Tag color={versionColors[version]}>{version.toUpperCase()}</Tag>}
    >
      <Descriptions column={1} size="small">
        <Descriptions.Item label="Requests">
          {metrics.request_count.toLocaleString()}
        </Descriptions.Item>
        <Descriptions.Item label="Error Rate">
          <Text type={metrics.error_rate <= 0.05 ? 'success' : 'danger'}>
            {(metrics.error_rate * 100).toFixed(2)}%
          </Text>
        </Descriptions.Item>
        <Descriptions.Item label="Avg Latency">
          <Text type={metrics.avg_latency_ms <= 3000 ? 'success' : 'warning'}>
            {metrics.avg_latency_ms.toFixed(0)}ms
          </Text>
        </Descriptions.Item>
        <Descriptions.Item label="Technologies">
          {metrics.technology_count}
        </Descriptions.Item>
      </Descriptions>
      <Divider style={{ margin: '12px 0' }} />
      <Text type="secondary" style={{ fontSize: 12 }}>
        {description}
      </Text>
    </Card>
  );

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div style={{ padding: 24 }}>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}>
            <ThunderboltOutlined style={{ marginRight: 8 }} />
            Evolution Cycle Dashboard
          </Title>
        </Col>
        <Col>
          <Space>
            <Tag color={cycleStatus?.running ? 'green' : 'default'}>
              {cycleStatus?.running ? (
                <>
                  <SyncOutlined spin /> Cycle #{cycleStatus.cycle_count}
                </>
              ) : (
                'Stopped'
              )}
            </Tag>
            <Button type="primary" icon={<ReloadOutlined />} onClick={fetchData}>
              Refresh
            </Button>
          </Space>
        </Col>
      </Row>

      {/* Cycle Flow Diagram */}
      <Alert
        message="Three-Version Evolution Cycle"
        description={
          <div style={{ textAlign: 'center', padding: '16px 0' }}>
            <Space size="large" align="center">
              <div>
                <Tag color="orange" style={{ fontSize: 16, padding: '8px 16px' }}>
                  <ExperimentOutlined /> V1 EXPERIMENTATION
                </Tag>
                <br />
                <Text type="secondary">Test new tech</Text>
              </div>
              <ArrowUpOutlined style={{ fontSize: 24, color: '#52c41a' }} />
              <div>
                <Tag color="green" style={{ fontSize: 16, padding: '8px 16px' }}>
                  <RocketOutlined /> V2 PRODUCTION
                </Tag>
                <br />
                <Text type="secondary">Stable for users</Text>
              </div>
              <ArrowDownOutlined style={{ fontSize: 24, color: '#ff4d4f' }} />
              <div>
                <Tag color="red" style={{ fontSize: 16, padding: '8px 16px' }}>
                  <StopOutlined /> V3 QUARANTINE
                </Tag>
                <br />
                <Text type="secondary">Failed tech archive</Text>
              </div>
            </Space>
          </div>
        }
        type="info"
        style={{ marginBottom: 24 }}
      />

      {/* Summary Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Cycles"
              value={cycleStatus?.cycle_count || 0}
              prefix={<SyncOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Promotions (V1âV2)"
              value={cycleStatus?.promotions || 0}
              prefix={<ArrowUpOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Degradations (V2âV3)"
              value={cycleStatus?.degradations || 0}
              prefix={<ArrowDownOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Active Experiments"
              value={experiments.filter((e) => e.status === 'running').length}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Version Cards */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} md={8}>
          {cycleStatus &&
            renderVersionCard(
              'v1',
              'V1 Experimentation',
              <ExperimentOutlined />,
              cycleStatus.v1_metrics,
              'Testing ground for new AI technologies. Relaxed thresholds allow trial-and-error.'
            )}
        </Col>
        <Col xs={24} md={8}>
          {cycleStatus &&
            renderVersionCard(
              'v2',
              'V2 Production',
              <RocketOutlined />,
              cycleStatus.v2_metrics,
              'Stable user-facing AI. Only proven technologies with strict SLO enforcement.'
            )}
        </Col>
        <Col xs={24} md={8}>
          {cycleStatus &&
            renderVersionCard(
              'v3',
              'V3 Quarantine',
              <StopOutlined />,
              cycleStatus.v3_metrics,
              'Archive for failed experiments. Technologies here may retry after 30 days.'
            )}
        </Col>
      </Row>

      {/* Main Content */}
      <Card>
        <Tabs defaultActiveKey="technologies">
          <TabPane tab="Technologies" key="technologies">
            <Space style={{ marginBottom: 16 }}>
              <Select
                style={{ width: 150 }}
                value={selectedVersion}
                onChange={(value: string) => setSelectedVersion(value)}
                options={[
                  { value: 'all', label: 'All Versions' },
                  { value: 'v1', label: 'V1 Experimental' },
                  { value: 'v2', label: 'V2 Production' },
                  { value: 'v3', label: 'V3 Quarantine' },
                ]}
              />
            </Space>
            <Table
              columns={techColumns}
              dataSource={filteredTechnologies}
              rowKey="tech_id"
              pagination={{ pageSize: 10 }}
            />
          </TabPane>

          <TabPane tab="Active Experiments" key="experiments">
            <Table
              columns={expColumns}
              dataSource={experiments}
              rowKey="experiment_id"
              pagination={{ pageSize: 10 }}
            />
          </TabPane>

          <TabPane tab="Promotion History" key="history">
            <Timeline mode="left">
              {promotionHistory.map((record) => (
                <Timeline.Item
                  key={record.record_id}
                  color={record.to_version === 'v2' ? 'green' : 'red'}
                  label={new Date(record.timestamp).toLocaleString()}
                >
                  <Space direction="vertical" size={0}>
                    <Text strong>{record.technology_name}</Text>
                    <Space>
                      <Tag color={versionColors[record.from_version]}>
                        {record.from_version.toUpperCase()}
                      </Tag>
                      â?
                      <Tag color={versionColors[record.to_version]}>
                        {record.to_version.toUpperCase()}
                      </Tag>
                    </Space>
                    <Text type="secondary">{record.reason}</Text>
                  </Space>
                </Timeline.Item>
              ))}
            </Timeline>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default EvolutionCycleDashboard;
