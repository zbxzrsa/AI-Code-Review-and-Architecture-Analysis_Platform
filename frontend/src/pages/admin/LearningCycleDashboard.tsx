/**
 * Learning Cycle Dashboard
 * 
 * Dashboard for monitoring the AI continuous learning cycle:
 * - Learning sources and channels
 * - Knowledge base updates
 * - Model fine-tuning status
 * - Learning metrics and trends
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Tag,
  Space,
  Typography,
  Statistic,
  Progress,
  Alert,
  Badge,
  Spin,
  Switch,
  List,
  Avatar,
} from 'antd';
import {
  BookOutlined,
  CloudDownloadOutlined,
  SyncOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  BranchesOutlined,
  FileTextOutlined,
  GithubOutlined,
  GlobalOutlined,
  RocketOutlined,
  BarChartOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text } = Typography;

interface LearningSource {
  id: string;
  name: string;
  type: 'github' | 'papers' | 'blogs' | 'docs' | 'feedback';
  enabled: boolean;
  lastSync: string;
  itemsProcessed: number;
  status: 'active' | 'syncing' | 'error' | 'paused';
}

interface KnowledgeUpdate {
  id: string;
  source: string;
  title: string;
  type: string;
  timestamp: string;
  impact: 'high' | 'medium' | 'low';
}

interface LearningMetrics {
  totalKnowledgeItems: number;
  itemsToday: number;
  learningAccuracy: number;
  modelVersion: string;
  lastFineTune: string;
  nextScheduledTune: string;
}

const LearningCycleDashboard: React.FC = () => {
  const { t: _t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [sources, setSources] = useState<LearningSource[]>([]);
  const [updates, setUpdates] = useState<KnowledgeUpdate[]>([]);
  const [metrics, setMetrics] = useState<LearningMetrics | null>(null);
  const [cycleRunning, setCycleRunning] = useState(true);

  useEffect(() => {
    // Simulate fetching data
    const timer = setTimeout(() => {
      setSources([
        {
          id: 'src-001',
          name: 'GitHub Repositories',
          type: 'github',
          enabled: true,
          lastSync: new Date().toISOString(),
          itemsProcessed: 15420,
          status: 'active',
        },
        {
          id: 'src-002',
          name: 'arXiv Papers',
          type: 'papers',
          enabled: true,
          lastSync: new Date(Date.now() - 3600000).toISOString(),
          itemsProcessed: 2340,
          status: 'active',
        },
        {
          id: 'src-003',
          name: 'Tech Blogs',
          type: 'blogs',
          enabled: true,
          lastSync: new Date(Date.now() - 7200000).toISOString(),
          itemsProcessed: 8750,
          status: 'syncing',
        },
        {
          id: 'src-004',
          name: 'Documentation',
          type: 'docs',
          enabled: true,
          lastSync: new Date(Date.now() - 1800000).toISOString(),
          itemsProcessed: 4280,
          status: 'active',
        },
        {
          id: 'src-005',
          name: 'User Feedback',
          type: 'feedback',
          enabled: true,
          lastSync: new Date().toISOString(),
          itemsProcessed: 1250,
          status: 'active',
        },
      ]);

      setUpdates([
        {
          id: 'upd-001',
          source: 'GitHub',
          title: 'New React 19 patterns detected',
          type: 'Pattern',
          timestamp: new Date().toISOString(),
          impact: 'high',
        },
        {
          id: 'upd-002',
          source: 'arXiv',
          title: 'GQA attention optimization paper',
          type: 'Research',
          timestamp: new Date(Date.now() - 1800000).toISOString(),
          impact: 'medium',
        },
        {
          id: 'upd-003',
          source: 'Feedback',
          title: 'Security pattern improvement',
          type: 'Improvement',
          timestamp: new Date(Date.now() - 3600000).toISOString(),
          impact: 'high',
        },
        {
          id: 'upd-004',
          source: 'Blogs',
          title: 'TypeScript 5.3 features',
          type: 'Pattern',
          timestamp: new Date(Date.now() - 7200000).toISOString(),
          impact: 'medium',
        },
      ]);

      setMetrics({
        totalKnowledgeItems: 32040,
        itemsToday: 245,
        learningAccuracy: 0.94,
        modelVersion: 'v2.3.1',
        lastFineTune: new Date(Date.now() - 86400000).toISOString(),
        nextScheduledTune: new Date(Date.now() + 43200000).toISOString(),
      });

      setLoading(false);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  const getSourceIcon = (type: string) => {
    switch (type) {
      case 'github':
        return <GithubOutlined />;
      case 'papers':
        return <FileTextOutlined />;
      case 'blogs':
        return <GlobalOutlined />;
      case 'docs':
        return <BookOutlined />;
      case 'feedback':
        return <BranchesOutlined />;
      default:
        return <DatabaseOutlined />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success';
      case 'syncing':
        return 'processing';
      case 'error':
        return 'error';
      case 'paused':
        return 'default';
      default:
        return 'default';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high':
        return 'red';
      case 'medium':
        return 'orange';
      case 'low':
        return 'blue';
      default:
        return 'default';
    }
  };

  const sourceColumns = [
    {
      title: 'Source',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: LearningSource) => (
        <Space>
          {getSourceIcon(record.type)}
          <Text strong>{name}</Text>
        </Space>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge
          status={getStatusColor(status) as any}
          text={status.charAt(0).toUpperCase() + status.slice(1)}
        />
      ),
    },
    {
      title: 'Items Processed',
      dataIndex: 'itemsProcessed',
      key: 'itemsProcessed',
      render: (count: number) => count.toLocaleString(),
    },
    {
      title: 'Last Sync',
      dataIndex: 'lastSync',
      key: 'lastSync',
      render: (date: string) => new Date(date).toLocaleTimeString(),
    },
    {
      title: 'Enabled',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled: boolean, record: LearningSource) => (
        <Switch
          checked={enabled}
          size="small"
          onChange={(checked) => {
            setSources((prev) =>
              prev.map((s) =>
                s.id === record.id ? { ...s, enabled: checked } : s
              )
            );
          }}
        />
      ),
    },
  ];

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
            <BookOutlined style={{ marginRight: 8 }} />
            Learning Cycle Dashboard
          </Title>
        </Col>
        <Col>
          <Space>
            <Badge
              status={cycleRunning ? 'processing' : 'default'}
              text={cycleRunning ? 'Learning Active' : 'Paused'}
            />
            <Button
              type={cycleRunning ? 'default' : 'primary'}
              icon={cycleRunning ? <ClockCircleOutlined /> : <RocketOutlined />}
              onClick={() => setCycleRunning(!cycleRunning)}
            >
              {cycleRunning ? 'Pause Learning' : 'Resume Learning'}
            </Button>
            <Button icon={<SyncOutlined />}>Sync All</Button>
          </Space>
        </Col>
      </Row>

      {/* Metrics */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Knowledge Items"
              value={metrics?.totalKnowledgeItems || 0}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Learned Today"
              value={metrics?.itemsToday || 0}
              prefix={<CloudDownloadOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Learning Accuracy"
              value={(metrics?.learningAccuracy || 0) * 100}
              precision={1}
              suffix="%"
              prefix={<BarChartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Model Version"
              value={metrics?.modelVersion || 'N/A'}
              prefix={<BranchesOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Fine-tuning Status */}
      <Alert
        message="Next Fine-tuning Scheduled"
        description={
          <Space>
            <Text>
              Last fine-tune:{' '}
              {metrics?.lastFineTune
                ? new Date(metrics.lastFineTune).toLocaleString()
                : 'Never'}
            </Text>
            <Text type="secondary">|</Text>
            <Text>
              Next scheduled:{' '}
              {metrics?.nextScheduledTune
                ? new Date(metrics.nextScheduledTune).toLocaleString()
                : 'Not scheduled'}
            </Text>
          </Space>
        }
        type="info"
        showIcon
        icon={<ClockCircleOutlined />}
        style={{ marginBottom: 24 }}
      />

      <Row gutter={24}>
        {/* Learning Sources */}
        <Col xs={24} lg={14}>
          <Card title="Learning Sources" style={{ marginBottom: 24 }}>
            <Table
              columns={sourceColumns}
              dataSource={sources}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </Card>
        </Col>

        {/* Recent Updates */}
        <Col xs={24} lg={10}>
          <Card title="Recent Knowledge Updates">
            <List
              itemLayout="horizontal"
              dataSource={updates}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Avatar
                        style={{
                          backgroundColor:
                            item.impact === 'high'
                              ? '#ff4d4f'
                              : item.impact === 'medium'
                              ? '#faad14'
                              : '#1890ff',
                        }}
                        icon={<BookOutlined />}
                      />
                    }
                    title={
                      <Space>
                        <Text strong>{item.title}</Text>
                        <Tag color={getImpactColor(item.impact)}>
                          {item.impact.toUpperCase()}
                        </Tag>
                      </Space>
                    }
                    description={
                      <Space>
                        <Tag>{item.source}</Tag>
                        <Text type="secondary">
                          {new Date(item.timestamp).toLocaleTimeString()}
                        </Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      {/* Learning Progress */}
      <Card title="Learning Progress by Category" style={{ marginTop: 24 }}>
        <Row gutter={[16, 16]}>
          {[
            { name: 'Security Patterns', progress: 92, color: '#ff4d4f' },
            { name: 'Code Quality', progress: 88, color: '#52c41a' },
            { name: 'Performance', progress: 75, color: '#1890ff' },
            { name: 'Best Practices', progress: 95, color: '#722ed1' },
            { name: 'New Frameworks', progress: 68, color: '#faad14' },
          ].map((category) => (
            <Col xs={24} sm={12} md={8} lg={4} key={category.name}>
              <div style={{ textAlign: 'center' }}>
                <Progress
                  type="circle"
                  percent={category.progress}
                  strokeColor={category.color}
                  width={80}
                />
                <div style={{ marginTop: 8 }}>
                  <Text>{category.name}</Text>
                </div>
              </div>
            </Col>
          ))}
        </Row>
      </Card>
    </div>
  );
};

export default LearningCycleDashboard;
