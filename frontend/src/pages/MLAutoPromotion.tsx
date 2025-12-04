/**
 * ML Auto-Promotion Dashboard
 * 
 * Intelligent machine learning-based version promotion:
 * - Model performance metrics
 * - Auto-promotion rules
 * - A/B testing results
 * - Promotion history
 * - Rollback controls
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Progress,
  Tag,
  Table,
  Statistic,
  Switch,
  Slider,
  Alert,
  Timeline,
  Modal,
  Tooltip,
  Badge,
  Divider,
  message,
} from 'antd';
import {
  RocketOutlined,
  ExperimentOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  SettingOutlined,
  HistoryOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
  LineChartOutlined,
  WarningOutlined,
  RollbackOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
} from '@ant-design/icons';
import './MLAutoPromotion.css';

const { Title, Text, Paragraph } = Typography;

// Mock data types
interface ModelVersion {
  id: string;
  version: string;
  stage: 'development' | 'staging' | 'production' | 'quarantine';
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  latency: number;
  errorRate: number;
  deployedAt: string | null;
  promotedBy: 'auto' | 'manual' | null;
}

interface PromotionRule {
  id: string;
  name: string;
  enabled: boolean;
  metric: string;
  operator: 'gt' | 'lt' | 'gte' | 'lte' | 'eq';
  threshold: number;
  priority: number;
}

interface PromotionEvent {
  id: string;
  fromVersion: string;
  toVersion: string;
  timestamp: string;
  status: 'success' | 'failed' | 'rolled_back';
  reason: string;
  promotedBy: 'auto' | 'manual';
  metrics: Record<string, number>;
}

// Mock data
const mockVersions: ModelVersion[] = [
  {
    id: '1',
    version: 'v3.2.1',
    stage: 'production',
    accuracy: 0.945,
    precision: 0.932,
    recall: 0.918,
    f1Score: 0.925,
    latency: 245,
    errorRate: 0.012,
    deployedAt: '2024-01-15T10:30:00Z',
    promotedBy: 'auto',
  },
  {
    id: '2',
    version: 'v3.3.0-beta',
    stage: 'staging',
    accuracy: 0.958,
    precision: 0.951,
    recall: 0.943,
    f1Score: 0.947,
    latency: 232,
    errorRate: 0.008,
    deployedAt: '2024-01-18T14:20:00Z',
    promotedBy: null,
  },
  {
    id: '3',
    version: 'v3.3.1-dev',
    stage: 'development',
    accuracy: 0.962,
    precision: 0.955,
    recall: 0.948,
    f1Score: 0.951,
    latency: 228,
    errorRate: 0.006,
    deployedAt: null,
    promotedBy: null,
  },
  {
    id: '4',
    version: 'v3.1.0',
    stage: 'quarantine',
    accuracy: 0.891,
    precision: 0.878,
    recall: 0.865,
    f1Score: 0.871,
    latency: 312,
    errorRate: 0.045,
    deployedAt: null,
    promotedBy: null,
  },
];

const mockRules: PromotionRule[] = [
  { id: '1', name: 'Accuracy Threshold', enabled: true, metric: 'accuracy', operator: 'gte', threshold: 0.90, priority: 1 },
  { id: '2', name: 'Error Rate Limit', enabled: true, metric: 'errorRate', operator: 'lte', threshold: 0.02, priority: 2 },
  { id: '3', name: 'Latency Limit', enabled: true, metric: 'latency', operator: 'lte', threshold: 300, priority: 3 },
  { id: '4', name: 'F1 Score Minimum', enabled: true, metric: 'f1Score', operator: 'gte', threshold: 0.85, priority: 4 },
];

const mockEvents: PromotionEvent[] = [
  {
    id: '1',
    fromVersion: 'v3.2.0',
    toVersion: 'v3.2.1',
    timestamp: '2024-01-15T10:30:00Z',
    status: 'success',
    reason: 'All promotion criteria met',
    promotedBy: 'auto',
    metrics: { accuracy: 0.945, latency: 245, errorRate: 0.012 },
  },
  {
    id: '2',
    fromVersion: 'v3.1.0',
    toVersion: 'v3.2.0',
    timestamp: '2024-01-10T08:15:00Z',
    status: 'success',
    reason: 'Manual promotion by admin',
    promotedBy: 'manual',
    metrics: { accuracy: 0.938, latency: 258, errorRate: 0.015 },
  },
  {
    id: '3',
    fromVersion: 'v3.0.5',
    toVersion: 'v3.1.0',
    timestamp: '2024-01-05T16:45:00Z',
    status: 'rolled_back',
    reason: 'Error rate exceeded threshold after 2 hours',
    promotedBy: 'auto',
    metrics: { accuracy: 0.891, latency: 312, errorRate: 0.045 },
  },
];

export const MLAutoPromotion: React.FC = () => {
  const [versions, setVersions] = useState<ModelVersion[]>(mockVersions);
  const [rules, setRules] = useState<PromotionRule[]>(mockRules);
  const [events] = useState<PromotionEvent[]>(mockEvents);
  const [autoPromotionEnabled, setAutoPromotionEnabled] = useState(true);
  const [selectedVersion, setSelectedVersion] = useState<ModelVersion | null>(null);
  const [showPromoteModal, setShowPromoteModal] = useState(false);
  const [isPromoting, setIsPromoting] = useState(false);

  // Simulated real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setVersions(prev => prev.map(v => ({
        ...v,
        accuracy: v.accuracy + (Math.random() - 0.5) * 0.001,
        latency: v.latency + (Math.random() - 0.5) * 5,
      })));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleToggleRule = (ruleId: string) => {
    setRules(prev => prev.map(r => 
      r.id === ruleId ? { ...r, enabled: !r.enabled } : r
    ));
  };

  const handlePromote = async () => {
    if (!selectedVersion) return;
    
    setIsPromoting(true);
    // Simulate promotion
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    setVersions(prev => prev.map(v => {
      if (v.id === selectedVersion.id) {
        return { ...v, stage: 'production', promotedBy: 'manual', deployedAt: new Date().toISOString() };
      }
      if (v.stage === 'production') {
        return { ...v, stage: 'staging' };
      }
      return v;
    }));
    
    setIsPromoting(false);
    setShowPromoteModal(false);
    message.success(`Successfully promoted ${selectedVersion.version} to production!`);
  };

  const getStageColor = (stage: ModelVersion['stage']) => {
    switch (stage) {
      case 'production': return '#22c55e';
      case 'staging': return '#3b82f6';
      case 'development': return '#f59e0b';
      case 'quarantine': return '#ef4444';
      default: return '#64748b';
    }
  };

  const getStageIcon = (stage: ModelVersion['stage']) => {
    switch (stage) {
      case 'production': return <RocketOutlined />;
      case 'staging': return <ExperimentOutlined />;
      case 'development': return <SettingOutlined />;
      case 'quarantine': return <WarningOutlined />;
      default: return null;
    }
  };

  const renderMetricGauge = (value: number, label: string, threshold: number, inverse = false) => {
    const percent = inverse ? (1 - value) * 100 : value * 100;
    const isGood = inverse ? value <= threshold : value >= threshold;
    
    return (
      <div className="metric-gauge">
        <Progress
          type="circle"
          percent={Math.min(percent, 100)}
          size={80}
          strokeColor={isGood ? '#22c55e' : '#ef4444'}
          format={() => (
            <span style={{ fontSize: 14, fontWeight: 600 }}>
              {inverse ? (value * 100).toFixed(1) + '%' : (value * 100).toFixed(1) + '%'}
            </span>
          )}
        />
        <Text type="secondary" style={{ marginTop: 8, display: 'block' }}>{label}</Text>
      </div>
    );
  };

  const versionColumns = [
    {
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
      render: (version: string, record: ModelVersion) => (
        <Space>
          <Badge status={record.stage === 'production' ? 'success' : 'default'} />
          <Text strong>{version}</Text>
        </Space>
      ),
    },
    {
      title: 'Stage',
      dataIndex: 'stage',
      key: 'stage',
      render: (stage: ModelVersion['stage']) => (
        <Tag icon={getStageIcon(stage)} color={getStageColor(stage)}>
          {stage.charAt(0).toUpperCase() + stage.slice(1)}
        </Tag>
      ),
    },
    {
      title: 'Accuracy',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (val: number) => (
        <Progress 
          percent={val * 100} 
          size="small" 
          format={(p) => `${p?.toFixed(1)}%`}
          strokeColor={val >= 0.9 ? '#22c55e' : val >= 0.85 ? '#f59e0b' : '#ef4444'}
        />
      ),
    },
    {
      title: 'Error Rate',
      dataIndex: 'errorRate',
      key: 'errorRate',
      render: (val: number) => (
        <Text type={val <= 0.02 ? 'success' : 'danger'}>
          {(val * 100).toFixed(2)}%
        </Text>
      ),
    },
    {
      title: 'Latency',
      dataIndex: 'latency',
      key: 'latency',
      render: (val: number) => (
        <Text type={val <= 300 ? 'success' : 'warning'}>
          {val.toFixed(0)}ms
        </Text>
      ),
    },
    {
      title: 'Promoted By',
      dataIndex: 'promotedBy',
      key: 'promotedBy',
      render: (val: 'auto' | 'manual' | null) => (
        val ? (
          <Tag color={val === 'auto' ? 'cyan' : 'purple'}>
            {val === 'auto' ? <ThunderboltOutlined /> : null} {val}
          </Tag>
        ) : <Text type="secondary">-</Text>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: unknown, record: ModelVersion) => (
        <Space>
          {record.stage !== 'production' && record.stage !== 'quarantine' && (
            <Tooltip title="Promote to next stage">
              <Button 
                type="primary" 
                size="small" 
                icon={<RocketOutlined />}
                onClick={() => {
                  setSelectedVersion(record);
                  setShowPromoteModal(true);
                }}
              >
                Promote
              </Button>
            </Tooltip>
          )}
          {record.stage === 'production' && (
            <Tooltip title="Rollback to previous version">
              <Button 
                danger 
                size="small" 
                icon={<RollbackOutlined />}
              >
                Rollback
              </Button>
            </Tooltip>
          )}
        </Space>
      ),
    },
  ];

  return (
    <div className="ml-auto-promotion-page">
      {/* Header */}
      <div className="page-header">
        <div className="header-content">
          <div>
            <Title level={3} style={{ margin: 0, color: 'white' }}>
              <ThunderboltOutlined /> ML Auto-Promotion
            </Title>
            <Text style={{ color: 'rgba(255,255,255,0.85)' }}>
              Intelligent version promotion powered by machine learning
            </Text>
          </div>
          <Space>
            <div className="auto-promotion-toggle">
              <Text style={{ color: 'white', marginRight: 12 }}>Auto-Promotion</Text>
              <Switch
                checked={autoPromotionEnabled}
                onChange={setAutoPromotionEnabled}
                checkedChildren={<CheckCircleOutlined />}
                unCheckedChildren={<CloseCircleOutlined />}
              />
            </div>
            <Button icon={<SettingOutlined />} ghost>Configure</Button>
          </Space>
        </div>
      </div>

      {/* Status Alert */}
      {autoPromotionEnabled && (
        <Alert
          type="success"
          icon={<SyncOutlined spin />}
          message="Auto-Promotion Active"
          description="The system is monitoring model performance and will automatically promote versions that meet all criteria."
          showIcon
          style={{ marginBottom: 24, borderRadius: 12 }}
        />
      )}

      {/* Key Metrics */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <Card className="metric-card">
            <Statistic
              title="Production Accuracy"
              value={versions.find(v => v.stage === 'production')?.accuracy || 0}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#22c55e' }}
              formatter={(val) => ((val as number) * 100).toFixed(1)}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card className="metric-card">
            <Statistic
              title="Avg Latency"
              value={versions.find(v => v.stage === 'production')?.latency || 0}
              suffix="ms"
              valueStyle={{ color: '#06b6d4' }}
              prefix={<LineChartOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card className="metric-card">
            <Statistic
              title="Error Rate"
              value={(versions.find(v => v.stage === 'production')?.errorRate || 0) * 100}
              precision={2}
              suffix="%"
              valueStyle={{ color: '#f59e0b' }}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card className="metric-card">
            <Statistic
              title="Models in Pipeline"
              value={versions.filter(v => v.stage !== 'quarantine').length}
              valueStyle={{ color: '#6366f1' }}
              prefix={<ExperimentOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={24}>
        {/* Model Versions */}
        <Col xs={24} lg={16}>
          <Card 
            title={<><RocketOutlined /> Model Versions</>}
            className="versions-card"
          >
            <Table
              dataSource={versions}
              columns={versionColumns}
              rowKey="id"
              pagination={false}
              size="middle"
            />
          </Card>

          {/* Promotion Rules */}
          <Card 
            title={<><SafetyCertificateOutlined /> Promotion Rules</>}
            className="rules-card"
            style={{ marginTop: 24 }}
          >
            <Row gutter={[16, 16]}>
              {rules.map(rule => (
                <Col xs={24} sm={12} key={rule.id}>
                  <div className={`rule-item ${rule.enabled ? 'enabled' : 'disabled'}`}>
                    <div className="rule-header">
                      <Text strong>{rule.name}</Text>
                      <Switch
                        size="small"
                        checked={rule.enabled}
                        onChange={() => handleToggleRule(rule.id)}
                      />
                    </div>
                    <div className="rule-details">
                      <Tag>{rule.metric}</Tag>
                      <Text type="secondary">
                        {rule.operator === 'gte' ? '≥' : rule.operator === 'lte' ? '≤' : rule.operator}
                      </Text>
                      <Text strong>{rule.threshold}</Text>
                    </div>
                    <Slider
                      value={rule.threshold * (rule.metric === 'accuracy' || rule.metric === 'f1Score' ? 100 : 1)}
                      disabled={!rule.enabled}
                      min={0}
                      max={rule.metric === 'latency' ? 500 : 100}
                      tooltip={{ formatter: (val) => rule.metric === 'latency' ? `${val}ms` : `${val}%` }}
                    />
                  </div>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* Timeline & Activity */}
        <Col xs={24} lg={8}>
          {/* Current Production */}
          <Card title={<><RocketOutlined /> Production Model</>} className="production-card">
            {(() => {
              const prod = versions.find(v => v.stage === 'production');
              if (!prod) return <Text type="secondary">No production model</Text>;
              
              return (
                <div className="production-details">
                  <Title level={4} style={{ marginBottom: 16 }}>{prod.version}</Title>
                  <Row gutter={16}>
                    {renderMetricGauge(prod.accuracy, 'Accuracy', 0.9)}
                    {renderMetricGauge(prod.f1Score, 'F1 Score', 0.85)}
                  </Row>
                  <Divider />
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div className="metric-row">
                      <Text type="secondary">Precision</Text>
                      <Text strong>{(prod.precision * 100).toFixed(1)}%</Text>
                    </div>
                    <div className="metric-row">
                      <Text type="secondary">Recall</Text>
                      <Text strong>{(prod.recall * 100).toFixed(1)}%</Text>
                    </div>
                    <div className="metric-row">
                      <Text type="secondary">Latency</Text>
                      <Text strong>{prod.latency.toFixed(0)}ms</Text>
                    </div>
                    <div className="metric-row">
                      <Text type="secondary">Error Rate</Text>
                      <Text strong type={prod.errorRate <= 0.02 ? 'success' : 'danger'}>
                        {(prod.errorRate * 100).toFixed(2)}%
                      </Text>
                    </div>
                  </Space>
                </div>
              );
            })()}
          </Card>

          {/* Promotion History */}
          <Card 
            title={<><HistoryOutlined /> Promotion History</>} 
            className="history-card"
            style={{ marginTop: 24 }}
          >
            <Timeline
              items={events.map(event => ({
                color: event.status === 'success' ? 'green' : event.status === 'rolled_back' ? 'red' : 'gray',
                dot: event.status === 'success' ? <CheckCircleOutlined /> : 
                     event.status === 'rolled_back' ? <RollbackOutlined /> : <CloseCircleOutlined />,
                children: (
                  <div className="timeline-item">
                    <div className="timeline-header">
                      <Text strong>{event.toVersion}</Text>
                      <Tag color={event.promotedBy === 'auto' ? 'cyan' : 'purple'} style={{ marginLeft: 8 }}>
                        {event.promotedBy}
                      </Tag>
                    </div>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {new Date(event.timestamp).toLocaleDateString()}
                    </Text>
                    <Paragraph type="secondary" style={{ marginTop: 4, marginBottom: 0, fontSize: 12 }}>
                      {event.reason}
                    </Paragraph>
                  </div>
                ),
              }))}
            />
          </Card>
        </Col>
      </Row>

      {/* Promote Modal */}
      <Modal
        title={<><RocketOutlined /> Promote Version</>}
        open={showPromoteModal}
        onCancel={() => setShowPromoteModal(false)}
        onOk={handlePromote}
        confirmLoading={isPromoting}
        okText="Promote to Production"
      >
        {selectedVersion && (
          <div className="promote-modal-content">
            <Alert
              type="warning"
              message="Manual Promotion"
              description="This will bypass auto-promotion rules. Make sure you've reviewed the metrics."
              showIcon
              style={{ marginBottom: 16 }}
            />
            <Paragraph>
              You are about to promote <Text strong>{selectedVersion.version}</Text> to production.
            </Paragraph>
            <div className="version-metrics">
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic title="Accuracy" value={selectedVersion.accuracy * 100} precision={1} suffix="%" />
                </Col>
                <Col span={12}>
                  <Statistic title="Error Rate" value={selectedVersion.errorRate * 100} precision={2} suffix="%" />
                </Col>
              </Row>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default MLAutoPromotion;
