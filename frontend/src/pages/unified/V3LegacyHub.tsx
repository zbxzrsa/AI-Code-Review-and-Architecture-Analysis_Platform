/**
 * V3 Legacy Hub
 *
 * Admin-only dashboard for legacy/deprecated features.
 * Used for comparison and rollback scenarios.
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  Layout,
  Card,
  Row,
  Col,
  Typography,
  Space,
  Tag,
  Alert,
  Statistic,
  Button,
  Table,
  Tooltip,
  Tabs,
  Timeline,
  Empty,
  Descriptions,
  Badge,
  Popconfirm,
} from 'antd';
import {
  HistoryOutlined,
  RollbackOutlined,
  CompressOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  InfoCircleOutlined,
  DeleteOutlined,
  ReloadOutlined,
  DatabaseOutlined,
  ApiOutlined,
  LineChartOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { AdminOnly } from '../../components/common/PermissionGate';

const { Content } = Layout;
const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

// =============================================================================
// V3 Legacy Functions
// =============================================================================

interface LegacyFunction {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  path: string;
  deprecatedDate: string;
  replacedBy?: string;
  canReactivate: boolean;
  lastUsed?: string;
}

const LEGACY_FUNCTIONS: LegacyFunction[] = [
  {
    id: 'version-comparison',
    name: 'Version Comparison',
    description: 'Compare different AI model versions',
    icon: <CompressOutlined />,
    path: '/admin/version-comparison',
    deprecatedDate: '2024-10-15',
    replacedBy: 'Model Comparison (V1)',
    canReactivate: true,
    lastUsed: '3 days ago',
  },
  {
    id: 'three-version',
    name: 'Three Version Control',
    description: 'Manage V1/V2/V3 lifecycle',
    icon: <ApiOutlined />,
    path: '/admin/three-version',
    deprecatedDate: '2024-11-01',
    canReactivate: true,
    lastUsed: '1 day ago',
  },
];

// =============================================================================
// Quarantined Models Data
// =============================================================================

const QUARANTINED_MODELS = [
  {
    key: '1',
    name: 'GPT-3.5 Legacy',
    version: 'v3.5.2',
    reason: 'Replaced by GPT-4',
    quarantinedDate: '2024-09-15',
    accuracy: 87.2,
    canRestore: true,
  },
  {
    key: '2',
    name: 'Code Review v1.0',
    version: 'v1.0.0',
    reason: 'Critical bugs found',
    quarantinedDate: '2024-08-20',
    accuracy: 78.5,
    canRestore: false,
  },
  {
    key: '3',
    name: 'Security Scanner v2',
    version: 'v2.1.0',
    reason: 'Performance issues',
    quarantinedDate: '2024-10-01',
    accuracy: 82.1,
    canRestore: true,
  },
];

// =============================================================================
// Comparison History
// =============================================================================

const COMPARISON_HISTORY = [
  {
    id: '1',
    v1Model: 'GPT-4 Turbo',
    v3Model: 'GPT-3.5 Legacy',
    date: '2024-11-28',
    v1Accuracy: 94.2,
    v3Accuracy: 87.2,
    winner: 'V1',
  },
  {
    id: '2',
    v1Model: 'Claude-3 Opus',
    v3Model: 'Claude-2.1',
    date: '2024-11-25',
    v1Accuracy: 93.8,
    v3Accuracy: 85.6,
    winner: 'V1',
  },
  {
    id: '3',
    v1Model: 'Code Llama',
    v3Model: 'StarCoder',
    date: '2024-11-20',
    v1Accuracy: 88.5,
    v3Accuracy: 86.2,
    winner: 'V1',
  },
];

// =============================================================================
// Main Component
// =============================================================================

const V3LegacyHub: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();

  const handleNavigate = useCallback((path: string) => {
    navigate(path);
  }, [navigate]);

  const handleRestore = useCallback((modelName: string) => {
    console.log('Restoring model:', modelName);
    // API call to restore model
  }, []);

  const handleReEvaluate = useCallback((modelName: string) => {
    console.log('Re-evaluating model:', modelName);
    // API call to re-evaluate
  }, []);

  const quarantinedColumns = [
    {
      title: 'Model',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: any) => (
        <Space>
          <DatabaseOutlined />
          <div>
            <Text strong>{name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 11 }}>{record.version}</Text>
          </div>
        </Space>
      ),
    },
    {
      title: 'Reason',
      dataIndex: 'reason',
      key: 'reason',
      render: (reason: string) => (
        <Tag color="orange" icon={<WarningOutlined />}>
          {reason}
        </Tag>
      ),
    },
    {
      title: 'Quarantined',
      dataIndex: 'quarantinedDate',
      key: 'quarantinedDate',
    },
    {
      title: 'Last Accuracy',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (val: number) => (
        <Text type={val < 85 ? 'danger' : 'secondary'}>{val}%</Text>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          <Tooltip title="Re-evaluate for V1 promotion">
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={() => handleReEvaluate(record.name)}
            >
              Re-evaluate
            </Button>
          </Tooltip>
          {record.canRestore && (
            <Popconfirm
              title="Restore this model?"
              description="This will move the model back to V1 for testing."
              onConfirm={() => handleRestore(record.name)}
            >
              <Button
                size="small"
                type="primary"
                icon={<RollbackOutlined />}
              >
                Restore
              </Button>
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ];

  const comparisonColumns = [
    {
      title: 'V1 Model',
      dataIndex: 'v1Model',
      key: 'v1Model',
      render: (name: string) => (
        <Tag color="orange">{name}</Tag>
      ),
    },
    {
      title: 'V3 Model',
      dataIndex: 'v3Model',
      key: 'v3Model',
      render: (name: string) => (
        <Tag color="default">{name}</Tag>
      ),
    },
    {
      title: 'Date',
      dataIndex: 'date',
      key: 'date',
    },
    {
      title: 'V1 Accuracy',
      dataIndex: 'v1Accuracy',
      key: 'v1Accuracy',
      render: (val: number) => <Text style={{ color: '#faad14' }}>{val}%</Text>,
    },
    {
      title: 'V3 Accuracy',
      dataIndex: 'v3Accuracy',
      key: 'v3Accuracy',
      render: (val: number) => <Text type="secondary">{val}%</Text>,
    },
    {
      title: 'Winner',
      dataIndex: 'winner',
      key: 'winner',
      render: (winner: string) => (
        <Tag color={winner === 'V1' ? 'green' : 'default'}>
          {winner === 'V1' ? <ArrowUpOutlined /> : <ArrowDownOutlined />} {winner}
        </Tag>
      ),
    },
  ];

  return (
    <AdminOnly fallback={
      <Alert
        message="Admin Access Required"
        description="V3 Legacy features are only available to administrators."
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
              <Tag color="default" style={{ fontSize: 14, padding: '4px 12px' }}>
                <HistoryOutlined /> V3 Legacy
              </Tag>
              <Tag color="red">Admin Only</Tag>
            </Space>

            <Title level={2} style={{ margin: 0 }}>
              Legacy Hub
            </Title>
            <Paragraph type="secondary" style={{ marginBottom: 0 }}>
              Quarantined models and deprecated features for comparison
            </Paragraph>
          </div>

          {/* Info Banner */}
          <Alert
            message="Legacy Environment"
            description="This area contains deprecated features and quarantined models. Use for comparison and rollback scenarios only."
            type="info"
            showIcon
            icon={<InfoCircleOutlined />}
            style={{ marginBottom: 24 }}
          />

          {/* Stats */}
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col xs={24} sm={8}>
              <Card>
                <Statistic
                  title="Quarantined Models"
                  value={QUARANTINED_MODELS.length}
                  prefix={<WarningOutlined style={{ color: '#faad14' }} />}
                  valueStyle={{ color: '#faad14' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={8}>
              <Card>
                <Statistic
                  title="Restorable"
                  value={QUARANTINED_MODELS.filter(m => m.canRestore).length}
                  prefix={<RollbackOutlined style={{ color: '#52c41a' }} />}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={8}>
              <Card>
                <Statistic
                  title="Comparisons Run"
                  value={COMPARISON_HISTORY.length}
                  prefix={<CompressOutlined style={{ color: '#1890ff' }} />}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
          </Row>

          <Tabs defaultActiveKey="quarantined">
            <TabPane
              tab={
                <Space>
                  <WarningOutlined />
                  Quarantined Models
                  <Badge count={QUARANTINED_MODELS.length} />
                </Space>
              }
              key="quarantined"
            >
              <Card
                title="Quarantined Models"
                extra={
                  <Button type="primary" icon={<ReloadOutlined />}>
                    Batch Re-evaluate
                  </Button>
                }
              >
                <Table
                  dataSource={QUARANTINED_MODELS}
                  columns={quarantinedColumns}
                  pagination={false}
                />
              </Card>
            </TabPane>

            <TabPane
              tab={
                <Space>
                  <CompressOutlined />
                  Comparison History
                </Space>
              }
              key="comparisons"
            >
              <Card title="V1 vs V3 Comparison History">
                <Table
                  dataSource={COMPARISON_HISTORY}
                  columns={comparisonColumns}
                  pagination={false}
                />
              </Card>
            </TabPane>

            <TabPane
              tab={
                <Space>
                  <HistoryOutlined />
                  Deprecated Features
                </Space>
              }
              key="deprecated"
            >
              <Row gutter={[16, 16]}>
                {LEGACY_FUNCTIONS.map(func => (
                  <Col xs={24} sm={12} key={func.id}>
                    <Card
                      hoverable
                      onClick={() => handleNavigate(func.path)}
                      style={{ borderLeft: '3px solid #8c8c8c' }}
                    >
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                          <div
                            style={{
                              width: 48,
                              height: 48,
                              borderRadius: 12,
                              background: '#8c8c8c15',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: 24,
                              color: '#8c8c8c',
                            }}
                          >
                            {func.icon}
                          </div>
                          <Tag color="default">Deprecated</Tag>
                        </Space>

                        <div>
                          <Text strong style={{ fontSize: 16 }}>{func.name}</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            {func.description}
                          </Text>
                        </div>

                        <Descriptions size="small" column={1}>
                          <Descriptions.Item label="Deprecated">
                            {func.deprecatedDate}
                          </Descriptions.Item>
                          {func.replacedBy && (
                            <Descriptions.Item label="Replaced by">
                              <Tag color="orange">{func.replacedBy}</Tag>
                            </Descriptions.Item>
                          )}
                          {func.lastUsed && (
                            <Descriptions.Item label="Last Used">
                              {func.lastUsed}
                            </Descriptions.Item>
                          )}
                        </Descriptions>

                        {func.canReactivate && (
                          <Button
                            size="small"
                            icon={<RollbackOutlined />}
                          >
                            Request Reactivation
                          </Button>
                        )}
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            </TabPane>

            <TabPane
              tab={
                <Space>
                  <LineChartOutlined />
                  Performance Baseline
                </Space>
              }
              key="baseline"
            >
              <Card title="V3 as Performance Baseline">
                <Alert
                  message="Baseline Comparison"
                  description="V3 models serve as the baseline for measuring V1 experimental improvements. All V1 models must outperform their V3 counterparts to be promoted to V2."
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                />

                <Descriptions bordered>
                  <Descriptions.Item label="Minimum Accuracy Improvement" span={3}>
                    +5% over V3 baseline
                  </Descriptions.Item>
                  <Descriptions.Item label="Maximum Error Rate" span={3}>
                    2% or lower
                  </Descriptions.Item>
                  <Descriptions.Item label="Latency Requirement" span={3}>
                    Within 20% of V3 latency
                  </Descriptions.Item>
                  <Descriptions.Item label="Promotion Criteria" span={3}>
                    Must pass all metrics for 24+ hours
                  </Descriptions.Item>
                </Descriptions>
              </Card>
            </TabPane>
          </Tabs>
        </Content>
      </Layout>
    </AdminOnly>
  );
};

export default V3LegacyHub;
