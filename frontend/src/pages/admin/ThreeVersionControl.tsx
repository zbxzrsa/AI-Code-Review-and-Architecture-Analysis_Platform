/**
 * 三版本演化控制面板 (Three-Version Evolution Control Panel)
 * 
 * 功能描述:
 *   管理三版本自演化循环的管理员界面。
 * 
 * 版本说明:
 *   - V1（新版）: 实验，试错
 *   - V2（稳定版）: 生产，修复 V1 错误
 *   - V3（旧版）: 隔离，比较，排除
 * 
 * 主要特性:
 *   - 版本状态监控
 *   - 升级/降级操作
 *   - 实验管理
 *   - 性能对比
 * 
 * 最后修改日期: 2024-12-07
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Tag,
  Spin,
  Alert,
  Tabs,
  Table,
  Modal,
  Form,
  Input,
  Select,
  Progress,
  Typography,
  Space,
  Statistic,
  Tooltip,
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  ExperimentOutlined,
  SafetyCertificateOutlined,
  InboxOutlined,
  BugOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import type { VersionAIStatus, FeedbackStats } from '../../services/threeVersionService';

const { Title, Text } = Typography;
const { TextArea } = Input;

// API base URL
const API_BASE = '/api/v1/evolution';

interface CycleStatus {
  running: boolean;
  current_cycle?: {
    cycle_id: string;
    phase: string;
    experiments_run: number;
    errors_fixed: number;
    promotions_made: number;
    degradations_made: number;
  };
  pending?: {
    promotions: number;
    degradations: number;
    reevaluations: number;
  };
}

/** Error report form values */
interface ErrorFormValues {
  error_type: string;
  description: string;
  stack_trace?: string;
}

/** Promotion form values */
interface PromotionFormValues {
  version_id: string;
  reason: string;
}

/** Degradation form values */
interface DegradationFormValues {
  version_id: string;
  reason: string;
}

/** Quarantine insight */
interface QuarantineInsight {
  category: string;
  insight: string;
  failure_count: number;
  avg_accuracy: number;
}

/** Quarantine stats */
interface QuarantineStats {
  total_quarantined?: number;
  recovery_rate?: number;
  permanent_exclusions?: number;
  temporary_exclusions?: number;
  insights?: QuarantineInsight[];
  statistics?: {
    total_quarantined: number;
    permanent_exclusions: number;
    temporary_exclusions: number;
  };
}

/** Metric row for table */
interface MetricRow {
  key: string;
  metric: string;
  value: number;
  color?: string;
}

const ThreeVersionControl: React.FC = () => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('v1');
  
  // State
  const [cycleStatus, setCycleStatus] = useState<CycleStatus | null>(null);
  const [aiStatus, setAiStatus] = useState<Record<string, VersionAIStatus>>({});
  const [quarantineStats, setQuarantineStats] = useState<QuarantineStats | null>(null);
  const [feedbackStats, setFeedbackStats] = useState<FeedbackStats | null>(null);
  
  // Dialog states
  const [errorModalOpen, setErrorModalOpen] = useState(false);
  const [promoteModalOpen, setPromoteModalOpen] = useState(false);
  const [degradeModalOpen, setDegradeModalOpen] = useState(false);
  
  // Forms
  const [errorForm] = Form.useForm();
  const [promoteForm] = Form.useForm();
  const [degradeForm] = Form.useForm();

  // Fetch status
  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true);
      const [statusRes, aiRes, quarantineRes] = await Promise.all([
        fetch(`${API_BASE}/status`),
        fetch(`${API_BASE}/ai/status`),
        fetch(`${API_BASE}/v3/quarantine`),
      ]);
      
      if (statusRes.ok) {
        const data = await statusRes.json();
        setCycleStatus(data.spiral_status);
        setFeedbackStats(data.spiral_status?.feedback_stats);
      }
      if (aiRes.ok) {
        setAiStatus(await aiRes.json());
      }
      if (quarantineRes.ok) {
        setQuarantineStats(await quarantineRes.json());
      }
      
      setError(null);
    } catch (_err) {
      setError('Failed to fetch status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Actions
  const startCycle = async () => {
    try {
      await fetch(`${API_BASE}/start`, { method: 'POST' });
      fetchStatus();
    } catch (_err) {
      setError('Failed to start cycle');
    }
  };

  const stopCycle = async () => {
    try {
      await fetch(`${API_BASE}/stop`, { method: 'POST' });
      fetchStatus();
    } catch (_err) {
      setError('Failed to stop cycle');
    }
  };

  const reportError = async (values: ErrorFormValues) => {
    try {
      await fetch(`${API_BASE}/v1/errors`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      });
      setErrorModalOpen(false);
      errorForm.resetFields();
      fetchStatus();
    } catch (_err) {
      setError('Failed to report error');
    }
  };

  const triggerPromotion = async (values: PromotionFormValues) => {
    try {
      await fetch(`${API_BASE}/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      });
      setPromoteModalOpen(false);
      promoteForm.resetFields();
      fetchStatus();
    } catch (_err) {
      setError('Failed to trigger promotion');
    }
  };

  const triggerDegradation = async (values: DegradationFormValues) => {
    try {
      await fetch(`${API_BASE}/degrade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      });
      setDegradeModalOpen(false);
      degradeForm.resetFields();
      fetchStatus();
    } catch (_err) {
      setError('Failed to trigger degradation');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'degraded': return 'warning';
      case 'offline': return 'error';
      default: return 'default';
    }
  };

  if (loading && !cycleStatus) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <Spin size="large" />
      </div>
    );
  }

  const tabItems = [
    {
      key: 'v1',
      label: <><ExperimentOutlined /> V1 Experimental</>,
      children: (
        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <Card title="V1 AI Status">
              <Table
                size="small"
                pagination={false}
                dataSource={[
                  {
                    key: 'vc',
                    name: 'VC-AI (Admin)',
                    status: aiStatus?.v1?.vc_ai?.status || 'unknown',
                  },
                  {
                    key: 'cr',
                    name: 'CR-AI (Testing)',
                    status: aiStatus?.v1?.cr_ai?.status || 'unknown',
                  },
                ]}
                columns={[
                  { title: 'Component', dataIndex: 'name', key: 'name' },
                  {
                    title: 'Status',
                    dataIndex: 'status',
                    key: 'status',
                    render: (status: string) => (
                      <Tag color={getStatusColor(status)}>{status}</Tag>
                    ),
                  },
                ]}
              />
            </Card>
          </Col>
          <Col xs={24} md={12}>
            <Card title="Actions">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Button
                  icon={<BugOutlined />}
                  onClick={() => setErrorModalOpen(true)}
                  block
                >
                  Report Error
                </Button>
                <Button
                  type="primary"
                  icon={<ArrowUpOutlined />}
                  onClick={() => setPromoteModalOpen(true)}
                  block
                >
                  Promote to V2
                </Button>
              </Space>
            </Card>
          </Col>
        </Row>
      ),
    },
    {
      key: 'v2',
      label: <><SafetyCertificateOutlined /> V2 Production</>,
      children: (
        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <Card title="V2 AI Status (User-Facing)">
              <Table
                size="small"
                pagination={false}
                dataSource={[
                  {
                    key: 'vc',
                    name: 'VC-AI (Admin)',
                    status: aiStatus?.v2?.vc_ai?.status || 'unknown',
                  },
                  {
                    key: 'cr',
                    name: 'CR-AI (Users)',
                    status: aiStatus?.v2?.cr_ai?.status || 'unknown',
                  },
                ]}
                columns={[
                  { title: 'Component', dataIndex: 'name', key: 'name' },
                  {
                    title: 'Status',
                    dataIndex: 'status',
                    key: 'status',
                    render: (status: string) => (
                      <Tag color={getStatusColor(status)}>{status}</Tag>
                    ),
                  },
                ]}
              />
            </Card>
          </Col>
          <Col xs={24} md={12}>
            <Card title="V1 Error Fixes">
              {feedbackStats ? (
                <>
                  <Text>Total Errors: {feedbackStats.total_errors_reported}</Text>
                  <br />
                  <Text>Fixes Generated: {feedbackStats.total_fixes_generated}</Text>
                  <br />
                  <Text>Success Rate:</Text>
                  <Progress
                    percent={Math.round(feedbackStats.fix_success_rate * 100)}
                    status="active"
                  />
                </>
              ) : (
                <Text type="secondary">No feedback stats available</Text>
              )}
              <Button
                danger
                icon={<ArrowDownOutlined />}
                onClick={() => setDegradeModalOpen(true)}
                style={{ marginTop: 16 }}
                block
              >
                Degrade to V3
              </Button>
            </Card>
          </Col>
        </Row>
      ),
    },
    {
      key: 'v3',
      label: <><InboxOutlined /> V3 Quarantine</>,
      children: (
        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <Card title="Quarantine Statistics">
              {quarantineStats?.statistics ? (
                <Table
                  size="small"
                  pagination={false}
                  dataSource={[
                    {
                      key: 'total',
                      metric: 'Total Quarantined',
                      value: quarantineStats.statistics.total_quarantined,
                    },
                    {
                      key: 'permanent',
                      metric: 'Permanent Exclusions',
                      value: quarantineStats.statistics.permanent_exclusions,
                      color: 'error',
                    },
                    {
                      key: 'temporary',
                      metric: 'Temporary Exclusions',
                      value: quarantineStats.statistics.temporary_exclusions,
                      color: 'warning',
                    },
                  ]}
                  columns={[
                    { title: 'Metric', dataIndex: 'metric', key: 'metric' },
                    {
                      title: 'Value',
                      dataIndex: 'value',
                      key: 'value',
                      render: (value: number, record: MetricRow) =>
                        record.color ? (
                          <Tag color={record.color}>{value}</Tag>
                        ) : (
                          value
                        ),
                    },
                  ]}
                />
              ) : (
                <Text type="secondary">No quarantine stats available</Text>
              )}
            </Card>
          </Col>
          <Col xs={24} md={12}>
            <Card title="Failure Insights">
              {quarantineStats?.insights?.map((insight: QuarantineInsight, index: number) => (
                <Alert
                  key={index}
                  type="info"
                  message={<strong>{insight.category}</strong>}
                  description={
                    <>
                      {insight.insight}
                      <br />
                      <small>
                        Failures: {insight.failure_count}, Avg Accuracy:{' '}
                        {(insight.avg_accuracy * 100).toFixed(1)}%
                      </small>
                    </>
                  }
                  style={{ marginBottom: 8 }}
                />
              )) || <Text type="secondary">No insights available</Text>}
            </Card>
          </Col>
        </Row>
      ),
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}>
        Three-Version Evolution Control
      </Title>
      <Text type="secondary">
        Manage the V1→V2→V3 spiral self-evolution cycle
      </Text>

      {error && (
        <Alert
          type="error"
          message={error}
          closable
          onClose={() => setError(null)}
          style={{ marginTop: 16, marginBottom: 16 }}
        />
      )}

      {/* Cycle Controls */}
      <Card style={{ marginTop: 16, marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col xs={24} md={12}>
            <Space>
              <Title level={4} style={{ margin: 0 }}>
                Evolution Cycle
              </Title>
              <Tag color={cycleStatus?.running ? 'success' : 'default'}>
                {cycleStatus?.running ? 'Running' : 'Stopped'}
              </Tag>
            </Space>
            {cycleStatus?.current_cycle && (
              <div style={{ marginTop: 8 }}>
                <Text type="secondary">
                  Phase: {cycleStatus.current_cycle.phase}
                </Text>
              </div>
            )}
          </Col>
          <Col xs={24} md={12} style={{ textAlign: 'right' }}>
            <Space>
              {cycleStatus?.running ? (
                <Button
                  type="primary"
                  danger
                  icon={<PauseCircleOutlined />}
                  onClick={stopCycle}
                >
                  Stop Cycle
                </Button>
              ) : (
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={startCycle}
                >
                  Start Cycle
                </Button>
              )}
              <Tooltip title="Refresh">
                <Button icon={<ReloadOutlined />} onClick={fetchStatus} />
              </Tooltip>
            </Space>
          </Col>
        </Row>

        {/* Cycle Metrics */}
        {cycleStatus?.current_cycle && (
          <Row gutter={16} style={{ marginTop: 16 }}>
            <Col xs={12} sm={6}>
              <Statistic
                title="Experiments"
                value={cycleStatus.current_cycle.experiments_run}
                prefix={<ExperimentOutlined />}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title="Errors Fixed"
                value={cycleStatus.current_cycle.errors_fixed}
                prefix={<CheckCircleOutlined />}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title="Promotions"
                value={cycleStatus.current_cycle.promotions_made}
                prefix={<ArrowUpOutlined />}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title="Degradations"
                value={cycleStatus.current_cycle.degradations_made}
                prefix={<ArrowDownOutlined />}
              />
            </Col>
          </Row>
        )}
      </Card>

      {/* Tabs */}
      <Tabs activeKey={activeTab} onChange={setActiveTab} items={tabItems} />

      {/* Error Report Modal */}
      <Modal
        title="Report V1 Error"
        open={errorModalOpen}
        onCancel={() => setErrorModalOpen(false)}
        footer={null}
      >
        <Form form={errorForm} onFinish={reportError} layout="vertical">
          <Form.Item
            name="tech_id"
            label="Technology ID"
            rules={[{ required: true }]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="tech_name"
            label="Technology Name"
            rules={[{ required: true }]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="error_type"
            label="Error Type"
            initialValue="compatibility"
          >
            <Select>
              <Select.Option value="compatibility">Compatibility</Select.Option>
              <Select.Option value="performance">Performance</Select.Option>
              <Select.Option value="security">Security</Select.Option>
              <Select.Option value="accuracy">Accuracy</Select.Option>
              <Select.Option value="stability">Stability</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="description" label="Description">
            <TextArea rows={3} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button onClick={() => setErrorModalOpen(false)}>Cancel</Button>
              <Button type="primary" htmlType="submit">
                Report Error
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Promotion Modal */}
      <Modal
        title="Promote Technology (V1 → V2)"
        open={promoteModalOpen}
        onCancel={() => setPromoteModalOpen(false)}
        footer={null}
      >
        <Form form={promoteForm} onFinish={triggerPromotion} layout="vertical">
          <Form.Item
            name="tech_id"
            label="Technology ID"
            rules={[{ required: true }]}
          >
            <Input />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button onClick={() => setPromoteModalOpen(false)}>Cancel</Button>
              <Button type="primary" htmlType="submit">
                Promote
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Degradation Modal */}
      <Modal
        title="Degrade Technology (V2 → V3)"
        open={degradeModalOpen}
        onCancel={() => setDegradeModalOpen(false)}
        footer={null}
      >
        <Form form={degradeForm} onFinish={triggerDegradation} layout="vertical">
          <Form.Item
            name="tech_id"
            label="Technology ID"
            rules={[{ required: true }]}
          >
            <Input />
          </Form.Item>
          <Form.Item name="reason" label="Reason">
            <Input />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button onClick={() => setDegradeModalOpen(false)}>Cancel</Button>
              <Button type="primary" danger htmlType="submit">
                Degrade
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ThreeVersionControl;
