/**
 * Auto-Fix Dashboard
 * 
 * Comprehensive dashboard for the AI auto-fix cycle:
 * - Cycle status and controls
 * - Vulnerability list
 * - Fix review and approval
 * - History and metrics
 */

import React, { useState } from 'react';
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
  Modal,
  Tabs,
  Timeline,
  Alert,
  Badge,
  Tooltip,
  Spin,
  Popconfirm,
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  SyncOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SafetyOutlined,
  BugOutlined,
  ThunderboltOutlined,
  EyeOutlined,
  RollbackOutlined,
  HistoryOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAutoFix } from '../../hooks/useAutoFix';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

const AutoFixDashboard: React.FC = () => {
  const { t: _t } = useTranslation();
  const {
    status,
    vulnerabilities,
    fixes,
    pendingFixes,
    loading,
    refresh,
    startCycle,
    stopCycle,
    approveFix,
    rejectFix,
    applyFix,
    rollbackFix,
  } = useAutoFix();

  const [selectedFix, setSelectedFix] = useState<any>(null);
  const [showFixModal, setShowFixModal] = useState(false);

  // Severity colors
  const severityColors: Record<string, string> = {
    critical: 'red',
    high: 'orange',
    medium: 'gold',
    low: 'blue',
    info: 'default',
  };

  // Status colors
  const statusColors: Record<string, string> = {
    pending: 'default',
    approved: 'processing',
    applied: 'cyan',
    verified: 'success',
    rejected: 'error',
    rolled_back: 'warning',
  };

  // Vulnerability columns
  const vulnColumns = [
    {
      title: 'ID',
      dataIndex: 'vuln_id',
      key: 'vuln_id',
      width: 100,
      render: (id: string) => <Text code>{id.slice(-6)}</Text>,
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: string) => (
        <Tag color={severityColors[severity]}>{severity.toUpperCase()}</Tag>
      ),
      filters: [
        { text: 'Critical', value: 'critical' },
        { text: 'High', value: 'high' },
        { text: 'Medium', value: 'medium' },
        { text: 'Low', value: 'low' },
      ],
      onFilter: (value: any, record: any) => record.severity === value,
    },
    {
      title: 'File',
      dataIndex: 'file_path',
      key: 'file_path',
      ellipsis: true,
      render: (path: string, record: any) => (
        <Tooltip title={path}>
          <Text code style={{ fontSize: 12 }}>
            {path.split('/').pop()}:{record.line_number}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (confidence: number) => (
        <Progress
          percent={Math.round(confidence * 100)}
          size="small"
          status={confidence >= 0.9 ? 'success' : confidence >= 0.7 ? 'normal' : 'exception'}
        />
      ),
    },
  ];

  // Fix columns
  const fixColumns = [
    {
      title: 'ID',
      dataIndex: 'fix_id',
      key: 'fix_id',
      width: 100,
      render: (id: string) => <Text code>{id.slice(-6)}</Text>,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={statusColors[status]}>{status.toUpperCase()}</Tag>
      ),
    },
    {
      title: 'File',
      dataIndex: 'file_path',
      key: 'file_path',
      ellipsis: true,
      render: (path: string) => (
        <Tooltip title={path}>
          <Text code style={{ fontSize: 12 }}>
            {path.split('/').pop()}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (confidence: number) => (
        <Progress
          percent={Math.round(confidence * 100)}
          size="small"
          status={confidence >= 0.9 ? 'success' : 'normal'}
        />
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 200,
      render: (_: any, record: any) => (
        <Space size="small">
          <Tooltip title="View Details">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedFix(record);
                setShowFixModal(true);
              }}
            />
          </Tooltip>
          {record.status === 'pending' && (
            <>
              <Tooltip title="Approve">
                <Button
                  type="text"
                  icon={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
                  onClick={() => approveFix(record.fix_id)}
                />
              </Tooltip>
              <Tooltip title="Reject">
                <Button
                  type="text"
                  icon={<CloseCircleOutlined style={{ color: '#ff4d4f' }} />}
                  onClick={() => rejectFix(record.fix_id)}
                />
              </Tooltip>
            </>
          )}
          {record.status === 'approved' && (
            <Tooltip title="Apply Fix">
              <Button
                type="text"
                icon={<ThunderboltOutlined style={{ color: '#1890ff' }} />}
                onClick={() => applyFix(record.fix_id)}
              />
            </Tooltip>
          )}
          {(record.status === 'applied' || record.status === 'verified') && (
            <Popconfirm
              title="Rollback this fix?"
              onConfirm={() => rollbackFix(record.fix_id)}
            >
              <Button
                type="text"
                danger
                icon={<RollbackOutlined />}
              />
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ];

  if (loading && !status) {
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
            AI Auto-Fix Dashboard
          </Title>
        </Col>
        <Col>
          <Space>
            <Badge
              status={status?.running ? 'processing' : 'default'}
              text={status?.running ? 'Running' : 'Stopped'}
            />
            {status?.running ? (
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
            <Button icon={<SyncOutlined />} onClick={refresh}>
              Refresh
            </Button>
          </Space>
        </Col>
      </Row>

      {/* Metrics */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Cycles Completed"
              value={status?.metrics.cycles_completed || 0}
              prefix={<SyncOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Vulnerabilities Detected"
              value={status?.metrics.vulnerabilities_detected || 0}
              prefix={<BugOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Fixes Applied"
              value={status?.metrics.fixes_applied || 0}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Success Rate"
              value={
                status?.metrics.fixes_applied
                  ? Math.round(
                      (status.metrics.fixes_verified /
                        status.metrics.fixes_applied) *
                        100
                    )
                  : 0
              }
              suffix="%"
              prefix={<SafetyOutlined />}
              valueStyle={{
                color:
                  status?.metrics.fixes_applied &&
                  status.metrics.fixes_verified / status.metrics.fixes_applied > 0.8
                    ? '#3f8600'
                    : '#faad14',
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Pending Fixes Alert */}
      {pendingFixes.length > 0 && (
        <Alert
          message={`${pendingFixes.length} fixes pending approval`}
          description="Review and approve fixes before they can be applied."
          type="warning"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {/* Tabs */}
      <Card>
        <Tabs defaultActiveKey="vulnerabilities">
          <TabPane
            tab={
              <span>
                <BugOutlined />
                Vulnerabilities ({vulnerabilities.length})
              </span>
            }
            key="vulnerabilities"
          >
            <Table
              columns={vulnColumns}
              dataSource={vulnerabilities}
              rowKey="vuln_id"
              size="small"
              pagination={{ pageSize: 10 }}
            />
          </TabPane>
          <TabPane
            tab={
              <span>
                <ThunderboltOutlined />
                Fixes ({fixes.length})
              </span>
            }
            key="fixes"
          >
            <Table
              columns={fixColumns}
              dataSource={fixes}
              rowKey="fix_id"
              size="small"
              pagination={{ pageSize: 10 }}
            />
          </TabPane>
          <TabPane
            tab={
              <span>
                <HistoryOutlined />
                History
              </span>
            }
            key="history"
          >
            <Timeline>
              <Timeline.Item color="green">
                <Text strong>Fix verified</Text>
                <br />
                <Text type="secondary">auth.py - Hardcoded secret removed</Text>
                <br />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {new Date().toLocaleString()}
                </Text>
              </Timeline.Item>
              <Timeline.Item color="blue">
                <Text strong>Fix applied</Text>
                <br />
                <Text type="secondary">reliability.py - Deprecated API updated</Text>
                <br />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {new Date(Date.now() - 3600000).toLocaleString()}
                </Text>
              </Timeline.Item>
              <Timeline.Item color="orange">
                <Text strong>Vulnerability detected</Text>
                <br />
                <Text type="secondary">JWT decode without validation</Text>
                <br />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {new Date(Date.now() - 7200000).toLocaleString()}
                </Text>
              </Timeline.Item>
              <Timeline.Item color="gray">
                <Text strong>Cycle completed</Text>
                <br />
                <Text type="secondary">Cycle #15 finished successfully</Text>
                <br />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {new Date(Date.now() - 10800000).toLocaleString()}
                </Text>
              </Timeline.Item>
            </Timeline>
          </TabPane>
        </Tabs>
      </Card>

      {/* Fix Detail Modal */}
      <Modal
        title="Fix Details"
        open={showFixModal}
        onCancel={() => setShowFixModal(false)}
        width={800}
        footer={null}
      >
        {selectedFix && (
          <div>
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={12}>
                <Text type="secondary">Fix ID:</Text>
                <br />
                <Text code>{selectedFix.fix_id}</Text>
              </Col>
              <Col span={12}>
                <Text type="secondary">Status:</Text>
                <br />
                <Tag color={statusColors[selectedFix.status]}>
                  {selectedFix.status.toUpperCase()}
                </Tag>
              </Col>
            </Row>
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={24}>
                <Text type="secondary">File:</Text>
                <br />
                <Text code>{selectedFix.file_path}</Text>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col span={12}>
                <Text type="secondary">Original Code:</Text>
                <pre
                  style={{
                    background: '#1a1a2e',
                    color: '#ff6b6b',
                    padding: 12,
                    borderRadius: 4,
                    overflow: 'auto',
                  }}
                >
                  {selectedFix.original_code || 'N/A'}
                </pre>
              </Col>
              <Col span={12}>
                <Text type="secondary">Fixed Code:</Text>
                <pre
                  style={{
                    background: '#1a1a2e',
                    color: '#00ff88',
                    padding: 12,
                    borderRadius: 4,
                    overflow: 'auto',
                  }}
                >
                  {selectedFix.fixed_code || 'N/A'}
                </pre>
              </Col>
            </Row>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default AutoFixDashboard;
