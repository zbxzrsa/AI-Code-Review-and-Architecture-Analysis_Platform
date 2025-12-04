/**
 * Admin Version Comparison View
 * 
 * Displays side-by-side comparison of V1/V2/V3 outputs for the same request.
 * Only accessible to administrators.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Tag,
  Button,
  Select,
  Space,
  Spin,
  Alert,
  Tabs,
  Table,
  Tooltip,
  Badge,
  Divider,
  Modal,
  Input,
  message,
  Statistic,
  Progress,
} from 'antd';
import {
  SyncOutlined,
  RollbackOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  ExperimentOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
  DollarOutlined,
  HistoryOutlined,
  DiffOutlined,
} from '@ant-design/icons';
import { DiffViewer } from '../../components/code/DiffViewer';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;
const { TextArea } = Input;

interface VersionOutput {
  version: 'v1' | 'v2' | 'v3';
  versionId: string;
  modelVersion: string;
  promptVersion: string;
  timestamp: string;
  latencyMs: number;
  cost: number;
  issues: Issue[];
  rawOutput: string;
  confidence: number;
  securityPassed: boolean;
}

interface Issue {
  id: string;
  type: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  message: string;
  file: string;
  line: number;
  suggestion?: string;
}

interface ComparisonRequest {
  requestId: string;
  code: string;
  language: string;
  timestamp: string;
  v1Output?: VersionOutput;
  v2Output?: VersionOutput;
  v3Output?: VersionOutput;
}

interface RollbackReason {
  id: string;
  label: string;
}

const ROLLBACK_REASONS: RollbackReason[] = [
  { id: 'accuracy_regression', label: 'Accuracy Regression' },
  { id: 'latency_increase', label: 'Latency Increase' },
  { id: 'security_failure', label: 'Security Check Failure' },
  { id: 'cost_overrun', label: 'Cost Budget Exceeded' },
  { id: 'user_feedback', label: 'Negative User Feedback' },
  { id: 'other', label: 'Other (specify)' },
];

const VersionComparison: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [requests, setRequests] = useState<ComparisonRequest[]>([]);
  const [selectedRequest, setSelectedRequest] = useState<ComparisonRequest | null>(null);
  const [rollbackModalVisible, setRollbackModalVisible] = useState(false);
  const [rollbackReason, setRollbackReason] = useState<string>('');
  const [rollbackNotes, setRollbackNotes] = useState('');
  const [rollbackVersion, setRollbackVersion] = useState<string>('');

  // Fetch recent comparison requests
  const fetchRequests = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/admin/comparison-requests?limit=50');
      const data = await response.json();
      setRequests(data.requests);
      if (data.requests.length > 0 && !selectedRequest) {
        setSelectedRequest(data.requests[0]);
      }
    } catch (error) {
      message.error('Failed to fetch comparison requests');
    } finally {
      setLoading(false);
    }
  }, [selectedRequest]);

  useEffect(() => {
    fetchRequests();
    // Poll for updates
    const interval = setInterval(fetchRequests, 30000);
    return () => clearInterval(interval);
  }, [fetchRequests]);

  // Handle rollback
  const handleRollback = async () => {
    if (!rollbackReason) {
      message.error('Please select a rollback reason');
      return;
    }

    try {
      await fetch('/api/admin/rollback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          versionId: rollbackVersion,
          reason: rollbackReason,
          notes: rollbackNotes,
        }),
      });
      message.success('Rollback initiated successfully');
      setRollbackModalVisible(false);
    } catch (error) {
      message.error('Rollback failed');
    }
  };

  // Render version card
  const renderVersionCard = (output: VersionOutput | undefined, label: string, color: string) => {
    if (!output) {
      return (
        <Card
          title={<span style={{ color }}>{label}</span>}
          style={{ height: '100%', opacity: 0.5 }}
        >
          <Text type="secondary">No output available</Text>
        </Card>
      );
    }

    const severityCounts = output.issues.reduce(
      (acc, issue) => {
        acc[issue.severity] = (acc[issue.severity] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );

    return (
      <Card
        title={
          <Space>
            <span style={{ color }}>{label}</span>
            <Tag color={output.securityPassed ? 'green' : 'red'}>
              {output.securityPassed ? (
                <><SafetyCertificateOutlined /> Secure</>
              ) : (
                <><WarningOutlined /> Security Failed</>
              )}
            </Tag>
          </Space>
        }
        extra={
          label === 'V1 Experiment' && (
            <Button
              danger
              size="small"
              icon={<RollbackOutlined />}
              onClick={() => {
                setRollbackVersion(output.versionId);
                setRollbackModalVisible(true);
              }}
            >
              Rollback
            </Button>
          )
        }
        style={{ height: '100%' }}
      >
        {/* Metrics */}
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Statistic
              title="Latency"
              value={output.latencyMs}
              suffix="ms"
              valueStyle={{
                color: output.latencyMs > 3000 ? '#cf1322' : '#3f8600',
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Cost"
              value={output.cost}
              prefix="$"
              precision={4}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Confidence"
              value={output.confidence * 100}
              suffix="%"
              precision={1}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Issues"
              value={output.issues.length}
            />
          </Col>
        </Row>

        {/* Model/Prompt Info */}
        <Space direction="vertical" size="small" style={{ width: '100%', marginBottom: 16 }}>
          <Text type="secondary">
            Model: <Tag>{output.modelVersion}</Tag>
          </Text>
          <Text type="secondary">
            Prompt: <Tag>{output.promptVersion}</Tag>
          </Text>
        </Space>

        {/* Severity Breakdown */}
        <div style={{ marginBottom: 16 }}>
          <Space>
            <Badge count={severityCounts.critical || 0} style={{ backgroundColor: '#ff4d4f' }}>
              <Tag color="red">Critical</Tag>
            </Badge>
            <Badge count={severityCounts.high || 0} style={{ backgroundColor: '#fa8c16' }}>
              <Tag color="orange">High</Tag>
            </Badge>
            <Badge count={severityCounts.medium || 0} style={{ backgroundColor: '#faad14' }}>
              <Tag color="gold">Medium</Tag>
            </Badge>
            <Badge count={severityCounts.low || 0} style={{ backgroundColor: '#52c41a' }}>
              <Tag color="green">Low</Tag>
            </Badge>
          </Space>
        </div>

        {/* Issues List */}
        <div style={{ maxHeight: 300, overflow: 'auto' }}>
          {output.issues.map((issue, index) => (
            <Alert
              key={issue.id}
              type={
                issue.severity === 'critical' ? 'error' :
                issue.severity === 'high' ? 'warning' :
                issue.severity === 'medium' ? 'info' : 'success'
              }
              message={
                <Space>
                  <Tag>{issue.type}</Tag>
                  <Text strong>{issue.file}:{issue.line}</Text>
                </Space>
              }
              description={issue.message}
              style={{ marginBottom: 8 }}
              showIcon
            />
          ))}
        </div>
      </Card>
    );
  };

  // Render diff view
  const renderDiffView = () => {
    if (!selectedRequest?.v1Output || !selectedRequest?.v2Output) {
      return <Text type="secondary">Select a request with both V1 and V2 outputs to view diff</Text>;
    }

    return (
      <DiffViewer
        oldCode={selectedRequest.v2Output.rawOutput}
        newCode={selectedRequest.v1Output.rawOutput}
        oldFileName="V2 Baseline"
        newFileName="V1 Experiment"
        outputFormat="side-by-side"
      />
    );
  };

  // Render executability test results
  const renderExecutabilityTests = () => {
    const testResults = [
      { name: 'Syntax Validity', v1: true, v2: true, v3: true },
      { name: 'Type Safety', v1: true, v2: true, v3: false },
      { name: 'Suggestion Applicability', v1: true, v2: true, v3: true },
      { name: 'No Breaking Changes', v1: false, v2: true, v3: true },
      { name: 'Test Coverage', v1: true, v2: true, v3: false },
    ];

    const columns = [
      { title: 'Test', dataIndex: 'name', key: 'name' },
      {
        title: 'V1',
        dataIndex: 'v1',
        key: 'v1',
        render: (passed: boolean) => passed ? (
          <CheckCircleOutlined style={{ color: '#52c41a' }} />
        ) : (
          <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
        ),
      },
      {
        title: 'V2',
        dataIndex: 'v2',
        key: 'v2',
        render: (passed: boolean) => passed ? (
          <CheckCircleOutlined style={{ color: '#52c41a' }} />
        ) : (
          <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
        ),
      },
      {
        title: 'V3',
        dataIndex: 'v3',
        key: 'v3',
        render: (passed: boolean) => passed ? (
          <CheckCircleOutlined style={{ color: '#52c41a' }} />
        ) : (
          <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
        ),
      },
    ];

    return <Table dataSource={testResults} columns={columns} pagination={false} size="small" />;
  };

  return (
    <div style={{ padding: 24 }}>
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={2}>
            <DiffOutlined /> Version Comparison (Admin Only)
          </Title>
          <Text type="secondary">
            Compare outputs from V1 Experiment, V2 Production, and V3 Legacy
          </Text>
        </Col>
        <Col>
          <Space>
            <Select
              style={{ width: 400 }}
              placeholder="Select a request to compare"
              value={selectedRequest?.requestId}
              onChange={(value) => {
                const request = requests.find((r) => r.requestId === value);
                setSelectedRequest(request || null);
              }}
              loading={loading}
            >
              {requests.map((r) => (
                <Option key={r.requestId} value={r.requestId}>
                  {r.requestId.substring(0, 8)} - {r.language} - {new Date(r.timestamp).toLocaleTimeString()}
                </Option>
              ))}
            </Select>
            <Button icon={<SyncOutlined />} onClick={fetchRequests} loading={loading}>
              Refresh
            </Button>
          </Space>
        </Col>
      </Row>

      {loading && !selectedRequest ? (
        <div style={{ textAlign: 'center', padding: 100 }}>
          <Spin size="large" />
        </div>
      ) : selectedRequest ? (
        <Tabs defaultActiveKey="comparison">
          <TabPane
            tab={<span><ExperimentOutlined /> Side-by-Side Comparison</span>}
            key="comparison"
          >
            <Row gutter={16}>
              <Col span={8}>
                {renderVersionCard(selectedRequest.v1Output, 'V1 Experiment', '#1890ff')}
              </Col>
              <Col span={8}>
                {renderVersionCard(selectedRequest.v2Output, 'V2 Production', '#52c41a')}
              </Col>
              <Col span={8}>
                {renderVersionCard(selectedRequest.v3Output, 'V3 Legacy', '#faad14')}
              </Col>
            </Row>
          </TabPane>

          <TabPane
            tab={<span><DiffOutlined /> Output Diff</span>}
            key="diff"
          >
            <Card>
              {renderDiffView()}
            </Card>
          </TabPane>

          <TabPane
            tab={<span><ThunderboltOutlined /> Executability Tests</span>}
            key="executability"
          >
            <Card title="Suggestion Executability Tests">
              {renderExecutabilityTests()}
            </Card>
          </TabPane>

          <TabPane
            tab={<span><SafetyCertificateOutlined /> Evidence Chain</span>}
            key="evidence"
          >
            <Card title="Evidence Chain for V1 Output">
              {selectedRequest.v1Output ? (
                <div>
                  <Paragraph>
                    <Text strong>Triggering Rules:</Text>
                    <br />
                    <Space wrap style={{ marginTop: 8 }}>
                      <Tag color="blue">CWE-79 (XSS)</Tag>
                      <Tag color="blue">OWASP-A03</Tag>
                      <Tag color="blue">ESLint/security</Tag>
                    </Space>
                  </Paragraph>

                  <Divider />

                  <Paragraph>
                    <Text strong>Code Locations:</Text>
                    <br />
                    <ul>
                      <li><code>src/api/users.ts:42:15</code> - User input not sanitized</li>
                      <li><code>src/utils/render.ts:88:3</code> - Dangerous innerHTML usage</li>
                    </ul>
                  </Paragraph>

                  <Divider />

                  <Paragraph>
                    <Text strong>Static Analysis References:</Text>
                    <br />
                    <ul>
                      <li>Semgrep: <code>javascript.browser.security.insecure-document-method</code></li>
                      <li>ESLint: <code>@typescript-eslint/no-unsafe-assignment</code></li>
                    </ul>
                  </Paragraph>

                  <Divider />

                  <Row gutter={16}>
                    <Col span={8}>
                      <Statistic
                        title="Model Confidence"
                        value={94.2}
                        suffix="%"
                        prefix={<CheckCircleOutlined />}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="Model Version"
                        value="gpt-4o-2024-05-13"
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="Prompt Version"
                        value="security-audit-v2"
                      />
                    </Col>
                  </Row>
                </div>
              ) : (
                <Text type="secondary">No V1 output available</Text>
              )}
            </Card>
          </TabPane>

          <TabPane
            tab={<span><HistoryOutlined /> History</span>}
            key="history"
          >
            <Card title="Request History">
              <Table
                dataSource={requests.slice(0, 20)}
                columns={[
                  { title: 'Request ID', dataIndex: 'requestId', key: 'requestId', render: (id) => id.substring(0, 8) },
                  { title: 'Language', dataIndex: 'language', key: 'language' },
                  { title: 'Timestamp', dataIndex: 'timestamp', key: 'timestamp', render: (ts) => new Date(ts).toLocaleString() },
                  {
                    title: 'V1',
                    key: 'v1',
                    render: (_, record) => record.v1Output ? (
                      <Tag color="blue">{record.v1Output.issues.length} issues</Tag>
                    ) : <Tag>N/A</Tag>,
                  },
                  {
                    title: 'V2',
                    key: 'v2',
                    render: (_, record) => record.v2Output ? (
                      <Tag color="green">{record.v2Output.issues.length} issues</Tag>
                    ) : <Tag>N/A</Tag>,
                  },
                  {
                    title: 'Actions',
                    key: 'actions',
                    render: (_, record) => (
                      <Button size="small" onClick={() => setSelectedRequest(record)}>
                        View
                      </Button>
                    ),
                  },
                ]}
                size="small"
              />
            </Card>
          </TabPane>
        </Tabs>
      ) : (
        <Alert
          type="info"
          message="No comparison requests available"
          description="Wait for shadow traffic to generate comparison data."
        />
      )}

      {/* Rollback Modal */}
      <Modal
        title={<><RollbackOutlined /> Initiate Rollback</>}
        open={rollbackModalVisible}
        onOk={handleRollback}
        onCancel={() => setRollbackModalVisible(false)}
        okText="Confirm Rollback"
        okButtonProps={{ danger: true }}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Alert
            type="warning"
            message="This will abort the current rollout and revert to the stable version."
          />

          <div>
            <Text strong>Version to rollback:</Text>
            <br />
            <Tag color="red">{rollbackVersion}</Tag>
          </div>

          <div>
            <Text strong>Reason for rollback:</Text>
            <Select
              style={{ width: '100%', marginTop: 8 }}
              placeholder="Select a reason"
              value={rollbackReason}
              onChange={setRollbackReason}
            >
              {ROLLBACK_REASONS.map((reason) => (
                <Option key={reason.id} value={reason.id}>
                  {reason.label}
                </Option>
              ))}
            </Select>
          </div>

          <div>
            <Text strong>Additional notes:</Text>
            <TextArea
              rows={4}
              placeholder="Describe the issue in detail..."
              value={rollbackNotes}
              onChange={(e) => setRollbackNotes(e.target.value)}
              style={{ marginTop: 8 }}
            />
          </div>
        </Space>
      </Modal>
    </div>
  );
};

export default VersionComparison;
