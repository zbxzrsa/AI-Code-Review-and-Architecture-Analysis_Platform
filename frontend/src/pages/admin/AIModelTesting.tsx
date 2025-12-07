/**
 * AI Model Testing Page
 *
 * Admin interface for testing AI models and technologies
 * from the three-version evolution cycle (V1/V2/V3).
 */

import React, { useState, useEffect } from 'react';
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
  Typography,
  Input,
  Spin,
  Alert,
  Form,
  Slider,
  Divider,
  message,
} from 'antd';
import {
  ExperimentOutlined,
  RocketOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  CodeOutlined,
  ThunderboltOutlined,
  LineChartOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { useTranslation } from 'react-i18next';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { TextArea } = Input;

// Types
interface AIModel {
  model_id: string;
  name: string;
  version: 'v1' | 'v2' | 'v3';
  type: string;
  status: 'active' | 'testing' | 'deprecated';
  accuracy: number;
  latency_ms: number;
  cost_per_1k: number;
  requests_today: number;
  last_used: string;
}

interface TestResult {
  test_id: string;
  model_id: string;
  input: string;
  output: string;
  latency_ms: number;
  tokens_used: number;
  cost: number;
  timestamp: string;
  success: boolean;
}

interface TestConfig {
  temperature: number;
  max_tokens: number;
  top_p: number;
  stream: boolean;
}

// Version colors
const versionColors: Record<string, string> = {
  v1: 'orange',
  v2: 'green',
  v3: 'red',
};

const statusColors: Record<string, string> = {
  active: 'green',
  testing: 'blue',
  deprecated: 'default',
};

const AIModelTesting: React.FC = () => {
  // Translation available: const { t } = useTranslation();
  useTranslation();  // For future i18n
  const [_form] = Form.useForm();  // Available for form validation

  // State
  const [models, setModels] = useState<AIModel[]>([]);
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [testing, setTesting] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [testInput, setTestInput] = useState('');
  const [testOutput, setTestOutput] = useState('');
  const [testConfig, setTestConfig] = useState<TestConfig>({
    temperature: 0.7,
    max_tokens: 1024,
    top_p: 0.9,
    stream: true,
  });

  // Fetch data
  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      // Mock data
      const mockModels: AIModel[] = [
        {
          model_id: 'model-001',
          name: 'GPT-4 Code Review',
          version: 'v2',
          type: 'code_review',
          status: 'active',
          accuracy: 0.94,
          latency_ms: 1800,
          cost_per_1k: 0.03,
          requests_today: 1250,
          last_used: new Date().toISOString(),
        },
        {
          model_id: 'model-002',
          name: 'Claude-3 Security Scanner',
          version: 'v2',
          type: 'security',
          status: 'active',
          accuracy: 0.92,
          latency_ms: 2100,
          cost_per_1k: 0.025,
          requests_today: 890,
          last_used: new Date().toISOString(),
        },
        {
          model_id: 'model-003',
          name: 'GQA Attention Model',
          version: 'v1',
          type: 'experimental',
          status: 'testing',
          accuracy: 0.87,
          latency_ms: 2500,
          cost_per_1k: 0.02,
          requests_today: 150,
          last_used: new Date().toISOString(),
        },
        {
          model_id: 'model-004',
          name: 'Legacy Code Analyzer',
          version: 'v3',
          type: 'code_review',
          status: 'deprecated',
          accuracy: 0.78,
          latency_ms: 3500,
          cost_per_1k: 0.015,
          requests_today: 0,
          last_used: new Date(Date.now() - 86400000).toISOString(),
        },
      ];

      const mockResults: TestResult[] = [
        {
          test_id: 'test-001',
          model_id: 'model-001',
          input: 'Review this Python function...',
          output: 'Analysis: Found 2 issues...',
          latency_ms: 1750,
          tokens_used: 850,
          cost: 0.0255,
          timestamp: new Date().toISOString(),
          success: true,
        },
        {
          test_id: 'test-002',
          model_id: 'model-002',
          input: 'Scan for security vulnerabilities...',
          output: 'Security scan complete. 1 critical issue found.',
          latency_ms: 2050,
          tokens_used: 620,
          cost: 0.0155,
          timestamp: new Date().toISOString(),
          success: true,
        },
      ];

      setModels(mockModels);
      setTestResults(mockResults);
    } catch (error) {
      message.error('Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  const handleTest = async () => {
    if (!selectedModel || !testInput.trim()) {
      message.warning('Please select a model and enter test input');
      return;
    }

    setTesting(true);
    setTestOutput('');

    try {
      // Simulate streaming response
      const words = [
        'Analyzing',
        'code...',
        '\n\n',
        '**Issues Found:**\n',
        '1. ',
        'Missing',
        ' type',
        ' annotations',
        '\n',
        '2. ',
        'Potential',
        ' null',
        ' reference',
        '\n\n',
        '**Suggestions:**\n',
        '- Add',
        ' TypeScript',
        ' types',
        '\n',
        '- Add',
        ' null',
        ' checks',
      ];

      for (const word of words) {
        await new Promise((r) => setTimeout(r, 50));
        setTestOutput((prev) => prev + word);
      }

      // Add to results
      const newResult: TestResult = {
        test_id: `test-${Date.now()}`,
        model_id: selectedModel,
        input: testInput.substring(0, 50) + '...',
        output: 'Analysis complete with 2 issues found.',
        latency_ms: 1200 + Math.random() * 500,
        tokens_used: 500 + Math.floor(Math.random() * 300),
        cost: 0.02 + Math.random() * 0.01,
        timestamp: new Date().toISOString(),
        success: true,
      };

      setTestResults((prev) => [newResult, ...prev]);
      message.success('Test completed successfully');
    } catch (error) {
      message.error('Test failed');
    } finally {
      setTesting(false);
    }
  };

  // Model table columns
  const modelColumns: ColumnsType<AIModel> = [
    {
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
      width: 80,
      render: (version: string) => (
        <Tag color={versionColors[version]}>{version.toUpperCase()}</Tag>
      ),
    },
    {
      title: 'Model',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: AIModel) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.type}
          </Text>
        </div>
      ),
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
      title: 'Accuracy',
      dataIndex: 'accuracy',
      key: 'accuracy',
      width: 120,
      render: (accuracy: number) => (
        <Progress
          percent={Math.round(accuracy * 100)}
          size="small"
          status={accuracy >= 0.85 ? 'success' : 'exception'}
        />
      ),
    },
    {
      title: 'Latency',
      dataIndex: 'latency_ms',
      key: 'latency_ms',
      width: 100,
      render: (ms: number) => (
        <Text type={ms <= 2000 ? 'success' : 'warning'}>{ms}ms</Text>
      ),
    },
    {
      title: 'Cost/1K',
      dataIndex: 'cost_per_1k',
      key: 'cost_per_1k',
      width: 80,
      render: (cost: number) => <Text>${cost.toFixed(3)}</Text>,
    },
    {
      title: 'Requests',
      dataIndex: 'requests_today',
      key: 'requests_today',
      width: 100,
      render: (count: number) => <Text>{count.toLocaleString()}</Text>,
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (_: unknown, record: AIModel) => (
        <Button
          type="primary"
          size="small"
          icon={<PlayCircleOutlined />}
          onClick={() => setSelectedModel(record.model_id)}
          disabled={record.status === 'deprecated'}
        >
          Test
        </Button>
      ),
    },
  ];

  // Test results columns
  const resultColumns: ColumnsType<TestResult> = [
    {
      title: 'Status',
      key: 'success',
      width: 80,
      render: (_: unknown, record: TestResult) =>
        record.success ? (
          <CheckCircleOutlined style={{ color: 'green', fontSize: 18 }} />
        ) : (
          <CloseCircleOutlined style={{ color: 'red', fontSize: 18 }} />
        ),
    },
    {
      title: 'Model',
      dataIndex: 'model_id',
      key: 'model_id',
      render: (id: string) => {
        const model = models.find((m) => m.model_id === id);
        return model?.name || id;
      },
    },
    {
      title: 'Input',
      dataIndex: 'input',
      key: 'input',
      ellipsis: true,
    },
    {
      title: 'Latency',
      dataIndex: 'latency_ms',
      key: 'latency_ms',
      width: 100,
      render: (ms: number) => `${ms.toFixed(0)}ms`,
    },
    {
      title: 'Tokens',
      dataIndex: 'tokens_used',
      key: 'tokens_used',
      width: 80,
    },
    {
      title: 'Cost',
      dataIndex: 'cost',
      key: 'cost',
      width: 80,
      render: (cost: number) => `$${cost.toFixed(4)}`,
    },
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
      render: (ts: string) => new Date(ts).toLocaleTimeString(),
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
            <ExperimentOutlined style={{ marginRight: 8 }} />
            AI Model Testing
          </Title>
        </Col>
        <Col>
          <Button icon={<ReloadOutlined />} onClick={fetchData}>
            Refresh
          </Button>
        </Col>
      </Row>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Active Models"
              value={models.filter((m) => m.status === 'active').length}
              prefix={<RocketOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="V1 Experiments"
              value={models.filter((m) => m.version === 'v1').length}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Tests Today"
              value={testResults.length}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Avg Latency"
              value={
                testResults.length > 0
                  ? Math.round(
                      testResults.reduce((a, b) => a + b.latency_ms, 0) /
                        testResults.length
                    )
                  : 0
              }
              suffix="ms"
              prefix={<LineChartOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={24}>
        {/* Model List */}
        <Col xs={24} lg={14}>
          <Card title="Available Models">
            <Table
              columns={modelColumns}
              dataSource={models}
              rowKey="model_id"
              pagination={false}
              size="small"
            />
          </Card>
        </Col>

        {/* Test Panel */}
        <Col xs={24} lg={10}>
          <Card
            title={
              <Space>
                <CodeOutlined />
                Model Testing
              </Space>
            }
            extra={
              selectedModel && (
                <Tag color="blue">
                  {models.find((m) => m.model_id === selectedModel)?.name}
                </Tag>
              )
            }
          >
            {selectedModel ? (
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                {/* Config */}
                <div>
                  <Text strong>Configuration</Text>
                  <Row gutter={16} style={{ marginTop: 8 }}>
                    <Col span={12}>
                      <Text type="secondary">Temperature: {testConfig.temperature}</Text>
                      <Slider
                        min={0}
                        max={1}
                        step={0.1}
                        value={testConfig.temperature}
                        onChange={(v) => setTestConfig({ ...testConfig, temperature: v })}
                      />
                    </Col>
                    <Col span={12}>
                      <Text type="secondary">Max Tokens: {testConfig.max_tokens}</Text>
                      <Slider
                        min={256}
                        max={4096}
                        step={256}
                        value={testConfig.max_tokens}
                        onChange={(v) => setTestConfig({ ...testConfig, max_tokens: v })}
                      />
                    </Col>
                  </Row>
                </div>

                <Divider style={{ margin: '8px 0' }} />

                {/* Input */}
                <div>
                  <Text strong>Test Input</Text>
                  <TextArea
                    rows={4}
                    placeholder="Enter code or prompt to test..."
                    value={testInput}
                    onChange={(e) => setTestInput(e.target.value)}
                    style={{ marginTop: 8 }}
                  />
                </div>

                {/* Run Button */}
                <Button
                  type="primary"
                  icon={testing ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                  loading={testing}
                  onClick={handleTest}
                  block
                >
                  {testing ? 'Testing...' : 'Run Test'}
                </Button>

                {/* Output */}
                {testOutput && (
                  <div>
                    <Text strong>Output</Text>
                    <div
                      style={{
                        marginTop: 8,
                        padding: 12,
                        background: '#1e1e1e',
                        borderRadius: 4,
                        color: '#d4d4d4',
                        fontFamily: 'monospace',
                        fontSize: 12,
                        whiteSpace: 'pre-wrap',
                        maxHeight: 200,
                        overflow: 'auto',
                      }}
                    >
                      {testOutput}
                    </div>
                  </div>
                )}
              </Space>
            ) : (
              <Alert
                message="Select a Model"
                description="Click 'Test' on any model from the list to start testing."
                type="info"
                showIcon
              />
            )}
          </Card>
        </Col>
      </Row>

      {/* Test History */}
      <Card title="Test History" style={{ marginTop: 24 }}>
        <Table
          columns={resultColumns}
          dataSource={testResults}
          rowKey="test_id"
          pagination={{ pageSize: 5 }}
          size="small"
        />
      </Card>
    </div>
  );
};

export default AIModelTesting;
