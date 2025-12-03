/**
 * Model Comparison Dashboard
 * 
 * Compare AI models across different metrics:
 * - Accuracy comparison
 * - Latency benchmarks
 * - Cost analysis
 * - A/B testing results
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
  Select,
  Radio,
  Tooltip,
  Spin,
  Divider,
  Badge,
} from 'antd';
import {
  BarChartOutlined,
  LineChartOutlined,
  SwapOutlined,
  ThunderboltOutlined,
  DollarOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  TrophyOutlined,
  ExperimentOutlined,
  RobotOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text } = Typography;
const { Option } = Select;

interface ModelMetrics {
  id: string;
  name: string;
  version: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  latencyP50: number;
  latencyP95: number;
  latencyP99: number;
  costPer1k: number;
  errorRate: number;
  throughput: number;
  requestsToday: number;
}

interface ComparisonResult {
  metric: string;
  model1Value: number | string;
  model2Value: number | string;
  winner: 'model1' | 'model2' | 'tie';
  difference: string;
}

const ModelComparison: React.FC = () => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [models, setModels] = useState<ModelMetrics[]>([]);
  const [selectedModel1, setSelectedModel1] = useState<string>('');
  const [selectedModel2, setSelectedModel2] = useState<string>('');
  const [comparisonType, setComparisonType] = useState<'accuracy' | 'performance' | 'cost'>('accuracy');

  useEffect(() => {
    const timer = setTimeout(() => {
      const mockModels: ModelMetrics[] = [
        {
          id: 'gpt4-review',
          name: 'GPT-4 Code Review',
          version: 'v2',
          accuracy: 0.94,
          precision: 0.92,
          recall: 0.95,
          f1Score: 0.935,
          latencyP50: 1800,
          latencyP95: 2500,
          latencyP99: 3200,
          costPer1k: 0.03,
          errorRate: 0.02,
          throughput: 150,
          requestsToday: 1250,
        },
        {
          id: 'claude3-security',
          name: 'Claude-3 Security',
          version: 'v2',
          accuracy: 0.92,
          precision: 0.94,
          recall: 0.90,
          f1Score: 0.92,
          latencyP50: 2100,
          latencyP95: 2800,
          latencyP99: 3500,
          costPer1k: 0.025,
          errorRate: 0.015,
          throughput: 120,
          requestsToday: 890,
        },
        {
          id: 'gqa-attention',
          name: 'GQA Attention Model',
          version: 'v1',
          accuracy: 0.87,
          precision: 0.85,
          recall: 0.88,
          f1Score: 0.865,
          latencyP50: 2500,
          latencyP95: 3200,
          latencyP99: 4000,
          costPer1k: 0.02,
          errorRate: 0.05,
          throughput: 100,
          requestsToday: 150,
        },
        {
          id: 'llama-quality',
          name: 'LLaMA Quality Analyzer',
          version: 'v1',
          accuracy: 0.85,
          precision: 0.83,
          recall: 0.86,
          f1Score: 0.845,
          latencyP50: 1500,
          latencyP95: 2000,
          latencyP99: 2500,
          costPer1k: 0.01,
          errorRate: 0.04,
          throughput: 200,
          requestsToday: 320,
        },
      ];
      setModels(mockModels);
      setSelectedModel1(mockModels[0].id);
      setSelectedModel2(mockModels[1].id);
      setLoading(false);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  const getModel = (id: string) => models.find((m) => m.id === id);

  const getComparisonResults = (): ComparisonResult[] => {
    const model1 = getModel(selectedModel1);
    const model2 = getModel(selectedModel2);
    if (!model1 || !model2) return [];

    const results: ComparisonResult[] = [];

    if (comparisonType === 'accuracy') {
      results.push(
        {
          metric: 'Accuracy',
          model1Value: `${(model1.accuracy * 100).toFixed(1)}%`,
          model2Value: `${(model2.accuracy * 100).toFixed(1)}%`,
          winner: model1.accuracy > model2.accuracy ? 'model1' : model1.accuracy < model2.accuracy ? 'model2' : 'tie',
          difference: `${Math.abs((model1.accuracy - model2.accuracy) * 100).toFixed(1)}%`,
        },
        {
          metric: 'Precision',
          model1Value: `${(model1.precision * 100).toFixed(1)}%`,
          model2Value: `${(model2.precision * 100).toFixed(1)}%`,
          winner: model1.precision > model2.precision ? 'model1' : model1.precision < model2.precision ? 'model2' : 'tie',
          difference: `${Math.abs((model1.precision - model2.precision) * 100).toFixed(1)}%`,
        },
        {
          metric: 'Recall',
          model1Value: `${(model1.recall * 100).toFixed(1)}%`,
          model2Value: `${(model2.recall * 100).toFixed(1)}%`,
          winner: model1.recall > model2.recall ? 'model1' : model1.recall < model2.recall ? 'model2' : 'tie',
          difference: `${Math.abs((model1.recall - model2.recall) * 100).toFixed(1)}%`,
        },
        {
          metric: 'F1 Score',
          model1Value: `${(model1.f1Score * 100).toFixed(1)}%`,
          model2Value: `${(model2.f1Score * 100).toFixed(1)}%`,
          winner: model1.f1Score > model2.f1Score ? 'model1' : model1.f1Score < model2.f1Score ? 'model2' : 'tie',
          difference: `${Math.abs((model1.f1Score - model2.f1Score) * 100).toFixed(1)}%`,
        },
        {
          metric: 'Error Rate',
          model1Value: `${(model1.errorRate * 100).toFixed(2)}%`,
          model2Value: `${(model2.errorRate * 100).toFixed(2)}%`,
          winner: model1.errorRate < model2.errorRate ? 'model1' : model1.errorRate > model2.errorRate ? 'model2' : 'tie',
          difference: `${Math.abs((model1.errorRate - model2.errorRate) * 100).toFixed(2)}%`,
        }
      );
    } else if (comparisonType === 'performance') {
      results.push(
        {
          metric: 'Latency P50',
          model1Value: `${model1.latencyP50}ms`,
          model2Value: `${model2.latencyP50}ms`,
          winner: model1.latencyP50 < model2.latencyP50 ? 'model1' : model1.latencyP50 > model2.latencyP50 ? 'model2' : 'tie',
          difference: `${Math.abs(model1.latencyP50 - model2.latencyP50)}ms`,
        },
        {
          metric: 'Latency P95',
          model1Value: `${model1.latencyP95}ms`,
          model2Value: `${model2.latencyP95}ms`,
          winner: model1.latencyP95 < model2.latencyP95 ? 'model1' : model1.latencyP95 > model2.latencyP95 ? 'model2' : 'tie',
          difference: `${Math.abs(model1.latencyP95 - model2.latencyP95)}ms`,
        },
        {
          metric: 'Latency P99',
          model1Value: `${model1.latencyP99}ms`,
          model2Value: `${model2.latencyP99}ms`,
          winner: model1.latencyP99 < model2.latencyP99 ? 'model1' : model1.latencyP99 > model2.latencyP99 ? 'model2' : 'tie',
          difference: `${Math.abs(model1.latencyP99 - model2.latencyP99)}ms`,
        },
        {
          metric: 'Throughput',
          model1Value: `${model1.throughput} req/s`,
          model2Value: `${model2.throughput} req/s`,
          winner: model1.throughput > model2.throughput ? 'model1' : model1.throughput < model2.throughput ? 'model2' : 'tie',
          difference: `${Math.abs(model1.throughput - model2.throughput)} req/s`,
        }
      );
    } else {
      results.push(
        {
          metric: 'Cost per 1K tokens',
          model1Value: `$${model1.costPer1k.toFixed(3)}`,
          model2Value: `$${model2.costPer1k.toFixed(3)}`,
          winner: model1.costPer1k < model2.costPer1k ? 'model1' : model1.costPer1k > model2.costPer1k ? 'model2' : 'tie',
          difference: `$${Math.abs(model1.costPer1k - model2.costPer1k).toFixed(3)}`,
        },
        {
          metric: 'Requests Today',
          model1Value: model1.requestsToday.toLocaleString(),
          model2Value: model2.requestsToday.toLocaleString(),
          winner: model1.requestsToday > model2.requestsToday ? 'model1' : model1.requestsToday < model2.requestsToday ? 'model2' : 'tie',
          difference: Math.abs(model1.requestsToday - model2.requestsToday).toLocaleString(),
        },
        {
          metric: 'Est. Daily Cost',
          model1Value: `$${((model1.requestsToday * model1.costPer1k) / 10).toFixed(2)}`,
          model2Value: `$${((model2.requestsToday * model2.costPer1k) / 10).toFixed(2)}`,
          winner: (model1.requestsToday * model1.costPer1k) < (model2.requestsToday * model2.costPer1k) ? 'model1' : 'model2',
          difference: `$${Math.abs((model1.requestsToday * model1.costPer1k - model2.requestsToday * model2.costPer1k) / 10).toFixed(2)}`,
        }
      );
    }

    return results;
  };

  const comparisonColumns = [
    {
      title: 'Metric',
      dataIndex: 'metric',
      key: 'metric',
      render: (text: string) => <Text strong>{text}</Text>,
    },
    {
      title: getModel(selectedModel1)?.name || 'Model 1',
      dataIndex: 'model1Value',
      key: 'model1Value',
      render: (value: string, record: ComparisonResult) => (
        <Space>
          <Text>{value}</Text>
          {record.winner === 'model1' && <TrophyOutlined style={{ color: '#52c41a' }} />}
        </Space>
      ),
    },
    {
      title: getModel(selectedModel2)?.name || 'Model 2',
      dataIndex: 'model2Value',
      key: 'model2Value',
      render: (value: string, record: ComparisonResult) => (
        <Space>
          <Text>{value}</Text>
          {record.winner === 'model2' && <TrophyOutlined style={{ color: '#52c41a' }} />}
        </Space>
      ),
    },
    {
      title: 'Difference',
      dataIndex: 'difference',
      key: 'difference',
      render: (text: string) => <Tag color="blue">{text}</Tag>,
    },
  ];

  const getWinnerCount = () => {
    const results = getComparisonResults();
    const model1Wins = results.filter((r) => r.winner === 'model1').length;
    const model2Wins = results.filter((r) => r.winner === 'model2').length;
    return { model1Wins, model2Wins };
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <Spin size="large" />
      </div>
    );
  }

  const { model1Wins, model2Wins } = getWinnerCount();

  return (
    <div style={{ padding: 24 }}>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}>
            <BarChartOutlined style={{ marginRight: 8 }} />
            Model Comparison
          </Title>
        </Col>
        <Col>
          <Button type="primary" icon={<ExperimentOutlined />}>
            Run A/B Test
          </Button>
        </Col>
      </Row>

      {/* Model Selection */}
      <Card style={{ marginBottom: 24 }}>
        <Row gutter={24} align="middle">
          <Col xs={24} md={10}>
            <Text type="secondary">Model 1</Text>
            <Select
              value={selectedModel1}
              onChange={setSelectedModel1}
              style={{ width: '100%', marginTop: 8 }}
              size="large"
            >
              {models.map((model) => (
                <Option key={model.id} value={model.id} disabled={model.id === selectedModel2}>
                  <Space>
                    <RobotOutlined />
                    {model.name}
                    <Tag color={model.version === 'v2' ? 'green' : 'orange'}>{model.version}</Tag>
                  </Space>
                </Option>
              ))}
            </Select>
          </Col>
          <Col xs={24} md={4} style={{ textAlign: 'center' }}>
            <SwapOutlined style={{ fontSize: 24, color: '#1890ff' }} />
          </Col>
          <Col xs={24} md={10}>
            <Text type="secondary">Model 2</Text>
            <Select
              value={selectedModel2}
              onChange={setSelectedModel2}
              style={{ width: '100%', marginTop: 8 }}
              size="large"
            >
              {models.map((model) => (
                <Option key={model.id} value={model.id} disabled={model.id === selectedModel1}>
                  <Space>
                    <RobotOutlined />
                    {model.name}
                    <Tag color={model.version === 'v2' ? 'green' : 'orange'}>{model.version}</Tag>
                  </Space>
                </Option>
              ))}
            </Select>
          </Col>
        </Row>
      </Card>

      {/* Winner Summary */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} md={8}>
          <Card>
            <Statistic
              title={getModel(selectedModel1)?.name}
              value={model1Wins}
              suffix={`/ ${getComparisonResults().length} wins`}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: model1Wins > model2Wins ? '#52c41a' : '#888' }}
            />
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card style={{ textAlign: 'center' }}>
            <Text type="secondary">Comparison Type</Text>
            <div style={{ marginTop: 12 }}>
              <Radio.Group
                value={comparisonType}
                onChange={(e) => setComparisonType(e.target.value)}
                buttonStyle="solid"
              >
                <Radio.Button value="accuracy">
                  <CheckCircleOutlined /> Accuracy
                </Radio.Button>
                <Radio.Button value="performance">
                  <ClockCircleOutlined /> Performance
                </Radio.Button>
                <Radio.Button value="cost">
                  <DollarOutlined /> Cost
                </Radio.Button>
              </Radio.Group>
            </div>
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card>
            <Statistic
              title={getModel(selectedModel2)?.name}
              value={model2Wins}
              suffix={`/ ${getComparisonResults().length} wins`}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: model2Wins > model1Wins ? '#52c41a' : '#888' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Comparison Table */}
      <Card title="Detailed Comparison">
        <Table
          columns={comparisonColumns}
          dataSource={getComparisonResults()}
          rowKey="metric"
          pagination={false}
          size="middle"
        />
      </Card>

      {/* Visual Comparison */}
      <Card title="Visual Comparison" style={{ marginTop: 24 }}>
        <Row gutter={[16, 16]}>
          {getComparisonResults().map((result) => (
            <Col xs={24} sm={12} md={8} key={result.metric}>
              <div style={{ textAlign: 'center', padding: 16 }}>
                <Text type="secondary">{result.metric}</Text>
                <div style={{ marginTop: 8, display: 'flex', justifyContent: 'center', gap: 16 }}>
                  <div>
                    <Progress
                      type="circle"
                      percent={typeof result.model1Value === 'string' && result.model1Value.includes('%')
                        ? parseFloat(result.model1Value)
                        : 75}
                      width={60}
                      strokeColor={result.winner === 'model1' ? '#52c41a' : '#1890ff'}
                      format={() => (
                        <Text style={{ fontSize: 10 }}>{String(result.model1Value).slice(0, 6)}</Text>
                      )}
                    />
                    <div style={{ marginTop: 4 }}>
                      <Text style={{ fontSize: 10 }}>
                        {getModel(selectedModel1)?.name.slice(0, 10)}
                      </Text>
                    </div>
                  </div>
                  <div>
                    <Progress
                      type="circle"
                      percent={typeof result.model2Value === 'string' && result.model2Value.includes('%')
                        ? parseFloat(result.model2Value)
                        : 75}
                      width={60}
                      strokeColor={result.winner === 'model2' ? '#52c41a' : '#1890ff'}
                      format={() => (
                        <Text style={{ fontSize: 10 }}>{String(result.model2Value).slice(0, 6)}</Text>
                      )}
                    />
                    <div style={{ marginTop: 4 }}>
                      <Text style={{ fontSize: 10 }}>
                        {getModel(selectedModel2)?.name.slice(0, 10)}
                      </Text>
                    </div>
                  </div>
                </div>
              </div>
            </Col>
          ))}
        </Row>
      </Card>
    </div>
  );
};

export default ModelComparison;
