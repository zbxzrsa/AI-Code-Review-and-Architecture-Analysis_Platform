/**
 * Code Quality Dashboard
 * 
 * Comprehensive code quality metrics and analysis:
 * - Code coverage tracking
 * - Complexity metrics (cyclomatic, cognitive)
 * - Code duplication detection
 * - Technical debt estimation
 * - Quality trends over time
 */

import React, { useState, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Statistic,
  Progress,
  Table,
  Tag,
  Space,
  Select,
  Tabs,
  List,
  Avatar,
  Tooltip,
} from 'antd';
import {
  CodeOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  FileTextOutlined,
  BranchesOutlined,
  ClockCircleOutlined,
  RiseOutlined,
  FallOutlined,
  BugOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  BarChart,
  Bar,
} from 'recharts';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

interface QualityMetric {
  name: string;
  score: number;
  trend: number;
  status: 'good' | 'warning' | 'critical';
}

interface FileMetric {
  file: string;
  coverage: number;
  complexity: number;
  duplications: number;
  issues: number;
  lines: number;
}

const CodeQualityDashboard: React.FC = () => {
  const { t } = useTranslation();
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');
  const [selectedProject, setSelectedProject] = useState<string>('all');

  // Quality metrics
  const qualityMetrics: QualityMetric[] = [
    { name: 'Code Coverage', score: 78, trend: 5, status: 'warning' },
    { name: 'Maintainability', score: 85, trend: 3, status: 'good' },
    { name: 'Reliability', score: 92, trend: -2, status: 'good' },
    { name: 'Security', score: 88, trend: 8, status: 'good' },
    { name: 'Duplications', score: 94, trend: 1, status: 'good' },
  ];

  // Radar chart data
  const radarData = [
    { subject: 'Coverage', A: 78, fullMark: 100 },
    { subject: 'Maintainability', A: 85, fullMark: 100 },
    { subject: 'Reliability', A: 92, fullMark: 100 },
    { subject: 'Security', A: 88, fullMark: 100 },
    { subject: 'Performance', A: 82, fullMark: 100 },
    { subject: 'Documentation', A: 65, fullMark: 100 },
  ];

  // Trend data
  const trendData = useMemo(() => {
    const data = [];
    for (let i = 29; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      data.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        coverage: 70 + Math.random() * 15 + (29 - i) * 0.2,
        quality: 80 + Math.random() * 10 + (29 - i) * 0.15,
        debt: 120 - (29 - i) * 2 + Math.random() * 20,
      });
    }
    return data;
  }, []);

  // File metrics
  const fileMetrics: FileMetric[] = [
    { file: 'src/services/ai/orchestrator.py', coverage: 45, complexity: 28, duplications: 3, issues: 8, lines: 520 },
    { file: 'src/api/analysis.py', coverage: 62, complexity: 22, duplications: 1, issues: 5, lines: 380 },
    { file: 'src/components/CodeEditor.tsx', coverage: 78, complexity: 15, duplications: 0, issues: 2, lines: 290 },
    { file: 'src/hooks/useAuth.ts', coverage: 85, complexity: 12, duplications: 0, issues: 1, lines: 180 },
    { file: 'src/utils/parser.py', coverage: 52, complexity: 35, duplications: 5, issues: 12, lines: 650 },
    { file: 'src/models/experiment.py', coverage: 90, complexity: 8, duplications: 0, issues: 0, lines: 120 },
  ];

  // Technical debt items
  const debtItems = [
    { id: 1, type: 'Code Smell', title: 'Long method detected', file: 'orchestrator.py:145', effort: '2h', severity: 'major' },
    { id: 2, type: 'Bug', title: 'Null pointer dereference', file: 'parser.py:234', effort: '30m', severity: 'critical' },
    { id: 3, type: 'Vulnerability', title: 'SQL injection risk', file: 'api/users.py:89', effort: '1h', severity: 'blocker' },
    { id: 4, type: 'Code Smell', title: 'Duplicated code block', file: 'utils/helpers.py:45', effort: '45m', severity: 'minor' },
    { id: 5, type: 'Code Smell', title: 'Complex conditional', file: 'services/auth.py:78', effort: '1h', severity: 'major' },
  ];

  // Complexity distribution
  const complexityData = [
    { range: '1-5', count: 45, label: 'Simple' },
    { range: '6-10', count: 32, label: 'Low' },
    { range: '11-20', count: 18, label: 'Moderate' },
    { range: '21-50', count: 8, label: 'High' },
    { range: '50+', count: 3, label: 'Very High' },
  ];

  const fileColumns = [
    {
      title: 'File',
      dataIndex: 'file',
      key: 'file',
      render: (file: string) => (
        <Space>
          <FileTextOutlined />
          <Text code style={{ fontSize: 12 }}>{file}</Text>
        </Space>
      ),
    },
    {
      title: 'Coverage',
      dataIndex: 'coverage',
      key: 'coverage',
      sorter: (a: FileMetric, b: FileMetric) => a.coverage - b.coverage,
      render: (val: number) => (
        <Progress 
          percent={val} 
          size="small" 
          status={val < 50 ? 'exception' : val < 80 ? 'normal' : 'success'}
          style={{ width: 100 }}
        />
      ),
    },
    {
      title: 'Complexity',
      dataIndex: 'complexity',
      key: 'complexity',
      sorter: (a: FileMetric, b: FileMetric) => a.complexity - b.complexity,
      render: (val: number) => (
        <Tag color={val > 25 ? 'red' : val > 15 ? 'orange' : 'green'}>
          {val}
        </Tag>
      ),
    },
    {
      title: 'Duplications',
      dataIndex: 'duplications',
      key: 'duplications',
      render: (val: number) => (
        <Tag color={val > 3 ? 'red' : val > 0 ? 'orange' : 'green'}>
          {val} blocks
        </Tag>
      ),
    },
    {
      title: 'Issues',
      dataIndex: 'issues',
      key: 'issues',
      sorter: (a: FileMetric, b: FileMetric) => a.issues - b.issues,
      render: (val: number) => (
        <Tag color={val > 5 ? 'red' : val > 0 ? 'orange' : 'green'}>
          {val}
        </Tag>
      ),
    },
    {
      title: 'Lines',
      dataIndex: 'lines',
      key: 'lines',
      render: (val: number) => <Text type="secondary">{val}</Text>,
    },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return '#52c41a';
      case 'warning': return '#faad14';
      case 'critical': return '#ff4d4f';
      default: return '#1890ff';
    }
  };

  const overallScore = Math.round(
    qualityMetrics.reduce((sum, m) => sum + m.score, 0) / qualityMetrics.length
  );

  return (
    <div style={{ padding: 24 }}>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}>
            <CodeOutlined style={{ marginRight: 8 }} />
            Code Quality Dashboard
          </Title>
        </Col>
        <Col>
          <Space>
            <Select value={selectedProject} onChange={setSelectedProject} style={{ width: 150 }}>
              <Option value="all">All Projects</Option>
              <Option value="backend">Backend API</Option>
              <Option value="frontend">Frontend App</Option>
              <Option value="ai-core">AI Core</Option>
            </Select>
            <Select value={timeRange} onChange={setTimeRange} style={{ width: 100 }}>
              <Option value="7d">7 Days</Option>
              <Option value="30d">30 Days</Option>
              <Option value="90d">90 Days</Option>
            </Select>
          </Space>
        </Col>
      </Row>

      {/* Overall Score & Key Metrics */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} md={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <Progress
                type="dashboard"
                percent={overallScore}
                width={150}
                strokeColor={{
                  '0%': '#ff4d4f',
                  '50%': '#faad14',
                  '100%': '#52c41a',
                }}
                format={(percent) => (
                  <div>
                    <div style={{ fontSize: 32, fontWeight: 700 }}>{percent}</div>
                    <div style={{ fontSize: 14, color: '#888' }}>Quality Score</div>
                  </div>
                )}
              />
              <div style={{ marginTop: 16 }}>
                <Tag color="green">Grade: B+</Tag>
              </div>
            </div>
          </Card>
        </Col>
        <Col xs={24} md={16}>
          <Card title="Quality Metrics">
            <Row gutter={[16, 16]}>
              {qualityMetrics.map((metric) => (
                <Col xs={12} sm={8} key={metric.name}>
                  <Statistic
                    title={metric.name}
                    value={metric.score}
                    suffix="%"
                    valueStyle={{ color: getStatusColor(metric.status) }}
                    prefix={metric.trend > 0 ? <RiseOutlined /> : <FallOutlined />}
                  />
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {metric.trend > 0 ? '+' : ''}{metric.trend}% vs last period
                  </Text>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>
      </Row>

      {/* Charts Row */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="Quality Radar">
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} />
                <Radar
                  name="Score"
                  dataKey="A"
                  stroke="#1890ff"
                  fill="#1890ff"
                  fillOpacity={0.5}
                />
                <RechartsTooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Quality Trend">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 100]} />
                <RechartsTooltip />
                <Line type="monotone" dataKey="coverage" stroke="#52c41a" name="Coverage" />
                <Line type="monotone" dataKey="quality" stroke="#1890ff" name="Quality" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      {/* Tabs Section */}
      <Card>
        <Tabs defaultActiveKey="files">
          <TabPane tab={<span><FileTextOutlined /> Files ({fileMetrics.length})</span>} key="files">
            <Table
              columns={fileColumns}
              dataSource={fileMetrics}
              rowKey="file"
              pagination={{ pageSize: 10 }}
              size="middle"
            />
          </TabPane>
          
          <TabPane tab={<span><BugOutlined /> Technical Debt ({debtItems.length})</span>} key="debt">
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={6}>
                <Statistic title="Total Debt" value="12h 45m" prefix={<ClockCircleOutlined />} />
              </Col>
              <Col span={6}>
                <Statistic title="Blockers" value={1} valueStyle={{ color: '#ff4d4f' }} />
              </Col>
              <Col span={6}>
                <Statistic title="Critical" value={1} valueStyle={{ color: '#fa8c16' }} />
              </Col>
              <Col span={6}>
                <Statistic title="Major" value={2} valueStyle={{ color: '#faad14' }} />
              </Col>
            </Row>
            <List
              dataSource={debtItems}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Avatar
                        style={{
                          backgroundColor:
                            item.severity === 'blocker' ? '#ff4d4f' :
                            item.severity === 'critical' ? '#fa8c16' :
                            item.severity === 'major' ? '#faad14' : '#52c41a'
                        }}
                        icon={item.type === 'Bug' ? <BugOutlined /> : <WarningOutlined />}
                      />
                    }
                    title={
                      <Space>
                        <Text strong>{item.title}</Text>
                        <Tag>{item.type}</Tag>
                      </Space>
                    }
                    description={
                      <Space>
                        <Text code>{item.file}</Text>
                        <Text type="secondary">Est. effort: {item.effort}</Text>
                      </Space>
                    }
                  />
                  <Tag color={
                    item.severity === 'blocker' ? 'red' :
                    item.severity === 'critical' ? 'orange' :
                    item.severity === 'major' ? 'gold' : 'green'
                  }>
                    {item.severity.toUpperCase()}
                  </Tag>
                </List.Item>
              )}
            />
          </TabPane>

          <TabPane tab={<span><BranchesOutlined /> Complexity</span>} key="complexity">
            <Row gutter={16}>
              <Col xs={24} md={12}>
                <Title level={5}>Cyclomatic Complexity Distribution</Title>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={complexityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
                    <RechartsTooltip />
                    <Bar dataKey="count" fill="#1890ff" />
                  </BarChart>
                </ResponsiveContainer>
              </Col>
              <Col xs={24} md={12}>
                <Title level={5}>Complexity Summary</Title>
                <List
                  dataSource={complexityData}
                  renderItem={(item) => (
                    <List.Item>
                      <List.Item.Meta
                        title={`${item.label} (${item.range})`}
                        description={`${item.count} functions`}
                      />
                      <Progress
                        percent={Math.round((item.count / 106) * 100)}
                        size="small"
                        style={{ width: 100 }}
                        strokeColor={
                          item.range === '50+' ? '#ff4d4f' :
                          item.range === '21-50' ? '#fa8c16' :
                          item.range === '11-20' ? '#faad14' : '#52c41a'
                        }
                      />
                    </List.Item>
                  )}
                />
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default CodeQualityDashboard;
