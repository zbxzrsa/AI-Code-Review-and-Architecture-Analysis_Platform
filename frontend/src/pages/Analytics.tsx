/**
 * Analytics Dashboard Page
 * 分析仪表板页面
 * 
 * Comprehensive analytics with:
 * - Code quality trends over time
 * - Issue resolution metrics
 * - Team performance insights
 * - AI model usage statistics
 * - Cost analytics
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Typography,
  Select,
  DatePicker,
  Space,
  Table,
  Tag,
  Progress,
  Tabs,
  Tooltip,
  Badge,
} from 'antd';
import {
  LineChartOutlined,
  BarChartOutlined,
  PieChartOutlined,
  RiseOutlined,
  FallOutlined,
  BugOutlined,
  SafetyCertificateOutlined,
  ClockCircleOutlined,
  DollarOutlined,
  TeamOutlined,
  CodeOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  ThunderboltOutlined,
  TrophyOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { api } from '../services/api';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;

// Mock chart component (in production, use ECharts or Recharts)
const SimpleChart: React.FC<{ data: number[]; color: string; height?: number }> = ({ data, color, height = 60 }) => {
  const max = Math.max(...data);
  return (
    <div style={{ display: 'flex', alignItems: 'end', gap: 2, height }}>
      {data.map((value, index) => (
        <div
          key={index}
          style={{
            flex: 1,
            height: `${(value / max) * 100}%`,
            backgroundColor: color,
            borderRadius: 2,
            minHeight: 4,
          }}
        />
      ))}
    </div>
  );
};

export const Analytics: React.FC = () => {
  const { t } = useTranslation();
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(30, 'day'),
    dayjs(),
  ]);
  const [selectedProject, setSelectedProject] = useState<string>('all');
  const [loading, setLoading] = useState(false);

  // Mock analytics data
  const overviewStats = {
    totalAnalyses: 1547,
    analysesChange: 12.5,
    issuesFound: 4832,
    issuesChange: -8.3,
    issuesResolved: 4156,
    resolvedChange: 15.2,
    avgResponseTime: 2.3,
    responseTimeChange: -5.1,
    totalCost: 1250.45,
    costChange: 8.7,
    securityIssues: 127,
    securityChange: -22.4,
  };

  const weeklyData = [45, 52, 38, 65, 72, 58, 81];
  const monthlyTrend = [120, 135, 128, 145, 162, 158, 175, 190, 185, 210, 225, 240];

  const issuesByCategory = [
    { category: 'Security', count: 127, percentage: 15, color: '#ff4d4f' },
    { category: 'Performance', count: 243, percentage: 28, color: '#faad14' },
    { category: 'Code Quality', count: 312, percentage: 36, color: '#1890ff' },
    { category: 'Best Practices', count: 182, percentage: 21, color: '#52c41a' },
  ];

  const topIssues = [
    { rule: 'security/no-hardcoded-secrets', count: 45, severity: 'critical' },
    { rule: 'security/sql-injection', count: 32, severity: 'critical' },
    { rule: 'performance/n-plus-one-query', count: 78, severity: 'high' },
    { rule: 'quality/cyclomatic-complexity', count: 124, severity: 'medium' },
    { rule: 'quality/duplicate-code', count: 89, severity: 'medium' },
  ];

  const teamPerformance = [
    { name: 'Frontend Team', analyses: 450, issues: 1200, resolved: 1150, rate: 95.8 },
    { name: 'Backend Team', analyses: 380, issues: 980, resolved: 920, rate: 93.9 },
    { name: 'DevOps Team', analyses: 220, issues: 450, resolved: 445, rate: 98.9 },
    { name: 'Mobile Team', analyses: 180, issues: 520, resolved: 480, rate: 92.3 },
  ];

  const aiModelUsage = [
    { model: 'GPT-4 Turbo', requests: 8500, cost: 680.00, accuracy: 94.2 },
    { model: 'Claude 3 Opus', requests: 5200, cost: 520.00, accuracy: 92.8 },
    { model: 'GPT-4', requests: 1200, cost: 48.00, accuracy: 93.5 },
    { model: 'Claude 3 Sonnet', requests: 800, cost: 2.45, accuracy: 91.2 },
  ];

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'red';
      case 'high': return 'orange';
      case 'medium': return 'gold';
      case 'low': return 'blue';
      default: return 'default';
    }
  };

  return (
    <div className="analytics-page">
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <Title level={3}>
            <LineChartOutlined /> {t('analytics.title', 'Analytics & Insights')}
          </Title>
          <Text type="secondary">
            {t('analytics.subtitle', 'Track code quality trends and team performance')}
          </Text>
        </div>
        <Space>
          <Select
            value={selectedProject}
            onChange={setSelectedProject}
            style={{ width: 200 }}
            options={[
              { value: 'all', label: 'All Projects' },
              { value: 'proj_1', label: 'AI Code Review Platform' },
              { value: 'proj_2', label: 'Backend Services' },
              { value: 'proj_3', label: 'Mobile App' },
            ]}
          />
          <RangePicker
            value={dateRange}
            onChange={(dates) => dates && setDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])}
          />
        </Space>
      </div>

      {/* Overview Stats */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={8} lg={4}>
          <Card>
            <Statistic
              title="Total Analyses"
              value={overviewStats.totalAnalyses}
              prefix={<CodeOutlined />}
              suffix={
                <Text type={overviewStats.analysesChange > 0 ? 'success' : 'danger'} style={{ fontSize: 14 }}>
                  {overviewStats.analysesChange > 0 ? <RiseOutlined /> : <FallOutlined />}
                  {Math.abs(overviewStats.analysesChange)}%
                </Text>
              }
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={4}>
          <Card>
            <Statistic
              title="Issues Found"
              value={overviewStats.issuesFound}
              prefix={<BugOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={4}>
          <Card>
            <Statistic
              title="Issues Resolved"
              value={overviewStats.issuesResolved}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={4}>
          <Card>
            <Statistic
              title="Security Issues"
              value={overviewStats.securityIssues}
              prefix={<SafetyCertificateOutlined />}
              valueStyle={{ color: overviewStats.securityChange < 0 ? '#52c41a' : '#ff4d4f' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={4}>
          <Card>
            <Statistic
              title="Avg Response"
              value={overviewStats.avgResponseTime}
              suffix="s"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={4}>
          <Card>
            <Statistic
              title="Total Cost"
              value={overviewStats.totalCost}
              prefix={<DollarOutlined />}
              precision={2}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* Analyses Trend */}
        <Col xs={24} lg={16}>
          <Card
            title={<><BarChartOutlined /> Analysis Trend</>}
            extra={<Text type="secondary">Last 30 days</Text>}
          >
            <div style={{ padding: '20px 0' }}>
              <SimpleChart data={monthlyTrend} color="#1890ff" height={200} />
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
                {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].map(m => (
                  <Text key={m} type="secondary" style={{ fontSize: 12 }}>{m}</Text>
                ))}
              </div>
            </div>
          </Card>
        </Col>

        {/* Issues by Category */}
        <Col xs={24} lg={8}>
          <Card title={<><PieChartOutlined /> Issues by Category</>}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {issuesByCategory.map(item => (
                <div key={item.category}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <Text>{item.category}</Text>
                    <Text strong>{item.count}</Text>
                  </div>
                  <Progress 
                    percent={item.percentage} 
                    strokeColor={item.color}
                    showInfo={false}
                    size="small"
                  />
                </div>
              ))}
            </Space>
          </Card>
        </Col>

        {/* Top Issues */}
        <Col xs={24} lg={12}>
          <Card title={<><WarningOutlined /> Top Issues</>}>
            <Table
              dataSource={topIssues}
              rowKey="rule"
              pagination={false}
              size="small"
              columns={[
                {
                  title: 'Rule',
                  dataIndex: 'rule',
                  render: (rule) => <Text code>{rule}</Text>,
                },
                {
                  title: 'Count',
                  dataIndex: 'count',
                  width: 80,
                  sorter: (a, b) => a.count - b.count,
                },
                {
                  title: 'Severity',
                  dataIndex: 'severity',
                  width: 100,
                  render: (severity) => (
                    <Tag color={getSeverityColor(severity)}>
                      {severity.toUpperCase()}
                    </Tag>
                  ),
                },
              ]}
            />
          </Card>
        </Col>

        {/* AI Model Usage */}
        <Col xs={24} lg={12}>
          <Card title={<><ThunderboltOutlined /> AI Model Usage</>}>
            <Table
              dataSource={aiModelUsage}
              rowKey="model"
              pagination={false}
              size="small"
              columns={[
                {
                  title: 'Model',
                  dataIndex: 'model',
                },
                {
                  title: 'Requests',
                  dataIndex: 'requests',
                  render: (v) => v.toLocaleString(),
                },
                {
                  title: 'Cost',
                  dataIndex: 'cost',
                  render: (v) => `$${v.toFixed(2)}`,
                },
                {
                  title: 'Accuracy',
                  dataIndex: 'accuracy',
                  render: (v) => (
                    <Text type={v >= 93 ? 'success' : 'warning'}>{v}%</Text>
                  ),
                },
              ]}
            />
          </Card>
        </Col>

        {/* Team Performance */}
        <Col xs={24}>
          <Card title={<><TrophyOutlined /> Team Performance</>}>
            <Table
              dataSource={teamPerformance}
              rowKey="name"
              pagination={false}
              columns={[
                {
                  title: 'Team',
                  dataIndex: 'name',
                  render: (name) => (
                    <Space>
                      <TeamOutlined />
                      {name}
                    </Space>
                  ),
                },
                {
                  title: 'Analyses',
                  dataIndex: 'analyses',
                  sorter: (a, b) => a.analyses - b.analyses,
                },
                {
                  title: 'Issues Found',
                  dataIndex: 'issues',
                  sorter: (a, b) => a.issues - b.issues,
                },
                {
                  title: 'Issues Resolved',
                  dataIndex: 'resolved',
                  sorter: (a, b) => a.resolved - b.resolved,
                },
                {
                  title: 'Resolution Rate',
                  dataIndex: 'rate',
                  render: (rate) => (
                    <Space>
                      <Progress 
                        percent={rate} 
                        size="small" 
                        style={{ width: 100 }}
                        strokeColor={rate >= 95 ? '#52c41a' : rate >= 90 ? '#faad14' : '#ff4d4f'}
                      />
                      <Text strong>{rate}%</Text>
                    </Space>
                  ),
                  sorter: (a, b) => a.rate - b.rate,
                },
              ]}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Analytics;
