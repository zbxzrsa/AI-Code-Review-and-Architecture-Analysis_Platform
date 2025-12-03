/**
 * Code Metrics Page
 * 代码度量页面
 * 
 * Features:
 * - Code complexity metrics
 * - Coverage statistics
 * - Technical debt tracking
 * - Quality trends
 */

import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Progress,
  Tag,
  Select,
  Table,
  Badge,
} from 'antd';
import type { TableProps } from 'antd';
import {
  BarChartOutlined,
  CodeOutlined,
  FileTextOutlined,
  FunctionOutlined,
  ApartmentOutlined,
  SafetyCertificateOutlined,
  RiseOutlined,
  FallOutlined,
  CheckCircleOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text } = Typography;

interface FileMetric {
  path: string;
  lines: number;
  complexity: number;
  coverage: number;
  issues: number;
  grade: 'A' | 'B' | 'C' | 'D' | 'F';
}

const mockFileMetrics: FileMetric[] = [
  { path: 'src/api/authentication.py', lines: 245, complexity: 8, coverage: 92, issues: 1, grade: 'A' },
  { path: 'src/api/users.py', lines: 312, complexity: 15, coverage: 78, issues: 3, grade: 'B' },
  { path: 'src/services/analysis.py', lines: 528, complexity: 22, coverage: 65, issues: 5, grade: 'C' },
  { path: 'src/utils/helpers.py', lines: 156, complexity: 5, coverage: 95, issues: 0, grade: 'A' },
  { path: 'src/models/project.py', lines: 189, complexity: 12, coverage: 88, issues: 2, grade: 'B' },
  { path: 'src/core/orchestrator.py', lines: 612, complexity: 28, coverage: 55, issues: 8, grade: 'D' },
];

const gradeConfig = {
  A: { color: '#22c55e', label: 'Excellent' },
  B: { color: '#84cc16', label: 'Good' },
  C: { color: '#f59e0b', label: 'Fair' },
  D: { color: '#f97316', label: 'Poor' },
  F: { color: '#ef4444', label: 'Critical' },
};

const MetricCard: React.FC<{
  title: string;
  value: number | string;
  suffix?: string;
  icon: React.ReactNode;
  trend?: number;
  color?: string;
  description?: string;
}> = ({ title, value, suffix, icon, trend, color, description }) => (
  <Card style={{ borderRadius: 12, height: '100%' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
      <div>
        <Text type="secondary" style={{ fontSize: 13 }}>{title}</Text>
        <div style={{ marginTop: 8 }}>
          <span style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>
            {value}
          </span>
          {suffix && <span style={{ fontSize: 16, color: '#64748b', marginLeft: 4 }}>{suffix}</span>}
        </div>
        {trend !== undefined && (
          <div style={{ marginTop: 8 }}>
            <Tag color={trend >= 0 ? 'green' : 'red'} icon={trend >= 0 ? <RiseOutlined /> : <FallOutlined />}>
              {trend >= 0 ? '+' : ''}{trend}% this week
            </Tag>
          </div>
        )}
        {description && (
          <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
            {description}
          </Text>
        )}
      </div>
      <div style={{
        width: 48,
        height: 48,
        borderRadius: 12,
        background: `${color || '#2563eb'}15`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: color || '#2563eb',
        fontSize: 22,
      }}>
        {icon}
      </div>
    </div>
  </Card>
);

export const CodeMetrics: React.FC = () => {
  const { t: _t } = useTranslation();
  const [selectedProject, setSelectedProject] = useState('all');
  const [timeRange, setTimeRange] = useState('7d');

  const overallMetrics = {
    codeQuality: 78,
    coverage: 72,
    technicalDebt: 24,
    complexity: 12.5,
    duplications: 3.2,
    issues: 45,
  };

  const columns: TableProps<FileMetric>['columns'] = [
    {
      title: 'File',
      dataIndex: 'path',
      render: (path: string) => (
        <Space>
          <FileTextOutlined style={{ color: '#64748b' }} />
          <Text code style={{ fontSize: 12 }}>{path}</Text>
        </Space>
      ),
    },
    {
      title: 'Lines',
      dataIndex: 'lines',
      width: 100,
      sorter: (a, b) => a.lines - b.lines,
    },
    {
      title: 'Complexity',
      dataIndex: 'complexity',
      width: 120,
      sorter: (a, b) => a.complexity - b.complexity,
      render: (complexity: number) => (
        <Tag color={complexity <= 10 ? 'green' : complexity <= 20 ? 'orange' : 'red'}>
          {complexity}
        </Tag>
      ),
    },
    {
      title: 'Coverage',
      dataIndex: 'coverage',
      width: 150,
      sorter: (a, b) => a.coverage - b.coverage,
      render: (coverage: number) => (
        <Progress
          percent={coverage}
          size="small"
          strokeColor={coverage >= 80 ? '#22c55e' : coverage >= 60 ? '#f59e0b' : '#ef4444'}
          format={p => `${p}%`}
        />
      ),
    },
    {
      title: 'Issues',
      dataIndex: 'issues',
      width: 100,
      sorter: (a, b) => a.issues - b.issues,
      render: (issues: number) => (
        issues > 0 ? (
          <Badge count={issues} style={{ backgroundColor: issues > 5 ? '#ef4444' : '#f59e0b' }} />
        ) : (
          <CheckCircleOutlined style={{ color: '#22c55e' }} />
        )
      ),
    },
    {
      title: 'Grade',
      dataIndex: 'grade',
      width: 100,
      render: (grade: keyof typeof gradeConfig) => {
        const config = gradeConfig[grade];
        return (
          <Tag color={config.color} style={{ fontWeight: 700, fontSize: 14 }}>
            {grade}
          </Tag>
        );
      },
    },
  ];

  return (
    <div className="code-metrics-page" style={{ maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <BarChartOutlined style={{ color: '#2563eb' }} /> Code Metrics
          </Title>
          <Text type="secondary">Monitor code quality, complexity, and technical debt</Text>
        </div>
        <Space>
          <Select
            value={selectedProject}
            onChange={setSelectedProject}
            style={{ width: 180 }}
            options={[
              { value: 'all', label: 'All Projects' },
              { value: 'backend', label: 'Backend' },
              { value: 'frontend', label: 'Frontend' },
            ]}
          />
          <Select
            value={timeRange}
            onChange={setTimeRange}
            style={{ width: 120 }}
            options={[
              { value: '7d', label: 'Last 7 days' },
              { value: '30d', label: 'Last 30 days' },
              { value: '90d', label: 'Last 90 days' },
            ]}
          />
        </Space>
      </div>

      {/* Overall Quality Score */}
      <Card style={{ marginBottom: 24, borderRadius: 12 }}>
        <Row gutter={24} align="middle">
          <Col xs={24} md={8} style={{ textAlign: 'center' }}>
            <Progress
              type="dashboard"
              percent={overallMetrics.codeQuality}
              size={180}
              strokeColor={{
                '0%': '#3b82f6',
                '100%': '#22c55e',
              }}
              format={percent => (
                <div>
                  <div style={{ fontSize: 36, fontWeight: 700 }}>{percent}</div>
                  <div style={{ fontSize: 14, color: '#64748b' }}>Quality Score</div>
                </div>
              )}
            />
            <div style={{ marginTop: 16 }}>
              <Tag color="green" icon={<RiseOutlined />}>+5% from last week</Tag>
            </div>
          </Col>
          <Col xs={24} md={16}>
            <Row gutter={[16, 16]}>
              <Col xs={12} md={8}>
                <div style={{ padding: 16, background: '#f8fafc', borderRadius: 12 }}>
                  <Text type="secondary">Coverage</Text>
                  <div style={{ fontSize: 24, fontWeight: 700 }}>{overallMetrics.coverage}%</div>
                  <Progress percent={overallMetrics.coverage} showInfo={false} strokeColor="#22c55e" />
                </div>
              </Col>
              <Col xs={12} md={8}>
                <div style={{ padding: 16, background: '#f8fafc', borderRadius: 12 }}>
                  <Text type="secondary">Tech Debt</Text>
                  <div style={{ fontSize: 24, fontWeight: 700 }}>{overallMetrics.technicalDebt}h</div>
                  <Progress percent={100 - overallMetrics.technicalDebt} showInfo={false} strokeColor="#f59e0b" />
                </div>
              </Col>
              <Col xs={12} md={8}>
                <div style={{ padding: 16, background: '#f8fafc', borderRadius: 12 }}>
                  <Text type="secondary">Duplications</Text>
                  <div style={{ fontSize: 24, fontWeight: 700 }}>{overallMetrics.duplications}%</div>
                  <Progress percent={100 - overallMetrics.duplications * 10} showInfo={false} strokeColor="#8b5cf6" />
                </div>
              </Col>
              <Col xs={12} md={8}>
                <div style={{ padding: 16, background: '#f8fafc', borderRadius: 12 }}>
                  <Text type="secondary">Avg Complexity</Text>
                  <div style={{ fontSize: 24, fontWeight: 700 }}>{overallMetrics.complexity}</div>
                  <Tag color={overallMetrics.complexity <= 10 ? 'green' : 'orange'}>
                    {overallMetrics.complexity <= 10 ? 'Low' : 'Medium'}
                  </Tag>
                </div>
              </Col>
              <Col xs={12} md={8}>
                <div style={{ padding: 16, background: '#f8fafc', borderRadius: 12 }}>
                  <Text type="secondary">Open Issues</Text>
                  <div style={{ fontSize: 24, fontWeight: 700 }}>{overallMetrics.issues}</div>
                  <Tag color="orange" icon={<WarningOutlined />}>Needs attention</Tag>
                </div>
              </Col>
              <Col xs={12} md={8}>
                <div style={{ padding: 16, background: '#f8fafc', borderRadius: 12 }}>
                  <Text type="secondary">Security</Text>
                  <div style={{ fontSize: 24, fontWeight: 700 }}>A</div>
                  <Tag color="green" icon={<SafetyCertificateOutlined />}>Secure</Tag>
                </div>
              </Col>
            </Row>
          </Col>
        </Row>
      </Card>

      {/* Detailed Metrics */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={12} md={6}>
          <MetricCard
            title="Total Lines of Code"
            value="45.2K"
            icon={<CodeOutlined />}
            trend={8}
            description="Across all files"
          />
        </Col>
        <Col xs={12} md={6}>
          <MetricCard
            title="Functions"
            value={856}
            icon={<FunctionOutlined />}
            trend={12}
            color="#8b5cf6"
          />
        </Col>
        <Col xs={12} md={6}>
          <MetricCard
            title="Classes"
            value={124}
            icon={<ApartmentOutlined />}
            trend={5}
            color="#22c55e"
          />
        </Col>
        <Col xs={12} md={6}>
          <MetricCard
            title="Files"
            value={312}
            icon={<FileTextOutlined />}
            trend={3}
            color="#f59e0b"
          />
        </Col>
      </Row>

      {/* File Metrics Table */}
      <Card
        title={<><FileTextOutlined /> File-Level Metrics</>}
        extra={<Text type="secondary">Sorted by complexity</Text>}
        style={{ borderRadius: 12 }}
      >
        <Table
          columns={columns}
          dataSource={mockFileMetrics}
          rowKey="path"
          pagination={{ pageSize: 10 }}
        />
      </Card>
    </div>
  );
};

export default CodeMetrics;
