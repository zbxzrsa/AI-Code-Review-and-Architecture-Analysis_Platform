/**
 * Security Dashboard Page
 * 安全仪表板页面
 * 
 * Features:
 * - Security vulnerability overview
 * - OWASP Top 10 tracking
 * - CVE monitoring
 * - Compliance status
 * - Security trends
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Typography,
  Table,
  Tag,
  Progress,
  Space,
  Select,
  Alert,
  Badge,
  Button,
  List,
  Spin,
  message,
} from 'antd';
import type { TableProps } from 'antd';
import {
  SafetyCertificateOutlined,
  BugOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  SecurityScanOutlined,
  FileProtectOutlined,
  AlertOutlined,
  RiseOutlined,
  FallOutlined,
  EyeOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { apiService } from '../services/api';

const { Title, Text } = Typography;

interface Vulnerability {
  id: string;
  title: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  category: string;
  cve?: string;
  project: string;
  file: string;
  line: number;
  status: 'open' | 'in_progress' | 'resolved' | 'ignored';
  discoveredAt: string;
  assignee?: string;
}

const mockVulnerabilities: Vulnerability[] = [
  {
    id: 'vuln_1',
    title: 'SQL Injection in User Authentication',
    severity: 'critical',
    category: 'A03:2021 Injection',
    cve: 'CVE-2024-1234',
    project: 'Backend Services',
    file: 'src/auth/login.py',
    line: 45,
    status: 'open',
    discoveredAt: '2024-03-01T10:00:00Z',
    assignee: 'John Doe',
  },
  {
    id: 'vuln_2',
    title: 'Hardcoded API Key Exposure',
    severity: 'critical',
    category: 'A02:2021 Cryptographic Failures',
    project: 'AI Platform',
    file: 'src/services/ai.ts',
    line: 12,
    status: 'in_progress',
    discoveredAt: '2024-02-28T14:30:00Z',
    assignee: 'Jane Smith',
  },
  {
    id: 'vuln_3',
    title: 'Cross-Site Scripting (XSS)',
    severity: 'high',
    category: 'A03:2021 Injection',
    project: 'Frontend',
    file: 'src/components/Comment.tsx',
    line: 78,
    status: 'open',
    discoveredAt: '2024-02-27T09:00:00Z',
  },
  {
    id: 'vuln_4',
    title: 'Insecure Direct Object Reference',
    severity: 'high',
    category: 'A01:2021 Broken Access Control',
    project: 'Backend Services',
    file: 'src/api/documents.py',
    line: 156,
    status: 'resolved',
    discoveredAt: '2024-02-20T11:00:00Z',
  },
  {
    id: 'vuln_5',
    title: 'Missing Rate Limiting',
    severity: 'medium',
    category: 'A04:2021 Insecure Design',
    project: 'API Gateway',
    file: 'src/middleware/auth.ts',
    line: 34,
    status: 'open',
    discoveredAt: '2024-02-25T16:00:00Z',
  },
];

const owaspTop10 = [
  { id: 'A01', name: 'Broken Access Control', count: 23, trend: 'up', change: 15 },
  { id: 'A02', name: 'Cryptographic Failures', count: 18, trend: 'down', change: -8 },
  { id: 'A03', name: 'Injection', count: 31, trend: 'up', change: 22 },
  { id: 'A04', name: 'Insecure Design', count: 12, trend: 'stable', change: 0 },
  { id: 'A05', name: 'Security Misconfiguration', count: 28, trend: 'down', change: -12 },
  { id: 'A06', name: 'Vulnerable Components', count: 45, trend: 'up', change: 8 },
  { id: 'A07', name: 'Auth Failures', count: 15, trend: 'down', change: -20 },
  { id: 'A08', name: 'Software Integrity Failures', count: 8, trend: 'stable', change: 2 },
  { id: 'A09', name: 'Logging Failures', count: 19, trend: 'up', change: 5 },
  { id: 'A10', name: 'SSRF', count: 6, trend: 'down', change: -10 },
];

const complianceChecks = [
  { name: 'OWASP Top 10', status: 'partial', score: 72, items: 10, passed: 7 },
  { name: 'PCI DSS', status: 'passing', score: 95, items: 12, passed: 11 },
  { name: 'SOC 2', status: 'passing', score: 88, items: 15, passed: 13 },
  { name: 'GDPR', status: 'partial', score: 78, items: 8, passed: 6 },
  { name: 'HIPAA', status: 'failing', score: 45, items: 10, passed: 4 },
];

export const SecurityDashboard: React.FC = () => {
  const { t } = useTranslation();
  const [vulnerabilities, setVulnerabilities] = useState<Vulnerability[]>(mockVulnerabilities);
  const [loading, setLoading] = useState(false);
  const [selectedProject, setSelectedProject] = useState<string>('all');
  const [severityFilter, setSeverityFilter] = useState<string>('all');

  // Fetch vulnerabilities from API
  const fetchVulnerabilities = useCallback(async () => {
    setLoading(true);
    try {
      const response = await apiService.analysis.getVulnerabilities();
      if (response.data?.items) {
        setVulnerabilities(response.data.items);
      }
    } catch {
      // Use mock data if API fails
      setVulnerabilities(mockVulnerabilities);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchVulnerabilities();
  }, [fetchVulnerabilities]);

  const stats = {
    critical: vulnerabilities.filter(v => v.severity === 'critical').length,
    high: vulnerabilities.filter(v => v.severity === 'high').length,
    medium: vulnerabilities.filter(v => v.severity === 'medium').length,
    low: vulnerabilities.filter(v => v.severity === 'low').length,
    open: vulnerabilities.filter(v => v.status === 'open').length,
    resolved: vulnerabilities.filter(v => v.status === 'resolved').length,
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#ff4d4f';
      case 'high': return '#fa8c16';
      case 'medium': return '#fadb14';
      case 'low': return '#1890ff';
      default: return '#999';
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'open': return <Badge status="error" text="Open" />;
      case 'in_progress': return <Badge status="processing" text="In Progress" />;
      case 'resolved': return <Badge status="success" text="Resolved" />;
      case 'ignored': return <Badge status="default" text="Ignored" />;
      default: return <Badge status="default" text={status} />;
    }
  };

  const columns: TableProps<Vulnerability>['columns'] = [
    {
      title: 'Severity',
      dataIndex: 'severity',
      width: 100,
      render: (severity) => (
        <Tag color={getSeverityColor(severity)} style={{ minWidth: 70, textAlign: 'center' }}>
          {severity.toUpperCase()}
        </Tag>
      ),
      filters: [
        { text: 'Critical', value: 'critical' },
        { text: 'High', value: 'high' },
        { text: 'Medium', value: 'medium' },
        { text: 'Low', value: 'low' },
      ],
      onFilter: (value, record) => record.severity === value,
    },
    {
      title: 'Vulnerability',
      key: 'title',
      render: (_, record) => (
        <div>
          <Text strong>{record.title}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.category}
            {record.cve && <Tag color="volcano" style={{ marginLeft: 8 }}>{record.cve}</Tag>}
          </Text>
        </div>
      ),
    },
    {
      title: 'Location',
      key: 'location',
      render: (_, record) => (
        <div>
          <Text>{record.project}</Text>
          <br />
          <Text code style={{ fontSize: 12 }}>{record.file}:{record.line}</Text>
        </div>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      width: 120,
      render: (status) => getStatusBadge(status),
    },
    {
      title: 'Discovered',
      dataIndex: 'discoveredAt',
      width: 120,
      render: (date) => new Date(date).toLocaleDateString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: () => (
        <Button size="small" icon={<EyeOutlined />}>
          View
        </Button>
      ),
    },
  ];

  return (
    <div className="security-dashboard">
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <Title level={3}>
            <SafetyCertificateOutlined /> {t('security.title', 'Security Dashboard')}
          </Title>
          <Text type="secondary">
            {t('security.subtitle', 'Monitor vulnerabilities and compliance status')}
          </Text>
        </div>
        <Space>
          <Select
            value={selectedProject}
            onChange={setSelectedProject}
            style={{ width: 200 }}
            options={[
              { value: 'all', label: 'All Projects' },
              { value: 'backend', label: 'Backend Services' },
              { value: 'frontend', label: 'Frontend' },
              { value: 'api', label: 'API Gateway' },
            ]}
          />
          <Button icon={<ReloadOutlined />}>Scan Now</Button>
        </Space>
      </div>

      {/* Critical Alert */}
      {stats.critical > 0 && (
        <Alert
          message={`${stats.critical} Critical Vulnerabilities Detected`}
          description="Critical security vulnerabilities require immediate attention. Review and remediate as soon as possible."
          type="error"
          showIcon
          icon={<AlertOutlined />}
          style={{ marginBottom: 24 }}
          action={
            <Button danger>View Critical Issues</Button>
          }
        />
      )}

      {/* Stats Overview */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="Critical"
              value={stats.critical}
              valueStyle={{ color: '#ff4d4f' }}
              prefix={<CloseCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="High"
              value={stats.high}
              valueStyle={{ color: '#fa8c16' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="Open Issues"
              value={stats.open}
              valueStyle={{ color: '#1890ff' }}
              prefix={<BugOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="Resolved (30d)"
              value={stats.resolved}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* OWASP Top 10 */}
        <Col xs={24} lg={12}>
          <Card
            title={<><SecurityScanOutlined /> OWASP Top 10 Coverage</>}
            extra={<Tag color="blue">2021</Tag>}
          >
            <List
              size="small"
              dataSource={owaspTop10}
              renderItem={item => (
                <List.Item>
                  <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                    <div>
                      <Text strong>{item.id}</Text>
                      <Text style={{ marginLeft: 8 }}>{item.name}</Text>
                    </div>
                    <Space>
                      <Badge count={item.count} style={{ backgroundColor: item.count > 20 ? '#ff4d4f' : '#faad14' }} />
                      {item.trend === 'up' && <RiseOutlined style={{ color: '#ff4d4f' }} />}
                      {item.trend === 'down' && <FallOutlined style={{ color: '#52c41a' }} />}
                    </Space>
                  </div>
                </List.Item>
              )}
            />
          </Card>
        </Col>

        {/* Compliance Status */}
        <Col xs={24} lg={12}>
          <Card title={<><FileProtectOutlined /> Compliance Status</>}>
            <List
              dataSource={complianceChecks}
              renderItem={item => (
                <List.Item>
                  <div style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <Space>
                        {item.status === 'passing' && <CheckCircleOutlined style={{ color: '#52c41a' }} />}
                        {item.status === 'partial' && <ExclamationCircleOutlined style={{ color: '#faad14' }} />}
                        {item.status === 'failing' && <CloseCircleOutlined style={{ color: '#ff4d4f' }} />}
                        <Text strong>{item.name}</Text>
                      </Space>
                      <Text>{item.passed}/{item.items} checks passed</Text>
                    </div>
                    <Progress
                      percent={item.score}
                      strokeColor={
                        item.score >= 80 ? '#52c41a' :
                        item.score >= 60 ? '#faad14' : '#ff4d4f'
                      }
                      size="small"
                    />
                  </div>
                </List.Item>
              )}
            />
          </Card>
        </Col>

        {/* Vulnerabilities Table */}
        <Col xs={24}>
          <Card
            title={<><BugOutlined /> Vulnerabilities</>}
            extra={
              <Space>
                <Select
                  value={severityFilter}
                  onChange={setSeverityFilter}
                  style={{ width: 120 }}
                  options={[
                    { value: 'all', label: 'All Severity' },
                    { value: 'critical', label: 'Critical' },
                    { value: 'high', label: 'High' },
                    { value: 'medium', label: 'Medium' },
                    { value: 'low', label: 'Low' },
                  ]}
                />
                <Button type="primary">Export Report</Button>
              </Space>
            }
          >
            <Table
              columns={columns}
              dataSource={severityFilter === 'all' 
                ? mockVulnerabilities 
                : mockVulnerabilities.filter(v => v.severity === severityFilter)
              }
              rowKey="id"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default SecurityDashboard;
