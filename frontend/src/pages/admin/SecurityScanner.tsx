/**
 * AI Security Scanner Dashboard
 * 
 * Automated security vulnerability detection and remediation:
 * - Real-time scanning status
 * - Vulnerability categorization (OWASP Top 10)
 * - Auto-fix suggestions with AI confidence
 * - Compliance tracking (SOC2, GDPR, HIPAA)
 */

import React, { useState, useEffect } from 'react';
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
  Button,
  Alert,
  Badge,
  Timeline,
  Tabs,
  Switch,
  Tooltip,
  Modal,
} from 'antd';
import {
  SafetyCertificateOutlined,
  BugOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  ThunderboltOutlined,
  EyeOutlined,
  SecurityScanOutlined,
  FileProtectOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

interface Vulnerability {
  id: string;
  title: string;
  category: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  cweId: string;
  file: string;
  line: number;
  description: string;
  autoFixAvailable: boolean;
  aiConfidence: number;
  status: 'open' | 'fixing' | 'fixed' | 'ignored';
  discoveredAt: string;
}

interface ComplianceStatus {
  framework: string;
  status: 'compliant' | 'partial' | 'non-compliant';
  score: number;
  issues: number;
}

const SecurityScanner: React.FC = () => {
  const { t } = useTranslation();
  const [scanning, setScanning] = useState(false);
  const [autoFixEnabled, setAutoFixEnabled] = useState(true);
  const [vulnerabilities, setVulnerabilities] = useState<Vulnerability[]>([]);
  const [selectedVuln, setSelectedVuln] = useState<Vulnerability | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);

  // Mock data
  useEffect(() => {
    setVulnerabilities([
      {
        id: 'vuln-001',
        title: 'SQL Injection Vulnerability',
        category: 'A03:2021-Injection',
        severity: 'critical',
        cweId: 'CWE-89',
        file: 'src/api/users.py',
        line: 145,
        description: 'User input directly concatenated into SQL query without sanitization.',
        autoFixAvailable: true,
        aiConfidence: 0.96,
        status: 'open',
        discoveredAt: new Date(Date.now() - 3600000).toISOString(),
      },
      {
        id: 'vuln-002',
        title: 'Cross-Site Scripting (XSS)',
        category: 'A03:2021-Injection',
        severity: 'high',
        cweId: 'CWE-79',
        file: 'src/components/Comment.tsx',
        line: 67,
        description: 'User-provided HTML rendered without escaping.',
        autoFixAvailable: true,
        aiConfidence: 0.92,
        status: 'fixing',
        discoveredAt: new Date(Date.now() - 7200000).toISOString(),
      },
      {
        id: 'vuln-003',
        title: 'Insecure Direct Object Reference',
        category: 'A01:2021-Broken Access Control',
        severity: 'high',
        cweId: 'CWE-639',
        file: 'src/api/documents.py',
        line: 89,
        description: 'Document access not properly authorized based on user permissions.',
        autoFixAvailable: true,
        aiConfidence: 0.88,
        status: 'open',
        discoveredAt: new Date(Date.now() - 14400000).toISOString(),
      },
      {
        id: 'vuln-004',
        title: 'Hardcoded Credentials',
        category: 'A07:2021-Identification Failures',
        severity: 'critical',
        cweId: 'CWE-798',
        file: 'src/config/database.py',
        line: 23,
        description: 'Database credentials hardcoded in configuration file.',
        autoFixAvailable: true,
        aiConfidence: 0.99,
        status: 'fixed',
        discoveredAt: new Date(Date.now() - 86400000).toISOString(),
      },
      {
        id: 'vuln-005',
        title: 'Missing Rate Limiting',
        category: 'A04:2021-Insecure Design',
        severity: 'medium',
        cweId: 'CWE-770',
        file: 'src/api/auth.py',
        line: 34,
        description: 'Login endpoint lacks rate limiting, vulnerable to brute force.',
        autoFixAvailable: false,
        aiConfidence: 0.75,
        status: 'open',
        discoveredAt: new Date(Date.now() - 172800000).toISOString(),
      },
      {
        id: 'vuln-006',
        title: 'Outdated Dependency',
        category: 'A06:2021-Vulnerable Components',
        severity: 'medium',
        cweId: 'CWE-1104',
        file: 'package.json',
        line: 15,
        description: 'lodash@4.17.20 has known prototype pollution vulnerability.',
        autoFixAvailable: true,
        aiConfidence: 0.95,
        status: 'open',
        discoveredAt: new Date(Date.now() - 259200000).toISOString(),
      },
    ]);
  }, []);

  const complianceData: ComplianceStatus[] = [
    { framework: 'SOC 2 Type II', status: 'compliant', score: 94, issues: 3 },
    { framework: 'GDPR', status: 'partial', score: 87, issues: 8 },
    { framework: 'HIPAA', status: 'partial', score: 82, issues: 12 },
    { framework: 'PCI DSS', status: 'compliant', score: 96, issues: 2 },
  ];

  const severityColors: Record<string, string> = {
    critical: '#ff4d4f',
    high: '#fa8c16',
    medium: '#faad14',
    low: '#52c41a',
  };

  const severityData = [
    { name: 'Critical', value: vulnerabilities.filter(v => v.severity === 'critical').length, color: severityColors.critical },
    { name: 'High', value: vulnerabilities.filter(v => v.severity === 'high').length, color: severityColors.high },
    { name: 'Medium', value: vulnerabilities.filter(v => v.severity === 'medium').length, color: severityColors.medium },
    { name: 'Low', value: vulnerabilities.filter(v => v.severity === 'low').length, color: severityColors.low },
  ];

  const categoryData = [
    { name: 'Injection', count: 2 },
    { name: 'Access Control', count: 1 },
    { name: 'Auth Failures', count: 1 },
    { name: 'Insecure Design', count: 1 },
    { name: 'Vuln Components', count: 1 },
  ];

  const handleScan = () => {
    setScanning(true);
    setTimeout(() => setScanning(false), 3000);
  };

  const handleAutoFix = (vuln: Vulnerability) => {
    setVulnerabilities(prev =>
      prev.map(v => v.id === vuln.id ? { ...v, status: 'fixing' } : v)
    );
    setTimeout(() => {
      setVulnerabilities(prev =>
        prev.map(v => v.id === vuln.id ? { ...v, status: 'fixed' } : v)
      );
    }, 2000);
  };

  const vulnColumns = [
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: string) => (
        <Tag color={severityColors[severity]}>{severity.toUpperCase()}</Tag>
      ),
      sorter: (a: Vulnerability, b: Vulnerability) => {
        const order = { critical: 0, high: 1, medium: 2, low: 3 };
        return order[a.severity] - order[b.severity];
      },
    },
    {
      title: 'Vulnerability',
      dataIndex: 'title',
      key: 'title',
      render: (title: string, record: Vulnerability) => (
        <Space direction="vertical" size={0}>
          <Text strong>{title}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>{record.cweId}</Text>
        </Space>
      ),
    },
    {
      title: 'Category',
      dataIndex: 'category',
      key: 'category',
      render: (cat: string) => <Tag>{cat}</Tag>,
    },
    {
      title: 'Location',
      key: 'location',
      render: (_: any, record: Vulnerability) => (
        <Text code>{record.file}:{record.line}</Text>
      ),
    },
    {
      title: 'AI Fix',
      key: 'autoFix',
      width: 120,
      render: (_: any, record: Vulnerability) => (
        record.autoFixAvailable ? (
          <Tooltip title={`AI Confidence: ${(record.aiConfidence * 100).toFixed(0)}%`}>
            <Tag color="blue" icon={<ThunderboltOutlined />}>
              {(record.aiConfidence * 100).toFixed(0)}%
            </Tag>
          </Tooltip>
        ) : (
          <Tag>Manual</Tag>
        )
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => {
        const config: Record<string, { color: string; icon: React.ReactNode }> = {
          open: { color: 'red', icon: <ExclamationCircleOutlined /> },
          fixing: { color: 'blue', icon: <SyncOutlined spin /> },
          fixed: { color: 'green', icon: <CheckCircleOutlined /> },
          ignored: { color: 'default', icon: <CloseCircleOutlined /> },
        };
        return (
          <Tag color={config[status]?.color} icon={config[status]?.icon}>
            {status.toUpperCase()}
          </Tag>
        );
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (_: any, record: Vulnerability) => (
        <Space>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedVuln(record);
              setDetailModalOpen(true);
            }}
          />
          {record.autoFixAvailable && record.status === 'open' && (
            <Button
              size="small"
              type="primary"
              icon={<ThunderboltOutlined />}
              onClick={() => handleAutoFix(record)}
            >
              Fix
            </Button>
          )}
        </Space>
      ),
    },
  ];

  const openVulns = vulnerabilities.filter(v => v.status === 'open').length;
  const criticalVulns = vulnerabilities.filter(v => v.severity === 'critical' && v.status !== 'fixed').length;
  const fixedVulns = vulnerabilities.filter(v => v.status === 'fixed').length;
  const autoFixable = vulnerabilities.filter(v => v.autoFixAvailable && v.status === 'open').length;

  return (
    <div style={{ padding: 24 }}>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}>
            <SecurityScanOutlined style={{ marginRight: 8 }} />
            AI Security Scanner
          </Title>
        </Col>
        <Col>
          <Space>
            <span>Auto-Fix:</span>
            <Switch checked={autoFixEnabled} onChange={setAutoFixEnabled} />
            <Button
              type="primary"
              icon={<SyncOutlined spin={scanning} />}
              onClick={handleScan}
              loading={scanning}
            >
              {scanning ? 'Scanning...' : 'Run Scan'}
            </Button>
          </Space>
        </Col>
      </Row>

      {/* Critical Alert */}
      {criticalVulns > 0 && (
        <Alert
          message={`${criticalVulns} Critical Vulnerabilities Detected`}
          description="Immediate attention required. AI auto-fix is available for most issues."
          type="error"
          showIcon
          icon={<BugOutlined />}
          style={{ marginBottom: 24 }}
          action={
            <Button size="small" danger onClick={() => {}}>
              Fix All Critical
            </Button>
          }
        />
      )}

      {/* Overview Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Open Vulnerabilities"
              value={openVulns}
              valueStyle={{ color: openVulns > 0 ? '#cf1322' : '#3f8600' }}
              prefix={<BugOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Critical Issues"
              value={criticalVulns}
              valueStyle={{ color: criticalVulns > 0 ? '#cf1322' : '#3f8600' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Fixed This Week"
              value={fixedVulns}
              valueStyle={{ color: '#3f8600' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Auto-Fixable"
              value={autoFixable}
              valueStyle={{ color: '#1890ff' }}
              prefix={<ThunderboltOutlined />}
              suffix={
                <Button type="link" size="small" onClick={() => {}}>
                  Fix All
                </Button>
              }
            />
          </Card>
        </Col>
      </Row>

      {/* Charts Row */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} md={8}>
          <Card title="Severity Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={severityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }: any) => value > 0 ? `${name}: ${value}` : ''}
                >
                  {severityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col xs={24} md={16}>
          <Card title="Vulnerabilities by Category (OWASP)">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={categoryData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 12 }} />
                <RechartsTooltip />
                <Bar dataKey="count" fill="#1890ff" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      {/* Tabs for Vulnerabilities and Compliance */}
      <Card>
        <Tabs defaultActiveKey="vulnerabilities">
          <TabPane tab={<span><BugOutlined /> Vulnerabilities ({vulnerabilities.length})</span>} key="vulnerabilities">
            <Table
              columns={vulnColumns}
              dataSource={vulnerabilities}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              size="middle"
            />
          </TabPane>
          <TabPane tab={<span><FileProtectOutlined /> Compliance</span>} key="compliance">
            <Row gutter={[16, 16]}>
              {complianceData.map((comp) => (
                <Col xs={24} sm={12} md={6} key={comp.framework}>
                  <Card size="small">
                    <div style={{ textAlign: 'center', marginBottom: 16 }}>
                      <Progress
                        type="circle"
                        percent={comp.score}
                        status={comp.status === 'compliant' ? 'success' : comp.status === 'partial' ? 'normal' : 'exception'}
                        width={100}
                      />
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <Text strong>{comp.framework}</Text>
                      <br />
                      <Tag color={
                        comp.status === 'compliant' ? 'green' : 
                        comp.status === 'partial' ? 'orange' : 'red'
                      }>
                        {comp.status.toUpperCase()}
                      </Tag>
                      <br />
                      <Text type="secondary">{comp.issues} issues</Text>
                    </div>
                  </Card>
                </Col>
              ))}
            </Row>
          </TabPane>
          <TabPane tab={<span><SafetyCertificateOutlined /> Scan History</span>} key="history">
            <Timeline>
              <Timeline.Item color="green">
                <Text strong>Full Security Scan Completed</Text>
                <br />
                <Text type="secondary">Today at 3:45 PM - Found 6 issues, 4 auto-fixed</Text>
              </Timeline.Item>
              <Timeline.Item color="blue">
                <Text strong>Dependency Audit</Text>
                <br />
                <Text type="secondary">Today at 2:30 PM - 2 outdated packages identified</Text>
              </Timeline.Item>
              <Timeline.Item color="green">
                <Text strong>SAST Analysis</Text>
                <br />
                <Text type="secondary">Yesterday at 11:00 AM - No new issues</Text>
              </Timeline.Item>
              <Timeline.Item color="orange">
                <Text strong>Container Scan</Text>
                <br />
                <Text type="secondary">Yesterday at 9:15 AM - 1 medium severity CVE</Text>
              </Timeline.Item>
            </Timeline>
          </TabPane>
        </Tabs>
      </Card>

      {/* Detail Modal */}
      <Modal
        title={selectedVuln?.title}
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        footer={[
          <Button key="close" onClick={() => setDetailModalOpen(false)}>Close</Button>,
          selectedVuln?.autoFixAvailable && selectedVuln.status === 'open' && (
            <Button
              key="fix"
              type="primary"
              icon={<ThunderboltOutlined />}
              onClick={() => {
                if (selectedVuln) handleAutoFix(selectedVuln);
                setDetailModalOpen(false);
              }}
            >
              Apply AI Fix
            </Button>
          ),
        ]}
        width={700}
      >
        {selectedVuln && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Row gutter={16}>
              <Col span={12}>
                <Text type="secondary">Severity</Text>
                <br />
                <Tag color={severityColors[selectedVuln.severity]}>
                  {selectedVuln.severity.toUpperCase()}
                </Tag>
              </Col>
              <Col span={12}>
                <Text type="secondary">CWE ID</Text>
                <br />
                <Text strong>{selectedVuln.cweId}</Text>
              </Col>
            </Row>
            <div>
              <Text type="secondary">Category</Text>
              <br />
              <Tag>{selectedVuln.category}</Tag>
            </div>
            <div>
              <Text type="secondary">Location</Text>
              <br />
              <Text code>{selectedVuln.file}:{selectedVuln.line}</Text>
            </div>
            <div>
              <Text type="secondary">Description</Text>
              <Paragraph>{selectedVuln.description}</Paragraph>
            </div>
            {selectedVuln.autoFixAvailable && (
              <Alert
                message={`AI Auto-Fix Available (${(selectedVuln.aiConfidence * 100).toFixed(0)}% confidence)`}
                description="Our AI has analyzed this vulnerability and can automatically apply a secure fix."
                type="info"
                showIcon
                icon={<ThunderboltOutlined />}
              />
            )}
          </Space>
        )}
      </Modal>
    </div>
  );
};

export default SecurityScanner;
