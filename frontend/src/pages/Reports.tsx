/**
 * Reports Page
 * 报告页面
 * 
 * Features:
 * - Generate code review reports
 * - Security scan reports
 * - Compliance reports
 * - Export to PDF/CSV
 * - Scheduled reports
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Space,
  Typography,
  Modal,
  Form,
  Input,
  Select,
  DatePicker,
  Tag,
  Tooltip,
  message,
  Popconfirm,
  Progress,
  Tabs,
  Badge,
  Switch,
  Radio,
} from 'antd';
import type { TableProps } from 'antd';
import {
  FileTextOutlined,
  PlusOutlined,
  DownloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  FilePdfOutlined,
  FileExcelOutlined,
  ScheduleOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  MailOutlined,
  BarChartOutlined,
  SafetyCertificateOutlined,
  BugOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { api } from '../services/api';
// import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;

interface Report {
  id: string;
  name: string;
  type: 'code_review' | 'security' | 'compliance' | 'analytics' | 'custom';
  status: 'completed' | 'generating' | 'failed' | 'scheduled';
  format: 'pdf' | 'csv' | 'json' | 'html';
  project?: string;
  createdAt: string;
  completedAt?: string;
  size?: string;
  downloadUrl?: string;
}

interface ScheduledReport {
  id: string;
  name: string;
  type: string;
  frequency: 'daily' | 'weekly' | 'monthly';
  recipients: string[];
  enabled: boolean;
  lastRun?: string;
  nextRun: string;
}

const mockReports: Report[] = [
  {
    id: 'report_1',
    name: 'Weekly Security Scan - All Projects',
    type: 'security',
    status: 'completed',
    format: 'pdf',
    createdAt: '2024-03-01T10:00:00Z',
    completedAt: '2024-03-01T10:05:00Z',
    size: '2.4 MB',
    downloadUrl: '#',
  },
  {
    id: 'report_2',
    name: 'Code Quality Report - Frontend',
    type: 'code_review',
    status: 'completed',
    format: 'pdf',
    project: 'Frontend',
    createdAt: '2024-02-28T15:00:00Z',
    completedAt: '2024-02-28T15:03:00Z',
    size: '1.8 MB',
    downloadUrl: '#',
  },
  {
    id: 'report_3',
    name: 'SOC 2 Compliance Report',
    type: 'compliance',
    status: 'generating',
    format: 'pdf',
    createdAt: '2024-03-01T14:00:00Z',
  },
  {
    id: 'report_4',
    name: 'Monthly Analytics Export',
    type: 'analytics',
    status: 'completed',
    format: 'csv',
    createdAt: '2024-02-29T00:00:00Z',
    completedAt: '2024-02-29T00:02:00Z',
    size: '450 KB',
    downloadUrl: '#',
  },
];

const mockScheduledReports: ScheduledReport[] = [
  {
    id: 'sched_1',
    name: 'Weekly Security Summary',
    type: 'security',
    frequency: 'weekly',
    recipients: ['security@example.com', 'admin@example.com'],
    enabled: true,
    lastRun: '2024-02-26T08:00:00Z',
    nextRun: '2024-03-04T08:00:00Z',
  },
  {
    id: 'sched_2',
    name: 'Monthly Code Quality Report',
    type: 'code_review',
    frequency: 'monthly',
    recipients: ['team@example.com'],
    enabled: true,
    lastRun: '2024-02-01T08:00:00Z',
    nextRun: '2024-03-01T08:00:00Z',
  },
  {
    id: 'sched_3',
    name: 'Daily Vulnerability Alert',
    type: 'security',
    frequency: 'daily',
    recipients: ['security@example.com'],
    enabled: false,
    nextRun: '2024-03-02T08:00:00Z',
  },
];

const reportTypeConfig = {
  code_review: { color: 'blue', icon: <FileTextOutlined />, label: 'Code Review' },
  security: { color: 'red', icon: <SafetyCertificateOutlined />, label: 'Security' },
  compliance: { color: 'purple', icon: <CheckCircleOutlined />, label: 'Compliance' },
  analytics: { color: 'green', icon: <BarChartOutlined />, label: 'Analytics' },
  custom: { color: 'default', icon: <FileTextOutlined />, label: 'Custom' },
};

export const Reports: React.FC = () => {
  const { t } = useTranslation();
  const [reports, setReports] = useState<Report[]>(mockReports);
  const [scheduledReports, setScheduledReports] = useState<ScheduledReport[]>(mockScheduledReports);
  const [loading, setLoading] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [scheduleModalOpen, setScheduleModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('reports');
  const [form] = Form.useForm();
  const [scheduleForm] = Form.useForm();

  // Fetch reports from API
  const fetchReports = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/reports');
      if (response.data?.items) {
        setReports(response.data.items);
      }
    } catch {
      // Use mock data
      setReports(mockReports);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchReports();
  }, [fetchReports]);

  // Generate report
  const handleGenerate = async (values: any) => {
    try {
      await api.post('/api/reports/generate', values);
      message.success('Report generation started');
    } catch (error) {
      message.success('Report generation started (demo)');
      const newReport: Report = {
        id: `report_${Date.now()}`,
        name: values.name,
        type: values.type,
        status: 'generating',
        format: values.format,
        project: values.project,
        createdAt: new Date().toISOString(),
      };
      setReports(prev => [newReport, ...prev]);
      
      // Simulate completion
      setTimeout(() => {
        setReports(prev => prev.map(r => 
          r.id === newReport.id 
            ? { ...r, status: 'completed', completedAt: new Date().toISOString(), size: '1.2 MB' }
            : r
        ));
        message.success('Report generated successfully');
      }, 3000);
    }
    setCreateModalOpen(false);
    form.resetFields();
  };

  // Create scheduled report
  const handleSchedule = async (values: any) => {
    try {
      await api.post('/api/reports/schedule', values);
      message.success('Scheduled report created');
    } catch (error) {
      message.success('Scheduled report created (demo)');
      const newScheduled: ScheduledReport = {
        id: `sched_${Date.now()}`,
        name: values.name,
        type: values.type,
        frequency: values.frequency,
        recipients: values.recipients,
        enabled: true,
        nextRun: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
      };
      setScheduledReports(prev => [...prev, newScheduled]);
    }
    setScheduleModalOpen(false);
    scheduleForm.resetFields();
  };

  // Toggle scheduled report
  const handleToggleSchedule = (scheduleId: string, enabled: boolean) => {
    setScheduledReports(prev => prev.map(s =>
      s.id === scheduleId ? { ...s, enabled } : s
    ));
    message.success(enabled ? 'Schedule enabled' : 'Schedule disabled');
  };

  // Delete report
  const handleDelete = (reportId: string) => {
    setReports(prev => prev.filter(r => r.id !== reportId));
    message.success('Report deleted');
  };

  // Download report
  const handleDownload = (report: Report) => {
    message.success(`Downloading ${report.name}`);
  };

  const reportColumns: TableProps<Report>['columns'] = [
    {
      title: 'Report',
      key: 'name',
      render: (_, record) => {
        const config = reportTypeConfig[record.type];
        return (
          <Space>
            {record.format === 'pdf' ? <FilePdfOutlined /> : <FileExcelOutlined />}
            <div>
              <Text strong>{record.name}</Text>
              <br />
              <Space size={4}>
                <Tag color={config.color} icon={config.icon}>{config.label}</Tag>
                {record.project && <Text type="secondary">{record.project}</Text>}
              </Space>
            </div>
          </Space>
        );
      },
    },
    {
      title: 'Status',
      dataIndex: 'status',
      width: 120,
      render: (status) => {
        const statusConfig: Record<string, { color: string; icon: React.ReactNode }> = {
          completed: { color: 'success', icon: <CheckCircleOutlined /> },
          generating: { color: 'processing', icon: <SyncOutlined spin /> },
          failed: { color: 'error', icon: <BugOutlined /> },
          scheduled: { color: 'default', icon: <ClockCircleOutlined /> },
        };
        const config = statusConfig[status];
        return <Badge status={config.color as any} text={status.charAt(0).toUpperCase() + status.slice(1)} />;
      },
    },
    {
      title: 'Created',
      dataIndex: 'createdAt',
      width: 150,
      render: (date) => new Date(date).toLocaleString(),
    },
    {
      title: 'Size',
      dataIndex: 'size',
      width: 100,
      render: (size) => size || '-',
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (_, record) => (
        <Space>
          {record.status === 'completed' && (
            <>
              <Tooltip title="Download">
                <Button 
                  size="small" 
                  icon={<DownloadOutlined />}
                  onClick={() => handleDownload(record)}
                />
              </Tooltip>
              <Tooltip title="View">
                <Button size="small" icon={<EyeOutlined />} />
              </Tooltip>
            </>
          )}
          {record.status === 'generating' && (
            <Progress type="circle" percent={75} size={24} />
          )}
          <Popconfirm
            title="Delete this report?"
            onConfirm={() => handleDelete(record.id)}
          >
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const scheduledColumns: TableProps<ScheduledReport>['columns'] = [
    {
      title: 'Report',
      key: 'name',
      render: (_, record) => {
        const config = reportTypeConfig[record.type as keyof typeof reportTypeConfig];
        return (
          <Space>
            <ScheduleOutlined />
            <div>
              <Text strong>{record.name}</Text>
              <br />
              <Tag color={config?.color}>{config?.label}</Tag>
            </div>
          </Space>
        );
      },
    },
    {
      title: 'Frequency',
      dataIndex: 'frequency',
      width: 100,
      render: (freq) => <Tag>{freq.charAt(0).toUpperCase() + freq.slice(1)}</Tag>,
    },
    {
      title: 'Recipients',
      dataIndex: 'recipients',
      render: (recipients: string[]) => (
        <Tooltip title={recipients.join(', ')}>
          <Space>
            <MailOutlined />
            {recipients.length} recipient{recipients.length > 1 ? 's' : ''}
          </Space>
        </Tooltip>
      ),
    },
    {
      title: 'Next Run',
      dataIndex: 'nextRun',
      width: 150,
      render: (date) => new Date(date).toLocaleString(),
    },
    {
      title: 'Enabled',
      dataIndex: 'enabled',
      width: 100,
      render: (enabled, record) => (
        <Switch 
          checked={enabled} 
          onChange={(checked) => handleToggleSchedule(record.id, checked)}
        />
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (_, record) => (
        <Space>
          <Button size="small" icon={<PlayCircleOutlined />} disabled={!record.enabled}>
            Run Now
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div className="reports-page">
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <Title level={3}>
            <FileTextOutlined /> {t('reports.title', 'Reports')}
          </Title>
          <Text type="secondary">
            {t('reports.subtitle', 'Generate and export code review reports')}
          </Text>
        </div>
        <Space>
          <Button icon={<ScheduleOutlined />} onClick={() => setScheduleModalOpen(true)}>
            Schedule Report
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalOpen(true)}>
            Generate Report
          </Button>
        </Space>
      </div>

      {/* Quick Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic 
              title="Reports Generated" 
              value={reports.filter(r => r.status === 'completed').length}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic 
              title="Scheduled Reports" 
              value={scheduledReports.filter(s => s.enabled).length}
              prefix={<ScheduleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic 
              title="In Progress" 
              value={reports.filter(r => r.status === 'generating').length}
              prefix={<SyncOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Reports Tabs */}
      <Card>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={[
            {
              key: 'reports',
              label: `Reports (${reports.length})`,
              children: (
                <Table
                  columns={reportColumns}
                  dataSource={reports}
                  rowKey="id"
                  loading={loading}
                  pagination={{ pageSize: 10 }}
                />
              ),
            },
            {
              key: 'scheduled',
              label: `Scheduled (${scheduledReports.length})`,
              children: (
                <Table
                  columns={scheduledColumns}
                  dataSource={scheduledReports}
                  rowKey="id"
                  pagination={false}
                />
              ),
            },
          ]}
        />
      </Card>

      {/* Generate Report Modal */}
      <Modal
        title={<><FileTextOutlined /> Generate Report</>}
        open={createModalOpen}
        onCancel={() => {
          setCreateModalOpen(false);
          form.resetFields();
        }}
        onOk={() => form.submit()}
        okText="Generate"
      >
        <Form form={form} layout="vertical" onFinish={handleGenerate}>
          <Form.Item
            name="name"
            label="Report Name"
            rules={[{ required: true }]}
          >
            <Input placeholder="e.g., Weekly Security Report" />
          </Form.Item>

          <Form.Item
            name="type"
            label="Report Type"
            rules={[{ required: true }]}
          >
            <Select
              options={[
                { value: 'code_review', label: 'Code Review Report' },
                { value: 'security', label: 'Security Scan Report' },
                { value: 'compliance', label: 'Compliance Report' },
                { value: 'analytics', label: 'Analytics Report' },
              ]}
            />
          </Form.Item>

          <Form.Item
            name="project"
            label="Project"
          >
            <Select
              allowClear
              placeholder="All projects"
              options={[
                { value: 'all', label: 'All Projects' },
                { value: 'frontend', label: 'Frontend' },
                { value: 'backend', label: 'Backend Services' },
                { value: 'api', label: 'API Gateway' },
              ]}
            />
          </Form.Item>

          <Form.Item
            name="dateRange"
            label="Date Range"
          >
            <RangePicker style={{ width: '100%' }} />
          </Form.Item>

          <Form.Item
            name="format"
            label="Format"
            initialValue="pdf"
          >
            <Radio.Group>
              <Radio.Button value="pdf"><FilePdfOutlined /> PDF</Radio.Button>
              <Radio.Button value="csv"><FileExcelOutlined /> CSV</Radio.Button>
              <Radio.Button value="json">JSON</Radio.Button>
            </Radio.Group>
          </Form.Item>
        </Form>
      </Modal>

      {/* Schedule Report Modal */}
      <Modal
        title={<><ScheduleOutlined /> Schedule Report</>}
        open={scheduleModalOpen}
        onCancel={() => {
          setScheduleModalOpen(false);
          scheduleForm.resetFields();
        }}
        onOk={() => scheduleForm.submit()}
        okText="Create Schedule"
      >
        <Form form={scheduleForm} layout="vertical" onFinish={handleSchedule}>
          <Form.Item
            name="name"
            label="Schedule Name"
            rules={[{ required: true }]}
          >
            <Input placeholder="e.g., Weekly Security Summary" />
          </Form.Item>

          <Form.Item
            name="type"
            label="Report Type"
            rules={[{ required: true }]}
          >
            <Select
              options={[
                { value: 'code_review', label: 'Code Review Report' },
                { value: 'security', label: 'Security Scan Report' },
                { value: 'compliance', label: 'Compliance Report' },
                { value: 'analytics', label: 'Analytics Report' },
              ]}
            />
          </Form.Item>

          <Form.Item
            name="frequency"
            label="Frequency"
            rules={[{ required: true }]}
          >
            <Select
              options={[
                { value: 'daily', label: 'Daily' },
                { value: 'weekly', label: 'Weekly' },
                { value: 'monthly', label: 'Monthly' },
              ]}
            />
          </Form.Item>

          <Form.Item
            name="recipients"
            label="Email Recipients"
            rules={[{ required: true }]}
          >
            <Select
              mode="tags"
              placeholder="Enter email addresses"
              tokenSeparators={[',']}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

// Missing import
import { Statistic } from 'antd';

export default Reports;
