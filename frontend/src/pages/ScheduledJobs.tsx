/**
 * Scheduled Jobs Page
 * 计划任务页面
 * 
 * Features:
 * - View scheduled analysis jobs
 * - Configure job schedules
 * - Job execution history
 * - Enable/disable jobs
 */

import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Table,
  Tag,
  Switch,
  Modal,
  Form,
  Input,
  Select,
  Statistic,
  Badge,
  Tooltip,
  message,
} from 'antd';
import type { TableProps } from 'antd';
import {
  ClockCircleOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  HistoryOutlined,
  ScheduleOutlined,
  SafetyCertificateOutlined,
  CalendarOutlined,
  CodeOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text } = Typography;

interface ScheduledJob {
  id: string;
  name: string;
  type: 'analysis' | 'security' | 'cleanup' | 'backup' | 'report';
  schedule: string;
  nextRun: string;
  lastRun?: string;
  lastStatus?: 'success' | 'failed' | 'running';
  enabled: boolean;
  project?: string;
  duration?: number;
}

const mockJobs: ScheduledJob[] = [
  {
    id: 'job_1',
    name: 'Daily Security Scan',
    type: 'security',
    schedule: '0 2 * * *',
    nextRun: new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString(),
    lastRun: new Date(Date.now() - 16 * 60 * 60 * 1000).toISOString(),
    lastStatus: 'success',
    enabled: true,
    project: 'All Projects',
    duration: 1250,
  },
  {
    id: 'job_2',
    name: 'Code Analysis - Backend',
    type: 'analysis',
    schedule: '0 */6 * * *',
    nextRun: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(),
    lastRun: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
    lastStatus: 'success',
    enabled: true,
    project: 'Backend Services',
    duration: 845,
  },
  {
    id: 'job_3',
    name: 'Weekly Full Scan',
    type: 'analysis',
    schedule: '0 0 * * 0',
    nextRun: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString(),
    lastRun: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
    lastStatus: 'success',
    enabled: true,
    project: 'All Projects',
    duration: 5420,
  },
  {
    id: 'job_4',
    name: 'Database Backup',
    type: 'backup',
    schedule: '0 3 * * *',
    nextRun: new Date(Date.now() + 9 * 60 * 60 * 1000).toISOString(),
    lastRun: new Date(Date.now() - 15 * 60 * 60 * 1000).toISOString(),
    lastStatus: 'success',
    enabled: true,
    duration: 320,
  },
  {
    id: 'job_5',
    name: 'Auto-Fix Vulnerabilities',
    type: 'security',
    schedule: '0 4 * * *',
    nextRun: new Date(Date.now() + 10 * 60 * 60 * 1000).toISOString(),
    lastRun: new Date(Date.now() - 14 * 60 * 60 * 1000).toISOString(),
    lastStatus: 'failed',
    enabled: true,
    project: 'All Projects',
    duration: 2100,
  },
  {
    id: 'job_6',
    name: 'Cleanup Old Sessions',
    type: 'cleanup',
    schedule: '0 5 * * *',
    nextRun: new Date(Date.now() + 11 * 60 * 60 * 1000).toISOString(),
    lastRun: new Date(Date.now() - 13 * 60 * 60 * 1000).toISOString(),
    lastStatus: 'success',
    enabled: false,
    duration: 45,
  },
];

const typeConfig = {
  analysis: { icon: <CodeOutlined />, color: '#3b82f6', label: 'Analysis' },
  security: { icon: <SafetyCertificateOutlined />, color: '#ef4444', label: 'Security' },
  cleanup: { icon: <DeleteOutlined />, color: '#64748b', label: 'Cleanup' },
  backup: { icon: <HistoryOutlined />, color: '#22c55e', label: 'Backup' },
  report: { icon: <CalendarOutlined />, color: '#8b5cf6', label: 'Report' },
};

const parseCron = (cron: string): string => {
  const parts = cron.split(' ');
  if (parts[1] === '*' && parts[2] === '*') return `Every hour at :${parts[0].padStart(2, '0')}`;
  if (parts[1].startsWith('*/')) return `Every ${parts[1].replace('*/', '')} hours`;
  if (parts[4] === '0') return `Weekly on Sunday at ${parts[1]}:${parts[0].padStart(2, '0')}`;
  if (parts[2] === '*') return `Daily at ${parts[1]}:${parts[0].padStart(2, '0')}`;
  return cron;
};

export const ScheduledJobs: React.FC = () => {
  const { t } = useTranslation();
  const [jobs, setJobs] = useState<ScheduledJob[]>(mockJobs);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [form] = Form.useForm();

  const handleToggle = (jobId: string, enabled: boolean) => {
    setJobs(prev => prev.map(j => j.id === jobId ? { ...j, enabled } : j));
    message.success(enabled ? 'Job enabled' : 'Job disabled');
  };

  const handleRunNow = (job: ScheduledJob) => {
    message.loading(`Running ${job.name}...`);
    setTimeout(() => {
      message.success(`${job.name} completed successfully`);
    }, 2000);
  };

  const stats = {
    total: jobs.length,
    active: jobs.filter(j => j.enabled).length,
    failed: jobs.filter(j => j.lastStatus === 'failed').length,
  };

  const columns: TableProps<ScheduledJob>['columns'] = [
    {
      title: 'Job',
      key: 'job',
      render: (_, record) => {
        const config = typeConfig[record.type];
        return (
          <div>
            <Space>
              <Badge status={record.enabled ? 'success' : 'default'} />
              <span style={{ color: config.color }}>{config.icon}</span>
              <Text strong>{record.name}</Text>
            </Space>
            {record.project && (
              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>{record.project}</Text>
              </div>
            )}
          </div>
        );
      },
    },
    {
      title: 'Schedule',
      key: 'schedule',
      width: 200,
      render: (_, record) => (
        <div>
          <Tag icon={<ClockCircleOutlined />}>{parseCron(record.schedule)}</Tag>
          <div style={{ marginTop: 4 }}>
            <Text type="secondary" style={{ fontSize: 11 }}>{record.schedule}</Text>
          </div>
        </div>
      ),
    },
    {
      title: 'Next Run',
      dataIndex: 'nextRun',
      width: 180,
      render: (nextRun: string) => (
        <Space direction="vertical" size={0}>
          <Text>{new Date(nextRun).toLocaleDateString()}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {new Date(nextRun).toLocaleTimeString()}
          </Text>
        </Space>
      ),
    },
    {
      title: 'Last Run',
      key: 'lastRun',
      width: 150,
      render: (_, record) => (
        record.lastRun ? (
          <Space>
            {record.lastStatus === 'success' ? (
              <CheckCircleOutlined style={{ color: '#22c55e' }} />
            ) : record.lastStatus === 'failed' ? (
              <CloseCircleOutlined style={{ color: '#ef4444' }} />
            ) : (
              <SyncOutlined spin style={{ color: '#3b82f6' }} />
            )}
            <Text type="secondary" style={{ fontSize: 12 }}>
              {new Date(record.lastRun).toLocaleString()}
            </Text>
          </Space>
        ) : (
          <Text type="secondary">Never</Text>
        )
      ),
    },
    {
      title: 'Duration',
      dataIndex: 'duration',
      width: 100,
      render: (duration?: number) => duration ? `${(duration / 1000).toFixed(1)}s` : '-',
    },
    {
      title: 'Enabled',
      dataIndex: 'enabled',
      width: 80,
      render: (enabled: boolean, record) => (
        <Switch
          checked={enabled}
          size="small"
          onChange={(checked) => handleToggle(record.id, checked)}
        />
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 140,
      render: (_, record) => (
        <Space>
          <Tooltip title="Run Now">
            <Button
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => handleRunNow(record)}
              disabled={!record.enabled}
            />
          </Tooltip>
          <Tooltip title="Edit">
            <Button size="small" icon={<EditOutlined />} />
          </Tooltip>
          <Tooltip title="Delete">
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Tooltip>
        </Space>
      ),
    },
  ];

  return (
    <div className="scheduled-jobs-page" style={{ maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <ScheduleOutlined style={{ color: '#2563eb' }} /> Scheduled Jobs
          </Title>
          <Text type="secondary">Automate code analysis and maintenance tasks</Text>
        </div>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalOpen(true)}>
          Create Job
        </Button>
      </div>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={8}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Total Jobs" value={stats.total} prefix={<ScheduleOutlined />} />
          </Card>
        </Col>
        <Col xs={8}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Active" value={stats.active} valueStyle={{ color: '#22c55e' }} prefix={<CheckCircleOutlined />} />
          </Card>
        </Col>
        <Col xs={8}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Failed (Last Run)" value={stats.failed} valueStyle={{ color: '#ef4444' }} prefix={<CloseCircleOutlined />} />
          </Card>
        </Col>
      </Row>

      {/* Jobs Table */}
      <Card title={<><ClockCircleOutlined /> All Scheduled Jobs</>} style={{ borderRadius: 12 }}>
        <Table columns={columns} dataSource={jobs} rowKey="id" pagination={false} />
      </Card>

      {/* Create Modal */}
      <Modal
        title={<><PlusOutlined /> Create Scheduled Job</>}
        open={createModalOpen}
        onCancel={() => setCreateModalOpen(false)}
        onOk={() => form.submit()}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="Job Name" rules={[{ required: true }]}>
            <Input placeholder="e.g., Daily Security Scan" />
          </Form.Item>
          <Form.Item name="type" label="Job Type" rules={[{ required: true }]}>
            <Select
              options={Object.entries(typeConfig).map(([key, config]) => ({
                value: key,
                label: <Space>{config.icon} {config.label}</Space>,
              }))}
            />
          </Form.Item>
          <Form.Item name="project" label="Target Project">
            <Select
              placeholder="Select project"
              options={[
                { value: 'all', label: 'All Projects' },
                { value: 'backend', label: 'Backend Services' },
                { value: 'frontend', label: 'Frontend' },
              ]}
            />
          </Form.Item>
          <Form.Item name="schedule" label="Schedule (Cron)" rules={[{ required: true }]}>
            <Input placeholder="0 2 * * * (daily at 2:00 AM)" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ScheduledJobs;
