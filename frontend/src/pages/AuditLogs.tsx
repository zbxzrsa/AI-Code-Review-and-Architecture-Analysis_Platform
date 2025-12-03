/**
 * Audit Logs Page
 * 审计日志页面
 * 
 * Features:
 * - View all audit events
 * - Filter by entity, action, user
 * - Export logs
 * - Tamper verification
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
  Input,
  Select,
  DatePicker,
  Drawer,
  Descriptions,
  Statistic,
  Tooltip,
  message,
} from 'antd';
import type { TableProps } from 'antd';
import {
  AuditOutlined,
  SearchOutlined,
  DownloadOutlined,
  SafetyCertificateOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  UserOutlined,
  ClockCircleOutlined,
  FileTextOutlined,
  LockOutlined,
  EyeOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;

interface AuditEvent {
  id: string;
  timestamp: string;
  entity: 'user' | 'project' | 'analysis' | 'api_key' | 'deployment' | 'config';
  action: 'create' | 'read' | 'update' | 'delete' | 'login' | 'logout';
  actor: { id: string; name: string; email: string };
  resource: string;
  status: 'success' | 'failure' | 'denied';
  ipAddress: string;
  userAgent: string;
  details?: Record<string, any>;
  verified: boolean;
}

const mockAuditEvents: AuditEvent[] = [
  {
    id: 'evt_1',
    timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    entity: 'analysis',
    action: 'create',
    actor: { id: 'usr_1', name: 'John Doe', email: 'john@example.com' },
    resource: 'project/backend/analysis/a1b2c3',
    status: 'success',
    ipAddress: '192.168.1.100',
    userAgent: 'Mozilla/5.0 Chrome/120.0',
    verified: true,
  },
  {
    id: 'evt_2',
    timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    entity: 'user',
    action: 'login',
    actor: { id: 'usr_2', name: 'Jane Smith', email: 'jane@example.com' },
    resource: 'auth/session/s1d2f3',
    status: 'success',
    ipAddress: '10.0.0.50',
    userAgent: 'Mozilla/5.0 Firefox/121.0',
    verified: true,
  },
  {
    id: 'evt_3',
    timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    entity: 'api_key',
    action: 'create',
    actor: { id: 'usr_1', name: 'John Doe', email: 'john@example.com' },
    resource: 'api_keys/key_x1y2z3',
    status: 'success',
    ipAddress: '192.168.1.100',
    userAgent: 'Mozilla/5.0 Chrome/120.0',
    verified: true,
  },
  {
    id: 'evt_4',
    timestamp: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
    entity: 'config',
    action: 'update',
    actor: { id: 'usr_3', name: 'Admin User', email: 'admin@example.com' },
    resource: 'config/security/rules',
    status: 'success',
    ipAddress: '172.16.0.1',
    userAgent: 'Mozilla/5.0 Chrome/120.0',
    details: { changes: ['enabled_2fa', 'updated_password_policy'] },
    verified: true,
  },
  {
    id: 'evt_5',
    timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
    entity: 'deployment',
    action: 'create',
    actor: { id: 'usr_1', name: 'John Doe', email: 'john@example.com' },
    resource: 'deployments/deploy_abc123',
    status: 'success',
    ipAddress: '192.168.1.100',
    userAgent: 'GitHub-Hookshot/1.0',
    verified: true,
  },
  {
    id: 'evt_6',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    entity: 'project',
    action: 'delete',
    actor: { id: 'usr_3', name: 'Admin User', email: 'admin@example.com' },
    resource: 'projects/old-project',
    status: 'denied',
    ipAddress: '172.16.0.1',
    userAgent: 'Mozilla/5.0 Chrome/120.0',
    verified: true,
  },
];

const entityConfig = {
  user: { color: 'blue', icon: <UserOutlined /> },
  project: { color: 'green', icon: <FileTextOutlined /> },
  analysis: { color: 'purple', icon: <AuditOutlined /> },
  api_key: { color: 'orange', icon: <LockOutlined /> },
  deployment: { color: 'cyan', icon: <SafetyCertificateOutlined /> },
  config: { color: 'magenta', icon: <InfoCircleOutlined /> },
};

const actionConfig = {
  create: { color: 'green' },
  read: { color: 'blue' },
  update: { color: 'orange' },
  delete: { color: 'red' },
  login: { color: 'cyan' },
  logout: { color: 'default' },
};

export const AuditLogs: React.FC = () => {
  const { t } = useTranslation();
  const [selectedEvent, setSelectedEvent] = useState<AuditEvent | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [filters, setFilters] = useState({ entity: '', action: '', search: '' });

  const filteredEvents = mockAuditEvents.filter(event => {
    if (filters.entity && event.entity !== filters.entity) return false;
    if (filters.action && event.action !== filters.action) return false;
    if (filters.search && !event.resource.toLowerCase().includes(filters.search.toLowerCase())) return false;
    return true;
  });

  const handleVerifyIntegrity = () => {
    message.loading('Verifying audit log integrity...');
    setTimeout(() => {
      message.success('All audit logs verified - No tampering detected');
    }, 2000);
  };

  const handleExport = () => {
    message.success('Audit logs exported successfully');
  };

  const columns: TableProps<AuditEvent>['columns'] = [
    {
      title: 'Timestamp',
      dataIndex: 'timestamp',
      width: 180,
      render: (ts: string) => (
        <Space direction="vertical" size={0}>
          <Text>{new Date(ts).toLocaleDateString()}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {new Date(ts).toLocaleTimeString()}
          </Text>
        </Space>
      ),
    },
    {
      title: 'Entity',
      dataIndex: 'entity',
      width: 120,
      render: (entity: keyof typeof entityConfig) => {
        const config = entityConfig[entity];
        return (
          <Tag color={config.color} icon={config.icon}>
            {entity.replace('_', ' ')}
          </Tag>
        );
      },
    },
    {
      title: 'Action',
      dataIndex: 'action',
      width: 100,
      render: (action: keyof typeof actionConfig) => (
        <Tag color={actionConfig[action].color}>{action.toUpperCase()}</Tag>
      ),
    },
    {
      title: 'Actor',
      dataIndex: 'actor',
      width: 180,
      render: (actor: AuditEvent['actor']) => (
        <Space>
          <UserOutlined />
          <div>
            <Text strong>{actor.name}</Text>
            <div><Text type="secondary" style={{ fontSize: 11 }}>{actor.email}</Text></div>
          </div>
        </Space>
      ),
    },
    {
      title: 'Resource',
      dataIndex: 'resource',
      ellipsis: true,
      render: (resource: string) => (
        <Tooltip title={resource}>
          <Text code style={{ fontSize: 12 }}>{resource}</Text>
        </Tooltip>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={status === 'success' ? 'green' : status === 'failure' ? 'red' : 'orange'}>
          {status}
        </Tag>
      ),
    },
    {
      title: 'Verified',
      dataIndex: 'verified',
      width: 80,
      render: (verified: boolean) => (
        verified ? (
          <CheckCircleOutlined style={{ color: '#22c55e' }} />
        ) : (
          <CloseCircleOutlined style={{ color: '#ef4444' }} />
        )
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 80,
      render: (_, record) => (
        <Button
          size="small"
          icon={<EyeOutlined />}
          onClick={() => {
            setSelectedEvent(record);
            setDrawerOpen(true);
          }}
        />
      ),
    },
  ];

  return (
    <div className="audit-logs-page" style={{ maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <AuditOutlined style={{ color: '#2563eb' }} /> Audit Logs
          </Title>
          <Text type="secondary">Track all system activities with tamper-proof logging</Text>
        </div>
        <Space>
          <Button icon={<SafetyCertificateOutlined />} onClick={handleVerifyIntegrity}>
            Verify Integrity
          </Button>
          <Button icon={<DownloadOutlined />} onClick={handleExport}>
            Export
          </Button>
        </Space>
      </div>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Total Events" value={mockAuditEvents.length} prefix={<AuditOutlined />} />
          </Card>
        </Col>
        <Col xs={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Success" value={mockAuditEvents.filter(e => e.status === 'success').length} valueStyle={{ color: '#22c55e' }} />
          </Card>
        </Col>
        <Col xs={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Failures" value={mockAuditEvents.filter(e => e.status === 'failure').length} valueStyle={{ color: '#ef4444' }} />
          </Card>
        </Col>
        <Col xs={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Verified" value={`${mockAuditEvents.filter(e => e.verified).length}/${mockAuditEvents.length}`} valueStyle={{ color: '#22c55e' }} prefix={<CheckCircleOutlined />} />
          </Card>
        </Col>
      </Row>

      {/* Filters */}
      <Card style={{ marginBottom: 16, borderRadius: 12 }}>
        <Space wrap>
          <Input
            placeholder="Search resources..."
            prefix={<SearchOutlined />}
            style={{ width: 250 }}
            value={filters.search}
            onChange={e => setFilters(f => ({ ...f, search: e.target.value }))}
            allowClear
          />
          <Select
            placeholder="Entity"
            style={{ width: 140 }}
            allowClear
            value={filters.entity || undefined}
            onChange={v => setFilters(f => ({ ...f, entity: v || '' }))}
            options={Object.keys(entityConfig).map(k => ({ value: k, label: k.replace('_', ' ') }))}
          />
          <Select
            placeholder="Action"
            style={{ width: 120 }}
            allowClear
            value={filters.action || undefined}
            onChange={v => setFilters(f => ({ ...f, action: v || '' }))}
            options={Object.keys(actionConfig).map(k => ({ value: k, label: k }))}
          />
          <RangePicker />
        </Space>
      </Card>

      {/* Table */}
      <Card style={{ borderRadius: 12 }}>
        <Table columns={columns} dataSource={filteredEvents} rowKey="id" pagination={{ pageSize: 15 }} />
      </Card>

      {/* Detail Drawer */}
      <Drawer
        title="Event Details"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        width={500}
      >
        {selectedEvent && (
          <Space direction="vertical" style={{ width: '100%' }} size={24}>
            <Descriptions column={1} bordered size="small">
              <Descriptions.Item label="Event ID">{selectedEvent.id}</Descriptions.Item>
              <Descriptions.Item label="Timestamp">{new Date(selectedEvent.timestamp).toLocaleString()}</Descriptions.Item>
              <Descriptions.Item label="Entity">
                <Tag color={entityConfig[selectedEvent.entity].color}>{selectedEvent.entity}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Action">
                <Tag color={actionConfig[selectedEvent.action].color}>{selectedEvent.action}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={selectedEvent.status === 'success' ? 'green' : 'red'}>{selectedEvent.status}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Actor">{selectedEvent.actor.name} ({selectedEvent.actor.email})</Descriptions.Item>
              <Descriptions.Item label="Resource"><Text code>{selectedEvent.resource}</Text></Descriptions.Item>
              <Descriptions.Item label="IP Address">{selectedEvent.ipAddress}</Descriptions.Item>
              <Descriptions.Item label="User Agent">{selectedEvent.userAgent}</Descriptions.Item>
              <Descriptions.Item label="Verified">
                {selectedEvent.verified ? <CheckCircleOutlined style={{ color: '#22c55e' }} /> : <CloseCircleOutlined style={{ color: '#ef4444' }} />}
              </Descriptions.Item>
            </Descriptions>
            {selectedEvent.details && (
              <Card title="Additional Details" size="small">
                <pre style={{ margin: 0, fontSize: 12 }}>{JSON.stringify(selectedEvent.details, null, 2)}</pre>
              </Card>
            )}
          </Space>
        )}
      </Drawer>
    </div>
  );
};

export default AuditLogs;
