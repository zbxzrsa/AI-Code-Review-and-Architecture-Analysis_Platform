/**
 * Webhooks Management Page
 * Webhook管理页面
 * 
 * Features:
 * - Webhook configuration
 * - Event subscriptions
 * - Delivery logs
 * - Testing interface
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
  Modal,
  Form,
  Input,
  Select,
  Tabs,
  Badge,
  Tooltip,
  Alert,
  Statistic,
  Timeline,
  message,
} from 'antd';
import type { TableProps } from 'antd';
import {
  ApiOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  LinkOutlined,
  KeyOutlined,
  BellOutlined,
  CodeOutlined,
  SecurityScanOutlined,
  RocketOutlined,
  EyeOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text } = Typography;

interface Webhook {
  id: string;
  name: string;
  url: string;
  secret?: string;
  events: string[];
  status: 'active' | 'inactive' | 'failed';
  lastDelivery?: string;
  lastStatus?: number;
  successRate: number;
  createdAt: string;
}

interface DeliveryLog {
  id: string;
  webhookId: string;
  event: string;
  status: number;
  duration: number;
  timestamp: string;
  requestBody: string;
  responseBody: string;
}

const mockWebhooks: Webhook[] = [
  {
    id: 'wh_1',
    name: 'Slack Notifications',
    url: 'https://hooks.slack.com/services/xxx/yyy/zzz',
    events: ['analysis.completed', 'security.alert'],
    status: 'active',
    lastDelivery: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
    lastStatus: 200,
    successRate: 99.5,
    createdAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'wh_2',
    name: 'CI/CD Pipeline Trigger',
    url: 'https://api.github.com/repos/myorg/myrepo/dispatches',
    events: ['deployment.success', 'deployment.failed'],
    status: 'active',
    lastDelivery: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    lastStatus: 204,
    successRate: 100,
    createdAt: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'wh_3',
    name: 'Security Alert System',
    url: 'https://security.example.com/api/alerts',
    events: ['security.vulnerability', 'security.alert'],
    status: 'failed',
    lastDelivery: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    lastStatus: 500,
    successRate: 85.2,
    createdAt: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString(),
  },
];

const mockDeliveryLogs: DeliveryLog[] = [
  {
    id: 'dl_1',
    webhookId: 'wh_1',
    event: 'analysis.completed',
    status: 200,
    duration: 245,
    timestamp: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
    requestBody: '{"event":"analysis.completed","project":"frontend"}',
    responseBody: '{"ok":true}',
  },
  {
    id: 'dl_2',
    webhookId: 'wh_1',
    event: 'security.alert',
    status: 200,
    duration: 312,
    timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    requestBody: '{"event":"security.alert","severity":"critical"}',
    responseBody: '{"ok":true}',
  },
  {
    id: 'dl_3',
    webhookId: 'wh_3',
    event: 'security.vulnerability',
    status: 500,
    duration: 5000,
    timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    requestBody: '{"event":"security.vulnerability"}',
    responseBody: '{"error":"Internal Server Error"}',
  },
];

const availableEvents = [
  { value: 'analysis.completed', label: 'Analysis Completed', icon: <CodeOutlined /> },
  { value: 'analysis.failed', label: 'Analysis Failed', icon: <CloseCircleOutlined /> },
  { value: 'security.alert', label: 'Security Alert', icon: <SecurityScanOutlined /> },
  { value: 'security.vulnerability', label: 'Vulnerability Found', icon: <SecurityScanOutlined /> },
  { value: 'deployment.success', label: 'Deployment Success', icon: <RocketOutlined /> },
  { value: 'deployment.failed', label: 'Deployment Failed', icon: <CloseCircleOutlined /> },
  { value: 'pr.opened', label: 'PR Opened', icon: <BellOutlined /> },
  { value: 'pr.merged', label: 'PR Merged', icon: <CheckCircleOutlined /> },
];

export const Webhooks: React.FC = () => {
  const { t: _t } = useTranslation();
  const [webhooks, setWebhooks] = useState<Webhook[]>(mockWebhooks);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [selectedWebhook, setSelectedWebhook] = useState<Webhook | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [form] = Form.useForm();

  const handleCreate = (values: any) => {
    const newWebhook: Webhook = {
      id: `wh_${Date.now()}`,
      ...values,
      status: 'active',
      successRate: 100,
      createdAt: new Date().toISOString(),
    };
    setWebhooks(prev => [...prev, newWebhook]);
    setCreateModalOpen(false);
    form.resetFields();
    message.success('Webhook created successfully');
  };

  const handleTest = async (_webhook: Webhook) => {
    message.loading('Sending test request...');
    setTimeout(() => {
      message.success('Test webhook delivered successfully');
    }, 1500);
  };

  const handleDelete = (webhookId: string) => {
    setWebhooks(prev => prev.filter(w => w.id !== webhookId));
    message.success('Webhook deleted');
  };

  const columns: TableProps<Webhook>['columns'] = [
    {
      title: 'Webhook',
      key: 'webhook',
      render: (_, record) => (
        <div>
          <Space>
            <ApiOutlined style={{ color: '#2563eb' }} />
            <Text strong>{record.name}</Text>
            <Badge
              status={record.status === 'active' ? 'success' : record.status === 'failed' ? 'error' : 'default'}
            />
          </Space>
          <div>
            <Text type="secondary" style={{ fontSize: 12 }} copyable={{ text: record.url }}>
              {record.url.length > 50 ? record.url.substring(0, 50) + '...' : record.url}
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: 'Events',
      dataIndex: 'events',
      width: 200,
      render: (events: string[]) => (
        <Space wrap size={4}>
          {events.slice(0, 2).map(event => (
            <Tag key={event} style={{ fontSize: 11 }}>{event}</Tag>
          ))}
          {events.length > 2 && <Tag>+{events.length - 2}</Tag>}
        </Space>
      ),
    },
    {
      title: 'Last Delivery',
      key: 'lastDelivery',
      width: 150,
      render: (_, record) => (
        record.lastDelivery ? (
          <Space direction="vertical" size={0}>
            <Tag color={record.lastStatus === 200 || record.lastStatus === 204 ? 'green' : 'red'}>
              {record.lastStatus}
            </Tag>
            <Text type="secondary" style={{ fontSize: 11 }}>
              {new Date(record.lastDelivery).toLocaleString()}
            </Text>
          </Space>
        ) : (
          <Text type="secondary">Never</Text>
        )
      ),
    },
    {
      title: 'Success Rate',
      dataIndex: 'successRate',
      width: 120,
      render: (rate: number) => (
        <Text style={{ color: rate >= 95 ? '#22c55e' : rate >= 80 ? '#f59e0b' : '#ef4444' }}>
          {rate}%
        </Text>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 180,
      render: (_, record) => (
        <Space>
          <Tooltip title="Test">
            <Button size="small" icon={<PlayCircleOutlined />} onClick={() => handleTest(record)} />
          </Tooltip>
          <Tooltip title="View Details">
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedWebhook(record);
                setDetailModalOpen(true);
              }}
            />
          </Tooltip>
          <Tooltip title="Edit">
            <Button size="small" icon={<EditOutlined />} />
          </Tooltip>
          <Tooltip title="Delete">
            <Button size="small" danger icon={<DeleteOutlined />} onClick={() => handleDelete(record.id)} />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const stats = {
    total: webhooks.length,
    active: webhooks.filter(w => w.status === 'active').length,
    failed: webhooks.filter(w => w.status === 'failed').length,
  };

  return (
    <div className="webhooks-page" style={{ maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <ApiOutlined style={{ color: '#2563eb' }} /> Webhooks
          </Title>
          <Text type="secondary">Configure webhooks to integrate with external services</Text>
        </div>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalOpen(true)}>
          Create Webhook
        </Button>
      </div>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={8}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Total Webhooks" value={stats.total} prefix={<ApiOutlined />} />
          </Card>
        </Col>
        <Col xs={8}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Active" value={stats.active} valueStyle={{ color: '#22c55e' }} prefix={<CheckCircleOutlined />} />
          </Card>
        </Col>
        <Col xs={8}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="Failed" value={stats.failed} valueStyle={{ color: '#ef4444' }} prefix={<CloseCircleOutlined />} />
          </Card>
        </Col>
      </Row>

      {/* Failed Webhook Alert */}
      {stats.failed > 0 && (
        <Alert
          type="error"
          showIcon
          message="Webhook Delivery Failures"
          description={`${stats.failed} webhook(s) have failed recently. Check the delivery logs for details.`}
          style={{ marginBottom: 24, borderRadius: 12 }}
          action={<Button size="small" danger>View Failures</Button>}
        />
      )}

      {/* Webhooks Table */}
      <Card title="Configured Webhooks" style={{ borderRadius: 12 }}>
        <Table columns={columns} dataSource={webhooks} rowKey="id" pagination={false} />
      </Card>

      {/* Create Modal */}
      <Modal
        title={<><PlusOutlined /> Create Webhook</>}
        open={createModalOpen}
        onCancel={() => {
          setCreateModalOpen(false);
          form.resetFields();
        }}
        onOk={() => form.submit()}
        okText="Create"
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          <Form.Item name="name" label="Name" rules={[{ required: true }]}>
            <Input placeholder="e.g., Slack Notifications" />
          </Form.Item>
          <Form.Item name="url" label="Payload URL" rules={[{ required: true, type: 'url' }]}>
            <Input placeholder="https://example.com/webhook" prefix={<LinkOutlined />} />
          </Form.Item>
          <Form.Item name="secret" label="Secret (Optional)">
            <Input.Password placeholder="Webhook secret for signature verification" prefix={<KeyOutlined />} />
          </Form.Item>
          <Form.Item name="events" label="Events" rules={[{ required: true }]}>
            <Select
              mode="multiple"
              placeholder="Select events to subscribe"
              options={availableEvents.map(e => ({ value: e.value, label: e.label }))}
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* Detail Modal */}
      <Modal
        title={<><ApiOutlined /> Webhook Details</>}
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        footer={null}
        width={700}
      >
        {selectedWebhook && (
          <Tabs
            items={[
              {
                key: 'details',
                label: 'Details',
                children: (
                  <Space direction="vertical" style={{ width: '100%' }} size={16}>
                    <div>
                      <Text type="secondary">Name</Text>
                      <div><Text strong>{selectedWebhook.name}</Text></div>
                    </div>
                    <div>
                      <Text type="secondary">URL</Text>
                      <div><Text copyable>{selectedWebhook.url}</Text></div>
                    </div>
                    <div>
                      <Text type="secondary">Events</Text>
                      <div>
                        <Space wrap>
                          {selectedWebhook.events.map(e => <Tag key={e}>{e}</Tag>)}
                        </Space>
                      </div>
                    </div>
                    <div>
                      <Text type="secondary">Success Rate</Text>
                      <div><Text strong>{selectedWebhook.successRate}%</Text></div>
                    </div>
                  </Space>
                ),
              },
              {
                key: 'deliveries',
                label: 'Recent Deliveries',
                children: (
                  <Timeline
                    items={mockDeliveryLogs.filter(l => l.webhookId === selectedWebhook.id).map(log => ({
                      color: log.status < 300 ? 'green' : 'red',
                      children: (
                        <div>
                          <Space>
                            <Tag color={log.status < 300 ? 'green' : 'red'}>{log.status}</Tag>
                            <Text>{log.event}</Text>
                            <Text type="secondary">{log.duration}ms</Text>
                          </Space>
                          <div>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              {new Date(log.timestamp).toLocaleString()}
                            </Text>
                          </div>
                        </div>
                      ),
                    }))}
                  />
                ),
              },
            ]}
          />
        )}
      </Modal>
    </div>
  );
};

export default Webhooks;
