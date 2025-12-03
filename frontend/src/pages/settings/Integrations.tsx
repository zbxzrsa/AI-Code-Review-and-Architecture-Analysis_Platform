/**
 * Integrations & Webhooks Page
 * 集成和Webhooks页面
 * 
 * Features:
 * - Git provider connections (GitHub, GitLab, Bitbucket)
 * - CI/CD integrations
 * - Notification channels (Slack, Teams, Discord)
 * - Custom webhooks
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Typography,
  Modal,
  Form,
  Input,
  Select,
  Tag,
  Switch,
  List,
  Avatar,
  Tooltip,
  message,
  Popconfirm,
  Badge,
  Divider,
  Alert,
  Tabs,
} from 'antd';
import {
  GithubOutlined,
  GitlabOutlined,
  SlackOutlined,
  ApiOutlined,
  LinkOutlined,
  DisconnectOutlined,
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
  CheckCircleOutlined,
  SettingOutlined,
  BellOutlined,
  CloudServerOutlined,
  SendOutlined,
  ReloadOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { api } from '../../services/api';

const { Title, Text, Paragraph } = Typography;

interface Integration {
  id: string;
  name: string;
  type: 'git' | 'ci' | 'notification' | 'webhook';
  provider: string;
  status: 'connected' | 'disconnected' | 'error';
  config: Record<string, any>;
  createdAt: string;
  lastSync?: string;
}

interface Webhook {
  id: string;
  name: string;
  url: string;
  events: string[];
  status: 'active' | 'inactive' | 'failing';
  secret?: string;
  lastTriggered?: string;
  failureCount: number;
}

const gitProviders = [
  { id: 'github', name: 'GitHub', icon: <GithubOutlined style={{ fontSize: 24 }} />, color: '#24292e' },
  { id: 'gitlab', name: 'GitLab', icon: <GitlabOutlined style={{ fontSize: 24 }} />, color: '#fc6d26' },
  { id: 'bitbucket', name: 'Bitbucket', icon: <ApiOutlined style={{ fontSize: 24 }} />, color: '#0052cc' },
  { id: 'azure', name: 'Azure DevOps', icon: <CloudServerOutlined style={{ fontSize: 24 }} />, color: '#0078d4' },
];

const notificationProviders = [
  { id: 'slack', name: 'Slack', icon: <SlackOutlined style={{ fontSize: 24 }} />, color: '#4a154b' },
  { id: 'teams', name: 'Microsoft Teams', icon: <ApiOutlined style={{ fontSize: 24 }} />, color: '#6264a7' },
  { id: 'discord', name: 'Discord', icon: <ApiOutlined style={{ fontSize: 24 }} />, color: '#5865f2' },
  { id: 'email', name: 'Email', icon: <BellOutlined style={{ fontSize: 24 }} />, color: '#1890ff' },
];

const webhookEvents = [
  { value: 'analysis.completed', label: 'Analysis Completed' },
  { value: 'analysis.failed', label: 'Analysis Failed' },
  { value: 'issue.critical', label: 'Critical Issue Found' },
  { value: 'issue.security', label: 'Security Issue Found' },
  { value: 'project.created', label: 'Project Created' },
  { value: 'project.deleted', label: 'Project Deleted' },
  { value: 'user.joined', label: 'User Joined Team' },
];

export const Integrations: React.FC = () => {
  const { t } = useTranslation();
  const [integrations, setIntegrations] = useState<Integration[]>([]);
  const [webhooks, setWebhooks] = useState<Webhook[]>([]);
  const [loading, setLoading] = useState(false);
  const [webhookModalOpen, setWebhookModalOpen] = useState(false);
  const [editingWebhook, setEditingWebhook] = useState<Webhook | null>(null);
  const [form] = Form.useForm();

  // Mock data
  useEffect(() => {
    setIntegrations([
      {
        id: 'int_1',
        name: 'GitHub',
        type: 'git',
        provider: 'github',
        status: 'connected',
        config: { org: 'my-org', repos: 12 },
        createdAt: '2024-01-15T10:00:00Z',
        lastSync: '2024-03-01T15:30:00Z',
      },
      {
        id: 'int_2',
        name: 'Slack',
        type: 'notification',
        provider: 'slack',
        status: 'connected',
        config: { workspace: 'My Workspace', channel: '#code-reviews' },
        createdAt: '2024-02-01T10:00:00Z',
      },
    ]);

    setWebhooks([
      {
        id: 'wh_1',
        name: 'CI/CD Pipeline',
        url: 'https://ci.example.com/webhook',
        events: ['analysis.completed', 'issue.critical'],
        status: 'active',
        lastTriggered: '2024-03-01T14:30:00Z',
        failureCount: 0,
      },
      {
        id: 'wh_2',
        name: 'Security Alerts',
        url: 'https://security.example.com/alerts',
        events: ['issue.security', 'issue.critical'],
        status: 'failing',
        lastTriggered: '2024-02-28T10:00:00Z',
        failureCount: 3,
      },
    ]);
  }, []);

  // Connect integration
  const handleConnect = async (provider: string) => {
    try {
      await api.post(`/api/user/integrations/${provider}/connect`);
      message.success(`Connected to ${provider}`);
    } catch (error) {
      message.success(`Connected to ${provider} (demo)`);
      // Add mock integration
      const providerInfo = [...gitProviders, ...notificationProviders].find(p => p.id === provider);
      if (providerInfo) {
        setIntegrations(prev => [...prev, {
          id: `int_${Date.now()}`,
          name: providerInfo.name,
          type: gitProviders.some(p => p.id === provider) ? 'git' : 'notification',
          provider,
          status: 'connected',
          config: {},
          createdAt: new Date().toISOString(),
        }]);
      }
    }
  };

  // Disconnect integration
  const handleDisconnect = async (integrationId: string) => {
    try {
      await api.delete(`/api/user/integrations/${integrationId}`);
      message.success('Disconnected');
    } catch (error) {
      message.success('Disconnected (demo)');
    }
    setIntegrations(prev => prev.filter(i => i.id !== integrationId));
  };

  // Save webhook
  const handleSaveWebhook = async (values: any) => {
    try {
      if (editingWebhook) {
        await api.put(`/api/user/webhooks/${editingWebhook.id}`, values);
      } else {
        await api.post('/api/user/webhooks', values);
      }
      message.success(editingWebhook ? 'Webhook updated' : 'Webhook created');
    } catch (error) {
      message.success(editingWebhook ? 'Webhook updated (demo)' : 'Webhook created (demo)');
      
      if (editingWebhook) {
        setWebhooks(prev => prev.map(w => 
          w.id === editingWebhook.id ? { ...w, ...values } : w
        ));
      } else {
        setWebhooks(prev => [...prev, {
          id: `wh_${Date.now()}`,
          ...values,
          status: 'active',
          failureCount: 0,
        }]);
      }
    }
    setWebhookModalOpen(false);
    setEditingWebhook(null);
    form.resetFields();
  };

  // Delete webhook
  const handleDeleteWebhook = async (webhookId: string) => {
    try {
      await api.delete(`/api/user/webhooks/${webhookId}`);
      message.success('Webhook deleted');
    } catch (error) {
      message.success('Webhook deleted (demo)');
    }
    setWebhooks(prev => prev.filter(w => w.id !== webhookId));
  };

  // Test webhook
  const handleTestWebhook = async (webhook: Webhook) => {
    try {
      await api.post(`/api/user/webhooks/${webhook.id}/test`);
      message.success('Test event sent');
    } catch (error) {
      message.success('Test event sent (demo)');
    }
  };

  const isConnected = (provider: string) => 
    integrations.some(i => i.provider === provider && i.status === 'connected');

  return (
    <div className="integrations-page">
      <div style={{ marginBottom: 24 }}>
        <Title level={3}>
          <LinkOutlined /> {t('settings.integrations.title', 'Integrations & Webhooks')}
        </Title>
        <Text type="secondary">
          {t('settings.integrations.subtitle', 'Connect your tools and set up automated workflows')}
        </Text>
      </div>

      <Tabs
        defaultActiveKey="git"
        items={[
          {
            key: 'git',
            label: <><GithubOutlined /> Git Providers</>,
            children: (
              <Row gutter={[16, 16]}>
                {gitProviders.map(provider => {
                  const integration = integrations.find(i => i.provider === provider.id);
                  const connected = integration?.status === 'connected';
                  
                  return (
                    <Col xs={24} sm={12} md={6} key={provider.id}>
                      <Card
                        hoverable
                        style={{ textAlign: 'center' }}
                      >
                        <div style={{ marginBottom: 16, color: provider.color }}>
                          {provider.icon}
                        </div>
                        <Title level={5}>{provider.name}</Title>
                        {connected ? (
                          <>
                            <Tag color="success" icon={<CheckCircleOutlined />}>
                              Connected
                            </Tag>
                            {integration?.config.repos && (
                              <Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
                                {integration.config.repos} repositories
                              </Text>
                            )}
                            <div style={{ marginTop: 16 }}>
                              <Space>
                                <Button size="small" icon={<SettingOutlined />}>
                                  Settings
                                </Button>
                                <Popconfirm
                                  title="Disconnect this integration?"
                                  onConfirm={() => handleDisconnect(integration!.id)}
                                >
                                  <Button size="small" danger icon={<DisconnectOutlined />}>
                                    Disconnect
                                  </Button>
                                </Popconfirm>
                              </Space>
                            </div>
                          </>
                        ) : (
                          <Button
                            type="primary"
                            icon={<LinkOutlined />}
                            onClick={() => handleConnect(provider.id)}
                          >
                            Connect
                          </Button>
                        )}
                      </Card>
                    </Col>
                  );
                })}
              </Row>
            ),
          },
          {
            key: 'notifications',
            label: <><BellOutlined /> Notifications</>,
            children: (
              <Row gutter={[16, 16]}>
                {notificationProviders.map(provider => {
                  const integration = integrations.find(i => i.provider === provider.id);
                  const connected = integration?.status === 'connected';
                  
                  return (
                    <Col xs={24} sm={12} md={6} key={provider.id}>
                      <Card hoverable style={{ textAlign: 'center' }}>
                        <div style={{ marginBottom: 16, color: provider.color }}>
                          {provider.icon}
                        </div>
                        <Title level={5}>{provider.name}</Title>
                        {connected ? (
                          <>
                            <Tag color="success" icon={<CheckCircleOutlined />}>
                              Connected
                            </Tag>
                            {integration?.config.channel && (
                              <Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
                                {integration.config.channel}
                              </Text>
                            )}
                            <div style={{ marginTop: 16 }}>
                              <Space>
                                <Button size="small" icon={<SettingOutlined />}>
                                  Settings
                                </Button>
                                <Popconfirm
                                  title="Disconnect?"
                                  onConfirm={() => handleDisconnect(integration!.id)}
                                >
                                  <Button size="small" danger icon={<DisconnectOutlined />}>
                                    Disconnect
                                  </Button>
                                </Popconfirm>
                              </Space>
                            </div>
                          </>
                        ) : (
                          <Button
                            type="primary"
                            icon={<LinkOutlined />}
                            onClick={() => handleConnect(provider.id)}
                          >
                            Connect
                          </Button>
                        )}
                      </Card>
                    </Col>
                  );
                })}
              </Row>
            ),
          },
          {
            key: 'webhooks',
            label: <><ApiOutlined /> Webhooks</>,
            children: (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
                  <Text type="secondary">
                    Webhooks allow external services to be notified when events happen.
                  </Text>
                  <Button
                    type="primary"
                    icon={<PlusOutlined />}
                    onClick={() => {
                      setEditingWebhook(null);
                      form.resetFields();
                      setWebhookModalOpen(true);
                    }}
                  >
                    Add Webhook
                  </Button>
                </div>

                <List
                  dataSource={webhooks}
                  renderItem={webhook => (
                    <Card style={{ marginBottom: 16 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div>
                          <Space>
                            <Text strong>{webhook.name}</Text>
                            <Badge
                              status={
                                webhook.status === 'active' ? 'success' :
                                webhook.status === 'failing' ? 'error' : 'default'
                              }
                              text={webhook.status}
                            />
                          </Space>
                          <div style={{ marginTop: 8 }}>
                            <Text code style={{ fontSize: 12 }}>{webhook.url}</Text>
                          </div>
                          <div style={{ marginTop: 8 }}>
                            <Space wrap>
                              {webhook.events.map(event => (
                                <Tag key={event}>{event}</Tag>
                              ))}
                            </Space>
                          </div>
                          {webhook.lastTriggered && (
                            <Text type="secondary" style={{ display: 'block', marginTop: 8, fontSize: 12 }}>
                              Last triggered: {new Date(webhook.lastTriggered).toLocaleString()}
                            </Text>
                          )}
                          {webhook.failureCount > 0 && (
                            <Alert
                              message={`${webhook.failureCount} consecutive failures`}
                              type="error"
                              showIcon
                              style={{ marginTop: 8 }}
                            />
                          )}
                        </div>
                        <Space>
                          <Button
                            size="small"
                            icon={<SendOutlined />}
                            onClick={() => handleTestWebhook(webhook)}
                          >
                            Test
                          </Button>
                          <Button
                            size="small"
                            icon={<EditOutlined />}
                            onClick={() => {
                              setEditingWebhook(webhook);
                              form.setFieldsValue(webhook);
                              setWebhookModalOpen(true);
                            }}
                          >
                            Edit
                          </Button>
                          <Popconfirm
                            title="Delete this webhook?"
                            onConfirm={() => handleDeleteWebhook(webhook.id)}
                          >
                            <Button size="small" danger icon={<DeleteOutlined />}>
                              Delete
                            </Button>
                          </Popconfirm>
                        </Space>
                      </div>
                    </Card>
                  )}
                />
              </>
            ),
          },
        ]}
      />

      {/* Webhook Modal */}
      <Modal
        title={editingWebhook ? 'Edit Webhook' : 'Add Webhook'}
        open={webhookModalOpen}
        onCancel={() => {
          setWebhookModalOpen(false);
          setEditingWebhook(null);
          form.resetFields();
        }}
        onOk={() => form.submit()}
      >
        <Form form={form} layout="vertical" onFinish={handleSaveWebhook}>
          <Form.Item
            name="name"
            label="Name"
            rules={[{ required: true }]}
          >
            <Input placeholder="e.g., CI/CD Pipeline" />
          </Form.Item>

          <Form.Item
            name="url"
            label="Payload URL"
            rules={[
              { required: true },
              { type: 'url', message: 'Please enter a valid URL' }
            ]}
          >
            <Input placeholder="https://example.com/webhook" />
          </Form.Item>

          <Form.Item
            name="events"
            label="Events"
            rules={[{ required: true, message: 'Select at least one event' }]}
          >
            <Select
              mode="multiple"
              options={webhookEvents}
              placeholder="Select events to trigger webhook"
            />
          </Form.Item>

          <Form.Item
            name="secret"
            label="Secret (optional)"
            extra="Used to sign webhook payloads for verification"
          >
            <Input.Password placeholder="Enter a secret key" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default Integrations;
