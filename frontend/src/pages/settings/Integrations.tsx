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

import React, { useState } from 'react';
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
  List,
  message,
  Popconfirm,
  Badge,
  Alert,
  Tabs,
  Spin,
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
  LoadingOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../../services/api';

const { Title, Text } = Typography;

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
  const queryClient = useQueryClient();
  const [webhooks, setWebhooks] = useState<Webhook[]>([
    {
      id: 'wh_1',
      name: 'CI/CD Pipeline',
      url: 'https://ci.example.com/webhook',
      events: ['analysis.completed', 'issue.critical'],
      status: 'active',
      lastTriggered: '2024-03-01T14:30:00Z',
      failureCount: 0,
    },
  ]);
  const [webhookModalOpen, setWebhookModalOpen] = useState(false);
  const [editingWebhook, setEditingWebhook] = useState<Webhook | null>(null);
  const [form] = Form.useForm();

  // Fetch OAuth connections from API
  const { data: connectionsData, isLoading: isLoadingConnections } = useQuery({
    queryKey: ['oauth-connections'],
    queryFn: async () => {
      const response = await apiService.oauth.getConnections();
      return response.data;
    },
  });

  // Map connections to integrations format
  const integrations: Integration[] = (connectionsData?.connections || []).map((conn: any) => ({
    id: conn.provider,
    name: gitProviders.find(p => p.id === conn.provider)?.name || conn.provider,
    type: 'git' as const,
    provider: conn.provider,
    status: 'connected' as const,
    config: { username: conn.username, email: conn.email },
    createdAt: conn.connected_at,
    lastSync: conn.connected_at,
  }));

  // Disconnect mutation
  const disconnectMutation = useMutation({
    mutationFn: async (provider: string) => {
      const response = await apiService.oauth.disconnect(provider);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['oauth-connections'] });
      message.success('Disconnected successfully');
    },
    onError: (error: any) => {
      message.error(error.response?.data?.detail || 'Failed to disconnect');
    },
  });

  // Connect to OAuth provider
  const handleConnect = async (provider: string) => {
    // Check if it's a Git provider (OAuth supported)
    if (gitProviders.some(p => p.id === provider)) {
      try {
        // Use fetch directly to avoid axios interceptors that might redirect on 401
        const response = await fetch(`/api/auth/oauth/connect/${provider}?return_url=${encodeURIComponent(window.location.href)}`);
        
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Failed to initiate OAuth');
        }
        
        const data = await response.json();
        
        if (data.authorization_url) {
          // Redirect to OAuth provider
          window.location.href = data.authorization_url;
        } else if (data.connected) {
          // Already connected (Bitbucket API Token)
          message.success(`${provider} is already connected via API Token`);
          queryClient.invalidateQueries({ queryKey: ['oauth-connections'] });
        }
      } catch (error: any) {
        // Show error message instead of redirecting
        message.error(error.message || `Failed to connect to ${provider}. Please check if OAuth is configured.`);
      }
    } else {
      // For notification providers, show a configuration modal (not OAuth)
      message.info(`${provider} integration coming soon!`);
    }
  };

  // Disconnect integration
  const handleDisconnect = async (provider: string) => {
    if (gitProviders.some(p => p.id === provider)) {
      disconnectMutation.mutate(provider);
    }
  };

  // Save webhook (using local state for now - can be connected to API later)
  const handleSaveWebhook = async (values: any) => {
    // Demo mode - save to local state
    if (editingWebhook) {
      setWebhooks(prev => prev.map(w => 
        w.id === editingWebhook.id ? { ...w, ...values } : w
      ));
      message.success('Webhook updated');
    } else {
      setWebhooks(prev => [...prev, {
        id: `wh_${Date.now()}`,
        ...values,
        status: 'active',
        failureCount: 0,
      }]);
      message.success('Webhook created');
    }
    setWebhookModalOpen(false);
    setEditingWebhook(null);
    form.resetFields();
  };

  // Delete webhook
  const handleDeleteWebhook = async (webhookId: string) => {
    setWebhooks(prev => prev.filter(w => w.id !== webhookId));
    message.success('Webhook deleted');
  };

  // Test webhook
  const handleTestWebhook = async (webhook: Webhook) => {
    message.success(`Test event sent to ${webhook.url}`);
  };

  const _isConnected = (provider: string) => 
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
            children: isLoadingConnections ? (
              <div style={{ textAlign: 'center', padding: 40 }}>
                <Spin indicator={<LoadingOutlined style={{ fontSize: 32 }} spin />} />
                <div style={{ marginTop: 16 }}>
                  <Text type="secondary">Loading connections...</Text>
                </div>
              </div>
            ) : (
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
            )
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
