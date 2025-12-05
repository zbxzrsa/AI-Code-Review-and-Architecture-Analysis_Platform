/**
 * Provider Management Page
 * 
 * Admin page for managing AI providers with:
 * - Provider list with status and metrics
 * - Provider configuration
 * - Model management
 * - Health monitoring
 * - Performance charts
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Button,
  Space,
  Tag,
  Badge,
  Progress,
  Statistic,
  List,
  Avatar,
  Form,
  Input,
  InputNumber,
  Select,
  Switch,
  Table,
  Alert,
  Divider,
  Tabs,
  Skeleton,
  Empty,
  Descriptions,
} from 'antd';
import type { TableProps, TabsProps } from 'antd';
import {
  ApiOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  ThunderboltOutlined,
  DollarOutlined,
  ReloadOutlined,
  SettingOutlined,
  EditOutlined,
  PauseCircleOutlined,
  KeyOutlined,
  RobotOutlined,
  ExperimentOutlined,
  DatabaseOutlined,
  CloudOutlined,
  DesktopOutlined,
  LineChartOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  useAdminStore,
  type AIProvider,
  type ProviderModel,
  type ProviderType,
  type ProviderStatus,
} from '../../store/adminStore';
import {
  useProviders,
  useProvider,
  useUpdateProvider,
  useUpdateProviderApiKey,
  useTestProvider,
  useProviderHealth,
  useProviderMetrics,
  useProviderModels,
  useUpdateProviderModel,
} from '../../hooks/useAdmin';
import './ProviderManagement.css';

const { Title, Text } = Typography;

/** Provider icons */
const providerIcons: Record<ProviderType, React.ReactNode> = {
  openai: <RobotOutlined />,
  anthropic: <ExperimentOutlined />,
  google: <CloudOutlined />,
  azure: <CloudOutlined />,
  huggingface: <DatabaseOutlined />,
  local: <DesktopOutlined />,
};

/** Provider colors */
const providerColors: Record<ProviderType, string> = {
  openai: '#10a37f',
  anthropic: '#cc785c',
  google: '#4285f4',
  azure: '#0078d4',
  huggingface: '#ff6b00',
  local: '#722ed1',
};

/** Status colors */
const statusColors: Record<ProviderStatus, string> = {
  active: 'green',
  inactive: 'default',
  error: 'red',
  rate_limited: 'orange',
};

/** Status icons */
const statusIcons: Record<ProviderStatus, React.ReactNode> = {
  active: <CheckCircleOutlined />,
  inactive: <PauseCircleOutlined />,
  error: <CloseCircleOutlined />,
  rate_limited: <ExclamationCircleOutlined />,
};

/**
 * Provider Card Component
 */
const ProviderCard: React.FC<{
  provider: AIProvider;
  selected: boolean;
  onSelect: () => void;
  onTest: () => void;
  testing: boolean;
}> = ({ provider, selected, onSelect, onTest, testing }) => {
  const { t } = useTranslation();

  return (
    <Card
      className={`provider-card ${selected ? 'selected' : ''}`}
      hoverable
      onClick={onSelect}
    >
      <div className="provider-card-header">
        <Avatar
          icon={providerIcons[provider.type]}
          style={{ backgroundColor: providerColors[provider.type] }}
          size={48}
        />
        <div className="provider-info">
          <Text strong>{provider.name}</Text>
          <Space>
            <Tag color={statusColors[provider.status]} icon={statusIcons[provider.status]}>
              {provider.status}
            </Tag>
            {provider.isDefault && (
              <Tag color="blue">{t('admin.providers.default', 'Default')}</Tag>
            )}
          </Space>
        </div>
        <Button
          size="small"
          icon={<ThunderboltOutlined />}
          onClick={(e) => {
            e.stopPropagation();
            onTest();
          }}
          loading={testing}
        >
          {t('admin.providers.test', 'Test')}
        </Button>
      </div>

      <Divider style={{ margin: '12px 0' }} />

      <Row gutter={16}>
        <Col span={8}>
          <Statistic
            title={t('admin.providers.requests', 'Requests')}
            value={provider.requestsToday}
            valueStyle={{ fontSize: 16 }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title={t('admin.providers.latency', 'Latency')}
            value={provider.avgResponseTime}
            suffix="ms"
            valueStyle={{ fontSize: 16 }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title={t('admin.providers.cost', 'Cost')}
            value={provider.costToday}
            prefix="$"
            precision={2}
            valueStyle={{ fontSize: 16 }}
          />
        </Col>
      </Row>

      <div className="provider-quota">
        <Text type="secondary">{t('admin.providers.quota', 'Quota Usage')}</Text>
        <Progress
          percent={Math.round((provider.quotaUsed / provider.quotaLimit) * 100)}
          size="small"
          status={provider.quotaUsed / provider.quotaLimit > 0.9 ? 'exception' : 'active'}
        />
      </div>

      {provider.lastError && (
        <Alert
          message={provider.lastError}
          type="error"
          showIcon
          style={{ marginTop: 12 }}
        />
      )}
    </Card>
  );
};

/**
 * Provider Configuration Panel
 */
const ProviderConfigPanel: React.FC<{
  providerId: string;
}> = ({ providerId }) => {
  const { t } = useTranslation();
  const [configForm] = Form.useForm();
  const [apiKeyForm] = Form.useForm();
  const [showApiKey, setShowApiKey] = useState(false);

  const { data: provider, isLoading } = useProvider(providerId);
  const { data: health } = useProviderHealth(providerId);
  const { data: metrics } = useProviderMetrics(providerId, 'day');
  const { data: models } = useProviderModels(providerId);
  
  const updateProvider = useUpdateProvider();
  const updateApiKey = useUpdateProviderApiKey();
  const updateModel = useUpdateProviderModel();

  const handleConfigSave = useCallback(async (values: any) => {
    await updateProvider.mutateAsync({ providerId, data: values });
  }, [providerId, updateProvider]);

  const handleApiKeySave = useCallback(async (values: any) => {
    await updateApiKey.mutateAsync({ providerId, apiKey: values.apiKey });
    apiKeyForm.resetFields();
    setShowApiKey(false);
  }, [providerId, updateApiKey, apiKeyForm]);

  const modelColumns: TableProps<ProviderModel>['columns'] = [
    {
      title: t('admin.providers.model', 'Model'),
      dataIndex: 'displayName',
      key: 'displayName',
    },
    {
      title: t('admin.providers.type', 'Type'),
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => <Tag>{type}</Tag>,
    },
    {
      title: t('admin.providers.enabled', 'Enabled'),
      dataIndex: 'isEnabled',
      key: 'isEnabled',
      render: (enabled: boolean, record) => (
        <Switch
          checked={enabled}
          onChange={(checked) => updateModel.mutate({
            providerId,
            modelId: record.id,
            data: { isEnabled: checked },
          })}
          size="small"
        />
      ),
    },
    {
      title: t('admin.providers.default', 'Default'),
      dataIndex: 'isDefault',
      key: 'isDefault',
      render: (isDefault: boolean, record) => (
        <Switch
          checked={isDefault}
          onChange={(checked) => updateModel.mutate({
            providerId,
            modelId: record.id,
            data: { isDefault: checked },
          })}
          size="small"
          disabled={!record.isEnabled}
        />
      ),
    },
    {
      title: t('admin.providers.input_cost', 'Input $/1K'),
      dataIndex: 'costPerInputToken',
      key: 'costPerInputToken',
      render: (cost: number) => `$${(cost * 1000).toFixed(4)}`,
    },
    {
      title: t('admin.providers.output_cost', 'Output $/1K'),
      dataIndex: 'costPerOutputToken',
      key: 'costPerOutputToken',
      render: (cost: number) => `$${(cost * 1000).toFixed(4)}`,
    },
    {
      title: t('admin.providers.requests', 'Requests'),
      dataIndex: 'requestsToday',
      key: 'requestsToday',
    },
  ];

  if (isLoading) {
    return <Skeleton active paragraph={{ rows: 10 }} />;
  }

  if (!provider) {
    return <Empty description={t('admin.providers.select', 'Select a provider')} />;
  }

  const tabItems: TabsProps['items'] = [
    {
      key: 'overview',
      label: <><ApiOutlined /> {t('admin.providers.overview', 'Overview')}</>,
      children: (
        <div className="provider-overview">
          <Descriptions bordered column={2} size="small">
            <Descriptions.Item label={t('admin.providers.status', 'Status')}>
              <Badge status={provider.status === 'active' ? 'success' : 'error'} />
              {provider.status}
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.providers.priority', 'Priority')}>
              {provider.priority}
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.providers.api_key', 'API Key')}>
              {provider.apiKeyConfigured ? (
                <Tag color="green" icon={<CheckCircleOutlined />}>
                  {t('admin.providers.configured', 'Configured')}
                </Tag>
              ) : (
                <Tag color="red" icon={<CloseCircleOutlined />}>
                  {t('admin.providers.not_configured', 'Not Configured')}
                </Tag>
              )}
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.providers.last_checked', 'Last Checked')}>
              {provider.lastChecked ? new Date(provider.lastChecked).toLocaleString() : '-'}
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.providers.error_rate', 'Error Rate')}>
              <Progress 
                percent={provider.errorRate * 100} 
                size="small" 
                status={provider.errorRate > 0.05 ? 'exception' : 'normal'}
                format={(p) => `${p?.toFixed(1)}%`}
              />
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.providers.monthly_cost', 'Monthly Cost')}>
              ${provider.costThisMonth.toFixed(2)}
            </Descriptions.Item>
          </Descriptions>

          {/* Health Status */}
          {health && (
            <Card title={t('admin.providers.health', 'Health Status')} size="small" className="health-card">
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic
                    title={t('admin.providers.uptime', 'Uptime')}
                    value={health.uptime}
                    suffix="%"
                    precision={2}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title={t('admin.providers.latency', 'Latency')}
                    value={health.latency}
                    suffix="ms"
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title={t('admin.providers.last_check', 'Last Check')}
                    value={new Date(health.lastCheck).toLocaleTimeString()}
                  />
                </Col>
              </Row>
              
              {health.recentErrors.length > 0 && (
                <div className="recent-errors">
                  <Text strong>{t('admin.providers.recent_errors', 'Recent Errors')}</Text>
                  <List
                    size="small"
                    dataSource={health.recentErrors.slice(0, 5)}
                    renderItem={(error) => (
                      <List.Item>
                        <Text type="danger">{error.message}</Text>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {new Date(error.timestamp).toLocaleString()}
                        </Text>
                      </List.Item>
                    )}
                  />
                </div>
              )}
            </Card>
          )}
        </div>
      ),
    },
    {
      key: 'config',
      label: <><SettingOutlined /> {t('admin.providers.configuration', 'Configuration')}</>,
      children: (
        <div className="provider-config">
          <Form
            form={configForm}
            layout="vertical"
            initialValues={{
              name: provider.name,
              priority: provider.priority,
              isDefault: provider.isDefault,
              status: provider.status,
            }}
            onFinish={handleConfigSave}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item name="name" label={t('admin.providers.name', 'Name')}>
                  <Input />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="priority" label={t('admin.providers.priority', 'Priority')}>
                  <InputNumber min={1} max={10} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item name="status" label={t('admin.providers.status', 'Status')}>
                  <Select
                    options={[
                      { value: 'active', label: 'Active' },
                      { value: 'inactive', label: 'Inactive' },
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="isDefault" label={t('admin.providers.default', 'Default Provider')} valuePropName="checked">
                  <Switch />
                </Form.Item>
              </Col>
            </Row>
            <Form.Item>
              <Button type="primary" htmlType="submit" loading={updateProvider.isPending}>
                {t('common.save', 'Save')}
              </Button>
            </Form.Item>
          </Form>

          <Divider />

          {/* API Key */}
          <Title level={5}><KeyOutlined /> {t('admin.providers.api_key', 'API Key')}</Title>
          {!showApiKey ? (
            <Space>
              <Text>
                {provider.apiKeyConfigured ? '••••••••••••••••' : t('admin.providers.no_api_key', 'No API key configured')}
              </Text>
              <Button icon={<EditOutlined />} onClick={() => setShowApiKey(true)}>
                {t('admin.providers.update_key', 'Update')}
              </Button>
            </Space>
          ) : (
            <Form form={apiKeyForm} layout="inline" onFinish={handleApiKeySave}>
              <Form.Item name="apiKey" rules={[{ required: true }]}>
                <Input.Password placeholder="Enter API key" style={{ width: 300 }} />
              </Form.Item>
              <Form.Item>
                <Space>
                  <Button type="primary" htmlType="submit" loading={updateApiKey.isPending}>
                    {t('common.save', 'Save')}
                  </Button>
                  <Button onClick={() => setShowApiKey(false)}>
                    {t('common.cancel', 'Cancel')}
                  </Button>
                </Space>
              </Form.Item>
            </Form>
          )}
        </div>
      ),
    },
    {
      key: 'models',
      label: <><RobotOutlined /> {t('admin.providers.models', 'Models')}</>,
      children: (
        <Table
          columns={modelColumns}
          dataSource={models || []}
          rowKey="id"
          size="small"
          pagination={false}
        />
      ),
    },
    {
      key: 'metrics',
      label: <><LineChartOutlined /> {t('admin.providers.metrics', 'Metrics')}</>,
      children: metrics ? (
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Card title={t('admin.providers.response_time', 'Response Time')} size="small">
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={metrics.responseTimeTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" fontSize={12} />
                  <YAxis fontSize={12} />
                  <RechartsTooltip />
                  <Line type="monotone" dataKey="value" stroke="#1890ff" strokeWidth={2} name="Response Time (ms)" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>
          <Col span={12}>
            <Card title={t('admin.providers.error_rate', 'Error Rate')} size="small">
              <ResponsiveContainer width="100%" height={150}>
                <AreaChart data={metrics.errorRateTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" fontSize={10} />
                  <YAxis fontSize={10} />
                  <RechartsTooltip />
                  <Area type="monotone" dataKey="value" stroke="#f5222d" fill="#fff1f0" name="Error Rate (%)" />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </Col>
          <Col span={12}>
            <Card title={t('admin.providers.cost_trend', 'Cost Trend')} size="small">
              <ResponsiveContainer width="100%" height={150}>
                <AreaChart data={metrics.costTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" fontSize={10} />
                  <YAxis fontSize={10} />
                  <RechartsTooltip />
                  <Area type="monotone" dataKey="value" stroke="#52c41a" fill="#f6ffed" name="Cost ($)" />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>
      ) : (
        <Empty description={t('admin.providers.no_metrics', 'No metrics available')} />
      ),
    },
  ];

  return (
    <div className="provider-config-panel">
      <div className="panel-header">
        <Space>
          <Avatar
            icon={providerIcons[provider.type]}
            style={{ backgroundColor: providerColors[provider.type] }}
          />
          <div>
            <Title level={4} style={{ margin: 0 }}>{provider.name}</Title>
            <Text type="secondary">{provider.type}</Text>
          </div>
        </Space>
      </div>
      <Tabs items={tabItems} />
    </div>
  );
};

/**
 * Main Provider Management Component
 */
export const ProviderManagement: React.FC = () => {
  const { t } = useTranslation();
  const { selectedProviderId, selectProvider } = useAdminStore();
  
  const { data: providers, isLoading, refetch, isFetching } = useProviders();
  const testProvider = useTestProvider();

  const handleTest = useCallback((providerId: string) => {
    testProvider.mutate(providerId);
  }, [testProvider]);

  // Summary stats - ensure providers is always an array
  const providerList = Array.isArray(providers) ? providers : [];
  
  const stats = useMemo(() => {
    if (!providerList.length) return null;
    return {
      total: providerList.length,
      active: providerList.filter(p => p.status === 'active').length,
      totalRequests: providerList.reduce((sum, p) => sum + (p.requestsToday || 0), 0),
      totalCost: providerList.reduce((sum, p) => sum + (p.costToday || 0), 0),
    };
  }, [providerList]);

  return (
    <div className="provider-management" role="main" aria-label={t('admin.providers.title', 'Provider Management')}>
      <div className="page-header">
        <Title level={2}><ApiOutlined /> {t('admin.providers.title', 'AI Provider Management')}</Title>
        <Button 
          icon={<ReloadOutlined />} 
          onClick={() => refetch()}
          loading={isFetching}
        >
          {t('common.refresh', 'Refresh')}
        </Button>
      </div>

      {/* Summary Stats */}
      {stats && (
        <Row gutter={16} className="provider-stats">
          <Col xs={12} sm={6}>
            <Card>
              <Statistic
                title={t('admin.providers.total', 'Total Providers')}
                value={stats.total}
                prefix={<ApiOutlined />}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card>
              <Statistic
                title={t('admin.providers.active', 'Active')}
                value={stats.active}
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card>
              <Statistic
                title={t('admin.providers.requests_today', 'Requests Today')}
                value={stats.totalRequests}
                prefix={<ThunderboltOutlined />}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card>
              <Statistic
                title={t('admin.providers.cost_today', 'Cost Today')}
                value={stats.totalCost}
                prefix={<DollarOutlined />}
                precision={2}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* Main Content */}
      <Row gutter={24}>
        {/* Provider List */}
        <Col xs={24} lg={8}>
          <Card 
            title={t('admin.providers.providers', 'Providers')} 
            className="providers-list-card"
            loading={isLoading}
          >
            {providerList.map(provider => (
              <ProviderCard
                key={provider.id}
                provider={provider}
                selected={selectedProviderId === provider.id}
                onSelect={() => selectProvider(provider.id)}
                onTest={() => handleTest(provider.id)}
                testing={testProvider.isPending}
              />
            ))}
            {!providerList.length && (
              <Empty description={t('admin.providers.no_providers', 'No providers configured')} />
            )}
          </Card>
        </Col>

        {/* Configuration Panel */}
        <Col xs={24} lg={16}>
          <Card className="config-panel-card">
            {selectedProviderId ? (
              <ProviderConfigPanel providerId={selectedProviderId} />
            ) : (
              <Empty description={t('admin.providers.select', 'Select a provider to configure')} />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default ProviderManagement;
