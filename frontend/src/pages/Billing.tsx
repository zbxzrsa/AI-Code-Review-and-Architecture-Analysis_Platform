/**
 * Billing & Usage Page
 * 账单与使用页�?
 * 
 * Features:
 * - Usage statistics and quotas
 * - AI token consumption tracking
 * - Cost breakdown
 * - Plan management
 */

import React from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Progress,
  Statistic,
  Table,
  Tag,
  Button,
  Alert,
  Tabs,
  List,
  Divider,
} from 'antd';
import type { TableProps } from 'antd';
import {
  DollarOutlined,
  ThunderboltOutlined,
  ApiOutlined,
  CloudOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  CrownOutlined,
  DownloadOutlined,
  BarChartOutlined,
  RiseOutlined,
  CalendarOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text } = Typography;

interface UsageRecord {
  id: string;
  date: string;
  type: 'analysis' | 'chat' | 'auto-fix' | 'api';
  tokens: number;
  cost: number;
  project?: string;
}

const mockUsageRecords: UsageRecord[] = [
  { id: '1', date: '2024-03-01', type: 'analysis', tokens: 45000, cost: 0.45, project: 'Frontend' },
  { id: '2', date: '2024-03-01', type: 'chat', tokens: 12000, cost: 0.12, project: 'Backend' },
  { id: '3', date: '2024-03-01', type: 'auto-fix', tokens: 8500, cost: 0.085, project: 'Backend' },
  { id: '4', date: '2024-02-29', type: 'analysis', tokens: 52000, cost: 0.52, project: 'Mobile' },
  { id: '5', date: '2024-02-29', type: 'api', tokens: 25000, cost: 0.25 },
  { id: '6', date: '2024-02-28', type: 'analysis', tokens: 38000, cost: 0.38, project: 'Frontend' },
  { id: '7', date: '2024-02-28', type: 'chat', tokens: 15000, cost: 0.15, project: 'Infrastructure' },
];

const currentPlan = {
  name: 'Professional',
  price: 99,
  tokensIncluded: 1000000,
  tokensUsed: 685000,
  analysesIncluded: 500,
  analysesUsed: 342,
  apiCallsIncluded: 10000,
  apiCallsUsed: 4521,
  nextBillingDate: '2024-04-01',
  features: [
    'Unlimited repositories',
    'AI-powered code review',
    'Auto-fix suggestions',
    'Priority support',
    'Custom rules',
    'Team collaboration',
  ],
};

const plans = [
  {
    name: 'Free',
    price: 0,
    tokens: 50000,
    analyses: 25,
    apiCalls: 100,
    features: ['1 repository', 'Basic code review', 'Community support'],
    current: false,
  },
  {
    name: 'Professional',
    price: 99,
    tokens: 1000000,
    analyses: 500,
    apiCalls: 10000,
    features: ['Unlimited repos', 'AI code review', 'Auto-fix', 'Priority support'],
    current: true,
    popular: true,
  },
  {
    name: 'Enterprise',
    price: 499,
    tokens: 10000000,
    analyses: 'Unlimited',
    apiCalls: 'Unlimited',
    features: ['Everything in Pro', 'SSO/SAML', 'Custom SLA', 'Dedicated support'],
    current: false,
  },
];

export const Billing: React.FC = () => {
  const { t: _t } = useTranslation();

  const tokenPercentage = Math.round((currentPlan.tokensUsed / currentPlan.tokensIncluded) * 100);
  const analysisPercentage = Math.round((currentPlan.analysesUsed / currentPlan.analysesIncluded) * 100);
  const apiPercentage = Math.round((currentPlan.apiCallsUsed / currentPlan.apiCallsIncluded) * 100);

  const usageColumns: TableProps<UsageRecord>['columns'] = [
    {
      title: 'Date',
      dataIndex: 'date',
      render: (date) => new Date(date).toLocaleDateString(),
    },
    {
      title: 'Type',
      dataIndex: 'type',
      render: (type) => {
        const config: Record<string, { color: string; icon: React.ReactNode }> = {
          analysis: { color: 'blue', icon: <BarChartOutlined /> },
          chat: { color: 'green', icon: <ApiOutlined /> },
          'auto-fix': { color: 'purple', icon: <ThunderboltOutlined /> },
          api: { color: 'orange', icon: <CloudOutlined /> },
        };
        const c = config[type];
        return <Tag color={c.color} icon={c.icon}>{type.charAt(0).toUpperCase() + type.slice(1)}</Tag>;
      },
    },
    {
      title: 'Project',
      dataIndex: 'project',
      render: (project) => project || <Text type="secondary">-</Text>,
    },
    {
      title: 'Tokens',
      dataIndex: 'tokens',
      render: (tokens) => tokens.toLocaleString(),
    },
    {
      title: 'Cost',
      dataIndex: 'cost',
      render: (cost) => `$${cost.toFixed(2)}`,
    },
  ];

  return (
    <div className="billing-page" style={{ maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <DollarOutlined style={{ color: '#2563eb' }} /> Billing & Usage
          </Title>
          <Text type="secondary">Monitor your usage and manage subscription</Text>
        </div>
        <Space>
          <Button icon={<DownloadOutlined />}>Download Invoice</Button>
          <Button type="primary" icon={<CrownOutlined />}>Upgrade Plan</Button>
        </Space>
      </div>

      {/* Usage Warning */}
      {tokenPercentage > 80 && (
        <Alert
          type="warning"
          showIcon
          icon={<WarningOutlined />}
          message="Approaching Token Limit"
          description={`You've used ${tokenPercentage}% of your monthly token quota. Consider upgrading your plan to avoid service interruption.`}
          style={{ marginBottom: 24 }}
          action={<Button type="primary" size="small">Upgrade</Button>}
        />
      )}

      {/* Current Plan Overview */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={8}>
          <Card 
            style={{ 
              borderRadius: 12, 
              background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
              border: 'none',
            }}
          >
            <div style={{ color: 'white' }}>
              <Space>
                <CrownOutlined style={{ fontSize: 24 }} />
                <Text style={{ color: 'white', fontSize: 18, fontWeight: 600 }}>{currentPlan.name} Plan</Text>
              </Space>
              <Title level={2} style={{ color: 'white', margin: '16px 0 8px' }}>
                ${currentPlan.price}<Text style={{ color: 'rgba(255,255,255,0.8)', fontSize: 16 }}>/month</Text>
              </Title>
              <Text style={{ color: 'rgba(255,255,255,0.8)' }}>
                <CalendarOutlined /> Next billing: {currentPlan.nextBillingDate}
              </Text>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={16}>
          <Card title="Current Usage" style={{ borderRadius: 12 }}>
            <Row gutter={24}>
              <Col span={8}>
                <div style={{ textAlign: 'center' }}>
                  <Progress
                    type="dashboard"
                    percent={tokenPercentage}
                    strokeColor={tokenPercentage > 80 ? '#ef4444' : '#2563eb'}
                    format={() => (
                      <div>
                        <div style={{ fontSize: 20, fontWeight: 600 }}>{tokenPercentage}%</div>
                        <div style={{ fontSize: 12, color: '#64748b' }}>Tokens</div>
                      </div>
                    )}
                  />
                  <Text type="secondary">
                    {(currentPlan.tokensUsed / 1000).toFixed(0)}K / {(currentPlan.tokensIncluded / 1000).toFixed(0)}K
                  </Text>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center' }}>
                  <Progress
                    type="dashboard"
                    percent={analysisPercentage}
                    strokeColor={analysisPercentage > 80 ? '#ef4444' : '#22c55e'}
                    format={() => (
                      <div>
                        <div style={{ fontSize: 20, fontWeight: 600 }}>{analysisPercentage}%</div>
                        <div style={{ fontSize: 12, color: '#64748b' }}>Analyses</div>
                      </div>
                    )}
                  />
                  <Text type="secondary">
                    {currentPlan.analysesUsed} / {currentPlan.analysesIncluded}
                  </Text>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center' }}>
                  <Progress
                    type="dashboard"
                    percent={apiPercentage}
                    strokeColor={apiPercentage > 80 ? '#ef4444' : '#8b5cf6'}
                    format={() => (
                      <div>
                        <div style={{ fontSize: 20, fontWeight: 600 }}>{apiPercentage}%</div>
                        <div style={{ fontSize: 12, color: '#64748b' }}>API Calls</div>
                      </div>
                    )}
                  />
                  <Text type="secondary">
                    {currentPlan.apiCallsUsed.toLocaleString()} / {currentPlan.apiCallsIncluded.toLocaleString()}
                  </Text>
                </div>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      {/* Cost Breakdown */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="This Month"
              value={32.45}
              prefix="$"
              valueStyle={{ color: '#2563eb' }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Last Month"
              value={28.90}
              prefix="$"
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Total Analyses"
              value={342}
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Cost Trend"
              value={12}
              suffix="%"
              prefix={<RiseOutlined />}
              valueStyle={{ color: '#f59e0b' }}
            />
          </Card>
        </Col>
      </Row>

      <Tabs
        defaultActiveKey="usage"
        items={[
          {
            key: 'usage',
            label: 'Usage History',
            children: (
              <Card style={{ borderRadius: 12 }}>
                <Table
                  columns={usageColumns}
                  dataSource={mockUsageRecords}
                  rowKey="id"
                  pagination={{ pageSize: 10 }}
                />
              </Card>
            ),
          },
          {
            key: 'plans',
            label: 'Available Plans',
            children: (
              <Row gutter={16}>
                {plans.map(plan => (
                  <Col key={plan.name} xs={24} md={8}>
                    <Card
                      style={{
                        borderRadius: 12,
                        border: plan.current ? '2px solid #2563eb' : '1px solid #e2e8f0',
                        position: 'relative',
                      }}
                    >
                      {plan.popular && (
                        <div style={{
                          position: 'absolute',
                          top: -12,
                          right: 16,
                          background: '#2563eb',
                          color: 'white',
                          padding: '4px 12px',
                          borderRadius: 12,
                          fontSize: 12,
                          fontWeight: 600,
                        }}>
                          Popular
                        </div>
                      )}
                      <div style={{ textAlign: 'center', marginBottom: 16 }}>
                        <Title level={4} style={{ margin: 0 }}>{plan.name}</Title>
                        <Title level={2} style={{ margin: '8px 0' }}>
                          ${plan.price}
                          <Text type="secondary" style={{ fontSize: 14 }}>/month</Text>
                        </Title>
                      </div>
                      <Divider />
                      <List
                        size="small"
                        dataSource={[
                          `${typeof plan.tokens === 'number' ? (plan.tokens / 1000).toFixed(0) + 'K' : plan.tokens} tokens`,
                          `${plan.analyses} analyses`,
                          `${typeof plan.apiCalls === 'number' ? plan.apiCalls.toLocaleString() : plan.apiCalls} API calls`,
                          ...plan.features,
                        ]}
                        renderItem={item => (
                          <List.Item style={{ padding: '8px 0', border: 'none' }}>
                            <Space>
                              <CheckCircleOutlined style={{ color: '#22c55e' }} />
                              {item}
                            </Space>
                          </List.Item>
                        )}
                      />
                      <Button
                        type={plan.current ? 'default' : 'primary'}
                        block
                        style={{ marginTop: 16 }}
                        disabled={plan.current}
                      >
                        {plan.current ? 'Current Plan' : 'Upgrade'}
                      </Button>
                    </Card>
                  </Col>
                ))}
              </Row>
            ),
          },
        ]}
      />
    </div>
  );
};

export default Billing;
