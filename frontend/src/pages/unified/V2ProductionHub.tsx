/**
 * V2 Production Hub
 *
 * Centralized dashboard for all V2 production functions.
 * Main interface for regular users with optimized navigation.
 */

import React, { useState, useMemo, useCallback, useEffect } from 'react';
import {
  Layout,
  Card,
  Row,
  Col,
  Input,
  Typography,
  Space,
  Tabs,
  Tag,
  Progress,
  Statistic,
  List,
  Avatar,
  Button,
  Tooltip,
  Badge,
  Divider,
  Timeline,
  Empty,
} from 'antd';
import {
  CodeOutlined,
  SafetyOutlined,
  BarChartOutlined,
  SearchOutlined,
  AppstoreOutlined,
  RocketOutlined,
  BugOutlined,
  TeamOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  FireOutlined,
  StarOutlined,
  ArrowRightOutlined,
  DatabaseOutlined,
  SyncOutlined,
  AimOutlined,
  AuditOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  AlertOutlined,
  LineChartOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const { Content } = Layout;
const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

// =============================================================================
// V2 Production Function Categories
// =============================================================================

interface QuickAction {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  path: string;
  color: string;
  stats?: {
    value: number;
    label: string;
    trend?: 'up' | 'down' | 'stable';
  };
}

const DEVELOPMENT_FUNCTIONS: QuickAction[] = [
  {
    id: 'code-review',
    name: 'Code Review',
    description: 'AI-powered code analysis',
    icon: <CodeOutlined />,
    path: '/review',
    color: '#1890ff',
    stats: { value: 156, label: 'Reviews Today', trend: 'up' },
  },
  {
    id: 'projects',
    name: 'Projects',
    description: 'Manage your projects',
    icon: <AppstoreOutlined />,
    path: '/projects',
    color: '#722ed1',
    stats: { value: 12, label: 'Active Projects' },
  },
  {
    id: 'repositories',
    name: 'Repositories',
    description: 'Connected repositories',
    icon: <DatabaseOutlined />,
    path: '/repositories',
    color: '#13c2c2',
    stats: { value: 8, label: 'Repositories' },
  },
  {
    id: 'pull-requests',
    name: 'Pull Requests',
    description: 'Review and merge PRs',
    icon: <SyncOutlined />,
    path: '/pull-requests',
    color: '#eb2f96',
    stats: { value: 5, label: 'Pending PRs', trend: 'down' },
  },
];

const INSIGHT_FUNCTIONS: QuickAction[] = [
  {
    id: 'analytics',
    name: 'Analytics',
    description: 'Code quality trends',
    icon: <BarChartOutlined />,
    path: '/analytics',
    color: '#52c41a',
    stats: { value: 94, label: 'Quality Score', trend: 'up' },
  },
  {
    id: 'reports',
    name: 'Reports',
    description: 'Generate reports',
    icon: <AuditOutlined />,
    path: '/reports',
    color: '#fa8c16',
    stats: { value: 24, label: 'Reports This Week' },
  },
  {
    id: 'metrics',
    name: 'Code Metrics',
    description: 'Detailed metrics',
    icon: <AimOutlined />,
    path: '/metrics',
    color: '#2f54eb',
    stats: { value: 87, label: 'Coverage %' },
  },
];

const SECURITY_FUNCTIONS: QuickAction[] = [
  {
    id: 'security',
    name: 'Security Dashboard',
    description: 'Vulnerability overview',
    icon: <SafetyOutlined />,
    path: '/security',
    color: '#f5222d',
    stats: { value: 3, label: 'Open Issues', trend: 'down' },
  },
];

const CONFIG_FUNCTIONS: QuickAction[] = [
  {
    id: 'rules',
    name: 'Quality Rules',
    description: 'Configure rules',
    icon: <CheckCircleOutlined />,
    path: '/rules',
    color: '#a0d911',
  },
  {
    id: 'teams',
    name: 'Teams',
    description: 'Team management',
    icon: <TeamOutlined />,
    path: '/teams',
    color: '#1890ff',
  },
  {
    id: 'settings',
    name: 'Settings',
    description: 'User settings',
    icon: <SettingOutlined />,
    path: '/settings',
    color: '#8c8c8c',
  },
];

// =============================================================================
// Quick Action Card
// =============================================================================

interface QuickActionCardProps {
  action: QuickAction;
  onNavigate: (path: string) => void;
  size?: 'default' | 'large';
}

const QuickActionCard: React.FC<QuickActionCardProps> = ({
  action,
  onNavigate,
  size = 'default',
}) => {
  const isLarge = size === 'large';

  return (
    <Card
      hoverable
      onClick={() => onNavigate(action.path)}
      style={{
        height: '100%',
        borderTop: `3px solid ${action.color}`,
      }}
      bodyStyle={{
        padding: isLarge ? 24 : 16,
      }}
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        <Space align="start" style={{ width: '100%', justifyContent: 'space-between' }}>
          <div
            style={{
              width: isLarge ? 56 : 44,
              height: isLarge ? 56 : 44,
              borderRadius: 12,
              background: `${action.color}15`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: isLarge ? 28 : 22,
              color: action.color,
            }}
          >
            {action.icon}
          </div>
          {action.stats && (
            <div style={{ textAlign: 'right' }}>
              <Text
                strong
                style={{
                  fontSize: isLarge ? 24 : 18,
                  color: action.color,
                }}
              >
                {action.stats.value}
              </Text>
              {action.stats.trend && (
                <span style={{
                  marginLeft: 4,
                  color: action.stats.trend === 'up' ? '#52c41a' :
                         action.stats.trend === 'down' ? '#f5222d' : '#8c8c8c',
                }}>
                  {action.stats.trend === 'up' ? '↑' :
                   action.stats.trend === 'down' ? '↓' : '→'}
                </span>
              )}
              <br />
              <Text type="secondary" style={{ fontSize: 11 }}>
                {action.stats.label}
              </Text>
            </div>
          )}
        </Space>

        <div>
          <Text strong style={{ fontSize: isLarge ? 16 : 14 }}>
            {action.name}
          </Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {action.description}
          </Text>
        </div>

        <Button
          type="link"
          size="small"
          style={{ padding: 0, color: action.color }}
        >
          Open <ArrowRightOutlined />
        </Button>
      </Space>
    </Card>
  );
};

// =============================================================================
// Recent Activity Item
// =============================================================================

interface RecentActivity {
  id: string;
  type: 'review' | 'commit' | 'pr' | 'alert';
  title: string;
  description: string;
  time: string;
  status: 'success' | 'warning' | 'error' | 'info';
}

const MOCK_ACTIVITIES: RecentActivity[] = [
  {
    id: '1',
    type: 'review',
    title: 'Code review completed',
    description: 'api/auth.py - 3 issues found',
    time: '5 min ago',
    status: 'warning',
  },
  {
    id: '2',
    type: 'pr',
    title: 'PR #142 merged',
    description: 'Feature: Add user authentication',
    time: '12 min ago',
    status: 'success',
  },
  {
    id: '3',
    type: 'alert',
    title: 'Security alert resolved',
    description: 'SQL injection vulnerability fixed',
    time: '1 hour ago',
    status: 'success',
  },
  {
    id: '4',
    type: 'commit',
    title: 'New commit pushed',
    description: 'main branch - 5 files changed',
    time: '2 hours ago',
    status: 'info',
  },
];

// =============================================================================
// Main Component
// =============================================================================

const V2ProductionHub: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();

  const [searchQuery, setSearchQuery] = useState('');

  // Filter functions by search
  const allFunctions = useMemo(() => [
    ...DEVELOPMENT_FUNCTIONS,
    ...INSIGHT_FUNCTIONS,
    ...SECURITY_FUNCTIONS,
    ...CONFIG_FUNCTIONS,
  ], []);

  const filteredFunctions = useMemo(() => {
    if (!searchQuery) return null;
    const query = searchQuery.toLowerCase();
    return allFunctions.filter(f =>
      f.name.toLowerCase().includes(query) ||
      f.description.toLowerCase().includes(query)
    );
  }, [searchQuery, allFunctions]);

  const handleNavigate = useCallback((path: string) => {
    navigate(path);
  }, [navigate]);

  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
      <Content style={{ padding: 24 }}>
        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <Space align="center" style={{ marginBottom: 8 }}>
            <Tag color="green" style={{ fontSize: 14, padding: '4px 12px' }}>
              <CheckCircleOutlined /> V2 Production
            </Tag>
            <Text type="secondary">Stable features for all users</Text>
          </Space>

          <Title level={2} style={{ margin: 0 }}>
            Production Hub
          </Title>
          <Paragraph type="secondary" style={{ marginBottom: 0 }}>
            Access all production-ready functions from one place
          </Paragraph>
        </div>

        {/* Search */}
        <Card style={{ marginBottom: 24 }}>
          <Input
            placeholder="Quick search functions... (code review, analytics, security)"
            prefix={<SearchOutlined style={{ color: '#bfbfbf' }} />}
            size="large"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            allowClear
          />

          {/* Search Results */}
          {filteredFunctions && (
            <div style={{ marginTop: 16 }}>
              {filteredFunctions.length > 0 ? (
                <Row gutter={[16, 16]}>
                  {filteredFunctions.map(f => (
                    <Col xs={24} sm={12} md={8} lg={6} key={f.id}>
                      <QuickActionCard action={f} onNavigate={handleNavigate} />
                    </Col>
                  ))}
                </Row>
              ) : (
                <Empty description="No functions found" />
              )}
            </div>
          )}
        </Card>

        {/* Main Content - Only show when not searching */}
        {!filteredFunctions && (
          <>
            {/* Quick Stats */}
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col xs={24} sm={12} md={6}>
                <Card>
                  <Statistic
                    title="Code Quality Score"
                    value={94}
                    suffix="%"
                    prefix={<LineChartOutlined />}
                    valueStyle={{ color: '#52c41a' }}
                  />
                  <Progress percent={94} showInfo={false} strokeColor="#52c41a" />
                </Card>
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Card>
                  <Statistic
                    title="Reviews Today"
                    value={156}
                    prefix={<CodeOutlined />}
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Card>
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Card>
                  <Statistic
                    title="Open Security Issues"
                    value={3}
                    prefix={<AlertOutlined />}
                    valueStyle={{ color: '#f5222d' }}
                  />
                </Card>
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Card>
                  <Statistic
                    title="Active Projects"
                    value={12}
                    prefix={<AppstoreOutlined />}
                    valueStyle={{ color: '#722ed1' }}
                  />
                </Card>
              </Col>
            </Row>

            {/* Development Functions */}
            <Card
              title={
                <Space>
                  <CodeOutlined style={{ color: '#1890ff' }} />
                  <span>Development</span>
                </Space>
              }
              style={{ marginBottom: 24 }}
            >
              <Row gutter={[16, 16]}>
                {DEVELOPMENT_FUNCTIONS.map(action => (
                  <Col xs={24} sm={12} md={6} key={action.id}>
                    <QuickActionCard
                      action={action}
                      onNavigate={handleNavigate}
                      size="large"
                    />
                  </Col>
                ))}
              </Row>
            </Card>

            <Row gutter={24}>
              <Col xs={24} lg={16}>
                {/* Insights */}
                <Card
                  title={
                    <Space>
                      <BarChartOutlined style={{ color: '#52c41a' }} />
                      <span>Insights & Analytics</span>
                    </Space>
                  }
                  style={{ marginBottom: 24 }}
                >
                  <Row gutter={[16, 16]}>
                    {INSIGHT_FUNCTIONS.map(action => (
                      <Col xs={24} sm={12} md={8} key={action.id}>
                        <QuickActionCard action={action} onNavigate={handleNavigate} />
                      </Col>
                    ))}
                  </Row>
                </Card>

                {/* Security */}
                <Card
                  title={
                    <Space>
                      <SafetyOutlined style={{ color: '#f5222d' }} />
                      <span>Security</span>
                      <Badge count={3} />
                    </Space>
                  }
                  style={{ marginBottom: 24 }}
                >
                  <Row gutter={[16, 16]}>
                    {SECURITY_FUNCTIONS.map(action => (
                      <Col xs={24} sm={12} key={action.id}>
                        <QuickActionCard action={action} onNavigate={handleNavigate} />
                      </Col>
                    ))}
                  </Row>
                </Card>

                {/* Configuration */}
                <Card
                  title={
                    <Space>
                      <SettingOutlined style={{ color: '#8c8c8c' }} />
                      <span>Configuration</span>
                    </Space>
                  }
                >
                  <Row gutter={[16, 16]}>
                    {CONFIG_FUNCTIONS.map(action => (
                      <Col xs={24} sm={8} key={action.id}>
                        <QuickActionCard action={action} onNavigate={handleNavigate} />
                      </Col>
                    ))}
                  </Row>
                </Card>
              </Col>

              <Col xs={24} lg={8}>
                {/* Recent Activity */}
                <Card
                  title={
                    <Space>
                      <ClockCircleOutlined />
                      <span>Recent Activity</span>
                    </Space>
                  }
                  extra={<Button type="link">View All</Button>}
                >
                  <Timeline>
                    {MOCK_ACTIVITIES.map(activity => (
                      <Timeline.Item
                        key={activity.id}
                        color={
                          activity.status === 'success' ? 'green' :
                          activity.status === 'warning' ? 'orange' :
                          activity.status === 'error' ? 'red' : 'blue'
                        }
                      >
                        <Text strong>{activity.title}</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {activity.description}
                        </Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: 11 }}>
                          {activity.time}
                        </Text>
                      </Timeline.Item>
                    ))}
                  </Timeline>
                </Card>

                {/* Quick Links */}
                <Card
                  title={
                    <Space>
                      <ThunderboltOutlined style={{ color: '#faad14' }} />
                      <span>Quick Actions</span>
                    </Space>
                  }
                  style={{ marginTop: 16 }}
                >
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Button
                      type="primary"
                      block
                      icon={<CodeOutlined />}
                      onClick={() => handleNavigate('/review')}
                    >
                      New Code Review
                    </Button>
                    <Button
                      block
                      icon={<AppstoreOutlined />}
                      onClick={() => handleNavigate('/projects/new')}
                    >
                      Create Project
                    </Button>
                    <Button
                      block
                      icon={<AuditOutlined />}
                      onClick={() => handleNavigate('/reports')}
                    >
                      Generate Report
                    </Button>
                  </Space>
                </Card>
              </Col>
            </Row>
          </>
        )}
      </Content>
    </Layout>
  );
};

export default V2ProductionHub;
