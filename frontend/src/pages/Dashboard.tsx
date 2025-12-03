import React, { useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Row,
  Col,
  Card,
  Statistic,
  Table,
  Button,
  Typography,
  Space,
  Tag,
  Progress,
  List,
  Avatar,
  Skeleton,
  Empty,
  Alert,
  Tooltip,
  Dropdown,
} from 'antd';
import {
  ProjectOutlined,
  BugOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  PlusOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  ReloadOutlined,
  MoreOutlined,
  EyeOutlined,
  SettingOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../hooks/useAuth';
import { apiService } from '../services/api';
import { useAsyncData } from '../hooks/useAsyncData';
import { getRelativeTime } from '../utils/formatters';
import { SelfEvolutionWidget, CodeQualityWidget, QuickActionsWidget } from '../components/dashboard';
import './Dashboard.css';

const { Title, Text } = Typography;

interface DashboardMetrics {
  total_projects: number;
  total_analyses: number;
  issues_found: number;
  issues_resolved: number;
  avg_resolution_time: number;
  trend: {
    projects: number;
    analyses: number;
    issues: number;
  };
}

interface RecentProject {
  id: string;
  name: string;
  language: string;
  last_analysis: string;
  issues_count: number;
  status: 'healthy' | 'warning' | 'critical';
}

interface RecentActivity {
  id: string;
  type: 'analysis' | 'fix' | 'review' | 'comment';
  project: string;
  description: string;
  timestamp: string;
  user: {
    name: string;
    avatar?: string;
  };
}

export const Dashboard: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { user } = useAuth();

  // ============================================
  // Data Fetching with useAsyncData
  // ============================================
  
  const {
    data: metrics,
    loading: metricsLoading,
    error: metricsError,
    refresh: refreshMetrics,
  } = useAsyncData<DashboardMetrics>(
    async () => {
      const response = await apiService.metrics.getDashboard();
      return response.data;
    },
    { immediate: true, retryCount: 2 }
  );

  const {
    data: recentProjects,
    loading: projectsLoading,
    error: projectsError,
    refresh: refreshProjects,
  } = useAsyncData<RecentProject[]>(
    async () => {
      const response = await apiService.projects.list({ limit: 5 });
      return response.data.items || [];
    },
    { immediate: true, retryCount: 2 }
  );

  // Mock activity data (replace with real API when available)
  const recentActivity = useMemo<RecentActivity[]>(() => [
    {
      id: '1',
      type: 'analysis',
      project: 'backend-api',
      description: 'Completed security analysis',
      timestamp: new Date().toISOString(),
      user: { name: user?.name || 'User' }
    },
    {
      id: '2',
      type: 'fix',
      project: 'frontend-app',
      description: 'Applied 3 security fixes',
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      user: { name: user?.name || 'User' }
    },
    {
      id: '3',
      type: 'review',
      project: 'api-gateway',
      description: 'Code review completed',
      timestamp: new Date(Date.now() - 7200000).toISOString(),
      user: { name: 'Jane Smith' }
    },
  ], [user?.name]);

  // Combined loading state
  const loading = metricsLoading || projectsLoading;

  // Refresh all data
  const handleRefreshAll = useCallback(async () => {
    await Promise.all([refreshMetrics(), refreshProjects()]);
  }, [refreshMetrics, refreshProjects]);

  // Calculate resolution rate
  const resolutionRate = useMemo(() => {
    if (!metrics?.issues_found) return 0;
    return Math.round((metrics.issues_resolved / metrics.issues_found) * 100);
  }, [metrics]);

  const projectColumns = [
    {
      title: t('dashboard.project_name', 'Project'),
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: RecentProject) => (
        <Space>
          <ProjectOutlined />
          <a onClick={() => navigate(`/projects/${record.id}`)}>{name}</a>
        </Space>
      )
    },
    {
      title: t('dashboard.language', 'Language'),
      dataIndex: 'language',
      key: 'language',
      render: (lang: string) => <Tag>{lang}</Tag>
    },
    {
      title: t('dashboard.issues', 'Issues'),
      dataIndex: 'issues_count',
      key: 'issues_count',
      render: (count: number) => (
        <Tag color={count > 10 ? 'red' : count > 5 ? 'orange' : 'green'}>
          {count}
        </Tag>
      )
    },
    {
      title: t('dashboard.status', 'Status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          healthy: 'green',
          warning: 'orange',
          critical: 'red'
        };
        return <Tag color={colors[status as keyof typeof colors]}>{status}</Tag>;
      }
    }
  ];

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'analysis':
        return <BugOutlined style={{ color: '#1890ff' }} />;
      case 'fix':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'review':
        return <ProjectOutlined style={{ color: '#722ed1' }} />;
      default:
        return <ClockCircleOutlined />;
    }
  };

  // Error state
  const hasError = metricsError || projectsError;

  return (
    <div className="dashboard-container">
      {/* Error Alert */}
      {hasError && (
        <Alert
          type="warning"
          message={t('dashboard.data_error', 'Some data could not be loaded')}
          description={t('dashboard.data_error_desc', 'Click refresh to try again')}
          showIcon
          icon={<ExclamationCircleOutlined />}
          action={
            <Button size="small" onClick={handleRefreshAll} icon={<ReloadOutlined />}>
              {t('common.retry', 'Retry')}
            </Button>
          }
          style={{ marginBottom: 16 }}
          closable
        />
      )}

      <div className="dashboard-header">
        <div>
          <Title level={2} style={{ marginBottom: 4 }}>
            {t('dashboard.welcome', 'Welcome back')}, {user?.name}!
          </Title>
          <Text type="secondary">
            {t('dashboard.subtitle', "Here's what's happening with your projects")}
          </Text>
        </div>
        <Space>
          <Tooltip title={t('dashboard.refresh', 'Refresh data')}>
            <Button
              icon={<ReloadOutlined spin={loading} />}
              onClick={handleRefreshAll}
              disabled={loading}
            />
          </Tooltip>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => navigate('/projects/new')}
          >
            {t('dashboard.new_project', 'New Project')}
          </Button>
        </Space>
      </div>

      {/* Metrics Cards */}
      <Row gutter={[16, 16]} className="dashboard-metrics">
        <Col xs={24} sm={12} lg={6}>
          <Card hoverable>
            <Skeleton loading={metricsLoading} active paragraph={false}>
              <Statistic
                title={t('dashboard.total_projects', 'Total Projects')}
                value={metrics?.total_projects || 0}
                prefix={<ProjectOutlined style={{ color: '#1890ff' }} />}
                suffix={
                  metrics?.trend?.projects ? (
                    <Tooltip title={`${metrics.trend.projects > 0 ? '+' : ''}${metrics.trend.projects}% from last month`}>
                      <span className={`trend ${metrics.trend.projects > 0 ? 'up' : 'down'}`}>
                        {metrics.trend.projects > 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                        {Math.abs(metrics.trend.projects)}%
                      </span>
                    </Tooltip>
                  ) : null
                }
              />
            </Skeleton>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card hoverable>
            <Skeleton loading={metricsLoading} active paragraph={false}>
              <Statistic
                title={t('dashboard.total_analyses', 'Total Analyses')}
                value={metrics?.total_analyses || 0}
                prefix={<BugOutlined style={{ color: '#722ed1' }} />}
                suffix={
                  metrics?.trend?.analyses ? (
                    <Tooltip title={`${metrics.trend.analyses > 0 ? '+' : ''}${metrics.trend.analyses}% from last month`}>
                      <span className={`trend ${metrics.trend.analyses > 0 ? 'up' : 'down'}`}>
                        {metrics.trend.analyses > 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                        {Math.abs(metrics.trend.analyses)}%
                      </span>
                    </Tooltip>
                  ) : null
                }
              />
            </Skeleton>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card hoverable>
            <Skeleton loading={metricsLoading} active paragraph={false}>
              <Statistic
                title={t('dashboard.issues_found', 'Issues Found')}
                value={metrics?.issues_found || 0}
                valueStyle={{ color: '#cf1322' }}
                prefix={<ExclamationCircleOutlined style={{ color: '#cf1322' }} />}
              />
            </Skeleton>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card hoverable>
            <Skeleton loading={metricsLoading} active paragraph={false}>
              <Statistic
                title={t('dashboard.issues_resolved', 'Issues Resolved')}
                value={metrics?.issues_resolved || 0}
                valueStyle={{ color: '#3f8600' }}
                prefix={<CheckCircleOutlined style={{ color: '#3f8600' }} />}
              />
            </Skeleton>
          </Card>
        </Col>
      </Row>

      {/* Resolution Progress */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card
            title={
              <Space>
                <ProjectOutlined />
                {t('dashboard.recent_projects', 'Recent Projects')}
              </Space>
            }
            extra={
              <Space>
                <Button 
                  type="link" 
                  icon={<ReloadOutlined spin={projectsLoading} />}
                  onClick={refreshProjects}
                  disabled={projectsLoading}
                >
                  {t('common.refresh', 'Refresh')}
                </Button>
                <Button type="link" onClick={() => navigate('/projects')}>
                  {t('dashboard.view_all', 'View All')} →
                </Button>
              </Space>
            }
          >
            {projectsError ? (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={
                  <Space direction="vertical">
                    <Text type="secondary">{t('dashboard.load_error', 'Failed to load projects')}</Text>
                    <Button size="small" onClick={refreshProjects} icon={<ReloadOutlined />}>
                      {t('common.retry', 'Retry')}
                    </Button>
                  </Space>
                }
              />
            ) : (
              <Table
                columns={projectColumns}
                dataSource={recentProjects || []}
                rowKey="id"
                pagination={false}
                loading={projectsLoading}
                locale={{ emptyText: t('dashboard.no_projects', 'No projects yet. Create your first project!') }}
                size="middle"
              />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card 
            title={
              <Space>
                <CheckCircleOutlined />
                {t('dashboard.resolution_rate', 'Resolution Rate')}
              </Space>
            }
          >
            <Skeleton loading={metricsLoading} active>
              <div className="resolution-progress" style={{ textAlign: 'center', padding: '20px 0' }}>
                <Progress
                  type="circle"
                  percent={resolutionRate}
                  strokeColor={{
                    '0%': '#108ee9',
                    '100%': '#87d068'
                  }}
                  format={percent => (
                    <div>
                      <div style={{ fontSize: 24, fontWeight: 600 }}>{percent}%</div>
                      <div style={{ fontSize: 12, color: '#8c8c8c' }}>Resolved</div>
                    </div>
                  )}
                  size={150}
                />
                <div className="resolution-stats" style={{ marginTop: 16 }}>
                  <Space split={<span style={{ color: '#d9d9d9' }}>|</span>}>
                    <Tooltip title="Issues found">
                      <Text type="secondary">
                        <ExclamationCircleOutlined style={{ color: '#cf1322', marginRight: 4 }} />
                        {metrics?.issues_found || 0}
                      </Text>
                    </Tooltip>
                    <Tooltip title="Issues resolved">
                      <Text type="secondary">
                        <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 4 }} />
                        {metrics?.issues_resolved || 0}
                      </Text>
                    </Tooltip>
                  </Space>
                </div>
              </div>
            </Skeleton>
          </Card>
        </Col>
      </Row>

      {/* Recent Activity */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card 
            title={
              <Space>
                <ClockCircleOutlined />
                {t('dashboard.recent_activity', 'Recent Activity')}
              </Space>
            }
            extra={
              <Button type="link" onClick={() => navigate('/activity')}>
                {t('dashboard.view_all', 'View All')} →
              </Button>
            }
          >
            <List
              itemLayout="horizontal"
              dataSource={recentActivity}
              locale={{ 
                emptyText: (
                  <Empty 
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                    description={t('dashboard.no_activity', 'No recent activity')}
                  />
                )
              }}
              renderItem={(item) => (
                <List.Item
                  actions={[
                    <Dropdown
                      key="more"
                      menu={{
                        items: [
                          { key: 'view', icon: <EyeOutlined />, label: 'View Details' },
                          { key: 'settings', icon: <SettingOutlined />, label: 'Settings' },
                        ]
                      }}
                      trigger={['click']}
                    >
                      <Button type="text" size="small" icon={<MoreOutlined />} />
                    </Dropdown>
                  ]}
                >
                  <List.Item.Meta
                    avatar={
                      <Avatar 
                        icon={getActivityIcon(item.type)} 
                        style={{ 
                          backgroundColor: 
                            item.type === 'analysis' ? '#e6f7ff' :
                            item.type === 'fix' ? '#f6ffed' :
                            item.type === 'review' ? '#f9f0ff' : '#f5f5f5'
                        }}
                      />
                    }
                    title={
                      <Space>
                        <Text 
                          strong 
                          style={{ cursor: 'pointer', color: '#1890ff' }}
                          onClick={() => navigate(`/projects/${item.project}`)}
                        >
                          {item.project}
                        </Text>
                        <Text type="secondary">•</Text>
                        <Text>{item.description}</Text>
                      </Space>
                    }
                    description={
                      <Space>
                        <Avatar size="small" style={{ backgroundColor: '#87d068' }}>
                          {item.user.name.charAt(0)}
                        </Avatar>
                        <Text type="secondary">{item.user.name}</Text>
                        <Text type="secondary">•</Text>
                        <Tooltip title={new Date(item.timestamp).toLocaleString()}>
                          <Text type="secondary">{getRelativeTime(item.timestamp)}</Text>
                        </Tooltip>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>

        {/* Self-Evolution Status Widget */}
        <Col xs={24} lg={12}>
          <SelfEvolutionWidget />
        </Col>
      </Row>

      {/* Code Quality & Quick Actions */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <CodeQualityWidget />
        </Col>
        <Col xs={24} lg={12}>
          <QuickActionsWidget />
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
