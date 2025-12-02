import React, { useEffect, useState } from 'react';
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
  Avatar
} from 'antd';
import {
  ProjectOutlined,
  BugOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  PlusOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../hooks/useAuth';
import { apiService } from '../services/api';
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
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [recentProjects, setRecentProjects] = useState<RecentProject[]>([]);
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const [metricsRes, projectsRes] = await Promise.all([
          apiService.metrics.getDashboard(),
          apiService.projects.list({ limit: 5 })
        ]);

        setMetrics(metricsRes.data);
        setRecentProjects(projectsRes.data.items || []);
        
        // Mock activity data for now
        setRecentActivity([
          {
            id: '1',
            type: 'analysis',
            project: 'backend-api',
            description: 'Completed security analysis',
            timestamp: new Date().toISOString(),
            user: { name: user?.name || 'User' }
          }
        ]);
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [user]);

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

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div>
          <Title level={2}>
            {t('dashboard.welcome', 'Welcome back')}, {user?.name}!
          </Title>
          <Text type="secondary">
            {t('dashboard.subtitle', "Here's what's happening with your projects")}
          </Text>
        </div>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => navigate('/projects/new')}
        >
          {t('dashboard.new_project', 'New Project')}
        </Button>
      </div>

      {/* Metrics Cards */}
      <Row gutter={[16, 16]} className="dashboard-metrics">
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title={t('dashboard.total_projects', 'Total Projects')}
              value={metrics?.total_projects || 0}
              prefix={<ProjectOutlined />}
              suffix={
                metrics?.trend?.projects ? (
                  <span className={`trend ${metrics.trend.projects > 0 ? 'up' : 'down'}`}>
                    {metrics.trend.projects > 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                    {Math.abs(metrics.trend.projects)}%
                  </span>
                ) : null
              }
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title={t('dashboard.total_analyses', 'Total Analyses')}
              value={metrics?.total_analyses || 0}
              prefix={<BugOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title={t('dashboard.issues_found', 'Issues Found')}
              value={metrics?.issues_found || 0}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title={t('dashboard.issues_resolved', 'Issues Resolved')}
              value={metrics?.issues_resolved || 0}
              valueStyle={{ color: '#3f8600' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Resolution Progress */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card
            title={t('dashboard.recent_projects', 'Recent Projects')}
            extra={
              <Button type="link" onClick={() => navigate('/projects')}>
                {t('dashboard.view_all', 'View All')}
              </Button>
            }
          >
            <Table
              columns={projectColumns}
              dataSource={recentProjects}
              rowKey="id"
              pagination={false}
              loading={loading}
              locale={{ emptyText: t('dashboard.no_projects', 'No projects yet') }}
            />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title={t('dashboard.resolution_rate', 'Resolution Rate')}>
            <div className="resolution-progress">
              <Progress
                type="circle"
                percent={
                  metrics?.issues_found
                    ? Math.round((metrics.issues_resolved / metrics.issues_found) * 100)
                    : 0
                }
                strokeColor={{
                  '0%': '#108ee9',
                  '100%': '#87d068'
                }}
              />
              <div className="resolution-stats">
                <Text>
                  {metrics?.issues_resolved || 0} / {metrics?.issues_found || 0}{' '}
                  {t('dashboard.issues_resolved_label', 'issues resolved')}
                </Text>
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      {/* Recent Activity */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24}>
          <Card title={t('dashboard.recent_activity', 'Recent Activity')}>
            <List
              itemLayout="horizontal"
              dataSource={recentActivity}
              loading={loading}
              locale={{ emptyText: t('dashboard.no_activity', 'No recent activity') }}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Avatar icon={getActivityIcon(item.type)} />
                    }
                    title={
                      <Space>
                        <Text strong>{item.project}</Text>
                        <Text type="secondary">•</Text>
                        <Text type="secondary">{item.description}</Text>
                      </Space>
                    }
                    description={
                      <Space>
                        <Text type="secondary">{item.user.name}</Text>
                        <Text type="secondary">•</Text>
                        <Text type="secondary">
                          {new Date(item.timestamp).toLocaleString()}
                        </Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
