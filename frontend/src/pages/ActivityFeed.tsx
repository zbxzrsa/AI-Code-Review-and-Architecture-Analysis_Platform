/**
 * Activity Feed Page
 * 活动动态页面
 * 
 * VSCode/GitHub-inspired activity timeline with:
 * - Real-time activity updates
 * - Filterable by type, user, project
 * - Rich activity cards with context
 */

import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Avatar,
  Tag,
  Button,
  Select,
  Input,
  Tooltip,
  Badge,
  Empty,
} from 'antd';
import {
  ClockCircleOutlined,
  CodeOutlined,
  UserOutlined,
  TeamOutlined,
  BranchesOutlined,
  MergeOutlined,
  CommentOutlined,
  EyeOutlined,
  FilterOutlined,
  ReloadOutlined,
  BellOutlined,
  RocketOutlined,
  SafetyCertificateOutlined,
  SettingOutlined,
  StarOutlined,
  StarFilled,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text } = Typography;

interface Activity {
  id: string;
  type: 'analysis' | 'review' | 'comment' | 'merge' | 'deploy' | 'security' | 'config';
  action: string;
  description: string;
  user: {
    name: string;
    avatar?: string;
  };
  project?: string;
  branch?: string;
  file?: string;
  metadata?: Record<string, any>;
  timestamp: string;
  isNew?: boolean;
}

const mockActivities: Activity[] = [
  {
    id: 'act_1',
    type: 'analysis',
    action: 'completed',
    description: 'Code analysis completed for frontend/src/components',
    user: { name: 'AI Assistant' },
    project: 'Frontend',
    branch: 'feature/new-dashboard',
    metadata: { issues: 3, warnings: 12, duration: '2.3s' },
    timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    isNew: true,
  },
  {
    id: 'act_2',
    type: 'security',
    action: 'detected',
    description: 'Critical vulnerability detected: SQL Injection',
    user: { name: 'Security Scanner' },
    project: 'Backend',
    file: 'src/auth/login.py',
    metadata: { severity: 'critical', cve: 'CVE-2024-1234' },
    timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    isNew: true,
  },
  {
    id: 'act_3',
    type: 'review',
    action: 'approved',
    description: 'Approved pull request: Add user authentication',
    user: { name: 'John Doe' },
    project: 'Backend',
    branch: 'feature/auth',
    timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
  },
  {
    id: 'act_4',
    type: 'merge',
    action: 'merged',
    description: 'Merged feature/auth into main',
    user: { name: 'Jane Smith' },
    project: 'Backend',
    branch: 'main',
    timestamp: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
  },
  {
    id: 'act_5',
    type: 'comment',
    action: 'commented',
    description: 'Suggested using parameterized queries instead',
    user: { name: 'Bob Wilson' },
    project: 'Backend',
    file: 'src/api/users.py:45',
    timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'act_6',
    type: 'deploy',
    action: 'deployed',
    description: 'Deployed v2.1.0 to production',
    user: { name: 'CI/CD Pipeline' },
    project: 'Frontend',
    metadata: { version: 'v2.1.0', environment: 'production' },
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'act_7',
    type: 'config',
    action: 'updated',
    description: 'Updated AI model configuration',
    user: { name: 'Admin' },
    metadata: { model: 'GPT-4 Turbo', temperature: 0.7 },
    timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'act_8',
    type: 'analysis',
    action: 'started',
    description: 'Started deep analysis for entire repository',
    user: { name: 'AI Assistant' },
    project: 'Mobile App',
    metadata: { files: 156, estimatedTime: '5 min' },
    timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
  },
];

const activityConfig = {
  analysis: { 
    icon: <CodeOutlined />, 
    color: '#2563eb',
    bgColor: 'rgba(37, 99, 235, 0.1)',
  },
  review: { 
    icon: <EyeOutlined />, 
    color: '#8b5cf6',
    bgColor: 'rgba(139, 92, 246, 0.1)',
  },
  comment: { 
    icon: <CommentOutlined />, 
    color: '#06b6d4',
    bgColor: 'rgba(6, 182, 212, 0.1)',
  },
  merge: { 
    icon: <MergeOutlined />, 
    color: '#10b981',
    bgColor: 'rgba(16, 185, 129, 0.1)',
  },
  deploy: { 
    icon: <RocketOutlined />, 
    color: '#f59e0b',
    bgColor: 'rgba(245, 158, 11, 0.1)',
  },
  security: { 
    icon: <SafetyCertificateOutlined />, 
    color: '#ef4444',
    bgColor: 'rgba(239, 68, 68, 0.1)',
  },
  config: { 
    icon: <SettingOutlined />, 
    color: '#64748b',
    bgColor: 'rgba(100, 116, 139, 0.1)',
  },
};

const formatTimeAgo = (timestamp: string): string => {
  const now = Date.now();
  const time = new Date(timestamp).getTime();
  const diff = now - time;
  
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);
  
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return `${days}d ago`;
};

export const ActivityFeed: React.FC = () => {
  const { t } = useTranslation();
  const [activities, _setActivities] = useState<Activity[]>(mockActivities);
  const [filter, setFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [starred, setStarred] = useState<Set<string>>(new Set());

  const filteredActivities = activities.filter(activity => {
    if (filter !== 'all' && activity.type !== filter) return false;
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        activity.description.toLowerCase().includes(query) ||
        activity.user.name.toLowerCase().includes(query) ||
        activity.project?.toLowerCase().includes(query)
      );
    }
    return true;
  });

  const toggleStar = (id: string) => {
    setStarred(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  const newCount = activities.filter(a => a.isNew).length;

  return (
    <div className="activity-feed-page" style={{ maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 12 }}>
            <BellOutlined style={{ color: '#2563eb' }} />
            {t('activity.title', 'Activity Feed')}
            {newCount > 0 && (
              <Badge count={newCount} style={{ marginLeft: 8 }} />
            )}
          </Title>
          <Text type="secondary">
            {t('activity.subtitle', 'Track all activities across your projects')}
          </Text>
        </div>
        <Space>
          <Input.Search
            placeholder="Search activities..."
            style={{ width: 250 }}
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            allowClear
          />
          <Select
            value={filter}
            onChange={setFilter}
            style={{ width: 150 }}
            options={[
              { value: 'all', label: 'All Types' },
              { value: 'analysis', label: 'Analysis' },
              { value: 'review', label: 'Reviews' },
              { value: 'security', label: 'Security' },
              { value: 'deploy', label: 'Deployments' },
              { value: 'merge', label: 'Merges' },
            ]}
          />
          <Button icon={<ReloadOutlined />}>Refresh</Button>
        </Space>
      </div>

      {/* Activity Timeline */}
      <Row gutter={24}>
        <Col xs={24} lg={16}>
          <Card 
            bodyStyle={{ padding: 0 }}
            style={{ 
              borderRadius: 16,
              overflow: 'hidden',
              border: '1px solid #e2e8f0',
            }}
          >
            {filteredActivities.length > 0 ? (
              <div>
                {filteredActivities.map((activity, index) => {
                  const config = activityConfig[activity.type];
                  return (
                    <div
                      key={activity.id}
                      style={{
                        padding: '20px 24px',
                        borderBottom: index < filteredActivities.length - 1 ? '1px solid #f1f5f9' : 'none',
                        background: activity.isNew ? 'rgba(37, 99, 235, 0.03)' : 'transparent',
                        transition: 'background 0.2s',
                      }}
                      className="activity-item"
                    >
                      <div style={{ display: 'flex', gap: 16 }}>
                        {/* Icon */}
                        <div
                          style={{
                            width: 42,
                            height: 42,
                            borderRadius: 12,
                            background: config.bgColor,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: config.color,
                            fontSize: 18,
                            flexShrink: 0,
                          }}
                        >
                          {config.icon}
                        </div>

                        {/* Content */}
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                            <div>
                              <Text strong style={{ fontSize: 14 }}>
                                {activity.description}
                              </Text>
                              {activity.isNew && (
                                <Tag color="blue" style={{ marginLeft: 8, fontSize: 11 }}>NEW</Tag>
                              )}
                            </div>
                            <Space size={4}>
                              <Tooltip title={starred.has(activity.id) ? 'Unstar' : 'Star'}>
                                <Button
                                  type="text"
                                  size="small"
                                  icon={starred.has(activity.id) ? 
                                    <StarFilled style={{ color: '#f59e0b' }} /> : 
                                    <StarOutlined />
                                  }
                                  onClick={() => toggleStar(activity.id)}
                                />
                              </Tooltip>
                              <Text type="secondary" style={{ fontSize: 12 }}>
                                {formatTimeAgo(activity.timestamp)}
                              </Text>
                            </Space>
                          </div>

                          {/* Metadata */}
                          <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                            <Space size={4}>
                              <Avatar size={20} icon={<UserOutlined />} src={activity.user.avatar} />
                              <Text type="secondary" style={{ fontSize: 13 }}>
                                {activity.user.name}
                              </Text>
                            </Space>
                            
                            {activity.project && (
                              <Tag 
                                style={{ 
                                  borderRadius: 6,
                                  background: '#f1f5f9',
                                  border: 'none',
                                  color: '#475569',
                                }}
                              >
                                <BranchesOutlined style={{ marginRight: 4 }} />
                                {activity.project}
                              </Tag>
                            )}
                            
                            {activity.branch && (
                              <Tag
                                style={{ 
                                  borderRadius: 6,
                                  background: '#eff6ff',
                                  border: 'none',
                                  color: '#2563eb',
                                }}
                              >
                                {activity.branch}
                              </Tag>
                            )}

                            {activity.file && (
                              <Text code style={{ fontSize: 12 }}>{activity.file}</Text>
                            )}

                            {activity.metadata?.severity === 'critical' && (
                              <Tag color="red">CRITICAL</Tag>
                            )}

                            {activity.metadata?.version && (
                              <Tag color="green">{activity.metadata.version}</Tag>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <Empty 
                description="No activities found" 
                style={{ padding: 48 }}
              />
            )}
          </Card>
        </Col>

        {/* Sidebar Stats */}
        <Col xs={24} lg={8}>
          {/* Activity Summary */}
          <Card 
            title={<><ClockCircleOutlined /> Today's Summary</>}
            style={{ marginBottom: 16, borderRadius: 16 }}
          >
            <Space direction="vertical" style={{ width: '100%' }} size={12}>
              {Object.entries(activityConfig).map(([type, config]) => {
                const count = activities.filter(a => a.type === type).length;
                return (
                  <div key={type} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Space>
                      <span style={{ color: config.color }}>{config.icon}</span>
                      <Text style={{ textTransform: 'capitalize' }}>{type}</Text>
                    </Space>
                    <Tag style={{ borderRadius: 12, minWidth: 32, textAlign: 'center' }}>{count}</Tag>
                  </div>
                );
              })}
            </Space>
          </Card>

          {/* Quick Filters */}
          <Card 
            title={<><FilterOutlined /> Quick Filters</>}
            style={{ marginBottom: 16, borderRadius: 16 }}
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button block style={{ textAlign: 'left' }}>
                <SafetyCertificateOutlined style={{ color: '#ef4444' }} /> Security Alerts
              </Button>
              <Button block style={{ textAlign: 'left' }}>
                <StarFilled style={{ color: '#f59e0b' }} /> Starred
              </Button>
              <Button block style={{ textAlign: 'left' }}>
                <UserOutlined /> My Activity
              </Button>
              <Button block style={{ textAlign: 'left' }}>
                <TeamOutlined /> Team Activity
              </Button>
            </Space>
          </Card>

          {/* Recent Projects */}
          <Card 
            title={<><BranchesOutlined /> Active Projects</>}
            style={{ borderRadius: 16 }}
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              {['Frontend', 'Backend', 'Mobile App', 'API Gateway'].map(project => (
                <div key={project} style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  padding: '8px 12px',
                  borderRadius: 8,
                  cursor: 'pointer',
                  transition: 'background 0.2s',
                }}
                className="project-item"
                >
                  <Text>{project}</Text>
                  <Badge 
                    count={Math.floor(Math.random() * 10)} 
                    style={{ backgroundColor: '#2563eb' }}
                  />
                </div>
              ))}
            </Space>
          </Card>
        </Col>
      </Row>

      <style>{`
        .activity-item:hover {
          background: rgba(37, 99, 235, 0.02) !important;
        }
        .project-item:hover {
          background: #f8fafc;
        }
      `}</style>
    </div>
  );
};

export default ActivityFeed;
