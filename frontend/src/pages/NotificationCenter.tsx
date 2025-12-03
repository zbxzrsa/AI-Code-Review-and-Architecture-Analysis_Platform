/**
 * Notification Center Page
 * 通知中心页面
 * 
 * Features:
 * - All notifications in one place
 * - Filter by type and status
 * - Mark as read/unread
 * - Notification preferences
 */

import React, { useState } from 'react';
import {
  Card,
  Typography,
  Space,
  Button,
  Tag,
  List,
  Avatar,
  Badge,
  Tabs,
  Empty,
  Checkbox,
  Dropdown,
  message,
} from 'antd';
import {
  BellOutlined,
  SecurityScanOutlined,
  CodeOutlined,
  TeamOutlined,
  RocketOutlined,
  SettingOutlined,
  DeleteOutlined,
  CheckOutlined,
  MoreOutlined,
  ClearOutlined,
  ClockCircleOutlined,
  BulbOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text, Paragraph } = Typography;

interface Notification {
  id: string;
  type: 'security' | 'review' | 'deployment' | 'team' | 'system' | 'ai';
  priority: 'high' | 'medium' | 'low';
  title: string;
  message: string;
  read: boolean;
  timestamp: string;
  actionUrl?: string;
  actionLabel?: string;
}

const mockNotifications: Notification[] = [
  {
    id: 'n1',
    type: 'security',
    priority: 'high',
    title: 'Critical Vulnerability Detected',
    message: 'SQL injection vulnerability found in src/api/users.py. AI Auto-Fix is available.',
    read: false,
    timestamp: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
    actionUrl: '/security',
    actionLabel: 'View Details',
  },
  {
    id: 'n2',
    type: 'ai',
    priority: 'medium',
    title: 'AI Auto-Fix Applied',
    message: '3 security fixes have been automatically applied and are pending review.',
    read: false,
    timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    actionUrl: '/admin/auto-fix',
    actionLabel: 'Review Fixes',
  },
  {
    id: 'n3',
    type: 'review',
    priority: 'medium',
    title: 'Code Review Completed',
    message: 'AI analysis complete for PR #142. Found 2 issues and 5 suggestions.',
    read: false,
    timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
    actionUrl: '/pull-requests',
    actionLabel: 'View PR',
  },
  {
    id: 'n4',
    type: 'deployment',
    priority: 'low',
    title: 'Deployment Successful',
    message: 'v2.1.5 has been deployed to production successfully.',
    read: true,
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    actionUrl: '/deployments',
    actionLabel: 'View Deployment',
  },
  {
    id: 'n5',
    type: 'team',
    priority: 'low',
    title: 'New Team Member',
    message: 'John Doe has joined the Frontend team.',
    read: true,
    timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'n6',
    type: 'system',
    priority: 'medium',
    title: 'Usage Limit Warning',
    message: 'You have used 85% of your monthly token quota.',
    read: true,
    timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    actionUrl: '/billing',
    actionLabel: 'Upgrade Plan',
  },
  {
    id: 'n7',
    type: 'security',
    priority: 'high',
    title: 'Hardcoded Secret Detected',
    message: 'API key found in src/config/settings.py. Immediate action required.',
    read: false,
    timestamp: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
    actionUrl: '/security',
    actionLabel: 'Fix Now',
  },
  {
    id: 'n8',
    type: 'ai',
    priority: 'low',
    title: 'AI Model Updated',
    message: 'GPT-4 Turbo model has been updated to the latest version.',
    read: true,
    timestamp: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
  },
];

const typeConfig = {
  security: { icon: <SecurityScanOutlined />, color: '#ef4444', label: 'Security' },
  review: { icon: <CodeOutlined />, color: '#3b82f6', label: 'Review' },
  deployment: { icon: <RocketOutlined />, color: '#22c55e', label: 'Deployment' },
  team: { icon: <TeamOutlined />, color: '#8b5cf6', label: 'Team' },
  system: { icon: <SettingOutlined />, color: '#f59e0b', label: 'System' },
  ai: { icon: <BulbOutlined />, color: '#06b6d4', label: 'AI' },
};

const priorityConfig = {
  high: { color: 'red', label: 'High' },
  medium: { color: 'orange', label: 'Medium' },
  low: { color: 'blue', label: 'Low' },
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

export const NotificationCenter: React.FC = () => {
  const { t: _t } = useTranslation();
  const [notifications, setNotifications] = useState<Notification[]>(mockNotifications);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState('all');

  const unreadCount = notifications.filter(n => !n.read).length;
  
  const filteredNotifications = notifications.filter(n => {
    if (filter === 'unread') return !n.read;
    if (filter === 'high') return n.priority === 'high';
    if (filter !== 'all') return n.type === filter;
    return true;
  });

  const handleMarkAsRead = (ids: string[]) => {
    setNotifications(prev => prev.map(n =>
      ids.includes(n.id) ? { ...n, read: true } : n
    ));
    message.success(`Marked ${ids.length} notification(s) as read`);
    setSelectedIds(new Set());
  };

  const handleMarkAllAsRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
    message.success('All notifications marked as read');
  };

  const handleDelete = (ids: string[]) => {
    setNotifications(prev => prev.filter(n => !ids.includes(n.id)));
    message.success(`Deleted ${ids.length} notification(s)`);
    setSelectedIds(new Set());
  };

  const handleClearAll = () => {
    setNotifications([]);
    message.success('All notifications cleared');
  };

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  return (
    <div className="notification-center-page" style={{ maxWidth: 1000, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <BellOutlined style={{ color: '#2563eb' }} /> Notifications
            {unreadCount > 0 && (
              <Badge count={unreadCount} style={{ marginLeft: 12 }} />
            )}
          </Title>
          <Text type="secondary">Stay updated with your code review activity</Text>
        </div>
        <Space>
          {selectedIds.size > 0 ? (
            <>
              <Button icon={<CheckOutlined />} onClick={() => handleMarkAsRead(Array.from(selectedIds))}>
                Mark as Read ({selectedIds.size})
              </Button>
              <Button danger icon={<DeleteOutlined />} onClick={() => handleDelete(Array.from(selectedIds))}>
                Delete ({selectedIds.size})
              </Button>
            </>
          ) : (
            <>
              <Button icon={<CheckOutlined />} onClick={handleMarkAllAsRead} disabled={unreadCount === 0}>
                Mark All as Read
              </Button>
              <Button icon={<ClearOutlined />} onClick={handleClearAll} disabled={notifications.length === 0}>
                Clear All
              </Button>
            </>
          )}
        </Space>
      </div>

      {/* Filter Tabs */}
      <Card style={{ marginBottom: 16, borderRadius: 12 }} bodyStyle={{ padding: '8px 16px' }}>
        <Tabs
          activeKey={filter}
          onChange={setFilter}
          items={[
            { key: 'all', label: `All (${notifications.length})` },
            { key: 'unread', label: `Unread (${unreadCount})` },
            { key: 'high', label: `High Priority (${notifications.filter(n => n.priority === 'high').length})` },
            { key: 'security', label: 'Security' },
            { key: 'review', label: 'Reviews' },
            { key: 'ai', label: 'AI Updates' },
          ]}
        />
      </Card>

      {/* Notifications List */}
      <Card style={{ borderRadius: 12 }} bodyStyle={{ padding: 0 }}>
        {filteredNotifications.length > 0 ? (
          <List
            dataSource={filteredNotifications}
            renderItem={notification => {
              const typeConf = typeConfig[notification.type];
              const priorityConf = priorityConfig[notification.priority];
              
              return (
                <List.Item
                  style={{
                    padding: '16px 24px',
                    background: notification.read ? 'transparent' : 'rgba(37, 99, 235, 0.03)',
                    borderLeft: notification.read ? 'none' : '3px solid #2563eb',
                    cursor: 'pointer',
                  }}
                  actions={[
                    notification.actionUrl && (
                      <Button key="action" type="link" size="small">
                        {notification.actionLabel}
                      </Button>
                    ),
                    <Dropdown key="dropdown"
                      menu={{
                        items: [
                          {
                            key: 'read',
                            icon: <CheckOutlined />,
                            label: notification.read ? 'Mark as Unread' : 'Mark as Read',
                            onClick: () => setNotifications(prev => prev.map(n =>
                              n.id === notification.id ? { ...n, read: !n.read } : n
                            )),
                          },
                          {
                            key: 'delete',
                            icon: <DeleteOutlined />,
                            label: 'Delete',
                            danger: true,
                            onClick: () => handleDelete([notification.id]),
                          },
                        ],
                      }}
                      trigger={['click']}
                    >
                      <Button type="text" icon={<MoreOutlined />} size="small" />
                    </Dropdown>,
                  ].filter(Boolean)}
                >
                  <List.Item.Meta
                    avatar={
                      <Space>
                        <Checkbox
                          checked={selectedIds.has(notification.id)}
                          onChange={() => toggleSelect(notification.id)}
                          onClick={e => e.stopPropagation()}
                        />
                        <Avatar
                          style={{
                            background: `${typeConf.color}15`,
                            color: typeConf.color,
                          }}
                          icon={typeConf.icon}
                        />
                      </Space>
                    }
                    title={
                      <Space>
                        <Text strong style={{ opacity: notification.read ? 0.7 : 1 }}>
                          {notification.title}
                        </Text>
                        {notification.priority === 'high' && (
                          <Tag color={priorityConf.color}>{priorityConf.label}</Tag>
                        )}
                        {!notification.read && (
                          <Badge status="processing" />
                        )}
                      </Space>
                    }
                    description={
                      <div>
                        <Paragraph
                          style={{ margin: 0, opacity: notification.read ? 0.6 : 0.8 }}
                          ellipsis={{ rows: 2 }}
                        >
                          {notification.message}
                        </Paragraph>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          <ClockCircleOutlined style={{ marginRight: 4 }} />
                          {formatTimeAgo(notification.timestamp)}
                        </Text>
                      </div>
                    }
                  />
                </List.Item>
              );
            }}
          />
        ) : (
          <Empty
            description="No notifications"
            style={{ padding: 48 }}
          />
        )}
      </Card>
    </div>
  );
};

export default NotificationCenter;
