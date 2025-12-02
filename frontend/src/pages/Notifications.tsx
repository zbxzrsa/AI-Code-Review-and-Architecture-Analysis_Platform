import React, { useState } from 'react';
import {
  Card,
  List,
  Typography,
  Button,
  Badge,
  Tag,
  Space,
  Empty,
  Dropdown,
  Tabs,
  Avatar,
} from 'antd';
import type { MenuProps } from 'antd';
import {
  BellOutlined,
  CheckOutlined,
  DeleteOutlined,
  MoreOutlined,
  InfoCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  BugOutlined,
  RocketOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import './Notifications.css';

const { Title, Text, Paragraph } = Typography;

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  category: 'analysis' | 'system' | 'security' | 'experiment';
  title: string;
  message: string;
  read: boolean;
  timestamp: string;
  link?: string;
}

// Mock notifications
const mockNotifications: Notification[] = [
  {
    id: '1',
    type: 'success',
    category: 'analysis',
    title: 'Analysis Complete',
    message: 'Code review for "auth-service" completed. Found 3 issues.',
    read: false,
    timestamp: '2024-12-02T10:30:00Z',
    link: '/review/proj_1',
  },
  {
    id: '2',
    type: 'warning',
    category: 'security',
    title: 'Security Alert',
    message: 'Potential SQL injection vulnerability detected in project "api-gateway".',
    read: false,
    timestamp: '2024-12-02T09:15:00Z',
    link: '/review/proj_2',
  },
  {
    id: '3',
    type: 'info',
    category: 'experiment',
    title: 'Experiment Promoted',
    message: 'Experiment "GPT-4 Turbo Test" has been promoted to production.',
    read: true,
    timestamp: '2024-12-01T15:45:00Z',
  },
  {
    id: '4',
    type: 'error',
    category: 'system',
    title: 'Service Degraded',
    message: 'AI provider "anthropic" is experiencing high latency.',
    read: true,
    timestamp: '2024-12-01T12:00:00Z',
  },
];

export const Notifications: React.FC = () => {
  const { t } = useTranslation();
  const [notifications, setNotifications] = useState<Notification[]>(mockNotifications);
  const [activeTab, setActiveTab] = useState('all');

  const unreadCount = notifications.filter(n => !n.read).length;

  const markAsRead = (id: string) => {
    setNotifications(prev =>
      prev.map(n => (n.id === id ? { ...n, read: true } : n))
    );
  };

  const markAllAsRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
  };

  const deleteNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const clearAll = () => {
    setNotifications([]);
  };

  const getIcon = (type: string, category: string) => {
    if (category === 'analysis') return <BugOutlined />;
    if (category === 'experiment') return <RocketOutlined />;
    
    switch (type) {
      case 'success': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'warning': return <WarningOutlined style={{ color: '#faad14' }} />;
      case 'error': return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      default: return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'success': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'processing';
    }
  };

  const filteredNotifications = activeTab === 'all' 
    ? notifications 
    : activeTab === 'unread'
    ? notifications.filter(n => !n.read)
    : notifications.filter(n => n.category === activeTab);

  const getItemActions = (item: Notification): MenuProps['items'] => [
    {
      key: 'read',
      label: item.read ? 'Mark as unread' : 'Mark as read',
      icon: <CheckOutlined />,
      onClick: () => markAsRead(item.id),
    },
    {
      key: 'delete',
      label: 'Delete',
      icon: <DeleteOutlined />,
      danger: true,
      onClick: () => deleteNotification(item.id),
    },
  ];

  const tabItems = [
    { key: 'all', label: `All (${notifications.length})` },
    { key: 'unread', label: `Unread (${unreadCount})` },
    { key: 'analysis', label: 'Analysis' },
    { key: 'security', label: 'Security' },
    { key: 'experiment', label: 'Experiments' },
    { key: 'system', label: 'System' },
  ];

  return (
    <div className="notifications-container">
      <div className="notifications-header">
        <Space>
          <Badge count={unreadCount}>
            <BellOutlined style={{ fontSize: 24 }} />
          </Badge>
          <Title level={2} style={{ margin: 0 }}>
            {t('notifications.title', 'Notifications')}
          </Title>
        </Space>
        <Space>
          <Button onClick={markAllAsRead} disabled={unreadCount === 0}>
            {t('notifications.mark_all_read', 'Mark all as read')}
          </Button>
          <Button danger onClick={clearAll} disabled={notifications.length === 0}>
            {t('notifications.clear_all', 'Clear all')}
          </Button>
        </Space>
      </div>

      <Card>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={tabItems}
        />

        {filteredNotifications.length === 0 ? (
          <Empty
            description={t('notifications.empty', 'No notifications')}
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          <List
            dataSource={filteredNotifications}
            renderItem={(item) => (
              <List.Item
                className={`notification-item ${!item.read ? 'unread' : ''}`}
                actions={[
                  <Dropdown menu={{ items: getItemActions(item) }} trigger={['click']}>
                    <Button type="text" icon={<MoreOutlined />} />
                  </Dropdown>
                ]}
              >
                <List.Item.Meta
                  avatar={
                    <Avatar
                      icon={getIcon(item.type, item.category)}
                      style={{
                        backgroundColor: item.type === 'error' ? '#fff1f0' :
                          item.type === 'warning' ? '#fffbe6' :
                          item.type === 'success' ? '#f6ffed' : '#e6f7ff'
                      }}
                    />
                  }
                  title={
                    <Space>
                      <Text strong={!item.read}>{item.title}</Text>
                      <Tag color={getTypeColor(item.type)}>{item.category}</Tag>
                      {!item.read && <Badge status="processing" />}
                    </Space>
                  }
                  description={
                    <>
                      <Paragraph ellipsis={{ rows: 2 }} style={{ marginBottom: 4 }}>
                        {item.message}
                      </Paragraph>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {new Date(item.timestamp).toLocaleString()}
                      </Text>
                    </>
                  }
                />
              </List.Item>
            )}
          />
        )}
      </Card>
    </div>
  );
};

export default Notifications;
