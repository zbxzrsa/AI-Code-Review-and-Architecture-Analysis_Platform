/**
 * Real-Time Notifications Hook
 * 
 * Hook for managing real-time notifications from the self-evolution system:
 * - Bug fixes applied
 * - Model promotions/degradations
 * - Security alerts
 * - System health updates
 */

import { useState, useEffect, useCallback } from 'react';
import { notification } from 'antd';
import React from 'react';
import * as Icons from '@ant-design/icons';

export interface SystemNotification {
  id: string;
  type: 'success' | 'info' | 'warning' | 'error';
  category: 'fix' | 'evolution' | 'security' | 'system';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
}

interface UseRealTimeNotificationsReturn {
  notifications: SystemNotification[];
  unreadCount: number;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  clearNotifications: () => void;
}

const getNotificationIcon = (category: string): React.ReactNode => {
  const iconStyle = { fontSize: 16 };
  switch (category) {
    case 'fix':
      return React.createElement(Icons.BugOutlined, { style: { ...iconStyle, color: '#52c41a' } });
    case 'evolution':
      return React.createElement(Icons.RocketOutlined, { style: { ...iconStyle, color: '#1890ff' } });
    case 'security':
      return React.createElement(Icons.SafetyOutlined, { style: { ...iconStyle, color: '#faad14' } });
    case 'system':
      return React.createElement(Icons.InfoCircleOutlined, { style: { ...iconStyle, color: '#722ed1' } });
    default:
      return React.createElement(Icons.InfoCircleOutlined, { style: iconStyle });
  }
};

export function useRealTimeNotifications(): UseRealTimeNotificationsReturn {
  const [notifications, setNotifications] = useState<SystemNotification[]>([]);

  // Simulate real-time notifications
  useEffect(() => {
    // Initial notifications
    const initialNotifications: SystemNotification[] = [
      {
        id: 'notif-1',
        type: 'success',
        category: 'fix',
        title: 'Bug Fix Applied',
        message: 'Hardcoded secret removed from auth.py',
        timestamp: new Date(),
        read: false,
      },
      {
        id: 'notif-2',
        type: 'info',
        category: 'evolution',
        title: 'Model Promoted',
        message: 'GQA Attention model promoted to V2',
        timestamp: new Date(Date.now() - 300000),
        read: false,
      },
    ];
    setNotifications(initialNotifications);

    // Simulate incoming notifications
    const interval = setInterval(() => {
      const types: Array<SystemNotification['type']> = ['success', 'info', 'warning'];
      const categories: Array<SystemNotification['category']> = ['fix', 'evolution', 'security', 'system'];
      
      const messages = {
        fix: [
          'Deprecated API usage fixed in utils.py',
          'Memory leak patched in worker.py',
          'SQL injection vulnerability patched',
        ],
        evolution: [
          'New experiment started: Flash Attention',
          'Model evaluation completed',
          'Technology moved to quarantine',
        ],
        security: [
          'Security scan completed - 0 critical issues',
          'New vulnerability pattern detected',
          'API key rotation recommended',
        ],
        system: [
          'System backup completed',
          'Cache cleared successfully',
          'Health check passed',
        ],
      };

      const category = categories[Math.floor(Math.random() * categories.length)];
      const type = types[Math.floor(Math.random() * types.length)];
      const categoryMessages = messages[category];
      const message = categoryMessages[Math.floor(Math.random() * categoryMessages.length)];

      const newNotification: SystemNotification = {
        id: `notif-${Date.now()}`,
        type,
        category,
        title: category.charAt(0).toUpperCase() + category.slice(1) + ' Update',
        message,
        timestamp: new Date(),
        read: false,
      };

      setNotifications((prev) => [newNotification, ...prev].slice(0, 50));

      // Show toast notification
      notification[type]({
        message: newNotification.title,
        description: newNotification.message,
        icon: getNotificationIcon(category),
        placement: 'bottomRight',
        duration: 4,
      });
    }, 30000); // Every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const unreadCount = notifications.filter((n) => !n.read).length;

  const markAsRead = useCallback((id: string) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
  }, []);

  const markAllAsRead = useCallback(() => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  }, []);

  const clearNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  return {
    notifications,
    unreadCount,
    markAsRead,
    markAllAsRead,
    clearNotifications,
  };
}

export default useRealTimeNotifications;
