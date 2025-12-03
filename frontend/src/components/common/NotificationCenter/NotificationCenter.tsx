/**
 * Notification Center Component
 * 
 * Global notification/toast system with:
 * - Multiple notification types (success, error, warning, info)
 * - Configurable position
 * - Auto-dismiss with manual close
 * - Action buttons (undo, retry, etc.)
 * - Queue management
 * - WCAG 2.1 AA accessibility compliance
 * - Screen reader compatibility
 * 
 * @example
 * ```tsx
 * const notify = useNotification();
 * notify.success('Profile updated successfully');
 * notify.error('Failed to update profile', { duration: 5000 });
 * notify.info('New message', { 
 *   action: { label: 'View', onClick: () => navigate('/messages') }
 * });
 * ```
 */

import React, { useCallback, useMemo, useEffect, useRef, memo } from 'react';
import { createPortal } from 'react-dom';
import { Button, Typography, Badge } from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  CloseOutlined,
} from '@ant-design/icons';
import { useUIStore, type NotificationPosition } from '../../../store/uiStore';
import './NotificationCenter.css';

const { Text } = Typography;

/**
 * Notification type definition
 */
export type NotificationType = 'success' | 'error' | 'warning' | 'info';

/**
 * Notification action button
 */
export interface NotificationAction {
  label: string;
  onClick: () => void;
}

/**
 * Notification interface matching the specification
 */
export interface Notification {
  id: string;
  type: NotificationType;
  message: string;
  duration?: number;
  action?: NotificationAction;
  dismissible?: boolean;
}

/**
 * Options for showing notifications
 */
export interface NotificationOptions {
  duration?: number;
  action?: NotificationAction;
  dismissible?: boolean;
}

/** Notification type icons */
const typeIcons: Record<NotificationType, React.ReactNode> = {
  success: <CheckCircleOutlined />,
  error: <CloseCircleOutlined />,
  warning: <ExclamationCircleOutlined />,
  info: <InfoCircleOutlined />,
};

/** ARIA live region settings by type */
const ariaLiveSettings: Record<NotificationType, 'polite' | 'assertive'> = {
  success: 'polite',
  error: 'assertive',
  warning: 'assertive',
  info: 'polite',
};

/** Position class mapping */
const positionClasses: Record<NotificationPosition, string> = {
  'top-right': 'notification-center--top-right',
  'top-center': 'notification-center--top-center',
  'bottom-right': 'notification-center--bottom-right',
  'bottom-center': 'notification-center--bottom-center',
};

/**
 * Single Notification Item (Memoized for performance)
 */
const NotificationItem = memo<{
  id: string;
  type: NotificationType;
  message: string;
  dismissible?: boolean;
  action?: NotificationAction;
  onClose: () => void;
}>(({ id: _id, type, message, dismissible = true, action, onClose }) => {
  const itemRef = useRef<HTMLDivElement>(null);

  // Focus management for accessibility
  useEffect(() => {
    if (type === 'error' && itemRef.current) {
      itemRef.current.focus();
    }
  }, [type]);

  const handleAction = useCallback(() => {
    action?.onClick();
    onClose();
  }, [action, onClose]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape' && dismissible) {
      onClose();
    }
    if (e.key === 'Enter' && action) {
      handleAction();
    }
  }, [dismissible, action, onClose, handleAction]);

  return (
    <div
      ref={itemRef}
      className={`notification-item notification-item--${type}`}
      role="alert"
      aria-live={ariaLiveSettings[type]}
      aria-atomic="true"
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      <div className="notification-item__icon" aria-hidden="true">
        {typeIcons[type]}
      </div>
      
      <div className="notification-item__content">
        <Text strong className="notification-item__message">{message}</Text>
        
        {action && (
          <div className="notification-item__actions">
            <Button
              size="small"
              type="link"
              onClick={handleAction}
              className="notification-item__action-btn"
            >
              {action.label}
            </Button>
          </div>
        )}
      </div>

      {dismissible && (
        <Button
          type="text"
          size="small"
          icon={<CloseOutlined />}
          onClick={onClose}
          className="notification-item__close"
          aria-label="Dismiss notification"
        />
      )}
    </div>
  );
});

NotificationItem.displayName = 'NotificationItem';

/**
 * Notification Center Component (Memoized)
 */
export const NotificationCenter: React.FC = memo(() => {
  const { notifications, notificationSettings, removeNotification } = useUIStore();
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Get visible notifications (limited by maxVisible)
  const visibleNotifications = useMemo(() => {
    return notifications.slice(-notificationSettings.maxVisible);
  }, [notifications, notificationSettings.maxVisible]);

  // Hidden count
  const hiddenCount = Math.max(0, notifications.length - notificationSettings.maxVisible);

  // Position class
  const positionClass = positionClasses[notificationSettings.position];

  // Don't render if no notifications
  if (notifications.length === 0) {
    return null;
  }

  const content = (
    <div 
      ref={containerRef}
      className={`notification-center ${positionClass}`}
      role="region"
      aria-label="Notifications"
      aria-live="polite"
    >
      {/* Screen reader announcements */}
      <div className="sr-only" aria-live="assertive" aria-atomic="true">
        {notifications.length > 0 && `${notifications.length} notification${notifications.length > 1 ? 's' : ''}`}
      </div>

      {/* Hidden notifications indicator */}
      {hiddenCount > 0 && (
        <div className="notification-center__hidden-count">
          <Badge count={hiddenCount} size="small">
            <Text type="secondary">
              +{hiddenCount} more notification{hiddenCount > 1 ? 's' : ''}
            </Text>
          </Badge>
        </div>
      )}

      {/* Notification items */}
      <div className="notification-center__list">
        {visibleNotifications.map((notification) => (
          <NotificationItem
            key={notification.id}
            id={notification.id}
            type={notification.type}
            message={notification.title || notification.message || ''}
            dismissible={notification.dismissible ?? true}
            action={notification.actions?.[0]}
            onClose={() => removeNotification(notification.id)}
          />
        ))}
      </div>
    </div>
  );

  // Render in portal to ensure it's above everything
  return createPortal(content, document.body);
});

NotificationCenter.displayName = 'NotificationCenter';

/**
 * Hook for showing notifications
 * 
 * @example
 * ```tsx
 * const notify = useNotification();
 * 
 * // Simple usage
 * notify.success('Profile updated successfully');
 * notify.error('Failed to update profile', { duration: 5000 });
 * 
 * // With action button
 * notify.info('New message received', { 
 *   action: { label: 'View', onClick: () => navigate('/messages') }
 * });
 * 
 * // With custom duration
 * notify.warning('Session expiring soon', { duration: 10000 });
 * ```
 */
export const useNotification = () => {
  const { addNotification, removeNotification, clearNotifications } = useUIStore();

  /**
   * Show a notification
   */
  const showNotification = useCallback((
    type: NotificationType,
    message: string,
    options?: NotificationOptions
  ): string => {
    return addNotification({
      type,
      title: message,
      message: options?.action ? undefined : undefined,
      duration: options?.duration,
      dismissible: options?.dismissible,
      actions: options?.action ? [options.action] : undefined,
    });
  }, [addNotification]);

  return {
    /**
     * Show success notification
     * @param message - Notification message
     * @param options - Optional settings (duration, action)
     */
    success: (message: string, options?: NotificationOptions): string =>
      showNotification('success', message, options),

    /**
     * Show error notification
     * @param message - Notification message
     * @param options - Optional settings (duration, action)
     */
    error: (message: string, options?: NotificationOptions): string =>
      showNotification('error', message, { duration: 5000, ...options }),

    /**
     * Show warning notification
     * @param message - Notification message
     * @param options - Optional settings (duration, action)
     */
    warning: (message: string, options?: NotificationOptions): string =>
      showNotification('warning', message, options),

    /**
     * Show info notification
     * @param message - Notification message
     * @param options - Optional settings (duration, action)
     */
    info: (message: string, options?: NotificationOptions): string =>
      showNotification('info', message, options),

    /**
     * Remove a specific notification by ID
     */
    remove: removeNotification,

    /**
     * Clear all notifications
     */
    clear: clearNotifications,
  };
};

export default NotificationCenter;
