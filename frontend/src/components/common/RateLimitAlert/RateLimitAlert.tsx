/**
 * Rate Limit Alert Component
 * 
 * Displays rate limit status with:
 * - Cooldown timer
 * - User-friendly messages
 * - Visual countdown
 */

import React, { useMemo } from 'react';
import { Alert, Progress, Space, Typography } from 'antd';
import { ClockCircleOutlined, WarningOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import type { RateLimitStatus } from '../../../hooks/useRateLimiter';
import './RateLimitAlert.css';

const { Text } = Typography;

interface RateLimitAlertProps {
  status: RateLimitStatus;
  totalSeconds?: number;
  message?: string;
  type?: 'warning' | 'error';
  showProgress?: boolean;
  onRetry?: () => void;
}

/**
 * Rate Limit Alert Component
 */
export const RateLimitAlert: React.FC<RateLimitAlertProps> = ({
  status,
  totalSeconds = 60,
  message,
  type = 'warning',
  showProgress = true,
  onRetry,
}) => {
  const { t } = useTranslation();

  // Calculate progress percentage
  const progressPercent = useMemo(() => {
    if (!status.isLimited || status.cooldownSeconds <= 0) return 100;
    return Math.round(((totalSeconds - status.cooldownSeconds) / totalSeconds) * 100);
  }, [status.isLimited, status.cooldownSeconds, totalSeconds]);

  // Format time remaining
  const timeRemaining = useMemo(() => {
    const seconds = status.cooldownSeconds;
    if (seconds < 60) {
      return t('rateLimit.seconds', '{{count}} seconds', { count: seconds });
    }
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (minutes < 60) {
      return secs > 0
        ? t('rateLimit.minutesSeconds', '{{minutes}}m {{seconds}}s', { minutes, seconds: secs })
        : t('rateLimit.minutes', '{{count}} minutes', { count: minutes });
    }
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0
      ? t('rateLimit.hoursMinutes', '{{hours}}h {{minutes}}m', { hours, minutes: mins })
      : t('rateLimit.hours', '{{count}} hours', { count: hours });
  }, [status.cooldownSeconds, t]);

  if (!status.isLimited) {
    return null;
  }

  const defaultMessage = t(
    'rateLimit.message',
    'Too many attempts. Please wait {{time}} before trying again.',
    { time: timeRemaining }
  );

  return (
    <Alert
      type={type}
      icon={<WarningOutlined />}
      message={
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
          <Space>
            <ClockCircleOutlined />
            <Text strong>{message || defaultMessage}</Text>
          </Space>
          
          {showProgress && (
            <div className="rate-limit-progress">
              <Progress
                percent={progressPercent}
                size="small"
                status={progressPercent < 100 ? 'active' : 'success'}
                showInfo={false}
                strokeColor={{
                  '0%': '#ff4d4f',
                  '100%': '#52c41a',
                }}
              />
              <Text type="secondary" className="rate-limit-timer">
                {timeRemaining}
              </Text>
            </div>
          )}
        </Space>
      }
      showIcon={false}
      className="rate-limit-alert"
    />
  );
};

/**
 * Inline Rate Limit Warning (for form fields)
 */
interface RateLimitWarningProps {
  isLimited: boolean;
  cooldownSeconds: number;
}

export const RateLimitWarning: React.FC<RateLimitWarningProps> = ({
  isLimited,
  cooldownSeconds,
}) => {
  const { t } = useTranslation();

  if (!isLimited) return null;

  return (
    <div className="rate-limit-warning">
      <ClockCircleOutlined />
      <Text type="warning">
        {t('rateLimit.wait', 'Please wait {{seconds}}s', { seconds: cooldownSeconds })}
      </Text>
    </div>
  );
};

/**
 * Rate Limit Info Display (for headers)
 */
interface RateLimitInfoProps {
  remaining: number;
  total: number;
  resetTime?: number | null;
}

export const RateLimitInfo: React.FC<RateLimitInfoProps> = ({
  remaining,
  total,
  resetTime,
}) => {
  const { t } = useTranslation();
  const percent = Math.round((remaining / total) * 100);
  
  const statusColor = useMemo(() => {
    if (percent > 50) return '#52c41a';
    if (percent > 20) return '#faad14';
    return '#ff4d4f';
  }, [percent]);

  return (
    <div className="rate-limit-info">
      <Text type="secondary">
        {t('rateLimit.remaining', '{{remaining}}/{{total}} requests remaining', { remaining, total })}
      </Text>
      <Progress
        percent={percent}
        size="small"
        showInfo={false}
        strokeColor={statusColor}
        style={{ width: 100 }}
      />
    </div>
  );
};

export default RateLimitAlert;
