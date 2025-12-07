/**
 * Access Denied Page
 *
 * Displayed when users attempt to access restricted admin functions.
 * Shows friendly error message with admin contact information.
 */

import React, { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Result, Button, Card, Space, Typography, Divider, Alert } from 'antd';
import {
  LockOutlined,
  HomeOutlined,
  ArrowLeftOutlined,
  MailOutlined,
  QuestionCircleOutlined,
  UserOutlined,
  SafetyOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { usePermissions } from '../hooks/usePermissions';
import { RoleBadge } from '../components/common/PermissionGate';

const { Text, Paragraph, Title } = Typography;

// Admin contact information
const ADMIN_CONTACT = {
  email: 'admin@coderev.example.com',
  supportUrl: 'https://coderev.example.com/support',
  documentationUrl: 'https://coderev.example.com/docs/permissions',
};

interface AccessDeniedProps {
  /** Optional custom message */
  message?: string;
  /** Required role that was denied */
  requiredRole?: string;
  /** Required permission that was denied */
  requiredPermission?: string;
  /** Show the attempted path */
  showAttemptedPath?: boolean;
}

const AccessDenied: React.FC<AccessDeniedProps> = ({
  message,
  requiredRole = 'admin',
  requiredPermission,
  showAttemptedPath = true,
}) => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const { user, role, getRoleLabel, isAuthenticated } = usePermissions();

  // Get the attempted path from location state or current path
  const attemptedPath = (location.state as any)?.from || location.pathname;

  // Log the access denial attempt
  useEffect(() => {
    if (isAuthenticated) {
      // Log to console for debugging
      console.warn('[AccessDenied] Unauthorized access attempt:', {
        user: user?.id,
        role,
        attemptedPath,
        requiredRole,
        requiredPermission,
        timestamp: new Date().toISOString(),
      });

      // Send to backend for audit logging
      fetch('/api/audit/access-denial', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: user?.id,
          userRole: role,
          attemptedPath,
          requiredRole,
          requiredPermission,
          timestamp: new Date().toISOString(),
        }),
        credentials: 'include',
      }).catch(() => {
        // Silently fail - audit logging shouldn't break the page
      });
    }
  }, [isAuthenticated, user, role, attemptedPath, requiredRole, requiredPermission]);

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%)',
        padding: 24,
      }}
    >
      <Card
        style={{
          maxWidth: 600,
          width: '100%',
          boxShadow: '0 4px 24px rgba(0, 0, 0, 0.1)',
          borderRadius: 12,
        }}
      >
        <Result
          status="403"
          icon={
            <div style={{ position: 'relative', display: 'inline-block' }}>
              <LockOutlined
                style={{
                  fontSize: 72,
                  color: '#ff4d4f',
                }}
              />
              <SafetyOutlined
                style={{
                  position: 'absolute',
                  bottom: -4,
                  right: -4,
                  fontSize: 28,
                  color: '#faad14',
                  background: '#fff',
                  borderRadius: '50%',
                  padding: 4,
                }}
              />
            </div>
          }
          title={
            <Title level={2} style={{ margin: 0 }}>
              {t('auth.access_denied', 'Access Denied')}
            </Title>
          }
          subTitle={
            <div style={{ textAlign: 'left', marginTop: 16 }}>
              <Alert
                type="error"
                showIcon
                icon={<LockOutlined />}
                message={t('auth.admin_only', 'Administrator Access Required')}
                description={
                  message ||
                  t('auth.admin_only_message', 'This function is only available to administrators. Regular users cannot access this area.')
                }
                style={{ marginBottom: 16 }}
              />

              {/* User Info Section */}
              <Card size="small" style={{ marginBottom: 16, background: '#fafafa' }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Text type="secondary">
                      <UserOutlined style={{ marginRight: 8 }} />
                      {t('auth.your_account', 'Your Account')}:
                    </Text>
                    <Text strong>{user?.email || user?.name || 'Unknown'}</Text>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Text type="secondary">
                      {t('auth.current_role', 'Current Role')}:
                    </Text>
                    <RoleBadge role={role} />
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Text type="secondary">
                      {t('auth.required_role', 'Required Role')}:
                    </Text>
                    <RoleBadge role={requiredRole as any} />
                  </div>

                  {showAttemptedPath && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Text type="secondary">
                        {t('auth.attempted_path', 'Attempted Path')}:
                      </Text>
                      <Text code style={{ fontSize: 12 }}>{attemptedPath}</Text>
                    </div>
                  )}
                </Space>
              </Card>

              <Divider style={{ margin: '16px 0' }} />

              {/* Contact Admin Section */}
              <div>
                <Title level={5} style={{ marginBottom: 12 }}>
                  <QuestionCircleOutlined style={{ marginRight: 8 }} />
                  {t('auth.need_access', 'Need Access?')}
                </Title>

                <Paragraph type="secondary" style={{ marginBottom: 12 }}>
                  {t(
                    'auth.contact_admin_message',
                    'If you believe you should have access to this feature, please contact your system administrator.'
                  )}
                </Paragraph>

                <Space direction="vertical" size="small" style={{ width: '100%' }}>
                  <Button
                    type="link"
                    icon={<MailOutlined />}
                    href={`mailto:${ADMIN_CONTACT.email}?subject=Access Request: ${attemptedPath}&body=I would like to request access to: ${attemptedPath}%0A%0AMy current role: ${getRoleLabel()}%0ARequired role: ${requiredRole}`}
                    style={{ padding: 0, height: 'auto' }}
                  >
                    {t('auth.email_admin', 'Email Administrator')}: {ADMIN_CONTACT.email}
                  </Button>

                  <Button
                    type="link"
                    icon={<QuestionCircleOutlined />}
                    href={ADMIN_CONTACT.supportUrl}
                    target="_blank"
                    style={{ padding: 0, height: 'auto' }}
                  >
                    {t('auth.visit_support', 'Visit Support Center')}
                  </Button>
                </Space>
              </div>
            </div>
          }
          extra={
            <Space size="middle" style={{ marginTop: 24 }}>
              <Button
                size="large"
                icon={<ArrowLeftOutlined />}
                onClick={() => navigate(-1)}
              >
                {t('common.go_back', 'Go Back')}
              </Button>
              <Button
                type="primary"
                size="large"
                icon={<HomeOutlined />}
                onClick={() => navigate('/dashboard')}
              >
                {t('common.go_home', 'Go to Dashboard')}
              </Button>
            </Space>
          }
        />
      </Card>
    </div>
  );
};

export default AccessDenied;
