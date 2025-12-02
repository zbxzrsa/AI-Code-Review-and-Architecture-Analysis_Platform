/**
 * Two-Factor Authentication Settings Component
 * 
 * Manages 2FA in user settings:
 * - View 2FA status
 * - Enable/disable 2FA
 * - View/regenerate backup codes
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Button,
  Typography,
  Space,
  Alert,
  Badge,
  Modal,
  List,
  Descriptions,
  Skeleton,
  message,
  Divider,
  Tag,
} from 'antd';
import {
  SafetyOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  KeyOutlined,
  ReloadOutlined,
  DownloadOutlined,
  CopyOutlined,
  ExclamationCircleOutlined,
  QrcodeOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { apiService } from '../../../services/api';
import { TwoFactorSetup, TwoFactorDisable } from '../../auth/TwoFactorAuth';
import { twoFactorAuth } from '../../../services/security';
import './TwoFactorSettings.css';

const { Title, Text, Paragraph } = Typography;

interface TwoFactorStatus {
  enabled: boolean;
  enabled_at: string | null;
  backup_codes_remaining: number;
  last_used: string | null;
}

export const TwoFactorSettings: React.FC = () => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState<TwoFactorStatus | null>(null);
  const [showSetup, setShowSetup] = useState(false);
  const [showDisable, setShowDisable] = useState(false);
  const [showBackupCodes, setShowBackupCodes] = useState(false);
  const [backupCodes, setBackupCodes] = useState<string[]>([]);
  const [regeneratingCodes, setRegeneratingCodes] = useState(false);

  // Fetch 2FA status
  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true);
      const response = await apiService.user.get2FAStatus();
      setStatus(response.data);
    } catch (error) {
      message.error(t('settings.2fa.fetchError', 'Failed to load 2FA status'));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Handle setup completion
  const handleSetupComplete = () => {
    setShowSetup(false);
    fetchStatus();
    message.success(t('settings.2fa.enabled', '2FA has been enabled!'));
  };

  // Handle disable completion
  const handleDisabled = () => {
    setShowDisable(false);
    fetchStatus();
    message.success(t('settings.2fa.disabled', '2FA has been disabled'));
  };

  // Regenerate backup codes
  const handleRegenerateBackupCodes = async () => {
    Modal.confirm({
      title: t('settings.2fa.regenerateTitle', 'Regenerate Backup Codes?'),
      icon: <ExclamationCircleOutlined />,
      content: t(
        'settings.2fa.regenerateWarning',
        'This will invalidate all existing backup codes. Make sure to save the new codes.'
      ),
      okText: t('common.confirm', 'Confirm'),
      cancelText: t('common.cancel', 'Cancel'),
      onOk: async () => {
        try {
          setRegeneratingCodes(true);
          const response = await apiService.user.regenerateBackupCodes('');
          setBackupCodes(response.data.backup_codes);
          setShowBackupCodes(true);
          fetchStatus();
        } catch (error) {
          message.error(t('settings.2fa.regenerateError', 'Failed to regenerate codes'));
        } finally {
          setRegeneratingCodes(false);
        }
      },
    });
  };

  // Copy backup codes
  const copyBackupCodes = () => {
    navigator.clipboard.writeText(backupCodes.join('\n'));
    message.success(t('common.copied', 'Copied to clipboard'));
  };

  // Download backup codes
  const downloadBackupCodes = () => {
    const content = [
      `AI Code Review - Backup Codes`,
      `Generated: ${new Date().toLocaleString()}`,
      '',
      'Keep these codes safe. Each code can only be used once.',
      '',
      ...backupCodes,
    ].join('\n');

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'backup-codes.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Format date for display
  const formatDate = (dateString: string | null) => {
    if (!dateString) return t('common.never', 'Never');
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <Card>
        <Skeleton active />
      </Card>
    );
  }

  return (
    <div className="two-factor-settings">
      <Card>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {/* Header */}
          <div className="settings-header">
            <Space>
              <SafetyOutlined className="settings-icon" />
              <div>
                <Title level={4} style={{ margin: 0 }}>
                  {t('settings.2fa.title', 'Two-Factor Authentication')}
                </Title>
                <Text type="secondary">
                  {t(
                    'settings.2fa.description',
                    'Add an extra layer of security to your account'
                  )}
                </Text>
              </div>
            </Space>
            <Badge
              status={status?.enabled ? 'success' : 'default'}
              text={
                status?.enabled
                  ? t('settings.2fa.statusEnabled', 'Enabled')
                  : t('settings.2fa.statusDisabled', 'Disabled')
              }
            />
          </div>

          <Divider />

          {/* Status Info */}
          {status?.enabled ? (
            <>
              <Alert
                type="success"
                icon={<CheckCircleOutlined />}
                message={t('settings.2fa.secureMessage', 'Your account is secured with 2FA')}
                description={t(
                  'settings.2fa.secureDescription',
                  'You will be asked for a verification code when logging in.'
                )}
                showIcon
              />

              <Descriptions column={1} bordered size="small">
                <Descriptions.Item label={t('settings.2fa.enabledAt', 'Enabled')}>
                  {formatDate(status.enabled_at)}
                </Descriptions.Item>
                <Descriptions.Item label={t('settings.2fa.lastUsed', 'Last Used')}>
                  {formatDate(status.last_used)}
                </Descriptions.Item>
                <Descriptions.Item label={t('settings.2fa.backupCodesRemaining', 'Backup Codes')}>
                  <Space>
                    <Tag color={status.backup_codes_remaining > 3 ? 'green' : 'orange'}>
                      {status.backup_codes_remaining} {t('settings.2fa.remaining', 'remaining')}
                    </Tag>
                    {status.backup_codes_remaining <= 3 && (
                      <Text type="warning">
                        {t('settings.2fa.lowBackupCodes', 'Consider regenerating backup codes')}
                      </Text>
                    )}
                  </Space>
                </Descriptions.Item>
              </Descriptions>

              {/* Actions when enabled */}
              <Space wrap>
                <Button
                  icon={<KeyOutlined />}
                  onClick={handleRegenerateBackupCodes}
                  loading={regeneratingCodes}
                >
                  {t('settings.2fa.regenerateBackupCodes', 'Regenerate Backup Codes')}
                </Button>
                <Button danger icon={<CloseCircleOutlined />} onClick={() => setShowDisable(true)}>
                  {t('settings.2fa.disable', 'Disable 2FA')}
                </Button>
              </Space>
            </>
          ) : (
            <>
              <Alert
                type="warning"
                icon={<ExclamationCircleOutlined />}
                message={t('settings.2fa.notEnabledMessage', '2FA is not enabled')}
                description={t(
                  'settings.2fa.notEnabledDescription',
                  'Enable two-factor authentication to add an extra layer of security to your account.'
                )}
                showIcon
              />

              <div className="two-factor-benefits">
                <Title level={5}>{t('settings.2fa.benefits', 'Benefits of 2FA')}</Title>
                <List
                  size="small"
                  dataSource={[
                    t('settings.2fa.benefit1', 'Protects against password theft'),
                    t('settings.2fa.benefit2', 'Blocks unauthorized access attempts'),
                    t('settings.2fa.benefit3', 'Alerts you to suspicious login activity'),
                    t('settings.2fa.benefit4', 'Works with popular authenticator apps'),
                  ]}
                  renderItem={(item) => (
                    <List.Item>
                      <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 8 }} />
                      {item}
                    </List.Item>
                  )}
                />
              </div>

              <Button type="primary" icon={<QrcodeOutlined />} onClick={() => setShowSetup(true)}>
                {t('settings.2fa.enable', 'Enable 2FA')}
              </Button>
            </>
          )}

          {/* Supported Apps */}
          <div className="supported-apps">
            <Text type="secondary">{t('settings.2fa.supportedApps', 'Supported authenticator apps:')}</Text>
            <Space style={{ marginTop: 8 }}>
              <Tag>Google Authenticator</Tag>
              <Tag>Authy</Tag>
              <Tag>Microsoft Authenticator</Tag>
              <Tag>1Password</Tag>
            </Space>
          </div>
        </Space>
      </Card>

      {/* Setup Modal */}
      <Modal
        title={null}
        open={showSetup}
        onCancel={() => setShowSetup(false)}
        footer={null}
        width={520}
        destroyOnClose
      >
        <TwoFactorSetup onComplete={handleSetupComplete} onCancel={() => setShowSetup(false)} />
      </Modal>

      {/* Disable Modal */}
      <TwoFactorDisable
        visible={showDisable}
        onClose={() => setShowDisable(false)}
        onDisabled={handleDisabled}
      />

      {/* Backup Codes Modal */}
      <Modal
        title={
          <Space>
            <KeyOutlined />
            {t('settings.2fa.backupCodesTitle', 'Your New Backup Codes')}
          </Space>
        }
        open={showBackupCodes}
        onCancel={() => setShowBackupCodes(false)}
        footer={[
          <Button key="copy" icon={<CopyOutlined />} onClick={copyBackupCodes}>
            {t('common.copy', 'Copy')}
          </Button>,
          <Button key="download" icon={<DownloadOutlined />} onClick={downloadBackupCodes}>
            {t('common.download', 'Download')}
          </Button>,
          <Button key="done" type="primary" onClick={() => setShowBackupCodes(false)}>
            {t('common.done', 'Done')}
          </Button>,
        ]}
        width={400}
      >
        <Alert
          type="warning"
          message={t(
            'settings.2fa.saveBackupCodes',
            'Save these codes in a secure place. Each code can only be used once.'
          )}
          showIcon
          style={{ marginBottom: 16 }}
        />
        <List
          bordered
          dataSource={backupCodes}
          renderItem={(code) => (
            <List.Item className="backup-code-item">
              <code>{twoFactorAuth.formatBackupCode(code)}</code>
            </List.Item>
          )}
        />
      </Modal>
    </div>
  );
};

export default TwoFactorSettings;
