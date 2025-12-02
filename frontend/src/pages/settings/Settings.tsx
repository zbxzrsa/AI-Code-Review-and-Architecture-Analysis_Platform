/**
 * Enhanced User Settings Page
 * 
 * Comprehensive settings page with:
 * - Preferences (theme, language, editor, analysis defaults)
 * - Security (password, 2FA, sessions, IP whitelist)
 * - API Keys management
 * - Integrations (Slack, Teams, webhooks)
 * - Notifications (email, in-app, DND)
 * - Data & Privacy
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  Card,
  Tabs,
  Form,
  Input,
  Select,
  Switch,
  Button,
  Space,
  Typography,
  Divider,
  message,
  Alert,
  List,
  Tag,
  Modal,
  Table,
  Popconfirm,
  Row,
  Col,
  Skeleton,
  Empty,
  Badge,
  Tooltip,
  InputNumber,
  TimePicker,
  Checkbox,
  Progress,
  Statistic,
  Steps,
} from 'antd';
import type { TabsProps, TableProps } from 'antd';
import {
  UserOutlined,
  LockOutlined,
  BellOutlined,
  GlobalOutlined,
  SafetyOutlined,
  ApiOutlined,
  DeleteOutlined,
  PlusOutlined,
  KeyOutlined,
  SettingOutlined,
  EditOutlined,
  CopyOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  DesktopOutlined,
  MobileOutlined,
  LaptopOutlined,
  EnvironmentOutlined,
  ClockCircleOutlined,
  SlackOutlined,
  WindowsOutlined,
  LinkOutlined,
  DisconnectOutlined,
  EyeOutlined,
  EyeInvisibleOutlined,
  MailOutlined,
  SoundOutlined,
  MoonOutlined,
  SunOutlined,
  CodeOutlined,
  ThunderboltOutlined,
  HistoryOutlined,
  QrcodeOutlined,
  ShieldOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useUIStore } from '../../store/uiStore';
import { 
  useAuthStore, 
  type Session, 
  type UserApiKey, 
  type Integration,
  type UserWebhook,
  defaultUserSettings,
} from '../../store/authStore';
import {
  useUserSettings,
  useUpdateSettings,
  useUpdateNotifications,
  useChangePassword,
  use2FAStatus,
  useSetup2FA,
  useEnable2FA,
  useDisable2FA,
  useRegenerateBackupCodes,
  useSessions,
  useRevokeSession,
  useRevokeAllSessions,
  useApiKeys,
  useCreateApiKey,
  useRevokeApiKey,
  useIntegrations,
  useConnectSlack,
  useDisconnectSlack,
  useConnectTeams,
  useDisconnectTeams,
  useUserWebhooks,
  useCreateUserWebhook,
  useUpdateUserWebhook,
  useDeleteUserWebhook,
  useTestUserWebhook,
  useIpWhitelist,
  useAddIpToWhitelist,
  useRemoveIpFromWhitelist,
  useLoginAlerts,
  useUpdateLoginAlerts,
} from '../../hooks/useUser';
import './Settings.css';

const { Title, Text, Paragraph } = Typography;

// ============================================
// Preferences Section
// ============================================
const PreferencesSection: React.FC = () => {
  const { t, i18n } = useTranslation();
  const { theme, setTheme, language, setLanguage } = useUIStore();
  const { settings } = useAuthStore();
  const updateSettings = useUpdateSettings();

  const handleThemeChange = (value: 'light' | 'dark' | 'system') => {
    setTheme(value);
    updateSettings.mutate({ theme: value });
  };

  const handleLanguageChange = (value: string) => {
    setLanguage(value as any);
    i18n.changeLanguage(value);
    updateSettings.mutate({ language: value });
  };

  return (
    <div className="settings-section">
      {/* Appearance */}
      <Title level={4}><SunOutlined /> {t('settings.appearance', 'Appearance')}</Title>
      <Form layout="vertical" className="settings-form">
        <Row gutter={16}>
          <Col xs={24} sm={12}>
            <Form.Item label={t('settings.theme', 'Theme')}>
              <Select
                value={theme}
                onChange={handleThemeChange}
                options={[
                  { value: 'light', label: <><SunOutlined /> {t('settings.theme_light', 'Light')}</> },
                  { value: 'dark', label: <><MoonOutlined /> {t('settings.theme_dark', 'Dark')}</> },
                  { value: 'system', label: <><DesktopOutlined /> {t('settings.theme_system', 'System')}</> },
                ]}
              />
            </Form.Item>
          </Col>
          <Col xs={24} sm={12}>
            <Form.Item label={t('settings.language', 'Language')}>
              <Select
                value={language}
                onChange={handleLanguageChange}
                options={[
                  { value: 'en', label: '🇺🇸 English' },
                  { value: 'zh-CN', label: '🇨🇳 简体中文' },
                  { value: 'zh-TW', label: '🇹🇼 繁體中文' },
                ]}
              />
            </Form.Item>
          </Col>
        </Row>
      </Form>

      <Divider />

      {/* Editor Preferences */}
      <Title level={4}><CodeOutlined /> {t('settings.editor', 'Editor Preferences')}</Title>
      <Form layout="vertical" className="settings-form">
        <Row gutter={16}>
          <Col xs={24} sm={12}>
            <Form.Item label={t('settings.font_size', 'Font Size')}>
              <Select
                value={settings?.editorFontSize || 14}
                onChange={(v) => updateSettings.mutate({ editorFontSize: v })}
                options={[
                  { value: 12, label: '12px' },
                  { value: 14, label: '14px (Default)' },
                  { value: 16, label: '16px' },
                  { value: 18, label: '18px' },
                  { value: 20, label: '20px' },
                ]}
              />
            </Form.Item>
          </Col>
          <Col xs={24} sm={12}>
            <Form.Item label={t('settings.tab_size', 'Tab Size')}>
              <Select
                value={settings?.editorTabSize || 2}
                onChange={(v) => updateSettings.mutate({ editorTabSize: v })}
                options={[
                  { value: 2, label: '2 spaces' },
                  { value: 4, label: '4 spaces' },
                ]}
              />
            </Form.Item>
          </Col>
        </Row>
        <Space direction="vertical" className="editor-toggles">
          <div className="toggle-item">
            <Switch
              checked={settings?.editorLineNumbers ?? true}
              onChange={(v) => updateSettings.mutate({ editorLineNumbers: v })}
            />
            <Text>{t('settings.line_numbers', 'Show line numbers')}</Text>
          </div>
          <div className="toggle-item">
            <Switch
              checked={settings?.editorMinimap ?? true}
              onChange={(v) => updateSettings.mutate({ editorMinimap: v })}
            />
            <Text>{t('settings.minimap', 'Show minimap')}</Text>
          </div>
          <div className="toggle-item">
            <Switch
              checked={settings?.editorWordWrap ?? true}
              onChange={(v) => updateSettings.mutate({ editorWordWrap: v })}
            />
            <Text>{t('settings.word_wrap', 'Word wrap')}</Text>
          </div>
        </Space>
      </Form>

      <Divider />

      {/* Analysis Defaults */}
      <Title level={4}><ThunderboltOutlined /> {t('settings.analysis_defaults', 'Analysis Defaults')}</Title>
      <Form layout="vertical" className="settings-form">
        <Row gutter={16}>
          <Col xs={24} sm={12}>
            <Form.Item label={t('settings.default_model', 'Default AI Model')}>
              <Select
                value={settings?.defaultAiModel || 'gpt-4'}
                onChange={(v) => updateSettings.mutate({ defaultAiModel: v })}
                options={[
                  { value: 'gpt-4', label: 'GPT-4 (Most Capable)' },
                  { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo (Fast)' },
                  { value: 'claude-3-opus', label: 'Claude 3 Opus' },
                  { value: 'claude-3-sonnet', label: 'Claude 3 Sonnet' },
                ]}
              />
            </Form.Item>
          </Col>
          <Col xs={24} sm={12}>
            <Form.Item label={t('settings.analysis_depth', 'Analysis Depth')}>
              <Select
                value={settings?.defaultAnalysisDepth || 'standard'}
                onChange={(v) => updateSettings.mutate({ defaultAnalysisDepth: v })}
                options={[
                  { value: 'quick', label: t('settings.depth_quick', 'Quick (Fast)') },
                  { value: 'standard', label: t('settings.depth_standard', 'Standard (Balanced)') },
                  { value: 'deep', label: t('settings.depth_deep', 'Deep (Thorough)') },
                ]}
              />
            </Form.Item>
          </Col>
        </Row>
        <div className="toggle-item">
          <Switch
            checked={settings?.autoAnalyzeOnPush ?? false}
            onChange={(v) => updateSettings.mutate({ autoAnalyzeOnPush: v })}
          />
          <Text>{t('settings.auto_analyze', 'Auto-analyze on git push')}</Text>
        </div>
      </Form>
    </div>
  );
};

// ============================================
// Security Section
// ============================================
const SecuritySection: React.FC = () => {
  const { t } = useTranslation();
  const [passwordForm] = Form.useForm();
  const [setup2FAModalOpen, setSetup2FAModalOpen] = useState(false);
  const [disable2FAModalOpen, setDisable2FAModalOpen] = useState(false);
  const [qrData, setQrData] = useState<{ qrCode: string; secret: string; backupCodes: string[] } | null>(null);
  const [verificationCode, setVerificationCode] = useState('');
  const [disablePassword, setDisablePassword] = useState('');
  const [disableCode, setDisableCode] = useState('');

  const changePassword = useChangePassword();
  const { data: twoFactorStatus, isLoading: loading2FA } = use2FAStatus();
  const setup2FA = useSetup2FA();
  const enable2FA = useEnable2FA();
  const disable2FA = useDisable2FA();
  const regenerateBackupCodes = useRegenerateBackupCodes();
  const { data: sessions, isLoading: sessionsLoading } = useSessions();
  const revokeSession = useRevokeSession();
  const revokeAllSessions = useRevokeAllSessions();
  const { data: ipWhitelist } = useIpWhitelist();
  const addIp = useAddIpToWhitelist();
  const removeIp = useRemoveIpFromWhitelist();
  const { data: loginAlerts } = useLoginAlerts();
  const updateLoginAlerts = useUpdateLoginAlerts();

  const handlePasswordChange = async (values: any) => {
    await changePassword.mutateAsync({
      currentPassword: values.currentPassword,
      newPassword: values.newPassword,
    });
    passwordForm.resetFields();
  };

  const handleSetup2FA = async () => {
    const result = await setup2FA.mutateAsync();
    setQrData(result);
    setSetup2FAModalOpen(true);
  };

  const handleEnable2FA = async () => {
    await enable2FA.mutateAsync(verificationCode);
    setSetup2FAModalOpen(false);
    setVerificationCode('');
    setQrData(null);
  };

  const handleDisable2FA = async () => {
    await disable2FA.mutateAsync({ code: disableCode, password: disablePassword });
    setDisable2FAModalOpen(false);
    setDisableCode('');
    setDisablePassword('');
  };

  const getDeviceIcon = (device: string) => {
    const lower = device.toLowerCase();
    if (lower.includes('mobile') || lower.includes('phone')) return <MobileOutlined />;
    if (lower.includes('tablet')) return <LaptopOutlined />;
    return <DesktopOutlined />;
  };

  const sessionColumns: TableProps<Session>['columns'] = [
    {
      title: t('settings.session_device', 'Device'),
      key: 'device',
      render: (_, record) => (
        <Space>
          {getDeviceIcon(record.device)}
          <div>
            <Text strong>{record.browser}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>{record.device} • {record.os}</Text>
          </div>
        </Space>
      ),
    },
    {
      title: t('settings.session_location', 'Location'),
      key: 'location',
      render: (_, record) => (
        <Space>
          <EnvironmentOutlined />
          <div>
            <Text>{record.location || 'Unknown'}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>{record.ip}</Text>
          </div>
        </Space>
      ),
    },
    {
      title: t('settings.session_last_active', 'Last Active'),
      dataIndex: 'lastActive',
      key: 'lastActive',
      render: (time: string) => new Date(time).toLocaleString(),
    },
    {
      title: '',
      key: 'actions',
      width: 100,
      render: (_, record) => (
        record.isCurrent ? (
          <Tag color="green">{t('settings.current_session', 'Current')}</Tag>
        ) : (
          <Button 
            type="link" 
            danger 
            onClick={() => revokeSession.mutate(record.id)}
            loading={revokeSession.isPending}
          >
            {t('settings.revoke', 'Revoke')}
          </Button>
        )
      ),
    },
  ];

  return (
    <div className="settings-section">
      {/* Change Password */}
      <Title level={4}><LockOutlined /> {t('settings.change_password', 'Change Password')}</Title>
      <Form
        form={passwordForm}
        layout="vertical"
        onFinish={handlePasswordChange}
        className="password-form"
      >
        <Form.Item
          name="currentPassword"
          label={t('settings.current_password', 'Current Password')}
          rules={[{ required: true, message: t('settings.current_password_required', 'Required') }]}
        >
          <Input.Password prefix={<LockOutlined />} />
        </Form.Item>
        <Form.Item
          name="newPassword"
          label={t('settings.new_password', 'New Password')}
          rules={[
            { required: true, message: t('settings.new_password_required', 'Required') },
            { min: 8, message: t('settings.password_min_length', 'At least 8 characters') },
            { 
              pattern: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])/,
              message: t('settings.password_requirements', 'Include uppercase, lowercase, number, and special character'),
            },
          ]}
        >
          <Input.Password prefix={<LockOutlined />} />
        </Form.Item>
        <Form.Item
          name="confirmPassword"
          label={t('settings.confirm_password', 'Confirm Password')}
          dependencies={['newPassword']}
          rules={[
            { required: true, message: t('settings.confirm_password_required', 'Required') },
            ({ getFieldValue }) => ({
              validator(_, value) {
                if (!value || getFieldValue('newPassword') === value) {
                  return Promise.resolve();
                }
                return Promise.reject(new Error(t('settings.passwords_not_match', 'Passwords do not match')));
              },
            }),
          ]}
        >
          <Input.Password prefix={<LockOutlined />} />
        </Form.Item>
        <Form.Item>
          <Button type="primary" htmlType="submit" loading={changePassword.isPending}>
            {t('settings.update_password', 'Update Password')}
          </Button>
        </Form.Item>
      </Form>

      <Divider />

      {/* Two-Factor Authentication */}
      <Title level={4}><ShieldOutlined /> {t('settings.two_factor', 'Two-Factor Authentication')}</Title>
      {loading2FA ? (
        <Skeleton active paragraph={{ rows: 2 }} />
      ) : twoFactorStatus?.enabled ? (
        <Alert
          message={t('settings.two_factor_enabled', 'Two-factor authentication is enabled')}
          description={
            <Space direction="vertical">
              <Text>{t('settings.backup_codes_remaining', 'Backup codes remaining')}: {twoFactorStatus.backupCodesRemaining}</Text>
              <Space>
                <Button onClick={() => regenerateBackupCodes.mutate('')}>
                  {t('settings.regenerate_codes', 'Regenerate Backup Codes')}
                </Button>
                <Button danger onClick={() => setDisable2FAModalOpen(true)}>
                  {t('settings.disable_2fa', 'Disable 2FA')}
                </Button>
              </Space>
            </Space>
          }
          type="success"
          showIcon
          icon={<CheckCircleOutlined />}
        />
      ) : (
        <Alert
          message={t('settings.two_factor_disabled', 'Two-factor authentication is not enabled')}
          description={t('settings.two_factor_desc', 'Add an extra layer of security to your account')}
          type="info"
          showIcon
          action={
            <Button type="primary" onClick={handleSetup2FA} loading={setup2FA.isPending}>
              {t('settings.enable_2fa', 'Enable 2FA')}
            </Button>
          }
        />
      )}

      <Divider />

      {/* Active Sessions */}
      <div className="section-header">
        <Title level={4}><DesktopOutlined /> {t('settings.sessions', 'Active Sessions')}</Title>
        <Button 
          danger 
          onClick={() => revokeAllSessions.mutate()}
          loading={revokeAllSessions.isPending}
        >
          {t('settings.revoke_all', 'Revoke All Others')}
        </Button>
      </div>
      <Table
        columns={sessionColumns}
        dataSource={sessions || []}
        rowKey="id"
        loading={sessionsLoading}
        pagination={false}
        size="small"
        locale={{ emptyText: <Empty description={t('settings.no_sessions', 'No active sessions')} /> }}
      />

      <Divider />

      {/* Login Alerts */}
      <Title level={4}><BellOutlined /> {t('settings.login_alerts', 'Login Alerts')}</Title>
      <Space direction="vertical" className="alert-toggles">
        <div className="toggle-item">
          <Switch
            checked={loginAlerts?.emailOnNewDevice ?? true}
            onChange={(v) => updateLoginAlerts.mutate({ emailOnNewDevice: v })}
          />
          <Text>{t('settings.alert_new_device', 'Email me when logging in from a new device')}</Text>
        </div>
        <div className="toggle-item">
          <Switch
            checked={loginAlerts?.emailOnNewLocation ?? true}
            onChange={(v) => updateLoginAlerts.mutate({ emailOnNewLocation: v })}
          />
          <Text>{t('settings.alert_new_location', 'Email me when logging in from a new location')}</Text>
        </div>
        <div className="toggle-item">
          <Switch
            checked={loginAlerts?.emailOnFailedAttempts ?? true}
            onChange={(v) => updateLoginAlerts.mutate({ emailOnFailedAttempts: v })}
          />
          <Text>{t('settings.alert_failed', 'Email me about failed login attempts')}</Text>
        </div>
      </Space>

      {/* 2FA Setup Modal */}
      <Modal
        title={t('settings.setup_2fa', 'Set Up Two-Factor Authentication')}
        open={setup2FAModalOpen}
        onCancel={() => {
          setSetup2FAModalOpen(false);
          setQrData(null);
          setVerificationCode('');
        }}
        footer={null}
        width={480}
      >
        {qrData && (
          <Steps
            direction="vertical"
            current={-1}
            items={[
              {
                title: t('settings.2fa_step1', 'Scan QR Code'),
                description: (
                  <div className="qr-code-section">
                    <img src={qrData.qrCode} alt="QR Code" className="qr-code-image" />
                    <Text type="secondary">{t('settings.2fa_scan_desc', 'Scan with your authenticator app')}</Text>
                    <div className="secret-code">
                      <Text code copyable>{qrData.secret}</Text>
                    </div>
                  </div>
                ),
              },
              {
                title: t('settings.2fa_step2', 'Save Backup Codes'),
                description: (
                  <div className="backup-codes">
                    <Alert
                      message={t('settings.backup_codes_warning', 'Save these codes in a safe place')}
                      type="warning"
                      showIcon
                    />
                    <div className="codes-grid">
                      {qrData.backupCodes.map((code, i) => (
                        <Text key={i} code>{code}</Text>
                      ))}
                    </div>
                  </div>
                ),
              },
              {
                title: t('settings.2fa_step3', 'Verify'),
                description: (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Input
                      placeholder={t('settings.enter_code', 'Enter 6-digit code')}
                      value={verificationCode}
                      onChange={(e) => setVerificationCode(e.target.value)}
                      maxLength={6}
                    />
                    <Button 
                      type="primary" 
                      onClick={handleEnable2FA}
                      loading={enable2FA.isPending}
                      disabled={verificationCode.length !== 6}
                      block
                    >
                      {t('settings.verify_enable', 'Verify & Enable')}
                    </Button>
                  </Space>
                ),
              },
            ]}
          />
        )}
      </Modal>

      {/* Disable 2FA Modal */}
      <Modal
        title={t('settings.disable_2fa', 'Disable Two-Factor Authentication')}
        open={disable2FAModalOpen}
        onCancel={() => {
          setDisable2FAModalOpen(false);
          setDisableCode('');
          setDisablePassword('');
        }}
        footer={[
          <Button key="cancel" onClick={() => setDisable2FAModalOpen(false)}>
            {t('common.cancel', 'Cancel')}
          </Button>,
          <Button 
            key="disable" 
            type="primary" 
            danger 
            onClick={handleDisable2FA}
            loading={disable2FA.isPending}
          >
            {t('settings.disable_2fa', 'Disable 2FA')}
          </Button>,
        ]}
      >
        <Alert
          message={t('settings.disable_2fa_warning', 'This will reduce your account security')}
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Form layout="vertical">
          <Form.Item label={t('settings.enter_2fa_code', 'Enter 2FA code')}>
            <Input
              value={disableCode}
              onChange={(e) => setDisableCode(e.target.value)}
              placeholder="000000"
              maxLength={6}
            />
          </Form.Item>
          <Form.Item label={t('settings.enter_password', 'Enter your password')}>
            <Input.Password
              value={disablePassword}
              onChange={(e) => setDisablePassword(e.target.value)}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

// ============================================
// API Keys Section
// ============================================
const ApiKeysSection: React.FC = () => {
  const { t } = useTranslation();
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [keyName, setKeyName] = useState('');
  const [keyPermissions, setKeyPermissions] = useState<string[]>([]);
  const [keyExpiry, setKeyExpiry] = useState<string | undefined>();

  const { data: apiKeys, isLoading } = useApiKeys();
  const createApiKey = useCreateApiKey();
  const revokeApiKey = useRevokeApiKey();

  const permissionOptions = [
    { value: 'read:projects', label: 'Read Projects' },
    { value: 'write:projects', label: 'Write Projects' },
    { value: 'read:analysis', label: 'Read Analysis' },
    { value: 'write:analysis', label: 'Trigger Analysis' },
    { value: 'read:issues', label: 'Read Issues' },
    { value: 'write:issues', label: 'Manage Issues' },
  ];

  const handleCreate = async () => {
    const result = await createApiKey.mutateAsync({
      name: keyName,
      permissions: keyPermissions,
      expiresAt: keyExpiry,
    });
    setNewKey(result.key);
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    message.success(t('common.copied', 'Copied!'));
  };

  const columns: TableProps<UserApiKey>['columns'] = [
    {
      title: t('settings.key_name', 'Name'),
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => <Text strong>{name}</Text>,
    },
    {
      title: t('settings.key_prefix', 'Key'),
      dataIndex: 'prefix',
      key: 'prefix',
      render: (prefix: string) => <Text code>{prefix}...</Text>,
    },
    {
      title: t('settings.key_permissions', 'Permissions'),
      dataIndex: 'permissions',
      key: 'permissions',
      render: (perms: string[]) => (
        <Space wrap>
          {perms.slice(0, 2).map(p => <Tag key={p}>{p}</Tag>)}
          {perms.length > 2 && <Tag>+{perms.length - 2}</Tag>}
        </Space>
      ),
    },
    {
      title: t('settings.key_usage', 'Usage'),
      dataIndex: 'usageCount',
      key: 'usageCount',
      render: (count: number) => count.toLocaleString(),
    },
    {
      title: t('settings.key_last_used', 'Last Used'),
      dataIndex: 'lastUsedAt',
      key: 'lastUsedAt',
      render: (date: string) => date ? new Date(date).toLocaleDateString() : '-',
    },
    {
      title: '',
      key: 'actions',
      width: 80,
      render: (_, record) => (
        <Popconfirm
          title={t('settings.revoke_key_confirm', 'Revoke this API key?')}
          description={t('settings.revoke_key_warning', 'Applications using this key will lose access.')}
          onConfirm={() => revokeApiKey.mutate(record.id)}
          okText={t('common.revoke', 'Revoke')}
          okType="danger"
        >
          <Button type="text" danger icon={<DeleteOutlined />} />
        </Popconfirm>
      ),
    },
  ];

  return (
    <div className="settings-section">
      <div className="section-header">
        <div>
          <Title level={4}><KeyOutlined /> {t('settings.api_keys', 'API Keys')}</Title>
          <Text type="secondary">{t('settings.api_keys_desc', 'Manage API keys for programmatic access')}</Text>
        </div>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModalOpen(true)}>
          {t('settings.create_api_key', 'Create API Key')}
        </Button>
      </div>

      <Table
        columns={columns}
        dataSource={apiKeys || []}
        rowKey="id"
        loading={isLoading}
        pagination={false}
        locale={{ emptyText: <Empty description={t('settings.no_api_keys', 'No API keys')} /> }}
      />

      {/* Create API Key Modal */}
      <Modal
        title={t('settings.create_api_key', 'Create API Key')}
        open={createModalOpen}
        onCancel={() => {
          setCreateModalOpen(false);
          setNewKey(null);
          setKeyName('');
          setKeyPermissions([]);
          setKeyExpiry(undefined);
        }}
        footer={newKey ? [
          <Button key="done" type="primary" onClick={() => {
            setCreateModalOpen(false);
            setNewKey(null);
            setKeyName('');
            setKeyPermissions([]);
          }}>
            {t('common.done', 'Done')}
          </Button>
        ] : [
          <Button key="cancel" onClick={() => setCreateModalOpen(false)}>
            {t('common.cancel', 'Cancel')}
          </Button>,
          <Button 
            key="create" 
            type="primary" 
            onClick={handleCreate}
            loading={createApiKey.isPending}
            disabled={!keyName || keyPermissions.length === 0}
          >
            {t('common.create', 'Create')}
          </Button>
        ]}
      >
        {newKey ? (
          <Alert
            type="success"
            message={t('settings.api_key_created', 'API Key Created')}
            description={
              <>
                <Paragraph>
                  {t('settings.api_key_warning', 'Copy your API key now. You won\'t be able to see it again!')}
                </Paragraph>
                <Input.Group compact>
                  <Input value={newKey} readOnly style={{ width: 'calc(100% - 80px)' }} />
                  <Button icon={<CopyOutlined />} onClick={() => handleCopy(newKey)}>
                    {t('common.copy', 'Copy')}
                  </Button>
                </Input.Group>
              </>
            }
          />
        ) : (
          <Form layout="vertical">
            <Form.Item label={t('settings.key_name', 'Key Name')} required>
              <Input
                value={keyName}
                onChange={(e) => setKeyName(e.target.value)}
                placeholder="e.g., CI/CD Pipeline"
              />
            </Form.Item>
            <Form.Item label={t('settings.key_permissions', 'Permissions')} required>
              <Select
                mode="multiple"
                value={keyPermissions}
                onChange={setKeyPermissions}
                options={permissionOptions}
                placeholder={t('settings.select_permissions', 'Select permissions')}
              />
            </Form.Item>
            <Form.Item label={t('settings.key_expiry', 'Expiration')}>
              <Select
                value={keyExpiry}
                onChange={setKeyExpiry}
                allowClear
                placeholder={t('settings.no_expiry', 'No expiration')}
                options={[
                  { value: '30d', label: '30 days' },
                  { value: '90d', label: '90 days' },
                  { value: '1y', label: '1 year' },
                ]}
              />
            </Form.Item>
          </Form>
        )}
      </Modal>
    </div>
  );
};

// ============================================
// Integrations Section
// ============================================
const IntegrationsSection: React.FC = () => {
  const { t } = useTranslation();
  const [webhookModalOpen, setWebhookModalOpen] = useState(false);
  const [editingWebhook, setEditingWebhook] = useState<UserWebhook | null>(null);
  const [webhookForm] = Form.useForm();

  const { data: integrations, isLoading } = useIntegrations();
  const disconnectSlack = useDisconnectSlack();
  const disconnectTeams = useDisconnectTeams();
  const { data: webhooks } = useUserWebhooks();
  const createWebhook = useCreateUserWebhook();
  const updateWebhook = useUpdateUserWebhook();
  const deleteWebhook = useDeleteUserWebhook();
  const testWebhook = useTestUserWebhook();

  const handleWebhookSubmit = async (values: any) => {
    if (editingWebhook) {
      await updateWebhook.mutateAsync({ webhookId: editingWebhook.id, data: values });
    } else {
      await createWebhook.mutateAsync(values);
    }
    setWebhookModalOpen(false);
    setEditingWebhook(null);
    webhookForm.resetFields();
  };

  const handleEditWebhook = (webhook: UserWebhook) => {
    setEditingWebhook(webhook);
    webhookForm.setFieldsValue(webhook);
    setWebhookModalOpen(true);
  };

  const slackIntegration = integrations?.find(i => i.type === 'slack');
  const teamsIntegration = integrations?.find(i => i.type === 'teams');

  const webhookEvents = [
    { value: 'analysis.completed', label: 'Analysis Completed' },
    { value: 'analysis.failed', label: 'Analysis Failed' },
    { value: 'issue.created', label: 'Issue Created' },
    { value: 'issue.resolved', label: 'Issue Resolved' },
    { value: 'project.created', label: 'Project Created' },
    { value: 'project.archived', label: 'Project Archived' },
  ];

  return (
    <div className="settings-section">
      <Title level={4}><LinkOutlined /> {t('settings.integrations', 'Integrations')}</Title>

      {/* Slack */}
      <Card className="integration-card">
        <div className="integration-header">
          <Space>
            <Avatar icon={<SlackOutlined />} style={{ backgroundColor: '#4a154b' }} />
            <div>
              <Text strong>Slack</Text>
              <br />
              <Text type="secondary">{t('settings.slack_desc', 'Receive notifications in Slack')}</Text>
            </div>
          </Space>
          {slackIntegration?.connected ? (
            <Space>
              <Badge status="success" text={slackIntegration.name} />
              <Button danger onClick={() => disconnectSlack.mutate()}>
                {t('settings.disconnect', 'Disconnect')}
              </Button>
            </Space>
          ) : (
            <Button type="primary" style={{ backgroundColor: '#4a154b' }}>
              {t('settings.connect', 'Connect Slack')}
            </Button>
          )}
        </div>
      </Card>

      {/* Teams */}
      <Card className="integration-card">
        <div className="integration-header">
          <Space>
            <Avatar icon={<WindowsOutlined />} style={{ backgroundColor: '#5059c9' }} />
            <div>
              <Text strong>Microsoft Teams</Text>
              <br />
              <Text type="secondary">{t('settings.teams_desc', 'Receive notifications in Teams')}</Text>
            </div>
          </Space>
          {teamsIntegration?.connected ? (
            <Space>
              <Badge status="success" text={teamsIntegration.name} />
              <Button danger onClick={() => disconnectTeams.mutate()}>
                {t('settings.disconnect', 'Disconnect')}
              </Button>
            </Space>
          ) : (
            <Button type="primary" style={{ backgroundColor: '#5059c9' }}>
              {t('settings.connect', 'Connect Teams')}
            </Button>
          )}
        </div>
      </Card>

      <Divider />

      {/* Webhooks */}
      <div className="section-header">
        <div>
          <Title level={4}><ApiOutlined /> {t('settings.webhooks', 'Webhooks')}</Title>
          <Text type="secondary">{t('settings.webhooks_desc', 'Send events to external URLs')}</Text>
        </div>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setWebhookModalOpen(true)}>
          {t('settings.add_webhook', 'Add Webhook')}
        </Button>
      </div>

      <List
        dataSource={webhooks || []}
        locale={{ emptyText: <Empty description={t('settings.no_webhooks', 'No webhooks configured')} /> }}
        renderItem={(webhook) => (
          <List.Item
            actions={[
              <Button 
                type="text" 
                icon={<ThunderboltOutlined />} 
                onClick={() => testWebhook.mutate(webhook.id)}
              >
                Test
              </Button>,
              <Button type="text" icon={<EditOutlined />} onClick={() => handleEditWebhook(webhook)}>
                Edit
              </Button>,
              <Popconfirm
                title={t('settings.delete_webhook_confirm', 'Delete this webhook?')}
                onConfirm={() => deleteWebhook.mutate(webhook.id)}
              >
                <Button type="text" danger icon={<DeleteOutlined />} />
              </Popconfirm>,
            ]}
          >
            <List.Item.Meta
              title={
                <Space>
                  <Text strong>{webhook.name}</Text>
                  <Badge status={webhook.isActive ? 'success' : 'default'} />
                </Space>
              }
              description={
                <Space direction="vertical">
                  <Text code style={{ fontSize: 12 }}>{webhook.url}</Text>
                  <Space>
                    {webhook.events.slice(0, 2).map(e => <Tag key={e}>{e}</Tag>)}
                    {webhook.events.length > 2 && <Tag>+{webhook.events.length - 2}</Tag>}
                  </Space>
                </Space>
              }
            />
          </List.Item>
        )}
      />

      {/* Webhook Modal */}
      <Modal
        title={editingWebhook ? t('settings.edit_webhook', 'Edit Webhook') : t('settings.add_webhook', 'Add Webhook')}
        open={webhookModalOpen}
        onCancel={() => {
          setWebhookModalOpen(false);
          setEditingWebhook(null);
          webhookForm.resetFields();
        }}
        footer={null}
      >
        <Form form={webhookForm} layout="vertical" onFinish={handleWebhookSubmit}>
          <Form.Item name="name" label={t('settings.webhook_name', 'Name')} rules={[{ required: true }]}>
            <Input placeholder="e.g., Discord Notifications" />
          </Form.Item>
          <Form.Item 
            name="url" 
            label={t('settings.webhook_url', 'URL')} 
            rules={[
              { required: true },
              { type: 'url', message: t('common.invalid_url', 'Invalid URL') },
            ]}
          >
            <Input placeholder="https://example.com/webhook" />
          </Form.Item>
          <Form.Item name="events" label={t('settings.webhook_events', 'Events')} rules={[{ required: true }]}>
            <Select mode="multiple" options={webhookEvents} placeholder="Select events" />
          </Form.Item>
          <Form.Item name="secret" label={t('settings.webhook_secret', 'Secret (optional)')}>
            <Input.Password placeholder="Optional signing secret" />
          </Form.Item>
          {editingWebhook && (
            <Form.Item name="isActive" label={t('settings.webhook_active', 'Active')} valuePropName="checked">
              <Switch />
            </Form.Item>
          )}
          <Form.Item>
            <Space>
              <Button onClick={() => setWebhookModalOpen(false)}>{t('common.cancel', 'Cancel')}</Button>
              <Button type="primary" htmlType="submit" loading={createWebhook.isPending || updateWebhook.isPending}>
                {editingWebhook ? t('common.save', 'Save') : t('common.create', 'Create')}
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

// ============================================
// Notifications Section
// ============================================
const NotificationsSection: React.FC = () => {
  const { t } = useTranslation();
  const { settings } = useAuthStore();
  const updateNotifications = useUpdateNotifications();

  const handleChange = (field: string, value: any) => {
    updateNotifications.mutate({ [field]: value });
  };

  const emailNotifications = [
    { key: 'emailAnalysisComplete', label: t('settings.notif_analysis_complete', 'Analysis Complete'), desc: t('settings.notif_analysis_complete_desc', 'When code analysis finishes') },
    { key: 'emailNewIssues', label: t('settings.notif_new_issues', 'New Issues Found'), desc: t('settings.notif_new_issues_desc', 'When new issues are discovered') },
    { key: 'emailWeeklyDigest', label: t('settings.notif_weekly_digest', 'Weekly Digest'), desc: t('settings.notif_weekly_digest_desc', 'Weekly summary of your projects') },
    { key: 'emailSecurityAlerts', label: t('settings.notif_security', 'Security Alerts'), desc: t('settings.notif_security_desc', 'Important security notifications') },
    { key: 'emailProductUpdates', label: t('settings.notif_product', 'Product Updates'), desc: t('settings.notif_product_desc', 'New features and updates') },
  ];

  return (
    <div className="settings-section">
      <Title level={4}><MailOutlined /> {t('settings.email_notifications', 'Email Notifications')}</Title>
      <Space direction="vertical" style={{ width: '100%' }} className="notification-list">
        {emailNotifications.map((notif) => (
          <div key={notif.key} className="notification-item">
            <div>
              <Text strong>{notif.label}</Text>
              <br />
              <Text type="secondary">{notif.desc}</Text>
            </div>
            <Switch
              checked={settings?.notifications?.[notif.key as keyof typeof settings.notifications] as boolean}
              onChange={(v) => handleChange(notif.key, v)}
            />
          </div>
        ))}
      </Space>

      <Divider />

      <Title level={4}><ClockCircleOutlined /> {t('settings.digest_frequency', 'Digest Frequency')}</Title>
      <Select
        value={settings?.notifications?.digestFrequency || 'immediate'}
        onChange={(v) => handleChange('digestFrequency', v)}
        style={{ width: 200 }}
        options={[
          { value: 'immediate', label: t('settings.freq_immediate', 'Immediate') },
          { value: 'daily', label: t('settings.freq_daily', 'Daily Digest') },
          { value: 'weekly', label: t('settings.freq_weekly', 'Weekly Digest') },
          { value: 'none', label: t('settings.freq_none', 'None') },
        ]}
      />

      <Divider />

      <Title level={4}><BellOutlined /> {t('settings.in_app_notifications', 'In-App Notifications')}</Title>
      <Space direction="vertical">
        <div className="toggle-item">
          <Switch
            checked={settings?.notifications?.inAppNotifications ?? true}
            onChange={(v) => handleChange('inAppNotifications', v)}
          />
          <Text>{t('settings.in_app', 'Show in-app notifications')}</Text>
        </div>
        <div className="toggle-item">
          <Switch
            checked={settings?.notifications?.desktopNotifications ?? true}
            onChange={(v) => handleChange('desktopNotifications', v)}
          />
          <Text>{t('settings.desktop_notifications', 'Desktop notifications')}</Text>
        </div>
      </Space>

      <Divider />

      <Title level={4}><MoonOutlined /> {t('settings.dnd', 'Do Not Disturb')}</Title>
      <div className="toggle-item" style={{ marginBottom: 16 }}>
        <Switch
          checked={settings?.notifications?.dndEnabled ?? false}
          onChange={(v) => handleChange('dndEnabled', v)}
        />
        <Text>{t('settings.enable_dnd', 'Enable Do Not Disturb')}</Text>
      </div>
      {settings?.notifications?.dndEnabled && (
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item label={t('settings.dnd_start', 'Start Time')}>
              <TimePicker format="HH:mm" />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item label={t('settings.dnd_end', 'End Time')}>
              <TimePicker format="HH:mm" />
            </Form.Item>
          </Col>
        </Row>
      )}
    </div>
  );
};

// ============================================
// Main Settings Component
// ============================================
export const Settings: React.FC = () => {
  const { t } = useTranslation();
  const { isLoading } = useUserSettings();

  const tabItems: TabsProps['items'] = [
    {
      key: 'preferences',
      label: <><SettingOutlined /> {t('settings.preferences', 'Preferences')}</>,
      children: <PreferencesSection />,
    },
    {
      key: 'security',
      label: <><SafetyOutlined /> {t('settings.security', 'Security')}</>,
      children: <SecuritySection />,
    },
    {
      key: 'api-keys',
      label: <><KeyOutlined /> {t('settings.api_keys', 'API Keys')}</>,
      children: <ApiKeysSection />,
    },
    {
      key: 'integrations',
      label: <><LinkOutlined /> {t('settings.integrations', 'Integrations')}</>,
      children: <IntegrationsSection />,
    },
    {
      key: 'notifications',
      label: <><BellOutlined /> {t('settings.notifications', 'Notifications')}</>,
      children: <NotificationsSection />,
    },
  ];

  if (isLoading) {
    return (
      <div className="settings-container">
        <Skeleton active paragraph={{ rows: 10 }} />
      </div>
    );
  }

  return (
    <div className="settings-container" role="main" aria-label={t('settings.title', 'Settings')}>
      <Title level={2}>{t('settings.title', 'Settings')}</Title>
      <Card className="settings-card">
        <Tabs
          defaultActiveKey="preferences"
          items={tabItems}
          tabPosition="left"
          className="settings-tabs"
        />
      </Card>
    </div>
  );
};

export default Settings;
