import React, { useState } from 'react';
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
  Modal
} from 'antd';
import {
  LockOutlined,
  BellOutlined,
  GlobalOutlined,
  SafetyOutlined,
  ApiOutlined,
  DeleteOutlined,
  PlusOutlined,
  KeyOutlined
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useUIStore } from '../store/uiStore';
import { useAuthStore } from '../store/authStore';
import { apiService } from '../services/api';
import './Settings.css';

const { Title, Text, Paragraph } = Typography;

interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used?: string;
}

export const Settings: React.FC = () => {
  const { t, i18n } = useTranslation();
  const { theme, setTheme, language, setLanguage } = useUIStore();
  const { user } = useAuthStore();
  const [passwordForm] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([
    {
      id: '1',
      name: 'Development Key',
      prefix: 'sk-dev-****',
      created_at: new Date().toISOString(),
      last_used: new Date().toISOString()
    }
  ]);
  const [isApiKeyModalOpen, setIsApiKeyModalOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [newApiKey, setNewApiKey] = useState<string | null>(null);

  // Handle password change
  const handlePasswordChange = async (values: any) => {
    setLoading(true);
    try {
      await apiService.auth.changePassword(values.currentPassword, values.newPassword);
      message.success(t('settings.password_changed', 'Password changed successfully'));
      passwordForm.resetFields();
    } catch (error: any) {
      message.error(error.response?.data?.detail || t('settings.password_error', 'Failed to change password'));
    } finally {
      setLoading(false);
    }
  };

  // Handle language change
  const handleLanguageChange = (newLanguage: string) => {
    setLanguage(newLanguage as any);
    i18n.changeLanguage(newLanguage);
    message.success(t('settings.language_changed', 'Language updated'));
  };

  // Handle create API key
  const handleCreateApiKey = async () => {
    if (!newKeyName.trim()) {
      message.error(t('settings.api_key_name_required', 'Please enter a name for the API key'));
      return;
    }

    try {
      // In production, this would call the API
      const mockKey = `sk-${newKeyName.toLowerCase()}-${Math.random().toString(36).substring(2, 15)}`;
      setNewApiKey(mockKey);
      
      setApiKeys([...apiKeys, {
        id: Date.now().toString(),
        name: newKeyName,
        prefix: `sk-${newKeyName.toLowerCase().substring(0, 3)}-****`,
        created_at: new Date().toISOString()
      }]);
      
      setNewKeyName('');
    } catch (error) {
      message.error(t('settings.api_key_error', 'Failed to create API key'));
    }
  };

  // Handle delete API key
  const handleDeleteApiKey = (keyId: string) => {
    Modal.confirm({
      title: t('settings.delete_api_key', 'Delete API Key?'),
      content: t('settings.delete_api_key_warning', 'This action cannot be undone. Applications using this key will lose access.'),
      okText: t('common.delete', 'Delete'),
      okType: 'danger',
      onOk: () => {
        setApiKeys(apiKeys.filter(k => k.id !== keyId));
        message.success(t('settings.api_key_deleted', 'API key deleted'));
      }
    });
  };

  const tabItems = [
    {
      key: 'general',
      label: (
        <span>
          <GlobalOutlined />
          {t('settings.general', 'General')}
        </span>
      ),
      children: (
        <Card>
          <Title level={4}>{t('settings.appearance', 'Appearance')}</Title>
          <Form layout="vertical">
            <Form.Item label={t('settings.theme', 'Theme')}>
              <Select
                value={theme}
                onChange={setTheme}
                style={{ width: 200 }}
                options={[
                  { value: 'light', label: t('settings.theme_light', 'Light') },
                  { value: 'dark', label: t('settings.theme_dark', 'Dark') },
                  { value: 'system', label: t('settings.theme_system', 'System') }
                ]}
              />
            </Form.Item>
            <Form.Item label={t('settings.language', 'Language')}>
              <Select
                value={language}
                onChange={handleLanguageChange}
                style={{ width: 200 }}
                options={[
                  { value: 'en', label: 'English' },
                  { value: 'zh-CN', label: '简体中文' },
                  { value: 'zh-TW', label: '繁體中文' }
                ]}
              />
            </Form.Item>
          </Form>

          <Divider />

          <Title level={4}>{t('settings.editor', 'Editor Preferences')}</Title>
          <Form layout="vertical">
            <Form.Item label={t('settings.font_size', 'Font Size')}>
              <Select
                defaultValue={14}
                style={{ width: 200 }}
                options={[
                  { value: 12, label: '12px' },
                  { value: 14, label: '14px (Default)' },
                  { value: 16, label: '16px' },
                  { value: 18, label: '18px' }
                ]}
              />
            </Form.Item>
            <Form.Item label={t('settings.tab_size', 'Tab Size')}>
              <Select
                defaultValue={2}
                style={{ width: 200 }}
                options={[
                  { value: 2, label: '2 spaces' },
                  { value: 4, label: '4 spaces' }
                ]}
              />
            </Form.Item>
            <Form.Item>
              <Space direction="vertical">
                <div>
                  <Switch defaultChecked /> {t('settings.line_numbers', 'Show line numbers')}
                </div>
                <div>
                  <Switch defaultChecked /> {t('settings.minimap', 'Show minimap')}
                </div>
                <div>
                  <Switch defaultChecked /> {t('settings.word_wrap', 'Word wrap')}
                </div>
              </Space>
            </Form.Item>
          </Form>
        </Card>
      )
    },
    {
      key: 'notifications',
      label: (
        <span>
          <BellOutlined />
          {t('settings.notifications', 'Notifications')}
        </span>
      ),
      children: (
        <Card>
          <Title level={4}>{t('settings.email_notifications', 'Email Notifications')}</Title>
          <Form layout="vertical">
            <Form.Item>
              <Space direction="vertical" style={{ width: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <Text strong>{t('settings.analysis_complete', 'Analysis Complete')}</Text>
                    <br />
                    <Text type="secondary">{t('settings.analysis_complete_desc', 'Get notified when code analysis finishes')}</Text>
                  </div>
                  <Switch defaultChecked />
                </div>
                <Divider style={{ margin: '12px 0' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <Text strong>{t('settings.new_issues', 'New Issues Found')}</Text>
                    <br />
                    <Text type="secondary">{t('settings.new_issues_desc', 'Get notified when new issues are discovered')}</Text>
                  </div>
                  <Switch defaultChecked />
                </div>
                <Divider style={{ margin: '12px 0' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <Text strong>{t('settings.weekly_digest', 'Weekly Digest')}</Text>
                    <br />
                    <Text type="secondary">{t('settings.weekly_digest_desc', 'Receive a weekly summary of your projects')}</Text>
                  </div>
                  <Switch />
                </div>
              </Space>
            </Form.Item>
          </Form>

          <Divider />

          <Title level={4}>{t('settings.in_app_notifications', 'In-App Notifications')}</Title>
          <Form layout="vertical">
            <Form.Item>
              <Space direction="vertical">
                <div>
                  <Switch defaultChecked /> {t('settings.desktop_notifications', 'Desktop notifications')}
                </div>
                <div>
                  <Switch defaultChecked /> {t('settings.sound_notifications', 'Sound notifications')}
                </div>
              </Space>
            </Form.Item>
          </Form>
        </Card>
      )
    },
    {
      key: 'security',
      label: (
        <span>
          <SafetyOutlined />
          {t('settings.security', 'Security')}
        </span>
      ),
      children: (
        <Card>
          <Title level={4}>{t('settings.change_password', 'Change Password')}</Title>
          <Form
            form={passwordForm}
            layout="vertical"
            onFinish={handlePasswordChange}
            style={{ maxWidth: 400 }}
          >
            <Form.Item
              name="currentPassword"
              label={t('settings.current_password', 'Current Password')}
              rules={[{ required: true, message: t('settings.current_password_required', 'Please enter current password') }]}
            >
              <Input.Password prefix={<LockOutlined />} />
            </Form.Item>
            <Form.Item
              name="newPassword"
              label={t('settings.new_password', 'New Password')}
              rules={[
                { required: true, message: t('settings.new_password_required', 'Please enter new password') },
                { min: 8, message: t('settings.password_min_length', 'Password must be at least 8 characters') }
              ]}
            >
              <Input.Password prefix={<LockOutlined />} />
            </Form.Item>
            <Form.Item
              name="confirmPassword"
              label={t('settings.confirm_password', 'Confirm Password')}
              dependencies={['newPassword']}
              rules={[
                { required: true, message: t('settings.confirm_password_required', 'Please confirm password') },
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    if (!value || getFieldValue('newPassword') === value) {
                      return Promise.resolve();
                    }
                    return Promise.reject(new Error(t('settings.passwords_not_match', 'Passwords do not match')));
                  }
                })
              ]}
            >
              <Input.Password prefix={<LockOutlined />} />
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit" loading={loading}>
                {t('settings.update_password', 'Update Password')}
              </Button>
            </Form.Item>
          </Form>

          <Divider />

          <Title level={4}>{t('settings.two_factor', 'Two-Factor Authentication')}</Title>
          <Alert
            message={t('settings.two_factor_disabled', 'Two-factor authentication is not enabled')}
            description={t('settings.two_factor_desc', 'Add an extra layer of security to your account')}
            type="info"
            showIcon
            action={
              <Button size="small" type="primary">
                {t('settings.enable_2fa', 'Enable 2FA')}
              </Button>
            }
          />

          <Divider />

          <Title level={4}>{t('settings.sessions', 'Active Sessions')}</Title>
          <List
            dataSource={[
              { device: 'Chrome on Windows', location: 'Vietnam', current: true, lastActive: 'Now' },
              { device: 'Safari on iPhone', location: 'Vietnam', current: false, lastActive: '2 hours ago' }
            ]}
            renderItem={(item: any) => (
              <List.Item
                actions={item.current ? [<Tag key="current" color="green">Current</Tag>] : [
                  <Button key="revoke" type="link" danger>Revoke</Button>
                ]}
              >
                <List.Item.Meta
                  title={item.device}
                  description={`${item.location} • ${item.lastActive}`}
                />
              </List.Item>
            )}
          />
        </Card>
      )
    },
    {
      key: 'api',
      label: (
        <span>
          <ApiOutlined />
          {t('settings.api_keys', 'API Keys')}
        </span>
      ),
      children: (
        <Card>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <div>
              <Title level={4} style={{ margin: 0 }}>{t('settings.api_keys', 'API Keys')}</Title>
              <Text type="secondary">{t('settings.api_keys_desc', 'Manage API keys for programmatic access')}</Text>
            </div>
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setIsApiKeyModalOpen(true)}>
              {t('settings.create_api_key', 'Create API Key')}
            </Button>
          </div>

          <List
            dataSource={apiKeys}
            locale={{ emptyText: t('settings.no_api_keys', 'No API keys') }}
            renderItem={(item) => (
              <List.Item
                actions={[
                  <Button
                    key="delete"
                    type="text"
                    danger
                    icon={<DeleteOutlined />}
                    onClick={() => handleDeleteApiKey(item.id)}
                  />
                ]}
              >
                <List.Item.Meta
                  avatar={<KeyOutlined style={{ fontSize: 24 }} />}
                  title={item.name}
                  description={
                    <Space>
                      <Text code>{item.prefix}</Text>
                      <Text type="secondary">•</Text>
                      <Text type="secondary">Created {new Date(item.created_at).toLocaleDateString()}</Text>
                      {item.last_used && (
                        <>
                          <Text type="secondary">•</Text>
                          <Text type="secondary">Last used {new Date(item.last_used).toLocaleDateString()}</Text>
                        </>
                      )}
                    </Space>
                  }
                />
              </List.Item>
            )}
          />

          <Modal
            title={t('settings.create_api_key', 'Create API Key')}
            open={isApiKeyModalOpen}
            onCancel={() => {
              setIsApiKeyModalOpen(false);
              setNewKeyName('');
              setNewApiKey(null);
            }}
            footer={newApiKey ? [
              <Button key="close" type="primary" onClick={() => {
                setIsApiKeyModalOpen(false);
                setNewApiKey(null);
              }}>
                Done
              </Button>
            ] : [
              <Button key="cancel" onClick={() => setIsApiKeyModalOpen(false)}>Cancel</Button>,
              <Button key="create" type="primary" onClick={handleCreateApiKey}>Create</Button>
            ]}
          >
            {newApiKey ? (
              <Alert
                type="success"
                message={t('settings.api_key_created', 'API Key Created')}
                description={
                  <>
                    <Paragraph>
                      {t('settings.api_key_warning', 'Make sure to copy your API key now. You won\'t be able to see it again!')}
                    </Paragraph>
                    <Space.Compact style={{ width: '100%' }}>
                      <Input.Password
                        value={newApiKey}
                        readOnly
                        style={{ flex: 1 }}
                      />
                      <Button
                        type="primary"
                        onClick={() => {
                          navigator.clipboard.writeText(newApiKey);
                          message.success('Copied!');
                        }}
                      >
                        Copy
                      </Button>
                    </Space.Compact>
                  </>
                }
              />
            ) : (
              <Form layout="vertical">
                <Form.Item label={t('settings.api_key_name', 'Key Name')}>
                  <Input
                    placeholder="e.g., Production Key"
                    value={newKeyName}
                    onChange={(e) => setNewKeyName(e.target.value)}
                  />
                </Form.Item>
              </Form>
            )}
          </Modal>
        </Card>
      )
    }
  ];

  return (
    <div className="settings-container">
      <Title level={2}>{t('settings.title', 'Settings')}</Title>
      <Tabs
        defaultActiveKey="general"
        items={tabItems}
        tabPosition="left"
        style={{ minHeight: 500 }}
      />
    </div>
  );
};

export default Settings;
