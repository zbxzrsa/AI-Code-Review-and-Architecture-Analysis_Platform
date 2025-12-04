/**
 * Enhanced User Profile Page
 * 
 * Comprehensive profile page with:
 * - Profile information with avatar upload
 * - Account linking (OAuth)
 * - Privacy settings
 * - Activity section (login history, API activity)
 * - Danger zone (account deletion, data export)
 */

import React, { useState, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Avatar,
  Typography,
  Button,
  Form,
  Input,
  Upload,
  Space,
  Divider,
  Statistic,
  Tag,
  message,
  List,
  Badge,
  Tooltip,
  Switch,
  Modal,
  Tabs,
  Alert,
  Skeleton,
  Empty,
  Table,
  Popconfirm,
  Descriptions,
  Progress,
} from 'antd';
import type { UploadProps, TabsProps, TableProps } from 'antd';
import {
  UserOutlined,
  EditOutlined,
  CameraOutlined,
  MailOutlined,
  CalendarOutlined,
  ProjectOutlined,
  BugOutlined,
  CheckCircleOutlined,
  TrophyOutlined,
  FireOutlined,
  GithubOutlined,
  GitlabOutlined,
  GoogleOutlined,
  WindowsOutlined,
  LinkOutlined,
  DisconnectOutlined,
  EyeOutlined,
  LockOutlined,
  HistoryOutlined,
  ApiOutlined,
  DeleteOutlined,
  DownloadOutlined,
  ExclamationCircleOutlined,
  SafetyOutlined,
  GlobalOutlined,
  WarningOutlined,
  ClockCircleOutlined,
  EnvironmentOutlined,
  LaptopOutlined,
  MobileOutlined,
  DesktopOutlined,
  SaveOutlined,
  CloseOutlined,
  InboxOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuthStore, type LoginHistory, type ApiActivity } from '../../store/authStore';
import {
  useUserProfile,
  useUpdateProfile,
  useUploadAvatar,
  useDeleteAvatar,
  useOAuthConnections,
  useConnectOAuth,
  useDisconnectOAuth,
  useUpdatePrivacy,
  useLoginHistory,
  useApiActivity,
  useDownloadPersonalData,
  useRequestAccountDeletion,
} from '../../hooks/useUser';
import './Profile.css';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { Dragger } = Upload;

/** Profile form values */
interface ProfileFormValues {
  name?: string;
  bio?: string;
  company?: string;
  location?: string;
  website?: string;
}

/** Extended login history with API variations */
interface LoginHistoryRecord extends Partial<LoginHistory> {
  device_type?: string;
  user_agent?: string;
  created_at?: string;
  status?: string;
}

/** Extended API activity with API variations */
interface ApiActivityRecord extends Partial<ApiActivity> {
  response_time_ms?: number;
  status_code?: number;
  created_at?: string;
}

/** OAuth provider icons (reserved for future use) */
const _providerIcons: Record<string, React.ReactNode> = {
  github: <GithubOutlined />,
  gitlab: <GitlabOutlined />,
  google: <GoogleOutlined />,
  microsoft: <WindowsOutlined />,
};

/** OAuth provider colors (reserved for future use) */
const _providerColors: Record<string, string> = {
  github: '#24292e',
  gitlab: '#fc6d26',
  google: '#4285f4',
  microsoft: '#00a4ef',
};

/**
 * Profile Information Section
 */
const ProfileInfoSection: React.FC = () => {
  const { t } = useTranslation();
  const { user } = useAuthStore();
  const [isEditing, setIsEditing] = useState(false);
  const [form] = Form.useForm();
  
  const updateProfile = useUpdateProfile();
  const uploadAvatar = useUploadAvatar();
  const deleteAvatar = useDeleteAvatar();

  // Handle profile update
  const handleSave = useCallback(async (values: ProfileFormValues) => {
    await updateProfile.mutateAsync(values);
    setIsEditing(false);
  }, [updateProfile]);

  // Avatar upload props
  const uploadProps: UploadProps = {
    name: 'avatar',
    showUploadList: false,
    accept: 'image/jpeg,image/png,image/webp',
    beforeUpload: (file) => {
      const isValidType = ['image/jpeg', 'image/png', 'image/webp'].includes(file.type);
      if (!isValidType) {
        message.error(t('profile.avatar_type_error', 'Only JPG, PNG, and WebP files are allowed'));
        return false;
      }
      const isLt5M = file.size / 1024 / 1024 < 5;
      if (!isLt5M) {
        message.error(t('profile.avatar_size_error', 'Image must be smaller than 5MB'));
        return false;
      }
      uploadAvatar.mutate(file);
      return false;
    },
  };

  // Dragger props for drag-and-drop
  const draggerProps: UploadProps = {
    ...uploadProps,
    multiple: false,
  };

  return (
    <Card className="profile-info-card">
      <div className="profile-header">
        {/* Avatar Section */}
        <div className="avatar-section">
          <div className="avatar-container">
            <Avatar
              size={120}
              src={user?.avatar}
              icon={!user?.avatar && <UserOutlined />}
              className="profile-avatar"
            />
            <Upload {...uploadProps}>
              <Button
                type="primary"
                shape="circle"
                icon={<CameraOutlined />}
                size="small"
                className="avatar-upload-btn"
                loading={uploadAvatar.isPending}
              />
            </Upload>
          </div>
          {user?.avatar && (
            <Button 
              type="link" 
              danger 
              size="small"
              onClick={() => deleteAvatar.mutate()}
              loading={deleteAvatar.isPending}
            >
              {t('profile.remove_avatar', 'Remove')}
            </Button>
          )}
        </div>

        {/* Profile Info */}
        <div className="profile-info">
          {!isEditing ? (
            <>
              <Title level={3} className="profile-name">{user?.name}</Title>
              {user?.username && <Text type="secondary">@{user.username}</Text>}
              <div className="profile-meta">
                <Tag color="blue">{user?.role}</Tag>
                {user?.emailVerified && (
                  <Tooltip title={t('profile.email_verified', 'Email verified')}>
                    <Tag color="green" icon={<CheckCircleOutlined />}>
                      {t('profile.verified', 'Verified')}
                    </Tag>
                  </Tooltip>
                )}
                {user?.twoFactorEnabled && (
                  <Tooltip title={t('profile.2fa_enabled', '2FA enabled')}>
                    <Tag color="purple" icon={<SafetyOutlined />}>
                      2FA
                    </Tag>
                  </Tooltip>
                )}
              </div>
            </>
          ) : (
            <Form
              form={form}
              layout="vertical"
              initialValues={{
                name: user?.name,
                username: user?.username,
                bio: user?.bio,
              }}
              onFinish={handleSave}
              className="profile-edit-form"
            >
              <Form.Item
                name="name"
                label={t('profile.name', 'Full Name')}
                rules={[{ required: true, message: t('profile.name_required', 'Please enter your name') }]}
              >
                <Input prefix={<UserOutlined />} />
              </Form.Item>
              <Form.Item
                name="username"
                label={t('profile.username', 'Username')}
                rules={[
                  { pattern: /^[a-zA-Z0-9_]+$/, message: t('profile.username_invalid', 'Only letters, numbers, and underscores') },
                ]}
              >
                <Input prefix={<span>@</span>} />
              </Form.Item>
              <Form.Item
                name="bio"
                label={t('profile.bio', 'Bio')}
              >
                <TextArea rows={3} maxLength={200} showCount />
              </Form.Item>
              <Form.Item>
                <Space>
                  <Button type="primary" htmlType="submit" loading={updateProfile.isPending} icon={<SaveOutlined />}>
                    {t('common.save', 'Save')}
                  </Button>
                  <Button onClick={() => setIsEditing(false)} icon={<CloseOutlined />}>
                    {t('common.cancel', 'Cancel')}
                  </Button>
                </Space>
              </Form.Item>
            </Form>
          )}
        </div>
      </div>

      {!isEditing && (
        <>
          {user?.bio && (
            <Paragraph className="profile-bio">{user.bio}</Paragraph>
          )}
          
          <Descriptions column={{ xs: 1, sm: 2 }} className="profile-details">
            <Descriptions.Item label={<><MailOutlined /> {t('profile.email', 'Email')}</>}>
              {user?.email}
            </Descriptions.Item>
            <Descriptions.Item label={<><CalendarOutlined /> {t('profile.joined', 'Joined')}</>}>
              {user?.createdAt ? new Date(user.createdAt).toLocaleDateString() : '-'}
            </Descriptions.Item>
            <Descriptions.Item label={<><ClockCircleOutlined /> {t('profile.last_login', 'Last Login')}</>}>
              {user?.lastLoginAt ? new Date(user.lastLoginAt).toLocaleString() : '-'}
            </Descriptions.Item>
            <Descriptions.Item label={<><SafetyOutlined /> {t('profile.role', 'Role')}</>}>
              <Tag color="blue">{user?.role}</Tag>
            </Descriptions.Item>
          </Descriptions>

          <Button
            type="primary"
            icon={<EditOutlined />}
            onClick={() => {
              form.setFieldsValue({
                name: user?.name,
                username: user?.username,
                bio: user?.bio,
              });
              setIsEditing(true);
            }}
            block
            className="edit-profile-btn"
          >
            {t('profile.edit', 'Edit Profile')}
          </Button>
        </>
      )}

      {/* Drag and Drop Zone */}
      {isEditing && (
        <Dragger {...draggerProps} className="avatar-dragger">
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">{t('profile.drag_avatar', 'Drag avatar here or click to upload')}</p>
          <p className="ant-upload-hint">{t('profile.avatar_hint', 'JPG, PNG, WebP up to 5MB')}</p>
        </Dragger>
      )}
    </Card>
  );
};

/**
 * OAuth Account Linking Section
 */
const AccountLinkingSection: React.FC = () => {
  const { t } = useTranslation();
  const { data: connections, isLoading } = useOAuthConnections();
  const connectOAuth = useConnectOAuth();
  const disconnectOAuth = useDisconnectOAuth();

  const providers = [
    { id: 'github', name: 'GitHub', icon: <GithubOutlined />, color: '#24292e' },
    { id: 'gitlab', name: 'GitLab', icon: <GitlabOutlined />, color: '#fc6d26' },
    { id: 'google', name: 'Google', icon: <GoogleOutlined />, color: '#4285f4' },
    { id: 'microsoft', name: 'Microsoft', icon: <WindowsOutlined />, color: '#00a4ef' },
  ];

  interface OAuthConnection {
    provider: string;
    connected: boolean;
    username?: string;
    email?: string;
    connectedAt?: string;
  }

  const getConnection = (provider: string): OAuthConnection | undefined => {
    if (!connections) return undefined;
    const connectionList: OAuthConnection[] = Array.isArray(connections) 
      ? connections 
      : (connections as unknown as { items?: OAuthConnection[] })?.items || [];
    return connectionList.find((c) => c.provider === provider);
  };

  if (isLoading) {
    return <Card><Skeleton active /></Card>;
  }

  return (
    <Card 
      title={<><LinkOutlined /> {t('profile.account_linking', 'Connected Accounts')}</>}
      className="account-linking-card"
    >
      <List
        dataSource={providers}
        renderItem={(provider) => {
          const connection = getConnection(provider.id);
          const isConnected = connection?.connected;

          return (
            <List.Item
              actions={[
                isConnected ? (
                  <Popconfirm
                    title={t('profile.disconnect_confirm', 'Disconnect this account?')}
                    onConfirm={() => disconnectOAuth.mutate(provider.id)}
                    okText={t('common.yes', 'Yes')}
                    cancelText={t('common.no', 'No')}
                  >
                    <Button 
                      danger 
                      icon={<DisconnectOutlined />}
                      loading={disconnectOAuth.isPending}
                    >
                      {t('profile.disconnect', 'Disconnect')}
                    </Button>
                  </Popconfirm>
                ) : (
                  <Button
                    type="primary"
                    icon={<LinkOutlined />}
                    onClick={() => connectOAuth.mutate(provider.id)}
                    loading={connectOAuth.isPending}
                    style={{ backgroundColor: provider.color, borderColor: provider.color }}
                  >
                    {t('profile.connect', 'Connect')}
                  </Button>
                ),
              ]}
            >
              <List.Item.Meta
                avatar={
                  <Avatar 
                    icon={provider.icon} 
                    style={{ backgroundColor: provider.color }}
                  />
                }
                title={provider.name}
                description={
                  isConnected ? (
                    <Space>
                      <Badge status="success" text={connection.username || connection.email} />
                      {connection.connectedAt && (
                        <Text type="secondary">
                          {t('profile.connected_on', 'Connected on')} {new Date(connection.connectedAt).toLocaleDateString()}
                        </Text>
                      )}
                    </Space>
                  ) : (
                    <Text type="secondary">{t('profile.not_connected', 'Not connected')}</Text>
                  )
                }
              />
            </List.Item>
          );
        }}
      />
    </Card>
  );
};

/**
 * Privacy Settings Section
 */
const PrivacySection: React.FC = () => {
  const { t } = useTranslation();
  const { settings } = useAuthStore();
  const updatePrivacy = useUpdatePrivacy();

  const handleChange = useCallback((field: string, value: boolean | string) => {
    updatePrivacy.mutate({ [field]: value });
  }, [updatePrivacy]);

  const privacyOptions = [
    { key: 'showEmail', label: t('profile.privacy.show_email', 'Show email publicly'), icon: <MailOutlined /> },
    { key: 'showActivity', label: t('profile.privacy.show_activity', 'Show activity feed'), icon: <HistoryOutlined /> },
    { key: 'showProjects', label: t('profile.privacy.show_projects', 'Show projects'), icon: <ProjectOutlined /> },
    { key: 'showStatistics', label: t('profile.privacy.show_stats', 'Show statistics'), icon: <TrophyOutlined /> },
    { key: 'allowDataSharing', label: t('profile.privacy.data_sharing', 'Allow anonymous data sharing'), icon: <GlobalOutlined /> },
    { key: 'allowAnalytics', label: t('profile.privacy.analytics', 'Allow usage analytics'), icon: <ApiOutlined /> },
  ];

  return (
    <Card 
      title={<><EyeOutlined /> {t('profile.privacy.title', 'Privacy Settings')}</>}
      className="privacy-card"
    >
      <Form layout="vertical">
        <Form.Item label={t('profile.privacy.visibility', 'Profile Visibility')}>
          <Select
            value={settings?.privacy?.profileVisibility || 'public'}
            onChange={(value) => handleChange('profileVisibility', value)}
            options={[
              { value: 'public', label: t('profile.privacy.public', 'Public') },
              { value: 'private', label: t('profile.privacy.private', 'Private') },
              { value: 'connections', label: t('profile.privacy.connections_only', 'Connections Only') },
            ]}
            style={{ width: 200 }}
          />
        </Form.Item>

        <Divider />

        <Space direction="vertical" style={{ width: '100%' }}>
          {privacyOptions.map((option) => (
            <div key={option.key} className="privacy-option">
              <Space>
                {option.icon}
                <Text>{option.label}</Text>
              </Space>
              <Switch
                checked={settings?.privacy?.[option.key as keyof typeof settings.privacy] as boolean}
                onChange={(checked) => handleChange(option.key, checked)}
                loading={updatePrivacy.isPending}
              />
            </div>
          ))}
        </Space>
      </Form>
    </Card>
  );
};

// Need to import Select
import { Select } from 'antd';

/**
 * Activity Section (Login History & API Activity)
 */
const ActivitySection: React.FC = () => {
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState('logins');
  
  const { data: loginHistory, isLoading: loginLoading } = useLoginHistory({ limit: 10 });
  const { data: apiActivity, isLoading: apiLoading } = useApiActivity({ limit: 10 });

  const getDeviceIcon = (device?: string) => {
    if (!device) return <DesktopOutlined />;
    const lower = device.toLowerCase();
    if (lower.includes('mobile') || lower.includes('phone')) return <MobileOutlined />;
    if (lower.includes('tablet')) return <LaptopOutlined />;
    return <DesktopOutlined />;
  };

  const loginColumns: TableProps<LoginHistoryRecord>['columns'] = [
    {
      title: t('profile.activity.device', 'Device'),
      dataIndex: 'device',
      key: 'device',
      render: (device: string | undefined, record: LoginHistoryRecord) => {
        // Handle both 'device' and 'device_type' from API
        const deviceInfo = device || record.device_type || 'desktop';
        const browserInfo = record.browser || record.user_agent?.split(' ')[0] || 'Unknown';
        return (
          <Space>
            {getDeviceIcon(deviceInfo)}
            <div>
              <Text strong>{browserInfo}</Text>
              <br />
              <Text type="secondary" style={{ fontSize: 12 }}>{deviceInfo}</Text>
            </div>
          </Space>
        );
      },
    },
    {
      title: t('profile.activity.location', 'Location'),
      key: 'location',
      render: (_: unknown, record: LoginHistoryRecord) => (
        <Space>
          <EnvironmentOutlined />
          <div>
            <Text>{record.location || 'Unknown'}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>{record.ip || (record as any).ip_address || 'Unknown'}</Text>
          </div>
        </Space>
      ),
    },
    {
      title: t('profile.activity.time', 'Time'),
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (time: string | undefined, record: LoginHistoryRecord) => {
        const timestamp = time || record.created_at;
        return timestamp ? new Date(timestamp).toLocaleString() : 'Unknown';
      },
    },
    {
      title: t('profile.activity.status', 'Status'),
      dataIndex: 'success',
      key: 'success',
      render: (success: boolean | undefined, record: LoginHistoryRecord) => {
        // Handle both 'success' boolean and 'status' string from API
        const isSuccess = success === true || record.status === 'success';
        return isSuccess ? (
          <Tag color="green">{t('profile.activity.success', 'Success')}</Tag>
        ) : (
          <Tooltip title={record.failureReason || (record as any).failure_reason}>
            <Tag color="red">{t('profile.activity.failed', 'Failed')}</Tag>
          </Tooltip>
        );
      },
    },
  ];

  const apiColumns: TableProps<ApiActivityRecord>['columns'] = [
    {
      title: t('profile.activity.endpoint', 'Endpoint'),
      key: 'endpoint',
      render: (_, record) => (
        <Space>
          <Tag color={record.method === 'GET' ? 'blue' : record.method === 'POST' ? 'green' : 'orange'}>
            {record.method}
          </Tag>
          <Text code>{record.endpoint}</Text>
        </Space>
      ),
    },
    {
      title: t('profile.activity.api_key', 'API Key'),
      dataIndex: 'apiKeyName',
      key: 'apiKeyName',
      render: (name: string) => name || '-',
    },
    {
      title: t('profile.activity.response_time', 'Response'),
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: (time: number | undefined, record: ApiActivityRecord) => {
        const responseTime = time || record.response_time_ms || 0;
        return `${responseTime}ms`;
      },
    },
    {
      title: t('profile.activity.status', 'Status'),
      dataIndex: 'statusCode',
      key: 'statusCode',
      render: (code: number | undefined, record: ApiActivityRecord) => {
        const statusCode = code || record.status_code || 200;
        return (
          <Tag color={statusCode < 300 ? 'green' : statusCode < 400 ? 'orange' : 'red'}>
            {statusCode}
          </Tag>
        );
      },
    },
    {
      title: t('profile.activity.time', 'Time'),
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (time: string | undefined, record: ApiActivityRecord) => {
        const timestamp = time || record.created_at;
        return timestamp ? new Date(timestamp).toLocaleString() : 'Unknown';
      },
    },
  ];

  const tabItems: TabsProps['items'] = [
    {
      key: 'logins',
      label: <><HistoryOutlined /> {t('profile.activity.login_history', 'Login History')}</>,
      children: (
        <Table
          columns={loginColumns as any}
          dataSource={loginHistory?.items || []}
          rowKey="id"
          loading={loginLoading}
          pagination={{ pageSize: 5 }}
          size="small"
          locale={{ emptyText: <Empty description={t('profile.activity.no_logins', 'No login history')} /> }}
        />
      ),
    },
    {
      key: 'api',
      label: <><ApiOutlined /> {t('profile.activity.api_activity', 'API Activity')}</>,
      children: (
        <Table
          columns={apiColumns as any}
          dataSource={apiActivity?.items || []}
          rowKey="id"
          loading={apiLoading}
          pagination={{ pageSize: 5 }}
          size="small"
          locale={{ emptyText: <Empty description={t('profile.activity.no_api', 'No API activity')} /> }}
        />
      ),
    },
  ];

  return (
    <Card 
      title={<><HistoryOutlined /> {t('profile.activity.title', 'Activity')}</>}
      className="activity-card"
    >
      <Tabs activeKey={activeTab} onChange={setActiveTab} items={tabItems} />
    </Card>
  );
};

/**
 * Danger Zone Section
 */
const DangerZoneSection: React.FC = () => {
  const { t } = useTranslation();
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [password, setPassword] = useState('');
  const [confirmText, setConfirmText] = useState('');
  const { user: _user } = useAuthStore();
  
  const downloadData = useDownloadPersonalData();
  const requestDeletion = useRequestAccountDeletion();

  const handleDeleteRequest = useCallback(() => {
    if (confirmText !== 'DELETE') {
      message.error(t('profile.danger.confirm_text_error', 'Please type DELETE to confirm'));
      return;
    }
    requestDeletion.mutate(password, {
      onSuccess: () => {
        setDeleteModalOpen(false);
        setPassword('');
        setConfirmText('');
      },
    });
  }, [confirmText, password, requestDeletion, t]);

  return (
    <Card 
      title={<><WarningOutlined /> {t('profile.danger.title', 'Danger Zone')}</>}
      className="danger-zone-card"
    >
      <Alert
        message={t('profile.danger.warning', 'Caution')}
        description={t('profile.danger.warning_desc', 'Actions in this section are irreversible. Please proceed carefully.')}
        type="warning"
        showIcon
        style={{ marginBottom: 24 }}
      />

      {/* Download Data */}
      <div className="danger-action">
        <div>
          <Text strong>{t('profile.danger.download_data', 'Download Your Data')}</Text>
          <br />
          <Text type="secondary">
            {t('profile.danger.download_desc', 'Download all your personal data in JSON format (GDPR compliance)')}
          </Text>
        </div>
        <Button
          icon={<DownloadOutlined />}
          onClick={() => downloadData.mutate()}
          loading={downloadData.isPending}
        >
          {t('profile.danger.download_btn', 'Download')}
        </Button>
      </div>

      <Divider />

      {/* Delete Account */}
      <div className="danger-action">
        <div>
          <Text strong type="danger">{t('profile.danger.delete_account', 'Delete Account')}</Text>
          <br />
          <Text type="secondary">
            {t('profile.danger.delete_desc', 'Permanently delete your account and all associated data')}
          </Text>
        </div>
        <Button
          danger
          type="primary"
          icon={<DeleteOutlined />}
          onClick={() => setDeleteModalOpen(true)}
        >
          {t('profile.danger.delete_btn', 'Delete Account')}
        </Button>
      </div>

      {/* Delete Confirmation Modal */}
      <Modal
        title={
          <Space>
            <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
            {t('profile.danger.delete_confirm_title', 'Confirm Account Deletion')}
          </Space>
        }
        open={deleteModalOpen}
        onCancel={() => {
          setDeleteModalOpen(false);
          setPassword('');
          setConfirmText('');
        }}
        footer={[
          <Button key="cancel" onClick={() => setDeleteModalOpen(false)}>
            {t('common.cancel', 'Cancel')}
          </Button>,
          <Button
            key="delete"
            type="primary"
            danger
            loading={requestDeletion.isPending}
            onClick={handleDeleteRequest}
            disabled={confirmText !== 'DELETE' || !password}
          >
            {t('profile.danger.delete_btn', 'Delete Account')}
          </Button>,
        ]}
      >
        <Alert
          message={t('profile.danger.delete_warning_title', 'This action cannot be undone')}
          description={t('profile.danger.delete_warning', 'All your data, projects, and settings will be permanently deleted.')}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
        />

        <Form layout="vertical">
          <Form.Item label={t('profile.danger.enter_password', 'Enter your password to confirm')}>
            <Input.Password
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              prefix={<LockOutlined />}
            />
          </Form.Item>
          <Form.Item label={t('profile.danger.type_delete', 'Type DELETE to confirm')}>
            <Input
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              placeholder="DELETE"
            />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
};

/**
 * Stats Section
 */
const StatsSection: React.FC = () => {
  const { t } = useTranslation();
  
  // Mock stats - would come from API
  const stats = {
    projects: 12,
    analyses: 156,
    issues_found: 847,
    issues_fixed: 723,
    streak: 15,
  };

  const fixRate = Math.round((stats.issues_fixed / stats.issues_found) * 100);

  return (
    <Card className="stats-card">
      <Row gutter={[16, 16]}>
        <Col xs={12} sm={8}>
          <Statistic
            title={t('profile.projects', 'Projects')}
            value={stats.projects}
            prefix={<ProjectOutlined />}
          />
        </Col>
        <Col xs={12} sm={8}>
          <Statistic
            title={t('profile.analyses', 'Analyses')}
            value={stats.analyses}
            prefix={<BugOutlined />}
          />
        </Col>
        <Col xs={12} sm={8}>
          <Statistic
            title={t('profile.streak', 'Day Streak')}
            value={stats.streak}
            prefix={<FireOutlined />}
            valueStyle={{ color: '#fa8c16' }}
          />
        </Col>
        <Col xs={12} sm={8}>
          <Statistic
            title={t('profile.issues_found', 'Issues Found')}
            value={stats.issues_found}
            valueStyle={{ color: '#cf1322' }}
          />
        </Col>
        <Col xs={12} sm={8}>
          <Statistic
            title={t('profile.issues_fixed', 'Issues Fixed')}
            value={stats.issues_fixed}
            valueStyle={{ color: '#3f8600' }}
          />
        </Col>
        <Col xs={12} sm={8}>
          <div>
            <Text type="secondary">{t('profile.fix_rate', 'Fix Rate')}</Text>
            <Progress percent={fixRate} status="active" strokeColor="#1890ff" />
          </div>
        </Col>
      </Row>
    </Card>
  );
};

/**
 * Main Profile Page Component
 */
export const Profile: React.FC = () => {
  const { t } = useTranslation();
  const { isLoading } = useUserProfile();

  if (isLoading) {
    return (
      <div className="profile-container">
        <Skeleton active avatar paragraph={{ rows: 8 }} />
      </div>
    );
  }

  return (
    <div className="profile-container" role="main" aria-label={t('profile.title', 'User Profile')}>
      <Row gutter={[24, 24]}>
        {/* Left Column */}
        <Col xs={24} lg={8}>
          <ProfileInfoSection />
          <div style={{ marginTop: 24 }}>
            <AccountLinkingSection />
          </div>
        </Col>

        {/* Right Column */}
        <Col xs={24} lg={16}>
          <StatsSection />
          
          <div style={{ marginTop: 24 }}>
            <PrivacySection />
          </div>
          
          <div style={{ marginTop: 24 }}>
            <ActivitySection />
          </div>
          
          <div style={{ marginTop: 24 }}>
            <DangerZoneSection />
          </div>
        </Col>
      </Row>
    </div>
  );
};

export default Profile;
