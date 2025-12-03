/**
 * Project Settings Page
 * 
 * Comprehensive settings page for managing project configuration including:
 * - Project information
 * - Analysis settings
 * - Team members
 * - Webhooks
 * - API keys
 * - Notification preferences
 * - Danger zone (archive/delete)
 */

import React, { useState, useCallback, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Card,
  Tabs,
  Form,
  Input,
  Select,
  Button,
  Space,
  Typography,
  Row,
  Col,
  Switch,
  Table,
  Tag,
  Modal,
  Popconfirm,
  Avatar,
  Skeleton,
  Alert,
  Tooltip,
  Divider,
  Timeline,
  Badge,
  message,
  Descriptions,
  Empty,
  Collapse,
} from 'antd';
import type { TabsProps, TableProps } from 'antd';
import {
  ArrowLeftOutlined,
  SaveOutlined,
  UserOutlined,
  SettingOutlined,
  LinkOutlined,
  KeyOutlined,
  DeleteOutlined,
  PlusOutlined,
  EditOutlined,
  CopyOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  HistoryOutlined,
  WarningOutlined,
  TeamOutlined,
  ApiOutlined,
  InboxOutlined,
  SendOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useProjectStore, type TeamMember, type Webhook, type APIKey, type ActivityLog } from '../../store/projectStore';
import {
  useProject,
  useProjectActivity,
  useProjectTeam,
  useProjectWebhooks,
  useProjectApiKeys,
  useUpdateProject,
  useDeleteProject,
  useArchiveProject,
  useInviteTeamMember,
  useUpdateMemberRole,
  useRemoveTeamMember,
  useCreateWebhook,
  useUpdateWebhook,
  useDeleteWebhook,
  useTestWebhook,
  useCreateApiKey,
  useRevokeApiKey,
} from '../../hooks/useProjects';
import './ProjectSettings.css';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { Panel } = Collapse;

/** Role options */
const roleOptions = [
  { value: 'owner', label: 'Owner', color: 'gold' },
  { value: 'admin', label: 'Admin', color: 'purple' },
  { value: 'member', label: 'Member', color: 'blue' },
  { value: 'viewer', label: 'Viewer', color: 'default' },
];

/** Webhook event options */
const webhookEventOptions = [
  { value: 'analysis.started', label: 'Analysis Started' },
  { value: 'analysis.completed', label: 'Analysis Completed' },
  { value: 'analysis.failed', label: 'Analysis Failed' },
  { value: 'issue.created', label: 'Issue Created' },
  { value: 'issue.resolved', label: 'Issue Resolved' },
];

/** API permission options */
const apiPermissionOptions = [
  { value: 'read:project', label: 'Read Project' },
  { value: 'write:project', label: 'Write Project' },
  { value: 'read:analysis', label: 'Read Analysis' },
  { value: 'write:analysis', label: 'Trigger Analysis' },
  { value: 'read:issues', label: 'Read Issues' },
  { value: 'write:issues', label: 'Manage Issues' },
];

/**
 * Project Information Section
 */
const ProjectInfoSection: React.FC<{
  project: any;
  onSave: (values: any) => void;
  isSaving: boolean;
}> = ({ project, onSave, isSaving }) => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const { setUnsavedChanges } = useProjectStore();

  useEffect(() => {
    if (project) {
      form.setFieldsValue({
        name: project.name,
        description: project.description,
        repository_url: project.repository_url,
      });
    }
  }, [project, form]);

  return (
    <Form
      form={form}
      layout="vertical"
      onFinish={onSave}
      onValuesChange={() => setUnsavedChanges(true)}
    >
      <Row gutter={16}>
        <Col xs={24} md={12}>
          <Form.Item
            name="name"
            label={t('projects.form.name', 'Project Name')}
            rules={[{ required: true, message: t('projects.form.name_required', 'Required') }]}
          >
            <Input prefix={<SettingOutlined />} />
          </Form.Item>
        </Col>
        <Col xs={24} md={12}>
          <Form.Item
            name="repository_url"
            label={t('projects.form.repo', 'Repository URL')}
          >
            <Input prefix={<LinkOutlined />} />
          </Form.Item>
        </Col>
      </Row>

      <Form.Item
        name="description"
        label={t('projects.form.description', 'Description')}
      >
        <TextArea rows={3} maxLength={500} showCount />
      </Form.Item>

      <Descriptions bordered size="small" column={{ xs: 1, sm: 2 }}>
        <Descriptions.Item label={t('projects.settings.owner', 'Owner')}>
          {project?.owner_name || '-'}
        </Descriptions.Item>
        <Descriptions.Item label={t('projects.settings.created', 'Created')}>
          {project?.created_at ? new Date(project.created_at).toLocaleDateString() : '-'}
        </Descriptions.Item>
        <Descriptions.Item label={t('projects.settings.language', 'Language')}>
          <Tag>{project?.language}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label={t('projects.settings.status', 'Status')}>
          <Badge status={project?.status === 'active' ? 'processing' : 'default'} text={project?.status} />
        </Descriptions.Item>
      </Descriptions>

      <Divider />

      <Form.Item>
        <Button type="primary" htmlType="submit" icon={<SaveOutlined />} loading={isSaving}>
          {t('common.save', 'Save Changes')}
        </Button>
      </Form.Item>
    </Form>
  );
};

/**
 * Team Management Section
 */
const TeamSection: React.FC<{ projectId: string }> = ({ projectId }) => {
  const { t } = useTranslation();
  const [inviteModalOpen, setInviteModalOpen] = useState(false);
  const [inviteForm] = Form.useForm();

  const { data: team, isLoading } = useProjectTeam(projectId);
  const inviteMember = useInviteTeamMember(projectId);
  const updateRole = useUpdateMemberRole(projectId);
  const removeMember = useRemoveTeamMember(projectId);

  const handleInvite = useCallback(async (values: { email: string; role: string }) => {
    await inviteMember.mutateAsync(values);
    setInviteModalOpen(false);
    inviteForm.resetFields();
  }, [inviteMember, inviteForm]);

  const columns: TableProps<TeamMember>['columns'] = [
    {
      title: t('projects.team.member', 'Member'),
      key: 'member',
      render: (_, record) => (
        <Space>
          <Avatar src={record.avatar} icon={<UserOutlined />} />
          <div>
            <Text strong>{record.name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>{record.email}</Text>
          </div>
        </Space>
      ),
    },
    {
      title: t('projects.team.role', 'Role'),
      dataIndex: 'role',
      key: 'role',
      render: (role: string, record) => (
        <Select
          value={role}
          onChange={(newRole) => updateRole.mutate({ memberId: record.id, role: newRole })}
          options={roleOptions}
          disabled={role === 'owner'}
          style={{ width: 120 }}
        />
      ),
    },
    {
      title: t('projects.team.joined', 'Joined'),
      dataIndex: 'accepted_at',
      key: 'joined',
      render: (date: string) => date ? new Date(date).toLocaleDateString() : 
        <Tag color="orange">{t('projects.team.pending', 'Pending')}</Tag>,
    },
    {
      title: '',
      key: 'actions',
      width: 60,
      render: (_, record) => (
        record.role !== 'owner' && (
          <Popconfirm
            title={t('projects.team.remove_confirm', 'Remove this member?')}
            onConfirm={() => removeMember.mutate(record.id)}
            okText={t('common.yes', 'Yes')}
            cancelText={t('common.no', 'No')}
          >
            <Button type="text" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        )
      ),
    },
  ];

  return (
    <div>
      <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
        <Col>
          <Text strong>{t('projects.team.title', 'Team Members')}</Text>
        </Col>
        <Col>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setInviteModalOpen(true)}>
            {t('projects.team.invite', 'Invite Member')}
          </Button>
        </Col>
      </Row>

      <Table
        columns={columns}
        dataSource={team || []}
        rowKey="id"
        loading={isLoading}
        pagination={false}
        locale={{ emptyText: <Empty description={t('projects.team.no_members', 'No team members yet')} /> }}
      />

      <Modal
        title={t('projects.team.invite_title', 'Invite Team Member')}
        open={inviteModalOpen}
        onCancel={() => setInviteModalOpen(false)}
        footer={null}
      >
        <Form form={inviteForm} layout="vertical" onFinish={handleInvite}>
          <Form.Item
            name="email"
            label={t('projects.team.email', 'Email Address')}
            rules={[
              { required: true, message: t('common.required', 'Required') },
              { type: 'email', message: t('common.invalid_email', 'Invalid email') },
            ]}
          >
            <Input prefix={<UserOutlined />} placeholder="user@example.com" />
          </Form.Item>
          <Form.Item
            name="role"
            label={t('projects.team.role', 'Role')}
            initialValue="member"
          >
            <Select options={roleOptions.filter(r => r.value !== 'owner')} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button onClick={() => setInviteModalOpen(false)}>
                {t('common.cancel', 'Cancel')}
              </Button>
              <Button type="primary" htmlType="submit" loading={inviteMember.isPending}>
                {t('projects.team.send_invite', 'Send Invite')}
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

/**
 * Webhooks Section
 */
const WebhooksSection: React.FC<{ projectId: string }> = ({ projectId }) => {
  const { t } = useTranslation();
  const [modalOpen, setModalOpen] = useState(false);
  const [editingWebhook, setEditingWebhook] = useState<Webhook | null>(null);
  const [form] = Form.useForm();

  const { data: webhooks, isLoading } = useProjectWebhooks(projectId);
  const createWebhook = useCreateWebhook(projectId);
  const updateWebhook = useUpdateWebhook(projectId);
  const deleteWebhook = useDeleteWebhook(projectId);
  const testWebhook = useTestWebhook(projectId);

  const handleSubmit = useCallback(async (values: any) => {
    if (editingWebhook) {
      await updateWebhook.mutateAsync({ webhookId: editingWebhook.id, data: values });
    } else {
      await createWebhook.mutateAsync(values);
    }
    setModalOpen(false);
    setEditingWebhook(null);
    form.resetFields();
  }, [editingWebhook, createWebhook, updateWebhook, form]);

  const handleEdit = useCallback((webhook: Webhook) => {
    setEditingWebhook(webhook);
    form.setFieldsValue(webhook);
    setModalOpen(true);
  }, [form]);

  const columns: TableProps<Webhook>['columns'] = [
    {
      title: t('projects.webhooks.url', 'URL'),
      dataIndex: 'url',
      key: 'url',
      ellipsis: true,
      render: (url: string) => (
        <Tooltip title={url}>
          <Text copyable={{ text: url }}>{url}</Text>
        </Tooltip>
      ),
    },
    {
      title: t('projects.webhooks.events', 'Events'),
      dataIndex: 'events',
      key: 'events',
      render: (events: string[]) => (
        <Space wrap>
          {events.slice(0, 2).map(e => <Tag key={e}>{e}</Tag>)}
          {events.length > 2 && <Tag>+{events.length - 2}</Tag>}
        </Space>
      ),
    },
    {
      title: t('projects.webhooks.status', 'Status'),
      key: 'status',
      render: (_, record) => (
        <Space>
          <Badge status={record.is_active ? 'success' : 'default'} />
          {record.last_status === 'success' ? 
            <CheckCircleOutlined style={{ color: '#52c41a' }} /> :
            record.last_status === 'failure' ?
            <CloseCircleOutlined style={{ color: '#ff4d4f' }} /> : null
          }
        </Space>
      ),
    },
    {
      title: '',
      key: 'actions',
      width: 150,
      render: (_, record) => (
        <Space>
          <Tooltip title={t('projects.webhooks.test', 'Test')}>
            <Button 
              type="text" 
              icon={<SendOutlined />} 
              onClick={() => testWebhook.mutate(record.id)}
              loading={testWebhook.isPending}
            />
          </Tooltip>
          <Tooltip title={t('common.edit', 'Edit')}>
            <Button type="text" icon={<EditOutlined />} onClick={() => handleEdit(record)} />
          </Tooltip>
          <Popconfirm
            title={t('projects.webhooks.delete_confirm', 'Delete this webhook?')}
            onConfirm={() => deleteWebhook.mutate(record.id)}
          >
            <Button type="text" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
        <Col>
          <Text strong>{t('projects.webhooks.title', 'Webhooks')}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {t('projects.webhooks.description', 'Receive real-time notifications about project events')}
          </Text>
        </Col>
        <Col>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalOpen(true)}>
            {t('projects.webhooks.add', 'Add Webhook')}
          </Button>
        </Col>
      </Row>

      <Table
        columns={columns}
        dataSource={webhooks || []}
        rowKey="id"
        loading={isLoading}
        pagination={false}
        locale={{ emptyText: <Empty description={t('projects.webhooks.no_webhooks', 'No webhooks configured')} /> }}
      />

      <Modal
        title={editingWebhook ? t('projects.webhooks.edit', 'Edit Webhook') : t('projects.webhooks.add', 'Add Webhook')}
        open={modalOpen}
        onCancel={() => { setModalOpen(false); setEditingWebhook(null); form.resetFields(); }}
        footer={null}
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={handleSubmit}>
          <Form.Item
            name="url"
            label={t('projects.webhooks.url', 'Webhook URL')}
            rules={[
              { required: true, message: t('common.required', 'Required') },
              { type: 'url', message: t('common.invalid_url', 'Invalid URL') },
            ]}
          >
            <Input placeholder="https://example.com/webhook" />
          </Form.Item>
          <Form.Item
            name="events"
            label={t('projects.webhooks.events', 'Events')}
            rules={[{ required: true, message: t('common.required', 'Required') }]}
          >
            <Select mode="multiple" options={webhookEventOptions} placeholder={t('projects.webhooks.select_events', 'Select events')} />
          </Form.Item>
          <Form.Item name="secret" label={t('projects.webhooks.secret', 'Secret (optional)')}>
            <Input.Password placeholder="Optional signing secret" />
          </Form.Item>
          {editingWebhook && (
            <Form.Item name="is_active" label={t('projects.webhooks.active', 'Active')} valuePropName="checked">
              <Switch />
            </Form.Item>
          )}
          <Form.Item>
            <Space>
              <Button onClick={() => { setModalOpen(false); setEditingWebhook(null); }}>
                {t('common.cancel', 'Cancel')}
              </Button>
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

/**
 * API Keys Section
 */
const ApiKeysSection: React.FC<{ projectId: string }> = ({ projectId }) => {
  const { t } = useTranslation();
  const [modalOpen, setModalOpen] = useState(false);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [form] = Form.useForm();

  const { data: apiKeys, isLoading } = useProjectApiKeys(projectId);
  const createApiKey = useCreateApiKey(projectId);
  const revokeApiKey = useRevokeApiKey(projectId);

  const handleCreate = useCallback(async (values: any) => {
    const result = await createApiKey.mutateAsync(values);
    setNewKey(result.key);
    form.resetFields();
  }, [createApiKey, form]);

  const handleCopy = useCallback((key: string) => {
    navigator.clipboard.writeText(key);
    message.success(t('common.copied', 'Copied to clipboard'));
  }, [t]);

  const columns: TableProps<APIKey>['columns'] = [
    {
      title: t('projects.api_keys.name', 'Name'),
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: t('projects.api_keys.key', 'Key'),
      dataIndex: 'key_prefix',
      key: 'key',
      render: (prefix: string) => (
        <Text code>{prefix}...</Text>
      ),
    },
    {
      title: t('projects.api_keys.permissions', 'Permissions'),
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
      title: t('projects.api_keys.usage', 'Usage'),
      dataIndex: 'usage_count',
      key: 'usage',
      render: (count: number) => count.toLocaleString(),
    },
    {
      title: t('projects.api_keys.last_used', 'Last Used'),
      dataIndex: 'last_used_at',
      key: 'last_used',
      render: (date: string) => date ? new Date(date).toLocaleDateString() : '-',
    },
    {
      title: '',
      key: 'actions',
      width: 80,
      render: (_, record) => (
        <Popconfirm
          title={t('projects.api_keys.revoke_confirm', 'Revoke this API key?')}
          description={t('projects.api_keys.revoke_warning', 'This action cannot be undone.')}
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
    <div>
      <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
        <Col>
          <Text strong>{t('projects.api_keys.title', 'API Keys')}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {t('projects.api_keys.description', 'Manage API keys for programmatic access')}
          </Text>
        </Col>
        <Col>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalOpen(true)}>
            {t('projects.api_keys.create', 'Create API Key')}
          </Button>
        </Col>
      </Row>

      <Table
        columns={columns}
        dataSource={apiKeys || []}
        rowKey="id"
        loading={isLoading}
        pagination={false}
        locale={{ emptyText: <Empty description={t('projects.api_keys.no_keys', 'No API keys')} /> }}
      />

      <Modal
        title={t('projects.api_keys.create', 'Create API Key')}
        open={modalOpen}
        onCancel={() => { setModalOpen(false); setNewKey(null); form.resetFields(); }}
        footer={null}
      >
        {newKey ? (
          <div>
            <Alert
              message={t('projects.api_keys.copy_warning', 'Copy your API key now')}
              description={t('projects.api_keys.copy_warning_desc', 'This is the only time you will see this key.')}
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
            />
            <Input.Group compact style={{ marginBottom: 16 }}>
              <Input value={newKey} readOnly style={{ width: 'calc(100% - 80px)' }} />
              <Button icon={<CopyOutlined />} onClick={() => handleCopy(newKey)}>
                {t('common.copy', 'Copy')}
              </Button>
            </Input.Group>
            <Button type="primary" block onClick={() => { setModalOpen(false); setNewKey(null); }}>
              {t('common.done', 'Done')}
            </Button>
          </div>
        ) : (
          <Form form={form} layout="vertical" onFinish={handleCreate}>
            <Form.Item
              name="name"
              label={t('projects.api_keys.name', 'Key Name')}
              rules={[{ required: true, message: t('common.required', 'Required') }]}
            >
              <Input placeholder="e.g., CI/CD Pipeline" />
            </Form.Item>
            <Form.Item
              name="permissions"
              label={t('projects.api_keys.permissions', 'Permissions')}
              rules={[{ required: true, message: t('common.required', 'Required') }]}
            >
              <Select mode="multiple" options={apiPermissionOptions} placeholder={t('projects.api_keys.select_permissions', 'Select permissions')} />
            </Form.Item>
            <Form.Item name="expires_at" label={t('projects.api_keys.expiry', 'Expiration (optional)')}>
              <Select
                placeholder={t('projects.api_keys.no_expiry', 'No expiration')}
                allowClear
                options={[
                  { value: '30d', label: '30 days' },
                  { value: '90d', label: '90 days' },
                  { value: '1y', label: '1 year' },
                ]}
              />
            </Form.Item>
            <Form.Item>
              <Space>
                <Button onClick={() => setModalOpen(false)}>
                  {t('common.cancel', 'Cancel')}
                </Button>
                <Button type="primary" htmlType="submit" loading={createApiKey.isPending}>
                  {t('common.create', 'Create')}
                </Button>
              </Space>
            </Form.Item>
          </Form>
        )}
      </Modal>
    </div>
  );
};

/**
 * Activity Log Section
 */
const ActivitySection: React.FC<{ projectId: string }> = ({ projectId }) => {
  const { t } = useTranslation();
  const { data, isLoading, refetch } = useProjectActivity(projectId, { limit: 20 });

  return (
    <div>
      <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
        <Col>
          <Text strong>{t('projects.activity.title', 'Activity Log')}</Text>
        </Col>
        <Col>
          <Button icon={<ReloadOutlined />} onClick={() => refetch()}>
            {t('common.refresh', 'Refresh')}
          </Button>
        </Col>
      </Row>

      {isLoading ? (
        <Skeleton active paragraph={{ rows: 5 }} />
      ) : (
        <Timeline
          items={(data?.items || []).map((item: ActivityLog) => ({
            color: item.action.includes('delete') ? 'red' : 
                   item.action.includes('create') ? 'green' : 'blue',
            children: (
              <div>
                <Space>
                  <Avatar src={item.actor.avatar} icon={<UserOutlined />} size="small" />
                  <Text strong>{item.actor.name}</Text>
                </Space>
                <Paragraph style={{ margin: '4px 0' }}>{item.description}</Paragraph>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {new Date(item.created_at).toLocaleString()}
                </Text>
              </div>
            ),
          }))}
        />
      )}

      {!isLoading && (!data?.items || data.items.length === 0) && (
        <Empty description={t('projects.activity.no_activity', 'No activity yet')} />
      )}
    </div>
  );
};

/**
 * Danger Zone Section
 */
const DangerZoneSection: React.FC<{
  project: any;
  onArchive: () => void;
  onDelete: () => void;
  isArchiving: boolean;
  isDeleting: boolean;
}> = ({ project, onArchive, onDelete, isArchiving, isDeleting }) => {
  const { t } = useTranslation();
  const [deleteConfirmText, setDeleteConfirmText] = useState('');
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);

  const canDelete = deleteConfirmText === project?.name;

  return (
    <div className="danger-zone">
      <Alert
        message={t('projects.danger.warning', 'Danger Zone')}
        description={t('projects.danger.warning_desc', 'These actions are irreversible. Please proceed with caution.')}
        type="error"
        showIcon
        icon={<WarningOutlined />}
        style={{ marginBottom: 24 }}
      />

      <Card className="danger-card">
        <Row justify="space-between" align="middle">
          <Col>
            <Text strong>{project?.status === 'archived' ? t('projects.danger.restore', 'Restore Project') : t('projects.danger.archive', 'Archive Project')}</Text>
            <br />
            <Text type="secondary">
              {project?.status === 'archived' 
                ? t('projects.danger.restore_desc', 'Restore this project to active status')
                : t('projects.danger.archive_desc', 'Archive this project. It can be restored later.')
              }
            </Text>
          </Col>
          <Col>
            <Button 
              danger={project?.status !== 'archived'}
              icon={<InboxOutlined />}
              onClick={onArchive}
              loading={isArchiving}
            >
              {project?.status === 'archived' ? t('projects.danger.restore_btn', 'Restore') : t('projects.danger.archive_btn', 'Archive')}
            </Button>
          </Col>
        </Row>
      </Card>

      <Card className="danger-card">
        <Row justify="space-between" align="middle">
          <Col>
            <Text strong type="danger">{t('projects.danger.delete', 'Delete Project')}</Text>
            <br />
            <Text type="secondary">
              {t('projects.danger.delete_desc', 'Permanently delete this project and all associated data.')}
            </Text>
          </Col>
          <Col>
            <Button 
              type="primary"
              danger
              icon={<DeleteOutlined />}
              onClick={() => setDeleteModalOpen(true)}
            >
              {t('projects.danger.delete_btn', 'Delete Project')}
            </Button>
          </Col>
        </Row>
      </Card>

      <Modal
        title={<><ExclamationCircleOutlined style={{ color: '#ff4d4f', marginRight: 8 }} />{t('projects.danger.delete_confirm', 'Confirm Deletion')}</>}
        open={deleteModalOpen}
        onCancel={() => { setDeleteModalOpen(false); setDeleteConfirmText(''); }}
        footer={[
          <Button key="cancel" onClick={() => setDeleteModalOpen(false)}>
            {t('common.cancel', 'Cancel')}
          </Button>,
          <Button 
            key="delete" 
            type="primary" 
            danger 
            disabled={!canDelete}
            loading={isDeleting}
            onClick={onDelete}
          >
            {t('projects.danger.delete_btn', 'Delete Project')}
          </Button>,
        ]}
      >
        <p>{t('projects.danger.delete_modal_text', 'This action cannot be undone. Please type the project name to confirm:')}</p>
        <p><Text strong>{project?.name}</Text></p>
        <Input 
          placeholder={project?.name}
          value={deleteConfirmText}
          onChange={(e) => setDeleteConfirmText(e.target.value)}
        />
      </Modal>
    </div>
  );
};

/**
 * Project Settings Page Component
 */
export const ProjectSettings: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  
  // Store
  const { unsavedChanges, setUnsavedChanges } = useProjectStore();
  
  // Queries
  const { data: project, isLoading, isError } = useProject(id);
  
  // Mutations
  const updateProject = useUpdateProject();
  const deleteProject = useDeleteProject();
  const archiveProject = useArchiveProject();
  
  // Handle save
  const handleSave = useCallback(async (values: any) => {
    if (!id) return;
    await updateProject.mutateAsync({ id, data: values });
    setUnsavedChanges(false);
  }, [id, updateProject, setUnsavedChanges]);
  
  // Handle archive/restore
  const handleArchive = useCallback(() => {
    if (!id) return;
    archiveProject.mutate(id);
  }, [id, archiveProject]);
  
  // Handle delete
  const handleDelete = useCallback(() => {
    if (!id) return;
    deleteProject.mutate(id, {
      onSuccess: () => navigate('/projects'),
    });
  }, [id, deleteProject, navigate]);
  
  // Warn on navigation with unsaved changes
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (unsavedChanges) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [unsavedChanges]);
  
  // Tab items
  const tabItems: TabsProps['items'] = [
    {
      key: 'general',
      label: <><SettingOutlined /> {t('projects.settings.general', 'General')}</>,
      children: project ? (
        <ProjectInfoSection
          project={project}
          onSave={handleSave}
          isSaving={updateProject.isPending}
        />
      ) : null,
    },
    {
      key: 'team',
      label: <><TeamOutlined /> {t('projects.settings.team', 'Team')}</>,
      children: id ? <TeamSection projectId={id} /> : null,
    },
    {
      key: 'webhooks',
      label: <><ApiOutlined /> {t('projects.settings.webhooks', 'Webhooks')}</>,
      children: id ? <WebhooksSection projectId={id} /> : null,
    },
    {
      key: 'api-keys',
      label: <><KeyOutlined /> {t('projects.settings.api_keys', 'API Keys')}</>,
      children: id ? <ApiKeysSection projectId={id} /> : null,
    },
    {
      key: 'activity',
      label: <><HistoryOutlined /> {t('projects.settings.activity', 'Activity')}</>,
      children: id ? <ActivitySection projectId={id} /> : null,
    },
    {
      key: 'danger',
      label: <><WarningOutlined /> {t('projects.settings.danger', 'Danger Zone')}</>,
      children: project ? (
        <DangerZoneSection
          project={project}
          onArchive={handleArchive}
          onDelete={handleDelete}
          isArchiving={archiveProject.isPending}
          isDeleting={deleteProject.isPending}
        />
      ) : null,
    },
  ];

  if (isLoading) {
    return (
      <div className="project-settings-container">
        <Skeleton active paragraph={{ rows: 10 }} />
      </div>
    );
  }

  if (isError || !project) {
    return (
      <div className="project-settings-container">
        <Alert
          message={t('projects.settings.not_found', 'Project not found')}
          type="error"
          showIcon
          action={
            <Button onClick={() => navigate('/projects')}>
              {t('common.back', 'Back to Projects')}
            </Button>
          }
        />
      </div>
    );
  }

  return (
    <div className="project-settings-container" role="main">
      {/* Header */}
      <div className="settings-header">
        <Space>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/projects')} />
          <div>
            <Title level={3} style={{ margin: 0 }}>{project.name}</Title>
            <Text type="secondary">{t('projects.settings.title', 'Project Settings')}</Text>
          </div>
        </Space>
        {unsavedChanges && (
          <Tag color="warning">{t('projects.settings.unsaved', 'Unsaved changes')}</Tag>
        )}
      </div>

      {/* Settings Tabs */}
      <Card className="settings-card">
        <Tabs
          defaultActiveKey="general"
          items={tabItems}
          tabPosition="left"
          className="settings-tabs"
        />
      </Card>
    </div>
  );
};

export default ProjectSettings;
