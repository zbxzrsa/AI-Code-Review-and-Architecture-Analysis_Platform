/**
 * 用户管理页面 (User Management Page)
 * 
 * 功能描述:
 *   管理员用户管理界面。
 * 
 * 主要特性:
 *   - 用户列表（支持筛选、排序、分页）
 *   - 用户统计和图表
 *   - 批量操作
 *   - 用户详情弹窗
 *   - 用户角色管理
 *   - 账户状态控制
 * 
 * 最后修改日期: 2024-12-07
 */

import React, { useState, useCallback } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Typography,
  Input,
  Select,
  Tag,
  Avatar,
  Modal,
  Form,
  Dropdown,
  Statistic,
  Row,
  Col,
  Badge,
  Tooltip,
  Upload,
  Skeleton,
  Empty,
  Descriptions,
  Divider,
} from 'antd';
import type { TableProps, MenuProps, UploadProps } from 'antd';
import {
  UserOutlined,
  EditOutlined,
  DeleteOutlined,
  DownloadOutlined,
  UploadOutlined,
  MoreOutlined,
  TeamOutlined,
  UserAddOutlined,
  LockOutlined,
  MailOutlined,
  ReloadOutlined,
  FilterOutlined,
  CheckCircleOutlined,
  StopOutlined,
  PlayCircleOutlined,
  KeyOutlined,
  ExperimentOutlined,
  CrownOutlined,
  EyeOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import {
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  useAdminStore,
  type AdminUser,
  type UserRole,
  type UserStatus,
} from '../../store/adminStore';
import {
  useAdminUsers,
  useUserStats,
  useUpdateUser,
  useDeleteUser,
  useSuspendUser,
  useReactivateUser,
  useResetUserPassword,
  useResendWelcome,
  useBulkUserAction,
  useImportUsers,
  useExportUsers,
} from '../../hooks/useAdmin';
import './UserManagement.css';

const { Title, Text } = Typography;
const { Search } = Input;

/** Status colors (reserved for future use) */
const _statusColors: Record<UserStatus, string> = {
  active: 'green',
  inactive: 'orange',
  suspended: 'red',
};

/** Role colors */
const roleColors: Record<UserRole, string> = {
  admin: 'purple',
  analyst: 'blue',
  viewer: 'default',
};

/** Role icons */
const roleIcons: Record<UserRole, React.ReactNode> = {
  admin: <CrownOutlined />,
  analyst: <ExperimentOutlined />,
  viewer: <EyeOutlined />,
};

/** Chart colors */
const CHART_COLORS = ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1'];

/**
 * User Statistics Section
 */
const UserStatsSection: React.FC = () => {
  const { t } = useTranslation();
  const { data: stats, isLoading } = useUserStats();

  if (isLoading) {
    return <Skeleton active paragraph={{ rows: 2 }} />;
  }

  if (!stats) {
    return null;
  }

  const roleData = (stats.byRole || []).map((r, i) => ({
    name: r.role,
    value: r.count,
    color: CHART_COLORS[i % CHART_COLORS.length],
  }));

  return (
    <Row gutter={[16, 16]} className="user-stats">
      <Col xs={12} sm={6}>
        <Card>
          <Statistic
            title={t('admin.users.total', 'Total Users')}
            value={stats.total}
            prefix={<TeamOutlined />}
          />
        </Card>
      </Col>
      <Col xs={12} sm={6}>
        <Card>
          <Statistic
            title={t('admin.users.active', 'Active Users')}
            value={stats.active}
            prefix={<CheckCircleOutlined />}
            valueStyle={{ color: '#52c41a' }}
          />
        </Card>
      </Col>
      <Col xs={12} sm={6}>
        <Card>
          <Statistic
            title={t('admin.users.suspended', 'Suspended')}
            value={stats.suspended}
            prefix={<StopOutlined />}
            valueStyle={{ color: '#f5222d' }}
          />
        </Card>
      </Col>
      <Col xs={12} sm={6}>
        <Card>
          <Statistic
            title={t('admin.users.recently_joined', 'New (7 days)')}
            value={stats.recentlyJoined}
            prefix={<UserAddOutlined />}
            valueStyle={{ color: '#1890ff' }}
          />
        </Card>
      </Col>
      
      {/* Charts */}
      <Col xs={24} md={12}>
        <Card title={t('admin.users.by_role', 'Users by Role')} size="small">
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={roleData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
                label={({ name, percent }: any) => `${name || ''} (${((percent || 0) * 100).toFixed(0)}%)`}
              >
                {roleData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <RechartsTooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </Col>
      <Col xs={24} md={12}>
        <Card title={t('admin.users.activity_trend', 'Activity Trend')} size="small">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={stats.activityTrend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" fontSize={12} />
              <YAxis fontSize={12} />
              <RechartsTooltip />
              <Line type="monotone" dataKey="count" stroke="#1890ff" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </Col>
    </Row>
  );
};

/**
 * User Detail Modal
 */
const UserDetailModal: React.FC<{
  user: AdminUser | null;
  open: boolean;
  onClose: () => void;
  onEdit: (user: AdminUser) => void;
}> = ({ user, open, onClose, onEdit }) => {
  const { t } = useTranslation();

  if (!user) return null;

  return (
    <Modal
      title={
        <Space>
          <Avatar src={user.avatar} icon={<UserOutlined />} size={40} />
          <div>
            <Text strong>{user.name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>@{user.username}</Text>
          </div>
        </Space>
      }
      open={open}
      onCancel={onClose}
      footer={[
        <Button key="close" onClick={onClose}>
          {t('common.close', 'Close')}
        </Button>,
        <Button key="edit" type="primary" onClick={() => onEdit(user)}>
          {t('common.edit', 'Edit')}
        </Button>,
      ]}
      width={600}
    >
      <Divider />
      <Descriptions column={2} bordered size="small">
        <Descriptions.Item label={t('admin.users.email', 'Email')} span={2}>
          <Space>
            <MailOutlined />
            {user.email}
            {user.emailVerified && (
              <Tag color="green" icon={<CheckCircleOutlined />}>
                {t('admin.users.verified', 'Verified')}
              </Tag>
            )}
          </Space>
        </Descriptions.Item>
        <Descriptions.Item label={t('admin.users.role', 'Role')}>
          <Tag color={roleColors[user.role]} icon={roleIcons[user.role]}>
            {user.role.toUpperCase()}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label={t('admin.users.status', 'Status')}>
          <Badge status={user.status === 'active' ? 'success' : user.status === 'suspended' ? 'error' : 'warning'} />
          {user.status}
        </Descriptions.Item>
        <Descriptions.Item label={t('admin.users.joined', 'Joined')}>
          {new Date(user.joinedAt).toLocaleDateString()}
        </Descriptions.Item>
        <Descriptions.Item label={t('admin.users.last_login', 'Last Login')}>
          {user.lastLoginAt ? new Date(user.lastLoginAt).toLocaleString() : '-'}
        </Descriptions.Item>
        <Descriptions.Item label={t('admin.users.projects', 'Projects')}>
          {user.projectCount}
        </Descriptions.Item>
        <Descriptions.Item label={t('admin.users.analyses', 'Analyses')}>
          {user.analysisCount}
        </Descriptions.Item>
        <Descriptions.Item label={t('admin.users.2fa', '2FA')} span={2}>
          {user.twoFactorEnabled ? (
            <Tag color="green" icon={<LockOutlined />}>{t('common.enabled', 'Enabled')}</Tag>
          ) : (
            <Tag>{t('common.disabled', 'Disabled')}</Tag>
          )}
        </Descriptions.Item>
      </Descriptions>
    </Modal>
  );
};

/**
 * Edit User Modal
 */
const EditUserModal: React.FC<{
  user: AdminUser | null;
  open: boolean;
  onClose: () => void;
}> = ({ user, open, onClose }) => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const updateUser = useUpdateUser();

  const handleSubmit = useCallback(async (values: any) => {
    if (!user) return;
    await updateUser.mutateAsync({ userId: user.id, data: values });
    onClose();
  }, [user, updateUser, onClose]);

  React.useEffect(() => {
    if (user && open) {
      form.setFieldsValue({
        email: user.email,
        name: user.name,
        role: user.role,
        status: user.status,
      });
    }
  }, [user, open, form]);

  return (
    <Modal
      title={t('admin.users.edit_user', 'Edit User')}
      open={open}
      onCancel={onClose}
      footer={null}
    >
      <Form form={form} layout="vertical" onFinish={handleSubmit}>
        <Form.Item
          name="name"
          label={t('admin.users.name', 'Name')}
          rules={[{ required: true }]}
        >
          <Input prefix={<UserOutlined />} />
        </Form.Item>
        <Form.Item
          name="email"
          label={t('admin.users.email', 'Email')}
          rules={[{ required: true, type: 'email' }]}
        >
          <Input prefix={<MailOutlined />} />
        </Form.Item>
        <Form.Item
          name="role"
          label={t('admin.users.role', 'Role')}
          rules={[{ required: true }]}
        >
          <Select
            options={[
              { value: 'admin', label: 'Admin' },
              { value: 'analyst', label: 'Analyst' },
              { value: 'viewer', label: 'Viewer' },
            ]}
          />
        </Form.Item>
        <Form.Item
          name="status"
          label={t('admin.users.status', 'Status')}
          rules={[{ required: true }]}
        >
          <Select
            options={[
              { value: 'active', label: 'Active' },
              { value: 'inactive', label: 'Inactive' },
              { value: 'suspended', label: 'Suspended' },
            ]}
          />
        </Form.Item>
        <Form.Item>
          <Space>
            <Button onClick={onClose}>{t('common.cancel', 'Cancel')}</Button>
            <Button type="primary" htmlType="submit" loading={updateUser.isPending}>
              {t('common.save', 'Save')}
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Modal>
  );
};

/**
 * Bulk Action Dropdown
 */
const BulkActionDropdown: React.FC<{
  selectedCount: number;
  onAction: (action: string, role?: string) => void;
  loading: boolean;
}> = ({ selectedCount, onAction, loading }) => {
  const { t } = useTranslation();

  const items: MenuProps['items'] = [
    {
      key: 'change_role',
      label: t('admin.users.change_role', 'Change Role'),
      icon: <CrownOutlined />,
      children: [
        { key: 'role_admin', label: 'Admin', onClick: () => onAction('change_role', 'admin') },
        { key: 'role_analyst', label: 'Analyst', onClick: () => onAction('change_role', 'analyst') },
        { key: 'role_viewer', label: 'Viewer', onClick: () => onAction('change_role', 'viewer') },
      ],
    },
    { type: 'divider' },
    {
      key: 'suspend',
      label: t('admin.users.suspend', 'Suspend'),
      icon: <StopOutlined />,
      danger: true,
      onClick: () => onAction('suspend'),
    },
    {
      key: 'reactivate',
      label: t('admin.users.reactivate', 'Reactivate'),
      icon: <PlayCircleOutlined />,
      onClick: () => onAction('reactivate'),
    },
    { type: 'divider' },
    {
      key: 'delete',
      label: t('admin.users.delete', 'Delete'),
      icon: <DeleteOutlined />,
      danger: true,
      onClick: () => onAction('delete'),
    },
  ];

  return (
    <Dropdown menu={{ items }} disabled={selectedCount === 0 || loading}>
      <Button loading={loading}>
        {t('admin.users.bulk_actions', 'Bulk Actions')} ({selectedCount})
        <MoreOutlined />
      </Button>
    </Dropdown>
  );
};

/**
 * Main User Management Component
 */
export const UserManagement: React.FC = () => {
  const { t } = useTranslation();
  const [viewUser, setViewUser] = useState<AdminUser | null>(null);
  const [editUser, setEditUser] = useState<AdminUser | null>(null);
  
  const {
    userFilters,
    userPagination,
    userSort: _userSort,
    selectedUserIds,
    setUserFilters,
    resetUserFilters,
    setUserPagination,
    setUserSort,
    selectUser,
    deselectUser,
    selectAllUsers,
    clearUserSelection,
  } = useAdminStore();

  const { data, isLoading, refetch, isFetching } = useAdminUsers();
  const deleteUser = useDeleteUser();
  const suspendUser = useSuspendUser();
  const reactivateUser = useReactivateUser();
  const resetPassword = useResetUserPassword();
  const resendWelcome = useResendWelcome();
  const bulkAction = useBulkUserAction();
  const importUsers = useImportUsers();
  const exportUsers = useExportUsers();

  // Handle search with debounce
  const handleSearch = useCallback((value: string) => {
    setUserFilters({ search: value });
  }, [setUserFilters]);

  // Handle bulk action
  const handleBulkAction = useCallback((action: string, role?: string) => {
    Modal.confirm({
      title: t('admin.users.confirm_bulk', 'Confirm Bulk Action'),
      content: t('admin.users.confirm_bulk_desc', `This will ${action} ${selectedUserIds.length} users.`),
      okType: action === 'delete' || action === 'suspend' ? 'danger' : 'primary',
      onOk: () => {
        bulkAction.mutate({
          userIds: selectedUserIds,
          action: action as any,
          role,
        });
      },
    });
  }, [selectedUserIds, bulkAction, t]);

  // Import props
  const uploadProps: UploadProps = {
    accept: '.csv',
    showUploadList: false,
    beforeUpload: (file) => {
      importUsers.mutate(file);
      return false;
    },
  };

  // Table columns
  const columns: TableProps<AdminUser>['columns'] = [
    {
      title: t('admin.users.user', 'User'),
      key: 'user',
      fixed: 'left',
      width: 250,
      render: (_, record) => (
        <Space>
          <Avatar src={record.avatar} icon={<UserOutlined />} />
          <div>
            <Text strong style={{ cursor: 'pointer' }} onClick={() => setViewUser(record)}>
              {record.name}
            </Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>@{record.username}</Text>
          </div>
        </Space>
      ),
    },
    {
      title: t('admin.users.email', 'Email'),
      dataIndex: 'email',
      key: 'email',
      width: 200,
      ellipsis: true,
      render: (email: string, record) => (
        <Space>
          <Text>{email}</Text>
          {record.emailVerified && (
            <Tooltip title={t('admin.users.verified', 'Verified')}>
              <CheckCircleOutlined style={{ color: '#52c41a' }} />
            </Tooltip>
          )}
        </Space>
      ),
    },
    {
      title: t('admin.users.role', 'Role'),
      dataIndex: 'role',
      key: 'role',
      width: 120,
      filters: [
        { text: 'Admin', value: 'admin' },
        { text: 'Analyst', value: 'analyst' },
        { text: 'Viewer', value: 'viewer' },
      ],
      render: (role: UserRole) => (
        <Tag color={roleColors[role]} icon={roleIcons[role]}>
          {role.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: t('admin.users.status', 'Status'),
      dataIndex: 'status',
      key: 'status',
      width: 100,
      filters: [
        { text: 'Active', value: 'active' },
        { text: 'Inactive', value: 'inactive' },
        { text: 'Suspended', value: 'suspended' },
      ],
      render: (status: UserStatus) => (
        <Badge
          status={status === 'active' ? 'success' : status === 'suspended' ? 'error' : 'warning'}
          text={status}
        />
      ),
    },
    {
      title: t('admin.users.joined', 'Joined'),
      dataIndex: 'joinedAt',
      key: 'joinedAt',
      width: 120,
      sorter: true,
      render: (date: string) => new Date(date).toLocaleDateString(),
    },
    {
      title: t('admin.users.last_login', 'Last Login'),
      dataIndex: 'lastLoginAt',
      key: 'lastLoginAt',
      width: 150,
      sorter: true,
      render: (date: string) => date ? new Date(date).toLocaleString() : '-',
    },
    {
      title: t('admin.users.actions', 'Actions'),
      key: 'actions',
      fixed: 'right',
      width: 180,
      render: (_, record) => {
        const menuItems: MenuProps['items'] = [
          {
            key: 'view',
            label: t('admin.users.view', 'View Details'),
            icon: <EyeOutlined />,
            onClick: () => setViewUser(record),
          },
          {
            key: 'edit',
            label: t('admin.users.edit', 'Edit'),
            icon: <EditOutlined />,
            onClick: () => setEditUser(record),
          },
          { type: 'divider' },
          {
            key: 'reset_password',
            label: t('admin.users.reset_password', 'Reset Password'),
            icon: <KeyOutlined />,
            onClick: () => resetPassword.mutate(record.id),
          },
          {
            key: 'resend_welcome',
            label: t('admin.users.resend_welcome', 'Resend Welcome'),
            icon: <MailOutlined />,
            onClick: () => resendWelcome.mutate(record.id),
          },
          { type: 'divider' },
          record.status === 'suspended' ? {
            key: 'reactivate',
            label: t('admin.users.reactivate', 'Reactivate'),
            icon: <PlayCircleOutlined />,
            onClick: () => reactivateUser.mutate(record.id),
          } : {
            key: 'suspend',
            label: t('admin.users.suspend', 'Suspend'),
            icon: <StopOutlined />,
            danger: true,
            onClick: () => {
              Modal.confirm({
                title: t('admin.users.suspend_confirm', 'Suspend User?'),
                content: t('admin.users.suspend_desc', 'This user will not be able to access the platform.'),
                okType: 'danger',
                onOk: () => suspendUser.mutate({ userId: record.id }),
              });
            },
          },
          { type: 'divider' },
          {
            key: 'delete',
            label: t('admin.users.delete', 'Delete'),
            icon: <DeleteOutlined />,
            danger: true,
            onClick: () => {
              Modal.confirm({
                title: t('admin.users.delete_confirm', 'Delete User?'),
                content: t('admin.users.delete_desc', 'This action cannot be undone.'),
                okType: 'danger',
                onOk: () => deleteUser.mutate(record.id),
              });
            },
          },
        ];

        return (
          <Space>
            <Button size="small" icon={<EditOutlined />} onClick={() => setEditUser(record)} />
            <Dropdown menu={{ items: menuItems }}>
              <Button size="small" icon={<MoreOutlined />} />
            </Dropdown>
          </Space>
        );
      },
    },
  ];

  // Row selection
  const rowSelection: TableProps<AdminUser>['rowSelection'] = {
    selectedRowKeys: selectedUserIds,
    onChange: (selectedKeys) => {
      selectAllUsers(selectedKeys as string[]);
    },
    onSelect: (record, selected) => {
      if (selected) {
        selectUser(record.id);
      } else {
        deselectUser(record.id);
      }
    },
    onSelectAll: (selected, selectedRows) => {
      if (selected) {
        selectAllUsers(selectedRows.map(r => r.id));
      } else {
        clearUserSelection();
      }
    },
  };

  // Handle table change
  const handleTableChange: TableProps<AdminUser>['onChange'] = (pagination, filters, sorter) => {
    if (pagination.current && pagination.pageSize) {
      setUserPagination({
        page: pagination.current,
        pageSize: pagination.pageSize,
      });
    }
    
    if (!Array.isArray(sorter) && sorter.field) {
      setUserSort({
        field: sorter.field as string,
        order: sorter.order === 'ascend' ? 'asc' : 'desc',
      });
    }
    
    if (filters.role) {
      setUserFilters({ role: (filters.role[0] as UserRole) || 'all' });
    }
    if (filters.status) {
      setUserFilters({ status: (filters.status[0] as UserStatus) || 'all' });
    }
  };

  return (
    <div className="user-management" role="main" aria-label={t('admin.users.title', 'User Management')}>
      <div className="page-header">
        <Title level={2}><TeamOutlined /> {t('admin.users.title', 'User Management')}</Title>
        <Space>
          <Upload {...uploadProps}>
            <Button icon={<UploadOutlined />} loading={importUsers.isPending}>
              {t('admin.users.import', 'Import')}
            </Button>
          </Upload>
          <Button 
            icon={<DownloadOutlined />} 
            onClick={() => exportUsers.mutate({ format: 'csv' })}
            loading={exportUsers.isPending}
          >
            {t('admin.users.export', 'Export')}
          </Button>
        </Space>
      </div>

      {/* Statistics */}
      <UserStatsSection />

      {/* Filters and Actions */}
      <Card className="filters-card">
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} sm={12} md={8}>
            <Search
              placeholder={t('admin.users.search_placeholder', 'Search by name or email')}
              allowClear
              onSearch={handleSearch}
              defaultValue={userFilters.search}
            />
          </Col>
          <Col xs={12} sm={6} md={4}>
            <Select
              placeholder={t('admin.users.filter_role', 'Role')}
              value={userFilters.role}
              onChange={(value) => setUserFilters({ role: value })}
              options={[
                { value: 'all', label: 'All Roles' },
                { value: 'admin', label: 'Admin' },
                { value: 'analyst', label: 'Analyst' },
                { value: 'viewer', label: 'Viewer' },
              ]}
              style={{ width: '100%' }}
            />
          </Col>
          <Col xs={12} sm={6} md={4}>
            <Select
              placeholder={t('admin.users.filter_status', 'Status')}
              value={userFilters.status}
              onChange={(value) => setUserFilters({ status: value })}
              options={[
                { value: 'all', label: 'All Status' },
                { value: 'active', label: 'Active' },
                { value: 'inactive', label: 'Inactive' },
                { value: 'suspended', label: 'Suspended' },
              ]}
              style={{ width: '100%' }}
            />
          </Col>
          <Col flex="auto" style={{ textAlign: 'right' }}>
            <Space>
              <Button icon={<FilterOutlined />} onClick={resetUserFilters}>
                {t('admin.users.reset', 'Reset')}
              </Button>
              <BulkActionDropdown
                selectedCount={selectedUserIds.length}
                onAction={handleBulkAction}
                loading={bulkAction.isPending}
              />
              <Button 
                icon={<ReloadOutlined />} 
                onClick={() => refetch()}
                loading={isFetching}
              />
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Users Table */}
      <Card className="users-table-card">
        <Table
          columns={columns}
          dataSource={data?.items || []}
          rowKey="id"
          loading={isLoading}
          rowSelection={rowSelection}
          onChange={handleTableChange}
          pagination={{
            current: userPagination.page,
            pageSize: userPagination.pageSize,
            total: data?.total || 0,
            showSizeChanger: true,
            showTotal: (total) => t('admin.users.total_users', `Total ${total} users`),
            pageSizeOptions: ['10', '20', '50', '100'],
          }}
          scroll={{ x: 1200 }}
          locale={{
            emptyText: <Empty description={t('admin.users.no_users', 'No users found')} />,
          }}
        />
      </Card>

      {/* Modals */}
      <UserDetailModal
        user={viewUser}
        open={!!viewUser}
        onClose={() => setViewUser(null)}
        onEdit={(user) => {
          setViewUser(null);
          setEditUser(user);
        }}
      />
      
      <EditUserModal
        user={editUser}
        open={!!editUser}
        onClose={() => setEditUser(null)}
      />
    </div>
  );
};

export default UserManagement;
