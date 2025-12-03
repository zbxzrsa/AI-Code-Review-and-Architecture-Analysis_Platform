/**
 * Audit Logs Page
 * 
 * Admin page for viewing audit logs with:
 * - Audit log table with filtering and search
 * - Analytics and charts
 * - Security alerts
 * - Export functionality
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Typography,
  Input,
  Select,
  Tag,
  Badge,
  Modal,
  Row,
  Col,
  Statistic,
  Tooltip,
  DatePicker,
  Dropdown,
  Tabs,
  List,
  Alert,
  Descriptions,
  Empty,
  Skeleton,
  Timeline,
} from 'antd';
import type { TableProps, TabsProps, MenuProps } from 'antd';
import {
  AuditOutlined,
  SearchOutlined,
  FilterOutlined,
  DownloadOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  UserOutlined,
  LoginOutlined,
  LogoutOutlined,
  EditOutlined,
  DeleteOutlined,
  PlusCircleOutlined,
  EyeOutlined,
  SettingOutlined,
  KeyOutlined,
  LockOutlined,
  UnlockOutlined,
  GlobalOutlined,
  EnvironmentOutlined,
  ClockCircleOutlined,
  SafetyOutlined,
  AlertOutlined,
  FileTextOutlined,
  BarChartOutlined,
  PieChartOutlined,
  HeatMapOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import {
  useAdminStore,
  type AuditLog,
  type AuditAction,
  type AuditResource,
  type AuditStatus,
  type SecurityAlert,
} from '../../store/adminStore';
import {
  useAuditLogs,
  useAuditLog,
  useExportAuditLogs,
  useAuditAnalytics,
  useSecurityAlerts,
  useResolveAlert,
  useLoginPatterns,
} from '../../hooks/useAdmin';
import './AuditLogs.css';

const { Title, Text, Paragraph } = Typography;
const { Search } = Input;
const { RangePicker } = DatePicker;

/** Action icons */
const actionIcons: Record<AuditAction, React.ReactNode> = {
  CREATE: <PlusCircleOutlined />,
  READ: <EyeOutlined />,
  UPDATE: <EditOutlined />,
  DELETE: <DeleteOutlined />,
  LOGIN: <LoginOutlined />,
  LOGOUT: <LogoutOutlined />,
  LOGIN_FAILED: <CloseCircleOutlined />,
  PASSWORD_CHANGE: <KeyOutlined />,
  PASSWORD_RESET: <UnlockOutlined />,
  ROLE_CHANGE: <UserOutlined />,
  STATUS_CHANGE: <SettingOutlined />,
  ANALYZE: <BarChartOutlined />,
  EXPORT: <DownloadOutlined />,
  IMPORT: <PlusCircleOutlined />,
  PROVIDER_CONFIG: <SettingOutlined />,
  SETTINGS_CHANGE: <SettingOutlined />,
  BACKUP: <SafetyOutlined />,
  RESTORE: <ReloadOutlined />,
  SYSTEM: <SettingOutlined />,
};

/** Action colors */
const actionColors: Record<AuditAction, string> = {
  CREATE: 'green',
  READ: 'blue',
  UPDATE: 'orange',
  DELETE: 'red',
  LOGIN: 'cyan',
  LOGOUT: 'default',
  LOGIN_FAILED: 'red',
  PASSWORD_CHANGE: 'purple',
  PASSWORD_RESET: 'purple',
  ROLE_CHANGE: 'magenta',
  STATUS_CHANGE: 'gold',
  ANALYZE: 'geekblue',
  EXPORT: 'lime',
  IMPORT: 'lime',
  PROVIDER_CONFIG: 'volcano',
  SETTINGS_CHANGE: 'volcano',
  BACKUP: 'cyan',
  RESTORE: 'cyan',
  SYSTEM: 'default',
};

/** Status colors */
const statusColors: Record<AuditStatus, string> = {
  success: 'green',
  failure: 'red',
  warning: 'orange',
};

/** Alert severity colors */
const severityColors: Record<string, string> = {
  low: 'default',
  medium: 'orange',
  high: 'red',
  critical: 'magenta',
};

/** Chart colors */
const CHART_COLORS = ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1', '#13c2c2', '#eb2f96', '#fa8c16'];

/**
 * Audit Statistics Section
 */
const AuditStatsSection: React.FC = () => {
  const { t } = useTranslation();
  const { data: analytics, isLoading } = useAuditAnalytics();
  const { data: alerts } = useSecurityAlerts({ resolved: false });

  if (isLoading) {
    return <Skeleton active paragraph={{ rows: 2 }} />;
  }

  if (!analytics) {
    return null;
  }

  return (
    <Row gutter={[16, 16]} className="audit-stats">
      <Col xs={12} sm={6}>
        <Card>
          <Statistic
            title={t('admin.audit.total_logs', 'Total Logs')}
            value={analytics.totalLogs}
            prefix={<FileTextOutlined />}
          />
        </Card>
      </Col>
      <Col xs={12} sm={6}>
        <Card>
          <Statistic
            title={t('admin.audit.success_rate', 'Success Rate')}
            value={analytics.successRate}
            suffix="%"
            precision={1}
            valueStyle={{ color: '#52c41a' }}
            prefix={<CheckCircleOutlined />}
          />
        </Card>
      </Col>
      <Col xs={12} sm={6}>
        <Card>
          <Statistic
            title={t('admin.audit.failure_rate', 'Failure Rate')}
            value={analytics.failureRate}
            suffix="%"
            precision={1}
            valueStyle={{ color: '#f5222d' }}
            prefix={<CloseCircleOutlined />}
          />
        </Card>
      </Col>
      <Col xs={12} sm={6}>
        <Card>
          <Statistic
            title={t('admin.audit.active_alerts', 'Active Alerts')}
            value={alerts?.total || 0}
            valueStyle={{ color: alerts?.total ? '#f5222d' : undefined }}
            prefix={<AlertOutlined />}
          />
        </Card>
      </Col>
    </Row>
  );
};

/**
 * Audit Analytics Charts
 */
const AuditAnalyticsSection: React.FC = () => {
  const { t } = useTranslation();
  const { data: analytics } = useAuditAnalytics();
  const { data: loginPatterns } = useLoginPatterns();

  if (!analytics) {
    return <Skeleton active paragraph={{ rows: 4 }} />;
  }

  const actionData = analytics.actionDistribution.map((item, index) => ({
    name: item.action,
    value: item.count,
    color: CHART_COLORS[index % CHART_COLORS.length],
  }));

  const activeUsersData = analytics.mostActiveUsers.slice(0, 5);

  return (
    <Row gutter={[16, 16]}>
      {/* Action Distribution */}
      <Col xs={24} md={12}>
        <Card title={<><PieChartOutlined /> {t('admin.audit.action_distribution', 'Action Distribution')}</>} size="small">
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={actionData}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={90}
                paddingAngle={2}
                dataKey="value"
                label={({ name, percent }: any) => `${name || ''} (${((percent || 0) * 100).toFixed(0)}%)`}
                labelLine={false}
              >
                {actionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <RechartsTooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </Col>

      {/* Most Active Users */}
      <Col xs={24} md={12}>
        <Card title={<><BarChartOutlined /> {t('admin.audit.most_active', 'Most Active Users')}</>} size="small">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={activeUsersData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" fontSize={12} />
              <YAxis dataKey="username" type="category" width={100} fontSize={12} />
              <RechartsTooltip />
              <Bar dataKey="count" fill="#1890ff" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </Col>

      {/* Failed Actions Timeline */}
      <Col xs={24}>
        <Card title={<><AreaChart /> {t('admin.audit.failed_actions', 'Failed Actions Timeline')}</>} size="small">
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={analytics.failedActionsTimeline}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" fontSize={12} />
              <YAxis fontSize={12} />
              <RechartsTooltip />
              <Area type="monotone" dataKey="count" stroke="#f5222d" fill="#fff1f0" />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      </Col>
    </Row>
  );
};

/**
 * Security Alerts Section
 */
const SecurityAlertsSection: React.FC = () => {
  const { t } = useTranslation();
  const { data: alertsData, isLoading } = useSecurityAlerts({ resolved: false });
  const resolveAlert = useResolveAlert();

  if (isLoading) {
    return <Skeleton active paragraph={{ rows: 3 }} />;
  }

  if (!alertsData?.items.length) {
    return (
      <Empty
        image={<SafetyOutlined style={{ fontSize: 48, color: '#52c41a' }} />}
        description={t('admin.audit.no_alerts', 'No active security alerts')}
      />
    );
  }

  return (
    <List
      dataSource={alertsData.items}
      renderItem={(alert: SecurityAlert) => (
        <List.Item
          actions={[
            <Button
              size="small"
              onClick={() => resolveAlert.mutate({ alertId: alert.id })}
              loading={resolveAlert.isPending}
            >
              {t('admin.audit.resolve', 'Resolve')}
            </Button>,
          ]}
        >
          <List.Item.Meta
            avatar={
              <Badge count={<ExclamationCircleOutlined style={{ color: severityColors[alert.severity] === 'magenta' ? '#eb2f96' : severityColors[alert.severity] === 'red' ? '#f5222d' : '#faad14' }} />}>
                <AlertOutlined style={{ fontSize: 24 }} />
              </Badge>
            }
            title={
              <Space>
                <Tag color={severityColors[alert.severity]}>{alert.severity.toUpperCase()}</Tag>
                <Text strong>{alert.type.replace(/_/g, ' ')}</Text>
              </Space>
            }
            description={
              <Space direction="vertical" size={4}>
                <Text>{alert.description}</Text>
                <Space>
                  {alert.username && <Tag icon={<UserOutlined />}>{alert.username}</Tag>}
                  {alert.ipAddress && <Tag icon={<GlobalOutlined />}>{alert.ipAddress}</Tag>}
                  <Text type="secondary">{new Date(alert.timestamp).toLocaleString()}</Text>
                </Space>
              </Space>
            }
          />
        </List.Item>
      )}
    />
  );
};

/**
 * Audit Log Detail Modal
 */
const AuditLogDetailModal: React.FC<{
  logId: string | null;
  open: boolean;
  onClose: () => void;
}> = ({ logId, open, onClose }) => {
  const { t } = useTranslation();
  const { data: log, isLoading } = useAuditLog(logId || '');

  if (!logId) return null;

  return (
    <Modal
      title={<><FileTextOutlined /> {t('admin.audit.log_details', 'Audit Log Details')}</>}
      open={open}
      onCancel={onClose}
      footer={[
        <Button key="close" onClick={onClose}>
          {t('common.close', 'Close')}
        </Button>,
      ]}
      width={700}
    >
      {isLoading ? (
        <Skeleton active paragraph={{ rows: 6 }} />
      ) : log ? (
        <div className="audit-log-detail">
          <Descriptions bordered column={2} size="small">
            <Descriptions.Item label={t('admin.audit.timestamp', 'Timestamp')} span={2}>
              <ClockCircleOutlined /> {new Date(log.timestamp).toLocaleString()}
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.audit.user', 'User')}>
              <Space>
                <UserOutlined />
                {log.username}
              </Space>
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.audit.email', 'Email')}>
              {log.userEmail}
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.audit.action', 'Action')}>
              <Tag color={actionColors[log.action]} icon={actionIcons[log.action]}>
                {log.action}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.audit.status', 'Status')}>
              <Tag color={statusColors[log.status]}>
                {log.status === 'success' ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
                {' '}{log.status}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.audit.resource', 'Resource')}>
              {log.resource} {log.resourceId && `(${log.resourceId})`}
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.audit.resource_name', 'Resource Name')}>
              {log.resourceName || '-'}
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.audit.ip_address', 'IP Address')}>
              <GlobalOutlined /> {log.ipAddress}
            </Descriptions.Item>
            <Descriptions.Item label={t('admin.audit.location', 'Location')}>
              <EnvironmentOutlined /> {log.location || '-'}
            </Descriptions.Item>
            {log.duration && (
              <Descriptions.Item label={t('admin.audit.duration', 'Duration')}>
                {log.duration}ms
              </Descriptions.Item>
            )}
            {log.details && (
              <Descriptions.Item label={t('admin.audit.details', 'Details')} span={2}>
                {log.details}
              </Descriptions.Item>
            )}
          </Descriptions>

          {(log.oldValue || log.newValue) && (
            <>
              <Title level={5} style={{ marginTop: 16 }}>{t('admin.audit.changes', 'Changes')}</Title>
              <Row gutter={16}>
                {log.oldValue && (
                  <Col span={12}>
                    <Card size="small" title={t('admin.audit.old_value', 'Old Value')}>
                      <pre className="json-viewer">
                        {JSON.stringify(log.oldValue, null, 2)}
                      </pre>
                    </Card>
                  </Col>
                )}
                {log.newValue && (
                  <Col span={log.oldValue ? 12 : 24}>
                    <Card size="small" title={t('admin.audit.new_value', 'New Value')}>
                      <pre className="json-viewer">
                        {JSON.stringify(log.newValue, null, 2)}
                      </pre>
                    </Card>
                  </Col>
                )}
              </Row>
            </>
          )}
        </div>
      ) : (
        <Empty description={t('admin.audit.log_not_found', 'Log not found')} />
      )}
    </Modal>
  );
};

/**
 * Main Audit Logs Component
 */
export const AuditLogs: React.FC = () => {
  const { t } = useTranslation();
  const [selectedLogId, setSelectedLogId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('logs');

  const {
    auditFilters,
    auditPagination,
    setAuditFilters,
    resetAuditFilters,
    setAuditPagination,
  } = useAdminStore();

  const { data, isLoading, refetch, isFetching } = useAuditLogs();
  const exportLogs = useExportAuditLogs();

  // Handle search
  const handleSearch = useCallback((value: string) => {
    setAuditFilters({ search: value });
  }, [setAuditFilters]);

  // Export menu
  const exportMenuItems: MenuProps['items'] = [
    { key: 'csv', label: 'Export as CSV', onClick: () => exportLogs.mutate('csv') },
    { key: 'json', label: 'Export as JSON', onClick: () => exportLogs.mutate('json') },
    { key: 'pdf', label: 'Export as PDF', onClick: () => exportLogs.mutate('pdf') },
  ];

  // Table columns
  const columns: TableProps<AuditLog>['columns'] = [
    {
      title: t('admin.audit.timestamp', 'Timestamp'),
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 160,
      sorter: true,
      render: (time: string) => (
        <Tooltip title={new Date(time).toLocaleString()}>
          <Text>{new Date(time).toLocaleString()}</Text>
        </Tooltip>
      ),
    },
    {
      title: t('admin.audit.user', 'User'),
      key: 'user',
      width: 150,
      render: (_, record) => (
        <Space>
          <UserOutlined />
          <div>
            <Text strong>{record.username}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 11 }}>{record.userEmail}</Text>
          </div>
        </Space>
      ),
    },
    {
      title: t('admin.audit.action', 'Action'),
      dataIndex: 'action',
      key: 'action',
      width: 130,
      filters: Object.keys(actionColors).map(action => ({ text: action, value: action })),
      render: (action: AuditAction) => (
        <Tag color={actionColors[action]} icon={actionIcons[action]}>
          {action}
        </Tag>
      ),
    },
    {
      title: t('admin.audit.resource', 'Resource'),
      key: 'resource',
      width: 180,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Tag>{record.resource}</Tag>
          {record.resourceName && <Text type="secondary" style={{ fontSize: 12 }}>{record.resourceName}</Text>}
        </Space>
      ),
    },
    {
      title: t('admin.audit.ip_address', 'IP Address'),
      dataIndex: 'ipAddress',
      key: 'ipAddress',
      width: 130,
      render: (ip: string, record) => (
        <Tooltip title={record.location}>
          <Space>
            <GlobalOutlined />
            <Text>{ip}</Text>
          </Space>
        </Tooltip>
      ),
    },
    {
      title: t('admin.audit.status', 'Status'),
      dataIndex: 'status',
      key: 'status',
      width: 100,
      filters: [
        { text: 'Success', value: 'success' },
        { text: 'Failure', value: 'failure' },
        { text: 'Warning', value: 'warning' },
      ],
      render: (status: AuditStatus) => (
        <Tag color={statusColors[status]} icon={status === 'success' ? <CheckCircleOutlined /> : <CloseCircleOutlined />}>
          {status}
        </Tag>
      ),
    },
    {
      title: '',
      key: 'actions',
      width: 60,
      render: (_, record) => (
        <Button
          type="text"
          icon={<EyeOutlined />}
          onClick={() => setSelectedLogId(record.id)}
        />
      ),
    },
  ];

  // Handle table change
  const handleTableChange: TableProps<AuditLog>['onChange'] = (pagination, filters) => {
    if (pagination.current && pagination.pageSize) {
      setAuditPagination({
        page: pagination.current,
        pageSize: pagination.pageSize,
      });
    }
    
    if (filters.action) {
      setAuditFilters({ action: (filters.action[0] as AuditAction) || 'all' });
    }
    if (filters.status) {
      setAuditFilters({ status: (filters.status[0] as AuditStatus) || 'all' });
    }
  };

  const tabItems: TabsProps['items'] = [
    {
      key: 'logs',
      label: <><FileTextOutlined /> {t('admin.audit.logs', 'Logs')}</>,
      children: (
        <>
          {/* Filters */}
          <Card className="filters-card">
            <Row gutter={[16, 16]} align="middle">
              <Col xs={24} sm={12} md={6}>
                <Search
                  placeholder={t('admin.audit.search_placeholder', 'Search logs...')}
                  allowClear
                  onSearch={handleSearch}
                  defaultValue={auditFilters.search}
                />
              </Col>
              <Col xs={12} sm={6} md={4}>
                <Select
                  placeholder={t('admin.audit.action', 'Action')}
                  value={auditFilters.action}
                  onChange={(value) => setAuditFilters({ action: value })}
                  options={[
                    { value: 'all', label: 'All Actions' },
                    ...Object.keys(actionColors).map(action => ({ value: action, label: action })),
                  ]}
                  style={{ width: '100%' }}
                />
              </Col>
              <Col xs={12} sm={6} md={4}>
                <Select
                  placeholder={t('admin.audit.resource', 'Resource')}
                  value={auditFilters.resource}
                  onChange={(value) => setAuditFilters({ resource: value })}
                  options={[
                    { value: 'all', label: 'All Resources' },
                    { value: 'user', label: 'User' },
                    { value: 'project', label: 'Project' },
                    { value: 'analysis', label: 'Analysis' },
                    { value: 'provider', label: 'Provider' },
                    { value: 'settings', label: 'Settings' },
                  ]}
                  style={{ width: '100%' }}
                />
              </Col>
              <Col xs={12} sm={6} md={4}>
                <Select
                  placeholder={t('admin.audit.status', 'Status')}
                  value={auditFilters.status}
                  onChange={(value) => setAuditFilters({ status: value })}
                  options={[
                    { value: 'all', label: 'All Status' },
                    { value: 'success', label: 'Success' },
                    { value: 'failure', label: 'Failure' },
                    { value: 'warning', label: 'Warning' },
                  ]}
                  style={{ width: '100%' }}
                />
              </Col>
              <Col flex="auto" style={{ textAlign: 'right' }}>
                <Space>
                  <Button icon={<FilterOutlined />} onClick={resetAuditFilters}>
                    {t('admin.audit.reset', 'Reset')}
                  </Button>
                  <Dropdown menu={{ items: exportMenuItems }}>
                    <Button icon={<DownloadOutlined />} loading={exportLogs.isPending}>
                      {t('admin.audit.export', 'Export')}
                    </Button>
                  </Dropdown>
                  <Button icon={<ReloadOutlined />} onClick={() => refetch()} loading={isFetching} />
                </Space>
              </Col>
            </Row>
          </Card>

          {/* Logs Table */}
          <Card className="logs-table-card">
            <Table
              columns={columns}
              dataSource={data?.items || []}
              rowKey="id"
              loading={isLoading}
              onChange={handleTableChange}
              pagination={{
                current: auditPagination.page,
                pageSize: auditPagination.pageSize,
                total: data?.total || 0,
                showSizeChanger: true,
                showTotal: (total) => t('admin.audit.total_logs_count', `Total ${total} logs`),
                pageSizeOptions: ['20', '50', '100', '200'],
              }}
              scroll={{ x: 1000 }}
              locale={{
                emptyText: <Empty description={t('admin.audit.no_logs', 'No audit logs found')} />,
              }}
            />
          </Card>
        </>
      ),
    },
    {
      key: 'analytics',
      label: <><BarChartOutlined /> {t('admin.audit.analytics', 'Analytics')}</>,
      children: <AuditAnalyticsSection />,
    },
    {
      key: 'alerts',
      label: <><AlertOutlined /> {t('admin.audit.security_alerts', 'Security Alerts')}</>,
      children: (
        <Card>
          <SecurityAlertsSection />
        </Card>
      ),
    },
  ];

  return (
    <div className="audit-logs" role="main" aria-label={t('admin.audit.title', 'Audit Logs')}>
      <div className="page-header">
        <Title level={2}><AuditOutlined /> {t('admin.audit.title', 'Audit Logs')}</Title>
      </div>

      {/* Statistics */}
      <AuditStatsSection />

      {/* Main Content */}
      <Card className="audit-content-card">
        <Tabs activeKey={activeTab} onChange={setActiveTab} items={tabItems} />
      </Card>

      {/* Log Detail Modal */}
      <AuditLogDetailModal
        logId={selectedLogId}
        open={!!selectedLogId}
        onClose={() => setSelectedLogId(null)}
      />
    </div>
  );
};

export default AuditLogs;
