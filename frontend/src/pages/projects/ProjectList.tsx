/**
 * Projects List Page
 * 
 * Displays all user projects with filtering, sorting, search, and pagination.
 * Supports both table and card view modes.
 */

import React, { useState, useCallback, useMemo, memo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  Table,
  Button,
  Space,
  Typography,
  Tag,
  Input,
  Select,
  Modal,
  Dropdown,
  Badge,
  Avatar,
  Tooltip,
  Empty,
  Row,
  Col,
  Statistic,
  Segmented,
  Skeleton,
  Alert,
} from 'antd';
import type { MenuProps, TableProps } from 'antd';
import {
  PlusOutlined,
  SearchOutlined,
  ReloadOutlined,
  MoreOutlined,
  FolderOutlined,
  ClockCircleOutlined,
  PlayCircleOutlined,
  EditOutlined,
  DeleteOutlined,
  SettingOutlined,
  AppstoreOutlined,
  UnorderedListOutlined,
  InboxOutlined,
  FilterOutlined,
  CheckCircleOutlined,
  PauseCircleOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useProjectStore, type Project, type ProjectStatus, type ViewMode } from '../../store/projectStore';
import { useProjects, useDeleteProject, useArchiveProject, useRestoreProject } from '../../hooks/useProjects';
import './ProjectList.css';

const { Title, Text, Paragraph } = Typography;
const { Search } = Input;

/** Language colors (GitHub-style) */
const languageColors: Record<string, string> = {
  python: '#3572A5',
  javascript: '#f1e05a',
  typescript: '#2b7489',
  java: '#b07219',
  go: '#00ADD8',
  rust: '#dea584',
  cpp: '#f34b7d',
  c: '#555555',
  csharp: '#178600',
  ruby: '#701516',
  php: '#4F5D95',
  swift: '#ffac45',
  kotlin: '#F18E33',
  scala: '#c22d40',
};

/** Status badge configuration */
const statusConfig: Record<ProjectStatus, { color: string; icon: React.ReactNode; text: string }> = {
  active: { color: 'processing', icon: <CheckCircleOutlined />, text: 'Active' },
  inactive: { color: 'default', icon: <PauseCircleOutlined />, text: 'Inactive' },
  archived: { color: 'warning', icon: <InboxOutlined />, text: 'Archived' },
};

/** Project Card Component */
const ProjectCard = memo(({ 
  project, 
  onView, 
  onSettings, 
  onAnalyze,
  onDelete,
  onArchive,
}: {
  project: Project;
  onView: (id: string) => void;
  onSettings: (id: string) => void;
  onAnalyze: (id: string) => void;
  onDelete: (project: Project) => void;
  onArchive: (project: Project) => void;
}) => {
  const { t } = useTranslation();
  const status = statusConfig[project.status] || statusConfig.active;
  
  return (
    <Card 
      className="project-card"
      hoverable
      actions={[
        <Tooltip title={t('projects.analyze', 'Analyze')} key="analyze">
          <Button type="text" icon={<PlayCircleOutlined />} onClick={() => onAnalyze(project.id)} />
        </Tooltip>,
        <Tooltip title={t('projects.settings', 'Settings')} key="settings">
          <Button type="text" icon={<SettingOutlined />} onClick={() => onSettings(project.id)} />
        </Tooltip>,
        <Dropdown
          key="more"
          menu={{
            items: [
              {
                key: 'archive',
                icon: <InboxOutlined />,
                label: project.status === 'archived' 
                  ? t('projects.restore', 'Restore') 
                  : t('projects.archive', 'Archive'),
                onClick: () => onArchive(project),
              },
              { type: 'divider' },
              {
                key: 'delete',
                icon: <DeleteOutlined />,
                label: t('common.delete', 'Delete'),
                danger: true,
                onClick: () => onDelete(project),
              },
            ],
          }}
          trigger={['click']}
        >
          <Button type="text" icon={<MoreOutlined />} />
        </Dropdown>,
      ]}
    >
      <Card.Meta
        avatar={
          <Avatar 
            shape="square" 
            size={48}
            style={{ backgroundColor: languageColors[project.language] || '#666' }}
            icon={<FolderOutlined />}
          />
        }
        title={
          <Space>
            <Text 
              strong 
              style={{ cursor: 'pointer' }} 
              onClick={() => onView(project.id)}
            >
              {project.name}
            </Text>
            <Badge status={status.color as any} />
          </Space>
        }
        description={
          <Paragraph 
            ellipsis={{ rows: 2 }} 
            type="secondary"
            style={{ marginBottom: 8 }}
          >
            {project.description || t('projects.no_description', 'No description')}
          </Paragraph>
        }
      />
      <div className="project-card-footer">
        <Space size="small" wrap>
          <Tag color={languageColors[project.language] || 'default'}>
            {project.language}
          </Tag>
          {project.framework && <Tag>{project.framework}</Tag>}
        </Space>
        <div className="project-card-meta">
          <Space>
            <ClockCircleOutlined />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {new Date(project.updated_at).toLocaleDateString()}
            </Text>
          </Space>
        </div>
      </div>
      {project.stats && (
        <div className="project-card-stats">
          <Statistic 
            title={t('projects.analyses', 'Analyses')} 
            value={project.stats.total_analyses} 
            valueStyle={{ fontSize: 14 }}
          />
          <Statistic 
            title={t('projects.issues', 'Issues')} 
            value={project.stats.total_issues - project.stats.resolved_issues} 
            valueStyle={{ fontSize: 14, color: project.stats.total_issues - project.stats.resolved_issues > 0 ? '#faad14' : '#52c41a' }}
          />
        </div>
      )}
    </Card>
  );
});

ProjectCard.displayName = 'ProjectCard';

/**
 * Projects List Page Component
 */
export const ProjectList: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  
  // Store state
  const { 
    viewMode, 
    filters, 
    pagination,
    setViewMode,
    setFilters,
    setPagination,
    resetFilters,
  } = useProjectStore();
  
  // Local state for delete confirmation
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<Project | null>(null);
  
  // Query hooks
  const { data, isLoading, isError, error, refetch, isFetching } = useProjects(filters, pagination);
  const deleteProject = useDeleteProject();
  const archiveProject = useArchiveProject();
  const restoreProject = useRestoreProject();
  
  // Computed values - memoize to prevent unnecessary re-renders
  const projects = useMemo(() => data?.items || [], [data?.items]);
  const total = data?.total || 0;
  
  // Get unique languages for filter
  const availableLanguages = useMemo(() => {
    const langs = [...new Set(projects.map(p => p.language))];
    return langs.sort();
  }, [projects]);
  
  // Handlers
  const handleSearch = useCallback((value: string) => {
    setFilters({ search: value });
  }, [setFilters]);
  
  const handleStatusFilter = useCallback((value: string) => {
    setFilters({ status: value as ProjectStatus | 'all' });
  }, [setFilters]);
  
  const handleLanguageFilter = useCallback((value: string) => {
    setFilters({ language: value });
  }, [setFilters]);
  
  const handleSortChange = useCallback((field: string) => {
    const newOrder = filters.sortField === field && filters.sortOrder === 'asc' ? 'desc' : 'asc';
    setFilters({ sortField: field as any, sortOrder: newOrder });
  }, [filters.sortField, filters.sortOrder, setFilters]);
  
  const handlePageChange = useCallback((page: number, pageSize: number) => {
    setPagination({ page, pageSize });
  }, [setPagination]);
  
  const handleView = useCallback((id: string) => {
    navigate(`/review/${id}`);
  }, [navigate]);
  
  const handleSettings = useCallback((id: string) => {
    navigate(`/projects/${id}/settings`);
  }, [navigate]);
  
  const handleAnalyze = useCallback((id: string) => {
    navigate(`/review/${id}`);
  }, [navigate]);
  
  const handleDelete = useCallback((project: Project) => {
    setProjectToDelete(project);
    setDeleteModalVisible(true);
  }, []);
  
  const confirmDelete = useCallback(() => {
    if (projectToDelete) {
      deleteProject.mutate(projectToDelete.id);
      setDeleteModalVisible(false);
      setProjectToDelete(null);
    }
  }, [projectToDelete, deleteProject]);
  
  const handleArchive = useCallback((project: Project) => {
    if (project.status === 'archived') {
      restoreProject.mutate(project.id);
    } else {
      archiveProject.mutate(project.id);
    }
  }, [archiveProject, restoreProject]);
  
  // Project action menu for table
  const getProjectActions = useCallback((project: Project): MenuProps['items'] => [
    {
      key: 'analyze',
      icon: <PlayCircleOutlined />,
      label: t('projects.analyze', 'Analyze'),
      onClick: () => handleAnalyze(project.id),
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: t('projects.settings', 'Settings'),
      onClick: () => handleSettings(project.id),
    },
    { type: 'divider' },
    {
      key: 'edit',
      icon: <EditOutlined />,
      label: t('common.edit', 'Edit'),
      onClick: () => handleSettings(project.id),
    },
    {
      key: 'archive',
      icon: <InboxOutlined />,
      label: project.status === 'archived' 
        ? t('projects.restore', 'Restore')
        : t('projects.archive', 'Archive'),
      onClick: () => handleArchive(project),
    },
    { type: 'divider' },
    {
      key: 'delete',
      icon: <DeleteOutlined />,
      label: t('common.delete', 'Delete'),
      danger: true,
      onClick: () => handleDelete(project),
    },
  ], [t, handleAnalyze, handleSettings, handleArchive, handleDelete]);
  
  // Table columns
  const columns: TableProps<Project>['columns'] = [
    {
      title: t('projects.name', 'Project'),
      dataIndex: 'name',
      key: 'name',
      sorter: true,
      sortOrder: filters.sortField === 'name' ? (filters.sortOrder === 'asc' ? 'ascend' : 'descend') : null,
      render: (name: string, record: Project) => (
        <Space>
          <Avatar
            shape="square"
            size="small"
            style={{ backgroundColor: languageColors[record.language] || '#666' }}
            icon={<FolderOutlined />}
          />
          <div>
            <Text 
              strong 
              style={{ cursor: 'pointer' }} 
              onClick={() => handleView(record.id)}
            >
              {name}
            </Text>
            {record.description && (
              <Paragraph 
                type="secondary" 
                ellipsis 
                style={{ margin: 0, fontSize: 12 }}
              >
                {record.description}
              </Paragraph>
            )}
          </div>
        </Space>
      ),
    },
    {
      title: t('projects.language', 'Language'),
      dataIndex: 'language',
      key: 'language',
      width: 120,
      render: (lang: string) => (
        <Tag color={languageColors[lang] || 'default'}>
          {lang}
        </Tag>
      ),
    },
    {
      title: t('projects.status', 'Status'),
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: ProjectStatus) => {
        const config = statusConfig[status] || statusConfig.active;
        return (
          <Badge 
            status={config.color as any} 
            text={config.text}
          />
        );
      },
    },
    {
      title: t('projects.created', 'Created'),
      dataIndex: 'created_at',
      key: 'created_at',
      width: 120,
      sorter: true,
      sortOrder: filters.sortField === 'created_at' ? (filters.sortOrder === 'asc' ? 'ascend' : 'descend') : null,
      render: (date: string) => (
        <Tooltip title={new Date(date).toLocaleString()}>
          {new Date(date).toLocaleDateString()}
        </Tooltip>
      ),
    },
    {
      title: t('projects.last_analyzed', 'Last Analyzed'),
      dataIndex: 'last_analyzed_at',
      key: 'last_analyzed_at',
      width: 130,
      sorter: true,
      sortOrder: filters.sortField === 'last_analyzed_at' ? (filters.sortOrder === 'asc' ? 'ascend' : 'descend') : null,
      render: (date: string) => date ? (
        <Tooltip title={new Date(date).toLocaleString()}>
          <Space>
            <ClockCircleOutlined />
            {new Date(date).toLocaleDateString()}
          </Space>
        </Tooltip>
      ) : (
        <Text type="secondary">-</Text>
      ),
    },
    {
      title: '',
      key: 'actions',
      width: 50,
      fixed: 'right',
      render: (_: unknown, record: Project) => (
        <Dropdown menu={{ items: getProjectActions(record) }} trigger={['click']}>
          <Button type="text" icon={<MoreOutlined />} />
        </Dropdown>
      ),
    },
  ];
  
  // Table change handler
  const handleTableChange: TableProps<Project>['onChange'] = (_, __, sorter) => {
    if (!Array.isArray(sorter) && sorter.field) {
      const field = sorter.field as string;
      const order = sorter.order === 'ascend' ? 'asc' : 'desc';
      setFilters({ sortField: field as any, sortOrder: order });
    }
  };
  
  // Calculate stats
  const stats = useMemo(() => ({
    total: total,
    active: projects.filter(p => p.status === 'active').length,
    recentAnalyses: projects.filter(p => {
      if (!p.last_analyzed_at) return false;
      const weekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
      return new Date(p.last_analyzed_at).getTime() > weekAgo;
    }).length,
  }), [projects, total]);
  
  return (
    <div className="project-list-container" role="main" aria-label={t('projects.title', 'Projects')}>
      {/* Header */}
      <div className="project-list-header">
        <Row justify="space-between" align="middle">
          <Col>
            <Title level={2} style={{ margin: 0 }}>
              {t('projects.title', 'Projects')}
            </Title>
            <Text type="secondary">
              {t('projects.subtitle', 'Manage your code review projects')}
            </Text>
          </Col>
          <Col>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => navigate('/projects/new')}
              size="large"
            >
              {t('projects.new', 'New Project')}
            </Button>
          </Col>
        </Row>
      </div>

      {/* Stats */}
      <Row gutter={16} className="project-stats">
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title={t('projects.total', 'Total Projects')}
              value={stats.total}
              prefix={<FolderOutlined />}
              loading={isLoading}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title={t('projects.active', 'Active Projects')}
              value={stats.active}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
              loading={isLoading}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title={t('projects.recent_analyses', 'Recent Analyses')}
              value={stats.recentAnalyses}
              prefix={<PlayCircleOutlined />}
              suffix={t('projects.this_week', 'this week')}
              loading={isLoading}
            />
          </Card>
        </Col>
      </Row>

      {/* Filters */}
      <Card className="project-filters">
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} sm={12} md={6}>
            <Search
              placeholder={t('projects.search', 'Search projects...')}
              prefix={<SearchOutlined />}
              value={filters.search}
              onChange={(e) => handleSearch(e.target.value)}
              allowClear
              aria-label={t('projects.search', 'Search projects')}
            />
          </Col>
          <Col xs={12} sm={6} md={4}>
            <Select
              style={{ width: '100%' }}
              placeholder={t('projects.filter_status', 'Status')}
              value={filters.status}
              onChange={handleStatusFilter}
              options={[
                { value: 'all', label: t('projects.all_status', 'All Status') },
                { value: 'active', label: t('projects.status_active', 'Active') },
                { value: 'inactive', label: t('projects.status_inactive', 'Inactive') },
                { value: 'archived', label: t('projects.status_archived', 'Archived') },
              ]}
              aria-label={t('projects.filter_status', 'Filter by status')}
            />
          </Col>
          <Col xs={12} sm={6} md={4}>
            <Select
              style={{ width: '100%' }}
              placeholder={t('projects.filter_language', 'Language')}
              value={filters.language}
              onChange={handleLanguageFilter}
              options={[
                { value: 'all', label: t('projects.all_languages', 'All Languages') },
                ...availableLanguages.map(lang => ({ value: lang, label: lang })),
              ]}
              aria-label={t('projects.filter_language', 'Filter by language')}
            />
          </Col>
          <Col flex="auto">
            <Row justify="end" gutter={8}>
              <Col>
                <Tooltip title={t('projects.reset_filters', 'Reset filters')}>
                  <Button 
                    icon={<FilterOutlined />} 
                    onClick={resetFilters}
                    disabled={filters.search === '' && filters.status === 'all' && filters.language === 'all'}
                  >
                    {t('projects.reset', 'Reset')}
                  </Button>
                </Tooltip>
              </Col>
              <Col>
                <Tooltip title={t('common.refresh', 'Refresh')}>
                  <Button 
                    icon={<ReloadOutlined spin={isFetching} />} 
                    onClick={() => refetch()}
                    loading={isFetching}
                  />
                </Tooltip>
              </Col>
              <Col>
                <Segmented
                  value={viewMode}
                  onChange={(value) => setViewMode(value as ViewMode)}
                  options={[
                    { value: 'table', icon: <UnorderedListOutlined />, label: '' },
                    { value: 'card', icon: <AppstoreOutlined />, label: '' },
                  ]}
                  aria-label={t('projects.view_mode', 'View mode')}
                />
              </Col>
            </Row>
          </Col>
        </Row>
      </Card>

      {/* Error State */}
      {isError && (
        <Alert
          message={t('projects.error_loading', 'Error loading projects')}
          description={(error as Error)?.message || t('common.unknown_error', 'An unknown error occurred')}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={() => refetch()}>
              {t('common.retry', 'Retry')}
            </Button>
          }
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Projects List */}
      <Card className="project-content">
        {isLoading ? (
          <Skeleton active paragraph={{ rows: 10 }} />
        ) : projects.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={
              filters.search || filters.status !== 'all' || filters.language !== 'all'
                ? t('projects.no_results', 'No projects match your filters')
                : t('projects.no_projects', 'No projects yet')
            }
          >
            {!filters.search && filters.status === 'all' && filters.language === 'all' && (
              <Button type="primary" onClick={() => navigate('/projects/new')}>
                {t('projects.create_first', 'Create your first project')}
              </Button>
            )}
          </Empty>
        ) : viewMode === 'table' ? (
          <Table
            columns={columns}
            dataSource={projects}
            rowKey="id"
            loading={isFetching}
            onChange={handleTableChange}
            pagination={{
              current: pagination.page,
              pageSize: pagination.pageSize,
              total: total,
              showSizeChanger: true,
              showTotal: (total) => t('projects.showing', `Showing ${total} projects`),
              onChange: handlePageChange,
            }}
            scroll={{ x: 800 }}
          />
        ) : (
          <>
            <Row gutter={[16, 16]}>
              {projects.map((project) => (
                <Col key={project.id} xs={24} sm={12} lg={8} xl={6}>
                  <ProjectCard
                    project={project}
                    onView={handleView}
                    onSettings={handleSettings}
                    onAnalyze={handleAnalyze}
                    onDelete={handleDelete}
                    onArchive={handleArchive}
                  />
                </Col>
              ))}
            </Row>
            <Row justify="center" style={{ marginTop: 24 }}>
              <Col>
                <Table
                  dataSource={[]}
                  columns={[]}
                  pagination={{
                    current: pagination.page,
                    pageSize: pagination.pageSize,
                    total: total,
                    showSizeChanger: true,
                    showTotal: (total) => t('projects.showing', `Showing ${total} projects`),
                    onChange: handlePageChange,
                  }}
                />
              </Col>
            </Row>
          </>
        )}
      </Card>

      {/* Delete Confirmation Modal */}
      <Modal
        title={t('projects.delete_confirm', 'Delete Project?')}
        open={deleteModalVisible}
        onCancel={() => setDeleteModalVisible(false)}
        onOk={confirmDelete}
        okText={t('common.delete', 'Delete')}
        okType="danger"
        cancelText={t('common.cancel', 'Cancel')}
        confirmLoading={deleteProject.isPending}
      >
        <p>
          {t(
            'projects.delete_warning',
            `Are you sure you want to delete "${projectToDelete?.name}"? This action cannot be undone.`
          )}
        </p>
      </Modal>
    </div>
  );
};

export default ProjectList;
