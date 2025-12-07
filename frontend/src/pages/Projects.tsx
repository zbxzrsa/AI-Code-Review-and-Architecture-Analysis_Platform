/**
 * 项目管理页面 (Projects Page)
 * 
 * 功能描述:
 *   项目列表和管理页面，提供项目的创建、编辑、删除等功能。
 * 
 * 主要特性:
 *   - 项目列表展示
 *   - 搜索和筛选
 *   - 项目创建向导
 *   - 项目设置管理
 *   - 代码分析入口
 * 
 * 最后修改日期: 2024-12-07
 */

import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
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
  Form,
  message,
  Dropdown,
  Badge,
  Avatar,
  Tooltip,
  Empty,
  Row,
  Col,
  Statistic
} from 'antd';
import type { MenuProps } from 'antd';
import {
  PlusOutlined,
  SearchOutlined,
  MoreOutlined,
  FolderOutlined,
  CodeOutlined,
  ClockCircleOutlined,
  EditOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  SettingOutlined
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { apiService } from '../services/api';
import type { Project } from '../store/projectStore';
import './Projects.css';

const { Title, Text, Paragraph } = Typography;
const { Search } = Input;

// Language colors (GitHub-style)
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
  scala: '#c22d40'
};

export const Projects: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { projectId: _projectId } = useParams<{ projectId: string }>();
  
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchText, setSearchText] = useState('');
  const [languageFilter, setLanguageFilter] = useState<string>('all');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingProject, setEditingProject] = useState<Project | null>(null);
  const [form] = Form.useForm();

  // Fetch projects
  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const response = await apiService.projects.list({ limit: 100 });
        setProjects(response.data.items || []);
      } catch (error) {
        console.error('Failed to fetch projects:', error);
        // Mock data for development
        setProjects([
          {
            id: '1',
            name: 'backend-api',
            description: 'Main backend API service',
            language: 'python',
            framework: 'FastAPI',
            repository_url: 'https://github.com/org/backend-api',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            owner_id: '1',
            is_public: false,
            status: 'active' as const,
            settings: {
              auto_review: true,
              review_on_push: true,
              review_on_pr: true,
              severity_threshold: 'warning',
              enabled_rules: [],
              ignored_paths: ['node_modules', '.git']
            }
          },
          {
            id: '2',
            name: 'frontend-app',
            description: 'React frontend application',
            language: 'typescript',
            framework: 'React',
            repository_url: 'https://github.com/org/frontend-app',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            owner_id: '1',
            is_public: true,
            status: 'active' as const,
            settings: {
              auto_review: false,
              review_on_push: false,
              review_on_pr: true,
              severity_threshold: 'error',
              enabled_rules: [],
              ignored_paths: ['dist', 'build']
            }
          }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchProjects();
  }, []);

  // Filter projects
  const filteredProjects = projects.filter(project => {
    const matchesSearch = project.name.toLowerCase().includes(searchText.toLowerCase()) ||
      project.description?.toLowerCase().includes(searchText.toLowerCase());
    const matchesLanguage = languageFilter === 'all' || project.language === languageFilter;
    return matchesSearch && matchesLanguage;
  });

  // Get unique languages
  const languages = [...new Set(projects.map(p => p.language))];

  // Handle create/edit project
  const handleSubmit = async (values: any) => {
    try {
      if (editingProject) {
        await apiService.projects.update(editingProject.id, values);
        message.success(t('projects.updated', 'Project updated successfully'));
      } else {
        await apiService.projects.create(values);
        message.success(t('projects.created', 'Project created successfully'));
      }
      setIsModalOpen(false);
      form.resetFields();
      setEditingProject(null);
      // Refresh projects
      const response = await apiService.projects.list({ limit: 100 });
      setProjects(response.data.items || []);
    } catch (error) {
      message.error(t('projects.error', 'Operation failed'));
    }
  };

  // Handle delete project
  const handleDelete = async (project: Project) => {
    Modal.confirm({
      title: t('projects.delete_confirm', 'Delete Project?'),
      content: t('projects.delete_warning', `Are you sure you want to delete "${project.name}"? This action cannot be undone.`),
      okText: t('common.delete', 'Delete'),
      okType: 'danger',
      cancelText: t('common.cancel', 'Cancel'),
      onOk: async () => {
        try {
          await apiService.projects.delete(project.id);
          message.success(t('projects.deleted', 'Project deleted'));
          setProjects(projects.filter(p => p.id !== project.id));
        } catch (error) {
          message.error(t('projects.delete_error', 'Failed to delete project'));
        }
      }
    });
  };

  // Project action menu
  const getProjectActions = (project: Project): MenuProps['items'] => [
    {
      key: 'analyze',
      icon: <PlayCircleOutlined />,
      label: t('projects.analyze', 'Analyze'),
      onClick: () => navigate(`/review/${project.id}`)
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: t('projects.settings', 'Settings'),
      onClick: () => navigate(`/projects/${project.id}/settings`)
    },
    { type: 'divider' },
    {
      key: 'edit',
      icon: <EditOutlined />,
      label: t('common.edit', 'Edit'),
      onClick: () => {
        setEditingProject(project);
        form.setFieldsValue(project);
        setIsModalOpen(true);
      }
    },
    {
      key: 'delete',
      icon: <DeleteOutlined />,
      label: t('common.delete', 'Delete'),
      danger: true,
      onClick: () => handleDelete(project)
    }
  ];

  // Table columns
  const columns = [
    {
      title: t('projects.name', 'Project'),
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: Project) => (
        <Space>
          <Avatar
            shape="square"
            size="small"
            style={{ backgroundColor: languageColors[record.language] || '#666' }}
            icon={<FolderOutlined />}
          />
          <div>
            <Text strong style={{ cursor: 'pointer' }} onClick={() => navigate(`/review/${record.id}`)}>
              {name}
            </Text>
            {record.description && (
              <Paragraph type="secondary" ellipsis style={{ margin: 0, fontSize: 12 }}>
                {record.description}
              </Paragraph>
            )}
          </div>
        </Space>
      )
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
      )
    },
    {
      title: t('projects.framework', 'Framework'),
      dataIndex: 'framework',
      key: 'framework',
      width: 120,
      render: (framework: string) => framework || '-'
    },
    {
      title: t('projects.status', 'Status'),
      key: 'status',
      width: 100,
      render: (_: any, record: Project) => (
        <Badge
          status={record.settings?.auto_review ? 'processing' : 'default'}
          text={record.settings?.auto_review ? 'Active' : 'Manual'}
        />
      )
    },
    {
      title: t('projects.updated', 'Last Updated'),
      dataIndex: 'updated_at',
      key: 'updated_at',
      width: 150,
      render: (date: string) => (
        <Tooltip title={new Date(date).toLocaleString()}>
          <Space>
            <ClockCircleOutlined />
            {new Date(date).toLocaleDateString()}
          </Space>
        </Tooltip>
      )
    },
    {
      title: '',
      key: 'actions',
      width: 50,
      render: (_: any, record: Project) => (
        <Dropdown menu={{ items: getProjectActions(record) }} trigger={['click']}>
          <Button type="text" icon={<MoreOutlined />} />
        </Dropdown>
      )
    }
  ];

  return (
    <div className="projects-container">
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
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
              onClick={() => {
                setEditingProject(null);
                form.resetFields();
                setIsModalOpen(true);
              }}
            >
              {t('projects.new', 'New Project')}
            </Button>
          </Col>
        </Row>
      </div>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title={t('projects.total', 'Total Projects')}
              value={projects.length}
              prefix={<FolderOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title={t('projects.active', 'Active Reviews')}
              value={projects.filter(p => p.settings?.auto_review).length}
              prefix={<PlayCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title={t('projects.languages', 'Languages')}
              value={languages.length}
              prefix={<CodeOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Filters */}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col xs={24} sm={12} md={8}>
            <Search
              placeholder={t('projects.search', 'Search projects...')}
              prefix={<SearchOutlined />}
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              allowClear
            />
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Select
              style={{ width: '100%' }}
              placeholder={t('projects.filter_language', 'Filter by language')}
              value={languageFilter}
              onChange={setLanguageFilter}
              options={[
                { value: 'all', label: t('projects.all_languages', 'All Languages') },
                ...languages.map(lang => ({ value: lang, label: lang }))
              ]}
            />
          </Col>
        </Row>
      </Card>

      {/* Projects Table */}
      <Card>
        <Table
          columns={columns}
          dataSource={filteredProjects}
          rowKey="id"
          loading={loading}
          pagination={{
            total: filteredProjects.length,
            pageSize: 10,
            showSizeChanger: true,
            showTotal: (total) => t('projects.showing', `Showing ${total} projects`)
          }}
          locale={{
            emptyText: (
              <Empty
                description={t('projects.no_projects', 'No projects yet')}
              >
                <Button type="primary" onClick={() => setIsModalOpen(true)}>
                  {t('projects.create_first', 'Create your first project')}
                </Button>
              </Empty>
            )
          }}
        />
      </Card>

      {/* Create/Edit Modal */}
      <Modal
        title={editingProject ? t('projects.edit', 'Edit Project') : t('projects.create', 'Create Project')}
        open={isModalOpen}
        onCancel={() => {
          setIsModalOpen(false);
          form.resetFields();
          setEditingProject(null);
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{
            language: 'python',
            is_public: false,
            settings: {
              auto_review: false,
              review_on_push: false,
              review_on_pr: true,
              severity_threshold: 'warning'
            }
          }}
        >
          <Form.Item
            name="name"
            label={t('projects.form.name', 'Project Name')}
            rules={[{ required: true, message: t('projects.form.name_required', 'Please enter project name') }]}
          >
            <Input placeholder="my-project" />
          </Form.Item>

          <Form.Item
            name="description"
            label={t('projects.form.description', 'Description')}
          >
            <Input.TextArea rows={3} placeholder="Brief description of your project" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="language"
                label={t('projects.form.language', 'Language')}
                rules={[{ required: true }]}
              >
                <Select
                  options={[
                    { value: 'python', label: 'Python' },
                    { value: 'javascript', label: 'JavaScript' },
                    { value: 'typescript', label: 'TypeScript' },
                    { value: 'java', label: 'Java' },
                    { value: 'go', label: 'Go' },
                    { value: 'rust', label: 'Rust' },
                    { value: 'cpp', label: 'C++' },
                    { value: 'csharp', label: 'C#' }
                  ]}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="framework"
                label={t('projects.form.framework', 'Framework')}
              >
                <Input placeholder="e.g., FastAPI, React" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="repository_url"
            label={t('projects.form.repo', 'Repository URL')}
          >
            <Input placeholder="https://github.com/username/repo" />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
            <Space>
              <Button onClick={() => setIsModalOpen(false)}>
                {t('common.cancel', 'Cancel')}
              </Button>
              <Button type="primary" htmlType="submit">
                {editingProject ? t('common.save', 'Save') : t('common.create', 'Create')}
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default Projects;
