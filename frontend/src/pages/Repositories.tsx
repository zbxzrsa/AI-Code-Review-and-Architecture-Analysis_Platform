/**
 * Repository Management Page
 * 仓库管理页面
 * 
 * GitHub-inspired repository view with:
 * - Repository list with stats
 * - Branch management
 * - Clone/connect functionality
 * - Repository settings
 * - Real OAuth integration
 */

import React, { useState, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Tag,
  Button,
  Input,
  Select,
  Tooltip,
  Progress,
  Modal,
  Form,
  Dropdown,
  Statistic,
  Divider,
  message,
  Spin,
  Empty,
  List,
  Avatar,
} from 'antd';
import {
  GithubOutlined,
  GitlabOutlined,
  ForkOutlined,
  StarOutlined,
  StarFilled,
  EyeOutlined,
  BranchesOutlined,
  ClockCircleOutlined,
  CodeOutlined,
  LockOutlined,
  UnlockOutlined,
  SettingOutlined,
  PlusOutlined,
  SyncOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  LinkOutlined,
  MoreOutlined,
  CopyOutlined,
  DeleteOutlined,
  SafetyCertificateOutlined,
  PlayCircleOutlined,
  LoadingOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../services/api';

const { Title, Text } = Typography;

interface Repository {
  id: string;
  name: string;
  fullName: string;
  description?: string;
  provider: 'github' | 'gitlab' | 'bitbucket' | 'azure';
  visibility: 'public' | 'private';
  defaultBranch: string;
  language: string;
  languageColor: string;
  stars: number;
  forks: number;
  issues: number;
  lastAnalysis?: string;
  analysisStatus?: 'passing' | 'failing' | 'pending' | 'none';
  healthScore: number;
  branches: number;
  size: string;
  updatedAt: string;
  isStarred?: boolean;
}

const mockRepositories: Repository[] = [
  {
    id: 'repo_1',
    name: 'ai-code-review-platform',
    fullName: 'myorg/ai-code-review-platform',
    description: 'AI-powered code review and architecture analysis platform',
    provider: 'github',
    visibility: 'private',
    defaultBranch: 'main',
    language: 'TypeScript',
    languageColor: '#3178c6',
    stars: 128,
    forks: 24,
    issues: 12,
    lastAnalysis: '2024-03-01T15:30:00Z',
    analysisStatus: 'passing',
    healthScore: 92,
    branches: 8,
    size: '45.2 MB',
    updatedAt: '2024-03-01T15:30:00Z',
    isStarred: true,
  },
  {
    id: 'repo_2',
    name: 'backend-services',
    fullName: 'myorg/backend-services',
    description: 'FastAPI backend microservices',
    provider: 'github',
    visibility: 'private',
    defaultBranch: 'main',
    language: 'Python',
    languageColor: '#3572A5',
    stars: 56,
    forks: 12,
    issues: 5,
    lastAnalysis: '2024-03-01T12:00:00Z',
    analysisStatus: 'failing',
    healthScore: 78,
    branches: 5,
    size: '28.1 MB',
    updatedAt: '2024-03-01T12:00:00Z',
    isStarred: false,
  },
  {
    id: 'repo_3',
    name: 'mobile-app',
    fullName: 'myorg/mobile-app',
    description: 'React Native mobile application',
    provider: 'gitlab',
    visibility: 'private',
    defaultBranch: 'develop',
    language: 'JavaScript',
    languageColor: '#f1e05a',
    stars: 34,
    forks: 8,
    issues: 8,
    lastAnalysis: '2024-02-28T10:00:00Z',
    analysisStatus: 'passing',
    healthScore: 85,
    branches: 12,
    size: '156.4 MB',
    updatedAt: '2024-02-28T10:00:00Z',
    isStarred: true,
  },
  {
    id: 'repo_4',
    name: 'infrastructure',
    fullName: 'myorg/infrastructure',
    description: 'Kubernetes manifests and Terraform configs',
    provider: 'github',
    visibility: 'public',
    defaultBranch: 'main',
    language: 'HCL',
    languageColor: '#844FBA',
    stars: 245,
    forks: 67,
    issues: 3,
    analysisStatus: 'none',
    healthScore: 0,
    branches: 3,
    size: '2.4 MB',
    updatedAt: '2024-02-25T08:00:00Z',
    isStarred: false,
  },
];

const providerIcons = {
  github: <GithubOutlined />,
  gitlab: <GitlabOutlined />,
  bitbucket: <CodeOutlined />,
  azure: <CodeOutlined />,
};

export const Repositories: React.FC = () => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState('');
  const [filterProvider, setFilterProvider] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [connectModalOpen, setConnectModalOpen] = useState(false);
  const [selectRepoModalOpen, setSelectRepoModalOpen] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [form] = Form.useForm();

  // Fetch repositories from API
  const { data: reposData, isLoading: isLoadingRepos, refetch: refetchRepos } = useQuery({
    queryKey: ['repositories'],
    queryFn: async () => {
      const response = await apiService.repositories.list();
      return response.data;
    },
  });

  // Fetch OAuth connections
  const { data: connectionsData } = useQuery({
    queryKey: ['oauth-connections'],
    queryFn: async () => {
      const response = await apiService.oauth.getConnections();
      return response.data;
    },
  });

  // Fetch repositories from OAuth provider
  const { data: providerRepos, isLoading: isLoadingProviderRepos, refetch: fetchProviderRepos } = useQuery({
    queryKey: ['provider-repos', selectedProvider],
    queryFn: async () => {
      if (!selectedProvider) return null;
      const response = await apiService.oauth.listRepositories(selectedProvider);
      return response.data;
    },
    enabled: !!selectedProvider,
  });

  // Connect repository mutation
  const connectRepoMutation = useMutation({
    mutationFn: async (data: { provider: string; repo_full_name: string }) => {
      const response = await apiService.repositories.connect(data);
      return response.data;
    },
    onSuccess: () => {
      message.success('Repository connected successfully');
      queryClient.invalidateQueries({ queryKey: ['repositories'] });
      setSelectRepoModalOpen(false);
      setSelectedProvider(null);
    },
    onError: (error: any) => {
      message.error(error.response?.data?.detail || 'Failed to connect repository');
    },
  });

  // Create repository from URL mutation
  const createRepoMutation = useMutation({
    mutationFn: async (data: { url: string; name?: string }) => {
      const response = await apiService.repositories.create(data);
      return response.data;
    },
    onSuccess: () => {
      message.success('Repository added successfully');
      queryClient.invalidateQueries({ queryKey: ['repositories'] });
      form.resetFields();
      setConnectModalOpen(false);
    },
    onError: (error: any) => {
      message.error(error.response?.data?.detail || 'Failed to add repository');
    },
  });

  // Delete repository mutation
  const deleteRepoMutation = useMutation({
    mutationFn: async (id: string) => {
      const response = await apiService.repositories.delete(id);
      return response.data;
    },
    onSuccess: () => {
      message.success('Repository disconnected');
      queryClient.invalidateQueries({ queryKey: ['repositories'] });
    },
    onError: (error: any) => {
      message.error(error.response?.data?.detail || 'Failed to disconnect repository');
    },
  });

  // Sync repository mutation
  const syncRepoMutation = useMutation({
    mutationFn: async (id: string) => {
      const response = await apiService.repositories.sync(id);
      return response.data;
    },
    onSuccess: () => {
      message.success('Repository sync started');
      queryClient.invalidateQueries({ queryKey: ['repositories'] });
    },
  });

  // Map API data to UI format
  const repositories: Repository[] = (reposData?.items || mockRepositories).map((repo: any) => ({
    id: repo.id,
    name: repo.name,
    fullName: repo.full_name || repo.fullName || repo.name,
    description: repo.description,
    provider: repo.provider || 'github',
    visibility: repo.is_private ? 'private' : 'public',
    defaultBranch: repo.default_branch || 'main',
    language: repo.language || 'Unknown',
    languageColor: '#3178c6',
    stars: repo.stars || 0,
    forks: repo.forks || 0,
    issues: repo.issues || 0,
    lastAnalysis: repo.last_analyzed_at,
    analysisStatus: repo.analysis_status || 'none',
    healthScore: repo.health_score || 0,
    branches: repo.branches || 1,
    size: repo.size || '0 MB',
    updatedAt: repo.updated_at || new Date().toISOString(),
    isStarred: repo.is_starred || false,
  }));

  const toggleStar = (repoId: string) => {
    // TODO: Implement star API
    message.info('Star functionality coming soon');
  };

  const handleConnectProvider = async (provider: string) => {
    // Check if already connected
    const isConnected = connectionsData?.connections?.some((c: any) => c.provider === provider);
    
    if (isConnected) {
      // Already connected, show repository selection
      setSelectedProvider(provider);
      setSelectRepoModalOpen(true);
      setConnectModalOpen(false);
    } else {
      // Redirect to OAuth
      try {
        const response = await apiService.oauth.connect(provider, window.location.href);
        window.location.href = response.data.authorization_url;
      } catch (error: any) {
        message.error(error.response?.data?.detail || `Failed to connect to ${provider}`);
      }
    }
  };

  const handleAddRepoByUrl = async (values: { url: string }) => {
    createRepoMutation.mutate(values);
  };

  const handleSelectRepo = (repoFullName: string) => {
    if (selectedProvider) {
      connectRepoMutation.mutate({
        provider: selectedProvider,
        repo_full_name: repoFullName,
      });
    }
  };

  const filteredRepos = repositories.filter(repo => {
    if (searchQuery && !repo.name.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    if (filterProvider !== 'all' && repo.provider !== filterProvider) return false;
    if (filterStatus !== 'all' && repo.analysisStatus !== filterStatus) return false;
    return true;
  });

  const getStatusBadge = (status?: string) => {
    switch (status) {
      case 'passing':
        return <Tag color="success" icon={<CheckCircleOutlined />}>Passing</Tag>;
      case 'failing':
        return <Tag color="error" icon={<WarningOutlined />}>Failing</Tag>;
      case 'pending':
        return <Tag color="processing" icon={<SyncOutlined spin />}>Analyzing</Tag>;
      default:
        return <Tag color="default">Not Analyzed</Tag>;
    }
  };

  return (
    <div className="repositories-page" style={{ maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 12 }}>
            <BranchesOutlined style={{ color: '#2563eb' }} />
            {t('repositories.title', 'Repositories')}
          </Title>
          <Text type="secondary">
            {t('repositories.subtitle', 'Manage connected repositories and analyze code')}
          </Text>
        </div>
        <Space>
          <Button 
            icon={<SyncOutlined spin={isLoadingRepos} />}
            onClick={() => refetchRepos()}
            loading={isLoadingRepos}
          >
            {isLoadingRepos ? 'Syncing...' : 'Refresh'}
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setConnectModalOpen(true)}>
            Connect Repository
          </Button>
        </Space>
      </div>

      {/* Stats Row */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12, background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)', border: 'none' }}>
            <Statistic 
              title={<span style={{ color: 'rgba(255,255,255,0.8)' }}>Total Repos</span>}
              value={repositories.length}
              valueStyle={{ color: 'white', fontWeight: 700 }}
              prefix={<BranchesOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic 
              title="Passing"
              value={repositories.filter(r => r.analysisStatus === 'passing').length}
              valueStyle={{ color: '#10b981' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic 
              title="Failing"
              value={repositories.filter(r => r.analysisStatus === 'failing').length}
              valueStyle={{ color: '#ef4444' }}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic 
              title="Avg Health"
              value={Math.round(repositories.filter(r => r.healthScore > 0).reduce((a, b) => a + b.healthScore, 0) / repositories.filter(r => r.healthScore > 0).length)}
              suffix="%"
              valueStyle={{ color: '#2563eb' }}
              prefix={<SafetyCertificateOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Filters */}
      <Card style={{ marginBottom: 16, borderRadius: 12 }}>
        <Space wrap>
          <Input.Search
            placeholder="Search repositories..."
            style={{ width: 280 }}
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            allowClear
          />
          <Select
            value={filterProvider}
            onChange={setFilterProvider}
            style={{ width: 150 }}
            options={[
              { value: 'all', label: 'All Providers' },
              { value: 'github', label: 'GitHub' },
              { value: 'gitlab', label: 'GitLab' },
              { value: 'bitbucket', label: 'Bitbucket' },
            ]}
          />
          <Select
            value={filterStatus}
            onChange={setFilterStatus}
            style={{ width: 150 }}
            options={[
              { value: 'all', label: 'All Status' },
              { value: 'passing', label: 'Passing' },
              { value: 'failing', label: 'Failing' },
              { value: 'pending', label: 'Pending' },
              { value: 'none', label: 'Not Analyzed' },
            ]}
          />
        </Space>
      </Card>

      {/* Repository List */}
      <div>
        {isLoadingRepos ? (
          <Card style={{ borderRadius: 12, textAlign: 'center', padding: 40 }}>
            <Spin indicator={<LoadingOutlined style={{ fontSize: 32 }} spin />} />
            <div style={{ marginTop: 16 }}>
              <Text type="secondary">Loading repositories...</Text>
            </div>
          </Card>
        ) : filteredRepos.length === 0 ? (
          <Card style={{ borderRadius: 12, textAlign: 'center', padding: 40 }}>
            <Empty
              description={
                searchQuery || filterProvider !== 'all' || filterStatus !== 'all'
                  ? 'No repositories match your filters'
                  : 'No repositories connected yet'
              }
            >
              {!searchQuery && filterProvider === 'all' && filterStatus === 'all' && (
                <Button type="primary" icon={<PlusOutlined />} onClick={() => setConnectModalOpen(true)}>
                  Connect Your First Repository
                </Button>
              )}
            </Empty>
          </Card>
        ) : filteredRepos.map(repo => (
          <Card 
            key={repo.id}
            style={{ 
              marginBottom: 12, 
              borderRadius: 12,
              border: '1px solid #e2e8f0',
              transition: 'all 0.2s',
            }}
            className="repo-card"
            bodyStyle={{ padding: '20px 24px' }}
          >
            <Row gutter={24} align="middle">
              {/* Main Info */}
              <Col flex="1">
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
                  {/* Provider Icon */}
                  <div style={{
                    width: 48,
                    height: 48,
                    borderRadius: 12,
                    background: '#f8fafc',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 24,
                    color: '#475569',
                  }}>
                    {providerIcons[repo.provider]}
                  </div>

                  <div style={{ flex: 1 }}>
                    {/* Title Row */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <Text strong style={{ fontSize: 16, color: '#2563eb', cursor: 'pointer' }}>
                        {repo.fullName}
                      </Text>
                      {repo.visibility === 'private' ? (
                        <Tag icon={<LockOutlined />} style={{ borderRadius: 12 }}>Private</Tag>
                      ) : (
                        <Tag icon={<UnlockOutlined />} color="green" style={{ borderRadius: 12 }}>Public</Tag>
                      )}
                      {getStatusBadge(repo.analysisStatus)}
                    </div>

                    {/* Description */}
                    {repo.description && (
                      <Text type="secondary" style={{ display: 'block', marginBottom: 8 }}>
                        {repo.description}
                      </Text>
                    )}

                    {/* Meta Info */}
                    <Space size={16} wrap>
                      <Space size={4}>
                        <span style={{
                          width: 12,
                          height: 12,
                          borderRadius: '50%',
                          background: repo.languageColor,
                          display: 'inline-block',
                        }} />
                        <Text type="secondary">{repo.language}</Text>
                      </Space>
                      
                      <Space size={4}>
                        <StarOutlined style={{ color: '#f59e0b' }} />
                        <Text type="secondary">{repo.stars}</Text>
                      </Space>

                      <Space size={4}>
                        <ForkOutlined />
                        <Text type="secondary">{repo.forks}</Text>
                      </Space>

                      <Space size={4}>
                        <BranchesOutlined />
                        <Text type="secondary">{repo.branches} branches</Text>
                      </Space>

                      {repo.lastAnalysis && (
                        <Space size={4}>
                          <ClockCircleOutlined />
                          <Text type="secondary">
                            Analyzed {new Date(repo.lastAnalysis).toLocaleDateString()}
                          </Text>
                        </Space>
                      )}
                    </Space>
                  </div>
                </div>
              </Col>

              {/* Health Score */}
              <Col>
                {repo.healthScore > 0 && (
                  <div style={{ textAlign: 'center', minWidth: 80 }}>
                    <Progress
                      type="circle"
                      percent={repo.healthScore}
                      size={60}
                      strokeColor={
                        repo.healthScore >= 80 ? '#10b981' :
                        repo.healthScore >= 60 ? '#f59e0b' : '#ef4444'
                      }
                      format={p => <span style={{ fontSize: 14, fontWeight: 600 }}>{p}</span>}
                    />
                    <div style={{ marginTop: 4 }}>
                      <Text type="secondary" style={{ fontSize: 12 }}>Health</Text>
                    </div>
                  </div>
                )}
              </Col>

              {/* Actions */}
              <Col>
                <Space>
                  <Tooltip title={repo.isStarred ? 'Unstar' : 'Star'}>
                    <Button
                      type="text"
                      icon={repo.isStarred ? <StarFilled style={{ color: '#f59e0b' }} /> : <StarOutlined />}
                      onClick={() => toggleStar(repo.id)}
                    />
                  </Tooltip>
                  <Tooltip title="Sync with remote">
                    <Button 
                      icon={<SyncOutlined spin={syncRepoMutation.isPending} />}
                      onClick={() => syncRepoMutation.mutate(repo.id)}
                    />
                  </Tooltip>
                  <Button type="primary" icon={<PlayCircleOutlined />}>
                    Analyze
                  </Button>
                  <Dropdown
                    menu={{
                      items: [
                        { key: 'view', icon: <EyeOutlined />, label: 'View Details' },
                        { key: 'settings', icon: <SettingOutlined />, label: 'Settings' },
                        { 
                          key: 'clone', 
                          icon: <CopyOutlined />, 
                          label: 'Copy Clone URL',
                          onClick: () => {
                            navigator.clipboard.writeText(`https://github.com/${repo.fullName}.git`);
                            message.success('Clone URL copied to clipboard');
                          }
                        },
                        { type: 'divider' },
                        { 
                          key: 'disconnect', 
                          icon: <DeleteOutlined />, 
                          label: 'Disconnect', 
                          danger: true,
                          onClick: () => {
                            Modal.confirm({
                              title: 'Disconnect Repository',
                              content: `Are you sure you want to disconnect "${repo.name}"?`,
                              okText: 'Disconnect',
                              okButtonProps: { danger: true },
                              onOk: () => deleteRepoMutation.mutate(repo.id),
                            });
                          }
                        },
                      ],
                    }}
                    trigger={['click']}
                  >
                    <Button icon={<MoreOutlined />} />
                  </Dropdown>
                </Space>
              </Col>
            </Row>
          </Card>
        ))}
      </div>

      {/* Connect Repository Modal */}
      <Modal
        title={<><LinkOutlined /> Connect Repository</>}
        open={connectModalOpen}
        onCancel={() => setConnectModalOpen(false)}
        footer={null}
        width={600}
      >
        <div style={{ padding: '24px 0' }}>
          <Row gutter={16}>
            {[
              { provider: 'github', name: 'GitHub', icon: <GithubOutlined style={{ fontSize: 32 }} />, color: '#24292e' },
              { provider: 'gitlab', name: 'GitLab', icon: <GitlabOutlined style={{ fontSize: 32 }} />, color: '#fc6d26' },
            ].map(p => {
              const isConnected = connectionsData?.connections?.some((c: any) => c.provider === p.provider);
              return (
                <Col span={12} key={p.provider}>
                  <Card
                    hoverable
                    onClick={() => handleConnectProvider(p.provider)}
                    style={{ 
                      textAlign: 'center', 
                      borderRadius: 12,
                      border: isConnected ? '2px solid #52c41a' : '2px solid #e2e8f0',
                      transition: 'all 0.2s',
                    }}
                    bodyStyle={{ padding: 24 }}
                  >
                    <div style={{ color: p.color, marginBottom: 12 }}>{p.icon}</div>
                    <Title level={5} style={{ margin: 0 }}>{p.name}</Title>
                    <Text type="secondary">
                      {isConnected ? 'Connected - Click to select repositories' : `Connect your ${p.name} account`}
                    </Text>
                    {isConnected && (
                      <Tag color="success" style={{ marginTop: 8 }}>
                        <CheckCircleOutlined /> Connected
                      </Tag>
                    )}
                    <Button 
                      type={isConnected ? "default" : "primary"} 
                      block 
                      style={{ marginTop: 16 }}
                    >
                      {isConnected ? 'Select Repositories' : `Connect ${p.name}`}
                    </Button>
                  </Card>
                </Col>
              );
            })}
          </Row>

          <Divider>Or enter repository URL</Divider>

          <Form form={form} layout="vertical" onFinish={handleAddRepoByUrl}>
            <Form.Item 
              name="url" 
              label="Repository URL"
              rules={[
                { required: true, message: 'Please enter a repository URL' },
                { type: 'url', message: 'Please enter a valid URL' },
              ]}
            >
              <Input placeholder="https://github.com/username/repository" />
            </Form.Item>
            <Form.Item>
              <Button 
                type="primary" 
                block 
                htmlType="submit"
                loading={createRepoMutation.isPending}
              >
                Add Repository
              </Button>
            </Form.Item>
          </Form>
        </div>
      </Modal>

      {/* Select Repository Modal */}
      <Modal
        title={
          <Space>
            {selectedProvider === 'github' ? <GithubOutlined /> : <GitlabOutlined />}
            Select Repository from {selectedProvider?.charAt(0).toUpperCase()}{selectedProvider?.slice(1)}
          </Space>
        }
        open={selectRepoModalOpen}
        onCancel={() => {
          setSelectRepoModalOpen(false);
          setSelectedProvider(null);
        }}
        footer={null}
        width={700}
      >
        {isLoadingProviderRepos ? (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Spin indicator={<LoadingOutlined style={{ fontSize: 32 }} spin />} />
            <div style={{ marginTop: 16 }}>
              <Text type="secondary">Loading repositories...</Text>
            </div>
          </div>
        ) : providerRepos?.repositories?.length > 0 ? (
          <List
            dataSource={providerRepos.repositories}
            style={{ maxHeight: 400, overflow: 'auto' }}
            renderItem={(repo: any) => (
              <List.Item
                key={repo.id}
                actions={[
                  <Button
                    type="primary"
                    size="small"
                    onClick={() => handleSelectRepo(repo.full_name)}
                    loading={connectRepoMutation.isPending}
                  >
                    Connect
                  </Button>
                ]}
              >
                <List.Item.Meta
                  avatar={
                    <Avatar 
                      icon={repo.is_private ? <LockOutlined /> : <UnlockOutlined />}
                      style={{ backgroundColor: repo.is_private ? '#faad14' : '#52c41a' }}
                    />
                  }
                  title={
                    <Space>
                      <Text strong>{repo.full_name}</Text>
                      {repo.is_private && <Tag color="orange">Private</Tag>}
                    </Space>
                  }
                  description={
                    <Space split={<Divider type="vertical" />}>
                      <Text type="secondary">{repo.description || 'No description'}</Text>
                      <Space size={12}>
                        <span><StarOutlined /> {repo.stars}</span>
                        <span><ForkOutlined /> {repo.forks}</span>
                      </Space>
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        ) : (
          <Empty description="No repositories found" />
        )}
      </Modal>

      <style>{`
        .repo-card:hover {
          border-color: #2563eb !important;
          box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
        }
      `}</style>
    </div>
  );
};

export default Repositories;
