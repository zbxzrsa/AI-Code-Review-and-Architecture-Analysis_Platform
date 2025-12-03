/**
 * Repository Management Page
 * 仓库管理页面
 * 
 * GitHub-inspired repository view with:
 * - Repository list with stats
 * - Branch management
 * - Clone/connect functionality
 * - Repository settings
 */

import React, { useState } from 'react';
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
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text, Paragraph } = Typography;

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
  const [repositories, setRepositories] = useState<Repository[]>(mockRepositories);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterProvider, setFilterProvider] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [connectModalOpen, setConnectModalOpen] = useState(false);
  const [form] = Form.useForm();

  const toggleStar = (repoId: string) => {
    setRepositories(prev => prev.map(r =>
      r.id === repoId ? { ...r, isStarred: !r.isStarred } : r
    ));
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
          <Button icon={<SyncOutlined />}>Sync All</Button>
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
        {filteredRepos.map(repo => (
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
                  <Button type="primary" icon={<PlayCircleOutlined />}>
                    Analyze
                  </Button>
                  <Dropdown
                    menu={{
                      items: [
                        { key: 'view', icon: <EyeOutlined />, label: 'View Details' },
                        { key: 'settings', icon: <SettingOutlined />, label: 'Settings' },
                        { key: 'clone', icon: <CopyOutlined />, label: 'Copy Clone URL' },
                        { type: 'divider' },
                        { key: 'disconnect', icon: <DeleteOutlined />, label: 'Disconnect', danger: true },
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
            ].map(p => (
              <Col span={12} key={p.provider}>
                <Card
                  hoverable
                  style={{ 
                    textAlign: 'center', 
                    borderRadius: 12,
                    border: '2px solid #e2e8f0',
                    transition: 'all 0.2s',
                  }}
                  bodyStyle={{ padding: 24 }}
                >
                  <div style={{ color: p.color, marginBottom: 12 }}>{p.icon}</div>
                  <Title level={5} style={{ margin: 0 }}>{p.name}</Title>
                  <Text type="secondary">Connect your {p.name} repositories</Text>
                  <Button type="primary" block style={{ marginTop: 16 }}>
                    Connect {p.name}
                  </Button>
                </Card>
              </Col>
            ))}
          </Row>

          <Divider>Or enter repository URL</Divider>

          <Form form={form} layout="vertical">
            <Form.Item name="url" label="Repository URL">
              <Input placeholder="https://github.com/username/repository" />
            </Form.Item>
            <Form.Item>
              <Button type="primary" block>Add Repository</Button>
            </Form.Item>
          </Form>
        </div>
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
