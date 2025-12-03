/**
 * Pull Requests Page
 * 拉取请求页面
 * 
 * GitHub-style PR management with:
 * - PR list with status
 * - AI-powered review comments
 * - Merge/close actions
 * - Diff viewer integration
 */

import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Tag,
  Space,
  Typography,
  Avatar,
  Badge,
  Tooltip,
  Select,
  Input,
  Tabs,
  Progress,
  Timeline,
  Divider,
  Modal,
  List,
  message,
} from 'antd';
import type { TableProps } from 'antd';
import {
  PullRequestOutlined,
  MergeOutlined,
  CloseCircleOutlined,
  CheckCircleOutlined,
  CommentOutlined,
  EyeOutlined,
  BranchesOutlined,
  ClockCircleOutlined,
  UserOutlined,
  RobotOutlined,
  CodeOutlined,
  WarningOutlined,
  SyncOutlined,
  PlusOutlined,
  SearchOutlined,
  FilterOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text, Paragraph } = Typography;

interface PullRequest {
  id: string;
  number: number;
  title: string;
  description?: string;
  author: {
    name: string;
    avatar?: string;
  };
  repository: string;
  sourceBranch: string;
  targetBranch: string;
  status: 'open' | 'merged' | 'closed' | 'draft';
  reviewStatus: 'pending' | 'approved' | 'changes_requested' | 'reviewing';
  aiReviewStatus?: 'pending' | 'in_progress' | 'completed';
  aiScore?: number;
  issuesFound?: number;
  securityIssues?: number;
  comments: number;
  commits: number;
  additions: number;
  deletions: number;
  createdAt: string;
  updatedAt: string;
  reviewers?: { name: string; avatar?: string; status: string }[];
}

const mockPullRequests: PullRequest[] = [
  {
    id: 'pr_1',
    number: 142,
    title: 'Add user authentication with JWT tokens',
    description: 'Implements secure JWT-based authentication with refresh tokens',
    author: { name: 'John Doe' },
    repository: 'backend-services',
    sourceBranch: 'feature/auth',
    targetBranch: 'main',
    status: 'open',
    reviewStatus: 'approved',
    aiReviewStatus: 'completed',
    aiScore: 92,
    issuesFound: 2,
    securityIssues: 0,
    comments: 8,
    commits: 5,
    additions: 450,
    deletions: 120,
    createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    updatedAt: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
    reviewers: [
      { name: 'Jane Smith', status: 'approved' },
      { name: 'AI Review', status: 'approved' },
    ],
  },
  {
    id: 'pr_2',
    number: 141,
    title: 'Fix SQL injection vulnerability in user query',
    description: 'Critical security fix for SQL injection vulnerability',
    author: { name: 'AI Auto-Fix' },
    repository: 'backend-services',
    sourceBranch: 'auto-fix/sql-injection',
    targetBranch: 'main',
    status: 'open',
    reviewStatus: 'pending',
    aiReviewStatus: 'completed',
    aiScore: 98,
    issuesFound: 0,
    securityIssues: 0,
    comments: 2,
    commits: 1,
    additions: 15,
    deletions: 8,
    createdAt: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
    updatedAt: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    reviewers: [
      { name: 'AI Review', status: 'approved' },
    ],
  },
  {
    id: 'pr_3',
    number: 140,
    title: 'Implement dashboard analytics components',
    author: { name: 'Bob Wilson' },
    repository: 'frontend',
    sourceBranch: 'feature/analytics',
    targetBranch: 'develop',
    status: 'open',
    reviewStatus: 'changes_requested',
    aiReviewStatus: 'completed',
    aiScore: 75,
    issuesFound: 8,
    securityIssues: 1,
    comments: 15,
    commits: 12,
    additions: 890,
    deletions: 45,
    createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
    updatedAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    reviewers: [
      { name: 'Jane Smith', status: 'changes_requested' },
      { name: 'AI Review', status: 'changes_requested' },
    ],
  },
  {
    id: 'pr_4',
    number: 139,
    title: 'Add Kubernetes deployment manifests',
    author: { name: 'Alice Brown' },
    repository: 'infrastructure',
    sourceBranch: 'feature/k8s',
    targetBranch: 'main',
    status: 'merged',
    reviewStatus: 'approved',
    aiReviewStatus: 'completed',
    aiScore: 95,
    issuesFound: 0,
    comments: 5,
    commits: 8,
    additions: 650,
    deletions: 20,
    createdAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    updatedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'pr_5',
    number: 138,
    title: 'WIP: Refactor API client',
    author: { name: 'John Doe' },
    repository: 'frontend',
    sourceBranch: 'refactor/api-client',
    targetBranch: 'develop',
    status: 'draft',
    reviewStatus: 'pending',
    comments: 0,
    commits: 3,
    additions: 200,
    deletions: 150,
    createdAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
    updatedAt: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
  },
];

export const PullRequests: React.FC = () => {
  const { t: _t } = useTranslation();
  const [pullRequests, setPullRequests] = useState<PullRequest[]>(mockPullRequests);
  const [selectedPR, setSelectedPR] = useState<PullRequest | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [filter, setFilter] = useState('open');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredPRs = pullRequests.filter(pr => {
    if (filter !== 'all' && pr.status !== filter) return false;
    if (searchQuery && !pr.title.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const getStatusBadge = (status: string) => {
    const configs: Record<string, { color: string; icon: React.ReactNode; bgColor: string }> = {
      open: { color: '#22c55e', icon: <PullRequestOutlined />, bgColor: '#dcfce7' },
      merged: { color: '#8b5cf6', icon: <MergeOutlined />, bgColor: '#ede9fe' },
      closed: { color: '#ef4444', icon: <CloseCircleOutlined />, bgColor: '#fee2e2' },
      draft: { color: '#64748b', icon: <CodeOutlined />, bgColor: '#f1f5f9' },
    };
    const config = configs[status] || configs.open;
    return (
      <Tag 
        icon={config.icon}
        style={{ 
          background: config.bgColor, 
          color: config.color, 
          border: 'none',
          borderRadius: 16,
          fontWeight: 500,
        }}
      >
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Tag>
    );
  };

  const getReviewStatusBadge = (status: string) => {
    const configs: Record<string, { color: string; text: string }> = {
      pending: { color: 'default', text: 'Pending Review' },
      approved: { color: 'success', text: 'Approved' },
      changes_requested: { color: 'warning', text: 'Changes Requested' },
      reviewing: { color: 'processing', text: 'In Review' },
    };
    const config = configs[status] || { color: 'default', text: status };
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const handleMerge = async (pr: PullRequest) => {
    message.success(`PR #${pr.number} merged successfully`);
    setPullRequests(prev => prev.map(p =>
      p.id === pr.id ? { ...p, status: 'merged' as const } : p
    ));
  };

  const handleClose = async (pr: PullRequest) => {
    message.info(`PR #${pr.number} closed`);
    setPullRequests(prev => prev.map(p =>
      p.id === pr.id ? { ...p, status: 'closed' as const } : p
    ));
  };

  const stats = {
    open: pullRequests.filter(pr => pr.status === 'open').length,
    merged: pullRequests.filter(pr => pr.status === 'merged').length,
    closed: pullRequests.filter(pr => pr.status === 'closed').length,
  };

  return (
    <div className="pull-requests-page" style={{ maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <PullRequestOutlined style={{ color: '#22c55e' }} /> Pull Requests
          </Title>
          <Text type="secondary">Review and merge code changes with AI assistance</Text>
        </div>
        <Space>
          <Input.Search
            placeholder="Search pull requests..."
            style={{ width: 280 }}
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            allowClear
          />
          <Select
            value={filter}
            onChange={setFilter}
            style={{ width: 130 }}
            options={[
              { value: 'all', label: 'All' },
              { value: 'open', label: 'Open' },
              { value: 'merged', label: 'Merged' },
              { value: 'closed', label: 'Closed' },
              { value: 'draft', label: 'Draft' },
            ]}
          />
          <Button type="primary" icon={<PlusOutlined />}>
            New PR
          </Button>
        </Space>
      </div>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={8}>
          <Card style={{ borderRadius: 12, borderLeft: '4px solid #22c55e' }}>
            <Statistic title="Open" value={stats.open} prefix={<PullRequestOutlined style={{ color: '#22c55e' }} />} />
          </Card>
        </Col>
        <Col xs={8}>
          <Card style={{ borderRadius: 12, borderLeft: '4px solid #8b5cf6' }}>
            <Statistic title="Merged" value={stats.merged} prefix={<MergeOutlined style={{ color: '#8b5cf6' }} />} />
          </Card>
        </Col>
        <Col xs={8}>
          <Card style={{ borderRadius: 12, borderLeft: '4px solid #ef4444' }}>
            <Statistic title="Closed" value={stats.closed} prefix={<CloseCircleOutlined style={{ color: '#ef4444' }} />} />
          </Card>
        </Col>
      </Row>

      {/* PR List */}
      <Card style={{ borderRadius: 12 }} bodyStyle={{ padding: 0 }}>
        {filteredPRs.map((pr, index) => (
          <div
            key={pr.id}
            style={{
              padding: '20px 24px',
              borderBottom: index < filteredPRs.length - 1 ? '1px solid #f1f5f9' : 'none',
              transition: 'background 0.2s',
              cursor: 'pointer',
            }}
            className="pr-item"
            onClick={() => {
              setSelectedPR(pr);
              setDetailModalOpen(true);
            }}
          >
            <Row gutter={24} align="middle">
              <Col flex="1">
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                  {/* Status Icon */}
                  <div style={{
                    width: 40,
                    height: 40,
                    borderRadius: 10,
                    background: pr.status === 'open' ? '#dcfce7' : 
                               pr.status === 'merged' ? '#ede9fe' : 
                               pr.status === 'closed' ? '#fee2e2' : '#f1f5f9',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: pr.status === 'open' ? '#22c55e' : 
                           pr.status === 'merged' ? '#8b5cf6' : 
                           pr.status === 'closed' ? '#ef4444' : '#64748b',
                    fontSize: 20,
                  }}>
                    {pr.status === 'merged' ? <MergeOutlined /> : 
                     pr.status === 'closed' ? <CloseCircleOutlined /> : 
                     <PullRequestOutlined />}
                  </div>

                  <div style={{ flex: 1 }}>
                    {/* Title Row */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <Text strong style={{ fontSize: 15 }}>
                        {pr.title}
                      </Text>
                      <Text type="secondary">#{pr.number}</Text>
                      {pr.status === 'draft' && <Tag>Draft</Tag>}
                    </div>

                    {/* Meta Info */}
                    <Space size={16} wrap>
                      <Space size={4}>
                        <Avatar size={20} icon={<UserOutlined />} />
                        <Text type="secondary">{pr.author.name}</Text>
                      </Space>
                      <Text type="secondary">
                        <BranchesOutlined /> {pr.sourceBranch} �?{pr.targetBranch}
                      </Text>
                      <Text type="secondary">
                        <ClockCircleOutlined /> {new Date(pr.createdAt).toLocaleDateString()}
                      </Text>
                    </Space>
                  </div>
                </div>
              </Col>

              {/* AI Review Score */}
              {pr.aiScore !== undefined && (
                <Col>
                  <Tooltip title="AI Review Score">
                    <div style={{ textAlign: 'center' }}>
                      <Progress
                        type="circle"
                        percent={pr.aiScore}
                        size={50}
                        strokeColor={pr.aiScore >= 90 ? '#22c55e' : pr.aiScore >= 70 ? '#faad14' : '#ef4444'}
                        format={p => <span style={{ fontSize: 12, fontWeight: 600 }}>{p}</span>}
                      />
                      <div>
                        <RobotOutlined style={{ color: '#2563eb', marginRight: 4 }} />
                        <Text type="secondary" style={{ fontSize: 11 }}>AI Score</Text>
                      </div>
                    </div>
                  </Tooltip>
                </Col>
              )}

              {/* Issues */}
              <Col style={{ minWidth: 100 }}>
                <Space direction="vertical" size={4} align="end">
                  {pr.securityIssues !== undefined && pr.securityIssues > 0 && (
                    <Tag color="red" icon={<SafetyCertificateOutlined />}>
                      {pr.securityIssues} security
                    </Tag>
                  )}
                  {pr.issuesFound !== undefined && pr.issuesFound > 0 && (
                    <Tag color="orange" icon={<WarningOutlined />}>
                      {pr.issuesFound} issues
                    </Tag>
                  )}
                  <Space>
                    <Tooltip title="Comments">
                      <span><CommentOutlined /> {pr.comments}</span>
                    </Tooltip>
                    <Tooltip title="Commits">
                      <span><CodeOutlined /> {pr.commits}</span>
                    </Tooltip>
                  </Space>
                </Space>
              </Col>

              {/* Review Status */}
              <Col style={{ minWidth: 140 }}>
                {getReviewStatusBadge(pr.reviewStatus)}
              </Col>

              {/* Actions */}
              <Col>
                <Space onClick={e => e.stopPropagation()}>
                  {pr.status === 'open' && pr.reviewStatus === 'approved' && (
                    <Button type="primary" size="small" icon={<MergeOutlined />} onClick={() => handleMerge(pr)}>
                      Merge
                    </Button>
                  )}
                  {pr.status === 'open' && (
                    <Button size="small" icon={<EyeOutlined />}>
                      Review
                    </Button>
                  )}
                </Space>
              </Col>
            </Row>
          </div>
        ))}
      </Card>

      {/* PR Detail Modal */}
      <Modal
        title={<><PullRequestOutlined /> PR #{selectedPR?.number}: {selectedPR?.title}</>}
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        width={900}
        footer={selectedPR?.status === 'open' ? [
          <Button key="close" danger onClick={() => {
            if (selectedPR) handleClose(selectedPR);
            setDetailModalOpen(false);
          }}>
            Close PR
          </Button>,
          selectedPR?.reviewStatus === 'approved' && (
            <Button key="merge" type="primary" icon={<MergeOutlined />} onClick={() => {
              if (selectedPR) handleMerge(selectedPR);
              setDetailModalOpen(false);
            }}>
              Merge Pull Request
            </Button>
          ),
        ].filter(Boolean) : null}
      >
        {selectedPR && (
          <div>
            <Row gutter={24}>
              <Col span={16}>
                <Paragraph>{selectedPR.description}</Paragraph>
                
                <Divider>Changes</Divider>
                <Space size={24}>
                  <Statistic title="Commits" value={selectedPR.commits} />
                  <Statistic 
                    title="Additions" 
                    value={selectedPR.additions} 
                    valueStyle={{ color: '#22c55e' }}
                    prefix="+"
                  />
                  <Statistic 
                    title="Deletions" 
                    value={selectedPR.deletions} 
                    valueStyle={{ color: '#ef4444' }}
                    prefix="-"
                  />
                </Space>

                {selectedPR.aiScore !== undefined && (
                  <>
                    <Divider>AI Review</Divider>
                    <Row gutter={16}>
                      <Col span={8}>
                        <Card size="small">
                          <Statistic 
                            title="AI Score" 
                            value={selectedPR.aiScore}
                            suffix="/ 100"
                            valueStyle={{ color: selectedPR.aiScore >= 90 ? '#22c55e' : '#faad14' }}
                          />
                        </Card>
                      </Col>
                      <Col span={8}>
                        <Card size="small">
                          <Statistic 
                            title="Issues Found" 
                            value={selectedPR.issuesFound || 0}
                            valueStyle={{ color: (selectedPR.issuesFound || 0) > 0 ? '#faad14' : '#22c55e' }}
                          />
                        </Card>
                      </Col>
                      <Col span={8}>
                        <Card size="small">
                          <Statistic 
                            title="Security Issues" 
                            value={selectedPR.securityIssues || 0}
                            valueStyle={{ color: (selectedPR.securityIssues || 0) > 0 ? '#ef4444' : '#22c55e' }}
                          />
                        </Card>
                      </Col>
                    </Row>
                  </>
                )}
              </Col>
              <Col span={8}>
                <Card size="small" title="Reviewers">
                  <List
                    size="small"
                    dataSource={selectedPR.reviewers || []}
                    renderItem={reviewer => (
                      <List.Item>
                        <Space>
                          <Avatar size="small" icon={reviewer.name === 'AI Review' ? <RobotOutlined /> : <UserOutlined />} />
                          <Text>{reviewer.name}</Text>
                        </Space>
                        {getReviewStatusBadge(reviewer.status)}
                      </List.Item>
                    )}
                  />
                </Card>
              </Col>
            </Row>
          </div>
        )}
      </Modal>

      <style>{`
        .pr-item:hover {
          background: #f8fafc !important;
        }
      `}</style>
    </div>
  );
};

// Import missing component
import { Statistic } from 'antd';

export default PullRequests;
