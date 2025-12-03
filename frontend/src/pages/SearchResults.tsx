/**
 * Search Results Page
 * 搜索结果页面
 * 
 * Features:
 * - Global search across all entities
 * - Filter by type
 * - Highlighted matches
 * - Quick actions
 */

import React, { useState, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Input,
  Tag,
  List,
  Avatar,
  Tabs,
  Empty,
  Button,
  Tooltip,
  Highlight,
} from 'antd';
import {
  SearchOutlined,
  FileTextOutlined,
  ProjectOutlined,
  BranchesOutlined,
  CodeOutlined,
  UserOutlined,
  SettingOutlined,
  BugOutlined,
  SafetyCertificateOutlined,
  RocketOutlined,
  ClockCircleOutlined,
  RightOutlined,
  FilterOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useSearchParams } from 'react-router-dom';

const { Title, Text, Paragraph } = Typography;

interface SearchResult {
  id: string;
  type: 'project' | 'file' | 'issue' | 'pr' | 'user' | 'setting' | 'deployment';
  title: string;
  description: string;
  path?: string;
  tags?: string[];
  timestamp?: string;
  relevance: number;
}

const mockResults: SearchResult[] = [
  {
    id: '1',
    type: 'file',
    title: 'authentication.py',
    description: 'User authentication module with JWT token handling',
    path: 'src/api/authentication.py',
    tags: ['python', 'auth'],
    relevance: 0.95,
  },
  {
    id: '2',
    type: 'issue',
    title: 'SQL Injection Vulnerability',
    description: 'Critical security issue in user query endpoint',
    path: 'src/api/users.py:45',
    tags: ['security', 'critical'],
    relevance: 0.92,
  },
  {
    id: '3',
    type: 'project',
    title: 'Backend Services',
    description: 'FastAPI backend microservices for the platform',
    tags: ['python', 'fastapi'],
    relevance: 0.88,
  },
  {
    id: '4',
    type: 'pr',
    title: 'PR #142: Add user authentication',
    description: 'Implements secure JWT-based authentication with refresh tokens',
    tags: ['feature', 'auth'],
    timestamp: '2024-03-01',
    relevance: 0.85,
  },
  {
    id: '5',
    type: 'file',
    title: 'auth.ts',
    description: 'Frontend authentication hook and utilities',
    path: 'src/hooks/useAuth.ts',
    tags: ['typescript', 'react'],
    relevance: 0.82,
  },
  {
    id: '6',
    type: 'deployment',
    title: 'Production Deploy v2.1.5',
    description: 'Latest production deployment with auth improvements',
    timestamp: '2024-03-01',
    relevance: 0.78,
  },
  {
    id: '7',
    type: 'setting',
    title: 'API Keys Configuration',
    description: 'Manage API keys and access tokens',
    path: '/settings/api-keys',
    relevance: 0.75,
  },
  {
    id: '8',
    type: 'user',
    title: 'John Doe',
    description: 'Senior Developer - Backend Team',
    relevance: 0.70,
  },
];

const typeConfig = {
  project: { icon: <ProjectOutlined />, color: '#3b82f6', label: 'Project' },
  file: { icon: <FileTextOutlined />, color: '#22c55e', label: 'File' },
  issue: { icon: <BugOutlined />, color: '#ef4444', label: 'Issue' },
  pr: { icon: <BranchesOutlined />, color: '#8b5cf6', label: 'Pull Request' },
  user: { icon: <UserOutlined />, color: '#f59e0b', label: 'User' },
  setting: { icon: <SettingOutlined />, color: '#64748b', label: 'Setting' },
  deployment: { icon: <RocketOutlined />, color: '#06b6d4', label: 'Deployment' },
};

export const SearchResults: React.FC = () => {
  const { t } = useTranslation();
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get('q') || '';
  const [searchQuery, setSearchQuery] = useState(query);
  const [activeFilter, setActiveFilter] = useState('all');

  const filteredResults = useMemo(() => {
    let results = mockResults;
    if (activeFilter !== 'all') {
      results = results.filter(r => r.type === activeFilter);
    }
    if (searchQuery) {
      results = results.filter(r =>
        r.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        r.description.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }
    return results.sort((a, b) => b.relevance - a.relevance);
  }, [searchQuery, activeFilter]);

  const resultCounts = {
    all: mockResults.length,
    file: mockResults.filter(r => r.type === 'file').length,
    issue: mockResults.filter(r => r.type === 'issue').length,
    pr: mockResults.filter(r => r.type === 'pr').length,
    project: mockResults.filter(r => r.type === 'project').length,
  };

  const handleSearch = (value: string) => {
    setSearchQuery(value);
    setSearchParams({ q: value });
  };

  return (
    <div className="search-results-page" style={{ maxWidth: 1000, margin: '0 auto' }}>
      {/* Search Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <SearchOutlined style={{ color: '#2563eb' }} /> Search Results
          </Title>
          {searchQuery && (
            <Text type="secondary">
              Found {filteredResults.length} results for "{searchQuery}"
            </Text>
          )}
        </div>
      </div>

      {/* Search Input */}
      <Card style={{ marginBottom: 24, borderRadius: 12 }}>
        <Input.Search
          placeholder="Search projects, files, issues, PRs..."
          size="large"
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          onSearch={handleSearch}
          allowClear
          style={{ maxWidth: 600 }}
        />
      </Card>

      {/* Filter Tabs */}
      <Tabs
        activeKey={activeFilter}
        onChange={setActiveFilter}
        style={{ marginBottom: 16 }}
        items={[
          { key: 'all', label: `All (${resultCounts.all})` },
          { key: 'file', label: `Files (${resultCounts.file})` },
          { key: 'issue', label: `Issues (${resultCounts.issue})` },
          { key: 'pr', label: `Pull Requests (${resultCounts.pr})` },
          { key: 'project', label: `Projects (${resultCounts.project})` },
        ]}
      />

      {/* Results List */}
      <Card style={{ borderRadius: 12 }} bodyStyle={{ padding: 0 }}>
        {filteredResults.length > 0 ? (
          <List
            dataSource={filteredResults}
            renderItem={result => {
              const config = typeConfig[result.type];
              return (
                <List.Item
                  style={{
                    padding: '16px 24px',
                    cursor: 'pointer',
                    transition: 'background 0.2s',
                  }}
                  className="search-result-item"
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start', width: '100%' }}>
                    <Avatar
                      style={{
                        background: `${config.color}15`,
                        color: config.color,
                        marginRight: 16,
                      }}
                      icon={config.icon}
                    />
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <Text strong>{result.title}</Text>
                        <Tag color={config.color}>{config.label}</Tag>
                        {result.tags?.map(tag => (
                          <Tag key={tag} style={{ fontSize: 11 }}>{tag}</Tag>
                        ))}
                      </div>
                      <Paragraph
                        type="secondary"
                        style={{ margin: '4px 0', fontSize: 13 }}
                        ellipsis={{ rows: 1 }}
                      >
                        {result.description}
                      </Paragraph>
                      {result.path && (
                        <Text code style={{ fontSize: 12 }}>{result.path}</Text>
                      )}
                      {result.timestamp && (
                        <Text type="secondary" style={{ fontSize: 12, marginLeft: result.path ? 12 : 0 }}>
                          <ClockCircleOutlined style={{ marginRight: 4 }} />
                          {new Date(result.timestamp).toLocaleDateString()}
                        </Text>
                      )}
                    </div>
                    <Tooltip title="View">
                      <Button type="text" icon={<RightOutlined />} />
                    </Tooltip>
                  </div>
                </List.Item>
              );
            }}
          />
        ) : (
          <Empty
            description={
              searchQuery
                ? `No results found for "${searchQuery}"`
                : 'Enter a search term to find results'
            }
            style={{ padding: 48 }}
          />
        )}
      </Card>

      {/* Search Tips */}
      {!searchQuery && (
        <Card title="Search Tips" style={{ marginTop: 24, borderRadius: 12 }}>
          <Row gutter={16}>
            {[
              { icon: <FileTextOutlined />, text: 'Search files by name or content' },
              { icon: <BugOutlined />, text: 'Find issues and vulnerabilities' },
              { icon: <BranchesOutlined />, text: 'Search pull requests' },
              { icon: <CodeOutlined />, text: 'Use quotes for exact matches' },
            ].map((tip, index) => (
              <Col key={index} xs={12} md={6}>
                <Space>
                  <span style={{ color: '#2563eb' }}>{tip.icon}</span>
                  <Text type="secondary" style={{ fontSize: 13 }}>{tip.text}</Text>
                </Space>
              </Col>
            ))}
          </Row>
        </Card>
      )}

      <style>{`
        .search-result-item:hover {
          background: #f8fafc !important;
        }
      `}</style>
    </div>
  );
};

export default SearchResults;
