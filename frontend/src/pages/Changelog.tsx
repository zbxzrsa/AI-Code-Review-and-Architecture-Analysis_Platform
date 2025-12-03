/**
 * Changelog Page
 * 更新日志页面
 * 
 * Features:
 * - Version history with filtering
 * - New features announcements
 * - Bug fixes and improvements
 * - Roadmap preview
 * - Statistics dashboard
 * - Search functionality
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Tag,
  Timeline,
  Button,
  Badge,
  Divider,
  Alert,
  List,
  Avatar,
  Input,
  Segmented,
  Tooltip,
  Empty,
  Statistic,
  message,
} from 'antd';
import {
  HistoryOutlined,
  RocketOutlined,
  BugOutlined,
  ThunderboltOutlined,
  SafetyCertificateOutlined,
  StarOutlined,
  ClockCircleOutlined,
  GiftOutlined,
  BulbOutlined,
  ToolOutlined,
  SearchOutlined,
  FilterOutlined,
  CopyOutlined,
  LinkOutlined,
  DownOutlined,
  UpOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text, Paragraph } = Typography;

// ============================================
// Type Definitions
// ============================================

type VersionType = 'major' | 'minor' | 'patch';
type ChangeType = 'feature' | 'improvement' | 'bugfix' | 'security';
type RoadmapStatus = 'in-progress' | 'planned' | 'completed';

interface Change {
  type: ChangeType;
  text: string;
  link?: string;
}

interface ChangelogEntry {
  version: string;
  date: string;
  type: VersionType;
  title: string;
  description: string;
  changes: Change[];
  isNew?: boolean;
  breaking?: boolean;
}

interface RoadmapItem {
  id: string;
  title: string;
  description?: string;
  status: RoadmapStatus;
  eta: string;
  progress?: number;
}

interface ChangeTypeConfig {
  icon: React.ReactNode;
  color: string;
  label: string;
}

const changelog: ChangelogEntry[] = [
  {
    version: 'v2.2.0',
    date: '2024-03-01',
    type: 'minor',
    title: 'AI Auto-Fix & Enhanced Security',
    description: 'Introducing automated vulnerability fixing and improved security scanning',
    isNew: true,
    changes: [
      { type: 'feature', text: 'AI Auto-Fix system for automatic vulnerability remediation' },
      { type: 'feature', text: 'Real-time code comparison with AI analysis' },
      { type: 'feature', text: 'Enhanced notification center with filtering' },
      { type: 'improvement', text: 'Improved sidebar navigation with VSCode-inspired theme' },
      { type: 'security', text: 'Added OWASP Top 10 coverage in security dashboard' },
      { type: 'bugfix', text: 'Fixed icon import issues in various components' },
    ],
  },
  {
    version: 'v2.1.5',
    date: '2024-02-25',
    type: 'patch',
    title: 'Bug Fixes & Performance',
    description: 'Various bug fixes and performance improvements',
    changes: [
      { type: 'bugfix', text: 'Fixed null pointer exception in analysis service' },
      { type: 'improvement', text: 'Optimized API response times by 40%' },
      { type: 'improvement', text: 'Reduced memory usage in code diff viewer' },
      { type: 'bugfix', text: 'Fixed authentication token refresh issues' },
    ],
  },
  {
    version: 'v2.1.0',
    date: '2024-02-15',
    type: 'minor',
    title: 'Deployments & CI/CD Integration',
    description: 'New deployment management features and CI/CD pipeline support',
    changes: [
      { type: 'feature', text: 'Deployment pipeline visualization and management' },
      { type: 'feature', text: 'GitHub Actions and GitLab CI integration' },
      { type: 'feature', text: 'Rollback support for failed deployments' },
      { type: 'improvement', text: 'Enhanced pull request review workflow' },
      { type: 'security', text: 'Added secret scanning in CI/CD pipelines' },
    ],
  },
  {
    version: 'v2.0.0',
    date: '2024-02-01',
    type: 'major',
    title: 'Major Platform Redesign',
    description: 'Complete UI overhaul with new features and improved UX',
    changes: [
      { type: 'feature', text: 'New VSCode/GitHub-inspired design system' },
      { type: 'feature', text: 'AI-powered code review with GPT-4 Turbo' },
      { type: 'feature', text: 'Team collaboration features' },
      { type: 'feature', text: 'Comprehensive analytics dashboard' },
      { type: 'improvement', text: 'Complete codebase migration to TypeScript' },
      { type: 'security', text: 'Enhanced authentication with 2FA support' },
    ],
  },
];

const roadmap: RoadmapItem[] = [
  { id: '1', title: 'IDE Extensions', description: 'VS Code and JetBrains plugins', status: 'in-progress', eta: 'Q2 2024', progress: 65 },
  { id: '2', title: 'Custom AI Model Training', description: 'Train models on your codebase', status: 'planned', eta: 'Q2 2024' },
  { id: '3', title: 'Multi-language Support', description: 'Support for 10+ languages', status: 'planned', eta: 'Q3 2024' },
  { id: '4', title: 'Advanced Analytics', description: 'Deep insights and trends', status: 'planned', eta: 'Q3 2024' },
];

const changeTypeConfig: Record<ChangeType, ChangeTypeConfig> = {
  feature: { icon: <StarOutlined />, color: '#3b82f6', label: 'New' },
  improvement: { icon: <ThunderboltOutlined />, color: '#22c55e', label: 'Improved' },
  bugfix: { icon: <BugOutlined />, color: '#f59e0b', label: 'Fixed' },
  security: { icon: <SafetyCertificateOutlined />, color: '#ef4444', label: 'Security' },
};

const versionTypeColors: Record<VersionType, string> = {
  major: 'red',
  minor: 'blue',
  patch: 'green',
};

// ============================================
// Utility Functions
// ============================================

const formatDate = (dateString: string, locale = 'en-US'): string => {
  return new Date(dateString).toLocaleDateString(locale, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
};

const getRelativeTime = (dateString: string): string => {
  const date = new Date(dateString);
  const now = new Date();
  const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));
  
  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
  return `${Math.floor(diffDays / 365)} years ago`;
};

const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
};

export const Changelog: React.FC = () => {
  const { t } = useTranslation();
  
  // ============================================
  // State Management
  // ============================================
  const [expandedVersion, setExpandedVersion] = useState<string | null>('v2.2.0');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<VersionType | 'all'>('all');
  const [showAllVersions, setShowAllVersions] = useState(false);

  // ============================================
  // Memoized Computations
  // ============================================
  
  // Filter changelog based on search and type filter
  const filteredChangelog = useMemo(() => {
    let result = changelog;
    
    // Filter by version type
    if (filterType !== 'all') {
      result = result.filter(entry => entry.type === filterType);
    }
    
    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(entry =>
        entry.version.toLowerCase().includes(query) ||
        entry.title.toLowerCase().includes(query) ||
        entry.description.toLowerCase().includes(query) ||
        entry.changes.some(change => change.text.toLowerCase().includes(query))
      );
    }
    
    return result;
  }, [searchQuery, filterType]);

  // Calculate statistics
  const statistics = useMemo(() => {
    const stats = {
      total: changelog.length,
      major: 0,
      minor: 0,
      patch: 0,
      features: 0,
      improvements: 0,
      bugfixes: 0,
      security: 0,
    };
    
    changelog.forEach(entry => {
      stats[entry.type]++;
      entry.changes.forEach(change => {
        if (change.type === 'feature') stats.features++;
        else if (change.type === 'improvement') stats.improvements++;
        else if (change.type === 'bugfix') stats.bugfixes++;
        else if (change.type === 'security') stats.security++;
      });
    });
    
    return stats;
  }, []);

  // Versions to display (limited unless showAll is true)
  const displayedVersions = useMemo(() => {
    return showAllVersions ? filteredChangelog : filteredChangelog.slice(0, 3);
  }, [filteredChangelog, showAllVersions]);

  // ============================================
  // Event Handlers
  // ============================================
  
  const handleVersionToggle = useCallback((version: string) => {
    setExpandedVersion(prev => prev === version ? null : version);
  }, []);

  const handleCopyLink = useCallback(async (version: string) => {
    const url = `${window.location.origin}/changelog#${version}`;
    const success = await copyToClipboard(url);
    if (success) {
      message.success(t('changelog.link_copied', 'Link copied to clipboard'));
    } else {
      message.error(t('changelog.copy_failed', 'Failed to copy link'));
    }
  }, [t]);

  const _handleExpandAll = useCallback(() => {
    setExpandedVersion(expandedVersion ? null : changelog[0]?.version || null);
  }, [expandedVersion]);

  const handleSearchChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  }, []);

  const handleFilterChange = useCallback((value: string | number) => {
    setFilterType(value as VersionType | 'all');
  }, []);

  return (
    <div className="changelog-page" style={{ maxWidth: 1200, margin: '0 auto', padding: '0 16px' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <HistoryOutlined style={{ color: '#2563eb' }} /> {t('changelog.title')}
          </Title>
          <Text type="secondary">{t('changelog.subtitle')}</Text>
        </div>
        <Space wrap>
          <Button icon={<RocketOutlined />}>{t('changelog.view_roadmap')}</Button>
          <Button type="primary" icon={<GiftOutlined />}>{t('changelog.subscribe')}</Button>
        </Space>
      </div>

      {/* Latest Release Banner */}
      <Alert
        type="info"
        showIcon
        icon={<GiftOutlined />}
        message={
          <Space>
            <Text strong>Latest Release: {changelog[0].version}</Text>
            <Tag color="blue">NEW</Tag>
            <Text type="secondary">• {getRelativeTime(changelog[0].date)}</Text>
          </Space>
        }
        description={changelog[0].title}
        style={{ marginBottom: 24, borderRadius: 12 }}
        action={
          <Space>
            <Tooltip title={t('changelog.copy_link', 'Copy link')}>
              <Button size="small" icon={<LinkOutlined />} onClick={() => handleCopyLink(changelog[0].version)} />
            </Tooltip>
            <Button size="small" type="primary" onClick={() => handleVersionToggle(changelog[0].version)}>
              {t('changelog.see_whats_new')}
            </Button>
          </Space>
        }
      />

      {/* Search and Filter Bar */}
      <Card style={{ marginBottom: 24, borderRadius: 12 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} sm={12} md={8}>
            <Input
              placeholder={t('changelog.search_placeholder', 'Search versions, features...')}
              prefix={<SearchOutlined style={{ color: '#94a3b8' }} />}
              value={searchQuery}
              onChange={handleSearchChange}
              allowClear
              style={{ borderRadius: 8 }}
            />
          </Col>
          <Col xs={24} sm={12} md={10}>
            <Space>
              <FilterOutlined style={{ color: '#64748b' }} />
              <Segmented
                options={[
                  { label: 'All', value: 'all' },
                  { label: 'Major', value: 'major' },
                  { label: 'Minor', value: 'minor' },
                  { label: 'Patch', value: 'patch' },
                ]}
                value={filterType}
                onChange={handleFilterChange}
              />
            </Space>
          </Col>
          <Col xs={24} md={6} style={{ textAlign: 'right' }}>
            <Text type="secondary">
              {filteredChangelog.length} {filteredChangelog.length === 1 ? 'version' : 'versions'}
            </Text>
          </Col>
        </Row>
      </Card>

      <Row gutter={24}>
        {/* Changelog Timeline */}
        <Col xs={24} lg={16}>
          <Card style={{ borderRadius: 12 }}>
            {filteredChangelog.length === 0 ? (
              <Empty
                description={t('changelog.no_results', 'No versions found matching your search')}
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              />
            ) : (
              <>
                <Timeline
                  items={displayedVersions.map(entry => ({
                    color: versionTypeColors[entry.type],
                    dot: entry.isNew ? (
                      <Badge dot>
                        <Avatar size="small" style={{ background: '#2563eb' }} icon={<RocketOutlined />} />
                      </Badge>
                    ) : undefined,
                    children: (
                      <div 
                        id={entry.version} 
                        style={{ 
                          marginBottom: 24,
                          padding: 16,
                          borderRadius: 12,
                          background: expandedVersion === entry.version ? 'rgba(37, 99, 235, 0.02)' : 'transparent',
                          border: expandedVersion === entry.version ? '1px solid rgba(37, 99, 235, 0.1)' : '1px solid transparent',
                          transition: 'all 0.2s ease',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <div>
                            <Space wrap>
                              <Text strong style={{ fontSize: 18 }}>{entry.version}</Text>
                              <Tag color={versionTypeColors[entry.type]}>
                                {entry.type.toUpperCase()}
                              </Tag>
                              {entry.isNew && <Tag color="purple">NEW</Tag>}
                              {entry.breaking && <Tag color="volcano">BREAKING</Tag>}
                            </Space>
                            <div style={{ marginTop: 4 }}>
                              <Tooltip title={formatDate(entry.date)}>
                                <Text type="secondary" style={{ fontSize: 13 }}>
                                  <ClockCircleOutlined style={{ marginRight: 4 }} />
                                  {getRelativeTime(entry.date)}
                                </Text>
                              </Tooltip>
                            </div>
                          </div>
                          <Space>
                            <Tooltip title={t('changelog.copy_link', 'Copy link')}>
                              <Button
                                type="text"
                                size="small"
                                icon={<LinkOutlined />}
                                onClick={() => handleCopyLink(entry.version)}
                              />
                            </Tooltip>
                            <Button
                              type="text"
                              size="small"
                              icon={expandedVersion === entry.version ? <UpOutlined /> : <DownOutlined />}
                              onClick={() => handleVersionToggle(entry.version)}
                            >
                              {expandedVersion === entry.version ? t('changelog.collapse') : t('changelog.expand')}
                            </Button>
                          </Space>
                        </div>

                        <Title level={5} style={{ margin: '12px 0 8px' }}>{entry.title}</Title>
                        <Paragraph type="secondary" style={{ marginBottom: expandedVersion === entry.version ? 16 : 0 }}>
                          {entry.description}
                        </Paragraph>

                        {expandedVersion === entry.version && (
                          <List
                            size="small"
                            dataSource={entry.changes}
                            style={{ 
                              background: 'rgba(0,0,0,0.02)', 
                              borderRadius: 8, 
                              padding: '8px 12px',
                            }}
                            renderItem={change => {
                              const config = changeTypeConfig[change.type];
                              return (
                                <List.Item style={{ padding: '8px 0', border: 'none' }}>
                                  <Space align="start">
                                    <Tag 
                                      color={config.color} 
                                      icon={config.icon}
                                      style={{ minWidth: 90, textAlign: 'center' }}
                                    >
                                      {config.label}
                                    </Tag>
                                    <Text>{change.text}</Text>
                                  </Space>
                                </List.Item>
                              );
                            }}
                          />
                        )}
                      </div>
                    ),
                  }))}
                />
                
                {/* Show More Button */}
                {filteredChangelog.length > 3 && (
                  <div style={{ textAlign: 'center', marginTop: 16 }}>
                    <Button 
                      type="dashed" 
                      onClick={() => setShowAllVersions(!showAllVersions)}
                      icon={showAllVersions ? <UpOutlined /> : <DownOutlined />}
                    >
                      {showAllVersions 
                        ? t('changelog.show_less', 'Show Less')
                        : t('changelog.show_more', `Show ${filteredChangelog.length - 3} More Versions`)
                      }
                    </Button>
                  </div>
                )}
              </>
            )}
          </Card>
        </Col>

        {/* Sidebar */}
        <Col xs={24} lg={8}>
          {/* Statistics Card */}
          <Card 
            title={<><ToolOutlined /> {t('changelog.statistics', 'Statistics')}</>} 
            style={{ borderRadius: 12, marginBottom: 24 }}
          >
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Statistic 
                  title="Total Releases" 
                  value={statistics.total}
                  prefix={<RocketOutlined style={{ color: '#2563eb' }} />}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="Features" 
                  value={statistics.features}
                  valueStyle={{ color: '#3b82f6' }}
                  prefix={<StarOutlined />}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="Bug Fixes" 
                  value={statistics.bugfixes}
                  valueStyle={{ color: '#f59e0b' }}
                  prefix={<BugOutlined />}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="Security" 
                  value={statistics.security}
                  valueStyle={{ color: '#ef4444' }}
                  prefix={<SafetyCertificateOutlined />}
                />
              </Col>
            </Row>
            <Divider style={{ margin: '16px 0' }} />
            <Space direction="vertical" style={{ width: '100%' }} size={8}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Space>
                  <Tag color="red">MAJOR</Tag>
                  <Text type="secondary">Breaking changes</Text>
                </Space>
                <Text strong>{statistics.major}</Text>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Space>
                  <Tag color="blue">MINOR</Tag>
                  <Text type="secondary">New features</Text>
                </Space>
                <Text strong>{statistics.minor}</Text>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Space>
                  <Tag color="green">PATCH</Tag>
                  <Text type="secondary">Bug fixes</Text>
                </Space>
                <Text strong>{statistics.patch}</Text>
              </div>
            </Space>
          </Card>

          {/* Roadmap */}
          <Card 
            title={<><BulbOutlined /> {t('changelog.roadmap')}</>} 
            style={{ borderRadius: 12, marginBottom: 24 }}
            extra={<Tag color="processing">{roadmap.filter(r => r.status === 'in-progress').length} Active</Tag>}
          >
            <List
              size="small"
              dataSource={roadmap}
              renderItem={item => (
                <List.Item style={{ padding: '12px 0', display: 'block' }}>
                  <div style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <Text strong>{item.title}</Text>
                      <Tag color={
                        item.status === 'in-progress' ? 'processing' : 
                        item.status === 'completed' ? 'success' : 'default'
                      }>
                        {item.status === 'in-progress' ? t('changelog.in_progress') : 
                         item.status === 'completed' ? 'Completed' : t('changelog.planned')}
                      </Tag>
                    </div>
                    {item.description && (
                      <Text type="secondary" style={{ fontSize: 12, display: 'block' }}>
                        {item.description}
                      </Text>
                    )}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 8 }}>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        <ClockCircleOutlined style={{ marginRight: 4 }} />
                        ETA: {item.eta}
                      </Text>
                      {item.progress !== undefined && (
                        <Text type="secondary" style={{ fontSize: 12 }}>{item.progress}%</Text>
                      )}
                    </div>
                    {item.progress !== undefined && (
                      <div style={{ 
                        height: 4, 
                        background: '#e2e8f0', 
                        borderRadius: 2, 
                        marginTop: 8,
                        overflow: 'hidden'
                      }}>
                        <div style={{ 
                          height: '100%', 
                          width: `${item.progress}%`, 
                          background: '#2563eb',
                          borderRadius: 2,
                          transition: 'width 0.3s ease'
                        }} />
                      </div>
                    )}
                  </div>
                </List.Item>
              )}
            />
          </Card>

          {/* Current Version Card */}
          <Card 
            title={<><HistoryOutlined /> {t('changelog.release_stats')}</>} 
            style={{ borderRadius: 12, marginBottom: 24 }}
          >
            <Space direction="vertical" style={{ width: '100%' }} size={12}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text type="secondary">{t('changelog.current_version')}</Text>
                <Space>
                  <Text strong style={{ fontSize: 16 }}>{changelog[0].version}</Text>
                  <Tag color={versionTypeColors[changelog[0].type]}>{changelog[0].type}</Tag>
                </Space>
              </div>
              <Divider style={{ margin: '4px 0' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text type="secondary">{t('changelog.last_updated')}</Text>
                <Tooltip title={formatDate(changelog[0].date)}>
                  <Text>{getRelativeTime(changelog[0].date)}</Text>
                </Tooltip>
              </div>
              <Divider style={{ margin: '4px 0' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text type="secondary">Changes in Latest</Text>
                <Text>{changelog[0].changes.length} items</Text>
              </div>
            </Space>
          </Card>

          {/* Quick Links */}
          <Card title={t('changelog.quick_links')} style={{ borderRadius: 12 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button 
                block 
                type="text" 
                icon={<GiftOutlined />} 
                style={{ textAlign: 'left', height: 40 }}
              >
                {t('changelog.release_notes')}
              </Button>
              <Button 
                block 
                type="text" 
                icon={<BugOutlined />} 
                style={{ textAlign: 'left', height: 40 }}
                onClick={() => window.open('https://github.com/issues', '_blank')}
              >
                {t('changelog.report_issue')}
              </Button>
              <Button 
                block 
                type="text" 
                icon={<BulbOutlined />} 
                style={{ textAlign: 'left', height: 40 }}
              >
                {t('changelog.request_feature')}
              </Button>
              <Divider style={{ margin: '8px 0' }} />
              <Button 
                block 
                type="text" 
                icon={<CopyOutlined />} 
                style={{ textAlign: 'left', height: 40 }}
                onClick={() => handleCopyLink(changelog[0].version)}
              >
                Copy Latest Version Link
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Changelog;
