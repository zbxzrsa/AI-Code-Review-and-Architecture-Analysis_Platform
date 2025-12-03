/**
 * Changelog Page
 * 更新日志页面
 * 
 * Features:
 * - Version history
 * - New features announcements
 * - Bug fixes and improvements
 * - Roadmap preview
 */

import React, { useState } from 'react';
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
} from 'antd';
import {
  HistoryOutlined,
  RocketOutlined,
  BugOutlined,
  ThunderboltOutlined,
  SafetyCertificateOutlined,
  StarOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  GiftOutlined,
  BulbOutlined,
  ToolOutlined,
  EyeOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text, Paragraph } = Typography;

interface ChangelogEntry {
  version: string;
  date: string;
  type: 'major' | 'minor' | 'patch';
  title: string;
  description: string;
  changes: {
    type: 'feature' | 'improvement' | 'bugfix' | 'security';
    text: string;
  }[];
  isNew?: boolean;
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

const roadmap = [
  { title: 'IDE Extensions', status: 'in-progress', eta: 'Q2 2024' },
  { title: 'Custom AI Model Training', status: 'planned', eta: 'Q2 2024' },
  { title: 'Multi-language Support', status: 'planned', eta: 'Q3 2024' },
  { title: 'Advanced Analytics', status: 'planned', eta: 'Q3 2024' },
];

const changeTypeConfig = {
  feature: { icon: <StarOutlined />, color: '#3b82f6', label: 'New' },
  improvement: { icon: <ThunderboltOutlined />, color: '#22c55e', label: 'Improved' },
  bugfix: { icon: <BugOutlined />, color: '#f59e0b', label: 'Fixed' },
  security: { icon: <SafetyCertificateOutlined />, color: '#ef4444', label: 'Security' },
};

export const Changelog: React.FC = () => {
  const { t } = useTranslation();
  const [expandedVersion, setExpandedVersion] = useState<string | null>('v2.2.0');

  return (
    <div className="changelog-page" style={{ maxWidth: 1000, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <HistoryOutlined style={{ color: '#2563eb' }} /> Changelog
          </Title>
          <Text type="secondary">What's new and improved in the platform</Text>
        </div>
        <Space>
          <Button icon={<RocketOutlined />}>View Roadmap</Button>
          <Button type="primary" icon={<GiftOutlined />}>Subscribe to Updates</Button>
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
          </Space>
        }
        description={changelog[0].title}
        style={{ marginBottom: 24, borderRadius: 12 }}
        action={
          <Button size="small" type="primary">
            See What's New
          </Button>
        }
      />

      <Row gutter={24}>
        {/* Changelog Timeline */}
        <Col xs={24} lg={16}>
          <Card style={{ borderRadius: 12 }}>
            <Timeline
              items={changelog.map(entry => ({
                color: entry.type === 'major' ? 'red' : entry.type === 'minor' ? 'blue' : 'green',
                dot: entry.isNew ? (
                  <Badge dot>
                    <Avatar size="small" style={{ background: '#2563eb' }} icon={<RocketOutlined />} />
                  </Badge>
                ) : undefined,
                children: (
                  <div style={{ marginBottom: 24 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div>
                        <Space>
                          <Text strong style={{ fontSize: 18 }}>{entry.version}</Text>
                          <Tag color={
                            entry.type === 'major' ? 'red' :
                            entry.type === 'minor' ? 'blue' : 'green'
                          }>
                            {entry.type.toUpperCase()}
                          </Tag>
                          {entry.isNew && <Tag color="purple">NEW</Tag>}
                        </Space>
                        <div>
                          <Text type="secondary">
                            <ClockCircleOutlined style={{ marginRight: 4 }} />
                            {new Date(entry.date).toLocaleDateString('en-US', {
                              year: 'numeric',
                              month: 'long',
                              day: 'numeric',
                            })}
                          </Text>
                        </div>
                      </div>
                      <Button
                        type="text"
                        size="small"
                        onClick={() => setExpandedVersion(
                          expandedVersion === entry.version ? null : entry.version
                        )}
                      >
                        {expandedVersion === entry.version ? 'Collapse' : 'Expand'}
                      </Button>
                    </div>

                    <Title level={5} style={{ margin: '8px 0' }}>{entry.title}</Title>
                    <Paragraph type="secondary">{entry.description}</Paragraph>

                    {expandedVersion === entry.version && (
                      <List
                        size="small"
                        dataSource={entry.changes}
                        renderItem={change => {
                          const config = changeTypeConfig[change.type];
                          return (
                            <List.Item style={{ padding: '8px 0', border: 'none' }}>
                              <Space>
                                <Tag color={config.color} icon={config.icon}>
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
          </Card>
        </Col>

        {/* Sidebar */}
        <Col xs={24} lg={8}>
          {/* Roadmap */}
          <Card title={<><BulbOutlined /> Roadmap</>} style={{ borderRadius: 12, marginBottom: 24 }}>
            <List
              size="small"
              dataSource={roadmap}
              renderItem={item => (
                <List.Item style={{ padding: '12px 0' }}>
                  <div style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text strong>{item.title}</Text>
                      <Tag color={item.status === 'in-progress' ? 'processing' : 'default'}>
                        {item.status === 'in-progress' ? 'In Progress' : 'Planned'}
                      </Tag>
                    </div>
                    <Text type="secondary" style={{ fontSize: 12 }}>ETA: {item.eta}</Text>
                  </div>
                </List.Item>
              )}
            />
          </Card>

          {/* Version Stats */}
          <Card title={<><ToolOutlined /> Release Stats</>} style={{ borderRadius: 12, marginBottom: 24 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text type="secondary">Current Version</Text>
                <Text strong>{changelog[0].version}</Text>
              </div>
              <Divider style={{ margin: '8px 0' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text type="secondary">Total Releases</Text>
                <Text>{changelog.length}</Text>
              </div>
              <Divider style={{ margin: '8px 0' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text type="secondary">Last Updated</Text>
                <Text>{new Date(changelog[0].date).toLocaleDateString()}</Text>
              </div>
            </Space>
          </Card>

          {/* Quick Links */}
          <Card title="Quick Links" style={{ borderRadius: 12 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button block type="text" icon={<GiftOutlined />} style={{ textAlign: 'left' }}>
                Release Notes
              </Button>
              <Button block type="text" icon={<BugOutlined />} style={{ textAlign: 'left' }}>
                Report an Issue
              </Button>
              <Button block type="text" icon={<BulbOutlined />} style={{ textAlign: 'left' }}>
                Request a Feature
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Changelog;
