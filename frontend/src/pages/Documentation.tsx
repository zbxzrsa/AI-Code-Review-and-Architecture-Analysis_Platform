/**
 * Documentation & Knowledge Base Page
 * 文档与知识库页面
 * 
 * Features:
 * - API documentation
 * - Getting started guides
 * - Best practices
 * - Searchable knowledge base
 */

import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Input,
  Menu,
  Breadcrumb,
  Tag,
  Button,
  Collapse,
  Divider,
  Alert,
  List,
  Avatar,
} from 'antd';
import {
  BookOutlined,
  RocketOutlined,
  ApiOutlined,
  SafetyCertificateOutlined,
  QuestionCircleOutlined,
  BulbOutlined,
  FileTextOutlined,
  TeamOutlined,
  LinkOutlined,
  RightOutlined,
  HomeOutlined,
  BranchesOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import './Documentation.css';

const { Title, Text, Paragraph } = Typography;

interface DocSection {
  key: string;
  title: string;
  icon: React.ReactNode;
  description: string;
  articles: { title: string; slug: string; tags?: string[] }[];
}

const docSections: DocSection[] = [
  {
    key: 'getting-started',
    title: 'Getting Started',
    icon: <RocketOutlined />,
    description: 'Quick guides to get up and running',
    articles: [
      { title: 'Quick Start Guide', slug: 'quick-start', tags: ['beginner'] },
      { title: 'Installation & Setup', slug: 'installation', tags: ['setup'] },
      { title: 'Your First Code Review', slug: 'first-review', tags: ['tutorial'] },
      { title: 'Understanding AI Analysis', slug: 'ai-analysis', tags: ['ai'] },
    ],
  },
  {
    key: 'api-reference',
    title: 'API Reference',
    icon: <ApiOutlined />,
    description: 'Complete API documentation',
    articles: [
      { title: 'Authentication', slug: 'api-auth', tags: ['api', 'security'] },
      { title: 'Code Analysis Endpoints', slug: 'api-analysis', tags: ['api'] },
      { title: 'Projects API', slug: 'api-projects', tags: ['api'] },
      { title: 'Webhooks', slug: 'api-webhooks', tags: ['api', 'integration'] },
      { title: 'Rate Limits', slug: 'api-rate-limits', tags: ['api'] },
    ],
  },
  {
    key: 'integrations',
    title: 'Integrations',
    icon: <LinkOutlined />,
    description: 'Connect with your tools',
    articles: [
      { title: 'GitHub Integration', slug: 'github', tags: ['git', 'integration'] },
      { title: 'GitLab Integration', slug: 'gitlab', tags: ['git', 'integration'] },
      { title: 'VS Code Extension', slug: 'vscode', tags: ['ide'] },
      { title: 'CI/CD Pipelines', slug: 'cicd', tags: ['automation'] },
      { title: 'Slack Notifications', slug: 'slack', tags: ['notification'] },
    ],
  },
  {
    key: 'security',
    title: 'Security',
    icon: <SafetyCertificateOutlined />,
    description: 'Security best practices',
    articles: [
      { title: 'Security Overview', slug: 'security-overview', tags: ['security'] },
      { title: 'Vulnerability Detection', slug: 'vuln-detection', tags: ['security'] },
      { title: 'OWASP Compliance', slug: 'owasp', tags: ['compliance'] },
      { title: 'Secret Scanning', slug: 'secrets', tags: ['security'] },
    ],
  },
  {
    key: 'best-practices',
    title: 'Best Practices',
    icon: <BulbOutlined />,
    description: 'Optimize your workflow',
    articles: [
      { title: 'Code Review Guidelines', slug: 'review-guidelines', tags: ['guide'] },
      { title: 'AI Prompt Engineering', slug: 'prompt-engineering', tags: ['ai'] },
      { title: 'Custom Rules Setup', slug: 'custom-rules', tags: ['configuration'] },
      { title: 'Team Collaboration', slug: 'collaboration', tags: ['team'] },
    ],
  },
  {
    key: 'three-version',
    title: 'Three-Version Evolution',
    icon: <BranchesOutlined />,
    description: 'Self-evolving AI architecture',
    articles: [
      { title: 'Architecture Overview', slug: 'three-version-overview', tags: ['architecture', 'admin'] },
      { title: 'V1 Experimentation Zone', slug: 'v1-experimentation', tags: ['v1', 'experiments'] },
      { title: 'V2 Production Zone', slug: 'v2-production', tags: ['v2', 'production'] },
      { title: 'V3 Quarantine Zone', slug: 'v3-quarantine', tags: ['v3', 'quarantine'] },
      { title: 'Dual-AI System', slug: 'dual-ai', tags: ['ai', 'vcai', 'crai'] },
      { title: 'Spiral Evolution Cycle', slug: 'spiral-evolution', tags: ['evolution', 'cycle'] },
      { title: 'Promotion & Degradation', slug: 'promotion-degradation', tags: ['promotion', 'degradation'] },
      { title: 'Admin Control Panel', slug: 'admin-control', tags: ['admin', 'ui'] },
    ],
  },
];

const faqs = [
  {
    question: 'How does AI code review work?',
    answer: 'Our AI analyzes your code using advanced language models to identify bugs, security vulnerabilities, and code quality issues. It provides suggestions and can even auto-fix certain problems.',
  },
  {
    question: 'What programming languages are supported?',
    answer: 'We support 20+ languages including Python, JavaScript, TypeScript, Java, Go, Rust, C++, Ruby, PHP, and more. The AI adapts its analysis based on language-specific patterns.',
  },
  {
    question: 'Is my code secure?',
    answer: 'Yes, we use enterprise-grade encryption for all data. Your code is processed securely and never stored permanently. We are SOC 2 Type II compliant.',
  },
  {
    question: 'How do I integrate with my CI/CD pipeline?',
    answer: 'We provide native integrations with GitHub Actions, GitLab CI, Jenkins, and other CI/CD tools. Simply add our action/step to your pipeline configuration.',
  },
  {
    question: 'Can I customize the analysis rules?',
    answer: 'Yes, you can create custom rules, disable specific checks, and adjust severity levels. Rules can be configured globally or per-project.',
  },
  {
    question: 'What is the Three-Version Evolution system?',
    answer: 'The Three-Version system enables safe AI experimentation through V1 (experiments), V2 (production), and V3 (quarantine) zones. New technologies are tested in V1, promoted to V2 when proven, and quarantined in V3 if they fail. This ensures zero-error user experience while enabling continuous AI improvement.',
  },
  {
    question: 'How do the Dual-AI models work?',
    answer: 'Each version has two AI instances: VC-AI (Version Control AI) for admin-only version management decisions, and CR-AI (Code Review AI) for user-facing code analysis. Users only access V2 CR-AI, ensuring stable production quality.',
  },
];

export const Documentation: React.FC = () => {
  const { t: _t } = useTranslation();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSection, setSelectedSection] = useState('getting-started');

  const currentSection = docSections.find(s => s.key === selectedSection);

  return (
    <div className="documentation-page" style={{ maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <BookOutlined style={{ color: '#2563eb' }} /> Documentation
          </Title>
          <Text type="secondary">Guides, API reference, and resources</Text>
        </div>
        <Input.Search
          placeholder="Search documentation..."
          style={{ width: 320 }}
          size="large"
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          allowClear
        />
      </div>

      {/* Quick Links */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        {[
          { icon: <RocketOutlined />, title: 'Quick Start', color: '#22c55e', key: 'getting-started' },
          { icon: <ApiOutlined />, title: 'API Docs', color: '#3b82f6', key: 'api-reference' },
          { icon: <BranchesOutlined />, title: 'Three-Version', color: '#8b5cf6', key: 'three-version' },
          { icon: <QuestionCircleOutlined />, title: 'FAQ', color: '#f59e0b', key: 'faq' },
        ].map(item => (
          <Col key={item.title} xs={12} sm={6}>
            <Card
              hoverable
              style={{ borderRadius: 12, textAlign: 'center' }}
              onClick={() => item.key !== 'faq' && setSelectedSection(item.key)}
            >
              <div style={{
                width: 48,
                height: 48,
                borderRadius: '50%',
                background: `${item.color}15`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 12px',
                fontSize: 24,
                color: item.color,
              }}>
                {item.icon}
              </div>
              <Text strong>{item.title}</Text>
            </Card>
          </Col>
        ))}
      </Row>

      <Row gutter={24}>
        {/* Sidebar Navigation */}
        <Col xs={24} md={6}>
          <Card style={{ borderRadius: 12 }} bodyStyle={{ padding: 0 }}>
            <Menu
              mode="inline"
              selectedKeys={[selectedSection]}
              style={{ border: 'none', borderRadius: 12 }}
              items={docSections.map(section => ({
                key: section.key,
                icon: section.icon,
                label: section.title,
                onClick: () => setSelectedSection(section.key),
              }))}
            />
          </Card>
        </Col>

        {/* Main Content */}
        <Col xs={24} md={18}>
          {currentSection && (
            <Card style={{ borderRadius: 12 }}>
              <Breadcrumb
                items={[
                  { href: '#', title: <><HomeOutlined /> Docs</> },
                  { title: currentSection.title },
                ]}
                style={{ marginBottom: 16 }}
              />

              <Title level={4}>
                {currentSection.icon} {currentSection.title}
              </Title>
              <Paragraph type="secondary">{currentSection.description}</Paragraph>

              <Divider />

              <List
                itemLayout="horizontal"
                dataSource={currentSection.articles}
                renderItem={article => (
                  <List.Item
                    style={{
                      padding: '16px',
                      borderRadius: 8,
                      marginBottom: 8,
                      cursor: 'pointer',
                      transition: 'background 0.2s',
                    }}
                    className="doc-article"
                  >
                    <List.Item.Meta
                      avatar={
                        <Avatar
                          style={{ background: '#eff6ff' }}
                          icon={<FileTextOutlined style={{ color: '#2563eb' }} />}
                        />
                      }
                      title={
                        <Space>
                          <Text strong>{article.title}</Text>
                          <RightOutlined style={{ fontSize: 12, color: '#94a3b8' }} />
                        </Space>
                      }
                      description={
                        <Space size={4}>
                          {article.tags?.map(tag => (
                            <Tag key={tag} style={{ fontSize: 11 }}>{tag}</Tag>
                          ))}
                        </Space>
                      }
                    />
                  </List.Item>
                )}
              />
            </Card>
          )}

          {/* FAQ Section */}
          <Card title={<><QuestionCircleOutlined /> Frequently Asked Questions</>} style={{ borderRadius: 12, marginTop: 24 }}>
            <Collapse
              ghost
              expandIconPosition="end"
              items={faqs.map((faq, index) => ({
                key: index.toString(),
                label: <Text strong>{faq.question}</Text>,
                children: <Paragraph>{faq.answer}</Paragraph>,
              }))}
            />
          </Card>

          {/* Help Banner */}
          <Alert
            type="info"
            showIcon
            icon={<TeamOutlined />}
            message="Need more help?"
            description={
              <Space>
                <Text>Can&apos;t find what you&apos;re looking for?</Text>
                <Button type="primary" size="small">Contact Support</Button>
                <Button size="small">Join Community</Button>
              </Space>
            }
            style={{ marginTop: 24, borderRadius: 12 }}
          />
        </Col>
      </Row>

      <style>{`
        .doc-article:hover {
          background: #f8fafc !important;
        }
      `}</style>
    </div>
  );
};

export default Documentation;
