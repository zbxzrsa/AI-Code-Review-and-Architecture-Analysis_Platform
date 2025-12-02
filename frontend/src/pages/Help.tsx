import React, { useState } from 'react';
import {
  Card,
  Typography,
  Collapse,
  Input,
  Row,
  Col,
  Space,
  Button,
  Divider,
  Tag,
} from 'antd';
import {
  SearchOutlined,
  BookOutlined,
  QuestionCircleOutlined,
  RocketOutlined,
  SafetyOutlined,
  ApiOutlined,
  SettingOutlined,
  BugOutlined,
  GithubOutlined,
  MailOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import './Help.css';

const { Title, Text, Paragraph, Link } = Typography;
const { Search } = Input;
const { Panel } = Collapse;

interface FAQ {
  question: string;
  answer: string;
  tags: string[];
}

const faqs: FAQ[] = [
  {
    question: 'How do I start a code review?',
    answer: 'Navigate to the Code Review page, paste your code or upload a file, select the programming language, and click "Analyze". The AI will process your code and provide detailed feedback on issues, suggestions, and improvements.',
    tags: ['getting-started', 'code-review'],
  },
  {
    question: 'What programming languages are supported?',
    answer: 'We support over 20 programming languages including Python, JavaScript, TypeScript, Java, C++, Go, Rust, Ruby, PHP, and more. The AI models are trained to understand language-specific patterns and best practices.',
    tags: ['languages', 'features'],
  },
  {
    question: 'How does the AI provider priority work?',
    answer: 'The platform uses a priority chain for AI providers. By default, it uses local Ollama (free) first, then falls back to cloud providers like OpenAI or Anthropic if needed. You can configure your own API keys in Settings.',
    tags: ['ai', 'providers', 'configuration'],
  },
  {
    question: 'Can I use my own API keys?',
    answer: 'Yes! Go to Settings > API Keys and add your OpenAI, Anthropic, or HuggingFace API keys. When available, your keys will be used instead of the default providers.',
    tags: ['api-keys', 'configuration'],
  },
  {
    question: 'What is the difference between V1, V2, and V3?',
    answer: 'V1 is the experimentation zone where new AI models are tested. V2 is the stable production environment with strict SLOs. V3 is the quarantine zone for failed experiments. Regular users only interact with V2.',
    tags: ['architecture', 'versions'],
  },
  {
    question: 'How do I report a bug or request a feature?',
    answer: 'You can report bugs or request features through our GitHub repository. Click the "Report Issue" button below or visit our GitHub page directly.',
    tags: ['support', 'feedback'],
  },
  {
    question: 'Is my code stored or shared?',
    answer: 'We take privacy seriously. Code submitted for review is processed in memory and is not permanently stored unless you explicitly save it to a project. We never share your code with third parties.',
    tags: ['privacy', 'security'],
  },
  {
    question: 'How accurate is the AI analysis?',
    answer: 'Our AI models achieve over 90% accuracy in detecting common code issues. The V2 production model is continuously evaluated and must maintain strict quality metrics before deployment.',
    tags: ['accuracy', 'quality'],
  },
];

interface GuideCard {
  title: string;
  description: string;
  icon: React.ReactNode;
  link: string;
}

const guides: GuideCard[] = [
  {
    title: 'Getting Started',
    description: 'Learn the basics of using the platform',
    icon: <RocketOutlined />,
    link: '/docs/getting-started',
  },
  {
    title: 'Code Review Guide',
    description: 'Best practices for effective code reviews',
    icon: <BugOutlined />,
    link: '/docs/code-review',
  },
  {
    title: 'API Reference',
    description: 'Integrate with our REST API',
    icon: <ApiOutlined />,
    link: '/docs/api',
  },
  {
    title: 'Security Best Practices',
    description: 'Keep your code secure',
    icon: <SafetyOutlined />,
    link: '/docs/security',
  },
  {
    title: 'Configuration Guide',
    description: 'Customize the platform for your needs',
    icon: <SettingOutlined />,
    link: '/docs/configuration',
  },
  {
    title: 'Troubleshooting',
    description: 'Common issues and solutions',
    icon: <QuestionCircleOutlined />,
    link: '/docs/troubleshooting',
  },
];

export const Help: React.FC = () => {
  const { t } = useTranslation();
  const [searchTerm, setSearchTerm] = useState('');

  const filteredFaqs = faqs.filter(
    faq =>
      faq.question.toLowerCase().includes(searchTerm.toLowerCase()) ||
      faq.answer.toLowerCase().includes(searchTerm.toLowerCase()) ||
      faq.tags.some(tag => tag.includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="help-container">
      <div className="help-header">
        <Title level={2}>
          <BookOutlined /> {t('help.title', 'Help Center')}
        </Title>
        <Paragraph type="secondary">
          {t('help.subtitle', 'Find answers to common questions and learn how to use the platform')}
        </Paragraph>
      </div>

      <Search
        placeholder={t('help.search_placeholder', 'Search for help...')}
        size="large"
        prefix={<SearchOutlined />}
        onChange={e => setSearchTerm(e.target.value)}
        style={{ marginBottom: 32, maxWidth: 500 }}
      />

      {/* Quick Links */}
      <Title level={4}>{t('help.guides', 'Documentation & Guides')}</Title>
      <Row gutter={[16, 16]} style={{ marginBottom: 32 }}>
        {guides.map((guide, index) => (
          <Col xs={24} sm={12} md={8} key={index}>
            <Card hoverable className="guide-card">
              <Space direction="vertical" size="small">
                <Text className="guide-icon">{guide.icon}</Text>
                <Text strong>{guide.title}</Text>
                <Text type="secondary" style={{ fontSize: 13 }}>
                  {guide.description}
                </Text>
              </Space>
            </Card>
          </Col>
        ))}
      </Row>

      <Divider />

      {/* FAQ Section */}
      <Title level={4}>
        <QuestionCircleOutlined /> {t('help.faq', 'Frequently Asked Questions')}
      </Title>
      
      {filteredFaqs.length === 0 ? (
        <Card>
          <Text type="secondary">
            {t('help.no_results', 'No results found. Try a different search term.')}
          </Text>
        </Card>
      ) : (
        <Collapse accordion className="faq-collapse">
          {filteredFaqs.map((faq, index) => (
            <Panel
              header={
                <Space>
                  <Text strong>{faq.question}</Text>
                </Space>
              }
              key={index}
              extra={faq.tags.map(tag => (
                <Tag key={tag} color="blue" style={{ marginRight: 4 }}>
                  {tag}
                </Tag>
              ))}
            >
              <Paragraph>{faq.answer}</Paragraph>
            </Panel>
          ))}
        </Collapse>
      )}

      <Divider />

      {/* Contact Section */}
      <Card className="contact-card">
        <Title level={4}>{t('help.still_need_help', 'Still need help?')}</Title>
        <Paragraph type="secondary">
          {t('help.contact_description', "Can't find what you're looking for? Get in touch with us.")}
        </Paragraph>
        <Space size="middle">
          <Button icon={<GithubOutlined />} href="https://github.com" target="_blank">
            {t('help.github', 'GitHub Issues')}
          </Button>
          <Button icon={<MailOutlined />} href="mailto:support@example.com">
            {t('help.email', 'Email Support')}
          </Button>
        </Space>
      </Card>
    </div>
  );
};

export default Help;
