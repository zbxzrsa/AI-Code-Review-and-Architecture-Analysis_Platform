/**
 * Onboarding Page
 * 新手引导页面
 * 
 * Features:
 * - Step-by-step setup wizard
 * - Quick start guide
 * - Feature highlights
 * - Integration setup
 */

import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Steps,
  Progress,
  Checkbox,
  Avatar,
} from 'antd';
import {
  RocketOutlined,
  CheckCircleOutlined,
  GithubOutlined,
  CodeOutlined,
  SafetyCertificateOutlined,
  TeamOutlined,
  SettingOutlined,
  ArrowRightOutlined,
  PlayCircleOutlined,
  BulbOutlined,
  ThunderboltOutlined,
  ApiOutlined,
  BookOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';

const { Title, Text, Paragraph } = Typography;

interface SetupStep {
  key: string;
  title: string;
  description: string;
  completed: boolean;
  icon: React.ReactNode;
  action: string;
  path: string;
}

const setupSteps: SetupStep[] = [
  {
    key: 'connect',
    title: 'Connect Repository',
    description: 'Link your GitHub or GitLab repositories',
    completed: false,
    icon: <GithubOutlined />,
    action: 'Connect',
    path: '/repositories',
  },
  {
    key: 'review',
    title: 'Run First Analysis',
    description: 'Analyze your code with AI-powered review',
    completed: false,
    icon: <CodeOutlined />,
    action: 'Analyze',
    path: '/review',
  },
  {
    key: 'rules',
    title: 'Configure Rules',
    description: 'Customize code quality rules for your project',
    completed: false,
    icon: <SettingOutlined />,
    action: 'Configure',
    path: '/rules',
  },
  {
    key: 'team',
    title: 'Invite Team',
    description: 'Collaborate with your team members',
    completed: false,
    icon: <TeamOutlined />,
    action: 'Invite',
    path: '/teams',
  },
  {
    key: 'integrate',
    title: 'Setup CI/CD',
    description: 'Integrate with your deployment pipeline',
    completed: false,
    icon: <RocketOutlined />,
    action: 'Setup',
    path: '/settings/integrations',
  },
];

const features = [
  {
    icon: <SafetyCertificateOutlined />,
    title: 'Security Scanning',
    description: 'Detect vulnerabilities and security issues automatically',
    color: '#ef4444',
  },
  {
    icon: <ThunderboltOutlined />,
    title: 'AI Auto-Fix',
    description: 'Let AI fix bugs and vulnerabilities for you',
    color: '#8b5cf6',
  },
  {
    icon: <CodeOutlined />,
    title: 'Code Review',
    description: 'Get intelligent suggestions to improve code quality',
    color: '#3b82f6',
  },
  {
    icon: <ApiOutlined />,
    title: 'API Integration',
    description: 'Integrate with your existing tools and workflows',
    color: '#22c55e',
  },
];

export const Onboarding: React.FC = () => {
  const { t: _t } = useTranslation();
  const navigate = useNavigate();
  const [steps, setSteps] = useState(setupSteps);
  const [currentStep, setCurrentStep] = useState(0);

  const completedCount = steps.filter(s => s.completed).length;
  const progress = Math.round((completedCount / steps.length) * 100);

  const toggleStep = (key: string) => {
    setSteps(prev => prev.map(s =>
      s.key === key ? { ...s, completed: !s.completed } : s
    ));
  };

  return (
    <div className="onboarding-page" style={{ maxWidth: 1200, margin: '0 auto' }}>
      {/* Welcome Header */}
      <Card
        style={{
          borderRadius: 16,
          marginBottom: 24,
          background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
          border: 'none',
        }}
      >
        <Row gutter={24} align="middle">
          <Col xs={24} md={16}>
            <Space direction="vertical" size={16}>
              <Title level={2} style={{ color: 'white', margin: 0 }}>
                Welcome to AI Code Review Platform! 🎉
              </Title>
              <Paragraph style={{ color: 'rgba(255,255,255,0.9)', fontSize: 16, margin: 0 }}>
                Let&apos;s get you set up in just a few minutes. Follow the steps below to unlock
                the full power of AI-powered code review.
              </Paragraph>
              <Space>
                <Button type="primary" size="large" ghost icon={<PlayCircleOutlined />}>
                  Watch Demo
                </Button>
                <Button size="large" style={{ background: 'rgba(255,255,255,0.2)', border: 'none', color: 'white' }} icon={<BookOutlined />}>
                  Read Docs
                </Button>
              </Space>
            </Space>
          </Col>
          <Col xs={24} md={8} style={{ textAlign: 'center' }}>
            <div style={{ padding: 24 }}>
              <Progress
                type="circle"
                percent={progress}
                size={140}
                strokeColor="#fff"
                trailColor="rgba(255,255,255,0.2)"
                format={() => (
                  <div style={{ color: 'white' }}>
                    <div style={{ fontSize: 32, fontWeight: 700 }}>{completedCount}/{steps.length}</div>
                    <div style={{ fontSize: 14 }}>Steps Done</div>
                  </div>
                )}
              />
            </div>
          </Col>
        </Row>
      </Card>

      <Row gutter={24}>
        {/* Setup Steps */}
        <Col xs={24} lg={14}>
          <Card title={<><RocketOutlined /> Quick Setup</>} style={{ borderRadius: 12, marginBottom: 24 }}>
            <Steps
              direction="vertical"
              current={currentStep}
              onChange={setCurrentStep}
              items={steps.map((step, index) => ({
                title: (
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                    <Space>
                      <Checkbox
                        checked={step.completed}
                        onChange={() => toggleStep(step.key)}
                      />
                      <Text strong style={{ textDecoration: step.completed ? 'line-through' : 'none' }}>
                        {step.title}
                      </Text>
                    </Space>
                    <Button
                      type={step.completed ? 'default' : 'primary'}
                      size="small"
                      onClick={() => navigate(step.path)}
                    >
                      {step.action} <ArrowRightOutlined />
                    </Button>
                  </div>
                ),
                description: (
                  <Text type="secondary">{step.description}</Text>
                ),
                icon: (
                  <Avatar
                    style={{
                      background: step.completed ? '#22c55e' : '#e2e8f0',
                      color: step.completed ? 'white' : '#64748b',
                    }}
                    icon={step.completed ? <CheckCircleOutlined /> : step.icon}
                  />
                ),
                status: step.completed ? 'finish' : index === currentStep ? 'process' : 'wait',
              }))}
            />
          </Card>

          {/* Quick Actions */}
          <Card title={<><BulbOutlined /> Quick Actions</>} style={{ borderRadius: 12 }}>
            <Row gutter={[16, 16]}>
              {[
                { icon: <GithubOutlined />, label: 'Connect GitHub', path: '/repositories' },
                { icon: <CodeOutlined />, label: 'Start Review', path: '/review' },
                { icon: <SafetyCertificateOutlined />, label: 'Security Scan', path: '/security' },
                { icon: <ApiOutlined />, label: 'Get API Key', path: '/settings/api-keys' },
              ].map((action, index) => (
                <Col key={index} xs={12} sm={6}>
                  <Button
                    block
                    style={{ height: 80, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}
                    onClick={() => navigate(action.path)}
                  >
                    <span style={{ fontSize: 24, marginBottom: 8 }}>{action.icon}</span>
                    <span style={{ fontSize: 12 }}>{action.label}</span>
                  </Button>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* Features Highlight */}
        <Col xs={24} lg={10}>
          <Card title={<><ThunderboltOutlined /> Key Features</>} style={{ borderRadius: 12, marginBottom: 24 }}>
            <Space direction="vertical" style={{ width: '100%' }} size={16}>
              {features.map((feature, index) => (
                <div
                  key={index}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 16,
                    padding: 16,
                    background: '#f8fafc',
                    borderRadius: 12,
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: 12,
                      background: `${feature.color}15`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: feature.color,
                      fontSize: 24,
                    }}
                  >
                    {feature.icon}
                  </div>
                  <div>
                    <Text strong>{feature.title}</Text>
                    <div>
                      <Text type="secondary" style={{ fontSize: 13 }}>
                        {feature.description}
                      </Text>
                    </div>
                  </div>
                </div>
              ))}
            </Space>
          </Card>

          {/* Resources */}
          <Card title="Resources" style={{ borderRadius: 12 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button block type="text" style={{ textAlign: 'left', height: 'auto', padding: '12px 16px' }}>
                <Space>
                  <BookOutlined style={{ color: '#2563eb' }} />
                  <div>
                    <div><Text strong>Documentation</Text></div>
                    <Text type="secondary" style={{ fontSize: 12 }}>Learn how to use the platform</Text>
                  </div>
                </Space>
              </Button>
              <Button block type="text" style={{ textAlign: 'left', height: 'auto', padding: '12px 16px' }}>
                <Space>
                  <ApiOutlined style={{ color: '#22c55e' }} />
                  <div>
                    <div><Text strong>API Reference</Text></div>
                    <Text type="secondary" style={{ fontSize: 12 }}>Integrate with your tools</Text>
                  </div>
                </Space>
              </Button>
              <Button block type="text" style={{ textAlign: 'left', height: 'auto', padding: '12px 16px' }}>
                <Space>
                  <TeamOutlined style={{ color: '#8b5cf6' }} />
                  <div>
                    <div><Text strong>Community</Text></div>
                    <Text type="secondary" style={{ fontSize: 12 }}>Join our Discord community</Text>
                  </div>
                </Space>
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Onboarding;
