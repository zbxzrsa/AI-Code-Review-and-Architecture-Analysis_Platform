/**
 * Welcome Dashboard
 * 
 * A beautiful, artistic landing dashboard with:
 * - Soothing color gradients
 * - Glassmorphism effects
 * - Smooth animations
 * - Comprehensive overview widgets
 */

import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Typography, Space, Button, Avatar, Badge, Tooltip } from 'antd';
import {
  CodeOutlined,
  SafetyCertificateOutlined,
  RocketOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  BellOutlined,
  ArrowRightOutlined,
  StarFilled,
  FireFilled,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import {
  GradientStat,
  ProgressRing,
  StatusBadge,
  ArtisticTimeline,
  MetricBar,
} from '../components/common/ArtisticWidgets';

const { Title, Text, Paragraph } = Typography;

const WelcomeDashboard: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { user } = useAuthStore();
  const [greeting, setGreeting] = useState('');

  // Set greeting based on time of day
  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting('Good Morning');
    else if (hour < 18) setGreeting('Good Afternoon');
    else setGreeting('Good Evening');
  }, []);

  // Quick actions
  const quickActions = [
    { icon: <CodeOutlined />, label: 'New Review', path: '/code-review', color: '#3b82f6' },
    { icon: <SafetyCertificateOutlined />, label: 'Security Scan', path: '/admin/security', color: '#10b981' },
    { icon: <RocketOutlined />, label: 'Deploy', path: '/admin/evolution', color: '#8b5cf6' },
    { icon: <ThunderboltOutlined />, label: 'Auto-Fix', path: '/admin/auto-fix', color: '#f59e0b' },
  ];

  // Recent activities
  const activities = [
    { icon: <CheckCircleOutlined />, title: 'Code review completed', description: 'backend-api/auth.py', time: '2 min ago', status: 'success' as const },
    { icon: <ThunderboltOutlined />, title: 'Auto-fix applied', description: '3 vulnerabilities fixed', time: '15 min ago', status: 'info' as const },
    { icon: <SafetyCertificateOutlined />, title: 'Security scan passed', description: 'No critical issues', time: '1 hour ago', status: 'success' as const },
    { icon: <RocketOutlined />, title: 'Model promoted to v2', description: 'GPT-4 Code Reviewer', time: '3 hours ago', status: 'info' as const },
  ];

  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%)',
        padding: 24,
      }}
    >
      {/* Welcome Header */}
      <div
        style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: 24,
          padding: '40px 32px',
          marginBottom: 24,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Decorative circles */}
        <div
          style={{
            position: 'absolute',
            top: -50,
            right: -50,
            width: 200,
            height: 200,
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '50%',
          }}
        />
        <div
          style={{
            position: 'absolute',
            bottom: -30,
            right: 100,
            width: 120,
            height: 120,
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '50%',
          }}
        />
        
        <Row align="middle" justify="space-between">
          <Col>
            <Space direction="vertical" size={4}>
              <Text style={{ color: 'rgba(255,255,255,0.8)', fontSize: 14 }}>
                {greeting}, {user?.name || 'Developer'}! 👋
              </Text>
              <Title level={2} style={{ color: 'white', margin: 0 }}>
                Welcome to AI Code Review Platform
              </Title>
              <Text style={{ color: 'rgba(255,255,255,0.7)', fontSize: 15 }}>
                Your intelligent companion for code quality and security
              </Text>
            </Space>
          </Col>
          <Col>
            <Space>
              <Tooltip title="Notifications">
                <Badge count={3} size="small">
                  <Button
                    shape="circle"
                    icon={<BellOutlined />}
                    style={{
                      background: 'rgba(255,255,255,0.2)',
                      border: 'none',
                      color: 'white',
                    }}
                  />
                </Badge>
              </Tooltip>
              <Avatar
                size={48}
                style={{
                  background: 'linear-gradient(135deg, #f59e0b 0%, #ef4444 100%)',
                  border: '3px solid rgba(255,255,255,0.3)',
                }}
              >
                {user?.name?.[0] || 'U'}
              </Avatar>
            </Space>
          </Col>
        </Row>
      </div>

      {/* Quick Actions */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        {quickActions.map((action, index) => (
          <Col xs={12} sm={6} key={index}>
            <Card
              hoverable
              onClick={() => navigate(action.path)}
              style={{
                borderRadius: 16,
                border: 'none',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.05)',
                textAlign: 'center',
                transition: 'all 0.3s ease',
              }}
              bodyStyle={{ padding: '24px 16px' }}
            >
              <div
                style={{
                  width: 56,
                  height: 56,
                  borderRadius: 16,
                  background: `${action.color}15`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 12px',
                  fontSize: 24,
                  color: action.color,
                }}
              >
                {action.icon}
              </div>
              <Text strong>{action.label}</Text>
            </Card>
          </Col>
        ))}
      </Row>

      {/* Stats Row */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <GradientStat
            title="Code Reviews"
            value={1247}
            trend={12}
            gradient="ocean"
            icon={<CodeOutlined />}
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <GradientStat
            title="Issues Fixed"
            value={892}
            trend={8}
            gradient="mint"
            icon={<CheckCircleOutlined />}
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <GradientStat
            title="Security Score"
            value={94}
            suffix="%"
            trend={5}
            gradient="aurora"
            icon={<SafetyCertificateOutlined />}
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <GradientStat
            title="AI Accuracy"
            value={96.8}
            suffix="%"
            trend={2}
            gradient="sunset"
            icon={<ThunderboltOutlined />}
          />
        </Col>
      </Row>

      {/* Main Content Row */}
      <Row gutter={24}>
        {/* Quality Overview */}
        <Col xs={24} lg={16}>
          <Card
            style={{
              borderRadius: 20,
              border: 'none',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.05)',
              marginBottom: 24,
            }}
            bodyStyle={{ padding: 28 }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
              <Title level={4} style={{ margin: 0 }}>Code Quality Overview</Title>
              <Button type="link" onClick={() => navigate('/admin/quality')}>
                View Details <ArrowRightOutlined />
              </Button>
            </div>
            
            <Row gutter={32} align="middle">
              <Col xs={24} md={8} style={{ textAlign: 'center', marginBottom: 24 }}>
                <ProgressRing
                  percent={85}
                  size={160}
                  gradient="aurora"
                  label="Overall Score"
                  sublabel="Grade: A-"
                />
              </Col>
              <Col xs={24} md={16}>
                <MetricBar label="Code Coverage" value={78} gradient="ocean" />
                <MetricBar label="Maintainability" value={85} gradient="mint" />
                <MetricBar label="Security" value={92} gradient="aurora" />
                <MetricBar label="Performance" value={88} gradient="sunset" />
                <MetricBar label="Documentation" value={65} gradient="ocean" />
              </Col>
            </Row>
          </Card>

          {/* AI Evolution Status */}
          <Card
            style={{
              borderRadius: 20,
              border: 'none',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.05)',
            }}
            bodyStyle={{ padding: 28 }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
              <Title level={4} style={{ margin: 0 }}>
                <RocketOutlined style={{ marginRight: 8, color: '#8b5cf6' }} />
                AI Self-Evolution Status
              </Title>
              <StatusBadge status="processing" text="Learning" animated />
            </div>
            
            <Row gutter={24}>
              {[
                { label: 'V1 Experimentation', value: 3, total: 5, status: 'Active experiments', color: '#3b82f6' },
                { label: 'V2 Production', value: 2, total: 2, status: 'All systems stable', color: '#10b981' },
                { label: 'V3 Quarantine', value: 1, total: 8, status: 'Pending review', color: '#f59e0b' },
              ].map((zone, index) => (
                <Col xs={24} md={8} key={index}>
                  <div
                    style={{
                      padding: 20,
                      background: `${zone.color}08`,
                      borderRadius: 12,
                      border: `1px solid ${zone.color}20`,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                      <Text strong>{zone.label}</Text>
                      <Badge
                        count={zone.value}
                        style={{ background: zone.color }}
                      />
                    </div>
                    <div
                      style={{
                        fontSize: 24,
                        fontWeight: 700,
                        color: zone.color,
                        marginBottom: 4,
                      }}
                    >
                      {zone.value}/{zone.total}
                    </div>
                    <Text type="secondary" style={{ fontSize: 12 }}>{zone.status}</Text>
                  </div>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* Right Sidebar */}
        <Col xs={24} lg={8}>
          {/* Recent Activity */}
          <Card
            style={{
              borderRadius: 20,
              border: 'none',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.05)',
              marginBottom: 24,
            }}
            bodyStyle={{ padding: 28 }}
          >
            <Title level={4} style={{ margin: '0 0 20px 0' }}>
              <ClockCircleOutlined style={{ marginRight: 8, color: '#6366f1' }} />
              Recent Activity
            </Title>
            <ArtisticTimeline items={activities} />
          </Card>

          {/* Top Projects */}
          <Card
            style={{
              borderRadius: 20,
              border: 'none',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.05)',
            }}
            bodyStyle={{ padding: 28 }}
          >
            <Title level={4} style={{ margin: '0 0 20px 0' }}>
              <StarFilled style={{ marginRight: 8, color: '#f59e0b' }} />
              Top Projects
            </Title>
            <Space direction="vertical" style={{ width: '100%' }} size={12}>
              {[
                { name: 'backend-api', score: 94, language: 'Python', hot: true },
                { name: 'frontend-app', score: 88, language: 'TypeScript', hot: false },
                { name: 'ai-core', score: 91, language: 'Python', hot: true },
                { name: 'mobile-app', score: 82, language: 'Kotlin', hot: false },
              ].map((project, index) => (
                <div
                  key={index}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '12px 16px',
                    background: '#f9fafb',
                    borderRadius: 12,
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                  }}
                >
                  <Space>
                    <div
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: project.score >= 90 ? '#10b981' : project.score >= 80 ? '#f59e0b' : '#ef4444',
                      }}
                    />
                    <div>
                      <Space size={4}>
                        <Text strong>{project.name}</Text>
                        {project.hot && <FireFilled style={{ color: '#ef4444', fontSize: 12 }} />}
                      </Space>
                      <div>
                        <Text type="secondary" style={{ fontSize: 12 }}>{project.language}</Text>
                      </div>
                    </div>
                  </Space>
                  <div
                    style={{
                      padding: '4px 10px',
                      background: project.score >= 90 ? '#d1fae5' : project.score >= 80 ? '#fef3c7' : '#fee2e2',
                      borderRadius: 8,
                      fontWeight: 600,
                      fontSize: 13,
                      color: project.score >= 90 ? '#065f46' : project.score >= 80 ? '#92400e' : '#991b1b',
                    }}
                  >
                    {project.score}
                  </div>
                </div>
              ))}
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default WelcomeDashboard;
