/**
 * Quick Actions Widget
 * 
 * Dashboard widget for quick access to common actions:
 * - Start new analysis
 * - Create project
 * - View reports
 * - AI Assistant
 */

import React from 'react';
import { Card, Row, Col, Button, Space, Typography } from 'antd';
import {
  PlusOutlined,
  ScanOutlined,
  FileSearchOutlined,
  RobotOutlined,
  SettingOutlined,
  TeamOutlined,
  SafetyOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const { Text } = Typography;

interface QuickAction {
  key: string;
  icon: React.ReactNode;
  label: string;
  description: string;
  path: string;
  color: string;
}

const QuickActionsWidget: React.FC = () => {
  const navigate = useNavigate();
  const { t } = useTranslation();

  const actions: QuickAction[] = [
    {
      key: 'new-project',
      icon: <PlusOutlined />,
      label: t('quick_actions.new_project', 'New Project'),
      description: t('quick_actions.new_project_desc', 'Create a new project'),
      path: '/projects/new',
      color: '#1890ff',
    },
    {
      key: 'analyze',
      icon: <ScanOutlined />,
      label: t('quick_actions.analyze', 'Start Analysis'),
      description: t('quick_actions.analyze_desc', 'Run code analysis'),
      path: '/code-review',
      color: '#52c41a',
    },
    {
      key: 'reports',
      icon: <FileSearchOutlined />,
      label: t('quick_actions.reports', 'View Reports'),
      description: t('quick_actions.reports_desc', 'See analysis reports'),
      path: '/reports',
      color: '#722ed1',
    },
    {
      key: 'ai-assistant',
      icon: <RobotOutlined />,
      label: t('quick_actions.ai_assistant', 'AI Assistant'),
      description: t('quick_actions.ai_assistant_desc', 'Chat with AI'),
      path: '/ai-assistant',
      color: '#eb2f96',
    },
    {
      key: 'security',
      icon: <SafetyOutlined />,
      label: t('quick_actions.security', 'Security Scan'),
      description: t('quick_actions.security_desc', 'Check vulnerabilities'),
      path: '/admin/vulnerabilities',
      color: '#fa541c',
    },
    {
      key: 'auto-fix',
      icon: <ThunderboltOutlined />,
      label: t('quick_actions.auto_fix', 'Auto-Fix'),
      description: t('quick_actions.auto_fix_desc', 'Apply AI fixes'),
      path: '/admin/auto-fix',
      color: '#faad14',
    },
  ];

  return (
    <Card
      title={
        <Space>
          <ThunderboltOutlined />
          {t('quick_actions.title', 'Quick Actions')}
        </Space>
      }
      size="small"
    >
      <Row gutter={[8, 8]}>
        {actions.map((action) => (
          <Col xs={12} sm={8} key={action.key}>
            <Button
              type="default"
              block
              style={{
                height: 'auto',
                padding: '12px 8px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 4,
              }}
              onClick={() => navigate(action.path)}
            >
              <span style={{ fontSize: 20, color: action.color }}>
                {action.icon}
              </span>
              <Text style={{ fontSize: 12 }}>{action.label}</Text>
            </Button>
          </Col>
        ))}
      </Row>
    </Card>
  );
};

export default QuickActionsWidget;
