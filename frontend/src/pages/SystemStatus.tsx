/**
 * System Status Page
 * 系统状态页面
 * 
 * Features:
 * - Service health monitoring
 * - Uptime statistics
 * - Incident history
 * - Performance metrics
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Tag,
  Progress,
  Timeline,
  Statistic,
  Badge,
  Tooltip,
  Alert,
  Divider,
  List,
} from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  SyncOutlined,
  CloudServerOutlined,
  ApiOutlined,
  DatabaseOutlined,
  RobotOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
  GlobalOutlined,
  SafetyCertificateOutlined,
  LineChartOutlined,
  HistoryOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text, Paragraph } = Typography;

interface Service {
  id: string;
  name: string;
  description: string;
  status: 'operational' | 'degraded' | 'outage' | 'maintenance';
  uptime: number;
  responseTime: number;
  icon: React.ReactNode;
}

interface Incident {
  id: string;
  title: string;
  status: 'investigating' | 'identified' | 'monitoring' | 'resolved';
  severity: 'critical' | 'major' | 'minor';
  startedAt: string;
  resolvedAt?: string;
  updates: { time: string; message: string }[];
}

const services: Service[] = [
  {
    id: 'api',
    name: 'API Gateway',
    description: 'Core API services',
    status: 'operational',
    uptime: 99.99,
    responseTime: 45,
    icon: <ApiOutlined />,
  },
  {
    id: 'web',
    name: 'Web Application',
    description: 'Frontend dashboard',
    status: 'operational',
    uptime: 99.98,
    responseTime: 120,
    icon: <GlobalOutlined />,
  },
  {
    id: 'ai',
    name: 'AI Analysis Engine',
    description: 'Code review AI models',
    status: 'operational',
    uptime: 99.95,
    responseTime: 850,
    icon: <RobotOutlined />,
  },
  {
    id: 'db',
    name: 'Database',
    description: 'PostgreSQL cluster',
    status: 'operational',
    uptime: 99.999,
    responseTime: 12,
    icon: <DatabaseOutlined />,
  },
  {
    id: 'queue',
    name: 'Job Queue',
    description: 'Background processing',
    status: 'operational',
    uptime: 99.97,
    responseTime: 25,
    icon: <ThunderboltOutlined />,
  },
  {
    id: 'auth',
    name: 'Authentication',
    description: 'OAuth & SSO services',
    status: 'operational',
    uptime: 99.99,
    responseTime: 35,
    icon: <SafetyCertificateOutlined />,
  },
];

const incidents: Incident[] = [
  {
    id: 'inc_1',
    title: 'Elevated API Response Times',
    status: 'resolved',
    severity: 'minor',
    startedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    resolvedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000 + 45 * 60 * 1000).toISOString(),
    updates: [
      { time: '10:00', message: 'Investigating elevated response times' },
      { time: '10:25', message: 'Identified database connection pool issue' },
      { time: '10:45', message: 'Issue resolved, monitoring for stability' },
    ],
  },
  {
    id: 'inc_2',
    title: 'Scheduled Maintenance - AI Model Update',
    status: 'resolved',
    severity: 'minor',
    startedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    resolvedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000 + 2 * 60 * 60 * 1000).toISOString(),
    updates: [
      { time: '02:00', message: 'Starting scheduled maintenance' },
      { time: '04:00', message: 'Maintenance completed successfully' },
    ],
  },
];

const uptimeData = [
  { day: 'Mon', uptime: 100 },
  { day: 'Tue', uptime: 100 },
  { day: 'Wed', uptime: 99.95 },
  { day: 'Thu', uptime: 100 },
  { day: 'Fri', uptime: 100 },
  { day: 'Sat', uptime: 100 },
  { day: 'Sun', uptime: 100 },
];

const statusConfig = {
  operational: { color: '#22c55e', icon: <CheckCircleOutlined />, text: 'Operational' },
  degraded: { color: '#f59e0b', icon: <WarningOutlined />, text: 'Degraded' },
  outage: { color: '#ef4444', icon: <CloseCircleOutlined />, text: 'Outage' },
  maintenance: { color: '#3b82f6', icon: <SyncOutlined spin />, text: 'Maintenance' },
};

export const SystemStatus: React.FC = () => {
  const { t } = useTranslation();
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const allOperational = services.every(s => s.status === 'operational');
  const overallUptime = services.reduce((acc, s) => acc + s.uptime, 0) / services.length;

  return (
    <div className="system-status-page" style={{ maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <CloudServerOutlined style={{ color: '#2563eb' }} /> System Status
          </Title>
          <Text type="secondary">
            Real-time service health and uptime monitoring
          </Text>
        </div>
        <Text type="secondary">
          Last updated: {currentTime.toLocaleTimeString()}
        </Text>
      </div>

      {/* Overall Status Banner */}
      <Alert
        type={allOperational ? 'success' : 'warning'}
        icon={allOperational ? <CheckCircleOutlined /> : <WarningOutlined />}
        message={
          <Space>
            <Text strong style={{ fontSize: 16 }}>
              {allOperational ? 'All Systems Operational' : 'Some Systems Degraded'}
            </Text>
          </Space>
        }
        description={
          allOperational
            ? 'All services are running normally. No incidents reported.'
            : 'We are currently experiencing issues with some services.'
        }
        style={{ marginBottom: 24, borderRadius: 12 }}
      />

      {/* Stats Overview */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Overall Uptime"
              value={overallUptime.toFixed(2)}
              suffix="%"
              valueStyle={{ color: '#22c55e' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Active Services"
              value={services.filter(s => s.status === 'operational').length}
              suffix={`/ ${services.length}`}
              valueStyle={{ color: '#2563eb' }}
              prefix={<CloudServerOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Avg Response"
              value={Math.round(services.reduce((a, s) => a + s.responseTime, 0) / services.length)}
              suffix="ms"
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Incidents (30d)"
              value={incidents.length}
              valueStyle={{ color: incidents.length > 0 ? '#f59e0b' : '#22c55e' }}
              prefix={<HistoryOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={24}>
        {/* Services Status */}
        <Col xs={24} lg={14}>
          <Card title={<><CloudServerOutlined /> Service Status</>} style={{ borderRadius: 12, marginBottom: 24 }}>
            <List
              dataSource={services}
              renderItem={service => {
                const config = statusConfig[service.status];
                return (
                  <List.Item
                    style={{
                      padding: '16px 0',
                      borderBottom: '1px solid #f1f5f9',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                      <div style={{
                        width: 40,
                        height: 40,
                        borderRadius: 10,
                        background: '#f1f5f9',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        marginRight: 16,
                        fontSize: 18,
                        color: '#64748b',
                      }}>
                        {service.icon}
                      </div>
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div>
                            <Text strong>{service.name}</Text>
                            <div>
                              <Text type="secondary" style={{ fontSize: 12 }}>
                                {service.description}
                              </Text>
                            </div>
                          </div>
                          <Space>
                            <Tooltip title="Response Time">
                              <Tag>{service.responseTime}ms</Tag>
                            </Tooltip>
                            <Tooltip title={`${service.uptime}% uptime`}>
                              <Tag color={config.color} icon={config.icon}>
                                {config.text}
                              </Tag>
                            </Tooltip>
                          </Space>
                        </div>
                      </div>
                    </div>
                  </List.Item>
                );
              }}
            />
          </Card>

          {/* Uptime Chart (Simplified) */}
          <Card title={<><LineChartOutlined /> 7-Day Uptime</>} style={{ borderRadius: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', height: 80 }}>
              {uptimeData.map((day, index) => (
                <Tooltip key={index} title={`${day.day}: ${day.uptime}%`}>
                  <div style={{ textAlign: 'center' }}>
                    <div
                      style={{
                        width: 40,
                        height: day.uptime === 100 ? 60 : 50,
                        background: day.uptime === 100 ? '#22c55e' : '#f59e0b',
                        borderRadius: 6,
                        marginBottom: 8,
                      }}
                    />
                    <Text type="secondary" style={{ fontSize: 12 }}>{day.day}</Text>
                  </div>
                </Tooltip>
              ))}
            </div>
          </Card>
        </Col>

        {/* Incident History */}
        <Col xs={24} lg={10}>
          <Card title={<><HistoryOutlined /> Recent Incidents</>} style={{ borderRadius: 12 }}>
            {incidents.length > 0 ? (
              <Timeline
                items={incidents.map(incident => ({
                  color: incident.status === 'resolved' ? 'green' : 'orange',
                  children: (
                    <div>
                      <Space>
                        <Text strong>{incident.title}</Text>
                        <Tag color={
                          incident.severity === 'critical' ? 'red' :
                          incident.severity === 'major' ? 'orange' : 'blue'
                        }>
                          {incident.severity}
                        </Tag>
                      </Space>
                      <div>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {new Date(incident.startedAt).toLocaleDateString()}
                          {incident.resolvedAt && ` - Resolved`}
                        </Text>
                      </div>
                      <div style={{ marginTop: 8 }}>
                        {incident.updates.slice(-1).map((update, idx) => (
                          <Text key={idx} style={{ fontSize: 13 }}>
                            {update.message}
                          </Text>
                        ))}
                      </div>
                    </div>
                  ),
                }))}
              />
            ) : (
              <div style={{ textAlign: 'center', padding: 24 }}>
                <CheckCircleOutlined style={{ fontSize: 48, color: '#22c55e', marginBottom: 16 }} />
                <div>
                  <Text type="secondary">No incidents in the last 30 days</Text>
                </div>
              </div>
            )}
          </Card>

          {/* Quick Info */}
          <Card title="Service Information" style={{ borderRadius: 12, marginTop: 24 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text type="secondary">API Version</Text>
                <Text>v2.1.5</Text>
              </div>
              <Divider style={{ margin: '8px 0' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text type="secondary">Region</Text>
                <Text>US-East-1</Text>
              </div>
              <Divider style={{ margin: '8px 0' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text type="secondary">Last Deploy</Text>
                <Text>2 hours ago</Text>
              </div>
              <Divider style={{ margin: '8px 0' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text type="secondary">SSL Certificate</Text>
                <Tag color="green">Valid</Tag>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default SystemStatus;
