/**
 * System Health Dashboard
 * 
 * Monitor the health of all system components:
 * - Services (Backend, AI, Database)
 * - Evolution Cycle (V1/V2/V3)
 * - Auto-Fix Cycle
 * - Resource Usage
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Tag,
  Space,
  Progress,
  Statistic,
  Typography,
  Timeline,
  Alert,
  Badge,
  Tooltip,
  Spin,
  message,
} from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  SyncOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  ApiOutlined,
  RobotOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
  DesktopOutlined,
  HddOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text } = Typography;

// Types
interface ServiceHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency_ms: number;
  uptime_percent: number;
  last_check: string;
  error?: string;
}

interface ResourceUsage {
  cpu_percent: number;
  memory_percent: number;
  disk_percent: number;
  network_in_mbps: number;
  network_out_mbps: number;
}

interface CycleHealth {
  name: string;
  running: boolean;
  cycle_count: number;
  last_cycle: string;
  success_rate: number;
}

interface HealthEvent {
  timestamp: string;
  type: 'info' | 'warning' | 'error' | 'success';
  message: string;
  service: string;
}

// Status colors and icons
const statusConfig = {
  healthy: { color: 'green', icon: <CheckCircleOutlined /> },
  degraded: { color: 'orange', icon: <WarningOutlined /> },
  unhealthy: { color: 'red', icon: <CloseCircleOutlined /> },
};

const SystemHealth: React.FC = () => {
  const { t: _t } = useTranslation();
  
  // State
  const [services, setServices] = useState<ServiceHealth[]>([]);
  const [resources, setResources] = useState<ResourceUsage | null>(null);
  const [cycles, setCycles] = useState<CycleHealth[]>([]);
  const [events, setEvents] = useState<HealthEvent[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch data
  const fetchData = useCallback(async () => {
    try {
      // Mock data
      const mockServices: ServiceHealth[] = [
        {
          name: 'API Gateway',
          status: 'healthy',
          latency_ms: 45,
          uptime_percent: 99.99,
          last_check: new Date().toISOString(),
        },
        {
          name: 'Auth Service',
          status: 'healthy',
          latency_ms: 32,
          uptime_percent: 99.98,
          last_check: new Date().toISOString(),
        },
        {
          name: 'Analysis Service',
          status: 'healthy',
          latency_ms: 180,
          uptime_percent: 99.95,
          last_check: new Date().toISOString(),
        },
        {
          name: 'AI Orchestrator',
          status: 'healthy',
          latency_ms: 250,
          uptime_percent: 99.90,
          last_check: new Date().toISOString(),
        },
        {
          name: 'PostgreSQL',
          status: 'healthy',
          latency_ms: 5,
          uptime_percent: 99.99,
          last_check: new Date().toISOString(),
        },
        {
          name: 'Redis',
          status: 'healthy',
          latency_ms: 1,
          uptime_percent: 99.99,
          last_check: new Date().toISOString(),
        },
        {
          name: 'Neo4j',
          status: 'degraded',
          latency_ms: 85,
          uptime_percent: 99.5,
          last_check: new Date().toISOString(),
          error: 'High memory usage detected',
        },
      ];

      const mockResources: ResourceUsage = {
        cpu_percent: 42,
        memory_percent: 68,
        disk_percent: 35,
        network_in_mbps: 125,
        network_out_mbps: 85,
      };

      const mockCycles: CycleHealth[] = [
        {
          name: 'Evolution Cycle',
          running: true,
          cycle_count: 24,
          last_cycle: new Date().toISOString(),
          success_rate: 0.95,
        },
        {
          name: 'Auto-Fix Cycle',
          running: true,
          cycle_count: 15,
          last_cycle: new Date().toISOString(),
          success_rate: 0.88,
        },
        {
          name: 'Learning Cycle',
          running: true,
          cycle_count: 120,
          last_cycle: new Date().toISOString(),
          success_rate: 0.92,
        },
      ];

      const mockEvents: HealthEvent[] = [
        {
          timestamp: new Date().toISOString(),
          type: 'success',
          message: 'Evolution cycle #24 completed successfully',
          service: 'Evolution',
        },
        {
          timestamp: new Date(Date.now() - 300000).toISOString(),
          type: 'warning',
          message: 'Neo4j memory usage above 80%',
          service: 'Neo4j',
        },
        {
          timestamp: new Date(Date.now() - 600000).toISOString(),
          type: 'info',
          message: 'Auto-fix applied 2 patches',
          service: 'Auto-Fix',
        },
        {
          timestamp: new Date(Date.now() - 900000).toISOString(),
          type: 'success',
          message: 'Model GPT-4 promoted to V2',
          service: 'AI Models',
        },
        {
          timestamp: new Date(Date.now() - 1200000).toISOString(),
          type: 'info',
          message: 'System backup completed',
          service: 'System',
        },
      ];

      setServices(mockServices);
      setResources(mockResources);
      setCycles(mockCycles);
      setEvents(mockEvents);
    } catch (error) {
      message.error('Failed to load health data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Calculate overall health
  const overallHealth = services.every((s) => s.status === 'healthy')
    ? 'healthy'
    : services.some((s) => s.status === 'unhealthy')
    ? 'unhealthy'
    : 'degraded';

  const healthyCount = services.filter((s) => s.status === 'healthy').length;

  // Get service icon
  const getServiceIcon = (name: string) => {
    if (name.includes('Gateway') || name.includes('API')) return <ApiOutlined />;
    if (name.includes('Database') || name.includes('SQL')) return <DatabaseOutlined />;
    if (name.includes('Redis')) return <HddOutlined />;
    if (name.includes('AI') || name.includes('Analysis')) return <RobotOutlined />;
    return <CloudServerOutlined />;
  };

  // Get event color
  const getEventColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'green';
      case 'warning':
        return 'orange';
      case 'error':
        return 'red';
      default:
        return 'blue';
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div style={{ padding: 24 }}>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}>
            <DesktopOutlined style={{ marginRight: 8 }} />
            System Health
          </Title>
        </Col>
        <Col>
          <Space>
            <Tag
              color={statusConfig[overallHealth].color}
              icon={statusConfig[overallHealth].icon}
              style={{ fontSize: 14, padding: '4px 12px' }}
            >
              {overallHealth.toUpperCase()}
            </Tag>
            <Text type="secondary">
              <SyncOutlined spin /> Auto-refreshing
            </Text>
          </Space>
        </Col>
      </Row>

      {/* Alert if degraded */}
      {overallHealth !== 'healthy' && (
        <Alert
          message="System Status Warning"
          description={
            services
              .filter((s) => s.status !== 'healthy')
              .map((s) => `${s.name}: ${s.error || s.status}`)
              .join('; ')
          }
          type={overallHealth === 'unhealthy' ? 'error' : 'warning'}
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {/* Summary Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Services Healthy"
              value={healthyCount}
              suffix={`/ ${services.length}`}
              valueStyle={{
                color: healthyCount === services.length ? '#52c41a' : '#faad14',
              }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="CPU Usage"
              value={resources?.cpu_percent || 0}
              suffix="%"
              valueStyle={{
                color: (resources?.cpu_percent || 0) > 80 ? '#ff4d4f' : '#52c41a',
              }}
            />
            <Progress
              percent={resources?.cpu_percent || 0}
              showInfo={false}
              strokeColor={(resources?.cpu_percent || 0) > 80 ? '#ff4d4f' : '#52c41a'}
              size="small"
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Memory Usage"
              value={resources?.memory_percent || 0}
              suffix="%"
              valueStyle={{
                color: (resources?.memory_percent || 0) > 80 ? '#ff4d4f' : '#52c41a',
              }}
            />
            <Progress
              percent={resources?.memory_percent || 0}
              showInfo={false}
              strokeColor={(resources?.memory_percent || 0) > 80 ? '#ff4d4f' : '#52c41a'}
              size="small"
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Disk Usage"
              value={resources?.disk_percent || 0}
              suffix="%"
              valueStyle={{
                color: (resources?.disk_percent || 0) > 80 ? '#ff4d4f' : '#52c41a',
              }}
            />
            <Progress
              percent={resources?.disk_percent || 0}
              showInfo={false}
              strokeColor={(resources?.disk_percent || 0) > 80 ? '#ff4d4f' : '#52c41a'}
              size="small"
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={24}>
        {/* Services */}
        <Col xs={24} lg={12}>
          <Card title="Service Status" style={{ marginBottom: 24 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {services.map((service) => (
                <div
                  key={service.name}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '8px 0',
                    borderBottom: '1px solid #f0f0f0',
                  }}
                >
                  <Space>
                    {getServiceIcon(service.name)}
                    <Text strong>{service.name}</Text>
                  </Space>
                  <Space>
                    <Tooltip title={`Latency: ${service.latency_ms}ms`}>
                      <Text type="secondary">{service.latency_ms}ms</Text>
                    </Tooltip>
                    <Tooltip title={`Uptime: ${service.uptime_percent}%`}>
                      <Text type="secondary">{service.uptime_percent}%</Text>
                    </Tooltip>
                    <Badge
                      status={
                        service.status === 'healthy'
                          ? 'success'
                          : service.status === 'degraded'
                          ? 'warning'
                          : 'error'
                      }
                      text={service.status}
                    />
                  </Space>
                </div>
              ))}
            </Space>
          </Card>

          {/* AI Cycles */}
          <Card title="AI Cycles">
            <Space direction="vertical" style={{ width: '100%' }}>
              {cycles.map((cycle) => (
                <div
                  key={cycle.name}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '8px 0',
                    borderBottom: '1px solid #f0f0f0',
                  }}
                >
                  <Space>
                    <ThunderboltOutlined />
                    <Text strong>{cycle.name}</Text>
                  </Space>
                  <Space>
                    <Tag color={cycle.running ? 'green' : 'default'}>
                      {cycle.running ? (
                        <>
                          <SyncOutlined spin /> Running
                        </>
                      ) : (
                        'Stopped'
                      )}
                    </Tag>
                    <Text type="secondary">#{cycle.cycle_count}</Text>
                    <Progress
                      type="circle"
                      percent={Math.round(cycle.success_rate * 100)}
                      width={40}
                      strokeColor={cycle.success_rate >= 0.9 ? '#52c41a' : '#faad14'}
                    />
                  </Space>
                </div>
              ))}
            </Space>
          </Card>
        </Col>

        {/* Events Timeline */}
        <Col xs={24} lg={12}>
          <Card title="Recent Events">
            <Timeline>
              {events.map((event, index) => (
                <Timeline.Item
                  key={index}
                  color={getEventColor(event.type)}
                  dot={
                    event.type === 'success' ? (
                      <CheckCircleOutlined />
                    ) : event.type === 'warning' ? (
                      <WarningOutlined />
                    ) : event.type === 'error' ? (
                      <CloseCircleOutlined />
                    ) : (
                      <ClockCircleOutlined />
                    )
                  }
                >
                  <div>
                    <Text strong>{event.message}</Text>
                    <br />
                    <Space>
                      <Tag>{event.service}</Tag>
                      <Text type="secondary">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </Text>
                    </Space>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default SystemHealth;
