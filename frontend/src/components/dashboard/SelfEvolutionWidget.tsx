/**
 * Self-Evolution Widget
 * 
 * Dashboard widget showing the status of the AI self-evolution cycles:
 * - Bug Fix Cycle status
 * - Evolution Cycle (V1/V2/V3)
 * - Recent activity
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Tag,
  Progress,
  Statistic,
  Space,
  Typography,
  Timeline,
  Badge,
  Tooltip,
  Spin,
} from 'antd';
import {
  ThunderboltOutlined,
  RocketOutlined,
  SafetyOutlined,
  SyncOutlined,
  WarningOutlined,
  ExperimentOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

const { Text } = Typography;

interface CycleStatus {
  running: boolean;
  cycles: number;
  lastCycle: string;
  metrics: {
    success: number;
    failed: number;
  };
}

interface EvolutionActivity {
  id: string;
  type: 'promotion' | 'degradation' | 'fix' | 'experiment';
  message: string;
  timestamp: string;
}

const SelfEvolutionWidget: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [bugFixCycle] = useState<CycleStatus>({
    running: true,
    cycles: 15,
    lastCycle: new Date().toISOString(),
    metrics: { success: 12, failed: 3 },
  });
  const [evolutionCycle] = useState<CycleStatus>({
    running: true,
    cycles: 24,
    lastCycle: new Date().toISOString(),
    metrics: { success: 5, failed: 2 },
  });
  const [activities, setActivities] = useState<EvolutionActivity[]>([]);

  useEffect(() => {
    // Simulate fetching data
    const timer = setTimeout(() => {
      setActivities([
        {
          id: '1',
          type: 'fix',
          message: 'Auto-fixed hardcoded secret in auth.py',
          timestamp: new Date().toISOString(),
        },
        {
          id: '2',
          type: 'promotion',
          message: 'GQA Attention promoted to V2',
          timestamp: new Date(Date.now() - 3600000).toISOString(),
        },
        {
          id: '3',
          type: 'experiment',
          message: 'Started DPO alignment experiment',
          timestamp: new Date(Date.now() - 7200000).toISOString(),
        },
        {
          id: '4',
          type: 'degradation',
          message: 'SWA model moved to V3 quarantine',
          timestamp: new Date(Date.now() - 10800000).toISOString(),
        },
      ]);
      setLoading(false);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'promotion':
        return <ArrowUpOutlined style={{ color: '#52c41a' }} />;
      case 'degradation':
        return <ArrowDownOutlined style={{ color: '#ff4d4f' }} />;
      case 'fix':
        return <SafetyOutlined style={{ color: '#1890ff' }} />;
      case 'experiment':
        return <ExperimentOutlined style={{ color: '#faad14' }} />;
      default:
        return <SyncOutlined />;
    }
  };

  if (loading) {
    return (
      <Card title="Self-Evolution Status">
        <div style={{ textAlign: 'center', padding: 40 }}>
          <Spin />
        </div>
      </Card>
    );
  }

  return (
    <Card
      title={
        <Space>
          <ThunderboltOutlined />
          Self-Evolution Status
        </Space>
      }
      extra={
        <Space>
          <Badge
            status={bugFixCycle.running && evolutionCycle.running ? 'processing' : 'default'}
            text={bugFixCycle.running && evolutionCycle.running ? 'Active' : 'Paused'}
          />
        </Space>
      }
    >
      <Row gutter={[16, 16]}>
        {/* Bug Fix Cycle */}
        <Col xs={24} md={12}>
          <Card
            size="small"
            title={
              <Space>
                <SafetyOutlined />
                Bug Fix Cycle
              </Space>
            }
            extra={
              <Tag color={bugFixCycle.running ? 'green' : 'default'}>
                {bugFixCycle.running ? <SyncOutlined spin /> : null}
                {bugFixCycle.running ? ' Running' : ' Stopped'}
              </Tag>
            }
            hoverable
            onClick={() => navigate('/admin/vulnerabilities')}
            style={{ cursor: 'pointer' }}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="Cycles"
                  value={bugFixCycle.cycles}
                  valueStyle={{ fontSize: 20 }}
                />
              </Col>
              <Col span={12}>
                <Text type="secondary">Success Rate</Text>
                <Progress
                  percent={Math.round(
                    (bugFixCycle.metrics.success /
                      (bugFixCycle.metrics.success + bugFixCycle.metrics.failed)) *
                      100
                  )}
                  size="small"
                  status="active"
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Evolution Cycle */}
        <Col xs={24} md={12}>
          <Card
            size="small"
            title={
              <Space>
                <RocketOutlined />
                Evolution Cycle
              </Space>
            }
            extra={
              <Tag color={evolutionCycle.running ? 'green' : 'default'}>
                {evolutionCycle.running ? <SyncOutlined spin /> : null}
                {evolutionCycle.running ? ' Running' : ' Stopped'}
              </Tag>
            }
            hoverable
            onClick={() => navigate('/admin/evolution')}
            style={{ cursor: 'pointer' }}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="Cycles"
                  value={evolutionCycle.cycles}
                  valueStyle={{ fontSize: 20 }}
                />
              </Col>
              <Col span={12}>
                <Text type="secondary">Promotions</Text>
                <div>
                  <Tag color="green">{evolutionCycle.metrics.success} promoted</Tag>
                  <Tag color="red">{evolutionCycle.metrics.failed} degraded</Tag>
                </div>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Version Status */}
        <Col span={24}>
          <Card size="small" title="Version Status">
            <Row gutter={16} justify="space-around">
              <Col>
                <Tooltip title="Experimental - Testing new technologies">
                  <div style={{ textAlign: 'center' }}>
                    <Badge count={3} style={{ backgroundColor: '#faad14' }}>
                      <Tag color="orange" style={{ fontSize: 14, padding: '4px 12px' }}>
                        <ExperimentOutlined /> V1
                      </Tag>
                    </Badge>
                    <div style={{ marginTop: 4 }}>
                      <Text type="secondary">Experimental</Text>
                    </div>
                  </div>
                </Tooltip>
              </Col>
              <Col>
                <Tooltip title="Production - Stable for users">
                  <div style={{ textAlign: 'center' }}>
                    <Badge count={5} style={{ backgroundColor: '#52c41a' }}>
                      <Tag color="green" style={{ fontSize: 14, padding: '4px 12px' }}>
                        <RocketOutlined /> V2
                      </Tag>
                    </Badge>
                    <div style={{ marginTop: 4 }}>
                      <Text type="secondary">Production</Text>
                    </div>
                  </div>
                </Tooltip>
              </Col>
              <Col>
                <Tooltip title="Quarantine - Failed technologies">
                  <div style={{ textAlign: 'center' }}>
                    <Badge count={2} style={{ backgroundColor: '#ff4d4f' }}>
                      <Tag color="red" style={{ fontSize: 14, padding: '4px 12px' }}>
                        <WarningOutlined /> V3
                      </Tag>
                    </Badge>
                    <div style={{ marginTop: 4 }}>
                      <Text type="secondary">Quarantine</Text>
                    </div>
                  </div>
                </Tooltip>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Recent Activity */}
        <Col span={24}>
          <Card size="small" title="Recent Activity">
            <Timeline
              items={activities.map((activity) => ({
                dot: getActivityIcon(activity.type),
                children: (
                  <div>
                    <Text>{activity.message}</Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {new Date(activity.timestamp).toLocaleTimeString()}
                    </Text>
                  </div>
                ),
              }))}
            />
          </Card>
        </Col>
      </Row>
    </Card>
  );
};

export default SelfEvolutionWidget;
