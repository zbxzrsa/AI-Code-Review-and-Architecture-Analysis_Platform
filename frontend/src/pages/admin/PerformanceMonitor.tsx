/**
 * Performance Monitor Dashboard
 * 
 * Real-time performance monitoring for the AI platform:
 * - Service latency metrics
 * - Request throughput
 * - Error rates
 * - Resource utilization
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Statistic,
  Progress,
  Table,
  Tag,
  Space,
  Select,
  Button,
  Alert,
  Badge,
} from 'antd';
import {
  DashboardOutlined,
  ThunderboltOutlined,
  ApiOutlined,
  ClockCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';

const { Title, Text } = Typography;
const { Option } = Select;

interface ServiceMetric {
  name: string;
  latencyP50: number;
  latencyP95: number;
  latencyP99: number;
  requestsPerSecond: number;
  errorRate: number;
  status: 'healthy' | 'degraded' | 'unhealthy';
}

interface TimeSeriesData {
  time: string;
  latency: number;
  requests: number;
  errors: number;
}

const PerformanceMonitor: React.FC = () => {
  const { t } = useTranslation();
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('1h');
  const [refreshing, setRefreshing] = useState(false);
  const [services, setServices] = useState<ServiceMetric[]>([]);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);

  // Generate mock data
  useEffect(() => {
    const generateMockData = () => {
      // Services
      setServices([
        { name: 'API Gateway', latencyP50: 45, latencyP95: 120, latencyP99: 250, requestsPerSecond: 1250, errorRate: 0.02, status: 'healthy' },
        { name: 'Auth Service', latencyP50: 32, latencyP95: 85, latencyP99: 150, requestsPerSecond: 450, errorRate: 0.01, status: 'healthy' },
        { name: 'Analysis Service', latencyP50: 1800, latencyP95: 2500, latencyP99: 3500, requestsPerSecond: 85, errorRate: 0.03, status: 'healthy' },
        { name: 'AI Orchestrator', latencyP50: 2200, latencyP95: 3000, latencyP99: 4500, requestsPerSecond: 45, errorRate: 0.05, status: 'degraded' },
        { name: 'Database', latencyP50: 5, latencyP95: 15, latencyP99: 35, requestsPerSecond: 2500, errorRate: 0.001, status: 'healthy' },
        { name: 'Redis Cache', latencyP50: 1, latencyP95: 3, latencyP99: 8, requestsPerSecond: 5000, errorRate: 0.0001, status: 'healthy' },
      ]);

      // Time series
      const now = Date.now();
      const data: TimeSeriesData[] = [];
      for (let i = 59; i >= 0; i--) {
        const time = new Date(now - i * 60000);
        data.push({
          time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          latency: 150 + Math.random() * 100 + Math.sin(i / 10) * 50,
          requests: 1000 + Math.random() * 500 + Math.sin(i / 5) * 200,
          errors: Math.floor(Math.random() * 5),
        });
      }
      setTimeSeriesData(data);
    };

    generateMockData();
    const interval = setInterval(generateMockData, 30000);
    return () => clearInterval(interval);
  }, [timeRange]);

  const handleRefresh = () => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1000);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'error';
      default: return 'default';
    }
  };

  const serviceColumns = [
    {
      title: 'Service',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: ServiceMetric) => (
        <Space>
          <Badge status={getStatusColor(record.status) as any} />
          <Text strong>{name}</Text>
        </Space>
      ),
    },
    {
      title: 'P50 Latency',
      dataIndex: 'latencyP50',
      key: 'latencyP50',
      render: (val: number) => <Text>{val}ms</Text>,
      sorter: (a: ServiceMetric, b: ServiceMetric) => a.latencyP50 - b.latencyP50,
    },
    {
      title: 'P95 Latency',
      dataIndex: 'latencyP95',
      key: 'latencyP95',
      render: (val: number) => <Text>{val}ms</Text>,
      sorter: (a: ServiceMetric, b: ServiceMetric) => a.latencyP95 - b.latencyP95,
    },
    {
      title: 'P99 Latency',
      dataIndex: 'latencyP99',
      key: 'latencyP99',
      render: (val: number) => (
        <Text type={val > 3000 ? 'danger' : undefined}>{val}ms</Text>
      ),
      sorter: (a: ServiceMetric, b: ServiceMetric) => a.latencyP99 - b.latencyP99,
    },
    {
      title: 'RPS',
      dataIndex: 'requestsPerSecond',
      key: 'requestsPerSecond',
      render: (val: number) => <Text>{val.toLocaleString()}</Text>,
      sorter: (a: ServiceMetric, b: ServiceMetric) => a.requestsPerSecond - b.requestsPerSecond,
    },
    {
      title: 'Error Rate',
      dataIndex: 'errorRate',
      key: 'errorRate',
      render: (val: number) => (
        <Tag color={val > 0.05 ? 'red' : val > 0.02 ? 'orange' : 'green'}>
          {(val * 100).toFixed(2)}%
        </Tag>
      ),
      sorter: (a: ServiceMetric, b: ServiceMetric) => a.errorRate - b.errorRate,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
      ),
    },
  ];

  // Calculate aggregates
  const totalRequests = services.reduce((sum, s) => sum + s.requestsPerSecond, 0);
  const avgLatency = services.reduce((sum, s) => sum + s.latencyP50, 0) / services.length;
  const avgErrorRate = services.reduce((sum, s) => sum + s.errorRate, 0) / services.length;
  const healthyServices = services.filter(s => s.status === 'healthy').length;

  return (
    <div style={{ padding: 24 }}>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}>
            <DashboardOutlined style={{ marginRight: 8 }} />
            Performance Monitor
          </Title>
        </Col>
        <Col>
          <Space>
            <Select value={timeRange} onChange={setTimeRange} style={{ width: 100 }}>
              <Option value="1h">1 Hour</Option>
              <Option value="6h">6 Hours</Option>
              <Option value="24h">24 Hours</Option>
              <Option value="7d">7 Days</Option>
            </Select>
            <Button 
              icon={<SyncOutlined spin={refreshing} />} 
              onClick={handleRefresh}
            >
              Refresh
            </Button>
          </Space>
        </Col>
      </Row>

      {/* Alert for degraded services */}
      {services.some(s => s.status !== 'healthy') && (
        <Alert
          message="Performance Degradation Detected"
          description={`${services.filter(s => s.status !== 'healthy').length} service(s) showing degraded performance.`}
          type="warning"
          showIcon
          icon={<WarningOutlined />}
          style={{ marginBottom: 24 }}
        />
      )}

      {/* Overview Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Requests/sec"
              value={totalRequests}
              prefix={<ThunderboltOutlined />}
              suffix={
                <Text type="success" style={{ fontSize: 14 }}>
                  <ArrowUpOutlined /> 12%
                </Text>
              }
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Avg Latency (P50)"
              value={Math.round(avgLatency)}
              suffix="ms"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Avg Error Rate"
              value={(avgErrorRate * 100).toFixed(2)}
              suffix="%"
              prefix={<WarningOutlined />}
              valueStyle={{ color: avgErrorRate > 0.03 ? '#cf1322' : '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Healthy Services"
              value={healthyServices}
              suffix={`/ ${services.length}`}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: healthyServices === services.length ? '#3f8600' : '#faad14' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Charts */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="Latency Trend (ms)">
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={timeSeriesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Area 
                  type="monotone" 
                  dataKey="latency" 
                  stroke="#1890ff" 
                  fill="#1890ff" 
                  fillOpacity={0.3} 
                />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Request Throughput">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={timeSeriesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="requests" 
                  stroke="#52c41a" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      {/* Service Table */}
      <Card title="Service Metrics">
        <Table
          columns={serviceColumns}
          dataSource={services}
          rowKey="name"
          pagination={false}
          size="middle"
        />
      </Card>

      {/* SLO Compliance */}
      <Card title="SLO Compliance" style={{ marginTop: 24 }}>
        <Row gutter={[24, 24]}>
          {[
            { name: 'Availability (99.9%)', current: 99.95, target: 99.9 },
            { name: 'P95 Latency (< 3s)', current: 2.5, target: 3, unit: 's', inverse: true },
            { name: 'Error Rate (< 2%)', current: 1.2, target: 2, unit: '%', inverse: true },
            { name: 'Throughput (> 1000 RPS)', current: 1250, target: 1000, unit: ' RPS' },
          ].map((slo) => {
            const met = slo.inverse 
              ? slo.current <= slo.target 
              : slo.current >= slo.target;
            const percent = slo.inverse
              ? Math.min(100, (slo.target / slo.current) * 100)
              : Math.min(100, (slo.current / slo.target) * 100);
            
            return (
              <Col xs={24} sm={12} md={6} key={slo.name}>
                <div style={{ textAlign: 'center' }}>
                  <Progress
                    type="dashboard"
                    percent={percent}
                    status={met ? 'success' : 'exception'}
                    format={() => (
                      <div>
                        <div style={{ fontSize: 18, fontWeight: 600 }}>
                          {slo.current}{slo.unit || '%'}
                        </div>
                        <div style={{ fontSize: 12, color: '#888' }}>
                          Target: {slo.target}{slo.unit || '%'}
                        </div>
                      </div>
                    )}
                  />
                  <div style={{ marginTop: 8 }}>
                    <Text>{slo.name}</Text>
                  </div>
                </div>
              </Col>
            );
          })}
        </Row>
      </Card>
    </div>
  );
};

export default PerformanceMonitor;
