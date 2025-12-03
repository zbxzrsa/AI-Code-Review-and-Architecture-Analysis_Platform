/**
 * Code Quality Widget
 * 
 * Dashboard widget showing code quality metrics:
 * - Overall score
 * - Category breakdown
 * - Trend over time
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Progress,
  Statistic,
  Space,
  Typography,
  Tag,
  Tooltip,
  Spin,
} from 'antd';
import {
  SafetyOutlined,
  BugOutlined,
  CodeOutlined,
  ThunderboltOutlined,
  FileTextOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

interface QualityMetric {
  name: string;
  score: number;
  trend: number;
  icon: React.ReactNode;
  color: string;
}

const CodeQualityWidget: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [overallScore, setOverallScore] = useState(0);
  const [metrics, setMetrics] = useState<QualityMetric[]>([]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setOverallScore(85);
      setMetrics([
        {
          name: 'Security',
          score: 92,
          trend: 5,
          icon: <SafetyOutlined />,
          color: '#52c41a',
        },
        {
          name: 'Reliability',
          score: 88,
          trend: 3,
          icon: <BugOutlined />,
          color: '#1890ff',
        },
        {
          name: 'Maintainability',
          score: 78,
          trend: -2,
          icon: <CodeOutlined />,
          color: '#faad14',
        },
        {
          name: 'Performance',
          score: 85,
          trend: 8,
          icon: <ThunderboltOutlined />,
          color: '#722ed1',
        },
        {
          name: 'Documentation',
          score: 72,
          trend: 0,
          icon: <FileTextOutlined />,
          color: '#eb2f96',
        },
      ]);
      setLoading(false);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  const getScoreColor = (score: number) => {
    if (score >= 90) return '#52c41a';
    if (score >= 70) return '#faad14';
    return '#ff4d4f';
  };

  const getGrade = (score: number) => {
    if (score >= 90) return 'A';
    if (score >= 80) return 'B';
    if (score >= 70) return 'C';
    if (score >= 60) return 'D';
    return 'F';
  };

  if (loading) {
    return (
      <Card title="Code Quality">
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
          <CodeOutlined />
          Code Quality Score
        </Space>
      }
      extra={
        <Tag color={getScoreColor(overallScore)} style={{ fontSize: 16, padding: '4px 12px' }}>
          Grade: {getGrade(overallScore)}
        </Tag>
      }
    >
      {/* Overall Score */}
      <Row justify="center" style={{ marginBottom: 24 }}>
        <Col>
          <Progress
            type="dashboard"
            percent={overallScore}
            strokeColor={getScoreColor(overallScore)}
            format={(percent) => (
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 32, fontWeight: 'bold', color: getScoreColor(percent || 0) }}>
                  {percent}
                </div>
                <div style={{ fontSize: 12, color: '#888' }}>Overall Score</div>
              </div>
            )}
            width={150}
          />
        </Col>
      </Row>

      {/* Category Breakdown */}
      <Row gutter={[8, 12]}>
        {metrics.map((metric) => (
          <Col span={24} key={metric.name}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Tooltip title={metric.name}>
                <span style={{ color: metric.color, fontSize: 16, width: 24 }}>
                  {metric.icon}
                </span>
              </Tooltip>
              <Text style={{ width: 100 }}>{metric.name}</Text>
              <Progress
                percent={metric.score}
                size="small"
                strokeColor={metric.color}
                style={{ flex: 1 }}
                format={(percent) => `${percent}%`}
              />
              <span style={{ width: 50, textAlign: 'right' }}>
                {metric.trend > 0 ? (
                  <Text type="success">
                    <ArrowUpOutlined /> {metric.trend}%
                  </Text>
                ) : metric.trend < 0 ? (
                  <Text type="danger">
                    <ArrowDownOutlined /> {Math.abs(metric.trend)}%
                  </Text>
                ) : (
                  <Text type="secondary">-</Text>
                )}
              </span>
            </div>
          </Col>
        ))}
      </Row>
    </Card>
  );
};

export default CodeQualityWidget;
