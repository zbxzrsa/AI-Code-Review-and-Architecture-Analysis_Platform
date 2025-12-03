/**
 * Deployments Page
 * 部署管理页面
 * 
 * CI/CD Pipeline visualization with:
 * - Deployment history
 * - Pipeline status
 * - Rollback capability
 * - Environment management
 */

import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Tag,
  Space,
  Typography,
  Avatar,
  Tooltip,
  Select,
  Timeline,
  Modal,
  Statistic,
  Steps,
  Alert,
  Descriptions,
  message,
} from 'antd';
import type { TableProps } from 'antd';
import {
  RocketOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  ClockCircleOutlined,
  RollbackOutlined,
  EyeOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  EnvironmentOutlined,
  BranchesOutlined,
  UserOutlined,
  HistoryOutlined,
  CloudServerOutlined,
  ThunderboltOutlined,
  SafetyCertificateOutlined,
  CodeOutlined,
  BuildOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text, Paragraph } = Typography;

interface Deployment {
  id: string;
  version: string;
  environment: 'production' | 'staging' | 'development';
  status: 'success' | 'failed' | 'in_progress' | 'pending' | 'rolled_back';
  branch: string;
  commit: string;
  commitMessage: string;
  deployedBy: string;
  startedAt: string;
  completedAt?: string;
  duration?: number;
  stages: {
    name: string;
    status: 'success' | 'failed' | 'in_progress' | 'pending' | 'skipped';
    duration?: number;
  }[];
}

const mockDeployments: Deployment[] = [
  {
    id: 'deploy_1',
    version: 'v2.1.5',
    environment: 'production',
    status: 'success',
    branch: 'main',
    commit: 'a1b2c3d',
    commitMessage: 'Add user authentication with JWT tokens',
    deployedBy: 'CI/CD Pipeline',
    startedAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    completedAt: new Date(Date.now() - 2 * 60 * 60 * 1000 + 8 * 60 * 1000).toISOString(),
    duration: 480,
    stages: [
      { name: 'Build', status: 'success', duration: 120 },
      { name: 'Test', status: 'success', duration: 180 },
      { name: 'Security Scan', status: 'success', duration: 60 },
      { name: 'Deploy', status: 'success', duration: 120 },
    ],
  },
  {
    id: 'deploy_2',
    version: 'v2.1.4',
    environment: 'staging',
    status: 'in_progress',
    branch: 'develop',
    commit: 'e4f5g6h',
    commitMessage: 'Implement dashboard analytics',
    deployedBy: 'John Doe',
    startedAt: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    stages: [
      { name: 'Build', status: 'success', duration: 115 },
      { name: 'Test', status: 'success', duration: 165 },
      { name: 'Security Scan', status: 'in_progress' },
      { name: 'Deploy', status: 'pending' },
    ],
  },
  {
    id: 'deploy_3',
    version: 'v2.1.3',
    environment: 'production',
    status: 'failed',
    branch: 'main',
    commit: 'i7j8k9l',
    commitMessage: 'Update database connection pooling',
    deployedBy: 'CI/CD Pipeline',
    startedAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    completedAt: new Date(Date.now() - 24 * 60 * 60 * 1000 + 5 * 60 * 1000).toISOString(),
    duration: 300,
    stages: [
      { name: 'Build', status: 'success', duration: 118 },
      { name: 'Test', status: 'failed', duration: 182 },
      { name: 'Security Scan', status: 'skipped' },
      { name: 'Deploy', status: 'skipped' },
    ],
  },
  {
    id: 'deploy_4',
    version: 'v2.1.2',
    environment: 'production',
    status: 'rolled_back',
    branch: 'main',
    commit: 'm1n2o3p',
    commitMessage: 'Performance optimizations',
    deployedBy: 'Jane Smith',
    startedAt: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
    completedAt: new Date(Date.now() - 48 * 60 * 60 * 1000 + 7 * 60 * 1000).toISOString(),
    duration: 420,
    stages: [
      { name: 'Build', status: 'success', duration: 122 },
      { name: 'Test', status: 'success', duration: 175 },
      { name: 'Security Scan', status: 'success', duration: 58 },
      { name: 'Deploy', status: 'success', duration: 65 },
    ],
  },
];

const environmentConfig = {
  production: { color: '#ef4444', icon: <CloudServerOutlined />, label: 'Production' },
  staging: { color: '#f59e0b', icon: <EnvironmentOutlined />, label: 'Staging' },
  development: { color: '#22c55e', icon: <CodeOutlined />, label: 'Development' },
};

const statusConfig = {
  success: { color: 'success', icon: <CheckCircleOutlined />, text: 'Success' },
  failed: { color: 'error', icon: <CloseCircleOutlined />, text: 'Failed' },
  in_progress: { color: 'processing', icon: <SyncOutlined spin />, text: 'In Progress' },
  pending: { color: 'default', icon: <ClockCircleOutlined />, text: 'Pending' },
  rolled_back: { color: 'warning', icon: <RollbackOutlined />, text: 'Rolled Back' },
  skipped: { color: 'default', icon: <PauseCircleOutlined />, text: 'Skipped' },
};

export const Deployments: React.FC = () => {
  const { t: _t } = useTranslation();
  const [deployments, setDeployments] = useState<Deployment[]>(mockDeployments);
  const [selectedDeployment, setSelectedDeployment] = useState<Deployment | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [envFilter, setEnvFilter] = useState('all');

  const filteredDeployments = envFilter === 'all'
    ? deployments
    : deployments.filter(d => d.environment === envFilter);

  const handleRollback = async (deployment: Deployment) => {
    message.success(`Rolling back to ${deployment.version}`);
    setDeployments(prev => prev.map(d =>
      d.id === deployment.id ? { ...d, status: 'rolled_back' as const } : d
    ));
  };

  const handleRedeploy = async (deployment: Deployment) => {
    message.success(`Redeploying ${deployment.version}`);
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const currentInProgress = deployments.find(d => d.status === 'in_progress');
  const stats = {
    total: deployments.length,
    success: deployments.filter(d => d.status === 'success').length,
    failed: deployments.filter(d => d.status === 'failed').length,
  };

  const columns: TableProps<Deployment>['columns'] = [
    {
      title: 'Deployment',
      key: 'deployment',
      render: (_, record) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 40,
            height: 40,
            borderRadius: 10,
            background: statusConfig[record.status].color === 'success' ? '#dcfce7' :
                       statusConfig[record.status].color === 'error' ? '#fee2e2' :
                       statusConfig[record.status].color === 'processing' ? '#dbeafe' : '#f1f5f9',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: statusConfig[record.status].color === 'success' ? '#22c55e' :
                   statusConfig[record.status].color === 'error' ? '#ef4444' :
                   statusConfig[record.status].color === 'processing' ? '#3b82f6' : '#64748b',
            fontSize: 18,
          }}>
            <RocketOutlined />
          </div>
          <div>
            <Space>
              <Text strong>{record.version}</Text>
              <Tag color={statusConfig[record.status].color} icon={statusConfig[record.status].icon}>
                {statusConfig[record.status].text}
              </Tag>
            </Space>
            <div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {record.commitMessage}
              </Text>
            </div>
          </div>
        </div>
      ),
    },
    {
      title: 'Environment',
      dataIndex: 'environment',
      width: 130,
      render: (env) => {
        const config = environmentConfig[env as keyof typeof environmentConfig];
        return (
          <Tag color={config.color} icon={config.icon}>
            {config.label}
          </Tag>
        );
      },
    },
    {
      title: 'Branch',
      key: 'branch',
      width: 150,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Text><BranchesOutlined /> {record.branch}</Text>
          <Text code style={{ fontSize: 11 }}>{record.commit}</Text>
        </Space>
      ),
    },
    {
      title: 'Deployed By',
      dataIndex: 'deployedBy',
      width: 150,
      render: (user) => (
        <Space>
          <Avatar size="small" icon={<UserOutlined />} />
          <Text>{user}</Text>
        </Space>
      ),
    },
    {
      title: 'Duration',
      key: 'duration',
      width: 100,
      render: (_, record) => record.duration ? formatDuration(record.duration) : '-',
    },
    {
      title: 'Time',
      key: 'time',
      width: 150,
      render: (_, record) => (
        <Tooltip title={new Date(record.startedAt).toLocaleString()}>
          <Text type="secondary">
            <ClockCircleOutlined /> {new Date(record.startedAt).toLocaleDateString()}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (_, record) => (
        <Space>
          <Tooltip title="View Details">
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedDeployment(record);
                setDetailModalOpen(true);
              }}
            />
          </Tooltip>
          {record.status === 'success' && record.environment === 'production' && (
            <Tooltip title="Rollback">
              <Button
                size="small"
                icon={<RollbackOutlined />}
                onClick={() => handleRollback(record)}
              />
            </Tooltip>
          )}
          {record.status === 'failed' && (
            <Tooltip title="Redeploy">
              <Button
                size="small"
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={() => handleRedeploy(record)}
              />
            </Tooltip>
          )}
        </Space>
      ),
    },
  ];

  return (
    <div className="deployments-page" style={{ maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <RocketOutlined style={{ color: '#2563eb' }} /> Deployments
          </Title>
          <Text type="secondary">CI/CD pipeline management and deployment history</Text>
        </div>
        <Space>
          <Select
            value={envFilter}
            onChange={setEnvFilter}
            style={{ width: 150 }}
            options={[
              { value: 'all', label: 'All Environments' },
              { value: 'production', label: 'Production' },
              { value: 'staging', label: 'Staging' },
              { value: 'development', label: 'Development' },
            ]}
          />
          <Button type="primary" icon={<RocketOutlined />}>
            Deploy Now
          </Button>
        </Space>
      </div>

      {/* Active Deployment Alert */}
      {currentInProgress && (
        <Alert
          type="info"
          showIcon
          icon={<SyncOutlined spin />}
          message={`Deployment in Progress: ${currentInProgress.version}`}
          description={
            <div>
              <Text>{currentInProgress.commitMessage}</Text>
              <div style={{ marginTop: 12 }}>
                <Steps
                  size="small"
                  current={currentInProgress.stages.findIndex(s => s.status === 'in_progress')}
                  items={currentInProgress.stages.map(stage => ({
                    title: stage.name,
                    status: stage.status === 'success' ? 'finish' :
                           stage.status === 'in_progress' ? 'process' :
                           stage.status === 'failed' ? 'error' : 'wait',
                  }))}
                />
              </div>
            </div>
          }
          style={{ marginBottom: 24 }}
        />
      )}

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={8}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Total Deployments"
              value={stats.total}
              prefix={<RocketOutlined />}
            />
          </Card>
        </Col>
        <Col xs={8}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Successful"
              value={stats.success}
              valueStyle={{ color: '#22c55e' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={8}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic
              title="Success Rate"
              value={Math.round((stats.success / stats.total) * 100)}
              suffix="%"
              valueStyle={{ color: '#2563eb' }}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Pipeline Stages Overview */}
      <Card 
        title={<><BuildOutlined /> Pipeline Stages</>} 
        style={{ marginBottom: 24, borderRadius: 12 }}
      >
        <Row gutter={24}>
          {['Build', 'Test', 'Security Scan', 'Deploy'].map((stage, index) => (
            <Col key={stage} span={6}>
              <Card size="small" style={{ textAlign: 'center', borderRadius: 10 }}>
                <div style={{
                  width: 48,
                  height: 48,
                  borderRadius: '50%',
                  background: '#dbeafe',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 12px',
                  fontSize: 20,
                  color: '#2563eb',
                }}>
                  {index === 0 && <BuildOutlined />}
                  {index === 1 && <CheckCircleOutlined />}
                  {index === 2 && <SafetyCertificateOutlined />}
                  {index === 3 && <RocketOutlined />}
                </div>
                <Text strong>{stage}</Text>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>

      {/* Deployments Table */}
      <Card title={<><HistoryOutlined /> Deployment History</>} style={{ borderRadius: 12 }}>
        <Table
          columns={columns}
          dataSource={filteredDeployments}
          rowKey="id"
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* Detail Modal */}
      <Modal
        title={<><RocketOutlined /> Deployment Details</>}
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        width={700}
        footer={null}
      >
        {selectedDeployment && (
          <div>
            <Descriptions bordered column={2}>
              <Descriptions.Item label="Version">{selectedDeployment.version}</Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={statusConfig[selectedDeployment.status].color}>
                  {statusConfig[selectedDeployment.status].text}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Environment">
                <Tag color={environmentConfig[selectedDeployment.environment].color}>
                  {environmentConfig[selectedDeployment.environment].label}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Branch">{selectedDeployment.branch}</Descriptions.Item>
              <Descriptions.Item label="Commit" span={2}>
                <Text code>{selectedDeployment.commit}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="Message" span={2}>
                {selectedDeployment.commitMessage}
              </Descriptions.Item>
              <Descriptions.Item label="Deployed By">{selectedDeployment.deployedBy}</Descriptions.Item>
              <Descriptions.Item label="Duration">
                {selectedDeployment.duration ? formatDuration(selectedDeployment.duration) : '-'}
              </Descriptions.Item>
            </Descriptions>

            <Title level={5} style={{ marginTop: 24 }}>Pipeline Stages</Title>
            <Timeline
              items={selectedDeployment.stages.map(stage => ({
                color: stage.status === 'success' ? 'green' :
                       stage.status === 'failed' ? 'red' :
                       stage.status === 'in_progress' ? 'blue' : 'gray',
                children: (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Space>
                      {statusConfig[stage.status]?.icon}
                      <Text strong>{stage.name}</Text>
                    </Space>
                    <Text type="secondary">
                      {stage.duration ? formatDuration(stage.duration) : '-'}
                    </Text>
                  </div>
                ),
              }))}
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default Deployments;
