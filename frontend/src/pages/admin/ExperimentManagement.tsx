import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Modal,
  Form,
  Input,
  Select,
  Typography,
  Tooltip,
  Popconfirm,
  message,
  Progress,
  Descriptions
} from 'antd';
import {
  PlusOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExperimentOutlined,
  ArrowUpOutlined,
  StopOutlined
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { apiService } from '../../services/api';
import './ExperimentManagement.css';

const { Title, Text } = Typography;
const { TextArea } = Input;

interface Experiment {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'promoted' | 'quarantined';
  config: {
    model: string;
    temperature: number;
    prompt_template: string;
  };
  dataset_id: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    latency_p95: number;
    error_rate: number;
    cost_per_analysis: number;
  };
}

export const ExperimentManagement: React.FC = () => {
  const { t } = useTranslation();
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);
  const [form] = Form.useForm();

  // Fetch experiments
  useEffect(() => {
    fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    setLoading(true);
    try {
      const response = await apiService.experiments.list();
      setExperiments(response.data.items || []);
    } catch (error) {
      message.error(t('experiments.fetch_error', 'Failed to fetch experiments'));
    } finally {
      setLoading(false);
    }
  };

  // Create experiment
  const handleCreate = async (values: any) => {
    try {
      await apiService.experiments.create({
        name: values.name,
        config: {
          model: values.model,
          temperature: parseFloat(values.temperature),
          prompt_template: values.prompt_template
        },
        dataset_id: values.dataset_id
      });
      message.success(t('experiments.create_success', 'Experiment created'));
      setCreateModalVisible(false);
      form.resetFields();
      fetchExperiments();
    } catch (error) {
      message.error(t('experiments.create_error', 'Failed to create experiment'));
    }
  };

  // Start experiment
  const handleStart = async (id: string) => {
    try {
      await apiService.experiments.start(id);
      message.success(t('experiments.start_success', 'Experiment started'));
      fetchExperiments();
    } catch (error) {
      message.error(t('experiments.start_error', 'Failed to start experiment'));
    }
  };

  // Stop experiment
  const handleStop = async (id: string) => {
    try {
      await apiService.experiments.stop(id);
      message.success(t('experiments.stop_success', 'Experiment stopped'));
      fetchExperiments();
    } catch (error) {
      message.error(t('experiments.stop_error', 'Failed to stop experiment'));
    }
  };

  // Promote experiment
  const handlePromote = async (id: string) => {
    try {
      await apiService.experiments.promote(id);
      message.success(t('experiments.promote_success', 'Experiment promoted to v2'));
      fetchExperiments();
    } catch (error: any) {
      message.error(error.response?.data?.detail || t('experiments.promote_error', 'Failed to promote experiment'));
    }
  };

  // Quarantine experiment
  const handleQuarantine = async (id: string, reason: string) => {
    try {
      await apiService.experiments.quarantine(id, reason);
      message.success(t('experiments.quarantine_success', 'Experiment quarantined'));
      fetchExperiments();
    } catch (error) {
      message.error(t('experiments.quarantine_error', 'Failed to quarantine experiment'));
    }
  };

  // Get status tag
  const getStatusTag = (status: string) => {
    const statusConfig: Record<string, { color: string; icon: React.ReactNode }> = {
      pending: { color: 'default', icon: <ExperimentOutlined /> },
      running: { color: 'processing', icon: <PlayCircleOutlined /> },
      completed: { color: 'success', icon: <CheckCircleOutlined /> },
      failed: { color: 'error', icon: <CloseCircleOutlined /> },
      promoted: { color: 'green', icon: <ArrowUpOutlined /> },
      quarantined: { color: 'orange', icon: <StopOutlined /> }
    };
    const config = statusConfig[status] || statusConfig.pending;
    return (
      <Tag color={config.color} icon={config.icon}>
        {status.toUpperCase()}
      </Tag>
    );
  };

  // Table columns
  const columns = [
    {
      title: t('experiments.name', 'Name'),
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: Experiment) => (
        <a onClick={() => {
          setSelectedExperiment(record);
          setDetailModalVisible(true);
        }}>
          {name}
        </a>
      )
    },
    {
      title: t('experiments.model', 'Model'),
      dataIndex: ['config', 'model'],
      key: 'model',
      render: (model: string) => <Tag>{model}</Tag>
    },
    {
      title: t('experiments.status', 'Status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusTag(status)
    },
    {
      title: t('experiments.accuracy', 'Accuracy'),
      dataIndex: ['metrics', 'accuracy'],
      key: 'accuracy',
      render: (accuracy: number) => accuracy ? (
        <Progress
          percent={Math.round(accuracy * 100)}
          size="small"
          status={accuracy >= 0.85 ? 'success' : accuracy >= 0.7 ? 'normal' : 'exception'}
        />
      ) : '-'
    },
    {
      title: t('experiments.error_rate', 'Error Rate'),
      dataIndex: ['metrics', 'error_rate'],
      key: 'error_rate',
      render: (rate: number) => rate !== undefined ? (
        <Tag color={rate <= 0.05 ? 'green' : rate <= 0.1 ? 'orange' : 'red'}>
          {(rate * 100).toFixed(1)}%
        </Tag>
      ) : '-'
    },
    {
      title: t('experiments.created', 'Created'),
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleString()
    },
    {
      title: t('experiments.actions', 'Actions'),
      key: 'actions',
      render: (_: any, record: Experiment) => (
        <Space>
          {record.status === 'pending' && (
            <Tooltip title={t('experiments.start', 'Start')}>
              <Button
                type="primary"
                size="small"
                icon={<PlayCircleOutlined />}
                onClick={() => handleStart(record.id)}
              />
            </Tooltip>
          )}
          {record.status === 'running' && (
            <Tooltip title={t('experiments.stop', 'Stop')}>
              <Button
                size="small"
                icon={<PauseCircleOutlined />}
                onClick={() => handleStop(record.id)}
              />
            </Tooltip>
          )}
          {record.status === 'completed' && record.metrics && (
            <>
              <Popconfirm
                title={t('experiments.promote_confirm', 'Promote to v2?')}
                onConfirm={() => handlePromote(record.id)}
                disabled={record.metrics.accuracy < 0.85 || record.metrics.error_rate > 0.05}
              >
                <Tooltip title={
                  record.metrics.accuracy >= 0.85 && record.metrics.error_rate <= 0.05
                    ? t('experiments.promote', 'Promote')
                    : t('experiments.promote_disabled', 'Does not meet criteria')
                }>
                  <Button
                    type="primary"
                    size="small"
                    icon={<ArrowUpOutlined />}
                    disabled={record.metrics.accuracy < 0.85 || record.metrics.error_rate > 0.05}
                  />
                </Tooltip>
              </Popconfirm>
              <Popconfirm
                title={t('experiments.quarantine_confirm', 'Quarantine this experiment?')}
                onConfirm={() => handleQuarantine(record.id, 'Manual quarantine')}
              >
                <Tooltip title={t('experiments.quarantine', 'Quarantine')}>
                  <Button
                    danger
                    size="small"
                    icon={<StopOutlined />}
                  />
                </Tooltip>
              </Popconfirm>
            </>
          )}
        </Space>
      )
    }
  ];

  return (
    <div className="experiment-management">
      <div className="page-header">
        <Title level={2}>{t('experiments.title', 'Experiment Management')}</Title>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setCreateModalVisible(true)}
        >
          {t('experiments.create', 'Create Experiment')}
        </Button>
      </div>

      <Card>
        <Table
          columns={columns}
          dataSource={experiments}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* Create Modal */}
      <Modal
        title={t('experiments.create_title', 'Create New Experiment')}
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          <Form.Item
            name="name"
            label={t('experiments.name', 'Name')}
            rules={[{ required: true }]}
          >
            <Input placeholder="e.g., gpt-4-turbo-security-v2" />
          </Form.Item>
          <Form.Item
            name="model"
            label={t('experiments.model', 'Model')}
            rules={[{ required: true }]}
          >
            <Select>
              <Select.Option value="gpt-4-turbo">GPT-4 Turbo</Select.Option>
              <Select.Option value="gpt-4">GPT-4</Select.Option>
              <Select.Option value="claude-3-opus">Claude 3 Opus</Select.Option>
              <Select.Option value="claude-3-sonnet">Claude 3 Sonnet</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="temperature"
            label={t('experiments.temperature', 'Temperature')}
            rules={[{ required: true }]}
            initialValue="0.7"
          >
            <Input type="number" step="0.1" min="0" max="2" />
          </Form.Item>
          <Form.Item
            name="prompt_template"
            label={t('experiments.prompt_template', 'Prompt Template')}
            rules={[{ required: true }]}
          >
            <Select>
              <Select.Option value="security_expert_v1">Security Expert v1</Select.Option>
              <Select.Option value="security_expert_v2">Security Expert v2</Select.Option>
              <Select.Option value="code_quality_v1">Code Quality v1</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="dataset_id"
            label={t('experiments.dataset', 'Dataset')}
            rules={[{ required: true }]}
          >
            <Select>
              <Select.Option value="test-dataset-100">Test Dataset (100 samples)</Select.Option>
              <Select.Option value="test-dataset-500">Test Dataset (500 samples)</Select.Option>
              <Select.Option value="production-sample">Production Sample</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                {t('common.create', 'Create')}
              </Button>
              <Button onClick={() => setCreateModalVisible(false)}>
                {t('common.cancel', 'Cancel')}
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Detail Modal */}
      <Modal
        title={selectedExperiment?.name}
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={700}
      >
        {selectedExperiment && (
          <Descriptions bordered column={2}>
            <Descriptions.Item label={t('experiments.status', 'Status')}>
              {getStatusTag(selectedExperiment.status)}
            </Descriptions.Item>
            <Descriptions.Item label={t('experiments.model', 'Model')}>
              {selectedExperiment.config.model}
            </Descriptions.Item>
            <Descriptions.Item label={t('experiments.temperature', 'Temperature')}>
              {selectedExperiment.config.temperature}
            </Descriptions.Item>
            <Descriptions.Item label={t('experiments.prompt_template', 'Prompt Template')}>
              {selectedExperiment.config.prompt_template}
            </Descriptions.Item>
            {selectedExperiment.metrics && (
              <>
                <Descriptions.Item label={t('experiments.accuracy', 'Accuracy')}>
                  <Progress
                    percent={Math.round(selectedExperiment.metrics.accuracy * 100)}
                    status={selectedExperiment.metrics.accuracy >= 0.85 ? 'success' : 'exception'}
                  />
                </Descriptions.Item>
                <Descriptions.Item label={t('experiments.error_rate', 'Error Rate')}>
                  {(selectedExperiment.metrics.error_rate * 100).toFixed(2)}%
                </Descriptions.Item>
                <Descriptions.Item label={t('experiments.latency', 'Latency (p95)')}>
                  {selectedExperiment.metrics.latency_p95.toFixed(2)}s
                </Descriptions.Item>
                <Descriptions.Item label={t('experiments.cost', 'Cost/Analysis')}>
                  ${selectedExperiment.metrics.cost_per_analysis.toFixed(4)}
                </Descriptions.Item>
              </>
            )}
          </Descriptions>
        )}
      </Modal>
    </div>
  );
};

export default ExperimentManagement;
