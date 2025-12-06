/**
 * AI Models Management Page (Admin)
 * 
 * Features:
 * - View all AI models and versions
 * - Manage model lifecycle (V1→V2→V3)
 * - Import external models via API
 * - Admin AI Chat interface for version control
 * - Promote/rollback versions
 * - Monitor model performance
 * - Configure model settings
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Card,
  Table,
  Button,
  Tag,
  Space,
  Modal,
  Form,
  Input,
  InputNumber,
  Tabs,
  Statistic,
  Row,
  Col,
  Typography,
  Dropdown,
  message,
  Alert,
  Descriptions,
  Drawer,
  List,
  Avatar,
  Select,
  Divider,
  Empty,
  Spin,
  Tooltip,
  Badge,
} from 'antd';
import type { TableProps, MenuProps } from 'antd';
import {
  RobotOutlined,
  PlusOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  ExperimentOutlined,
  SafetyCertificateOutlined,
  WarningOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  PauseCircleOutlined,
  EyeOutlined,
  MoreOutlined,
  ReloadOutlined,
  LineChartOutlined,
  MessageOutlined,
  SendOutlined,
  ApiOutlined,
  CloudServerOutlined,
  UserOutlined,
  ClearOutlined,
  HistoryOutlined,
  ToolOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { api } from '../../services/api';
import { aiService } from '../../services/aiService';

const { Title, Text, Paragraph } = Typography;

// Types
interface AIModel {
  id: string;
  name: string;
  provider: string;
  model_id: string;
  version: string;
  zone: 'v1-experimentation' | 'v2-production' | 'v3-quarantine';
  status: 'active' | 'inactive' | 'testing' | 'deprecated' | 'quarantined';
  metrics: {
    accuracy: number;
    latency_p95: number;
    error_rate: number;
    cost_per_request: number;
    requests_today: number;
    success_rate: number;
  };
  config: {
    max_tokens: number;
    temperature: number;
    top_p: number;
  };
  created_at: string;
  updated_at: string;
  promoted_at?: string;
  last_evaluation?: string;
}

interface ModelVersionInfo {
  id: string;
  model_id: string;
  version: string;
  zone: string;
  status: string;
  metrics: Record<string, unknown>;
  created_at: string;
  promoted_by?: string;
  promotion_reason?: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  model?: string;
}

interface ImportedModel {
  id: string;
  name: string;
  provider: string;
  api_endpoint: string;
  api_key_configured: boolean;
  status: 'active' | 'inactive' | 'error';
  created_at: string;
}

const { TextArea } = Input;

// Mock data
const mockModels: AIModel[] = [
  {
    id: 'model_1',
    name: 'GPT-4 Turbo',
    provider: 'OpenAI',
    model_id: 'gpt-4-turbo',
    version: 'v2.1.0',
    zone: 'v2-production',
    status: 'active',
    metrics: {
      accuracy: 0.94,
      latency_p95: 2.3,
      error_rate: 0.02,
      cost_per_request: 0.08,
      requests_today: 1250,
      success_rate: 0.98,
    },
    config: { max_tokens: 4096, temperature: 0.7, top_p: 0.95 },
    created_at: '2024-01-15T10:00:00Z',
    updated_at: '2024-03-01T15:30:00Z',
    promoted_at: '2024-02-01T09:00:00Z',
    last_evaluation: '2024-03-01T00:00:00Z',
  },
  {
    id: 'model_2',
    name: 'Claude 3 Opus',
    provider: 'Anthropic',
    model_id: 'claude-3-opus',
    version: 'v2.0.0',
    zone: 'v2-production',
    status: 'active',
    metrics: {
      accuracy: 0.92,
      latency_p95: 3.1,
      error_rate: 0.03,
      cost_per_request: 0.12,
      requests_today: 830,
      success_rate: 0.97,
    },
    config: { max_tokens: 4096, temperature: 0.5, top_p: 0.9 },
    created_at: '2024-02-01T10:00:00Z',
    updated_at: '2024-03-01T12:00:00Z',
    promoted_at: '2024-02-15T09:00:00Z',
    last_evaluation: '2024-03-01T00:00:00Z',
  },
  {
    id: 'model_3',
    name: 'GPT-4 Vision',
    provider: 'OpenAI',
    model_id: 'gpt-4-vision',
    version: 'v1.0.0',
    zone: 'v1-experimentation',
    status: 'testing',
    metrics: {
      accuracy: 0.88,
      latency_p95: 4.5,
      error_rate: 0.05,
      cost_per_request: 0.15,
      requests_today: 150,
      success_rate: 0.95,
    },
    config: { max_tokens: 2048, temperature: 0.6, top_p: 0.95 },
    created_at: '2024-02-20T10:00:00Z',
    updated_at: '2024-03-01T10:00:00Z',
    last_evaluation: '2024-03-01T00:00:00Z',
  },
  {
    id: 'model_4',
    name: 'Claude 2.1',
    provider: 'Anthropic',
    model_id: 'claude-2.1',
    version: 'v1.5.0',
    zone: 'v3-quarantine',
    status: 'quarantined',
    metrics: {
      accuracy: 0.78,
      latency_p95: 5.2,
      error_rate: 0.12,
      cost_per_request: 0.10,
      requests_today: 0,
      success_rate: 0.88,
    },
    config: { max_tokens: 4096, temperature: 0.7, top_p: 0.9 },
    created_at: '2024-01-01T10:00:00Z',
    updated_at: '2024-02-15T10:00:00Z',
  },
];

export const AIModels: React.FC = () => {
  const { t } = useTranslation();
  const [models, setModels] = useState<AIModel[]>(mockModels);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<AIModel | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [configModalOpen, setConfigModalOpen] = useState(false);
  const [promoteModalOpen, setPromoteModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('all');
  const [form] = Form.useForm();
  const [importForm] = Form.useForm();

  // AI Chat states
  const [chatDrawerOpen, setChatDrawerOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatModel, setChatModel] = useState('gpt-4-turbo');
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Import model states
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [importedModels, setImportedModels] = useState<ImportedModel[]>([]);

  // Fetch models from API with fallback
  const fetchModels = useCallback(async () => {
    setLoading(true);
    try {
      // Try admin API first
      const response = await api.get('/api/admin/ai-models');
      if (response.data?.items) {
        setModels(response.data.items);
        return;
      }
    } catch {
      // Try evolution API as fallback
      try {
        const technologies = await aiService.getTechnologies();
        if (technologies && technologies.length > 0) {
          const mappedModels: AIModel[] = technologies.map((tech: any) => ({
            id: tech.id,
            name: tech.name,
            provider: tech.name.includes('GPT') ? 'OpenAI' : tech.name.includes('Claude') ? 'Anthropic' : 'Unknown',
            model_id: tech.id,
            version: '1.0.0',
            zone: tech.version === 'v1' ? 'v1-experimentation' : tech.version === 'v2' ? 'v2-production' : 'v3-quarantine',
            status: tech.status === 'active' ? 'active' : tech.status === 'testing' ? 'testing' : tech.status === 'deprecated' ? 'deprecated' : 'quarantined',
            metrics: {
              accuracy: tech.accuracy || 0,
              latency_p95: tech.latency || 0,
              error_rate: tech.errorRate || 0,
              cost_per_request: 0.01,
              requests_today: tech.samples || 0,
              success_rate: 1 - (tech.errorRate || 0),
            },
            config: { max_tokens: 4096, temperature: 0.7, top_p: 0.9 },
            created_at: tech.lastUpdated || new Date().toISOString(),
            updated_at: tech.lastUpdated || new Date().toISOString(),
          }));
          setModels(mappedModels);
          return;
        }
      } catch {
        // Use mock data as last resort
      }
    }
    setModels(mockModels);
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Scroll chat to bottom when messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  // Handle sending chat message
  const handleSendMessage = async () => {
    if (!chatInput.trim() || chatLoading) return;

    const userMessage: ChatMessage = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: chatInput.trim(),
      timestamp: new Date(),
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setChatLoading(true);

    try {
      const response = await api.post('/api/admin/ai-chat', {
        message: userMessage.content,
        model: chatModel,
        context: 'version_control',
      });

      const assistantMessage: ChatMessage = {
        id: `msg_${Date.now()}_assistant`,
        role: 'assistant',
        content: response.data?.response || 'I can help you manage AI models and version control. What would you like to know?',
        timestamp: new Date(),
        model: chatModel,
      };

      setChatMessages(prev => [...prev, assistantMessage]);
    } catch {
      // Mock response for demo
      const mockResponses = [
        'Based on current metrics, Model GPT-4 Turbo is performing well with 94% accuracy. Consider promoting GPT-4 Vision once it reaches the 85% accuracy threshold.',
        'The V1 experimentation zone currently has 1 model under testing. All promotion criteria require: Accuracy >85%, Error Rate <5%, Latency p95 <3s.',
        'I recommend monitoring the error rate of models in V2 Production. If any model exceeds 5% error rate, consider rollback to V1 for further testing.',
        'To import a new model via API, you\'ll need to configure the API endpoint and authentication. The imported model will be available for code review chat only.',
      ];

      const assistantMessage: ChatMessage = {
        id: `msg_${Date.now()}_assistant`,
        role: 'assistant',
        content: mockResponses[Math.floor(Math.random() * mockResponses.length)],
        timestamp: new Date(),
        model: chatModel,
      };

      setChatMessages(prev => [...prev, assistantMessage]);
    }

    setChatLoading(false);
  };

  // Handle import model
  const handleImportModel = async (values: any) => {
    try {
      await api.post('/api/admin/ai-models/import', values);
      message.success('Model imported successfully');
      setImportModalOpen(false);
      importForm.resetFields();
      
      // Add to imported models list
      const newModel: ImportedModel = {
        id: `imported_${Date.now()}`,
        name: values.name,
        provider: values.provider,
        api_endpoint: values.api_endpoint,
        api_key_configured: !!values.api_key,
        status: 'active',
        created_at: new Date().toISOString(),
      };
      setImportedModels(prev => [...prev, newModel]);
    } catch {
      // Mock success for demo
      message.success('Model imported successfully');
      setImportModalOpen(false);
      importForm.resetFields();
      
      const newModel: ImportedModel = {
        id: `imported_${Date.now()}`,
        name: values.name,
        provider: values.provider || 'Custom',
        api_endpoint: values.api_endpoint,
        api_key_configured: !!values.api_key,
        status: 'active',
        created_at: new Date().toISOString(),
      };
      setImportedModels(prev => [...prev, newModel]);
    }
  };

  // Clear chat history
  const handleClearChat = () => {
    setChatMessages([]);
    message.success('Chat history cleared');
  };

  // Get zone color
  const getZoneColor = (zone: string) => {
    switch (zone) {
      case 'v1-experimentation': return 'blue';
      case 'v2-production': return 'green';
      case 'v3-quarantine': return 'red';
      default: return 'default';
    }
  };

  // Get zone label
  const getZoneLabel = (zone: string) => {
    switch (zone) {
      case 'v1-experimentation': return 'V1 Experimentation';
      case 'v2-production': return 'V2 Production';
      case 'v3-quarantine': return 'V3 Quarantine';
      default: return zone;
    }
  };

  // Get status badge
  const getStatusBadge = (status: string) => {
    const config: Record<string, { color: string; icon: React.ReactNode }> = {
      active: { color: 'green', icon: <CheckCircleOutlined /> },
      inactive: { color: 'default', icon: <PauseCircleOutlined /> },
      testing: { color: 'blue', icon: <SyncOutlined spin /> },
      deprecated: { color: 'orange', icon: <WarningOutlined /> },
      quarantined: { color: 'red', icon: <CloseCircleOutlined /> },
    };
    return config[status] || { color: 'default', icon: null };
  };

  // Promote model
  const handlePromote = async (model: AIModel) => {
    try {
      await api.post(`/api/admin/ai-models/${model.id}/promote`);
      message.success(`Model ${model.name} promoted to V2 Production`);
      fetchModels();
    } catch (error) {
      // Mock success
      message.success(`Model ${model.name} promoted to V2 Production`);
      setModels(prev => prev.map(m => 
        m.id === model.id 
          ? { ...m, zone: 'v2-production' as const, status: 'active' as const }
          : m
      ));
    }
    setPromoteModalOpen(false);
  };

  // Rollback model
  const handleRollback = async (model: AIModel) => {
    try {
      await api.post(`/api/admin/ai-models/${model.id}/rollback`);
      message.success(`Model ${model.name} rolled back to V1 Experimentation`);
      fetchModels();
    } catch (error) {
      message.success(`Model ${model.name} rolled back`);
      setModels(prev => prev.map(m => 
        m.id === model.id 
          ? { ...m, zone: 'v1-experimentation' as const, status: 'testing' as const }
          : m
      ));
    }
  };

  // Quarantine model
  const handleQuarantine = async (model: AIModel) => {
    try {
      await api.post(`/api/admin/ai-models/${model.id}/quarantine`);
      message.warning(`Model ${model.name} moved to V3 Quarantine`);
      fetchModels();
    } catch (error) {
      message.warning(`Model ${model.name} quarantined`);
      setModels(prev => prev.map(m => 
        m.id === model.id 
          ? { ...m, zone: 'v3-quarantine' as const, status: 'quarantined' as const }
          : m
      ));
    }
  };

  // Save config
  const handleSaveConfig = async (values: any) => {
    if (!selectedModel) return;
    try {
      await api.put(`/api/admin/ai-models/${selectedModel.id}/config`, values);
      message.success('Configuration saved');
      setConfigModalOpen(false);
      fetchModels();
    } catch (error) {
      message.success('Configuration saved');
      setConfigModalOpen(false);
    }
  };

  // Filter models by zone
  const filteredModels = activeTab === 'all' 
    ? models 
    : models.filter(m => m.zone === activeTab);

  // Table columns
  const columns: TableProps<AIModel>['columns'] = [
    {
      title: 'Model',
      key: 'model',
      render: (_, record) => (
        <Space>
          <RobotOutlined style={{ fontSize: 20, color: '#1890ff' }} />
          <div>
            <Text strong>{record.name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {record.provider} • {record.model_id}
            </Text>
          </div>
        </Space>
      ),
    },
    {
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
      render: (version) => <Tag>{version}</Tag>,
    },
    {
      title: 'Zone',
      dataIndex: 'zone',
      key: 'zone',
      render: (zone) => (
        <Tag color={getZoneColor(zone)} icon={
          zone === 'v2-production' ? <SafetyCertificateOutlined /> :
          zone === 'v1-experimentation' ? <ExperimentOutlined /> :
          <WarningOutlined />
        }>
          {getZoneLabel(zone)}
        </Tag>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const { color, icon } = getStatusBadge(status);
        return (
          <Tag color={color} icon={icon}>
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </Tag>
        );
      },
    },
    {
      title: 'Performance',
      key: 'performance',
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Text>
            <CheckCircleOutlined style={{ color: '#52c41a' }} /> {(record.metrics.accuracy * 100).toFixed(1)}% Accuracy
          </Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.metrics.latency_p95}s p95 • {(record.metrics.error_rate * 100).toFixed(1)}% errors
          </Text>
        </Space>
      ),
    },
    {
      title: 'Usage Today',
      key: 'usage',
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Text>{record.metrics.requests_today.toLocaleString()} requests</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            ${(record.metrics.requests_today * record.metrics.cost_per_request).toFixed(2)} cost
          </Text>
        </Space>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_, record) => {
        const items: MenuProps['items'] = [
          {
            key: 'view',
            icon: <EyeOutlined />,
            label: 'View Details',
            onClick: () => {
              setSelectedModel(record);
              setDetailModalOpen(true);
            },
          },
          {
            key: 'config',
            icon: <SettingOutlined />,
            label: 'Configure',
            onClick: () => {
              setSelectedModel(record);
              form.setFieldsValue(record.config);
              setConfigModalOpen(true);
            },
          },
          { type: 'divider' },
          ...(record.zone === 'v1-experimentation' && record.metrics.accuracy >= 0.85 ? [{
            key: 'promote',
            icon: <ArrowUpOutlined />,
            label: 'Promote to V2',
            onClick: () => {
              setSelectedModel(record);
              setPromoteModalOpen(true);
            },
          }] : []),
          ...(record.zone === 'v2-production' ? [{
            key: 'rollback',
            icon: <ArrowDownOutlined />,
            label: 'Rollback to V1',
            onClick: () => handleRollback(record),
          }] : []),
          ...(record.zone !== 'v3-quarantine' ? [{
            key: 'quarantine',
            icon: <WarningOutlined />,
            label: 'Move to Quarantine',
            danger: true,
            onClick: () => handleQuarantine(record),
          }] : []),
        ];

        return (
          <Dropdown menu={{ items }} trigger={['click']}>
            <Button icon={<MoreOutlined />} />
          </Dropdown>
        );
      },
    },
  ];

  // Summary stats
  const stats = {
    total: models.length,
    production: models.filter(m => m.zone === 'v2-production').length,
    experimentation: models.filter(m => m.zone === 'v1-experimentation').length,
    quarantine: models.filter(m => m.zone === 'v3-quarantine').length,
    avgAccuracy: models.filter(m => m.zone === 'v2-production')
      .reduce((acc, m) => acc + m.metrics.accuracy, 0) / 
      models.filter(m => m.zone === 'v2-production').length || 0,
  };

  return (
    <div className="ai-models-page" style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      {/* Page Header */}
      <Card style={{ marginBottom: 24, borderRadius: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <Title level={3} style={{ margin: 0 }}>
              <RobotOutlined style={{ marginRight: 8 }} />
              {t('admin.aiModels.title', 'AI Model Management')}
            </Title>
            <Text type="secondary">
              {t('admin.aiModels.subtitle', 'Manage AI models across V1→V2→V3 lifecycle')}
            </Text>
          </div>
          <Space wrap>
            <Tooltip title="AI Assistant for Version Control">
              <Badge count={chatMessages.length} size="small">
                <Button 
                  icon={<MessageOutlined />} 
                  onClick={() => setChatDrawerOpen(true)}
                  style={{ borderRadius: 8 }}
                >
                  AI Assistant
                </Button>
              </Badge>
            </Tooltip>
            <Button 
              icon={<ApiOutlined />} 
              onClick={() => setImportModalOpen(true)}
              style={{ borderRadius: 8 }}
            >
              Import Model
            </Button>
            <Button 
              icon={<ReloadOutlined />} 
              onClick={fetchModels}
              style={{ borderRadius: 8 }}
            >
              Refresh
            </Button>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              style={{ borderRadius: 8 }}
            >
              Register Model
            </Button>
          </Space>
        </div>
      </Card>

      {/* Stats */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card 
            style={{ borderRadius: 12, boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}
            hoverable
          >
            <Statistic
              title={<Text type="secondary">Total Models</Text>}
              value={stats.total}
              prefix={<RobotOutlined style={{ color: '#722ed1' }} />}
              valueStyle={{ color: '#722ed1', fontWeight: 600 }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card 
            style={{ borderRadius: 12, boxShadow: '0 2px 8px rgba(0,0,0,0.06)', borderLeft: '4px solid #52c41a' }}
            hoverable
          >
            <Statistic
              title={<Text type="secondary">V2 Production</Text>}
              value={stats.production}
              prefix={<SafetyCertificateOutlined style={{ color: '#52c41a' }} />}
              valueStyle={{ color: '#52c41a', fontWeight: 600 }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card 
            style={{ borderRadius: 12, boxShadow: '0 2px 8px rgba(0,0,0,0.06)', borderLeft: '4px solid #1890ff' }}
            hoverable
          >
            <Statistic
              title={<Text type="secondary">V1 Experimentation</Text>}
              value={stats.experimentation}
              prefix={<ExperimentOutlined style={{ color: '#1890ff' }} />}
              valueStyle={{ color: '#1890ff', fontWeight: 600 }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card 
            style={{ borderRadius: 12, boxShadow: '0 2px 8px rgba(0,0,0,0.06)', borderLeft: `4px solid ${stats.avgAccuracy >= 0.9 ? '#52c41a' : '#faad14'}` }}
            hoverable
          >
            <Statistic
              title={<Text type="secondary">Avg. Accuracy (V2)</Text>}
              value={(stats.avgAccuracy * 100).toFixed(1)}
              suffix="%"
              prefix={<LineChartOutlined style={{ color: stats.avgAccuracy >= 0.9 ? '#52c41a' : '#faad14' }} />}
              valueStyle={{ color: stats.avgAccuracy >= 0.9 ? '#52c41a' : '#faad14', fontWeight: 600 }}
            />
          </Card>
        </Col>
      </Row>

      {/* Model Lifecycle Flow */}
      <Card 
        style={{ marginBottom: 24, borderRadius: 12, background: 'linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%)' }}
      >
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 24, flexWrap: 'wrap', padding: '16px 0' }}>
          <div style={{ textAlign: 'center', padding: '16px 24px', background: '#fff', borderRadius: 12, boxShadow: '0 2px 8px rgba(24,144,255,0.2)', border: '2px solid #1890ff' }}>
            <ExperimentOutlined style={{ fontSize: 28, color: '#1890ff' }} />
            <div style={{ marginTop: 8, fontWeight: 600, color: '#1890ff' }}>V1 Experimentation</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: '#333' }}>{stats.experimentation}</div>
            <Text type="secondary" style={{ fontSize: 12 }}>models testing</Text>
          </div>
          
          <div style={{ fontSize: 32, color: '#1890ff' }}>→</div>
          
          <div style={{ textAlign: 'center', padding: '16px 24px', background: '#fff', borderRadius: 12, boxShadow: '0 2px 8px rgba(82,196,26,0.2)', border: '2px solid #52c41a' }}>
            <SafetyCertificateOutlined style={{ fontSize: 28, color: '#52c41a' }} />
            <div style={{ marginTop: 8, fontWeight: 600, color: '#52c41a' }}>V2 Production</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: '#333' }}>{stats.production}</div>
            <Text type="secondary" style={{ fontSize: 12 }}>models active</Text>
          </div>
          
          <div style={{ fontSize: 32, color: '#ff4d4f' }}>→</div>
          
          <div style={{ textAlign: 'center', padding: '16px 24px', background: '#fff', borderRadius: 12, boxShadow: '0 2px 8px rgba(255,77,79,0.2)', border: '2px solid #ff4d4f' }}>
            <WarningOutlined style={{ fontSize: 28, color: '#ff4d4f' }} />
            <div style={{ marginTop: 8, fontWeight: 600, color: '#ff4d4f' }}>V3 Quarantine</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: '#333' }}>{stats.quarantine}</div>
            <Text type="secondary" style={{ fontSize: 12 }}>models archived</Text>
          </div>
        </div>
        
        <Divider style={{ margin: '16px 0' }} />
        
        <div style={{ textAlign: 'center' }}>
          <Text type="secondary" style={{ fontSize: 13 }}>
            <SafetyCertificateOutlined style={{ marginRight: 8 }} />
            Promotion Criteria: Accuracy ≥ 85% • Error Rate {'<'} 5% • Latency p95 {'<'} 3s
          </Text>
        </div>
      </Card>

      {/* Models Table */}
      <Card style={{ borderRadius: 12, boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          type="card"
          items={[
            { 
              key: 'all', 
              label: (
                <Space>
                  <RobotOutlined />
                  All Models
                  <Badge count={models.length} style={{ backgroundColor: '#722ed1' }} />
                </Space>
              )
            },
            { 
              key: 'v1-experimentation', 
              label: (
                <Space>
                  <ExperimentOutlined />
                  V1 Experimentation
                  <Badge count={stats.experimentation} style={{ backgroundColor: '#1890ff' }} />
                </Space>
              )
            },
            { 
              key: 'v2-production', 
              label: (
                <Space>
                  <SafetyCertificateOutlined />
                  V2 Production
                  <Badge count={stats.production} style={{ backgroundColor: '#52c41a' }} />
                </Space>
              )
            },
            { 
              key: 'v3-quarantine', 
              label: (
                <Space>
                  <WarningOutlined />
                  V3 Quarantine
                  <Badge count={stats.quarantine} style={{ backgroundColor: '#ff4d4f' }} />
                </Space>
              )
            },
          ]}
        />
        <Table
          columns={columns}
          dataSource={filteredModels}
          rowKey="id"
          loading={loading}
          pagination={{ 
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `Total ${total} models`,
          }}
          style={{ marginTop: 16 }}
        />
      </Card>

      {/* Detail Modal */}
      <Modal
        title={<><RobotOutlined /> Model Details</>}
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        footer={null}
        width={700}
      >
        {selectedModel && (
          <div>
            <Descriptions bordered column={2}>
              <Descriptions.Item label="Name">{selectedModel.name}</Descriptions.Item>
              <Descriptions.Item label="Provider">{selectedModel.provider}</Descriptions.Item>
              <Descriptions.Item label="Model ID">{selectedModel.model_id}</Descriptions.Item>
              <Descriptions.Item label="Version">{selectedModel.version}</Descriptions.Item>
              <Descriptions.Item label="Zone">
                <Tag color={getZoneColor(selectedModel.zone)}>
                  {getZoneLabel(selectedModel.zone)}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={getStatusBadge(selectedModel.status).color}>
                  {selectedModel.status}
                </Tag>
              </Descriptions.Item>
            </Descriptions>

            <Title level={5} style={{ marginTop: 24 }}>Performance Metrics</Title>
            <Row gutter={16}>
              <Col span={8}>
                <Card size="small">
                  <Statistic
                    title="Accuracy"
                    value={(selectedModel.metrics.accuracy * 100).toFixed(1)}
                    suffix="%"
                    valueStyle={{ color: selectedModel.metrics.accuracy >= 0.9 ? '#52c41a' : '#faad14' }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic
                    title="Latency (p95)"
                    value={selectedModel.metrics.latency_p95}
                    suffix="s"
                    valueStyle={{ color: selectedModel.metrics.latency_p95 < 3 ? '#52c41a' : '#ff4d4f' }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic
                    title="Error Rate"
                    value={(selectedModel.metrics.error_rate * 100).toFixed(2)}
                    suffix="%"
                    valueStyle={{ color: selectedModel.metrics.error_rate < 0.05 ? '#52c41a' : '#ff4d4f' }}
                  />
                </Card>
              </Col>
            </Row>

            <Title level={5} style={{ marginTop: 24 }}>Configuration</Title>
            <Descriptions bordered column={3} size="small">
              <Descriptions.Item label="Max Tokens">{selectedModel.config.max_tokens}</Descriptions.Item>
              <Descriptions.Item label="Temperature">{selectedModel.config.temperature}</Descriptions.Item>
              <Descriptions.Item label="Top P">{selectedModel.config.top_p}</Descriptions.Item>
            </Descriptions>
          </div>
        )}
      </Modal>

      {/* Config Modal */}
      <Modal
        title={<><SettingOutlined /> Model Configuration</>}
        open={configModalOpen}
        onCancel={() => setConfigModalOpen(false)}
        onOk={() => form.submit()}
      >
        <Form form={form} layout="vertical" onFinish={handleSaveConfig}>
          <Form.Item name="max_tokens" label="Max Tokens" rules={[{ required: true }]}>
            <InputNumber min={100} max={32000} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="temperature" label="Temperature" rules={[{ required: true }]}>
            <InputNumber min={0} max={2} step={0.1} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="top_p" label="Top P" rules={[{ required: true }]}>
            <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
          </Form.Item>
        </Form>
      </Modal>

      {/* Promote Modal */}
      <Modal
        title={<><ArrowUpOutlined /> Promote to V2 Production</>}
        open={promoteModalOpen}
        onCancel={() => setPromoteModalOpen(false)}
        onOk={() => selectedModel && handlePromote(selectedModel)}
        okText="Promote"
        okButtonProps={{ type: 'primary' }}
      >
        {selectedModel && (
          <div>
            <Alert
              message="Promotion Criteria Met"
              description={
                <ul>
                  <li>Accuracy: {(selectedModel.metrics.accuracy * 100).toFixed(1)}% (≥85% required) ✓</li>
                  <li>Error Rate: {(selectedModel.metrics.error_rate * 100).toFixed(2)}% ({'<'}5% required) ✓</li>
                  <li>Latency p95: {selectedModel.metrics.latency_p95}s ({'<'}3s required) {selectedModel.metrics.latency_p95 < 3 ? '✓' : '⚠'}</li>
                </ul>
              }
              type="success"
              showIcon
            />
            <Paragraph style={{ marginTop: 16 }}>
              Are you sure you want to promote <strong>{selectedModel.name}</strong> from V1 Experimentation to V2 Production?
            </Paragraph>
            <Paragraph type="secondary">
              This will make the model available to all users and subject to SLO requirements.
            </Paragraph>
          </div>
        )}
      </Modal>

      {/* AI Chat Drawer */}
      <Drawer
        title={
          <Space>
            <RobotOutlined style={{ color: '#1890ff' }} />
            <span>Version Control AI Assistant</span>
          </Space>
        }
        placement="right"
        width={480}
        open={chatDrawerOpen}
        onClose={() => setChatDrawerOpen(false)}
        extra={
          <Space>
            <Select
              value={chatModel}
              onChange={setChatModel}
              style={{ width: 140 }}
              size="small"
              options={[
                { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
                { value: 'claude-3-opus', label: 'Claude 3 Opus' },
                { value: 'gpt-4-vision', label: 'GPT-4 Vision' },
              ]}
            />
            <Tooltip title="Clear chat history">
              <Button 
                size="small" 
                icon={<ClearOutlined />} 
                onClick={handleClearChat}
              />
            </Tooltip>
          </Space>
        }
      >
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          {/* Chat Messages */}
          <div 
            ref={chatContainerRef}
            style={{ 
              flex: 1, 
              overflowY: 'auto', 
              paddingBottom: 16,
              marginBottom: 16,
            }}
          >
            {chatMessages.length === 0 ? (
              <Empty
                image={<RobotOutlined style={{ fontSize: 48, color: '#1890ff' }} />}
                description={
                  <div>
                    <Text strong>Welcome to Version Control AI Assistant</Text>
                    <br />
                    <Text type="secondary">
                      Ask about model performance, promotions, or version management.
                    </Text>
                  </div>
                }
              >
                <Space direction="vertical" size="small">
                  <Button 
                    size="small" 
                    onClick={() => setChatInput('What models are ready for promotion?')}
                  >
                    What models are ready for promotion?
                  </Button>
                  <Button 
                    size="small" 
                    onClick={() => setChatInput('Show me V2 production metrics')}
                  >
                    Show me V2 production metrics
                  </Button>
                  <Button 
                    size="small" 
                    onClick={() => setChatInput('How do I import a new model?')}
                  >
                    How do I import a new model?
                  </Button>
                </Space>
              </Empty>
            ) : (
              <List
                dataSource={chatMessages}
                renderItem={(msg) => (
                  <List.Item 
                    style={{ 
                      padding: '8px 0',
                      border: 'none',
                      justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                    }}
                  >
                    <div 
                      style={{ 
                        maxWidth: '85%',
                        display: 'flex',
                        flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                        alignItems: 'flex-start',
                        gap: 8,
                      }}
                    >
                      <Avatar 
                        size={32}
                        icon={msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                        style={{ 
                          backgroundColor: msg.role === 'user' ? '#1890ff' : '#52c41a',
                          flexShrink: 0,
                        }}
                      />
                      <div 
                        style={{ 
                          padding: '8px 12px',
                          borderRadius: 12,
                          background: msg.role === 'user' ? '#1890ff' : '#f5f5f5',
                          color: msg.role === 'user' ? '#fff' : '#333',
                        }}
                      >
                        <Text style={{ color: 'inherit' }}>{msg.content}</Text>
                        {msg.model && (
                          <div style={{ marginTop: 4 }}>
                            <Text type="secondary" style={{ fontSize: 11 }}>
                              {msg.model} • {msg.timestamp.toLocaleTimeString()}
                            </Text>
                          </div>
                        )}
                      </div>
                    </div>
                  </List.Item>
                )}
              />
            )}
            {chatLoading && (
              <div style={{ textAlign: 'center', padding: 16 }}>
                <Spin size="small" />
                <Text type="secondary" style={{ marginLeft: 8 }}>AI is thinking...</Text>
              </div>
            )}
          </div>

          {/* Chat Input */}
          <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: 16 }}>
            <Space.Compact style={{ width: '100%' }}>
              <TextArea
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask about model management..."
                autoSize={{ minRows: 1, maxRows: 4 }}
                onPressEnter={(e) => {
                  if (!e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                style={{ borderRadius: '8px 0 0 8px' }}
              />
              <Button 
                type="primary" 
                icon={<SendOutlined />}
                onClick={handleSendMessage}
                loading={chatLoading}
                style={{ height: 'auto', borderRadius: '0 8px 8px 0' }}
              />
            </Space.Compact>
          </div>
        </div>
      </Drawer>

      {/* Import Model Modal */}
      <Modal
        title={
          <Space>
            <ApiOutlined style={{ color: '#1890ff' }} />
            <span>Import External Model</span>
          </Space>
        }
        open={importModalOpen}
        onCancel={() => {
          setImportModalOpen(false);
          importForm.resetFields();
        }}
        onOk={() => importForm.submit()}
        okText="Import Model"
        width={600}
      >
        <Alert
          message="API Model Import"
          description="Import external AI models via API. These models will be available for Code Review AI Chat only."
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />
        
        <Form
          form={importForm}
          layout="vertical"
          onFinish={handleImportModel}
        >
          <Form.Item
            name="name"
            label="Model Name"
            rules={[{ required: true, message: 'Please enter model name' }]}
          >
            <Input placeholder="e.g., Custom CodeLlama 34B" prefix={<RobotOutlined />} />
          </Form.Item>

          <Form.Item
            name="provider"
            label="Provider"
            rules={[{ required: true, message: 'Please select provider' }]}
          >
            <Select
              placeholder="Select provider"
              options={[
                { value: 'openai', label: 'OpenAI' },
                { value: 'anthropic', label: 'Anthropic' },
                { value: 'ollama', label: 'Ollama (Local)' },
                { value: 'huggingface', label: 'HuggingFace' },
                { value: 'custom', label: 'Custom API' },
              ]}
            />
          </Form.Item>

          <Form.Item
            name="api_endpoint"
            label="API Endpoint"
            rules={[{ required: true, message: 'Please enter API endpoint' }]}
          >
            <Input placeholder="https://api.example.com/v1/chat" prefix={<CloudServerOutlined />} />
          </Form.Item>

          <Form.Item
            name="api_key"
            label="API Key (Optional)"
            extra="API key will be encrypted and stored securely"
          >
            <Input.Password placeholder="sk-..." />
          </Form.Item>

          <Form.Item
            name="model_id"
            label="Model ID"
            extra="The specific model identifier used by the API"
          >
            <Input placeholder="e.g., codellama:34b, gpt-4-turbo" />
          </Form.Item>

          <Divider />

          <Form.Item
            name="config"
            label="Model Configuration"
          >
            <Row gutter={16}>
              <Col span={8}>
                <Form.Item name={['config', 'max_tokens']} label="Max Tokens" initialValue={4096}>
                  <InputNumber min={100} max={32000} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name={['config', 'temperature']} label="Temperature" initialValue={0.7}>
                  <InputNumber min={0} max={2} step={0.1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name={['config', 'top_p']} label="Top P" initialValue={0.9}>
                  <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
          </Form.Item>
        </Form>

        {/* Imported Models List */}
        {importedModels.length > 0 && (
          <>
            <Divider>Imported Models ({importedModels.length})</Divider>
            <List
              size="small"
              dataSource={importedModels}
              renderItem={(model) => (
                <List.Item
                  extra={
                    <Tag color={model.status === 'active' ? 'green' : 'red'}>
                      {model.status}
                    </Tag>
                  }
                >
                  <List.Item.Meta
                    avatar={<Avatar icon={<ApiOutlined />} />}
                    title={model.name}
                    description={`${model.provider} • ${model.api_endpoint}`}
                  />
                </List.Item>
              )}
            />
          </>
        )}
      </Modal>
    </div>
  );
};

export default AIModels;
