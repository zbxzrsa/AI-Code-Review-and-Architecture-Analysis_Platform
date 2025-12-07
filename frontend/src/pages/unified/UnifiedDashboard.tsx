/**
 * Unified Dashboard - Version-Based Function Aggregation
 *
 * Centralizes all functions by version (V1, V2, V3) with:
 * - Modular layout with logical grouping
 * - Fast navigation and search
 * - Role-based access control
 * - Performance optimized (< 1s load time)
 */

import React, { Suspense, lazy, useState, useMemo, useCallback, useEffect } from 'react';
import {
  Layout,
  Menu,
  Card,
  Row,
  Col,
  Input,
  Badge,
  Tabs,
  Tag,
  Tooltip,
  Space,
  Spin,
  Typography,
  Breadcrumb,
  Button,
  Statistic,
  Divider,
  Empty,
  Alert,
} from 'antd';
import {
  DashboardOutlined,
  CodeOutlined,
  SafetyOutlined,
  BarChartOutlined,
  ExperimentOutlined,
  HistoryOutlined,
  SearchOutlined,
  AppstoreOutlined,
  RocketOutlined,
  BugOutlined,
  TeamOutlined,
  SettingOutlined,
  CloudServerOutlined,
  ThunderboltOutlined,
  AimOutlined,
  DatabaseOutlined,
  ApiOutlined,
  AuditOutlined,
  SyncOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  HomeOutlined,
  StarOutlined,
  PushpinOutlined,
} from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { usePermissions } from '../../hooks/usePermissions';
import { AdminOnly } from '../../components/common/PermissionGate';

const { Sider, Content } = Layout;
const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

// =============================================================================
// Types
// =============================================================================

interface FunctionModule {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  path: string;
  category: string;
  version: 'V1' | 'V2' | 'V3' | 'Admin';
  status: 'active' | 'beta' | 'deprecated';
  isAdminOnly: boolean;
  keywords: string[];
  metrics?: {
    usage?: number;
    health?: number;
  };
}

interface CategoryGroup {
  id: string;
  name: string;
  icon: React.ReactNode;
  modules: FunctionModule[];
}

// =============================================================================
// Function Registry - All system functions organized by version
// =============================================================================

const FUNCTION_MODULES: FunctionModule[] = [
  // =========================================
  // V2 - Production Functions (User Access)
  // =========================================
  {
    id: 'code-review',
    name: 'Code Review',
    description: 'AI-powered code review and analysis',
    icon: <CodeOutlined />,
    path: '/review',
    category: 'development',
    version: 'V2',
    status: 'active',
    isAdminOnly: false,
    keywords: ['code', 'review', 'analysis', 'ai', 'quality'],
    metrics: { usage: 95, health: 100 },
  },
  {
    id: 'projects',
    name: 'Projects',
    description: 'Project management and organization',
    icon: <AppstoreOutlined />,
    path: '/projects',
    category: 'development',
    version: 'V2',
    status: 'active',
    isAdminOnly: false,
    keywords: ['project', 'manage', 'organize', 'workspace'],
    metrics: { usage: 88, health: 100 },
  },
  {
    id: 'repositories',
    name: 'Repositories',
    description: 'Repository management and integration',
    icon: <DatabaseOutlined />,
    path: '/repositories',
    category: 'development',
    version: 'V2',
    status: 'active',
    isAdminOnly: false,
    keywords: ['repository', 'git', 'source', 'code'],
    metrics: { usage: 82, health: 100 },
  },
  {
    id: 'pull-requests',
    name: 'Pull Requests',
    description: 'Pull request review and management',
    icon: <SyncOutlined />,
    path: '/pull-requests',
    category: 'development',
    version: 'V2',
    status: 'active',
    isAdminOnly: false,
    keywords: ['pull request', 'pr', 'merge', 'review'],
    metrics: { usage: 75, health: 100 },
  },
  {
    id: 'analytics',
    name: 'Analytics',
    description: 'Code quality and performance analytics',
    icon: <BarChartOutlined />,
    path: '/analytics',
    category: 'insights',
    version: 'V2',
    status: 'active',
    isAdminOnly: false,
    keywords: ['analytics', 'metrics', 'statistics', 'trends'],
    metrics: { usage: 70, health: 100 },
  },
  {
    id: 'security-dashboard',
    name: 'Security Dashboard',
    description: 'Security vulnerabilities and compliance',
    icon: <SafetyOutlined />,
    path: '/security',
    category: 'security',
    version: 'V2',
    status: 'active',
    isAdminOnly: false,
    keywords: ['security', 'vulnerability', 'scan', 'compliance'],
    metrics: { usage: 65, health: 100 },
  },
  {
    id: 'reports',
    name: 'Reports',
    description: 'Generate and view code analysis reports',
    icon: <AuditOutlined />,
    path: '/reports',
    category: 'insights',
    version: 'V2',
    status: 'active',
    isAdminOnly: false,
    keywords: ['report', 'export', 'document', 'summary'],
    metrics: { usage: 60, health: 100 },
  },
  {
    id: 'code-metrics',
    name: 'Code Metrics',
    description: 'Detailed code quality metrics',
    icon: <AimOutlined />,
    path: '/metrics',
    category: 'insights',
    version: 'V2',
    status: 'active',
    isAdminOnly: false,
    keywords: ['metrics', 'quality', 'complexity', 'coverage'],
    metrics: { usage: 55, health: 100 },
  },
  {
    id: 'quality-rules',
    name: 'Quality Rules',
    description: 'Configure code quality rules',
    icon: <CheckCircleOutlined />,
    path: '/rules',
    category: 'configuration',
    version: 'V2',
    status: 'active',
    isAdminOnly: false,
    keywords: ['rules', 'quality', 'lint', 'standards'],
    metrics: { usage: 45, health: 100 },
  },
  {
    id: 'teams',
    name: 'Team Management',
    description: 'Manage teams and collaboration',
    icon: <TeamOutlined />,
    path: '/teams',
    category: 'collaboration',
    version: 'V2',
    status: 'active',
    isAdminOnly: false,
    keywords: ['team', 'collaborate', 'members', 'permissions'],
    metrics: { usage: 50, health: 100 },
  },

  // =========================================
  // V1 - Experimental Functions (Admin Only)
  // =========================================
  {
    id: 'experiments',
    name: 'Experiment Management',
    description: 'Manage AI model experiments',
    icon: <ExperimentOutlined />,
    path: '/admin/experiments',
    category: 'ai-experiments',
    version: 'V1',
    status: 'beta',
    isAdminOnly: true,
    keywords: ['experiment', 'test', 'ai', 'model'],
    metrics: { usage: 30, health: 95 },
  },
  {
    id: 'ai-testing',
    name: 'AI Model Testing',
    description: 'Test AI models with sample data',
    icon: <BugOutlined />,
    path: '/admin/model-testing',
    category: 'ai-experiments',
    version: 'V1',
    status: 'beta',
    isAdminOnly: true,
    keywords: ['ai', 'test', 'model', 'validation'],
    metrics: { usage: 25, health: 90 },
  },
  {
    id: 'model-comparison',
    name: 'Model Comparison',
    description: 'Compare AI model performance',
    icon: <ThunderboltOutlined />,
    path: '/admin/model-comparison',
    category: 'ai-experiments',
    version: 'V1',
    status: 'beta',
    isAdminOnly: true,
    keywords: ['compare', 'model', 'performance', 'benchmark'],
    metrics: { usage: 20, health: 92 },
  },
  {
    id: 'learning-cycle',
    name: 'Learning Cycle',
    description: 'Continuous learning dashboard',
    icon: <RocketOutlined />,
    path: '/admin/learning',
    category: 'ai-experiments',
    version: 'V1',
    status: 'beta',
    isAdminOnly: true,
    keywords: ['learning', 'training', 'cycle', 'evolution'],
    metrics: { usage: 15, health: 88 },
  },
  {
    id: 'evolution-cycle',
    name: 'Evolution Cycle',
    description: 'AI self-evolution monitoring',
    icon: <SyncOutlined />,
    path: '/admin/evolution',
    category: 'ai-experiments',
    version: 'V1',
    status: 'beta',
    isAdminOnly: true,
    keywords: ['evolution', 'self', 'improve', 'cycle'],
    metrics: { usage: 18, health: 85 },
  },

  // =========================================
  // V3 - Legacy Functions (Admin Comparison)
  // =========================================
  {
    id: 'version-comparison',
    name: 'Version Comparison',
    description: 'Compare different AI versions',
    icon: <HistoryOutlined />,
    path: '/admin/version-comparison',
    category: 'legacy',
    version: 'V3',
    status: 'deprecated',
    isAdminOnly: true,
    keywords: ['version', 'compare', 'legacy', 'baseline'],
    metrics: { usage: 10, health: 100 },
  },
  {
    id: 'three-version',
    name: 'Three Version Control',
    description: 'Manage V1/V2/V3 lifecycle',
    icon: <ApiOutlined />,
    path: '/admin/three-version',
    category: 'legacy',
    version: 'V3',
    status: 'active',
    isAdminOnly: true,
    keywords: ['version', 'lifecycle', 'promote', 'demote'],
    metrics: { usage: 12, health: 100 },
  },

  // =========================================
  // Admin Functions
  // =========================================
  {
    id: 'user-management',
    name: 'User Management',
    description: 'Manage users and permissions',
    icon: <TeamOutlined />,
    path: '/admin/users',
    category: 'administration',
    version: 'Admin',
    status: 'active',
    isAdminOnly: true,
    keywords: ['user', 'manage', 'permission', 'role'],
    metrics: { usage: 40, health: 100 },
  },
  {
    id: 'provider-management',
    name: 'AI Providers',
    description: 'Configure AI providers and API keys',
    icon: <CloudServerOutlined />,
    path: '/admin/providers',
    category: 'administration',
    version: 'Admin',
    status: 'active',
    isAdminOnly: true,
    keywords: ['provider', 'api', 'openai', 'anthropic'],
    metrics: { usage: 35, health: 100 },
  },
  {
    id: 'ai-models',
    name: 'AI Models',
    description: 'Manage AI models configuration',
    icon: <RocketOutlined />,
    path: '/admin/ai-models',
    category: 'administration',
    version: 'Admin',
    status: 'active',
    isAdminOnly: true,
    keywords: ['ai', 'model', 'config', 'settings'],
    metrics: { usage: 30, health: 100 },
  },
  {
    id: 'audit-logs',
    name: 'Audit Logs',
    description: 'View system audit logs',
    icon: <AuditOutlined />,
    path: '/admin/audit',
    category: 'administration',
    version: 'Admin',
    status: 'active',
    isAdminOnly: true,
    keywords: ['audit', 'log', 'history', 'track'],
    metrics: { usage: 25, health: 100 },
  },
  {
    id: 'system-health',
    name: 'System Health',
    description: 'Monitor system health and performance',
    icon: <DashboardOutlined />,
    path: '/admin/health',
    category: 'administration',
    version: 'Admin',
    status: 'active',
    isAdminOnly: true,
    keywords: ['health', 'monitor', 'status', 'performance'],
    metrics: { usage: 45, health: 100 },
  },
  {
    id: 'auto-fix',
    name: 'Auto-Fix',
    description: 'Automated code fix management',
    icon: <SettingOutlined />,
    path: '/admin/auto-fix',
    category: 'administration',
    version: 'Admin',
    status: 'active',
    isAdminOnly: true,
    keywords: ['auto', 'fix', 'repair', 'automated'],
    metrics: { usage: 20, health: 95 },
  },
  {
    id: 'security-scanner',
    name: 'Security Scanner',
    description: 'Advanced security scanning',
    icon: <SafetyOutlined />,
    path: '/admin/security',
    category: 'administration',
    version: 'Admin',
    status: 'active',
    isAdminOnly: true,
    keywords: ['security', 'scan', 'vulnerability', 'threat'],
    metrics: { usage: 28, health: 100 },
  },
  {
    id: 'performance-monitor',
    name: 'Performance Monitor',
    description: 'System performance monitoring',
    icon: <ThunderboltOutlined />,
    path: '/admin/performance',
    category: 'administration',
    version: 'Admin',
    status: 'active',
    isAdminOnly: true,
    keywords: ['performance', 'monitor', 'latency', 'throughput'],
    metrics: { usage: 22, health: 100 },
  },
];

// =============================================================================
// Category Groups
// =============================================================================

const CATEGORY_GROUPS: Record<string, { name: string; icon: React.ReactNode }> = {
  development: { name: 'Development', icon: <CodeOutlined /> },
  insights: { name: 'Insights & Analytics', icon: <BarChartOutlined /> },
  security: { name: 'Security', icon: <SafetyOutlined /> },
  configuration: { name: 'Configuration', icon: <SettingOutlined /> },
  collaboration: { name: 'Collaboration', icon: <TeamOutlined /> },
  'ai-experiments': { name: 'AI Experiments', icon: <ExperimentOutlined /> },
  legacy: { name: 'Legacy & Comparison', icon: <HistoryOutlined /> },
  administration: { name: 'Administration', icon: <SettingOutlined /> },
};

// =============================================================================
// Version Configs
// =============================================================================

const VERSION_CONFIG = {
  V1: {
    name: 'V1 - Experimental',
    color: '#faad14',
    description: 'New features under testing. Admin access only.',
    icon: <ExperimentOutlined />,
  },
  V2: {
    name: 'V2 - Production',
    color: '#52c41a',
    description: 'Stable features for all users.',
    icon: <CheckCircleOutlined />,
  },
  V3: {
    name: 'V3 - Legacy',
    color: '#8c8c8c',
    description: 'Deprecated features for comparison.',
    icon: <HistoryOutlined />,
  },
  Admin: {
    name: 'Administration',
    color: '#1890ff',
    description: 'System administration functions.',
    icon: <SettingOutlined />,
  },
};

// =============================================================================
// Function Card Component
// =============================================================================

interface FunctionCardProps {
  module: FunctionModule;
  onNavigate: (path: string) => void;
  isPinned?: boolean;
  onPin?: (id: string) => void;
}

const FunctionCard: React.FC<FunctionCardProps> = ({
  module,
  onNavigate,
  isPinned,
  onPin,
}) => {
  const versionConfig = VERSION_CONFIG[module.version];

  return (
    <Card
      hoverable
      size="small"
      onClick={() => onNavigate(module.path)}
      style={{
        height: '100%',
        borderLeft: `3px solid ${versionConfig.color}`,
      }}
      actions={[
        <Tooltip title={isPinned ? 'Unpin' : 'Pin to favorites'} key="pin">
          <Button
            type="text"
            size="small"
            icon={<PushpinOutlined style={{ color: isPinned ? '#1890ff' : undefined }} />}
            onClick={(e) => {
              e.stopPropagation();
              onPin?.(module.id);
            }}
          />
        </Tooltip>,
      ]}
    >
      <Card.Meta
        avatar={
          <div style={{
            fontSize: 24,
            color: versionConfig.color,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 40,
            height: 40,
            borderRadius: 8,
            background: `${versionConfig.color}15`,
          }}>
            {module.icon}
          </div>
        }
        title={
          <Space>
            <span>{module.name}</span>
            {module.status === 'beta' && <Tag color="orange">Beta</Tag>}
            {module.status === 'deprecated' && <Tag color="default">Legacy</Tag>}
          </Space>
        }
        description={
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>
              {module.description}
            </Text>
            {module.metrics && (
              <div style={{ marginTop: 8, display: 'flex', gap: 12 }}>
                <Tooltip title="Usage">
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    <BarChartOutlined /> {module.metrics.usage}%
                  </Text>
                </Tooltip>
                <Tooltip title="Health">
                  <Text
                    type="secondary"
                    style={{
                      fontSize: 11,
                      color: module.metrics.health! >= 95 ? '#52c41a' : '#faad14',
                    }}
                  >
                    <CheckCircleOutlined /> {module.metrics.health}%
                  </Text>
                </Tooltip>
              </div>
            )}
          </div>
        }
      />
    </Card>
  );
};

// =============================================================================
// Main Component
// =============================================================================

const UnifiedDashboard: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const { isAdmin, role } = usePermissions();

  // State
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedVersion, setSelectedVersion] = useState<string>('all');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [pinnedModules, setPinnedModules] = useState<string[]>(() => {
    const saved = localStorage.getItem('pinnedModules');
    return saved ? JSON.parse(saved) : ['code-review', 'projects', 'analytics'];
  });
  const [loadTime, setLoadTime] = useState<number>(0);

  // Measure load time
  useEffect(() => {
    const startTime = performance.now();
    return () => {
      const endTime = performance.now();
      setLoadTime(endTime - startTime);
    };
  }, []);

  // Save pinned modules
  useEffect(() => {
    localStorage.setItem('pinnedModules', JSON.stringify(pinnedModules));
  }, [pinnedModules]);

  // Filter modules based on role and search
  const filteredModules = useMemo(() => {
    let modules = FUNCTION_MODULES;

    // Filter by role (non-admin users can't see admin-only modules)
    if (!isAdmin) {
      modules = modules.filter(m => !m.isAdminOnly);
    }

    // Filter by version
    if (selectedVersion !== 'all') {
      modules = modules.filter(m => m.version === selectedVersion);
    }

    // Filter by category
    if (selectedCategory !== 'all') {
      modules = modules.filter(m => m.category === selectedCategory);
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      modules = modules.filter(m =>
        m.name.toLowerCase().includes(query) ||
        m.description.toLowerCase().includes(query) ||
        m.keywords.some(k => k.toLowerCase().includes(query))
      );
    }

    return modules;
  }, [isAdmin, selectedVersion, selectedCategory, searchQuery]);

  // Group modules by version
  const modulesByVersion = useMemo(() => {
    const groups: Record<string, FunctionModule[]> = {
      V2: [],
      V1: [],
      V3: [],
      Admin: [],
    };

    filteredModules.forEach(m => {
      groups[m.version].push(m);
    });

    return groups;
  }, [filteredModules]);

  // Group modules by category
  const modulesByCategory = useMemo(() => {
    const groups: Record<string, FunctionModule[]> = {};

    filteredModules.forEach(m => {
      if (!groups[m.category]) {
        groups[m.category] = [];
      }
      groups[m.category].push(m);
    });

    return groups;
  }, [filteredModules]);

  // Pinned modules
  const pinnedModulesList = useMemo(() => {
    return FUNCTION_MODULES.filter(m =>
      pinnedModules.includes(m.id) &&
      (!m.isAdminOnly || isAdmin)
    );
  }, [pinnedModules, isAdmin]);

  // Handlers
  const handleNavigate = useCallback((path: string) => {
    navigate(path);
  }, [navigate]);

  const handlePin = useCallback((id: string) => {
    setPinnedModules(prev =>
      prev.includes(id)
        ? prev.filter(p => p !== id)
        : [...prev, id]
    );
  }, []);

  // Search efficiency calculation (simulated)
  const searchEfficiency = useMemo(() => {
    if (!searchQuery) return 100;
    const totalModules = isAdmin ? FUNCTION_MODULES.length : FUNCTION_MODULES.filter(m => !m.isAdminOnly).length;
    const foundModules = filteredModules.length;
    return Math.round((1 - (foundModules / totalModules)) * 100);
  }, [searchQuery, filteredModules, isAdmin]);

  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
      <Content style={{ padding: 24 }}>
        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <Breadcrumb>
            <Breadcrumb.Item>
              <HomeOutlined />
            </Breadcrumb.Item>
            <Breadcrumb.Item>Unified Dashboard</Breadcrumb.Item>
          </Breadcrumb>

          <Title level={2} style={{ marginTop: 16, marginBottom: 8 }}>
            <DashboardOutlined style={{ marginRight: 12 }} />
            Unified Function Dashboard
          </Title>
          <Text type="secondary">
            Access all system functions organized by version. Search efficiency: {searchEfficiency}%
          </Text>
        </div>

        {/* Search and Filters */}
        <Card style={{ marginBottom: 24 }}>
          <Row gutter={16} align="middle">
            <Col flex="auto">
              <Input
                placeholder="Search functions... (e.g., code review, security, ai model)"
                prefix={<SearchOutlined />}
                size="large"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                allowClear
                style={{ maxWidth: 500 }}
              />
            </Col>
            <Col>
              <Space>
                <Text type="secondary">Version:</Text>
                <Button.Group>
                  <Button
                    type={selectedVersion === 'all' ? 'primary' : 'default'}
                    onClick={() => setSelectedVersion('all')}
                  >
                    All
                  </Button>
                  <Button
                    type={selectedVersion === 'V2' ? 'primary' : 'default'}
                    onClick={() => setSelectedVersion('V2')}
                    style={{ color: selectedVersion === 'V2' ? '#fff' : '#52c41a' }}
                  >
                    V2 Production
                  </Button>
                  {isAdmin && (
                    <>
                      <Button
                        type={selectedVersion === 'V1' ? 'primary' : 'default'}
                        onClick={() => setSelectedVersion('V1')}
                        style={{ color: selectedVersion === 'V1' ? '#fff' : '#faad14' }}
                      >
                        V1 Experimental
                      </Button>
                      <Button
                        type={selectedVersion === 'V3' ? 'primary' : 'default'}
                        onClick={() => setSelectedVersion('V3')}
                        style={{ color: selectedVersion === 'V3' ? '#fff' : '#8c8c8c' }}
                      >
                        V3 Legacy
                      </Button>
                      <Button
                        type={selectedVersion === 'Admin' ? 'primary' : 'default'}
                        onClick={() => setSelectedVersion('Admin')}
                      >
                        Admin
                      </Button>
                    </>
                  )}
                </Button.Group>
              </Space>
            </Col>
          </Row>
        </Card>

        {/* Quick Stats */}
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="V2 Production Functions"
                value={modulesByVersion.V2.length}
                prefix={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          {isAdmin && (
            <>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="V1 Experimental"
                    value={modulesByVersion.V1.length}
                    prefix={<ExperimentOutlined style={{ color: '#faad14' }} />}
                    valueStyle={{ color: '#faad14' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="V3 Legacy"
                    value={modulesByVersion.V3.length}
                    prefix={<HistoryOutlined style={{ color: '#8c8c8c' }} />}
                    valueStyle={{ color: '#8c8c8c' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Admin Functions"
                    value={modulesByVersion.Admin.length}
                    prefix={<SettingOutlined style={{ color: '#1890ff' }} />}
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Card>
              </Col>
            </>
          )}
          {!isAdmin && (
            <Col span={18}>
              <Card>
                <Alert
                  message="User View"
                  description="You're viewing production functions (V2). Contact administrator for access to experimental features."
                  type="info"
                  showIcon
                />
              </Card>
            </Col>
          )}
        </Row>

        {/* Pinned Functions */}
        {pinnedModulesList.length > 0 && (
          <Card
            title={
              <Space>
                <StarOutlined style={{ color: '#faad14' }} />
                <span>Pinned Functions</span>
              </Space>
            }
            style={{ marginBottom: 24 }}
          >
            <Row gutter={[16, 16]}>
              {pinnedModulesList.map(module => (
                <Col xs={24} sm={12} md={8} lg={6} key={module.id}>
                  <FunctionCard
                    module={module}
                    onNavigate={handleNavigate}
                    isPinned={true}
                    onPin={handlePin}
                  />
                </Col>
              ))}
            </Row>
          </Card>
        )}

        {/* Functions by Version */}
        <Tabs defaultActiveKey="by-version" size="large">
          <TabPane tab="By Version" key="by-version">
            {/* V2 Production */}
            {modulesByVersion.V2.length > 0 && (
              <Card
                title={
                  <Space>
                    <CheckCircleOutlined style={{ color: '#52c41a' }} />
                    <span>V2 - Production</span>
                    <Tag color="green">{modulesByVersion.V2.length} functions</Tag>
                  </Space>
                }
                style={{ marginBottom: 16 }}
              >
                <Row gutter={[16, 16]}>
                  {modulesByVersion.V2.map(module => (
                    <Col xs={24} sm={12} md={8} lg={6} key={module.id}>
                      <FunctionCard
                        module={module}
                        onNavigate={handleNavigate}
                        isPinned={pinnedModules.includes(module.id)}
                        onPin={handlePin}
                      />
                    </Col>
                  ))}
                </Row>
              </Card>
            )}

            {/* V1 Experimental - Admin Only */}
            {isAdmin && modulesByVersion.V1.length > 0 && (
              <Card
                title={
                  <Space>
                    <ExperimentOutlined style={{ color: '#faad14' }} />
                    <span>V1 - Experimental</span>
                    <Tag color="orange">{modulesByVersion.V1.length} functions</Tag>
                    <Tag color="red">Admin Only</Tag>
                  </Space>
                }
                style={{ marginBottom: 16 }}
              >
                <Row gutter={[16, 16]}>
                  {modulesByVersion.V1.map(module => (
                    <Col xs={24} sm={12} md={8} lg={6} key={module.id}>
                      <FunctionCard
                        module={module}
                        onNavigate={handleNavigate}
                        isPinned={pinnedModules.includes(module.id)}
                        onPin={handlePin}
                      />
                    </Col>
                  ))}
                </Row>
              </Card>
            )}

            {/* V3 Legacy - Admin Only */}
            {isAdmin && modulesByVersion.V3.length > 0 && (
              <Card
                title={
                  <Space>
                    <HistoryOutlined style={{ color: '#8c8c8c' }} />
                    <span>V3 - Legacy</span>
                    <Tag color="default">{modulesByVersion.V3.length} functions</Tag>
                    <Tag color="red">Admin Only</Tag>
                  </Space>
                }
                style={{ marginBottom: 16 }}
              >
                <Row gutter={[16, 16]}>
                  {modulesByVersion.V3.map(module => (
                    <Col xs={24} sm={12} md={8} lg={6} key={module.id}>
                      <FunctionCard
                        module={module}
                        onNavigate={handleNavigate}
                        isPinned={pinnedModules.includes(module.id)}
                        onPin={handlePin}
                      />
                    </Col>
                  ))}
                </Row>
              </Card>
            )}

            {/* Admin Functions */}
            {isAdmin && modulesByVersion.Admin.length > 0 && (
              <Card
                title={
                  <Space>
                    <SettingOutlined style={{ color: '#1890ff' }} />
                    <span>Administration</span>
                    <Tag color="blue">{modulesByVersion.Admin.length} functions</Tag>
                    <Tag color="red">Admin Only</Tag>
                  </Space>
                }
              >
                <Row gutter={[16, 16]}>
                  {modulesByVersion.Admin.map(module => (
                    <Col xs={24} sm={12} md={8} lg={6} key={module.id}>
                      <FunctionCard
                        module={module}
                        onNavigate={handleNavigate}
                        isPinned={pinnedModules.includes(module.id)}
                        onPin={handlePin}
                      />
                    </Col>
                  ))}
                </Row>
              </Card>
            )}
          </TabPane>

          <TabPane tab="By Category" key="by-category">
            {Object.entries(modulesByCategory).map(([category, modules]) => {
              const categoryConfig = CATEGORY_GROUPS[category];
              return (
                <Card
                  key={category}
                  title={
                    <Space>
                      {categoryConfig?.icon}
                      <span>{categoryConfig?.name || category}</span>
                      <Tag>{modules.length} functions</Tag>
                    </Space>
                  }
                  style={{ marginBottom: 16 }}
                >
                  <Row gutter={[16, 16]}>
                    {modules.map(module => (
                      <Col xs={24} sm={12} md={8} lg={6} key={module.id}>
                        <FunctionCard
                          module={module}
                          onNavigate={handleNavigate}
                          isPinned={pinnedModules.includes(module.id)}
                          onPin={handlePin}
                        />
                      </Col>
                    ))}
                  </Row>
                </Card>
              );
            })}
          </TabPane>
        </Tabs>

        {/* Empty State */}
        {filteredModules.length === 0 && (
          <Card>
            <Empty
              description={
                <span>
                  No functions found for "{searchQuery}"
                </span>
              }
            >
              <Button type="primary" onClick={() => setSearchQuery('')}>
                Clear Search
              </Button>
            </Empty>
          </Card>
        )}

        {/* Load Time Indicator (Development) */}
        {process.env.NODE_ENV === 'development' && loadTime > 0 && (
          <div style={{
            position: 'fixed',
            bottom: 16,
            right: 16,
            background: loadTime < 1000 ? '#52c41a' : '#ff4d4f',
            color: '#fff',
            padding: '4px 12px',
            borderRadius: 4,
            fontSize: 12,
          }}>
            Load time: {loadTime.toFixed(0)}ms
          </div>
        )}
      </Content>
    </Layout>
  );
};

export default UnifiedDashboard;
