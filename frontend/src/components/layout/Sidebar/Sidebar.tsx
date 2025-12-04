/**
 * Enhanced Sidebar Component
 * 
 * Main navigation sidebar with:
 * - Collapsible/expandable toggle
 * - Active route highlighting
 * - Search functionality
 * - Quick access favorites
 * - User profile mini card
 * - Keyboard navigation
 * - Mobile drawer support
 */

import React, { useMemo, useCallback, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Menu,
  Input,
  Button,
  Avatar,
  Space,
  Typography,
  Dropdown,
  Drawer,
  Divider,
  theme,
} from 'antd';
import type { MenuProps } from 'antd';
import {
  DashboardOutlined,
  ProjectOutlined,
  CodeOutlined,
  BarChartOutlined,
  SettingOutlined,
  UserOutlined,
  LogoutOutlined,
  SearchOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  StarFilled,
  ExperimentOutlined,
  TeamOutlined,
  AuditOutlined,
  ApiOutlined,
  QuestionCircleOutlined,
  BellOutlined,
  RobotOutlined,
  SafetyCertificateOutlined,
  FileTextOutlined,
  KeyOutlined,
  LinkOutlined,
  GithubOutlined,
  PullRequestOutlined,
  ThunderboltOutlined,
  RocketOutlined,
  DiffOutlined,
  CheckSquareOutlined,
  DollarOutlined,
  BookOutlined,
  DesktopOutlined,
  HomeOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '../../../store/authStore';
import { useUIStore } from '../../../store/uiStore';
import './Sidebar.css';

const { Text } = Typography;

type MenuItem = Required<MenuProps>['items'][number];

interface NavItem {
  key: string;
  label: string;
  icon?: React.ReactNode;
  path?: string;
  children?: NavItem[];
  adminOnly?: boolean;
  badge?: number;
}

/**
 * Create menu item helper
 */
function createMenuItem(
  label: React.ReactNode,
  key: string,
  icon?: React.ReactNode,
  children?: MenuItem[],
  type?: 'group'
): MenuItem {
  return { key, icon, children, label, type } as MenuItem;
}

/**
 * Sidebar Component
 */
export const Sidebar: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const { token } = theme.useToken();
  
  const { user, logout } = useAuthStore();
  const {
    sidebar,
    toggleSidebar,
    setSidebarSearch,
    setSidebarExpandedKeys,
    closeMobileDrawer,
    addFavorite,
    removeFavorite,
  } = useUIStore();

  const searchInputRef = useRef<any>(null);

  // Define navigation items
  const navItems: NavItem[] = useMemo(() => [
    {
      key: 'welcome',
      label: t('nav.welcome', 'Welcome'),
      path: '/welcome',
      icon: <HomeOutlined />,
    },
    {
      key: 'dashboard',
      label: t('nav.dashboard', 'Dashboard'),
      icon: <DashboardOutlined />,
      path: '/dashboard',
    },
    {
      key: 'activity',
      label: t('nav.activity', 'Activity'),
      icon: <BellOutlined />,
      path: '/activity',
    },
    {
      key: 'repositories',
      label: t('nav.repositories', 'Repositories'),
      icon: <GithubOutlined />,
      path: '/repositories',
    },
    {
      key: 'projects',
      label: t('nav.projects', 'Projects'),
      icon: <ProjectOutlined />,
      children: [
        { key: 'projects-list', label: t('nav.all_projects', 'All Projects'), path: '/projects' },
        { key: 'projects-new', label: t('nav.new_project', 'New Project'), path: '/projects/new' },
      ],
    },
    {
      key: 'code-review',
      label: t('nav.code_review', 'Code Review'),
      icon: <CodeOutlined />,
      path: '/review',
    },
    {
      key: 'pull-requests',
      label: t('nav.pull_requests', 'Pull Requests'),
      icon: <PullRequestOutlined />,
      path: '/pull-requests',
    },
    {
      key: 'deployments',
      label: t('nav.deployments', 'Deployments'),
      icon: <RocketOutlined />,
      path: '/deployments',
    },
    {
      key: 'compare',
      label: t('nav.compare', 'Compare'),
      icon: <DiffOutlined />,
      path: '/compare',
    },
    {
      key: 'rules',
      label: t('nav.rules', 'Quality Rules'),
      icon: <CheckSquareOutlined />,
      path: '/rules',
    },
    {
      key: 'analytics',
      label: t('nav.analytics', 'Analytics'),
      icon: <BarChartOutlined />,
      path: '/analytics',
    },
    {
      key: 'security',
      label: t('nav.security', 'Security'),
      icon: <SafetyCertificateOutlined />,
      path: '/security',
    },
    {
      key: 'reports',
      label: t('nav.reports', 'Reports'),
      icon: <FileTextOutlined />,
      path: '/reports',
    },
    {
      key: 'teams',
      label: t('nav.teams', 'Teams'),
      icon: <TeamOutlined />,
      path: '/teams',
    },
    {
      key: 'settings',
      label: t('nav.settings', 'Settings'),
      icon: <SettingOutlined />,
      children: [
        { key: 'profile', label: t('nav.profile', 'Profile'), path: '/profile', icon: <UserOutlined /> },
        { key: 'settings-general', label: t('nav.preferences', 'Preferences'), path: '/settings' },
        { key: 'settings-api-keys', label: t('nav.api_keys', 'API Keys'), path: '/settings/api-keys', icon: <KeyOutlined /> },
        { key: 'settings-integrations', label: t('nav.integrations', 'Integrations'), path: '/settings/integrations', icon: <LinkOutlined /> },
        { key: 'billing', label: t('nav.billing', 'Billing'), path: '/billing', icon: <DollarOutlined /> },
      ],
    },
    {
      key: 'docs',
      label: t('nav.documentation', 'Documentation'),
      icon: <BookOutlined />,
      path: '/docs',
    },
    {
      key: 'admin',
      label: t('nav.admin', 'Administration'),
      icon: <SettingOutlined />,
      adminOnly: true,
      children: [
        { key: 'admin-users', label: t('nav.users', 'Users'), path: '/admin/users', icon: <TeamOutlined /> },
        { key: 'admin-ai-models', label: t('nav.ai_models', 'AI Models'), path: '/admin/ai-models', icon: <RobotOutlined /> },
        { key: 'admin-auto-fix', label: t('nav.auto_fix', 'AI Auto-Fix'), path: '/admin/auto-fix', icon: <ThunderboltOutlined /> },
        { key: 'admin-providers', label: t('nav.providers', 'AI Providers'), path: '/admin/providers', icon: <ApiOutlined /> },
        { key: 'admin-experiments', label: t('nav.experiments', 'Experiments'), path: '/admin/experiments', icon: <ExperimentOutlined /> },
        { key: 'admin-vulnerabilities', label: t('nav.vulnerabilities', 'Vulnerabilities'), path: '/admin/vulnerabilities', icon: <SafetyCertificateOutlined /> },
        { key: 'admin-security', label: t('nav.security', 'Security Scanner'), path: '/admin/security', icon: <SafetyCertificateOutlined /> },
        { key: 'admin-quality', label: t('nav.quality', 'Code Quality'), path: '/admin/quality', icon: <CodeOutlined /> },
        { key: 'admin-evolution', label: t('nav.evolution', 'Evolution Cycle'), path: '/admin/evolution', icon: <RocketOutlined /> },
        { key: 'admin-model-testing', label: t('nav.model_testing', 'Model Testing'), path: '/admin/model-testing', icon: <ExperimentOutlined /> },
        { key: 'admin-model-comparison', label: t('nav.model_comparison', 'Model Comparison'), path: '/admin/model-comparison', icon: <BarChartOutlined /> },
        { key: 'admin-version-comparison', label: t('nav.version_comparison', 'Version Comparison'), path: '/admin/version-comparison', icon: <DiffOutlined /> },
        { key: 'admin-performance', label: t('nav.performance', 'Performance'), path: '/admin/performance', icon: <DashboardOutlined /> },
        { key: 'admin-health', label: t('nav.system_health', 'System Health'), path: '/admin/health', icon: <DesktopOutlined /> },
        { key: 'admin-learning', label: t('nav.learning', 'Learning Cycle'), path: '/admin/learning', icon: <BookOutlined /> },
        { key: 'admin-ml-promotion', label: t('nav.ml_promotion', 'ML Auto-Promotion'), path: '/admin/ml-promotion', icon: <RocketOutlined /> },
        { key: 'admin-audit', label: t('nav.audit', 'Audit Logs'), path: '/admin/audit', icon: <AuditOutlined /> },
      ],
    },
  ], [t]);

  // Filter items based on role
  const filteredNavItems = useMemo(() => {
    return navItems.filter(item => {
      if (item.adminOnly && user?.role !== 'admin') return false;
      return true;
    });
  }, [navItems, user?.role]);

  // Search filter
  const searchFilteredItems = useMemo(() => {
    if (!sidebar.searchQuery) return filteredNavItems;
    
    const query = sidebar.searchQuery.toLowerCase();
    
    const filterItems = (items: NavItem[]): NavItem[] => {
      return items.reduce((acc: NavItem[], item) => {
        const labelMatch = item.label.toLowerCase().includes(query);
        const filteredChildren = item.children ? filterItems(item.children) : undefined;
        
        if (labelMatch || (filteredChildren && filteredChildren.length > 0)) {
          acc.push({
            ...item,
            children: filteredChildren,
          });
        }
        
        return acc;
      }, []);
    };
    
    return filterItems(filteredNavItems);
  }, [filteredNavItems, sidebar.searchQuery]);

  // Convert to Ant Design menu items
  const menuItems: MenuItem[] = useMemo(() => {
    const convertToMenuItem = (item: NavItem): MenuItem => {
      const children = item.children?.map(convertToMenuItem);
      return createMenuItem(
        item.label,
        item.path || item.key,
        item.icon,
        children
      );
    };
    
    return searchFilteredItems.map(convertToMenuItem);
  }, [searchFilteredItems]);

  // Handle menu click
  const handleMenuClick = useCallback((e: { key: string }) => {
    if (e.key.startsWith('/')) {
      navigate(e.key);
      closeMobileDrawer();
    }
  }, [navigate, closeMobileDrawer]);

  // Get selected key based on current path
  const getSelectedKey = useCallback(() => {
    const path = location.pathname;
    if (path.startsWith('/admin')) return path;
    if (path.startsWith('/review')) return '/review';
    if (path.startsWith('/projects/new')) return '/projects/new';
    if (path.startsWith('/projects')) return '/projects';
    return path;
  }, [location.pathname]);

  // Handle favorite toggle (reserved for future use)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _toggleFavorite = useCallback((item: NavItem) => {
    const existingFavorite = sidebar.favorites.find(f => f.path === item.path);
    if (existingFavorite) {
      removeFavorite(existingFavorite.id);
    } else if (item.path) {
      addFavorite({
        label: item.label,
        path: item.path,
        icon: item.key,
      });
    }
  }, [sidebar.favorites, addFavorite, removeFavorite]);

  // Check if item is favorited (reserved for future use)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _isFavorite = useCallback((path?: string) => {
    if (!path) return false;
    return sidebar.favorites.some(f => f.path === path);
  }, [sidebar.favorites]);

  // Handle logout
  const handleLogout = useCallback(() => {
    logout();
    navigate('/login');
  }, [logout, navigate]);

  // User dropdown menu
  const userMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: t('user.profile', 'Profile'),
      onClick: () => navigate('/profile'),
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: t('user.settings', 'Settings'),
      onClick: () => navigate('/settings'),
    },
    { type: 'divider' },
    {
      key: 'help',
      icon: <QuestionCircleOutlined />,
      label: t('user.help', 'Help'),
      onClick: () => navigate('/help'),
    },
    { type: 'divider' },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: t('user.logout', 'Logout'),
      onClick: handleLogout,
      danger: true,
    },
  ];

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl/Cmd + K to focus search
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const collapsed = !sidebar.isOpen;

  // Sidebar content
  const sidebarContent = (
    <div className="sidebar-content">
      {/* Logo */}
      <div className="sidebar-logo">
        {collapsed ? (
          <CodeOutlined style={{ fontSize: 24, color: token.colorPrimary }} />
        ) : (
          <Space>
            <CodeOutlined style={{ fontSize: 24, color: token.colorPrimary }} />
            <Text strong style={{ fontSize: 16 }}>Code Review AI</Text>
          </Space>
        )}
      </div>

      {/* Search */}
      {!collapsed && (
        <div className="sidebar-search">
          <Input
            ref={searchInputRef}
            prefix={<SearchOutlined />}
            placeholder={t('nav.search', 'Search...')}
            value={sidebar.searchQuery}
            onChange={(e) => setSidebarSearch(e.target.value)}
            allowClear
            size="small"
          />
          <Text type="secondary" className="sidebar-search-hint">
            ⌘K
          </Text>
        </div>
      )}

      {/* Favorites */}
      {!collapsed && sidebar.favorites.length > 0 && (
        <div className="sidebar-favorites">
          <Text type="secondary" className="sidebar-section-title">
            <StarFilled /> {t('nav.favorites', 'Favorites')}
          </Text>
          <Menu
            mode="inline"
            selectedKeys={[getSelectedKey()]}
            items={sidebar.favorites.map(fav => ({
              key: fav.path,
              label: fav.label,
              onClick: () => {
                navigate(fav.path);
                closeMobileDrawer();
              },
            }))}
            className="sidebar-favorites-menu"
          />
          <Divider style={{ margin: '8px 0' }} />
        </div>
      )}

      {/* Main Navigation */}
      <div className="sidebar-nav">
        <Menu
          mode="inline"
          selectedKeys={[getSelectedKey()]}
          openKeys={collapsed ? [] : sidebar.expandedKeys}
          onOpenChange={setSidebarExpandedKeys}
          items={menuItems}
          onClick={handleMenuClick}
          inlineCollapsed={collapsed}
        />
      </div>

      {/* User Profile Card */}
      <div className="sidebar-user">
        <Divider style={{ margin: '8px 0' }} />
        <Dropdown
          menu={{ items: userMenuItems }}
          placement="topRight"
          trigger={['click']}
        >
          <div className="sidebar-user-card">
            <Avatar
              src={user?.avatar}
              icon={!user?.avatar && <UserOutlined />}
              style={{ backgroundColor: token.colorPrimary }}
              size={collapsed ? 'default' : 40}
            />
            {!collapsed && (
              <div className="sidebar-user-info">
                <Text strong className="sidebar-user-name">{user?.name || 'User'}</Text>
                <Text type="secondary" className="sidebar-user-role">
                  {user?.role || 'user'}
                </Text>
              </div>
            )}
          </div>
        </Dropdown>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Sidebar */}
      <aside
        className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}
        style={{ background: token.colorBgContainer }}
        role="navigation"
        aria-label={t('nav.main_navigation', 'Main navigation')}
      >
        {/* Collapse Toggle */}
        <Button
          type="text"
          icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
          onClick={toggleSidebar}
          className="sidebar-collapse-btn"
          aria-label={collapsed ? t('nav.expand', 'Expand sidebar') : t('nav.collapse', 'Collapse sidebar')}
        />
        
        {sidebarContent}
      </aside>

      {/* Mobile Drawer */}
      <Drawer
        placement="left"
        open={sidebar.mobileDrawerOpen}
        onClose={closeMobileDrawer}
        width={280}
        className="sidebar-drawer"
        styles={{ body: { padding: 0 } }}
      >
        {sidebarContent}
      </Drawer>
    </>
  );
};

export default Sidebar;
