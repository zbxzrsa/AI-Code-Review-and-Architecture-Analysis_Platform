/**
 * Layout Component / 布局组件
 * 
 * Main application layout with:
 * 主应用布局，包含：
 * - Sidebar navigation / 侧边栏导航
 * - Header with user menu and language selector / 带用户菜单和语言选择器的头部
 * - Content area / 内容区域
 */

import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import {
  Layout as AntLayout,
  Menu,
  Avatar,
  Dropdown,
  Button,
  Space,
  Typography,
  Badge,
  Tooltip,
  theme
} from 'antd';
import type { MenuProps } from 'antd';
import {
  DashboardOutlined,
  ProjectOutlined,
  CodeOutlined,
  ExperimentOutlined,
  SettingOutlined,
  UserOutlined,
  LogoutOutlined,
  BellOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  QuestionCircleOutlined,
  SunOutlined,
  MoonOutlined
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '../../store/authStore';
import { useUIStore } from '../../store/uiStore';
import { LanguageSelector } from '../common/LanguageSelector';
import './Layout.css';

const { Header, Sider, Content } = AntLayout;
const { Text } = Typography;

type MenuItem = Required<MenuProps>['items'][number];

function getItem(
  label: React.ReactNode,
  key: React.Key,
  icon?: React.ReactNode,
  children?: MenuItem[],
  type?: 'group'
): MenuItem {
  return {
    key,
    icon,
    children,
    label,
    type,
  } as MenuItem;
}

export const Layout: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const { token } = theme.useToken();
  
  const { user, logout } = useAuthStore();
  const { sidebar, toggleSidebar, resolvedTheme, setTheme } = useUIStore();
  
  const [collapsed, setCollapsed] = useState(!sidebar.isOpen);
  const [notifications] = useState(3); // Mock notification count

  useEffect(() => {
    setCollapsed(!sidebar.isOpen);
  }, [sidebar.isOpen]);

  const handleCollapse = () => {
    toggleSidebar();
    setCollapsed(!collapsed);
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  // Navigation menu items
  const menuItems: MenuItem[] = [
    getItem(t('nav.dashboard', 'Dashboard'), '/dashboard', <DashboardOutlined />),
    getItem(t('nav.projects', 'Projects'), '/projects', <ProjectOutlined />),
    getItem(t('nav.code_review', 'Code Review'), '/review', <CodeOutlined />),
    
    // Admin menu items (only visible to admins)
    ...(user?.role === 'admin' ? [
      { type: 'divider' as const },
      getItem(t('nav.admin', 'Administration'), 'admin', <SettingOutlined />, [
        getItem(t('nav.experiments', 'Experiments'), '/admin/experiments', <ExperimentOutlined />),
        getItem(t('nav.users', 'Users'), '/admin/users', <UserOutlined />),
        getItem(t('nav.providers', 'AI Providers'), '/admin/providers'),
        getItem(t('nav.audit', 'Audit Logs'), '/admin/audit'),
      ]),
    ] : []),
  ];

  // User dropdown menu
  const userMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: t('user.profile', 'Profile'),
      onClick: () => navigate('/profile')
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: t('user.settings', 'Settings'),
      onClick: () => navigate('/settings')
    },
    { type: 'divider' },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: t('user.logout', 'Logout'),
      onClick: handleLogout,
      danger: true
    }
  ];

  const handleMenuClick = (e: { key: string }) => {
    if (e.key.startsWith('/')) {
      navigate(e.key);
    }
  };

  const getSelectedKey = () => {
    const path = location.pathname;
    // Handle nested routes
    if (path.startsWith('/admin')) {
      return path;
    }
    if (path.startsWith('/review')) {
      return '/review';
    }
    if (path.startsWith('/projects')) {
      return '/projects';
    }
    return path;
  };

  return (
    <AntLayout className="app-layout">
      {/* Sidebar */}
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={handleCollapse}
        trigger={null}
        width={240}
        collapsedWidth={80}
        className="app-sider"
        style={{
          background: token.colorBgContainer
        }}
      >
        {/* Logo */}
        <div className="app-logo">
          {collapsed ? (
            <CodeOutlined style={{ fontSize: 24, color: token.colorPrimary }} />
          ) : (
            <Space>
              <CodeOutlined style={{ fontSize: 24, color: token.colorPrimary }} />
              <Text strong style={{ fontSize: 16 }}>Code Review AI</Text>
            </Space>
          )}
        </div>

        {/* Navigation Menu */}
        <Menu
          mode="inline"
          selectedKeys={[getSelectedKey()]}
          items={menuItems}
          onClick={handleMenuClick}
          style={{ borderRight: 0 }}
        />
      </Sider>

      <AntLayout>
        {/* Header */}
        <Header className="app-header" style={{ background: token.colorBgContainer }}>
          <div className="header-left">
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={handleCollapse}
              className="collapse-btn"
            />
          </div>

          <div className="header-right">
            <Space size="middle">
              {/* Language Selector / 语言选择器 */}
              <LanguageSelector 
                mode="dropdown" 
                size="middle"
                showFlag={true}
                showNativeName={true}
              />

              {/* Theme Toggle / 主题切换 */}
              <Tooltip title={resolvedTheme === 'dark' ? t('accessibility.light_mode', 'Light Mode') : t('accessibility.dark_mode', 'Dark Mode')}>
                <Button
                  type="text"
                  icon={resolvedTheme === 'dark' ? <SunOutlined /> : <MoonOutlined />}
                  onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
                />
              </Tooltip>

              {/* Help / 帮助 */}
              <Tooltip title={t('header.help', 'Help')}>
                <Button
                  type="text"
                  icon={<QuestionCircleOutlined />}
                  onClick={() => navigate('/help')}
                />
              </Tooltip>

              {/* Notifications / 通知 */}
              <Tooltip title={t('header.notifications', 'Notifications')}>
                <Badge count={notifications} size="small">
                  <Button 
                    type="text" 
                    icon={<BellOutlined />} 
                    onClick={() => navigate('/notifications')}
                  />
                </Badge>
              </Tooltip>

              {/* User Menu */}
              <Dropdown
                menu={{ items: userMenuItems }}
                placement="bottomRight"
                trigger={['click']}
              >
                <Space className="user-menu" style={{ cursor: 'pointer' }}>
                  <Avatar
                    src={user?.avatar}
                    icon={!user?.avatar && <UserOutlined />}
                    style={{ backgroundColor: token.colorPrimary }}
                  />
                  {!collapsed && (
                    <div className="user-info">
                      <Text strong>{user?.name || 'User'}</Text>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {user?.role || 'user'}
                      </Text>
                    </div>
                  )}
                </Space>
              </Dropdown>
            </Space>
          </div>
        </Header>

        {/* Main Content */}
        <Content className="app-content">
          <Outlet />
        </Content>
      </AntLayout>
    </AntLayout>
  );
};

export default Layout;
