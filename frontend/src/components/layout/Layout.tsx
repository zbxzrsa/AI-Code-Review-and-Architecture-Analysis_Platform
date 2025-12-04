/**
 * Layout Component / 布局组件
 * 
 * Main application layout with:
 * 主应用布局，包含：
 * - Enhanced sidebar navigation / 增强的侧边栏导航
 * - Header with user menu, search, and notifications / 带用户菜单、搜索和通知的头部
 * - Content area with breadcrumbs / 带面包屑的内容区域
 * - Responsive design / 响应式设计
 */

import React, { useState, useMemo } from 'react';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import {
  Layout as AntLayout,
  Avatar,
  Dropdown,
  Button,
  Space,
  Typography,
  Badge,
  Tooltip,
  Input,
  Breadcrumb,
  theme,
} from 'antd';
import type { MenuProps } from 'antd';
import {
  UserOutlined,
  LogoutOutlined,
  BellOutlined,
  MenuOutlined,
  QuestionCircleOutlined,
  SunOutlined,
  MoonOutlined,
  SettingOutlined,
  HomeOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '../../store/authStore';
import { useUIStore } from '../../store/uiStore';
import { LanguageSelector } from '../common/LanguageSelector';
import { Sidebar } from './Sidebar';
import { FeedbackWidget } from '../feedback/FeedbackWidget';
import './Layout.css';

const { Header, Content } = AntLayout;
const { Text } = Typography;
const { Search } = Input;

/** Breadcrumb route mapping */
const routeToBreadcrumb: Record<string, { label: string; icon?: React.ReactNode }> = {
  '/dashboard': { label: 'Dashboard', icon: <HomeOutlined /> },
  '/projects': { label: 'Projects' },
  '/projects/new': { label: 'New Project' },
  '/review': { label: 'Code Review' },
  '/profile': { label: 'Profile' },
  '/settings': { label: 'Settings' },
  '/notifications': { label: 'Notifications' },
  '/help': { label: 'Help' },
  '/admin': { label: 'Administration' },
  '/admin/users': { label: 'Users' },
  '/admin/providers': { label: 'AI Providers' },
  '/admin/experiments': { label: 'Experiments' },
  '/admin/audit': { label: 'Audit Logs' },
};

export const Layout: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const { token } = theme.useToken();
  
  const { user, logout } = useAuthStore();
  const { 
    sidebar, 
    toggleMobileDrawer, 
    resolvedTheme, 
    setTheme,
    openCommandPalette,
  } = useUIStore();

  const [notifications] = useState(3); // Mock notification count
  const [globalSearch, setGlobalSearch] = useState('');

  // Generate breadcrumbs from current path
  const currentBreadcrumbs = useMemo(() => {
    const paths = location.pathname.split('/').filter(Boolean);
    const items: { title: React.ReactNode; onClick?: () => void; className?: string }[] = [
      {
        title: <><HomeOutlined /> {t('nav.dashboard', 'Dashboard')}</>,
        onClick: () => navigate('/dashboard'),
        className: 'breadcrumb-link',
      },
    ];

    let currentPath = '';
    for (const path of paths) {
      currentPath += `/${path}`;
      const route = routeToBreadcrumb[currentPath];
      if (route) {
        const isCurrentPage = currentPath === location.pathname;
        items.push({
          title: <span>{t(`nav.${path}`, route.label)}</span>,
          onClick: isCurrentPage ? undefined : () => navigate(currentPath),
          className: isCurrentPage ? undefined : 'breadcrumb-link',
        });
      }
    }

    return items.length > 1 ? items : [];
  }, [location.pathname, t, navigate]);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  // Handle global search
  const handleGlobalSearch = (value: string) => {
    if (value.trim()) {
      // Navigate to search results or open command palette
      openCommandPalette();
    }
  };

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

  const collapsed = !sidebar.isOpen;

  return (
    <AntLayout className="app-layout">
      {/* Enhanced Sidebar */}
      <Sidebar />

      <AntLayout className={`app-main ${collapsed ? 'sidebar-collapsed' : ''}`}>
        {/* Header */}
        <Header className="app-header" style={{ background: token.colorBgContainer }}>
          <div className="header-left">
            {/* Mobile Menu Button */}
            <Button
              type="text"
              icon={<MenuOutlined />}
              onClick={toggleMobileDrawer}
              className="mobile-menu-btn"
              aria-label={t('nav.open_menu', 'Open menu')}
            />

            {/* Global Search */}
            <Search
              placeholder={t('header.search', 'Search...')}
              value={globalSearch}
              onChange={(e) => setGlobalSearch(e.target.value)}
              onSearch={handleGlobalSearch}
              className="header-search"
              style={{ width: 240 }}
              allowClear
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
                  aria-label={resolvedTheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                />
              </Tooltip>

              {/* Help / 帮助 */}
              <Tooltip title={t('header.help', 'Help')}>
                <Button
                  type="text"
                  icon={<QuestionCircleOutlined />}
                  onClick={() => navigate('/help')}
                  aria-label={t('header.help', 'Help')}
                />
              </Tooltip>

              {/* Notifications / 通知 */}
              <Tooltip title={t('header.notifications', 'Notifications')}>
                <Badge count={notifications} size="small">
                  <Button 
                    type="text" 
                    icon={<BellOutlined />} 
                    onClick={() => navigate('/notifications')}
                    aria-label={`${t('header.notifications', 'Notifications')} (${notifications})`}
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
                  <div className="user-info-header">
                    <Text strong>{user?.name || 'User'}</Text>
                  </div>
                </Space>
              </Dropdown>
            </Space>
          </div>
        </Header>

        {/* Breadcrumbs */}
        {currentBreadcrumbs.length > 0 && (
          <div className="app-breadcrumbs" style={{ background: token.colorBgContainer }}>
            <Breadcrumb items={currentBreadcrumbs} />
          </div>
        )}

        {/* Main Content */}
        <Content className="app-content">
          <Outlet />
        </Content>

        {/* Feedback Widget */}
        <FeedbackWidget position="bottom-right" />
      </AntLayout>
    </AntLayout>
  );
};

export default Layout;
