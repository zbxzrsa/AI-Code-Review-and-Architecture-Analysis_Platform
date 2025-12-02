/**
 * Main Application Component / 主应用组件
 * 
 * This is the root component that sets up:
 * 这是设置以下内容的根组件：
 * - Internationalization (i18n) / 国际化
 * - Theme configuration / 主题配置
 * - Routing / 路由
 * - State management / 状态管理
 */

import React, { Suspense, lazy, useMemo } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider, theme, Spin } from 'antd';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';

// Ant Design locales / Ant Design 语言包
import enUS from 'antd/locale/en_US';
import zhCN from 'antd/locale/zh_CN';
import zhTW from 'antd/locale/zh_TW';

import { useUIStore } from './store/uiStore';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { ProtectedRoute, PublicRoute, AdminRoute } from './components/common/ProtectedRoute';
import { CommandPalette } from './components/common/CommandPalette';
import { Layout } from './components/layout';
import { isRTL } from './i18n/config';

// Initialize i18n / 初始化国际化
import './i18n';
import './App.css';

/**
 * Lazy load pages for code splitting
 * 懒加载页面组件，实现代码分割
 */
const Login = lazy(() => import('./pages/Login'));
const Register = lazy(() => import('./pages/Register'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const CodeReview = lazy(() => import('./pages/CodeReview'));
const Projects = lazy(() => import('./pages/Projects'));
const Settings = lazy(() => import('./pages/Settings'));
const Profile = lazy(() => import('./pages/Profile'));
const Notifications = lazy(() => import('./pages/Notifications'));
const Help = lazy(() => import('./pages/Help'));
const ExperimentManagement = lazy(() => import('./pages/admin/ExperimentManagement'));

// Query client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

/**
 * Loading fallback component / 加载回退组件
 */
const PageLoader: React.FC = () => (
  <div style={{ 
    display: 'flex', 
    flexDirection: 'column',
    justifyContent: 'center', 
    alignItems: 'center', 
    height: '100vh',
    gap: 16
  }}>
    <Spin size="large" />
    <span style={{ color: '#666' }}>Loading...</span>
  </div>
);

/**
 * Ant Design locale mapping / Ant Design 语言包映射
 */
const antdLocales: Record<string, typeof enUS> = {
  en: enUS,
  'zh-CN': zhCN,
  'zh-TW': zhTW,
};

export default function App() {
  const { resolvedTheme } = useUIStore();
  const { i18n } = useTranslation();

  // Get current language / 获取当前语言
  const currentLanguage = i18n.language;
  
  // Get Ant Design locale / 获取 Ant Design 语言包
  const antdLocale = useMemo(() => {
    return antdLocales[currentLanguage] || enUS;
  }, [currentLanguage]);

  // Check if current language is RTL / 检查当前语言是否从右到左
  const direction = useMemo(() => {
    return isRTL(currentLanguage) ? 'rtl' : 'ltr';
  }, [currentLanguage]);

  // Theme configuration / 主题配置
  const themeConfig = useMemo(() => ({
    token: {
      colorPrimary: '#1890ff',
      borderRadius: 6,
    },
    algorithm: resolvedTheme === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
  }), [resolvedTheme]);

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <ConfigProvider 
          theme={themeConfig} 
          locale={antdLocale}
          direction={direction}
        >
          <Router>
            <CommandPalette />
            <Suspense fallback={<PageLoader />}>
              <Routes>
                {/* Public routes / 公开路由 */}
                <Route path="/login" element={
                  <PublicRoute>
                    <Login />
                  </PublicRoute>
                } />
                
                <Route path="/register" element={
                  <PublicRoute>
                    <Register />
                  </PublicRoute>
                } />

                {/* Protected routes with Layout / 受保护的路由（带布局） */}
                <Route element={
                  <ProtectedRoute>
                    <Layout />
                  </ProtectedRoute>
                }>
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/projects" element={<Projects />} />
                  <Route path="/projects/:projectId" element={<Projects />} />
                  <Route path="/review" element={<CodeReview />} />
                  <Route path="/review/:projectId" element={<CodeReview />} />
                  <Route path="/settings" element={<Settings />} />
                  <Route path="/profile" element={<Profile />} />
                  <Route path="/notifications" element={<Notifications />} />
                  <Route path="/help" element={<Help />} />

                  {/* Admin routes */}
                  <Route path="/admin/experiments" element={
                    <AdminRoute>
                      <ExperimentManagement />
                    </AdminRoute>
                  } />
                </Route>

                {/* Redirect root to dashboard */}
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                
                {/* 404 redirect */}
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </Routes>
            </Suspense>
          </Router>
        </ConfigProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}
