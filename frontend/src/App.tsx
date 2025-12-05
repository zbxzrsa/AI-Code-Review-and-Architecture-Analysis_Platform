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
import { NotificationCenter } from './components/common/NotificationCenter';
import { Layout } from './components/layout';
import { isRTL } from './i18n/config';

// Artistic theme styles
import './styles/artistic-theme.css';

// Initialize i18n / 初始化国际化
import './i18n';
import './App.css';
import './styles/pixel-theme.css';

/**
 * Lazy load pages for code splitting
 * 懒加载页面组件，实现代码分割
 */
const Login = lazy(() => import('./pages/Login'));
const Register = lazy(() => import('./pages/Register'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const CodeReview = lazy(() => import('./pages/CodeReview'));
const Projects = lazy(() => import('./pages/Projects'));
const Notifications = lazy(() => import('./pages/Notifications'));
const Help = lazy(() => import('./pages/Help'));

// Profile & Settings pages / 个人资料和设置页面
const Profile = lazy(() => import('./pages/profile/Profile'));
const Settings = lazy(() => import('./pages/settings/Settings'));
const APIKeys = lazy(() => import('./pages/settings/APIKeys'));
const Integrations = lazy(() => import('./pages/settings/Integrations'));

// Additional pages / 其他页面
const Analytics = lazy(() => import('./pages/Analytics'));
const SecurityDashboard = lazy(() => import('./pages/SecurityDashboard'));
const TeamManagement = lazy(() => import('./pages/TeamManagement'));
const Reports = lazy(() => import('./pages/Reports'));
const ActivityFeed = lazy(() => import('./pages/ActivityFeed'));
const Repositories = lazy(() => import('./pages/Repositories'));
const PullRequests = lazy(() => import('./pages/PullRequests'));
// AutoFix page - uses AutoFixDashboard instead
const Deployments = lazy(() => import('./pages/Deployments'));
const CodeComparison = lazy(() => import('./pages/CodeComparison'));
const CodeQualityRules = lazy(() => import('./pages/CodeQualityRules'));
const Billing = lazy(() => import('./pages/Billing'));
const Documentation = lazy(() => import('./pages/Documentation'));
const NotificationCenterPage = lazy(() => import('./pages/NotificationCenter'));
const SystemStatus = lazy(() => import('./pages/SystemStatus'));
const SearchResults = lazy(() => import('./pages/SearchResults'));
const Onboarding = lazy(() => import('./pages/Onboarding'));
const Changelog = lazy(() => import('./pages/Changelog'));
const Shortcuts = lazy(() => import('./pages/Shortcuts'));
// AIAssistant redirects to CodeReview - import removed
const Webhooks = lazy(() => import('./pages/Webhooks'));
const CodeMetrics = lazy(() => import('./pages/CodeMetrics'));
const ScheduledJobs = lazy(() => import('./pages/ScheduledJobs'));
const ImportExport = lazy(() => import('./pages/ImportExport'));
const AuditLogs = lazy(() => import('./pages/AuditLogs'));

// Projects pages / 项目管理页面
const ProjectList = lazy(() => import('./pages/projects/ProjectList'));
const NewProject = lazy(() => import('./pages/projects/NewProject'));
const ProjectSettings = lazy(() => import('./pages/projects/ProjectSettings'));

// Admin pages / 管理员页面
const ExperimentManagement = lazy(() => import('./pages/admin/ExperimentManagement'));
const UserManagement = lazy(() => import('./pages/admin/UserManagement'));
const ProviderManagement = lazy(() => import('./pages/admin/ProviderManagement'));
const AdminAuditLogs = lazy(() => import('./pages/admin/AuditLogs'));
const AIModels = lazy(() => import('./pages/admin/AIModels'));

// Self-Evolution pages / 自演化页面
const VulnerabilityDashboard = lazy(() => import('./pages/VulnerabilityDashboard'));
const EvolutionCycleDashboard = lazy(() => import('./pages/EvolutionCycleDashboard'));
const AIModelTesting = lazy(() => import('./pages/admin/AIModelTesting'));
const SystemHealth = lazy(() => import('./pages/admin/SystemHealth'));
const AutoFixDashboard = lazy(() => import('./pages/admin/AutoFixDashboard'));
const LearningCycleDashboard = lazy(() => import('./pages/admin/LearningCycleDashboard'));
const ModelComparison = lazy(() => import('./pages/admin/ModelComparison'));
const PerformanceMonitor = lazy(() => import('./pages/admin/PerformanceMonitor'));
const SecurityScanner = lazy(() => import('./pages/admin/SecurityScanner'));
const CodeQualityDashboard = lazy(() => import('./pages/admin/CodeQualityDashboard'));
const VersionComparison = lazy(() => import('./pages/admin/VersionComparison'));
const ThreeVersionControl = lazy(() => import('./pages/admin/ThreeVersionControl'));
const WelcomeDashboard = lazy(() => import('./pages/WelcomeDashboard'));
const MLAutoPromotion = lazy(() => import('./pages/MLAutoPromotion'));

// AI Interaction pages / AI交互页面
const VersionControlAI = lazy(() => import('./pages/ai/VersionControlAI'));

// OAuth Callback / OAuth回调
const OAuthCallback = lazy(() => import('./pages/OAuthCallback'));

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
            <NotificationCenter />
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

                {/* OAuth Callback route / OAuth回调路由 */}
                <Route path="/oauth/callback/:provider" element={<OAuthCallback />} />

                {/* Protected routes with Layout / 受保护的路由（带布局） */}
                <Route element={
                  <ProtectedRoute>
                    <Layout />
                  </ProtectedRoute>
                }>
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/welcome" element={<WelcomeDashboard />} />
                  
                  {/* Projects routes / 项目路由 */}
                  <Route path="/projects" element={<ProjectList />} />
                  <Route path="/projects/new" element={<NewProject />} />
                  <Route path="/projects/:id/settings" element={<ProjectSettings />} />
                  
                  {/* Legacy projects route for backwards compatibility */}
                  <Route path="/projects/:projectId" element={<Projects />} />
                  
                  <Route path="/review" element={<CodeReview />} />
                  <Route path="/review/:projectId" element={<CodeReview />} />
                  <Route path="/settings" element={<Settings />} />
                  <Route path="/settings/api-keys" element={<APIKeys />} />
                  <Route path="/settings/integrations" element={<Integrations />} />
                  <Route path="/profile" element={<Profile />} />
                  <Route path="/notifications" element={<Notifications />} />
                  <Route path="/help" element={<Help />} />
                  
                  {/* Analytics & Security / 分析和安全 */}
                  <Route path="/analytics" element={<Analytics />} />
                  <Route path="/security" element={<SecurityDashboard />} />
                  <Route path="/teams" element={<TeamManagement />} />
                  <Route path="/reports" element={<Reports />} />
                  <Route path="/activity" element={<ActivityFeed />} />
                  <Route path="/repositories" element={<Repositories />} />
                  <Route path="/pull-requests" element={<PullRequests />} />
                  <Route path="/deployments" element={<Deployments />} />
                  <Route path="/compare" element={<CodeComparison />} />
                  <Route path="/rules" element={<CodeQualityRules />} />
                  <Route path="/billing" element={<Billing />} />
                  <Route path="/docs" element={<Documentation />} />
                  <Route path="/notification-center" element={<NotificationCenterPage />} />
                  <Route path="/status" element={<SystemStatus />} />
                  <Route path="/search" element={<SearchResults />} />
                  <Route path="/onboarding" element={<Onboarding />} />
                  <Route path="/changelog" element={<Changelog />} />
                  <Route path="/shortcuts" element={<Shortcuts />} />
                  {/* AI Assistant redirects to Code Review with AI Chat */}
                  <Route path="/ai-assistant" element={<Navigate to="/review" replace />} />
                  <Route path="/ai-code-review" element={<Navigate to="/review" replace />} />
                  <Route path="/webhooks" element={<Webhooks />} />
                  <Route path="/metrics" element={<CodeMetrics />} />
                  <Route path="/scheduled-jobs" element={<ScheduledJobs />} />
                  <Route path="/import-export" element={<ImportExport />} />
                  <Route path="/audit-logs" element={<AuditLogs />} />

                  {/* Admin routes */}
                  <Route path="/admin/auto-fix" element={
                    <AdminRoute>
                      <AutoFixDashboard />
                    </AdminRoute>
                  } />
                  <Route path="/admin/experiments" element={
                    <AdminRoute>
                      <ExperimentManagement />
                    </AdminRoute>
                  } />
                  <Route path="/admin/users" element={
                    <AdminRoute>
                      <UserManagement />
                    </AdminRoute>
                  } />
                  <Route path="/admin/providers" element={
                    <AdminRoute>
                      <ProviderManagement />
                    </AdminRoute>
                  } />
                  <Route path="/admin/audit" element={
                    <AdminRoute>
                      <AdminAuditLogs />
                    </AdminRoute>
                  } />
                  <Route path="/admin/ai-models" element={
                    <AdminRoute>
                      <AIModels />
                    </AdminRoute>
                  } />
                  <Route path="/admin/version-comparison" element={
                    <AdminRoute>
                      <VersionComparison />
                    </AdminRoute>
                  } />
                  <Route path="/admin/vulnerabilities" element={
                    <AdminRoute>
                      <VulnerabilityDashboard />
                    </AdminRoute>
                  } />
                  <Route path="/admin/evolution" element={
                    <AdminRoute>
                      <EvolutionCycleDashboard />
                    </AdminRoute>
                  } />
                  <Route path="/admin/model-testing" element={
                    <AdminRoute>
                      <AIModelTesting />
                    </AdminRoute>
                  } />
                  <Route path="/admin/health" element={
                    <AdminRoute>
                      <SystemHealth />
                    </AdminRoute>
                  } />
                  <Route path="/admin/learning" element={
                    <AdminRoute>
                      <LearningCycleDashboard />
                    </AdminRoute>
                  } />
                  <Route path="/admin/model-comparison" element={
                    <AdminRoute>
                      <ModelComparison />
                    </AdminRoute>
                  } />
                  <Route path="/admin/performance" element={
                    <AdminRoute>
                      <PerformanceMonitor />
                    </AdminRoute>
                  } />
                  <Route path="/admin/security" element={
                    <AdminRoute>
                      <SecurityScanner />
                    </AdminRoute>
                  } />
                  <Route path="/admin/quality" element={
                    <AdminRoute>
                      <CodeQualityDashboard />
                    </AdminRoute>
                  } />
                  <Route path="/admin/ml-promotion" element={
                    <AdminRoute>
                      <MLAutoPromotion />
                    </AdminRoute>
                  } />
                  <Route path="/admin/three-version" element={
                    <AdminRoute>
                      <ThreeVersionControl />
                    </AdminRoute>
                  } />
                  <Route path="/admin/vcai" element={
                    <AdminRoute>
                      <VersionControlAI />
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
