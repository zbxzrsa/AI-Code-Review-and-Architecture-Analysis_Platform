/**
 * Page exports / 页面导出
 * All main application pages are exported from here
 * 所有主要应用页面从此处导出
 */

// Authentication pages / 认证页面
export { Login } from './Login';
export { Register } from './Register';

// Main pages / 主要页面
export { Dashboard } from './Dashboard';
export { CodeReview } from './CodeReview';
export { Projects } from './Projects';
export { Notifications } from './Notifications';
export { Help } from './Help';

// Profile pages / 个人资料页面
export { Profile } from './Profile';

// Settings pages / 设置页面
export { Settings } from './Settings';

// Re-export Projects (already exported above)
// Additional project components can be added here if needed

// Admin pages / 管理员页面
export { 
  ExperimentManagement,
  UserManagement,
  ProviderManagement,
  AuditLogs,
} from './admin';
