import React, { useEffect } from 'react';
import { Navigate, useLocation, useNavigate } from 'react-router-dom';
import { Spin, Result, Button, Space } from 'antd';
import { HomeOutlined, ArrowLeftOutlined, LockOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '../../store/authStore';

/**
 * Permission types for fine-grained access control
 */
export type Permission = 
  | 'read:projects' | 'write:projects' | 'delete:projects'
  | 'read:analyses' | 'write:analyses'
  | 'read:users' | 'write:users' | 'delete:users'
  | 'read:providers' | 'write:providers'
  | 'read:audit' | 'admin:all'
  | string; // Allow custom permissions

/**
 * ProtectedRoute component props
 * @interface ProtectedRouteProps
 * @property {React.ReactNode} children - Child components to render when authorized
 * @property {string[]} requiredRoles - Roles required to access the route (OR logic)
 * @property {string[]} requiredPermissions - Permissions required to access the route
 * @property {boolean} requireAll - If true, ALL permissions required; otherwise ANY
 * @property {string} redirectTo - URL to redirect to when not authenticated
 * @property {React.ReactNode} fallback - Custom loading component
 * @property {() => void} onAccessDenied - Callback when access is denied
 */
interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRoles?: string[];
  requiredPermissions?: string[];
  requireAll?: boolean;
  redirectTo?: string;
  fallback?: React.ReactNode;
  onAccessDenied?: () => void;
}

/**
 * Protected Route Component
 * 
 * Wraps routes that require authentication and optionally specific roles.
 * Redirects to login if not authenticated or shows forbidden if lacking permissions.
 */
/**
 * Check if user has required permissions
 */
const hasPermissions = (
  userPermissions: string[] = [],
  requiredPermissions: Permission[] = [],
  requireAll: boolean = false
): boolean => {
  if (requiredPermissions.length === 0) return true;
  
  // Admin has all permissions
  if (userPermissions.includes('admin:all')) return true;
  
  if (requireAll) {
    return requiredPermissions.every(perm => userPermissions.includes(perm));
  } else {
    return requiredPermissions.some(perm => userPermissions.includes(perm));
  }
};

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredRoles = [],
  requiredPermissions = [],
  requireAll = false,
  redirectTo = '/login',
  fallback,
  onAccessDenied,
}) => {
  const { t } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const { isAuthenticated, user, isLoading } = useAuthStore();

  // Call onAccessDenied when access is denied
  useEffect(() => {
    if (!isLoading && isAuthenticated && user) {
      const roleCheck = requiredRoles.length === 0 || requiredRoles.includes(user.role);
      // Permissions are optional on User - use empty array if not present
      const userPermissions = (user as any).permissions || [];
      const permCheck = hasPermissions(userPermissions, requiredPermissions, requireAll);
      
      if (!roleCheck || !permCheck) {
        onAccessDenied?.();
      }
    }
  }, [isLoading, isAuthenticated, user, requiredRoles, requiredPermissions, requireAll, onAccessDenied]);

  // Show loading state while checking authentication
  if (isLoading) {
    return fallback || (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        gap: 16
      }}>
        <Spin size="large" />
        <span style={{ color: '#666' }}>{t('auth.checking', 'Checking authentication...')}</span>
      </div>
    );
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated || !user) {
    // Save the attempted URL for redirecting after login
    return (
      <Navigate 
        to={redirectTo} 
        state={{ from: location.pathname + location.search }} 
        replace 
      />
    );
  }

  // Check role-based access
  const hasRequiredRole = requiredRoles.length === 0 || requiredRoles.includes(user.role);
  
  // Check permission-based access (permissions are optional on User)
  const userPerms = (user as any).permissions || [];
  const hasRequiredPermissions = hasPermissions(userPerms, requiredPermissions, requireAll);

  if (!hasRequiredRole || !hasRequiredPermissions) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        padding: 24,
        background: '#f5f5f5'
      }}>
        <Result
          status="403"
          icon={<LockOutlined style={{ color: '#ff4d4f' }} />}
          title={t('auth.access_denied', 'Access Denied')}
          subTitle={t('auth.no_permission', "You don't have permission to access this page.")}
          extra={
            <Space>
              <Button 
                icon={<ArrowLeftOutlined />} 
                onClick={() => navigate(-1)}
              >
                {t('common.go_back', 'Go Back')}
              </Button>
              <Button 
                type="primary" 
                icon={<HomeOutlined />}
                onClick={() => navigate('/dashboard')}
              >
                {t('common.go_home', 'Go to Dashboard')}
              </Button>
            </Space>
          }
        />
      </div>
    );
  }

  // User is authenticated and has required role/permissions
  return <>{children}</>;
};

/**
 * Admin Route - Shortcut for admin-only routes
 */
export const AdminRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <ProtectedRoute requiredRoles={['admin']}>
    {children}
  </ProtectedRoute>
);

/**
 * User Route - Shortcut for authenticated user routes
 */
export const UserRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <ProtectedRoute requiredRoles={['admin', 'user']}>
    {children}
  </ProtectedRoute>
);

/**
 * Public Route - Only accessible when NOT authenticated
 * Redirects to dashboard if already logged in
 */
export const PublicRoute: React.FC<{
  children: React.ReactNode;
  redirectTo?: string;
}> = ({ children, redirectTo = '/dashboard' }) => {
  const { isAuthenticated } = useAuthStore();

  if (isAuthenticated) {
    return <Navigate to={redirectTo} replace />;
  }

  return <>{children}</>;
};

export default ProtectedRoute;
