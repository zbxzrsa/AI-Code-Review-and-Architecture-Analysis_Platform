import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { Spin, Result, Button } from 'antd';
import { useAuthStore } from '../../store/authStore';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRoles?: string[];
  redirectTo?: string;
  fallback?: React.ReactNode;
}

/**
 * Protected Route Component
 * 
 * Wraps routes that require authentication and optionally specific roles.
 * Redirects to login if not authenticated or shows forbidden if lacking permissions.
 */
export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredRoles = [],
  redirectTo = '/login',
  fallback
}) => {
  const location = useLocation();
  const { isAuthenticated, user, isLoading } = useAuthStore();

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
        <span style={{ color: '#666' }}>Checking authentication...</span>
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
  if (requiredRoles.length > 0 && !requiredRoles.includes(user.role)) {
    return (
      <Result
        status="403"
        title="Access Denied"
        subTitle="You don't have permission to access this page."
        extra={
          <Button type="primary" onClick={() => window.history.back()}>
            Go Back
          </Button>
        }
      />
    );
  }

  // User is authenticated and has required role
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
