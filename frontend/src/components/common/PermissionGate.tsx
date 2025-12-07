/**
 * Permission Gate Component
 *
 * Conditionally renders content based on user permissions.
 * Provides visual layer protection for admin-only features.
 *
 * IMPORTANT: This is visual protection only. Backend must verify permissions.
 */

import React, { useEffect } from 'react';
import { usePermissions, Permission, FeatureAccess } from '../../hooks/usePermissions';
import { UserRole } from '../../store/authStore';
import { useUIStore } from '../../store/uiStore';

// =============================================================================
// Types
// =============================================================================

interface PermissionGateProps {
  /** Content to render when permission is granted */
  children: React.ReactNode;
  /** Alternative content when access is denied (default: nothing) */
  fallback?: React.ReactNode;
  /** Required role (user must have this role or higher) */
  requiredRole?: UserRole;
  /** Required permission */
  requiredPermission?: Permission;
  /** Required permissions (any or all based on requireAll) */
  requiredPermissions?: Permission[];
  /** If true, user must have ALL permissions; otherwise ANY */
  requireAll?: boolean;
  /** If true, logs access denial attempts */
  logDenials?: boolean;
  /** Feature identifier for logging */
  featureId?: string;
  /** If true, show nothing when denied (default behavior) */
  hideOnDenied?: boolean;
}

interface AdminOnlyProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  featureId?: string;
}

interface UserOnlyProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

// =============================================================================
// Access Denial Logger (Frontend)
// =============================================================================

interface AccessDenialLog {
  timestamp: string;
  userId?: string;
  userRole?: string;
  featureId?: string;
  requiredRole?: string;
  requiredPermission?: string;
  path: string;
}

class FrontendAccessLogger {
  private static logs: AccessDenialLog[] = [];
  private static maxLogs = 100;

  static logDenial(log: Omit<AccessDenialLog, 'timestamp' | 'path'>) {
    const entry: AccessDenialLog = {
      ...log,
      timestamp: new Date().toISOString(),
      path: typeof window !== 'undefined' ? window.location.pathname : '',
    };

    this.logs.push(entry);

    // Trim old logs
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(-this.maxLogs);
    }

    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.warn('[PermissionGate] Access denied:', entry);
    }

    // Send to backend for audit logging
    this.sendToBackend(entry).catch(() => {
      // Silently fail - audit logging shouldn't break the app
    });
  }

  private static async sendToBackend(log: AccessDenialLog) {
    try {
      await fetch('/api/audit/access-denial', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(log),
        credentials: 'include',
      });
    } catch {
      // Ignore errors
    }
  }

  static getRecentDenials(): AccessDenialLog[] {
    return [...this.logs];
  }
}

// =============================================================================
// PermissionGate Component
// =============================================================================

/**
 * PermissionGate - Conditionally render content based on permissions
 *
 * @example
 * // Hide admin buttons from regular users
 * <PermissionGate requiredRole="admin">
 *   <Button>Delete All Users</Button>
 * </PermissionGate>
 *
 * @example
 * // Show alternative content when denied
 * <PermissionGate
 *   requiredPermission="admin:users"
 *   fallback={<p>You don't have access to user management</p>}
 * >
 *   <UserManagementPanel />
 * </PermissionGate>
 */
export const PermissionGate: React.FC<PermissionGateProps> = ({
  children,
  fallback = null,
  requiredRole,
  requiredPermission,
  requiredPermissions = [],
  requireAll = false,
  logDenials = true,
  featureId,
  hideOnDenied = true,
}) => {
  const {
    user,
    role,
    hasRole,
    hasPermission,
    hasAnyPermission,
    hasAllPermissions,
    isAuthenticated,
  } = usePermissions();

  // Check access
  let hasAccess = true;
  let denialReason = '';

  // Check role
  if (requiredRole && !hasRole(requiredRole)) {
    hasAccess = false;
    denialReason = `Requires ${requiredRole} role`;
  }

  // Check single permission
  if (hasAccess && requiredPermission && !hasPermission(requiredPermission)) {
    hasAccess = false;
    denialReason = `Requires ${requiredPermission} permission`;
  }

  // Check multiple permissions
  if (hasAccess && requiredPermissions.length > 0) {
    const permCheck = requireAll
      ? hasAllPermissions(requiredPermissions)
      : hasAnyPermission(requiredPermissions);

    if (!permCheck) {
      hasAccess = false;
      denialReason = `Requires ${requireAll ? 'all' : 'any'} of: ${requiredPermissions.join(', ')}`;
    }
  }

  // Log denial
  useEffect(() => {
    if (!hasAccess && logDenials && isAuthenticated) {
      FrontendAccessLogger.logDenial({
        userId: user?.id,
        userRole: role,
        featureId,
        requiredRole,
        requiredPermission: requiredPermission || requiredPermissions.join(', '),
      });
    }
  }, [hasAccess, logDenials, isAuthenticated, user?.id, role, featureId, requiredRole, requiredPermission, requiredPermissions]);

  // Render
  if (hasAccess) {
    return <>{children}</>;
  }

  if (hideOnDenied && !fallback) {
    return null;
  }

  return <>{fallback}</>;
};

// =============================================================================
// Convenience Components
// =============================================================================

/**
 * AdminOnly - Only render for admin users
 *
 * @example
 * <AdminOnly>
 *   <AdminSettingsPanel />
 * </AdminOnly>
 */
export const AdminOnly: React.FC<AdminOnlyProps> = ({
  children,
  fallback,
  featureId,
}) => (
  <PermissionGate
    requiredRole="admin"
    fallback={fallback}
    featureId={featureId}
    logDenials={true}
  >
    {children}
  </PermissionGate>
);

/**
 * UserOnly - Only render for authenticated users (not guests)
 */
export const UserOnly: React.FC<UserOnlyProps> = ({ children, fallback }) => (
  <PermissionGate requiredRole="user" fallback={fallback}>
    {children}
  </PermissionGate>
);

/**
 * ViewerOnly - Render for viewers and above
 */
export const ViewerOnly: React.FC<UserOnlyProps> = ({ children, fallback }) => (
  <PermissionGate requiredRole="viewer" fallback={fallback}>
    {children}
  </PermissionGate>
);

/**
 * NonAdminOnly - Only render for non-admin users
 */
export const NonAdminOnly: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAdmin } = usePermissions();

  if (isAdmin) {
    return null;
  }

  return <>{children}</>;
};

// =============================================================================
// HOC for Permission Checking
// =============================================================================

/**
 * withPermission - HOC to wrap components with permission checks
 *
 * @example
 * const ProtectedComponent = withPermission(MyComponent, { requiredRole: 'admin' });
 */
export function withPermission<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options: {
    requiredRole?: UserRole;
    requiredPermission?: Permission;
    requiredPermissions?: Permission[];
    requireAll?: boolean;
    fallback?: React.ReactNode;
  }
): React.FC<P> {
  const ComponentWithPermission: React.FC<P> = (props) => (
    <PermissionGate {...options}>
      <WrappedComponent {...props} />
    </PermissionGate>
  );

  ComponentWithPermission.displayName = `withPermission(${WrappedComponent.displayName || WrappedComponent.name || 'Component'})`;

  return ComponentWithPermission;
}

// =============================================================================
// Permission Indicator Component
// =============================================================================

interface PermissionIndicatorProps {
  permission: Permission;
  showGranted?: boolean;
  showDenied?: boolean;
}

/**
 * PermissionIndicator - Shows visual indicator of permission status
 */
export const PermissionIndicator: React.FC<PermissionIndicatorProps> = ({
  permission,
  showGranted = true,
  showDenied = true,
}) => {
  const { hasPermission } = usePermissions();
  const granted = hasPermission(permission);

  if (granted && showGranted) {
    return (
      <span style={{ color: '#52c41a', fontSize: '12px' }}>
        ✓ {permission}
      </span>
    );
  }

  if (!granted && showDenied) {
    return (
      <span style={{ color: '#ff4d4f', fontSize: '12px' }}>
        ✗ {permission}
      </span>
    );
  }

  return null;
};

// =============================================================================
// Role Badge Component
// =============================================================================

interface RoleBadgeProps {
  role?: UserRole;
  size?: 'small' | 'default' | 'large';
}

const ROLE_COLORS: Record<UserRole, string> = {
  admin: '#f50',
  user: '#108ee9',
  viewer: '#87d068',
  guest: '#d9d9d9',
};

const ROLE_LABELS: Record<UserRole, string> = {
  admin: 'Administrator',
  user: 'User',
  viewer: 'Viewer',
  guest: 'Guest',
};

/**
 * RoleBadge - Display user role as a badge
 */
export const RoleBadge: React.FC<RoleBadgeProps> = ({ role, size = 'default' }) => {
  const { role: currentRole } = usePermissions();
  const displayRole = role || currentRole;

  const fontSize = size === 'small' ? '11px' : size === 'large' ? '14px' : '12px';
  const padding = size === 'small' ? '2px 6px' : size === 'large' ? '6px 12px' : '4px 8px';

  return (
    <span
      style={{
        display: 'inline-block',
        padding,
        fontSize,
        fontWeight: 500,
        color: '#fff',
        backgroundColor: ROLE_COLORS[displayRole],
        borderRadius: '4px',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
      }}
    >
      {ROLE_LABELS[displayRole]}
    </span>
  );
};

// =============================================================================
// Exports
// =============================================================================

export { FrontendAccessLogger };

export default PermissionGate;
