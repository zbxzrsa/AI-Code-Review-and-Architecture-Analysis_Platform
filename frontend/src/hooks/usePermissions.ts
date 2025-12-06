/**
 * Permission Management Hook
 *
 * Provides centralized permission checking for the application.
 * Handles role-based and permission-based access control.
 */

import { useMemo, useCallback } from "react";
import { useAuthStore, UserRole } from "../store/authStore";

/**
 * Permission types for fine-grained access control
 */
export type Permission =
  | "read:projects"
  | "write:projects"
  | "delete:projects"
  | "read:analyses"
  | "write:analyses"
  | "read:users"
  | "write:users"
  | "delete:users"
  | "read:providers"
  | "write:providers"
  | "read:audit"
  | "admin:all"
  | "admin:users"
  | "admin:ai-models"
  | "admin:security"
  | "admin:system";

/**
 * Role hierarchy - higher roles inherit permissions from lower roles
 */
const ROLE_HIERARCHY: Record<UserRole, number> = {
  guest: 0,
  viewer: 1,
  user: 2,
  admin: 3,
};

/**
 * Default permissions for each role
 */
const ROLE_PERMISSIONS: Record<UserRole, Permission[]> = {
  guest: [],
  viewer: ["read:projects", "read:analyses"],
  user: ["read:projects", "write:projects", "read:analyses", "write:analyses"],
  admin: [
    "admin:all",
    "admin:users",
    "admin:ai-models",
    "admin:security",
    "admin:system",
    "read:projects",
    "write:projects",
    "delete:projects",
    "read:analyses",
    "write:analyses",
    "read:users",
    "write:users",
    "delete:users",
    "read:providers",
    "write:providers",
    "read:audit",
  ],
};

/**
 * Features that are restricted to specific roles
 */
export interface FeatureAccess {
  id: string;
  name: string;
  description: string;
  requiredRole: UserRole;
  requiredPermissions?: Permission[];
  path?: string;
  isAdminOnly: boolean;
}

/**
 * All application features with their access requirements
 */
export const APP_FEATURES: FeatureAccess[] = [
  // User Features (accessible by all authenticated users)
  {
    id: "dashboard",
    name: "Dashboard",
    description: "View dashboard",
    requiredRole: "user",
    path: "/dashboard",
    isAdminOnly: false,
  },
  {
    id: "projects",
    name: "Projects",
    description: "Manage projects",
    requiredRole: "user",
    path: "/projects",
    isAdminOnly: false,
  },
  {
    id: "code-review",
    name: "Code Review",
    description: "Review code",
    requiredRole: "user",
    path: "/review",
    isAdminOnly: false,
  },
  {
    id: "repositories",
    name: "Repositories",
    description: "Manage repositories",
    requiredRole: "user",
    path: "/repositories",
    isAdminOnly: false,
  },
  {
    id: "analytics",
    name: "Analytics",
    description: "View analytics",
    requiredRole: "user",
    path: "/analytics",
    isAdminOnly: false,
  },
  {
    id: "security",
    name: "Security",
    description: "Security dashboard",
    requiredRole: "user",
    path: "/security",
    isAdminOnly: false,
  },
  {
    id: "reports",
    name: "Reports",
    description: "View reports",
    requiredRole: "user",
    path: "/reports",
    isAdminOnly: false,
  },
  {
    id: "teams",
    name: "Teams",
    description: "Team management",
    requiredRole: "user",
    path: "/teams",
    isAdminOnly: false,
  },
  {
    id: "settings",
    name: "Settings",
    description: "User settings",
    requiredRole: "user",
    path: "/settings",
    isAdminOnly: false,
  },
  {
    id: "profile",
    name: "Profile",
    description: "User profile",
    requiredRole: "user",
    path: "/profile",
    isAdminOnly: false,
  },

  // Viewer Features (read-only access)
  {
    id: "view-projects",
    name: "View Projects",
    description: "View projects (read-only)",
    requiredRole: "viewer",
    path: "/projects",
    isAdminOnly: false,
  },

  // Admin Features (admin only)
  {
    id: "admin-users",
    name: "User Management",
    description: "Manage users",
    requiredRole: "admin",
    requiredPermissions: ["admin:users"],
    path: "/admin/users",
    isAdminOnly: true,
  },
  {
    id: "admin-ai-models",
    name: "AI Models",
    description: "Manage AI models",
    requiredRole: "admin",
    requiredPermissions: ["admin:ai-models"],
    path: "/admin/ai-models",
    isAdminOnly: true,
  },
  {
    id: "admin-providers",
    name: "AI Providers",
    description: "Manage AI providers",
    requiredRole: "admin",
    requiredPermissions: ["admin:ai-models"],
    path: "/admin/providers",
    isAdminOnly: true,
  },
  {
    id: "admin-security",
    name: "Security Settings",
    description: "System security",
    requiredRole: "admin",
    requiredPermissions: ["admin:security"],
    path: "/admin/security",
    isAdminOnly: true,
  },
  {
    id: "admin-audit",
    name: "Audit Logs",
    description: "View audit logs",
    requiredRole: "admin",
    requiredPermissions: ["read:audit"],
    path: "/admin/audit",
    isAdminOnly: true,
  },
  {
    id: "admin-health",
    name: "System Health",
    description: "System health monitoring",
    requiredRole: "admin",
    requiredPermissions: ["admin:system"],
    path: "/admin/health",
    isAdminOnly: true,
  },
  {
    id: "admin-performance",
    name: "Performance",
    description: "Performance monitoring",
    requiredRole: "admin",
    requiredPermissions: ["admin:system"],
    path: "/admin/performance",
    isAdminOnly: true,
  },
  {
    id: "admin-experiments",
    name: "Experiments",
    description: "AI experiments",
    requiredRole: "admin",
    requiredPermissions: ["admin:ai-models"],
    path: "/admin/experiments",
    isAdminOnly: true,
  },
  {
    id: "admin-evolution",
    name: "Evolution Cycle",
    description: "AI evolution cycle",
    requiredRole: "admin",
    requiredPermissions: ["admin:ai-models"],
    path: "/admin/evolution",
    isAdminOnly: true,
  },
  {
    id: "admin-auto-fix",
    name: "Auto-Fix",
    description: "Auto-fix management",
    requiredRole: "admin",
    requiredPermissions: ["admin:ai-models"],
    path: "/admin/auto-fix",
    isAdminOnly: true,
  },
  {
    id: "admin-vulnerabilities",
    name: "Vulnerabilities",
    description: "Vulnerability management",
    requiredRole: "admin",
    requiredPermissions: ["admin:security"],
    path: "/admin/vulnerabilities",
    isAdminOnly: true,
  },
];

/**
 * Permission hook result
 */
export interface UsePermissionsResult {
  // User info
  user: ReturnType<typeof useAuthStore>["user"];
  role: UserRole;
  isAuthenticated: boolean;
  isAdmin: boolean;
  isUser: boolean;
  isViewer: boolean;
  isGuest: boolean;

  // Permission checks
  hasRole: (role: UserRole) => boolean;
  hasPermission: (permission: Permission) => boolean;
  hasAnyPermission: (permissions: Permission[]) => boolean;
  hasAllPermissions: (permissions: Permission[]) => boolean;
  canAccess: (feature: FeatureAccess) => boolean;
  canAccessPath: (path: string) => boolean;

  // Feature access
  accessibleFeatures: FeatureAccess[];
  adminFeatures: FeatureAccess[];
  userFeatures: FeatureAccess[];

  // Utilities
  getPermissions: () => Permission[];
  getRoleLabel: () => string;
}

/**
 * usePermissions hook
 *
 * Provides centralized permission checking throughout the app.
 */
export function usePermissions(): UsePermissionsResult {
  const { user, isAuthenticated } = useAuthStore();

  // Get current role
  const role = useMemo<UserRole>(() => {
    if (!isAuthenticated || !user) return "guest";
    return user.role || "user";
  }, [isAuthenticated, user]);

  // Role checks
  const isAdmin = role === "admin";
  const isUser = role === "user" || role === "admin";
  const isViewer = role === "viewer" || isUser;
  const isGuest = role === "guest";

  // Get permissions for current role
  const getPermissions = useCallback((): Permission[] => {
    return ROLE_PERMISSIONS[role] || [];
  }, [role]);

  // Check if user has a specific role or higher
  const hasRole = useCallback(
    (requiredRole: UserRole): boolean => {
      return ROLE_HIERARCHY[role] >= ROLE_HIERARCHY[requiredRole];
    },
    [role]
  );

  // Check if user has a specific permission
  const hasPermission = useCallback(
    (permission: Permission): boolean => {
      const permissions = getPermissions();
      return (
        permissions.includes("admin:all") || permissions.includes(permission)
      );
    },
    [getPermissions]
  );

  // Check if user has any of the given permissions
  const hasAnyPermission = useCallback(
    (permissions: Permission[]): boolean => {
      return permissions.some((p) => hasPermission(p));
    },
    [hasPermission]
  );

  // Check if user has all of the given permissions
  const hasAllPermissions = useCallback(
    (permissions: Permission[]): boolean => {
      return permissions.every((p) => hasPermission(p));
    },
    [hasPermission]
  );

  // Check if user can access a feature
  const canAccess = useCallback(
    (feature: FeatureAccess): boolean => {
      if (!hasRole(feature.requiredRole)) return false;
      if (
        feature.requiredPermissions &&
        feature.requiredPermissions.length > 0
      ) {
        return hasAnyPermission(feature.requiredPermissions);
      }
      return true;
    },
    [hasRole, hasAnyPermission]
  );

  // Check if user can access a path
  const canAccessPath = useCallback(
    (path: string): boolean => {
      // Admin paths require admin role
      if (path.startsWith("/admin")) {
        return isAdmin;
      }
      // All other paths accessible to authenticated users
      return isAuthenticated;
    },
    [isAdmin, isAuthenticated]
  );

  // Get accessible features
  const accessibleFeatures = useMemo(() => {
    return APP_FEATURES.filter((f) => canAccess(f));
  }, [canAccess]);

  // Get admin-only features
  const adminFeatures = useMemo(() => {
    return APP_FEATURES.filter((f) => f.isAdminOnly);
  }, []);

  // Get user features (non-admin)
  const userFeatures = useMemo(() => {
    return APP_FEATURES.filter((f) => !f.isAdminOnly);
  }, []);

  // Get role label for display
  const getRoleLabel = useCallback((): string => {
    const labels: Record<UserRole, string> = {
      guest: "Guest",
      viewer: "Viewer",
      user: "User",
      admin: "Administrator",
    };
    return labels[role] || "Unknown";
  }, [role]);

  return {
    user,
    role,
    isAuthenticated,
    isAdmin,
    isUser,
    isViewer,
    isGuest,
    hasRole,
    hasPermission,
    hasAnyPermission,
    hasAllPermissions,
    canAccess,
    canAccessPath,
    accessibleFeatures,
    adminFeatures,
    userFeatures,
    getPermissions,
    getRoleLabel,
  };
}

export default usePermissions;
