/**
 * Permission Hook Tests
 *
 * Comprehensive tests for the usePermissions hook covering:
 * - Role-based access control
 * - Permission checks
 * - Admin-only feature restrictions
 * - Boundary cases
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { usePermissions, APP_FEATURES } from "../usePermissions";

// Mock user type
type MockUser = {
  id: string;
  email: string;
  name: string;
  role: "admin" | "user" | "viewer" | "guest";
  avatar: string | null;
};

// Mock the auth store
const mockUser: MockUser = {
  id: "1",
  email: "test@example.com",
  name: "Test User",
  role: "user",
  avatar: null,
};

const mockAdminUser: MockUser = {
  id: "1",
  email: "admin@example.com",
  name: "Admin User",
  role: "admin",
  avatar: null,
};

const mockViewerUser: MockUser = {
  id: "1",
  email: "viewer@example.com",
  name: "Viewer User",
  role: "viewer",
  avatar: null,
};

const mockGuestUser: MockUser = {
  id: "1",
  email: "guest@example.com",
  name: "Guest User",
  role: "guest",
  avatar: null,
};

let mockAuthState: { user: MockUser | null; isAuthenticated: boolean } = {
  user: mockUser,
  isAuthenticated: true,
};

vi.mock("../../store/authStore", () => ({
  useAuthStore: () => mockAuthState,
}));

describe("usePermissions Hook", () => {
  beforeEach(() => {
    mockAuthState.user = mockUser;
    mockAuthState.isAuthenticated = true;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // ============================================
  // Role Detection Tests
  // ============================================
  describe("Role Detection", () => {
    it("correctly identifies admin role", () => {
      mockAuthState.user = mockAdminUser;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.isAdmin).toBe(true);
      expect(result.current.isUser).toBe(true);
      expect(result.current.isViewer).toBe(true);
      expect(result.current.role).toBe("admin");
    });

    it("correctly identifies user role", () => {
      mockAuthState.user = mockUser;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.isAdmin).toBe(false);
      expect(result.current.isUser).toBe(true);
      expect(result.current.isViewer).toBe(true);
      expect(result.current.role).toBe("user");
    });

    it("correctly identifies viewer role", () => {
      mockAuthState.user = mockViewerUser;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.isAdmin).toBe(false);
      expect(result.current.isUser).toBe(false);
      expect(result.current.isViewer).toBe(true);
      expect(result.current.role).toBe("viewer");
    });

    it("correctly identifies guest role when not authenticated", () => {
      mockAuthState.user = null as any;
      mockAuthState.isAuthenticated = false;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.isAdmin).toBe(false);
      expect(result.current.isUser).toBe(false);
      expect(result.current.isViewer).toBe(false);
      expect(result.current.isGuest).toBe(true);
      expect(result.current.role).toBe("guest");
    });
  });

  // ============================================
  // Admin-Only Feature Access Tests
  // ============================================
  describe("Admin-Only Feature Access", () => {
    it("admin can access all admin features", () => {
      mockAuthState.user = mockAdminUser;
      const { result } = renderHook(() => usePermissions());

      const adminFeatures = APP_FEATURES.filter((f) => f.isAdminOnly);

      adminFeatures.forEach((feature) => {
        expect(result.current.canAccess(feature)).toBe(true);
      });
    });

    it("regular user cannot access admin features", () => {
      mockAuthState.user = mockUser;
      const { result } = renderHook(() => usePermissions());

      const adminFeatures = APP_FEATURES.filter((f) => f.isAdminOnly);

      adminFeatures.forEach((feature) => {
        expect(result.current.canAccess(feature)).toBe(false);
      });
    });

    it("viewer cannot access admin features", () => {
      mockAuthState.user = mockViewerUser;
      const { result } = renderHook(() => usePermissions());

      const adminFeatures = APP_FEATURES.filter((f) => f.isAdminOnly);

      adminFeatures.forEach((feature) => {
        expect(result.current.canAccess(feature)).toBe(false);
      });
    });

    it("guest cannot access admin features", () => {
      mockAuthState.user = mockGuestUser;
      mockAuthState.isAuthenticated = false;
      const { result } = renderHook(() => usePermissions());

      const adminFeatures = APP_FEATURES.filter((f) => f.isAdminOnly);

      adminFeatures.forEach((feature) => {
        expect(result.current.canAccess(feature)).toBe(false);
      });
    });
  });

  // ============================================
  // Non-Admin Feature Access Tests
  // ============================================
  describe("Non-Admin Feature Access", () => {
    it("admin can access all user features", () => {
      mockAuthState.user = mockAdminUser;
      const { result } = renderHook(() => usePermissions());

      const userFeatures = APP_FEATURES.filter((f) => !f.isAdminOnly && f.requiredRole === "user");

      userFeatures.forEach((feature) => {
        expect(result.current.canAccess(feature)).toBe(true);
      });
    });

    it("regular user can access user features", () => {
      mockAuthState.user = mockUser;
      const { result } = renderHook(() => usePermissions());

      const userFeatures = APP_FEATURES.filter((f) => !f.isAdminOnly && f.requiredRole === "user");

      userFeatures.forEach((feature) => {
        expect(result.current.canAccess(feature)).toBe(true);
      });
    });

    it("viewer can access viewer features", () => {
      mockAuthState.user = mockViewerUser;
      const { result } = renderHook(() => usePermissions());

      const viewerFeatures = APP_FEATURES.filter((f) => f.requiredRole === "viewer");

      viewerFeatures.forEach((feature) => {
        expect(result.current.canAccess(feature)).toBe(true);
      });
    });
  });

  // ============================================
  // Path-Based Access Control Tests
  // ============================================
  describe("Path-Based Access Control", () => {
    it("admin can access admin paths", () => {
      mockAuthState.user = mockAdminUser;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.canAccessPath("/admin/users")).toBe(true);
      expect(result.current.canAccessPath("/admin/ai-models")).toBe(true);
      expect(result.current.canAccessPath("/admin/security")).toBe(true);
      expect(result.current.canAccessPath("/admin/audit")).toBe(true);
    });

    it("regular user cannot access admin paths via direct URL", () => {
      mockAuthState.user = mockUser;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.canAccessPath("/admin/users")).toBe(false);
      expect(result.current.canAccessPath("/admin/ai-models")).toBe(false);
      expect(result.current.canAccessPath("/admin/security")).toBe(false);
      expect(result.current.canAccessPath("/admin/audit")).toBe(false);
    });

    it("viewer cannot access admin paths via direct URL", () => {
      mockAuthState.user = mockViewerUser;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.canAccessPath("/admin/users")).toBe(false);
      expect(result.current.canAccessPath("/admin/providers")).toBe(false);
    });

    it("all authenticated users can access non-admin paths", () => {
      const testPaths = ["/dashboard", "/projects", "/review", "/settings", "/profile"];

      // Test for admin
      mockAuthState.user = mockAdminUser;
      let { result } = renderHook(() => usePermissions());
      testPaths.forEach((path) => {
        expect(result.current.canAccessPath(path)).toBe(true);
      });

      // Test for user
      mockAuthState.user = mockUser;
      result = renderHook(() => usePermissions()).result;
      testPaths.forEach((path) => {
        expect(result.current.canAccessPath(path)).toBe(true);
      });

      // Test for viewer
      mockAuthState.user = mockViewerUser;
      result = renderHook(() => usePermissions()).result;
      testPaths.forEach((path) => {
        expect(result.current.canAccessPath(path)).toBe(true);
      });
    });

    it("unauthenticated users cannot access any protected paths", () => {
      mockAuthState.user = null as any;
      mockAuthState.isAuthenticated = false;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.canAccessPath("/dashboard")).toBe(false);
      expect(result.current.canAccessPath("/admin/users")).toBe(false);
      expect(result.current.canAccessPath("/projects")).toBe(false);
    });
  });

  // ============================================
  // Permission Checks
  // ============================================
  describe("Permission Checks", () => {
    it("admin has all permissions", () => {
      mockAuthState.user = mockAdminUser;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.hasPermission("admin:all")).toBe(true);
      expect(result.current.hasPermission("read:projects")).toBe(true);
      expect(result.current.hasPermission("write:projects")).toBe(true);
      expect(result.current.hasPermission("delete:projects")).toBe(true);
    });

    it("user has limited permissions", () => {
      mockAuthState.user = mockUser;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.hasPermission("admin:all")).toBe(false);
      expect(result.current.hasPermission("read:projects")).toBe(true);
      expect(result.current.hasPermission("write:projects")).toBe(true);
      expect(result.current.hasPermission("delete:projects")).toBe(false);
    });

    it("viewer has read-only permissions", () => {
      mockAuthState.user = mockViewerUser;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.hasPermission("read:projects")).toBe(true);
      expect(result.current.hasPermission("write:projects")).toBe(false);
      expect(result.current.hasPermission("admin:all")).toBe(false);
    });

    it("guest has no permissions", () => {
      mockAuthState.user = mockGuestUser;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.hasPermission("read:projects")).toBe(false);
      expect(result.current.hasPermission("admin:all")).toBe(false);
    });
  });

  // ============================================
  // Boundary Tests
  // ============================================
  describe("Boundary Tests", () => {
    it("handles null user gracefully", () => {
      mockAuthState.user = null as any;
      mockAuthState.isAuthenticated = false;

      const { result } = renderHook(() => usePermissions());

      expect(result.current.role).toBe("guest");
      expect(result.current.isAdmin).toBe(false);
      expect(result.current.accessibleFeatures).toEqual([]);
    });

    it("handles undefined role gracefully", () => {
      mockAuthState.user = { ...mockUser, role: undefined as any };

      const { result } = renderHook(() => usePermissions());

      // Should default to 'user' when role is undefined but authenticated
      expect(result.current.role).toBe("user");
    });

    it("handles hasAnyPermission correctly", () => {
      mockAuthState.user = mockUser;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.hasAnyPermission(["read:projects", "admin:all"])).toBe(true);
      expect(result.current.hasAnyPermission(["admin:all", "admin:users"])).toBe(false);
    });

    it("handles hasAllPermissions correctly", () => {
      mockAuthState.user = mockUser;
      const { result } = renderHook(() => usePermissions());

      expect(result.current.hasAllPermissions(["read:projects", "write:projects"])).toBe(true);
      expect(result.current.hasAllPermissions(["read:projects", "admin:all"])).toBe(false);
    });

    it("returns correct role label", () => {
      mockAuthState.user = mockAdminUser;
      let { result } = renderHook(() => usePermissions());
      expect(result.current.getRoleLabel()).toBe("Administrator");

      mockAuthState.user = mockUser;
      result = renderHook(() => usePermissions()).result;
      expect(result.current.getRoleLabel()).toBe("User");

      mockAuthState.user = mockViewerUser;
      result = renderHook(() => usePermissions()).result;
      expect(result.current.getRoleLabel()).toBe("Viewer");
    });

    it("adminFeatures only contains admin-only features", () => {
      const { result } = renderHook(() => usePermissions());

      result.current.adminFeatures.forEach((feature) => {
        expect(feature.isAdminOnly).toBe(true);
      });
    });

    it("userFeatures does not contain admin-only features", () => {
      const { result } = renderHook(() => usePermissions());

      result.current.userFeatures.forEach((feature) => {
        expect(feature.isAdminOnly).toBe(false);
      });
    });
  });

  // ============================================
  // Admin Routes Direct URL Access Tests
  // ============================================
  describe("Direct URL Access Prevention", () => {
    const adminPaths = [
      "/admin/users",
      "/admin/ai-models",
      "/admin/providers",
      "/admin/security",
      "/admin/audit",
      "/admin/health",
      "/admin/performance",
      "/admin/experiments",
      "/admin/evolution",
      "/admin/auto-fix",
      "/admin/vulnerabilities",
      "/admin/model-testing",
      "/admin/learning",
      "/admin/model-comparison",
      "/admin/quality",
      "/admin/ml-promotion",
      "/admin/version-comparison",
    ];

    it("blocks all admin paths for regular users", () => {
      mockAuthState.user = mockUser;
      const { result } = renderHook(() => usePermissions());

      adminPaths.forEach((path) => {
        expect(result.current.canAccessPath(path)).toBe(false);
      });
    });

    it("allows all admin paths for admin users", () => {
      mockAuthState.user = mockAdminUser;
      const { result } = renderHook(() => usePermissions());

      adminPaths.forEach((path) => {
        expect(result.current.canAccessPath(path)).toBe(true);
      });
    });
  });
});
