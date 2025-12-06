/**
 * useAuth Hook Tests
 *
 * Tests for authentication hook functionality including:
 * - Login/logout operations
 * - Role-based access control
 * - Token refresh
 * - Error handling
 */

import { renderHook, act, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";

// Mock dependencies before importing the hook
vi.mock("react-router-dom", () => ({
  useNavigate: () => vi.fn(),
}));

vi.mock("../../services/api", () => ({
  api: {
    post: vi.fn(),
    get: vi.fn(),
  },
}));

vi.mock("../../store/authStore", () => {
  let mockState = {
    user: null,
    token: null,
    refreshToken: null,
    isAuthenticated: false,
    isLoading: false,
    error: null,
    permissions: [],
  };

  const mockSetUser = vi.fn((user) => {
    mockState.user = user;
    mockState.isAuthenticated = !!user;
  });

  const mockSetTokens = vi.fn((token, refreshToken) => {
    mockState.token = token;
    mockState.refreshToken = refreshToken;
    mockState.isAuthenticated = true;
  });

  const mockSetLoading = vi.fn((loading) => {
    mockState.isLoading = loading;
  });

  const mockSetError = vi.fn((error) => {
    mockState.error = error;
  });

  const mockLogout = vi.fn(() => {
    mockState = {
      user: null,
      token: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      permissions: [],
    };
  });

  return {
    useAuthStore: () => ({
      ...mockState,
      setUser: mockSetUser,
      setTokens: mockSetTokens,
      setLoading: mockSetLoading,
      setError: mockSetError,
      logout: mockLogout,
    }),
    __resetMockState: () => {
      mockState = {
        user: null,
        token: null,
        refreshToken: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
        permissions: [],
      };
    },
    __setMockState: (newState: Partial<typeof mockState>) => {
      mockState = { ...mockState, ...newState };
    },
    __getMockState: () => mockState,
  };
});

import { useAuth } from "../useAuth";
import { api } from "../../services/api";
import { __resetMockState, __setMockState } from "../../store/authStore";

describe("useAuth", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    __resetMockState();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe("login", () => {
    it("should login successfully with valid credentials", async () => {
      const mockUser = {
        id: "user-1",
        email: "test@example.com",
        name: "Test User",
        role: "user",
      };

      const mockResponse = {
        data: {
          access_token: "mock-access-token",
          refresh_token: "mock-refresh-token",
          token_type: "bearer",
          expires_in: 3600,
          user: mockUser,
        },
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useAuth());

      let success: boolean;
      await act(async () => {
        success = await result.current.login({
          email: "test@example.com",
          password: "password123",
        });
      });

      expect(success!).toBe(true);
      expect(api.post).toHaveBeenCalledWith("/auth/login", {
        email: "test@example.com",
        password: "password123",
      });
    });

    it("should handle login failure with invalid credentials", async () => {
      const mockError = {
        response: {
          data: {
            detail: "Invalid email or password",
          },
        },
      };

      vi.mocked(api.post).mockRejectedValueOnce(mockError);

      const { result } = renderHook(() => useAuth());

      let success: boolean;
      await act(async () => {
        success = await result.current.login({
          email: "wrong@example.com",
          password: "wrongpassword",
        });
      });

      expect(success!).toBe(false);
    });

    it("should handle network errors during login", async () => {
      vi.mocked(api.post).mockRejectedValueOnce(new Error("Network error"));

      const { result } = renderHook(() => useAuth());

      let success: boolean;
      await act(async () => {
        success = await result.current.login({
          email: "test@example.com",
          password: "password123",
        });
      });

      expect(success!).toBe(false);
    });
  });

  describe("register", () => {
    it("should register successfully with valid data", async () => {
      const mockUser = {
        id: "user-new",
        email: "newuser@example.com",
        name: "New User",
        role: "user",
      };

      const mockResponse = {
        data: {
          access_token: "mock-access-token",
          refresh_token: "mock-refresh-token",
          token_type: "bearer",
          expires_in: 3600,
          user: mockUser,
        },
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useAuth());

      let success: boolean;
      await act(async () => {
        success = await result.current.register({
          email: "newuser@example.com",
          password: "securepassword123",
          name: "New User",
          invitation_code: "INVITE123",
        });
      });

      expect(success!).toBe(true);
      expect(api.post).toHaveBeenCalledWith("/auth/register", {
        email: "newuser@example.com",
        password: "securepassword123",
        name: "New User",
        invitation_code: "INVITE123",
      });
    });

    it("should handle registration failure", async () => {
      const mockError = {
        response: {
          data: {
            detail: "Email already registered",
          },
        },
      };

      vi.mocked(api.post).mockRejectedValueOnce(mockError);

      const { result } = renderHook(() => useAuth());

      let success: boolean;
      await act(async () => {
        success = await result.current.register({
          email: "existing@example.com",
          password: "password123",
          name: "Existing User",
          invitation_code: "INVITE123",
        });
      });

      expect(success!).toBe(false);
    });
  });

  describe("logout", () => {
    it("should clear auth state on logout", async () => {
      vi.mocked(api.post).mockResolvedValueOnce({ data: {} });

      __setMockState({
        user: {
          id: "1",
          email: "test@example.com",
          name: "Test",
          role: "user",
        },
        isAuthenticated: true,
        token: "mock-token",
      });

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        await result.current.logout();
      });

      expect(api.post).toHaveBeenCalledWith("/auth/logout");
    });

    it("should handle logout API failure gracefully", async () => {
      vi.mocked(api.post).mockRejectedValueOnce(new Error("API Error"));

      __setMockState({
        user: {
          id: "1",
          email: "test@example.com",
          name: "Test",
          role: "user",
        },
        isAuthenticated: true,
      });

      const { result } = renderHook(() => useAuth());

      // Should not throw even if API fails
      await act(async () => {
        await result.current.logout();
      });

      // Logout should still complete
      expect(api.post).toHaveBeenCalled();
    });
  });

  describe("hasRole", () => {
    it("should return true when user has the specified role", () => {
      __setMockState({
        user: {
          id: "1",
          email: "admin@example.com",
          name: "Admin",
          role: "admin",
        },
        isAuthenticated: true,
      });

      const { result } = renderHook(() => useAuth());

      expect(result.current.hasRole("admin")).toBe(true);
      expect(result.current.hasRole("user")).toBe(false);
    });

    it("should return true when user has one of multiple roles", () => {
      __setMockState({
        user: {
          id: "1",
          email: "user@example.com",
          name: "User",
          role: "user",
        },
        isAuthenticated: true,
      });

      const { result } = renderHook(() => useAuth());

      expect(result.current.hasRole(["admin", "user"])).toBe(true);
      expect(result.current.hasRole(["admin", "viewer"])).toBe(false);
    });

    it("should return false when user is not authenticated", () => {
      __setMockState({
        user: null,
        isAuthenticated: false,
      });

      const { result } = renderHook(() => useAuth());

      expect(result.current.hasRole("admin")).toBe(false);
      expect(result.current.hasRole("user")).toBe(false);
    });
  });

  describe("isAdmin", () => {
    it("should return true for admin users", () => {
      __setMockState({
        user: {
          id: "1",
          email: "admin@example.com",
          name: "Admin",
          role: "admin",
        },
        isAuthenticated: true,
      });

      const { result } = renderHook(() => useAuth());

      expect(result.current.isAdmin()).toBe(true);
    });

    it("should return false for non-admin users", () => {
      __setMockState({
        user: {
          id: "1",
          email: "user@example.com",
          name: "User",
          role: "user",
        },
        isAuthenticated: true,
      });

      const { result } = renderHook(() => useAuth());

      expect(result.current.isAdmin()).toBe(false);
    });

    it("should return false when not authenticated", () => {
      __setMockState({
        user: null,
        isAuthenticated: false,
      });

      const { result } = renderHook(() => useAuth());

      expect(result.current.isAdmin()).toBe(false);
    });
  });

  describe("authentication state", () => {
    it("should expose user data when authenticated", () => {
      const mockUser = {
        id: "user-1",
        email: "test@example.com",
        name: "Test User",
        role: "user",
      };

      __setMockState({
        user: mockUser,
        isAuthenticated: true,
      });

      const { result } = renderHook(() => useAuth());

      expect(result.current.user).toEqual(mockUser);
      expect(result.current.isAuthenticated).toBe(true);
    });

    it("should have null user when not authenticated", () => {
      __setMockState({
        user: null,
        isAuthenticated: false,
      });

      const { result } = renderHook(() => useAuth());

      expect(result.current.user).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
    });
  });
});
