/**
 * Auth Store Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';

// Mock API service
vi.mock('@/services/api', () => ({
  apiService: {
    auth: {
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
      getCurrentUser: vi.fn(),
      refreshToken: vi.fn(),
    },
  },
}));

// Mock security service
vi.mock('@/services/security', () => ({
  csrfToken: {
    fetch: vi.fn().mockResolvedValue('test-csrf-token'),
  },
  sessionSecurity: {
    clearSession: vi.fn(),
  },
}));

import { useAuthStore } from '@/store/authStore';
import { apiService } from '@/services/api';

describe('AuthStore', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset store state
    useAuthStore.setState({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      twoFactorRequired: false,
      twoFactorSessionId: null,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Initial State', () => {
    it('has correct initial state', () => {
      const { result } = renderHook(() => useAuthStore());
      
      expect(result.current.user).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).toBeNull();
    });
  });

  describe('Login', () => {
    it('sets loading state during login', async () => {
      const { result } = renderHook(() => useAuthStore());
      
      (apiService.auth.login as ReturnType<typeof vi.fn>).mockImplementation(
        () => new Promise(resolve => setTimeout(resolve, 100))
      );

      act(() => {
        result.current.login('test@example.com', 'password');
      });

      expect(result.current.isLoading).toBe(true);
    });

    it('sets user on successful login', async () => {
      const mockUser = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        role: 'user',
      };

      (apiService.auth.login as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: { user: mockUser },
      });

      const { result } = renderHook(() => useAuthStore());

      await act(async () => {
        await result.current.login('test@example.com', 'password');
      });

      expect(result.current.user).toEqual(mockUser);
      expect(result.current.isAuthenticated).toBe(true);
      expect(result.current.isLoading).toBe(false);
    });

    it('sets error on failed login', async () => {
      (apiService.auth.login as ReturnType<typeof vi.fn>).mockRejectedValue(
        new Error('Invalid credentials')
      );

      const { result } = renderHook(() => useAuthStore());

      await act(async () => {
        await result.current.login('test@example.com', 'wrong-password');
      });

      expect(result.current.error).toBe('Invalid credentials');
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.isLoading).toBe(false);
    });

    it('handles 2FA required response', async () => {
      (apiService.auth.login as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: {
          requires_two_factor: true,
          session_id: 'temp-session-123',
        },
      });

      const { result } = renderHook(() => useAuthStore());

      await act(async () => {
        await result.current.login('test@example.com', 'password');
      });

      expect(result.current.twoFactorRequired).toBe(true);
      expect(result.current.twoFactorSessionId).toBe('temp-session-123');
      expect(result.current.isAuthenticated).toBe(false);
    });
  });

  describe('Register', () => {
    it('registers new user successfully', async () => {
      const mockUser = {
        id: '123',
        email: 'new@example.com',
        name: 'New User',
      };

      (apiService.auth.register as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: { user: mockUser },
      });

      const { result } = renderHook(() => useAuthStore());

      await act(async () => {
        await result.current.register('new@example.com', 'password', 'New User');
      });

      expect(result.current.user).toEqual(mockUser);
      expect(result.current.isAuthenticated).toBe(true);
    });

    it('handles registration error', async () => {
      (apiService.auth.register as ReturnType<typeof vi.fn>).mockRejectedValue(
        new Error('Email already exists')
      );

      const { result } = renderHook(() => useAuthStore());

      await act(async () => {
        await result.current.register('existing@example.com', 'password', 'User');
      });

      expect(result.current.error).toBe('Email already exists');
      expect(result.current.isAuthenticated).toBe(false);
    });
  });

  describe('Logout', () => {
    it('clears user state on logout', async () => {
      // Set initial authenticated state
      useAuthStore.setState({
        user: { id: '123', email: 'test@example.com', name: 'Test', role: 'user' },
        isAuthenticated: true,
      });

      (apiService.auth.logout as ReturnType<typeof vi.fn>).mockResolvedValue({});

      const { result } = renderHook(() => useAuthStore());

      await act(async () => {
        await result.current.logout();
      });

      expect(result.current.user).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
    });

    it('clears session data on logout', async () => {
      const { sessionSecurity } = await import('@/services/security');
      
      useAuthStore.setState({
        user: { id: '123', email: 'test@example.com', name: 'Test', role: 'user' },
        isAuthenticated: true,
      });

      (apiService.auth.logout as ReturnType<typeof vi.fn>).mockResolvedValue({});

      const { result } = renderHook(() => useAuthStore());

      await act(async () => {
        await result.current.logout();
      });

      expect(sessionSecurity.clearSession).toHaveBeenCalled();
    });
  });

  describe('Check Auth', () => {
    it('fetches current user on checkAuth', async () => {
      const mockUser = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
      };

      (apiService.auth.getCurrentUser as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: mockUser,
      });

      const { result } = renderHook(() => useAuthStore());

      await act(async () => {
        await result.current.checkAuth();
      });

      expect(result.current.user).toEqual(mockUser);
      expect(result.current.isAuthenticated).toBe(true);
    });

    it('sets not authenticated when no session', async () => {
      (apiService.auth.getCurrentUser as ReturnType<typeof vi.fn>).mockRejectedValue(
        new Error('Unauthorized')
      );

      const { result } = renderHook(() => useAuthStore());

      await act(async () => {
        await result.current.checkAuth();
      });

      expect(result.current.user).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
    });
  });

  describe('Two-Factor Authentication', () => {
    it('verifies 2FA code successfully', async () => {
      const mockUser = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
      };

      useAuthStore.setState({
        twoFactorRequired: true,
        twoFactorSessionId: 'temp-session-123',
      });

      (apiService.auth.verify2FA as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: { user: mockUser },
      });

      const { result } = renderHook(() => useAuthStore());

      await act(async () => {
        await result.current.verify2FA('123456');
      });

      expect(result.current.user).toEqual(mockUser);
      expect(result.current.isAuthenticated).toBe(true);
      expect(result.current.twoFactorRequired).toBe(false);
    });

    it('handles invalid 2FA code', async () => {
      useAuthStore.setState({
        twoFactorRequired: true,
        twoFactorSessionId: 'temp-session-123',
      });

      (apiService.auth.verify2FA as ReturnType<typeof vi.fn>).mockRejectedValue(
        new Error('Invalid code')
      );

      const { result } = renderHook(() => useAuthStore());

      await act(async () => {
        await result.current.verify2FA('000000');
      });

      expect(result.current.error).toBe('Invalid code');
      expect(result.current.isAuthenticated).toBe(false);
    });
  });

  describe('User Updates', () => {
    it('updates user data', () => {
      useAuthStore.setState({
        user: { id: '123', email: 'test@example.com', name: 'Old Name', role: 'user' },
        isAuthenticated: true,
      });

      const { result } = renderHook(() => useAuthStore());

      act(() => {
        result.current.updateUser({ name: 'New Name' });
      });

      expect(result.current.user?.name).toBe('New Name');
    });
  });

  describe('Error Handling', () => {
    it('clears error on new action', async () => {
      useAuthStore.setState({ error: 'Previous error' });

      (apiService.auth.login as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: { user: { id: '123', email: 'test@example.com' } },
      });

      const { result } = renderHook(() => useAuthStore());

      await act(async () => {
        await result.current.login('test@example.com', 'password');
      });

      expect(result.current.error).toBeNull();
    });

    it('can manually clear error', () => {
      useAuthStore.setState({ error: 'Some error' });

      const { result } = renderHook(() => useAuthStore());

      act(() => {
        result.current.clearError();
      });

      expect(result.current.error).toBeNull();
    });
  });

  describe('Permissions', () => {
    it('checks user permissions', () => {
      useAuthStore.setState({
        user: {
          id: '123',
          email: 'test@example.com',
          name: 'Test',
          role: 'admin',
          permissions: ['read', 'write', 'delete'],
        },
        isAuthenticated: true,
      });

      const { result } = renderHook(() => useAuthStore());

      expect(result.current.hasPermission('read')).toBe(true);
      expect(result.current.hasPermission('execute')).toBe(false);
    });

    it('checks user role', () => {
      useAuthStore.setState({
        user: { id: '123', email: 'test@example.com', name: 'Test', role: 'admin' },
        isAuthenticated: true,
      });

      const { result } = renderHook(() => useAuthStore());

      expect(result.current.hasRole('admin')).toBe(true);
      expect(result.current.hasRole('user')).toBe(false);
    });
  });
});
