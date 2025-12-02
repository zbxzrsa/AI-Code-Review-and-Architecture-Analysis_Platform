import { useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { api } from '../services/api';

interface LoginCredentials {
  email: string;
  password: string;
  invitation_code?: string;
}

interface RegisterData {
  email: string;
  password: string;
  name: string;
  invitation_code: string;
}

interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: {
    id: string;
    email: string;
    name: string;
    role: string;
  };
}

export function useAuth() {
  const navigate = useNavigate();
  const {
    user,
    token,
    refreshToken,
    isAuthenticated,
    isLoading,
    error,
    setUser,
    setTokens,
    setLoading,
    setError,
    logout: clearAuth
  } = useAuthStore();

  // Check if token is expired
  const isTokenExpired = useCallback((token: string): boolean => {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      return payload.exp * 1000 < Date.now();
    } catch {
      return true;
    }
  }, []);

  // Refresh access token
  const refreshAccessToken = useCallback(async (): Promise<string | null> => {
    if (!refreshToken) return null;

    try {
      const response = await api.post<AuthResponse>('/auth/refresh', {
        refresh_token: refreshToken
      });

      const { access_token, refresh_token: newRefreshToken } = response.data;
      setTokens(access_token, newRefreshToken);
      return access_token;
    } catch (error) {
      clearAuth();
      return null;
    }
  }, [refreshToken, setTokens, clearAuth]);

  // Login
  const login = useCallback(async (credentials: LoginCredentials): Promise<boolean> => {
    setLoading(true);
    setError(null);

    try {
      const response = await api.post<AuthResponse>('/auth/login', credentials);
      const { access_token, refresh_token, user } = response.data;

      setTokens(access_token, refresh_token);
      setUser(user);

      return true;
    } catch (error: any) {
      const message = error.response?.data?.detail || 'Login failed';
      setError(message);
      return false;
    } finally {
      setLoading(false);
    }
  }, [setLoading, setError, setTokens, setUser]);

  // Register
  const register = useCallback(async (data: RegisterData): Promise<boolean> => {
    setLoading(true);
    setError(null);

    try {
      const response = await api.post<AuthResponse>('/auth/register', data);
      const { access_token, refresh_token, user } = response.data;

      setTokens(access_token, refresh_token);
      setUser(user);

      return true;
    } catch (error: any) {
      const message = error.response?.data?.detail || 'Registration failed';
      setError(message);
      return false;
    } finally {
      setLoading(false);
    }
  }, [setLoading, setError, setTokens, setUser]);

  // Logout
  const logout = useCallback(async (): Promise<void> => {
    try {
      await api.post('/auth/logout');
    } catch {
      // Ignore logout errors
    } finally {
      clearAuth();
      navigate('/login');
    }
  }, [clearAuth, navigate]);

  // Get current user
  const fetchCurrentUser = useCallback(async (): Promise<void> => {
    if (!token) return;

    setLoading(true);

    try {
      const response = await api.get('/auth/me');
      setUser(response.data);
    } catch (error) {
      // Token might be invalid
      const newToken = await refreshAccessToken();
      if (!newToken) {
        clearAuth();
      }
    } finally {
      setLoading(false);
    }
  }, [token, setLoading, setUser, refreshAccessToken, clearAuth]);

  // Check authentication on mount
  useEffect(() => {
    if (token && !user) {
      fetchCurrentUser();
    }
  }, [token, user, fetchCurrentUser]);

  // Auto-refresh token before expiry
  useEffect(() => {
    if (!token) return;

    const checkAndRefresh = async () => {
      if (isTokenExpired(token)) {
        await refreshAccessToken();
      }
    };

    // Check every minute
    const interval = setInterval(checkAndRefresh, 60 * 1000);

    return () => clearInterval(interval);
  }, [token, isTokenExpired, refreshAccessToken]);

  // Check if user has specific role
  const hasRole = useCallback((role: string | string[]): boolean => {
    if (!user) return false;

    const roles = Array.isArray(role) ? role : [role];
    return roles.includes(user.role);
  }, [user]);

  // Check if user is admin
  const isAdmin = useCallback((): boolean => {
    return hasRole('admin');
  }, [hasRole]);

  return {
    user,
    token,
    isAuthenticated,
    isLoading,
    error,
    login,
    register,
    logout,
    refreshAccessToken,
    fetchCurrentUser,
    hasRole,
    isAdmin
  };
}

export default useAuth;
