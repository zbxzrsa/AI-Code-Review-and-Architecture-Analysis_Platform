/**
 * ProtectedRoute Component Tests
 * 
 * Tests for authentication and authorization routing
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { ProtectedRoute, PublicRoute, AdminRoute, UserRoute } from '../ProtectedRoute';

// Mock auth store
const mockUseAuthStore = vi.fn();

vi.mock('../../../store/authStore', () => ({
  useAuthStore: () => mockUseAuthStore(),
}));

// Mock i18n
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key,
  }),
}));

// Mock error logging
vi.mock('../../../services/errorLogging', () => ({
  errorLoggingService: {
    log: vi.fn(),
  },
  ErrorCategory: {
    AUTHORIZATION: 'authorization',
  },
}));

// Test wrapper
const createWrapper = (initialEntries: string[] = ['/protected']) => {
  return ({ children }: { children: React.ReactNode }) => (
    <ConfigProvider>
      <MemoryRouter initialEntries={initialEntries}>
        {children}
      </MemoryRouter>
    </ConfigProvider>
  );
};

// Test components
const ProtectedContent = () => <div data-testid="protected">Protected Content</div>;
const LoginPage = () => <div data-testid="login">Login Page</div>;
const ForbiddenPage = () => <div data-testid="forbidden">Forbidden</div>;

describe('ProtectedRoute', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows loading state while checking authentication', () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: false,
      user: null,
      isLoading: true,
    });

    render(
      <ProtectedRoute>
        <ProtectedContent />
      </ProtectedRoute>,
      { wrapper: createWrapper() }
    );

    expect(screen.getByText('Checking authentication...')).toBeInTheDocument();
  });

  it('redirects to login when not authenticated', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: false,
      user: null,
      isLoading: false,
    });

    render(
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/protected" element={
          <ProtectedRoute>
            <ProtectedContent />
          </ProtectedRoute>
        } />
      </Routes>,
      { wrapper: createWrapper(['/protected']) }
    );

    await waitFor(() => {
      expect(screen.getByTestId('login')).toBeInTheDocument();
    });
  });

  it('renders children when authenticated', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: { id: '1', role: 'user', permissions: [] },
      isLoading: false,
    });

    render(
      <ProtectedRoute>
        <ProtectedContent />
      </ProtectedRoute>,
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(screen.getByTestId('protected')).toBeInTheDocument();
    });
  });

  it('shows 403 when user lacks required role', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: { id: '1', role: 'user', permissions: [] },
      isLoading: false,
    });

    render(
      <ProtectedRoute requiredRoles={['admin']}>
        <ProtectedContent />
      </ProtectedRoute>,
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(screen.getByText('Access Denied')).toBeInTheDocument();
    });
  });

  it('allows access when user has required role', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: { id: '1', role: 'admin', permissions: [] },
      isLoading: false,
    });

    render(
      <ProtectedRoute requiredRoles={['admin']}>
        <ProtectedContent />
      </ProtectedRoute>,
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(screen.getByTestId('protected')).toBeInTheDocument();
    });
  });

  it('shows 403 when user lacks required permissions', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: { id: '1', role: 'user', permissions: ['read:projects'] },
      isLoading: false,
    });

    render(
      <ProtectedRoute requiredPermissions={['write:projects', 'delete:projects']} requireAll={true}>
        <ProtectedContent />
      </ProtectedRoute>,
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(screen.getByText('Access Denied')).toBeInTheDocument();
    });
  });

  it('allows access when user has any required permission (requireAll=false)', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: { id: '1', role: 'user', permissions: ['read:projects'] },
      isLoading: false,
    });

    render(
      <ProtectedRoute requiredPermissions={['read:projects', 'write:projects']} requireAll={false}>
        <ProtectedContent />
      </ProtectedRoute>,
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(screen.getByTestId('protected')).toBeInTheDocument();
    });
  });

  it('allows admin:all permission to bypass checks', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: { id: '1', role: 'admin', permissions: ['admin:all'] },
      isLoading: false,
    });

    render(
      <ProtectedRoute requiredPermissions={['delete:users']} requireAll={true}>
        <ProtectedContent />
      </ProtectedRoute>,
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(screen.getByTestId('protected')).toBeInTheDocument();
    });
  });

  it('renders custom fallback during loading', () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: false,
      user: null,
      isLoading: true,
    });

    const customFallback = <div data-testid="custom-loading">Custom Loading</div>;

    render(
      <ProtectedRoute fallback={customFallback}>
        <ProtectedContent />
      </ProtectedRoute>,
      { wrapper: createWrapper() }
    );

    expect(screen.getByTestId('custom-loading')).toBeInTheDocument();
  });

  it('calls onAccessDenied when access is denied', async () => {
    const onAccessDenied = vi.fn();
    
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: { id: '1', role: 'user', permissions: [] },
      isLoading: false,
    });

    render(
      <ProtectedRoute requiredRoles={['admin']} onAccessDenied={onAccessDenied}>
        <ProtectedContent />
      </ProtectedRoute>,
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(onAccessDenied).toHaveBeenCalled();
    });
  });
});

describe('AdminRoute', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('allows admin users', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: { id: '1', role: 'admin', permissions: [] },
      isLoading: false,
    });

    render(
      <AdminRoute>
        <ProtectedContent />
      </AdminRoute>,
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(screen.getByTestId('protected')).toBeInTheDocument();
    });
  });

  it('denies non-admin users', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: { id: '1', role: 'user', permissions: [] },
      isLoading: false,
    });

    render(
      <AdminRoute>
        <ProtectedContent />
      </AdminRoute>,
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(screen.getByText('Access Denied')).toBeInTheDocument();
    });
  });
});

describe('PublicRoute', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders children when not authenticated', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: false,
      user: null,
      isLoading: false,
    });

    render(
      <PublicRoute>
        <LoginPage />
      </PublicRoute>,
      { wrapper: createWrapper(['/login']) }
    );

    await waitFor(() => {
      expect(screen.getByTestId('login')).toBeInTheDocument();
    });
  });

  it('redirects to dashboard when authenticated', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: { id: '1', role: 'user' },
      isLoading: false,
    });

    render(
      <Routes>
        <Route path="/dashboard" element={<div data-testid="dashboard">Dashboard</div>} />
        <Route path="/login" element={
          <PublicRoute>
            <LoginPage />
          </PublicRoute>
        } />
      </Routes>,
      { wrapper: createWrapper(['/login']) }
    );

    await waitFor(() => {
      expect(screen.getByTestId('dashboard')).toBeInTheDocument();
    });
  });
});

describe('ProtectedRoute Accessibility', () => {
  it('provides accessible error message', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: { id: '1', role: 'user', permissions: [] },
      isLoading: false,
    });

    render(
      <ProtectedRoute requiredRoles={['admin']}>
        <ProtectedContent />
      </ProtectedRoute>,
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      const heading = screen.getByRole('heading', { name: /access denied/i });
      expect(heading).toBeInTheDocument();
    });
  });

  it('provides navigation buttons for recovery', async () => {
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: { id: '1', role: 'user', permissions: [] },
      isLoading: false,
    });

    render(
      <ProtectedRoute requiredRoles={['admin']}>
        <ProtectedContent />
      </ProtectedRoute>,
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /go back/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /go to dashboard/i })).toBeInTheDocument();
    });
  });
});
