/**
 * User Management Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import { UserManagement } from '../UserManagement';

// Mock the hooks
vi.mock('../../../hooks/useAdmin', () => ({
  useAdminUsers: vi.fn(() => ({
    data: {
      items: [
        {
          id: '1',
          username: 'testuser',
          email: 'test@example.com',
          name: 'Test User',
          role: 'analyst',
          status: 'active',
          joinedAt: '2024-01-01T00:00:00Z',
          lastLoginAt: '2024-01-15T12:00:00Z',
          emailVerified: true,
          twoFactorEnabled: false,
          projectCount: 5,
          analysisCount: 100,
        },
      ],
      total: 1,
    },
    isLoading: false,
    refetch: vi.fn(),
    isFetching: false,
  })),
  useUserStats: vi.fn(() => ({
    data: {
      total: 100,
      active: 85,
      inactive: 10,
      suspended: 5,
      recentlyJoined: 12,
      byRole: [
        { role: 'admin', count: 5 },
        { role: 'analyst', count: 50 },
        { role: 'viewer', count: 45 },
      ],
      activityTrend: [],
    },
    isLoading: false,
  })),
  useUpdateUser: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useDeleteUser: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useSuspendUser: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useReactivateUser: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useResetUserPassword: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useResendWelcome: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useBulkUserAction: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useImportUsers: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useExportUsers: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
}));

// Mock the store
vi.mock('../../../store/adminStore', async () => {
  const actual = await vi.importActual('../../../store/adminStore');
  return {
    ...actual,
    useAdminStore: vi.fn(() => ({
      userFilters: { search: '', role: 'all', status: 'all' },
      userPagination: { page: 1, pageSize: 20, total: 0 },
      userSort: { field: 'joinedAt', order: 'desc' },
      selectedUserIds: [],
      setUserFilters: vi.fn(),
      resetUserFilters: vi.fn(),
      setUserPagination: vi.fn(),
      setUserSort: vi.fn(),
      selectUser: vi.fn(),
      deselectUser: vi.fn(),
      selectAllUsers: vi.fn(),
      clearUserSelection: vi.fn(),
    })),
  };
});

// Mock i18n
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key,
  }),
}));

// Test wrapper
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider>
        <BrowserRouter>
          {children}
        </BrowserRouter>
      </ConfigProvider>
    </QueryClientProvider>
  );
};

describe('UserManagement Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the page title', async () => {
    render(<UserManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('User Management')).toBeInTheDocument();
    });
  });

  it('displays user statistics', async () => {
    render(<UserManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Total Users')).toBeInTheDocument();
      expect(screen.getByText('Active Users')).toBeInTheDocument();
      expect(screen.getByText('100')).toBeInTheDocument();
    });
  });

  it('displays users in the table', async () => {
    render(<UserManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Test User')).toBeInTheDocument();
      expect(screen.getByText('@testuser')).toBeInTheDocument();
    });
  });

  it('shows import and export buttons', async () => {
    render(<UserManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Import')).toBeInTheDocument();
      expect(screen.getByText('Export')).toBeInTheDocument();
    });
  });

  it('shows search input', async () => {
    render(<UserManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search by name or email')).toBeInTheDocument();
    });
  });

  it('shows role and status filters', async () => {
    render(<UserManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('All Roles')).toBeInTheDocument();
      expect(screen.getByText('All Status')).toBeInTheDocument();
    });
  });
});

describe('UserManagement Accessibility', () => {
  it('has proper ARIA labels', async () => {
    render(<UserManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByRole('main', { name: /user management/i })).toBeInTheDocument();
    });
  });
});
