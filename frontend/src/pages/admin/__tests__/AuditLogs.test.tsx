/**
 * Audit Logs Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import { AuditLogs } from '../AuditLogs';

// Mock the hooks
vi.mock('../../../hooks/useAdmin', () => ({
  useAuditLogs: vi.fn(() => ({
    data: {
      items: [
        {
          id: '1',
          timestamp: '2024-01-15T12:00:00Z',
          userId: 'user1',
          username: 'testuser',
          userEmail: 'test@example.com',
          action: 'LOGIN',
          resource: 'user',
          ipAddress: '192.168.1.1',
          location: 'Vietnam',
          status: 'success',
        },
        {
          id: '2',
          timestamp: '2024-01-15T11:00:00Z',
          userId: 'user2',
          username: 'admin',
          userEmail: 'admin@example.com',
          action: 'UPDATE',
          resource: 'settings',
          ipAddress: '192.168.1.2',
          location: 'USA',
          status: 'success',
        },
      ],
      total: 2,
    },
    isLoading: false,
    refetch: vi.fn(),
    isFetching: false,
  })),
  useAuditLog: vi.fn(() => ({
    data: null,
    isLoading: false,
  })),
  useExportAuditLogs: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useAuditAnalytics: vi.fn(() => ({
    data: {
      totalLogs: 5000,
      successRate: 98.5,
      failureRate: 1.5,
      mostActiveUsers: [
        { userId: '1', username: 'admin', count: 500 },
        { userId: '2', username: 'testuser', count: 300 },
      ],
      actionDistribution: [
        { action: 'LOGIN', count: 1000 },
        { action: 'UPDATE', count: 500 },
        { action: 'CREATE', count: 300 },
      ],
      failedActionsTimeline: [],
      loginPatterns: [],
      topResources: [],
    },
    isLoading: false,
  })),
  useSecurityAlerts: vi.fn(() => ({
    data: { items: [], total: 0 },
    isLoading: false,
  })),
  useResolveAlert: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useLoginPatterns: vi.fn(() => ({
    data: [],
    isLoading: false,
  })),
}));

// Mock the store
vi.mock('../../../store/adminStore', async () => {
  const actual = await vi.importActual('../../../store/adminStore');
  return {
    ...actual,
    useAdminStore: vi.fn(() => ({
      auditFilters: { search: '', action: 'all', resource: 'all', status: 'all' },
      auditPagination: { page: 1, pageSize: 20, total: 0 },
      setAuditFilters: vi.fn(),
      resetAuditFilters: vi.fn(),
      setAuditPagination: vi.fn(),
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

describe('AuditLogs Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the page title', async () => {
    render(<AuditLogs />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Audit Logs')).toBeInTheDocument();
    });
  });

  it('displays audit statistics', async () => {
    render(<AuditLogs />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Total Logs')).toBeInTheDocument();
      expect(screen.getByText('Success Rate')).toBeInTheDocument();
      expect(screen.getByText('Failure Rate')).toBeInTheDocument();
    });
  });

  it('shows tabs for logs, analytics, and alerts', async () => {
    render(<AuditLogs />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Logs')).toBeInTheDocument();
      expect(screen.getByText('Analytics')).toBeInTheDocument();
      expect(screen.getByText('Security Alerts')).toBeInTheDocument();
    });
  });

  it('displays audit logs in table', async () => {
    render(<AuditLogs />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('testuser')).toBeInTheDocument();
      expect(screen.getByText('admin')).toBeInTheDocument();
    });
  });

  it('shows search input', async () => {
    render(<AuditLogs />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search logs...')).toBeInTheDocument();
    });
  });

  it('shows filter dropdowns', async () => {
    render(<AuditLogs />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('All Actions')).toBeInTheDocument();
      expect(screen.getByText('All Resources')).toBeInTheDocument();
      expect(screen.getByText('All Status')).toBeInTheDocument();
    });
  });

  it('shows export button', async () => {
    render(<AuditLogs />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Export')).toBeInTheDocument();
    });
  });
});

describe('AuditLogs Analytics Tab', () => {
  it('can switch to analytics tab', async () => {
    const user = userEvent.setup();
    render(<AuditLogs />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Analytics')).toBeInTheDocument();
    });
    
    await user.click(screen.getByText('Analytics'));
    
    await waitFor(() => {
      expect(screen.getByText('Action Distribution')).toBeInTheDocument();
      expect(screen.getByText('Most Active Users')).toBeInTheDocument();
    });
  });
});

describe('AuditLogs Security Alerts Tab', () => {
  it('can switch to security alerts tab', async () => {
    const user = userEvent.setup();
    render(<AuditLogs />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Security Alerts')).toBeInTheDocument();
    });
    
    await user.click(screen.getByText('Security Alerts'));
    
    await waitFor(() => {
      expect(screen.getByText('No active security alerts')).toBeInTheDocument();
    });
  });
});

describe('AuditLogs Accessibility', () => {
  it('has proper ARIA labels', async () => {
    render(<AuditLogs />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByRole('main', { name: /audit logs/i })).toBeInTheDocument();
    });
  });
});
