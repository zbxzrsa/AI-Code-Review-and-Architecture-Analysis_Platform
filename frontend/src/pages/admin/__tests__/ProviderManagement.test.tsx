/**
 * Provider Management Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import { ProviderManagement } from '../ProviderManagement';

// Mock the hooks
vi.mock('../../../hooks/useAdmin', () => ({
  useProviders: vi.fn(() => ({
    data: [
      {
        id: '1',
        type: 'openai',
        name: 'OpenAI',
        status: 'active',
        isDefault: true,
        priority: 1,
        apiKeyConfigured: true,
        avgResponseTime: 250,
        requestsToday: 500,
        costToday: 15.50,
        costThisMonth: 350,
        quotaUsed: 50000,
        quotaLimit: 100000,
        errorRate: 0.02,
      },
      {
        id: '2',
        type: 'anthropic',
        name: 'Anthropic',
        status: 'active',
        isDefault: false,
        priority: 2,
        apiKeyConfigured: true,
        avgResponseTime: 300,
        requestsToday: 200,
        costToday: 8.25,
        costThisMonth: 180,
        quotaUsed: 20000,
        quotaLimit: 50000,
        errorRate: 0.01,
      },
    ],
    isLoading: false,
    refetch: vi.fn(),
    isFetching: false,
  })),
  useProvider: vi.fn(() => ({
    data: null,
    isLoading: false,
  })),
  useUpdateProvider: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useUpdateProviderApiKey: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useTestProvider: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useProviderHealth: vi.fn(() => ({
    data: null,
    isLoading: false,
  })),
  useProviderMetrics: vi.fn(() => ({
    data: null,
    isLoading: false,
  })),
  useProviderModels: vi.fn(() => ({
    data: [],
    isLoading: false,
  })),
  useUpdateProviderModel: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useSetFallbackOrder: vi.fn(() => ({
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
      selectedProviderId: null,
      selectProvider: vi.fn(),
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

describe('ProviderManagement Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the page title', async () => {
    render(<ProviderManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('AI Provider Management')).toBeInTheDocument();
    });
  });

  it('displays provider statistics', async () => {
    render(<ProviderManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Total Providers')).toBeInTheDocument();
      expect(screen.getByText('Active')).toBeInTheDocument();
      expect(screen.getByText('Requests Today')).toBeInTheDocument();
      expect(screen.getByText('Cost Today')).toBeInTheDocument();
    });
  });

  it('displays provider cards', async () => {
    render(<ProviderManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('OpenAI')).toBeInTheDocument();
      expect(screen.getByText('Anthropic')).toBeInTheDocument();
    });
  });

  it('shows test buttons for each provider', async () => {
    render(<ProviderManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      const testButtons = screen.getAllByText('Test');
      expect(testButtons.length).toBeGreaterThan(0);
    });
  });

  it('shows default tag for default provider', async () => {
    render(<ProviderManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Default')).toBeInTheDocument();
    });
  });

  it('shows select provider message when no provider selected', async () => {
    render(<ProviderManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Select a provider to configure')).toBeInTheDocument();
    });
  });
});

describe('ProviderManagement Accessibility', () => {
  it('has proper ARIA labels', async () => {
    render(<ProviderManagement />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByRole('main', { name: /provider management/i })).toBeInTheDocument();
    });
  });
});
