/**
 * Layout Component Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import { Layout } from '../Layout';

// Mock stores
vi.mock('../../../store/authStore', () => ({
  useAuthStore: vi.fn(() => ({
    user: {
      id: '1',
      name: 'Test User',
      email: 'test@example.com',
      role: 'admin',
      avatar: null,
    },
    isAuthenticated: true,
    logout: vi.fn(),
  })),
}));

vi.mock('../../../store/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    sidebar: {
      isOpen: true,
      width: 280,
      searchQuery: '',
      expandedKeys: [],
      favorites: [],
      mobileDrawerOpen: false,
    },
    resolvedTheme: 'light',
    setTheme: vi.fn(),
    toggleMobileDrawer: vi.fn(),
    openCommandPalette: vi.fn(),
    breadcrumbs: [],
  })),
}));

// Mock i18n
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key,
  }),
}));

// Mock Sidebar
vi.mock('../Sidebar', () => ({
  Sidebar: () => <aside data-testid="sidebar">Sidebar</aside>,
}));

// Mock LanguageSelector
vi.mock('../../common/LanguageSelector', () => ({
  LanguageSelector: () => <div data-testid="language-selector">Language</div>,
}));

// Test wrapper
const createWrapper = (initialEntries: string[] = ['/dashboard']) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider>
        <MemoryRouter initialEntries={initialEntries}>
          {children}
        </MemoryRouter>
      </ConfigProvider>
    </QueryClientProvider>
  );
};

describe('Layout Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the layout with sidebar', async () => {
    render(<Layout />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByTestId('sidebar')).toBeInTheDocument();
    });
  });

  it('renders the header', async () => {
    render(<Layout />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByRole('banner')).toBeInTheDocument();
    });
  });

  it('shows user name in header', async () => {
    render(<Layout />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Test User')).toBeInTheDocument();
    });
  });

  it('shows language selector', async () => {
    render(<Layout />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByTestId('language-selector')).toBeInTheDocument();
    });
  });

  it('shows notifications badge', async () => {
    render(<Layout />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /notifications/i })).toBeInTheDocument();
    });
  });

  it('shows theme toggle button', async () => {
    render(<Layout />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /mode/i })).toBeInTheDocument();
    });
  });

  it('shows help button', async () => {
    render(<Layout />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /help/i })).toBeInTheDocument();
    });
  });
});

describe('Layout Breadcrumbs', () => {
  it('shows breadcrumbs on dashboard', async () => {
    render(<Layout />, { wrapper: createWrapper(['/dashboard']) });
    
    await waitFor(() => {
      // Dashboard shows home icon breadcrumb
      const breadcrumbs = document.querySelector('.app-breadcrumbs');
      expect(breadcrumbs).toBeInTheDocument();
    });
  });

  it('shows breadcrumbs for nested routes', async () => {
    render(<Layout />, { wrapper: createWrapper(['/admin/users']) });
    
    await waitFor(() => {
      const breadcrumbs = document.querySelector('.app-breadcrumbs');
      expect(breadcrumbs).toBeInTheDocument();
    });
  });
});

describe('Layout Responsive', () => {
  it('shows mobile menu button on mobile', async () => {
    render(<Layout />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      const mobileBtn = document.querySelector('.mobile-menu-btn');
      expect(mobileBtn).toBeInTheDocument();
    });
  });
});

describe('Layout Accessibility', () => {
  it('has accessible header', async () => {
    render(<Layout />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      const header = screen.getByRole('banner');
      expect(header).toBeInTheDocument();
    });
  });

  it('has accessible main content area', async () => {
    render(<Layout />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      const main = screen.getByRole('main');
      expect(main).toBeInTheDocument();
    });
  });
});
