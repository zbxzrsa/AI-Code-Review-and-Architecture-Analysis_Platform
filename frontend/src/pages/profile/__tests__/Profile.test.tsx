/**
 * Profile Page Tests
 * 
 * Unit tests for the enhanced profile page components.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import { Profile } from '../Profile';

// Mock the hooks
vi.mock('../../../hooks/useUser', () => ({
  useUserProfile: vi.fn(() => ({
    data: {
      id: '1',
      email: 'test@example.com',
      name: 'Test User',
      username: 'testuser',
      bio: 'Test bio',
      role: 'user',
      avatar: null,
      createdAt: '2024-01-01T00:00:00Z',
      lastLoginAt: '2024-01-15T12:00:00Z',
      emailVerified: true,
      twoFactorEnabled: false,
    },
    isLoading: false,
    isError: false,
  })),
  useUpdateProfile: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useUploadAvatar: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useDeleteAvatar: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useOAuthConnections: vi.fn(() => ({
    data: [
      { provider: 'github', connected: true, username: 'testuser', connectedAt: '2024-01-01T00:00:00Z' },
      { provider: 'gitlab', connected: false },
    ],
    isLoading: false,
  })),
  useConnectOAuth: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useDisconnectOAuth: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useUpdatePrivacy: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useLoginHistory: vi.fn(() => ({
    data: { items: [], total: 0 },
    isLoading: false,
  })),
  useApiActivity: vi.fn(() => ({
    data: { items: [], total: 0 },
    isLoading: false,
  })),
  useDownloadPersonalData: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useRequestAccountDeletion: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
}));

// Mock i18n
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key,
    i18n: { language: 'en' },
  }),
}));

// Mock the store
vi.mock('../../../store/authStore', async () => {
  const actual = await vi.importActual('../../../store/authStore');
  return {
    ...actual,
    useAuthStore: vi.fn(() => ({
      user: {
        id: '1',
        email: 'test@example.com',
        name: 'Test User',
        username: 'testuser',
        role: 'user',
      },
      settings: {
        privacy: {
          profileVisibility: 'public',
          showEmail: false,
          showActivity: true,
        },
      },
      updateUser: vi.fn(),
    })),
  };
});

// Test wrapper with providers
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

describe('Profile Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the profile page with user information', async () => {
    render(<Profile />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Test User')).toBeInTheDocument();
    });
  });

  it('renders the edit profile button', async () => {
    render(<Profile />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /edit profile/i })).toBeInTheDocument();
    });
  });

  it('shows connected accounts section', async () => {
    render(<Profile />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Connected Accounts')).toBeInTheDocument();
      expect(screen.getByText('GitHub')).toBeInTheDocument();
      expect(screen.getByText('GitLab')).toBeInTheDocument();
    });
  });

  it('shows GitHub as connected', async () => {
    render(<Profile />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('testuser')).toBeInTheDocument();
    });
  });

  it('renders privacy settings section', async () => {
    render(<Profile />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Privacy Settings')).toBeInTheDocument();
    });
  });

  it('renders activity section', async () => {
    render(<Profile />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Activity')).toBeInTheDocument();
      expect(screen.getByText('Login History')).toBeInTheDocument();
      expect(screen.getByText('API Activity')).toBeInTheDocument();
    });
  });

  it('renders danger zone section', async () => {
    render(<Profile />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Danger Zone')).toBeInTheDocument();
      expect(screen.getByText('Download Your Data')).toBeInTheDocument();
      expect(screen.getByText('Delete Account')).toBeInTheDocument();
    });
  });

  it('shows verified badge when email is verified', async () => {
    render(<Profile />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Verified')).toBeInTheDocument();
    });
  });

  it('renders statistics', async () => {
    render(<Profile />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Projects')).toBeInTheDocument();
      expect(screen.getByText('Analyses')).toBeInTheDocument();
      expect(screen.getByText('Day Streak')).toBeInTheDocument();
    });
  });
});

describe('Profile Accessibility', () => {
  it('has proper ARIA labels', async () => {
    render(<Profile />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByRole('main', { name: /user profile/i })).toBeInTheDocument();
    });
  });

  it('has keyboard navigable buttons', async () => {
    render(<Profile />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).not.toHaveAttribute('tabindex', '-1');
      });
    });
  });
});
