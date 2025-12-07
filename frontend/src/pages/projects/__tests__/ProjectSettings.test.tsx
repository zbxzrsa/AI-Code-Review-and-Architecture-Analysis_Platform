/**
 * ProjectSettings Component Tests
 *
 * Tests for the project settings page with all configuration sections.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import { ProjectSettings } from '../ProjectSettings';

// Mock the hooks
vi.mock('../../../hooks/useProjects', () => ({
  useProject: vi.fn(() => ({
    data: {
      id: '1',
      name: 'test-project',
      description: 'A test project',
      language: 'python',
      framework: 'FastAPI',
      status: 'active',
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-02T00:00:00Z',
      owner_id: '1',
      owner_name: 'Test User',
      is_public: false,
      settings: {
        auto_review: true,
        review_on_push: true,
        review_on_pr: true,
        severity_threshold: 'warning',
        enabled_rules: [],
        ignored_paths: [],
      },
    },
    isLoading: false,
    isError: false,
  })),
  useProjectActivity: vi.fn(() => ({
    data: { items: [], total: 0 },
    isLoading: false,
    refetch: vi.fn(),
  })),
  useProjectTeam: vi.fn(() => ({
    data: [
      {
        id: '1',
        user_id: '1',
        email: 'owner@example.com',
        name: 'Owner User',
        role: 'owner',
        invited_at: '2024-01-01T00:00:00Z',
        accepted_at: '2024-01-01T00:00:00Z',
      },
    ],
    isLoading: false,
  })),
  useProjectWebhooks: vi.fn(() => ({
    data: [],
    isLoading: false,
  })),
  useProjectApiKeys: vi.fn(() => ({
    data: [],
    isLoading: false,
  })),
  useUpdateProject: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useDeleteProject: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useArchiveProject: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useInviteTeamMember: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useUpdateMemberRole: vi.fn(() => ({
    mutate: vi.fn(),
  })),
  useRemoveTeamMember: vi.fn(() => ({
    mutate: vi.fn(),
  })),
  useCreateWebhook: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useUpdateWebhook: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useDeleteWebhook: vi.fn(() => ({
    mutate: vi.fn(),
  })),
  useTestWebhook: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useCreateApiKey: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useRevokeApiKey: vi.fn(() => ({
    mutate: vi.fn(),
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
vi.mock('../../../store/projectStore', async () => {
  const actual = await vi.importActual('../../../store/projectStore');
  return {
    ...actual,
    useProjectStore: vi.fn(() => ({
      unsavedChanges: false,
      setUnsavedChanges: vi.fn(),
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
          <Routes>
            <Route path="/projects/:id/settings" element={children} />
          </Routes>
        </BrowserRouter>
      </ConfigProvider>
    </QueryClientProvider>
  );
};

// Helper to render with route
const renderWithRoute = (component: React.ReactNode) => {
  globalThis.history.pushState({}, 'Test', '/projects/1/settings');
  return render(component, { wrapper: createWrapper() });
};

describe('ProjectSettings', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the project settings page with project name', async () => {
    renderWithRoute(<ProjectSettings />);

    await waitFor(() => {
      expect(screen.getByText('test-project')).toBeInTheDocument();
    });
  });

  it('renders all setting tabs', async () => {
    renderWithRoute(<ProjectSettings />);

    await waitFor(() => {
      expect(screen.getByText('General')).toBeInTheDocument();
      expect(screen.getByText('Team')).toBeInTheDocument();
      expect(screen.getByText('Webhooks')).toBeInTheDocument();
      expect(screen.getByText('API Keys')).toBeInTheDocument();
      expect(screen.getByText('Activity')).toBeInTheDocument();
      expect(screen.getByText('Danger Zone')).toBeInTheDocument();
    });
  });

  it('renders back button', async () => {
    renderWithRoute(<ProjectSettings />);

    await waitFor(() => {
      // Back button uses arrow-left icon, find by querying the icon or button class
      const backButton = document.querySelector('.ant-btn-icon-only');
      expect(backButton).toBeInTheDocument();
    });
  });
});

describe('ProjectSettings General Tab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders project information form', async () => {
    renderWithRoute(<ProjectSettings />);

    await waitFor(() => {
      expect(screen.getByLabelText(/project name/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/description/i)).toBeInTheDocument();
    });
  });

  it('shows project metadata', async () => {
    renderWithRoute(<ProjectSettings />);

    await waitFor(() => {
      expect(screen.getByText('Owner')).toBeInTheDocument();
      expect(screen.getByText('Created')).toBeInTheDocument();
      expect(screen.getByText('Language')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
    });
  });

  it('has save button', async () => {
    renderWithRoute(<ProjectSettings />);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /save changes/i })).toBeInTheDocument();
    });
  });
});

describe('ProjectSettings Accessibility', () => {
  it('has proper tab navigation', async () => {
    renderWithRoute(<ProjectSettings />);

    await waitFor(() => {
      const tabs = screen.getAllByRole('tab');
      expect(tabs.length).toBeGreaterThan(0);
    });
  });
});
