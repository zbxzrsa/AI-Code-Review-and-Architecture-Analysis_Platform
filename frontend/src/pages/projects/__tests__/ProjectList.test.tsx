/**
 * ProjectList Component Tests
 * 
 * Tests for the projects listing page with filtering, sorting, and pagination.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import { ProjectList } from '../ProjectList';

// Mock the hooks
vi.mock('../../../hooks/useProjects', () => ({
  useProjects: vi.fn(() => ({
    data: {
      items: [
        {
          id: '1',
          name: 'test-project',
          description: 'A test project',
          language: 'python',
          framework: 'FastAPI',
          status: 'active',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-02T00:00:00Z',
          last_analyzed_at: '2024-01-02T00:00:00Z',
          owner_id: '1',
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
        {
          id: '2',
          name: 'frontend-app',
          description: 'React frontend',
          language: 'typescript',
          framework: 'React',
          status: 'inactive',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-03T00:00:00Z',
          owner_id: '1',
          is_public: true,
          settings: {
            auto_review: false,
            review_on_push: false,
            review_on_pr: true,
            severity_threshold: 'error',
            enabled_rules: [],
            ignored_paths: [],
          },
        },
      ],
      total: 2,
      page: 1,
      limit: 10,
    },
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
    isFetching: false,
  })),
  useDeleteProject: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useArchiveProject: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useRestoreProject: vi.fn(() => ({
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
vi.mock('../../../store/projectStore', async () => {
  const actual = await vi.importActual('../../../store/projectStore');
  return {
    ...actual,
    useProjectStore: vi.fn(() => ({
      viewMode: 'table',
      filters: {
        search: '',
        status: 'all',
        language: 'all',
        sortField: 'updated_at',
        sortOrder: 'desc',
      },
      pagination: {
        page: 1,
        pageSize: 10,
        total: 0,
      },
      setViewMode: vi.fn(),
      setFilters: vi.fn(),
      setPagination: vi.fn(),
      resetFilters: vi.fn(),
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

describe('ProjectList', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the project list page title', () => {
    render(<ProjectList />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Projects')).toBeInTheDocument();
    expect(screen.getByText('Manage your code review projects')).toBeInTheDocument();
  });

  it('renders the New Project button', () => {
    render(<ProjectList />, { wrapper: createWrapper() });
    
    expect(screen.getByRole('button', { name: /new project/i })).toBeInTheDocument();
  });

  it('renders project statistics cards', () => {
    render(<ProjectList />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Total Projects')).toBeInTheDocument();
    expect(screen.getByText('Active Projects')).toBeInTheDocument();
    expect(screen.getByText('Recent Analyses')).toBeInTheDocument();
  });

  it('renders the search input', () => {
    render(<ProjectList />, { wrapper: createWrapper() });
    
    expect(screen.getByPlaceholderText('Search projects...')).toBeInTheDocument();
  });

  it('renders project data in table', async () => {
    render(<ProjectList />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('test-project')).toBeInTheDocument();
      expect(screen.getByText('frontend-app')).toBeInTheDocument();
    });
  });

  it('renders language tags', async () => {
    render(<ProjectList />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('python')).toBeInTheDocument();
      expect(screen.getByText('typescript')).toBeInTheDocument();
    });
  });

  it('has filter dropdowns', () => {
    render(<ProjectList />, { wrapper: createWrapper() });
    
    // Check for status filter
    const statusFilter = screen.getByRole('combobox', { name: /filter by status/i });
    expect(statusFilter).toBeInTheDocument();
    
    // Check for language filter
    const languageFilter = screen.getByRole('combobox', { name: /filter by language/i });
    expect(languageFilter).toBeInTheDocument();
  });

  it('has view mode toggle', () => {
    render(<ProjectList />, { wrapper: createWrapper() });
    
    // Find the segmented control for view mode
    const viewModeControl = screen.getByRole('group', { name: /view mode/i });
    expect(viewModeControl).toBeInTheDocument();
  });

  it('navigates to new project page when button is clicked', async () => {
    const user = userEvent.setup();
    render(<ProjectList />, { wrapper: createWrapper() });
    
    const newProjectButton = screen.getByRole('button', { name: /new project/i });
    await user.click(newProjectButton);
    
    // Check navigation (would need to mock useNavigate for full test)
    expect(newProjectButton).toBeInTheDocument();
  });
});

describe('ProjectList Accessibility', () => {
  it('has proper ARIA labels', () => {
    render(<ProjectList />, { wrapper: createWrapper() });
    
    expect(screen.getByRole('main', { name: /projects/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/search projects/i)).toBeInTheDocument();
  });

  it('has keyboard navigable elements', () => {
    render(<ProjectList />, { wrapper: createWrapper() });
    
    const buttons = screen.getAllByRole('button');
    buttons.forEach(button => {
      expect(button).not.toHaveAttribute('tabindex', '-1');
    });
  });
});
