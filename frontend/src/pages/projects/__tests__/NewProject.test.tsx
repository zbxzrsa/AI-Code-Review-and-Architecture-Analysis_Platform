/**
 * NewProject Wizard Component Tests
 * 
 * Tests for the multi-step project creation wizard.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import { NewProject } from '../NewProject';

// Mock the hooks
vi.mock('../../../hooks/useProjects', () => ({
  useCreateProject: vi.fn(() => ({
    mutate: vi.fn(),
    mutateAsync: vi.fn().mockResolvedValue({ id: 'new-project-id' }),
    isPending: false,
    isSuccess: false,
    data: null,
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
const mockSetDraft = vi.fn();
const mockUpdateDraft = vi.fn();
const mockSaveDraftToStorage = vi.fn();
const mockLoadDraftFromStorage = vi.fn();
const mockClearDraft = vi.fn();

vi.mock('../../../store/projectStore', async () => {
  const actual = await vi.importActual('../../../store/projectStore');
  return {
    ...actual,
    useProjectStore: vi.fn(() => ({
      draft: {
        step: 0,
        basic_info: {
          name: '',
          description: '',
          repository_url: '',
          language: 'python',
        },
        analysis_settings: {
          ai_model: 'gpt-4',
          analysis_frequency: 'on_pr',
          priority: 'medium',
          max_files_per_analysis: 100,
          excluded_patterns: ['node_modules/**', '.git/**', 'dist/**'],
        },
        notification_settings: {
          email_on_analysis_complete: true,
          email_on_critical_issues: true,
          email_digest: 'weekly',
        },
      },
      setDraft: mockSetDraft,
      updateDraft: mockUpdateDraft,
      saveDraftToStorage: mockSaveDraftToStorage,
      loadDraftFromStorage: mockLoadDraftFromStorage,
      clearDraft: mockClearDraft,
    })),
    defaultDraft: {
      step: 0,
      basic_info: {
        name: '',
        description: '',
        repository_url: '',
        language: 'python',
      },
      analysis_settings: {
        ai_model: 'gpt-4',
        analysis_frequency: 'on_pr',
        priority: 'medium',
        max_files_per_analysis: 100,
        excluded_patterns: ['node_modules/**', '.git/**', 'dist/**'],
      },
      notification_settings: {
        email_on_analysis_complete: true,
        email_on_critical_issues: true,
        email_digest: 'weekly',
      },
    },
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

describe('NewProject Wizard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the wizard title', () => {
    render(<NewProject />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Create New Project')).toBeInTheDocument();
  });

  it('renders all step indicators', () => {
    render(<NewProject />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Basic Info')).toBeInTheDocument();
    expect(screen.getByText('Analysis')).toBeInTheDocument();
    expect(screen.getByText('Notifications')).toBeInTheDocument();
    expect(screen.getByText('Review')).toBeInTheDocument();
  });

  it('renders step 1 form fields', () => {
    render(<NewProject />, { wrapper: createWrapper() });
    
    expect(screen.getByLabelText(/project name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/description/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/repository url/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/primary language/i)).toBeInTheDocument();
  });

  it('has Save Draft button', () => {
    render(<NewProject />, { wrapper: createWrapper() });
    
    expect(screen.getByRole('button', { name: /save draft/i })).toBeInTheDocument();
  });

  it('has Next button on first step', () => {
    render(<NewProject />, { wrapper: createWrapper() });
    
    expect(screen.getByRole('button', { name: /next/i })).toBeInTheDocument();
  });

  it('has Cancel button', () => {
    render(<NewProject />, { wrapper: createWrapper() });
    
    expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
  });

  it('saves draft when Save Draft is clicked', async () => {
    const user = userEvent.setup();
    render(<NewProject />, { wrapper: createWrapper() });
    
    const saveDraftButton = screen.getByRole('button', { name: /save draft/i });
    await user.click(saveDraftButton);
    
    await waitFor(() => {
      expect(mockSaveDraftToStorage).toHaveBeenCalled();
    });
  });

  it('validates required fields before navigation', async () => {
    const user = userEvent.setup();
    render(<NewProject />, { wrapper: createWrapper() });
    
    // Click Next without filling required fields
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);
    
    // Should show validation error
    await waitFor(() => {
      expect(screen.getByText(/please enter a project name/i)).toBeInTheDocument();
    });
  });
});

describe('NewProject Form Validation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows error for project name under 3 characters', async () => {
    const user = userEvent.setup();
    render(<NewProject />, { wrapper: createWrapper() });
    
    const nameInput = screen.getByLabelText(/project name/i);
    await user.type(nameInput, 'ab');
    await user.tab(); // Blur to trigger validation
    
    await waitFor(() => {
      expect(screen.getByText(/at least 3 characters/i)).toBeInTheDocument();
    });
  });

  it('shows error for invalid repository URL', async () => {
    const user = userEvent.setup();
    render(<NewProject />, { wrapper: createWrapper() });
    
    const repoInput = screen.getByLabelText(/repository url/i);
    await user.type(repoInput, 'not-a-valid-url');
    await user.tab();
    
    await waitFor(() => {
      expect(screen.getByText(/valid repository url/i)).toBeInTheDocument();
    });
  });

  it('accepts valid GitHub URL', async () => {
    const user = userEvent.setup();
    render(<NewProject />, { wrapper: createWrapper() });
    
    const repoInput = screen.getByLabelText(/repository url/i);
    await user.type(repoInput, 'https://github.com/user/repo');
    await user.tab();
    
    await waitFor(() => {
      expect(screen.queryByText(/valid repository url/i)).not.toBeInTheDocument();
    });
  });
});

describe('NewProject Accessibility', () => {
  it('has proper form structure', () => {
    render(<NewProject />, { wrapper: createWrapper() });
    
    // All form inputs should have labels
    const inputs = screen.getAllByRole('textbox');
    inputs.forEach(input => {
      expect(input).toHaveAccessibleName();
    });
  });

  it('has keyboard navigable steps', () => {
    render(<NewProject />, { wrapper: createWrapper() });
    
    const buttons = screen.getAllByRole('button');
    buttons.forEach(button => {
      expect(button).not.toHaveAttribute('tabindex', '-1');
    });
  });
});
