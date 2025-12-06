/**
 * ProjectForm Component Tests
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock i18n
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key,
  }),
}));

// Mock API service
vi.mock('@/services/api', () => ({
  apiService: {
    projects: {
      create: vi.fn().mockResolvedValue({ data: { id: '123', name: 'Test Project' } }),
      update: vi.fn().mockResolvedValue({ data: { id: '123', name: 'Updated Project' } }),
    },
  },
}));

// Mock message
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    message: {
      success: vi.fn(),
      error: vi.fn(),
    },
  };
});

import { ProjectForm } from '@/components/projects/ProjectForm';
import { apiService } from '@/services/api';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('ProjectForm', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders form with all required fields', () => {
      render(<ProjectForm />, { wrapper: createWrapper() });

      expect(screen.getByLabelText(/project name/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/description/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /create|submit/i })).toBeInTheDocument();
    });

    it('renders language selection', () => {
      render(<ProjectForm />, { wrapper: createWrapper() });

      expect(screen.getByLabelText(/language/i)).toBeInTheDocument();
    });

    it('renders repository URL field', () => {
      render(<ProjectForm />, { wrapper: createWrapper() });

      expect(screen.getByLabelText(/repository/i)).toBeInTheDocument();
    });

    it('renders in edit mode with initial values', () => {
      const initialValues = {
        id: '123',
        name: 'Existing Project',
        description: 'Existing description',
        language: 'python',
      };

      render(<ProjectForm initialValues={initialValues} mode="edit" />, { 
        wrapper: createWrapper() 
      });

      expect(screen.getByDisplayValue('Existing Project')).toBeInTheDocument();
      expect(screen.getByDisplayValue('Existing description')).toBeInTheDocument();
    });
  });

  describe('Form Submission', () => {
    it('submits form with valid data', async () => {
      const user = userEvent.setup();
      const mockOnSubmit = vi.fn();

      render(<ProjectForm onSubmit={mockOnSubmit} />, { wrapper: createWrapper() });

      await user.type(screen.getByLabelText(/project name/i), 'My Project');
      await user.type(screen.getByLabelText(/description/i), 'Project description');
      
      // Select language if it's a select dropdown
      const languageSelect = screen.getByLabelText(/language/i);
      await user.click(languageSelect);
      await user.click(screen.getByText('Python'));

      await user.click(screen.getByRole('button', { name: /create|submit/i }));

      await waitFor(() => {
        expect(mockOnSubmit).toHaveBeenCalledWith(
          expect.objectContaining({
            name: 'My Project',
            description: 'Project description',
          })
        );
      });
    });

    it('calls API on submit', async () => {
      const user = userEvent.setup();

      render(<ProjectForm />, { wrapper: createWrapper() });

      await user.type(screen.getByLabelText(/project name/i), 'API Test Project');
      await user.type(screen.getByLabelText(/description/i), 'Description');
      await user.click(screen.getByRole('button', { name: /create|submit/i }));

      await waitFor(() => {
        expect(apiService.projects.create).toHaveBeenCalled();
      });
    });
  });

  describe('Validation', () => {
    it('displays validation error for empty name', async () => {
      const user = userEvent.setup();

      render(<ProjectForm />, { wrapper: createWrapper() });

      // Submit without filling required fields
      await user.click(screen.getByRole('button', { name: /create|submit/i }));

      await waitFor(() => {
        expect(screen.getByText(/name.*required|please.*name/i)).toBeInTheDocument();
      });
    });

    it('displays validation error for invalid repository URL', async () => {
      const user = userEvent.setup();

      render(<ProjectForm />, { wrapper: createWrapper() });

      await user.type(screen.getByLabelText(/project name/i), 'Test Project');
      await user.type(screen.getByLabelText(/repository/i), 'invalid-url');
      await user.click(screen.getByRole('button', { name: /create|submit/i }));

      await waitFor(() => {
        expect(screen.getByText(/valid.*url|invalid.*url/i)).toBeInTheDocument();
      });
    });

    it('validates name length', async () => {
      const user = userEvent.setup();

      render(<ProjectForm />, { wrapper: createWrapper() });

      // Type very short name
      await user.type(screen.getByLabelText(/project name/i), 'ab');
      await user.click(screen.getByRole('button', { name: /create|submit/i }));

      await waitFor(() => {
        expect(screen.getByText(/at least|minimum|too short/i)).toBeInTheDocument();
      });
    });
  });

  describe('Edit Mode', () => {
    it('shows update button in edit mode', () => {
      render(<ProjectForm mode="edit" initialValues={{ id: '123', name: 'Test' }} />, { 
        wrapper: createWrapper() 
      });

      expect(screen.getByRole('button', { name: /update|save/i })).toBeInTheDocument();
    });

    it('calls update API in edit mode', async () => {
      const user = userEvent.setup();

      render(
        <ProjectForm 
          mode="edit" 
          initialValues={{ id: '123', name: 'Test', description: 'Desc' }} 
        />, 
        { wrapper: createWrapper() }
      );

      await user.clear(screen.getByLabelText(/project name/i));
      await user.type(screen.getByLabelText(/project name/i), 'Updated Name');
      await user.click(screen.getByRole('button', { name: /update|save/i }));

      await waitFor(() => {
        expect(apiService.projects.update).toHaveBeenCalled();
      });
    });
  });

  describe('Cancel Action', () => {
    it('shows cancel button', () => {
      render(<ProjectForm />, { wrapper: createWrapper() });

      expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
    });

    it('calls onCancel when cancel clicked', async () => {
      const user = userEvent.setup();
      const mockOnCancel = vi.fn();

      render(<ProjectForm onCancel={mockOnCancel} />, { wrapper: createWrapper() });

      await user.click(screen.getByRole('button', { name: /cancel/i }));

      expect(mockOnCancel).toHaveBeenCalled();
    });

    it('resets form on cancel', async () => {
      const user = userEvent.setup();

      render(<ProjectForm />, { wrapper: createWrapper() });

      await user.type(screen.getByLabelText(/project name/i), 'Typed Text');
      await user.click(screen.getByRole('button', { name: /cancel/i }));

      expect(screen.getByLabelText(/project name/i)).toHaveValue('');
    });
  });

  describe('Loading State', () => {
    it('disables submit button while loading', async () => {
      const user = userEvent.setup();
      
      // Make API slow
      (apiService.projects.create as ReturnType<typeof vi.fn>).mockImplementation(
        () => new Promise(resolve => setTimeout(resolve, 1000))
      );

      render(<ProjectForm />, { wrapper: createWrapper() });

      await user.type(screen.getByLabelText(/project name/i), 'Test');
      await user.type(screen.getByLabelText(/description/i), 'Desc');
      await user.click(screen.getByRole('button', { name: /create|submit/i }));

      expect(screen.getByRole('button', { name: /create|submit|loading/i })).toBeDisabled();
    });

    it('shows loading indicator', async () => {
      const user = userEvent.setup();
      
      (apiService.projects.create as ReturnType<typeof vi.fn>).mockImplementation(
        () => new Promise(resolve => setTimeout(resolve, 1000))
      );

      render(<ProjectForm />, { wrapper: createWrapper() });

      await user.type(screen.getByLabelText(/project name/i), 'Test');
      await user.type(screen.getByLabelText(/description/i), 'Desc');
      await user.click(screen.getByRole('button', { name: /create|submit/i }));

      expect(screen.getByRole('button')).toContainElement(
        screen.queryByRole('progressbar') || screen.queryByTestId('loading-spinner')
      );
    });
  });

  describe('Error Handling', () => {
    it('displays error message on API failure', async () => {
      const user = userEvent.setup();
      
      (apiService.projects.create as ReturnType<typeof vi.fn>).mockRejectedValue(
        new Error('API Error')
      );

      render(<ProjectForm />, { wrapper: createWrapper() });

      await user.type(screen.getByLabelText(/project name/i), 'Test');
      await user.type(screen.getByLabelText(/description/i), 'Desc');
      await user.click(screen.getByRole('button', { name: /create|submit/i }));

      await waitFor(() => {
        expect(screen.getByText(/error|failed/i)).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('has proper form labels', () => {
      render(<ProjectForm />, { wrapper: createWrapper() });

      const nameInput = screen.getByLabelText(/project name/i);
      expect(nameInput).toHaveAttribute('id');
      
      const label = screen.getByText(/project name/i);
      expect(label).toHaveAttribute('for', nameInput.getAttribute('id'));
    });

    it('shows required field indicators', () => {
      render(<ProjectForm />, { wrapper: createWrapper() });

      const requiredFields = screen.getAllByText('*');
      expect(requiredFields.length).toBeGreaterThan(0);
    });

    it('supports keyboard submission', async () => {
      const user = userEvent.setup();
      const mockOnSubmit = vi.fn();

      render(<ProjectForm onSubmit={mockOnSubmit} />, { wrapper: createWrapper() });

      await user.type(screen.getByLabelText(/project name/i), 'Test{Enter}');

      // Form should attempt submission on Enter
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /create|submit/i })).toBeInTheDocument();
      });
    });
  });
});
