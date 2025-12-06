/**
 * Sidebar Component Tests
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the stores
vi.mock('@/store/authStore', () => ({
  useAuthStore: vi.fn(() => ({
    user: { id: '1', name: 'Test User', email: 'test@example.com', role: 'user' },
    isAuthenticated: true,
    logout: vi.fn(),
  })),
}));

vi.mock('@/store/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    sidebarCollapsed: false,
    toggleSidebar: vi.fn(),
    theme: 'light',
  })),
}));

// Mock i18n
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key,
  }),
}));

// Import after mocks
import { Sidebar } from '@/components/layout/Sidebar';
import { useAuthStore } from '@/store/authStore';
import { useUIStore } from '@/store/uiStore';

const renderSidebar = () => {
  return render(
    <BrowserRouter>
      <Sidebar />
    </BrowserRouter>
  );
};

describe('Sidebar', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders sidebar with navigation items', () => {
      renderSidebar();
      
      expect(screen.getByRole('navigation')).toBeInTheDocument();
    });

    it('renders logo/brand', () => {
      renderSidebar();
      
      // Look for logo or brand element
      const logo = screen.queryByRole('img') || screen.queryByText(/code review/i);
      expect(logo).toBeInTheDocument();
    });

    it('renders navigation links', () => {
      renderSidebar();
      
      expect(screen.getByText(/dashboard/i)).toBeInTheDocument();
      expect(screen.getByText(/projects/i)).toBeInTheDocument();
    });

    it('renders user info when authenticated', () => {
      renderSidebar();
      
      expect(screen.getByText('Test User')).toBeInTheDocument();
    });
  });

  describe('Collapsed State', () => {
    it('shows collapsed sidebar when sidebarCollapsed is true', () => {
      (useUIStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
        sidebarCollapsed: true,
        toggleSidebar: vi.fn(),
        theme: 'light',
      });

      renderSidebar();
      
      const sidebar = screen.getByRole('navigation');
      expect(sidebar).toHaveClass('collapsed');
    });

    it('calls toggleSidebar when collapse button clicked', async () => {
      const mockToggle = vi.fn();
      (useUIStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
        sidebarCollapsed: false,
        toggleSidebar: mockToggle,
        theme: 'light',
      });

      renderSidebar();
      
      const toggleButton = screen.getByRole('button', { name: /toggle/i });
      await userEvent.click(toggleButton);
      
      expect(mockToggle).toHaveBeenCalled();
    });
  });

  describe('Navigation', () => {
    it('highlights active navigation item', () => {
      renderSidebar();
      
      // Dashboard should be active by default
      const dashboardLink = screen.getByText(/dashboard/i).closest('a');
      expect(dashboardLink).toHaveClass('active');
    });

    it('navigates when clicking menu item', async () => {
      renderSidebar();
      
      const projectsLink = screen.getByText(/projects/i);
      await userEvent.click(projectsLink);
      
      expect(window.location.pathname).toBe('/projects');
    });
  });

  describe('User Menu', () => {
    it('shows logout option', async () => {
      renderSidebar();
      
      const userMenu = screen.getByText('Test User');
      await userEvent.click(userMenu);
      
      await waitFor(() => {
        expect(screen.getByText(/logout/i)).toBeInTheDocument();
      });
    });

    it('calls logout when logout clicked', async () => {
      const mockLogout = vi.fn();
      (useAuthStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
        user: { id: '1', name: 'Test User', email: 'test@example.com', role: 'user' },
        isAuthenticated: true,
        logout: mockLogout,
      });

      renderSidebar();
      
      const userMenu = screen.getByText('Test User');
      await userEvent.click(userMenu);
      
      const logoutButton = await screen.findByText(/logout/i);
      await userEvent.click(logoutButton);
      
      expect(mockLogout).toHaveBeenCalled();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels', () => {
      renderSidebar();
      
      const nav = screen.getByRole('navigation');
      expect(nav).toHaveAttribute('aria-label');
    });

    it('supports keyboard navigation', async () => {
      renderSidebar();
      
      const firstLink = screen.getAllByRole('link')[0];
      firstLink.focus();
      
      expect(document.activeElement).toBe(firstLink);
      
      // Tab to next element
      await userEvent.tab();
      expect(document.activeElement).not.toBe(firstLink);
    });
  });
});
