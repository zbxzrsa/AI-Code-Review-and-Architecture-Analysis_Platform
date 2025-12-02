/**
 * NotificationCenter Component Tests
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock stores
const mockNotifications = [
  {
    id: '1',
    type: 'success',
    title: 'Success',
    message: 'Operation completed successfully',
    timestamp: new Date().toISOString(),
    read: false,
  },
  {
    id: '2',
    type: 'error',
    title: 'Error',
    message: 'Something went wrong',
    timestamp: new Date().toISOString(),
    read: false,
  },
  {
    id: '3',
    type: 'info',
    title: 'Info',
    message: 'New update available',
    timestamp: new Date().toISOString(),
    read: true,
  },
];

vi.mock('@/store/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    notifications: mockNotifications,
    unreadCount: 2,
    addNotification: vi.fn(),
    removeNotification: vi.fn(),
    markAsRead: vi.fn(),
    markAllAsRead: vi.fn(),
    clearNotifications: vi.fn(),
  })),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key,
  }),
}));

import { NotificationCenter } from '@/components/common/NotificationCenter';
import { useUIStore } from '@/store/uiStore';

describe('NotificationCenter', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders notification bell icon', () => {
      render(<NotificationCenter />);
      
      const bell = screen.getByRole('button', { name: /notification/i });
      expect(bell).toBeInTheDocument();
    });

    it('shows unread count badge', () => {
      render(<NotificationCenter />);
      
      expect(screen.getByText('2')).toBeInTheDocument();
    });

    it('hides badge when no unread notifications', () => {
      (useUIStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
        notifications: [],
        unreadCount: 0,
        addNotification: vi.fn(),
        removeNotification: vi.fn(),
        markAsRead: vi.fn(),
        markAllAsRead: vi.fn(),
        clearNotifications: vi.fn(),
      });

      render(<NotificationCenter />);
      
      expect(screen.queryByText('0')).not.toBeInTheDocument();
    });
  });

  describe('Dropdown', () => {
    it('opens dropdown on click', async () => {
      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      const bell = screen.getByRole('button', { name: /notification/i });
      await user.click(bell);
      
      await waitFor(() => {
        expect(screen.getByText('Success')).toBeInTheDocument();
        expect(screen.getByText('Error')).toBeInTheDocument();
      });
    });

    it('shows all notifications in dropdown', async () => {
      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      await user.click(screen.getByRole('button', { name: /notification/i }));
      
      await waitFor(() => {
        expect(screen.getByText('Operation completed successfully')).toBeInTheDocument();
        expect(screen.getByText('Something went wrong')).toBeInTheDocument();
        expect(screen.getByText('New update available')).toBeInTheDocument();
      });
    });

    it('closes dropdown on outside click', async () => {
      const user = userEvent.setup();
      render(
        <div>
          <NotificationCenter />
          <button>Outside</button>
        </div>
      );
      
      await user.click(screen.getByRole('button', { name: /notification/i }));
      await waitFor(() => {
        expect(screen.getByText('Success')).toBeInTheDocument();
      });
      
      await user.click(screen.getByText('Outside'));
      
      await waitFor(() => {
        expect(screen.queryByText('Success')).not.toBeInTheDocument();
      });
    });
  });

  describe('Notification Actions', () => {
    it('marks notification as read on click', async () => {
      const mockMarkAsRead = vi.fn();
      (useUIStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
        notifications: mockNotifications,
        unreadCount: 2,
        addNotification: vi.fn(),
        removeNotification: vi.fn(),
        markAsRead: mockMarkAsRead,
        markAllAsRead: vi.fn(),
        clearNotifications: vi.fn(),
      });

      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      await user.click(screen.getByRole('button', { name: /notification/i }));
      
      const successNotification = await screen.findByText('Success');
      await user.click(successNotification);
      
      expect(mockMarkAsRead).toHaveBeenCalledWith('1');
    });

    it('removes notification on dismiss', async () => {
      const mockRemove = vi.fn();
      (useUIStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
        notifications: mockNotifications,
        unreadCount: 2,
        addNotification: vi.fn(),
        removeNotification: mockRemove,
        markAsRead: vi.fn(),
        markAllAsRead: vi.fn(),
        clearNotifications: vi.fn(),
      });

      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      await user.click(screen.getByRole('button', { name: /notification/i }));
      
      const dismissButtons = await screen.findAllByRole('button', { name: /dismiss|close|remove/i });
      await user.click(dismissButtons[0]);
      
      expect(mockRemove).toHaveBeenCalled();
    });

    it('marks all as read', async () => {
      const mockMarkAllAsRead = vi.fn();
      (useUIStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
        notifications: mockNotifications,
        unreadCount: 2,
        addNotification: vi.fn(),
        removeNotification: vi.fn(),
        markAsRead: vi.fn(),
        markAllAsRead: mockMarkAllAsRead,
        clearNotifications: vi.fn(),
      });

      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      await user.click(screen.getByRole('button', { name: /notification/i }));
      
      const markAllButton = await screen.findByText(/mark all as read/i);
      await user.click(markAllButton);
      
      expect(mockMarkAllAsRead).toHaveBeenCalled();
    });

    it('clears all notifications', async () => {
      const mockClear = vi.fn();
      (useUIStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
        notifications: mockNotifications,
        unreadCount: 2,
        addNotification: vi.fn(),
        removeNotification: vi.fn(),
        markAsRead: vi.fn(),
        markAllAsRead: vi.fn(),
        clearNotifications: mockClear,
      });

      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      await user.click(screen.getByRole('button', { name: /notification/i }));
      
      const clearButton = await screen.findByText(/clear all/i);
      await user.click(clearButton);
      
      expect(mockClear).toHaveBeenCalled();
    });
  });

  describe('Notification Types', () => {
    it('displays success notification with correct styling', async () => {
      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      await user.click(screen.getByRole('button', { name: /notification/i }));
      
      const successItem = await screen.findByText('Success');
      expect(successItem.closest('.notification-item')).toHaveClass('success');
    });

    it('displays error notification with correct styling', async () => {
      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      await user.click(screen.getByRole('button', { name: /notification/i }));
      
      const errorItem = await screen.findByText('Error');
      expect(errorItem.closest('.notification-item')).toHaveClass('error');
    });

    it('shows appropriate icon for each type', async () => {
      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      await user.click(screen.getByRole('button', { name: /notification/i }));
      
      // Check for type-specific icons
      const successIcon = screen.getByTestId('success-icon') || screen.getByRole('img', { name: /success/i });
      const errorIcon = screen.getByTestId('error-icon') || screen.getByRole('img', { name: /error/i });
      
      expect(successIcon).toBeInTheDocument();
      expect(errorIcon).toBeInTheDocument();
    });
  });

  describe('Empty State', () => {
    it('shows empty state when no notifications', async () => {
      (useUIStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
        notifications: [],
        unreadCount: 0,
        addNotification: vi.fn(),
        removeNotification: vi.fn(),
        markAsRead: vi.fn(),
        markAllAsRead: vi.fn(),
        clearNotifications: vi.fn(),
      });

      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      await user.click(screen.getByRole('button', { name: /notification/i }));
      
      await waitFor(() => {
        expect(screen.getByText(/no notifications/i)).toBeInTheDocument();
      });
    });
  });

  describe('Time Display', () => {
    it('shows relative time for notifications', async () => {
      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      await user.click(screen.getByRole('button', { name: /notification/i }));
      
      await waitFor(() => {
        // Should show "just now" or similar for recent notifications
        expect(screen.getByText(/just now|seconds ago|a moment ago/i)).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA attributes', () => {
      render(<NotificationCenter />);
      
      const button = screen.getByRole('button', { name: /notification/i });
      expect(button).toHaveAttribute('aria-haspopup', 'true');
      expect(button).toHaveAttribute('aria-expanded', 'false');
    });

    it('updates aria-expanded when opened', async () => {
      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      const button = screen.getByRole('button', { name: /notification/i });
      await user.click(button);
      
      expect(button).toHaveAttribute('aria-expanded', 'true');
    });

    it('announces unread count to screen readers', () => {
      render(<NotificationCenter />);
      
      const srText = screen.getByText(/2 unread notification/i, { selector: '.sr-only' });
      expect(srText).toBeInTheDocument();
    });

    it('supports keyboard navigation', async () => {
      const user = userEvent.setup();
      render(<NotificationCenter />);
      
      const button = screen.getByRole('button', { name: /notification/i });
      button.focus();
      
      // Open with Enter
      await user.keyboard('{Enter}');
      
      await waitFor(() => {
        expect(screen.getByText('Success')).toBeInTheDocument();
      });
      
      // Close with Escape
      await user.keyboard('{Escape}');
      
      await waitFor(() => {
        expect(screen.queryByText('Success')).not.toBeInTheDocument();
      });
    });
  });
});
