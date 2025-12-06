/**
 * NotificationCenter Component Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { NotificationCenter } from '../NotificationCenter';
import { useUIStore } from '../../../../store/uiStore';

// Mock the UI store
const mockRemoveNotification = vi.fn();
const mockAddNotification = vi.fn(() => 'test-id');
const mockClearNotifications = vi.fn();

vi.mock('../../../../store/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    notifications: [],
    notificationSettings: {
      position: 'top-right',
      maxVisible: 3,
      defaultDuration: 5000,
    },
    removeNotification: mockRemoveNotification,
    addNotification: mockAddNotification,
    clearNotifications: mockClearNotifications,
  })),
}));

describe('NotificationCenter Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset portal root
    const portalRoot = document.getElementById('notification-root');
    if (portalRoot) {
      document.body.removeChild(portalRoot);
    }
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders nothing when no notifications', () => {
    render(<NotificationCenter />);
    
    expect(document.querySelector('.notification-center')).not.toBeInTheDocument();
  });

  it('renders notifications when present', () => {
    (useUIStore as any).mockReturnValue({
      notifications: [
        {
          id: '1',
          type: 'success',
          title: 'Success!',
          message: 'Operation completed',
          timestamp: Date.now(),
        },
      ],
      notificationSettings: {
        position: 'top-right',
        maxVisible: 3,
        defaultDuration: 5000,
      },
      removeNotification: mockRemoveNotification,
    });

    render(<NotificationCenter />);
    
    // The component uses title as the displayed message (title || message)
    expect(screen.getByText('Success!')).toBeInTheDocument();
  });

  it('renders different notification types', () => {
    (useUIStore as any).mockReturnValue({
      notifications: [
        { id: '1', type: 'success', title: 'Success', timestamp: Date.now() },
        { id: '2', type: 'error', title: 'Error', timestamp: Date.now() },
        { id: '3', type: 'warning', title: 'Warning', timestamp: Date.now() },
      ],
      notificationSettings: {
        position: 'top-right',
        maxVisible: 3,
        defaultDuration: 5000,
      },
      removeNotification: mockRemoveNotification,
    });

    render(<NotificationCenter />);
    
    expect(screen.getByText('Success')).toBeInTheDocument();
    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(screen.getByText('Warning')).toBeInTheDocument();
  });

  it('limits visible notifications to maxVisible', () => {
    (useUIStore as any).mockReturnValue({
      notifications: [
        { id: '1', type: 'info', title: 'Notification 1', timestamp: Date.now() },
        { id: '2', type: 'info', title: 'Notification 2', timestamp: Date.now() },
        { id: '3', type: 'info', title: 'Notification 3', timestamp: Date.now() },
        { id: '4', type: 'info', title: 'Notification 4', timestamp: Date.now() },
        { id: '5', type: 'info', title: 'Notification 5', timestamp: Date.now() },
      ],
      notificationSettings: {
        position: 'top-right',
        maxVisible: 3,
        defaultDuration: 5000,
      },
      removeNotification: mockRemoveNotification,
    });

    render(<NotificationCenter />);
    
    // Should show +2 more indicator
    expect(screen.getByText(/\+2 more/)).toBeInTheDocument();
  });

  it('calls removeNotification when close button clicked', async () => {
    const user = userEvent.setup();
    
    (useUIStore as any).mockReturnValue({
      notifications: [
        {
          id: 'test-1',
          type: 'success',
          title: 'Test Notification',
          timestamp: Date.now(),
          dismissible: true,
        },
      ],
      notificationSettings: {
        position: 'top-right',
        maxVisible: 3,
        defaultDuration: 5000,
      },
      removeNotification: mockRemoveNotification,
    });

    render(<NotificationCenter />);
    
    // The close button has aria-label="Dismiss notification"
    const closeButton = screen.getByRole('button', { name: /dismiss notification/i });
    await user.click(closeButton);
    
    expect(mockRemoveNotification).toHaveBeenCalledWith('test-1');
  });

  it('renders action buttons when provided', () => {
    const mockAction = vi.fn();
    
    (useUIStore as any).mockReturnValue({
      notifications: [
        {
          id: '1',
          type: 'info',
          title: 'Action Required',
          timestamp: Date.now(),
          actions: [
            { label: 'Undo', onClick: mockAction },
          ],
        },
      ],
      notificationSettings: {
        position: 'top-right',
        maxVisible: 3,
        defaultDuration: 5000,
      },
      removeNotification: mockRemoveNotification,
    });

    render(<NotificationCenter />);
    
    expect(screen.getByText('Undo')).toBeInTheDocument();
  });

  it('applies correct position class', () => {
    (useUIStore as any).mockReturnValue({
      notifications: [
        { id: '1', type: 'info', title: 'Test', timestamp: Date.now() },
      ],
      notificationSettings: {
        position: 'bottom-center',
        maxVisible: 3,
        defaultDuration: 5000,
      },
      removeNotification: mockRemoveNotification,
    });

    render(<NotificationCenter />);
    
    const center = document.querySelector('.notification-center');
    expect(center).toHaveClass('notification-center--bottom-center');
  });
});

describe('NotificationCenter Accessibility', () => {
  it('has proper ARIA attributes', () => {
    (useUIStore as any).mockReturnValue({
      notifications: [
        { id: '1', type: 'error', title: 'Error!', timestamp: Date.now() },
      ],
      notificationSettings: {
        position: 'top-right',
        maxVisible: 3,
        defaultDuration: 5000,
      },
      removeNotification: mockRemoveNotification,
    });

    render(<NotificationCenter />);
    
    const alert = screen.getByRole('alert');
    expect(alert).toBeInTheDocument();
    expect(alert).toHaveAttribute('aria-live', 'assertive');
  });

  it('uses polite aria-live for non-error notifications', () => {
    (useUIStore as any).mockReturnValue({
      notifications: [
        { id: '1', type: 'success', title: 'Success!', timestamp: Date.now() },
      ],
      notificationSettings: {
        position: 'top-right',
        maxVisible: 3,
        defaultDuration: 5000,
      },
      removeNotification: mockRemoveNotification,
    });

    render(<NotificationCenter />);
    
    const alert = screen.getByRole('alert');
    expect(alert).toHaveAttribute('aria-live', 'polite');
  });
});
