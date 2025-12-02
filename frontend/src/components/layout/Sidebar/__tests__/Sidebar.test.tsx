/**
 * Sidebar Component Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter, MemoryRouter } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { Sidebar } from '../Sidebar';

// Mock stores
const mockToggleSidebar = vi.fn();
const mockSetSidebarSearch = vi.fn();
const mockCloseMobileDrawer = vi.fn();
const mockAddFavorite = vi.fn();
const mockRemoveFavorite = vi.fn();
const mockLogout = vi.fn();

vi.mock('../../../../store/authStore', () => ({
  useAuthStore: vi.fn(() => ({
    user: {
      id: '1',
      name: 'Test User',
      email: 'test@example.com',
      role: 'admin',
      avatar: null,
    },
    logout: mockLogout,
  })),
}));

vi.mock('../../../../store/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    sidebar: {
      isOpen: true,
      width: 280,
      searchQuery: '',
      expandedKeys: [],
      favorites: [],
      mobileDrawerOpen: false,
    },
    toggleSidebar: mockToggleSidebar,
    setSidebarSearch: mockSetSidebarSearch,
    setSidebarExpandedKeys: vi.fn(),
    closeMobileDrawer: mockCloseMobileDrawer,
    addFavorite: mockAddFavorite,
    removeFavorite: mockRemoveFavorite,
  })),
}));

// Mock i18n
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key,
  }),
}));

// Test wrapper
const createWrapper = () => {
  return ({ children }: { children: React.ReactNode }) => (
    <ConfigProvider>
      <MemoryRouter>
        {children}
      </MemoryRouter>
    </ConfigProvider>
  );
};

describe('Sidebar Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the sidebar', async () => {
    render(<Sidebar />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByRole('navigation')).toBeInTheDocument();
    });
  });

  it('renders the logo', async () => {
    render(<Sidebar />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Code Review AI')).toBeInTheDocument();
    });
  });

  it('renders search input', async () => {
    render(<Sidebar />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search...')).toBeInTheDocument();
    });
  });

  it('renders navigation items', async () => {
    render(<Sidebar />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Projects')).toBeInTheDocument();
      expect(screen.getByText('Code Review')).toBeInTheDocument();
    });
  });

  it('renders admin menu for admin users', async () => {
    render(<Sidebar />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Administration')).toBeInTheDocument();
    });
  });

  it('renders user profile card', async () => {
    render(<Sidebar />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Test User')).toBeInTheDocument();
    });
  });

  it('calls toggleSidebar when collapse button clicked', async () => {
    const user = userEvent.setup();
    render(<Sidebar />, { wrapper: createWrapper() });
    
    const collapseBtn = screen.getByRole('button', { name: /collapse/i });
    await user.click(collapseBtn);
    
    expect(mockToggleSidebar).toHaveBeenCalled();
  });

  it('filters navigation items based on search', async () => {
    const user = userEvent.setup();
    render(<Sidebar />, { wrapper: createWrapper() });
    
    const searchInput = screen.getByPlaceholderText('Search...');
    await user.type(searchInput, 'dashboard');
    
    expect(mockSetSidebarSearch).toHaveBeenCalled();
  });
});

describe('Sidebar Collapsed State', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const { useUIStore } = require('../../../../store/uiStore');
    useUIStore.mockReturnValue({
      sidebar: {
        isOpen: false,
        width: 80,
        searchQuery: '',
        expandedKeys: [],
        favorites: [],
        mobileDrawerOpen: false,
      },
      toggleSidebar: mockToggleSidebar,
      setSidebarSearch: mockSetSidebarSearch,
      setSidebarExpandedKeys: vi.fn(),
      closeMobileDrawer: mockCloseMobileDrawer,
      addFavorite: mockAddFavorite,
      removeFavorite: mockRemoveFavorite,
    });
  });

  it('applies collapsed class', async () => {
    render(<Sidebar />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      const sidebar = document.querySelector('.sidebar');
      expect(sidebar).toHaveClass('sidebar--collapsed');
    });
  });

  it('hides search input when collapsed', async () => {
    render(<Sidebar />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.queryByPlaceholderText('Search...')).not.toBeInTheDocument();
    });
  });
});

describe('Sidebar Mobile Drawer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const { useUIStore } = require('../../../../store/uiStore');
    useUIStore.mockReturnValue({
      sidebar: {
        isOpen: true,
        width: 280,
        searchQuery: '',
        expandedKeys: [],
        favorites: [],
        mobileDrawerOpen: true,
      },
      toggleSidebar: mockToggleSidebar,
      setSidebarSearch: mockSetSidebarSearch,
      setSidebarExpandedKeys: vi.fn(),
      closeMobileDrawer: mockCloseMobileDrawer,
      addFavorite: mockAddFavorite,
      removeFavorite: mockRemoveFavorite,
    });
  });

  it('renders mobile drawer when open', async () => {
    render(<Sidebar />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(document.querySelector('.sidebar-drawer')).toBeInTheDocument();
    });
  });
});

describe('Sidebar Accessibility', () => {
  it('has proper navigation role', async () => {
    render(<Sidebar />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      const nav = screen.getByRole('navigation');
      expect(nav).toHaveAttribute('aria-label');
    });
  });

  it('collapse button has accessible label', async () => {
    render(<Sidebar />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      const collapseBtn = screen.getByRole('button', { name: /collapse sidebar/i });
      expect(collapseBtn).toBeInTheDocument();
    });
  });
});

describe('Sidebar Favorites', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const { useUIStore } = require('../../../../store/uiStore');
    useUIStore.mockReturnValue({
      sidebar: {
        isOpen: true,
        width: 280,
        searchQuery: '',
        expandedKeys: [],
        favorites: [
          { id: '1', label: 'Dashboard', path: '/dashboard', order: 0 },
        ],
        mobileDrawerOpen: false,
      },
      toggleSidebar: mockToggleSidebar,
      setSidebarSearch: mockSetSidebarSearch,
      setSidebarExpandedKeys: vi.fn(),
      closeMobileDrawer: mockCloseMobileDrawer,
      addFavorite: mockAddFavorite,
      removeFavorite: mockRemoveFavorite,
    });
  });

  it('renders favorites section when favorites exist', async () => {
    render(<Sidebar />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Favorites')).toBeInTheDocument();
    });
  });
});
