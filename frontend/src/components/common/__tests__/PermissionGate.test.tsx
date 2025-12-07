/**
 * PermissionGate Component Tests
 *
 * Tests for role-based access control in the frontend:
 * 1. Admin functions are not visible to regular users
 * 2. Correct rendering based on user role
 * 3. Access denial logging
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { PermissionGate, AdminOnly, UserOnly, RoleBadge } from '../PermissionGate';
import { usePermissions } from '../../../hooks/usePermissions';

// Mock the usePermissions hook
jest.mock('../../../hooks/usePermissions');

const mockUsePermissions = usePermissions as jest.MockedFunction<typeof usePermissions>;

describe('PermissionGate', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Admin role tests', () => {
    beforeEach(() => {
      mockUsePermissions.mockReturnValue({
        user: { id: '1', email: 'admin@test.com', name: 'Admin', role: 'admin' },
        role: 'admin',
        isAuthenticated: true,
        isAdmin: true,
        isUser: true,
        isViewer: true,
        isGuest: false,
        hasRole: jest.fn().mockReturnValue(true),
        hasPermission: jest.fn().mockReturnValue(true),
        hasAnyPermission: jest.fn().mockReturnValue(true),
        hasAllPermissions: jest.fn().mockReturnValue(true),
        canAccess: jest.fn().mockReturnValue(true),
        canAccessPath: jest.fn().mockReturnValue(true),
        accessibleFeatures: [],
        adminFeatures: [],
        userFeatures: [],
        getPermissions: jest.fn().mockReturnValue(['admin:all']),
        getRoleLabel: jest.fn().mockReturnValue('Administrator'),
      });
    });

    it('renders children for admin user', () => {
      render(
        <PermissionGate requiredRole="admin">
          <div data-testid="admin-content">Admin Content</div>
        </PermissionGate>
      );

      expect(screen.getByTestId('admin-content')).toBeInTheDocument();
    });

    it('renders admin-only content for admin', () => {
      render(
        <AdminOnly>
          <div data-testid="admin-only">Admin Only Content</div>
        </AdminOnly>
      );

      expect(screen.getByTestId('admin-only')).toBeInTheDocument();
    });
  });

  describe('Regular user role tests', () => {
    beforeEach(() => {
      mockUsePermissions.mockReturnValue({
        user: { id: '2', email: 'user@test.com', name: 'User', role: 'user' },
        role: 'user',
        isAuthenticated: true,
        isAdmin: false,
        isUser: true,
        isViewer: true,
        isGuest: false,
        hasRole: jest.fn((role) => role === 'user' || role === 'viewer' || role === 'guest'),
        hasPermission: jest.fn((perm) => !perm.startsWith('admin:')),
        hasAnyPermission: jest.fn().mockReturnValue(true),
        hasAllPermissions: jest.fn().mockReturnValue(false),
        canAccess: jest.fn().mockReturnValue(true),
        canAccessPath: jest.fn((path) => !path.startsWith('/admin')),
        accessibleFeatures: [],
        adminFeatures: [],
        userFeatures: [],
        getPermissions: jest.fn().mockReturnValue(['project:read', 'project:write']),
        getRoleLabel: jest.fn().mockReturnValue('User'),
      });
    });

    it('hides admin content from regular user', () => {
      render(
        <PermissionGate requiredRole="admin">
          <div data-testid="admin-content">Admin Content</div>
        </PermissionGate>
      );

      expect(screen.queryByTestId('admin-content')).not.toBeInTheDocument();
    });

    it('shows fallback when access denied', () => {
      render(
        <PermissionGate
          requiredRole="admin"
          fallback={<div data-testid="fallback">Access Denied</div>}
        >
          <div data-testid="admin-content">Admin Content</div>
        </PermissionGate>
      );

      expect(screen.queryByTestId('admin-content')).not.toBeInTheDocument();
      expect(screen.getByTestId('fallback')).toBeInTheDocument();
    });

    it('hides admin-only content from regular user', () => {
      render(
        <AdminOnly>
          <div data-testid="admin-only">Admin Only Content</div>
        </AdminOnly>
      );

      expect(screen.queryByTestId('admin-only')).not.toBeInTheDocument();
    });

    it('shows user content to regular user', () => {
      render(
        <UserOnly>
          <div data-testid="user-content">User Content</div>
        </UserOnly>
      );

      expect(screen.getByTestId('user-content')).toBeInTheDocument();
    });
  });

  describe('Permission-based access', () => {
    beforeEach(() => {
      mockUsePermissions.mockReturnValue({
        user: { id: '2', email: 'user@test.com', name: 'User', role: 'user' },
        role: 'user',
        isAuthenticated: true,
        isAdmin: false,
        isUser: true,
        isViewer: true,
        isGuest: false,
        hasRole: jest.fn().mockReturnValue(true),
        hasPermission: jest.fn((perm) => perm === 'project:read' || perm === 'project:write'),
        hasAnyPermission: jest.fn((perms) => perms.some((p: string) => p === 'project:read')),
        hasAllPermissions: jest.fn((perms) => perms.every((p: string) => p.startsWith('project:'))),
        canAccess: jest.fn().mockReturnValue(true),
        canAccessPath: jest.fn().mockReturnValue(true),
        accessibleFeatures: [],
        adminFeatures: [],
        userFeatures: [],
        getPermissions: jest.fn().mockReturnValue(['project:read', 'project:write']),
        getRoleLabel: jest.fn().mockReturnValue('User'),
      });
    });

    it('shows content when user has required permission', () => {
      render(
        <PermissionGate requiredPermission="project:read">
          <div data-testid="permitted-content">Permitted Content</div>
        </PermissionGate>
      );

      expect(screen.getByTestId('permitted-content')).toBeInTheDocument();
    });

    it('hides content when user lacks required permission', () => {
      render(
        <PermissionGate requiredPermission="admin:users">
          <div data-testid="admin-content">Admin Content</div>
        </PermissionGate>
      );

      expect(screen.queryByTestId('admin-content')).not.toBeInTheDocument();
    });
  });

  describe('Guest user tests', () => {
    beforeEach(() => {
      mockUsePermissions.mockReturnValue({
        user: null,
        role: 'guest',
        isAuthenticated: false,
        isAdmin: false,
        isUser: false,
        isViewer: false,
        isGuest: true,
        hasRole: jest.fn().mockReturnValue(false),
        hasPermission: jest.fn().mockReturnValue(false),
        hasAnyPermission: jest.fn().mockReturnValue(false),
        hasAllPermissions: jest.fn().mockReturnValue(false),
        canAccess: jest.fn().mockReturnValue(false),
        canAccessPath: jest.fn().mockReturnValue(false),
        accessibleFeatures: [],
        adminFeatures: [],
        userFeatures: [],
        getPermissions: jest.fn().mockReturnValue([]),
        getRoleLabel: jest.fn().mockReturnValue('Guest'),
      });
    });

    it('hides all protected content from guest', () => {
      render(
        <>
          <PermissionGate requiredRole="admin">
            <div data-testid="admin-content">Admin</div>
          </PermissionGate>
          <PermissionGate requiredRole="user">
            <div data-testid="user-content">User</div>
          </PermissionGate>
        </>
      );

      expect(screen.queryByTestId('admin-content')).not.toBeInTheDocument();
      expect(screen.queryByTestId('user-content')).not.toBeInTheDocument();
    });
  });
});

describe('RoleBadge', () => {
  beforeEach(() => {
    mockUsePermissions.mockReturnValue({
      user: { id: '1', email: 'admin@test.com', name: 'Admin', role: 'admin' },
      role: 'admin',
      isAuthenticated: true,
      isAdmin: true,
      isUser: true,
      isViewer: true,
      isGuest: false,
      hasRole: jest.fn().mockReturnValue(true),
      hasPermission: jest.fn().mockReturnValue(true),
      hasAnyPermission: jest.fn().mockReturnValue(true),
      hasAllPermissions: jest.fn().mockReturnValue(true),
      canAccess: jest.fn().mockReturnValue(true),
      canAccessPath: jest.fn().mockReturnValue(true),
      accessibleFeatures: [],
      adminFeatures: [],
      userFeatures: [],
      getPermissions: jest.fn().mockReturnValue(['admin:all']),
      getRoleLabel: jest.fn().mockReturnValue('Administrator'),
    });
  });

  it('renders admin badge correctly', () => {
    render(<RoleBadge role="admin" />);
    expect(screen.getByText('Administrator')).toBeInTheDocument();
  });

  it('renders user badge correctly', () => {
    render(<RoleBadge role="user" />);
    expect(screen.getByText('User')).toBeInTheDocument();
  });

  it('renders current user role when no role prop', () => {
    render(<RoleBadge />);
    expect(screen.getByText('Administrator')).toBeInTheDocument();
  });
});
