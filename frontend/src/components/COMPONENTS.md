# Core Layout Components Documentation

## Overview

This document describes the core layout and navigation components for the AI Code Review Platform.

## Components

### 1. Layout (`/components/layout/Layout.tsx`)

Main application wrapper layout with header, sidebar, and content area.

#### Features
- Responsive design (Desktop/Tablet/Mobile breakpoints)
- Collapsible sidebar
- Breadcrumb navigation
- Global search
- Theme toggle (light/dark)
- Language selector
- Notifications badge
- User menu dropdown

#### Usage
```tsx
import { Layout } from '@/components/layout';

// Used as route element wrapper
<Route element={<Layout />}>
  <Route path="/dashboard" element={<Dashboard />} />
</Route>
```

#### Responsive Breakpoints
| Breakpoint | Width | Behavior |
|------------|-------|----------|
| Desktop | ≥1200px | Full layout with expanded sidebar |
| Tablet | 768-1199px | Collapsed sidebar (icons only) |
| Mobile | <768px | Hidden sidebar with drawer menu |

---

### 2. Sidebar (`/components/layout/Sidebar/`)

Enhanced navigation sidebar with search, favorites, and keyboard navigation.

#### Features
- Collapsible/expandable toggle
- Active route highlighting
- Search functionality (⌘K shortcut)
- Quick access favorites
- User profile mini card
- Admin menu (role-based)
- Mobile drawer support
- ARIA labels & keyboard navigation

#### Navigation Items
```typescript
const navItems = [
  { key: 'dashboard', label: 'Dashboard', path: '/dashboard' },
  { key: 'projects', label: 'Projects', children: [...] },
  { key: 'code-review', label: 'Code Review', path: '/review' },
  { key: 'reports', label: 'Reports', path: '/reports' },
  { key: 'settings', label: 'Settings', children: [...] },
  { key: 'admin', label: 'Administration', adminOnly: true, children: [...] },
];
```

---

### 3. ProtectedRoute (`/components/common/ProtectedRoute.tsx`)

Route guard component for authentication and authorization.

#### Interface
```typescript
interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRoles?: string[];       // Roles required (OR logic)
  requiredPermissions?: string[]; // Permissions required
  requireAll?: boolean;           // ALL permissions vs ANY
  redirectTo?: string;            // Redirect URL when not authenticated
  fallback?: React.ReactNode;     // Custom loading component
  onAccessDenied?: () => void;    // Callback when access denied
}
```

#### Usage
```tsx
// Basic authentication
<ProtectedRoute>
  <Dashboard />
</ProtectedRoute>

// Role-based access
<ProtectedRoute requiredRoles={['admin']}>
  <AdminPanel />
</ProtectedRoute>

// Permission-based access
<ProtectedRoute 
  requiredPermissions={['write:projects', 'delete:projects']}
  requireAll={true}
>
  <ProjectAdmin />
</ProtectedRoute>
```

#### Shortcut Components
```tsx
<AdminRoute>   // Requires 'admin' role
<UserRoute>    // Requires 'admin' or 'user' role
<PublicRoute>  // Only for unauthenticated users
```

---

### 4. NotificationCenter (`/components/common/NotificationCenter/`)

Global toast notification system.

#### Interface
```typescript
type NotificationType = 'success' | 'error' | 'warning' | 'info';

interface Notification {
  id: string;
  type: NotificationType;
  message: string;
  duration?: number;
  action?: { label: string; onClick: () => void };
}

interface NotificationOptions {
  duration?: number;
  action?: { label: string; onClick: () => void };
  dismissible?: boolean;
}
```

#### Usage
```tsx
import { useNotification } from '@/components/common/NotificationCenter';

const MyComponent = () => {
  const notify = useNotification();

  // Simple notifications
  notify.success('Profile updated successfully');
  notify.error('Failed to update profile', { duration: 5000 });

  // With action button
  notify.info('New message received', {
    action: { 
      label: 'View', 
      onClick: () => navigate('/messages') 
    }
  });

  // With custom duration
  notify.warning('Session expiring soon', { duration: 10000 });

  // Remove specific notification
  const id = notify.info('Processing...');
  notify.remove(id);

  // Clear all
  notify.clear();
};
```

#### Configuration
```typescript
// In uiStore
notificationSettings: {
  position: 'top-right',  // 'top-right' | 'top-center' | 'bottom-right' | 'bottom-center'
  maxVisible: 3,          // Maximum visible at once
  defaultDuration: 5000,  // Auto-dismiss time (ms)
}
```

---

### 5. ErrorBoundary (`/components/common/ErrorBoundary.tsx`)

Global error handling component.

#### Interface
```typescript
interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}
```

#### Features
- Catches JavaScript errors in component tree
- Logs to error logging service
- User-friendly error messages
- Recovery options (retry, reload, go home)
- Development mode stack trace
- WCAG 2.1 AA compliant

#### Usage
```tsx
<ErrorBoundary fallback={<CustomErrorPage />}>
  <App />
</ErrorBoundary>

// With error callback
<ErrorBoundary 
  onError={(error, info) => analytics.trackError(error)}
>
  <MyComponent />
</ErrorBoundary>
```

---

### 6. Error Logging Service (`/services/errorLogging.ts`)

Centralized error handling and logging.

#### Error Categories
```typescript
enum ErrorCategory {
  CLIENT = 'client',           // JavaScript errors
  SERVER = 'server',           // 5xx errors
  NETWORK = 'network',         // Connection issues
  AUTHENTICATION = 'authentication',  // 401 errors
  AUTHORIZATION = 'authorization',    // 403 errors
  VALIDATION = 'validation',   // 400 errors
  UNKNOWN = 'unknown',
}
```

#### Usage
```typescript
import { errorLoggingService, useErrorLogging } from '@/services/errorLogging';

// Direct usage
errorLoggingService.log(error, ErrorCategory.CLIENT);
errorLoggingService.logNetworkError(error, '/api/users', 'GET', 500);

// React hook
const { log, getUserMessage, getRecoveryStrategy } = useErrorLogging();
log(error);
const message = getUserMessage(error); // User-friendly message
const strategy = getRecoveryStrategy(error); // { type: 'retry', maxRetries: 3 }
```

#### Recovery Strategies
| Category | Strategy |
|----------|----------|
| NETWORK | Retry (3 attempts, 1s delay) |
| SERVER | Retry (2 attempts, 2s delay) |
| AUTHENTICATION | Redirect to /login |
| AUTHORIZATION | Redirect to /dashboard |
| VALIDATION | Ignore (show message) |
| DEFAULT | Refresh page |

---

## Accessibility (WCAG 2.1 AA)

All components follow accessibility best practices:

### Focus Management
- Visible focus indicators on all interactive elements
- Focus trap in modals and drawers
- Skip links for keyboard navigation

### Screen Readers
- Proper ARIA labels on navigation
- Live regions for notifications (polite/assertive)
- Descriptive button labels

### Keyboard Navigation
- Tab navigation through all interactive elements
- Escape to close modals/drawers
- Arrow keys for menu navigation
- ⌘K (Ctrl+K) for search

### Color Contrast
- All text meets 4.5:1 contrast ratio
- Focus indicators visible against all backgrounds
- Error states don't rely solely on color

---

## Performance Optimizations

### Lazy Loading
```tsx
// Route components
const Dashboard = lazy(() => import('./pages/Dashboard'));

// Heavy components
const CodeEditor = lazy(() => import('./components/CodeEditor'));
```

### Memoization
```tsx
// Memoized components
const NotificationItem = memo(({ ... }) => { ... });
const NotificationCenter = memo(() => { ... });

// Memoized values
const menuItems = useMemo(() => [...], [dependencies]);
```

### Code Splitting
- Route-based splitting for pages
- Component-based splitting for heavy features
- Vendor chunk optimization

---

## Testing

### Unit Tests
```bash
# Run all tests
npm test

# Run specific component tests
npm test -- --grep "Layout"
npm test -- --grep "NotificationCenter"
```

### Test Files
- `/components/layout/__tests__/Layout.test.tsx`
- `/components/layout/Sidebar/__tests__/Sidebar.test.tsx`
- `/components/common/NotificationCenter/__tests__/NotificationCenter.test.tsx`

### Coverage Areas
- Component rendering
- User interactions
- Accessibility (ARIA)
- Responsive behavior
- Error states
