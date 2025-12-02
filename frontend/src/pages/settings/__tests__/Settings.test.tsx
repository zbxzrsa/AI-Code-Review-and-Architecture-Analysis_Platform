/**
 * Settings Page Tests
 * 
 * Unit tests for the enhanced settings page components.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import { Settings } from '../Settings';

// Mock the hooks
vi.mock('../../../hooks/useUser', () => ({
  useUserSettings: vi.fn(() => ({
    data: {
      theme: 'system',
      language: 'en',
      editorFontSize: 14,
      editorTabSize: 2,
      editorLineNumbers: true,
      editorMinimap: true,
      editorWordWrap: true,
      defaultAiModel: 'gpt-4',
      defaultAnalysisDepth: 'standard',
      autoAnalyzeOnPush: false,
      privacy: {},
      notifications: {
        emailAnalysisComplete: true,
        emailNewIssues: true,
        emailWeeklyDigest: false,
        digestFrequency: 'immediate',
        inAppNotifications: true,
        desktopNotifications: true,
        dndEnabled: false,
      },
    },
    isLoading: false,
    isError: false,
  })),
  useUpdateSettings: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useUpdateNotifications: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useChangePassword: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  use2FAStatus: vi.fn(() => ({
    data: { enabled: false, backupCodesRemaining: 0 },
    isLoading: false,
  })),
  useSetup2FA: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useEnable2FA: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useDisable2FA: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useRegenerateBackupCodes: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useSessions: vi.fn(() => ({
    data: [
      {
        id: '1',
        device: 'Chrome on Windows',
        browser: 'Chrome',
        os: 'Windows 11',
        ip: '192.168.1.1',
        location: 'Vietnam',
        lastActive: new Date().toISOString(),
        createdAt: new Date().toISOString(),
        isCurrent: true,
      },
    ],
    isLoading: false,
  })),
  useRevokeSession: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useRevokeAllSessions: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useApiKeys: vi.fn(() => ({
    data: [
      {
        id: '1',
        name: 'Test Key',
        prefix: 'sk-test-****',
        permissions: ['read:projects'],
        createdAt: '2024-01-01T00:00:00Z',
        usageCount: 100,
      },
    ],
    isLoading: false,
  })),
  useCreateApiKey: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useRevokeApiKey: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useIntegrations: vi.fn(() => ({
    data: [
      { type: 'slack', connected: false },
      { type: 'teams', connected: false },
    ],
    isLoading: false,
  })),
  useConnectSlack: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useDisconnectSlack: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useConnectTeams: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useDisconnectTeams: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useUserWebhooks: vi.fn(() => ({
    data: [],
    isLoading: false,
  })),
  useCreateUserWebhook: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useUpdateUserWebhook: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useDeleteUserWebhook: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useTestUserWebhook: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useIpWhitelist: vi.fn(() => ({
    data: [],
    isLoading: false,
  })),
  useAddIpToWhitelist: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useRemoveIpFromWhitelist: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
  useLoginAlerts: vi.fn(() => ({
    data: {
      emailOnNewDevice: true,
      emailOnNewLocation: true,
      emailOnFailedAttempts: true,
    },
    isLoading: false,
  })),
  useUpdateLoginAlerts: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
  })),
}));

// Mock i18n
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key,
    i18n: { 
      language: 'en',
      changeLanguage: vi.fn(),
    },
  }),
}));

// Mock the stores
vi.mock('../../../store/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    theme: 'system',
    setTheme: vi.fn(),
    language: 'en',
    setLanguage: vi.fn(),
  })),
}));

vi.mock('../../../store/authStore', async () => {
  const actual = await vi.importActual('../../../store/authStore');
  return {
    ...actual,
    useAuthStore: vi.fn(() => ({
      user: { id: '1', email: 'test@example.com', name: 'Test User' },
      settings: {
        editorFontSize: 14,
        editorTabSize: 2,
        editorLineNumbers: true,
        editorMinimap: true,
        editorWordWrap: true,
        defaultAiModel: 'gpt-4',
        defaultAnalysisDepth: 'standard',
        notifications: {
          emailAnalysisComplete: true,
          emailNewIssues: true,
          digestFrequency: 'immediate',
          inAppNotifications: true,
          desktopNotifications: true,
          dndEnabled: false,
        },
      },
      updateSettings: vi.fn(),
    })),
    defaultUserSettings: {
      theme: 'system',
      language: 'en',
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

describe('Settings Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the settings page with title', async () => {
    render(<Settings />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Settings')).toBeInTheDocument();
    });
  });

  it('renders all setting tabs', async () => {
    render(<Settings />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Preferences')).toBeInTheDocument();
      expect(screen.getByText('Security')).toBeInTheDocument();
      expect(screen.getByText('API Keys')).toBeInTheDocument();
      expect(screen.getByText('Integrations')).toBeInTheDocument();
      expect(screen.getByText('Notifications')).toBeInTheDocument();
    });
  });
});

describe('Preferences Tab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders appearance settings', async () => {
    render(<Settings />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Appearance')).toBeInTheDocument();
      expect(screen.getByText('Theme')).toBeInTheDocument();
      expect(screen.getByText('Language')).toBeInTheDocument();
    });
  });

  it('renders editor preferences', async () => {
    render(<Settings />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Editor Preferences')).toBeInTheDocument();
      expect(screen.getByText('Font Size')).toBeInTheDocument();
      expect(screen.getByText('Tab Size')).toBeInTheDocument();
    });
  });

  it('renders analysis defaults', async () => {
    render(<Settings />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Analysis Defaults')).toBeInTheDocument();
      expect(screen.getByText('Default AI Model')).toBeInTheDocument();
    });
  });
});

describe('Security Tab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders password change form', async () => {
    const user = userEvent.setup();
    render(<Settings />, { wrapper: createWrapper() });
    
    // Click on Security tab
    await waitFor(() => {
      expect(screen.getByText('Security')).toBeInTheDocument();
    });
    
    await user.click(screen.getByText('Security'));
    
    await waitFor(() => {
      expect(screen.getByText('Change Password')).toBeInTheDocument();
      expect(screen.getByText('Current Password')).toBeInTheDocument();
      expect(screen.getByText('New Password')).toBeInTheDocument();
    });
  });

  it('renders 2FA section', async () => {
    const user = userEvent.setup();
    render(<Settings />, { wrapper: createWrapper() });
    
    await user.click(screen.getByText('Security'));
    
    await waitFor(() => {
      expect(screen.getByText('Two-Factor Authentication')).toBeInTheDocument();
    });
  });

  it('renders active sessions', async () => {
    const user = userEvent.setup();
    render(<Settings />, { wrapper: createWrapper() });
    
    await user.click(screen.getByText('Security'));
    
    await waitFor(() => {
      expect(screen.getByText('Active Sessions')).toBeInTheDocument();
      expect(screen.getByText('Chrome')).toBeInTheDocument();
    });
  });
});

describe('API Keys Tab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders API keys list', async () => {
    const user = userEvent.setup();
    render(<Settings />, { wrapper: createWrapper() });
    
    await user.click(screen.getByText('API Keys'));
    
    await waitFor(() => {
      expect(screen.getByText('Test Key')).toBeInTheDocument();
      expect(screen.getByText('Create API Key')).toBeInTheDocument();
    });
  });

  it('shows create API key button', async () => {
    const user = userEvent.setup();
    render(<Settings />, { wrapper: createWrapper() });
    
    await user.click(screen.getByText('API Keys'));
    
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /create api key/i })).toBeInTheDocument();
    });
  });
});

describe('Integrations Tab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders integration options', async () => {
    const user = userEvent.setup();
    render(<Settings />, { wrapper: createWrapper() });
    
    await user.click(screen.getByText('Integrations'));
    
    await waitFor(() => {
      expect(screen.getByText('Slack')).toBeInTheDocument();
      expect(screen.getByText('Microsoft Teams')).toBeInTheDocument();
    });
  });

  it('renders webhooks section', async () => {
    const user = userEvent.setup();
    render(<Settings />, { wrapper: createWrapper() });
    
    await user.click(screen.getByText('Integrations'));
    
    await waitFor(() => {
      expect(screen.getByText('Webhooks')).toBeInTheDocument();
      expect(screen.getByText('Add Webhook')).toBeInTheDocument();
    });
  });
});

describe('Notifications Tab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders notification preferences', async () => {
    const user = userEvent.setup();
    render(<Settings />, { wrapper: createWrapper() });
    
    await user.click(screen.getByText('Notifications'));
    
    await waitFor(() => {
      expect(screen.getByText('Email Notifications')).toBeInTheDocument();
      expect(screen.getByText('In-App Notifications')).toBeInTheDocument();
    });
  });

  it('renders DND settings', async () => {
    const user = userEvent.setup();
    render(<Settings />, { wrapper: createWrapper() });
    
    await user.click(screen.getByText('Notifications'));
    
    await waitFor(() => {
      expect(screen.getByText('Do Not Disturb')).toBeInTheDocument();
    });
  });
});

describe('Settings Accessibility', () => {
  it('has proper ARIA labels', async () => {
    render(<Settings />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByRole('main', { name: /settings/i })).toBeInTheDocument();
    });
  });

  it('has keyboard navigable tabs', async () => {
    render(<Settings />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      const tabs = screen.getAllByRole('tab');
      expect(tabs.length).toBeGreaterThan(0);
    });
  });
});
