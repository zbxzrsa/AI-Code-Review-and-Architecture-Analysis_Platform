/**
 * Two-Factor Authentication Component Tests
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TwoFactorVerify, TwoFactorSetup, TwoFactorDisable } from '../TwoFactorAuth';

// Mock i18n
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue: string) => defaultValue,
  }),
}));

// Mock API service
vi.mock('../../../../services/api', () => ({
  apiService: {
    user: {
      setup2FA: vi.fn().mockResolvedValue({
        data: {
          qr_code: 'data:image/png;base64,mock-qr-code',
          secret: 'JBSWY3DPEHPK3PXP',
          backup_codes: ['ABCD1234', 'EFGH5678', 'IJKL9012'],
        },
      }),
      enable2FA: vi.fn().mockResolvedValue({ data: { success: true } }),
      disable2FA: vi.fn().mockResolvedValue({ data: { success: true } }),
    },
  },
}));

// Mock security service
vi.mock('../../../../services/security', () => ({
  twoFactorAuth: {
    validateCodeFormat: (code: string) => /^\d{6}$/.test(code),
    validateBackupCodeFormat: (code: string) => code.length >= 8,
    formatBackupCode: (code: string) => `${code.slice(0, 4)}-${code.slice(4)}`,
  },
}));

describe('TwoFactorVerify', () => {
  const mockOnVerify = vi.fn();
  const mockOnCancel = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders verification form', () => {
    render(
      <TwoFactorVerify
        onVerify={mockOnVerify}
        onCancel={mockOnCancel}
      />
    );

    expect(screen.getByText('Two-Factor Authentication')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /verify/i })).toBeInTheDocument();
  });

  it('has 6 input fields for code entry', () => {
    render(
      <TwoFactorVerify
        onVerify={mockOnVerify}
        onCancel={mockOnCancel}
      />
    );

    const inputs = screen.getAllByRole('textbox');
    expect(inputs).toHaveLength(6);
  });

  it('focuses first input on mount', async () => {
    render(
      <TwoFactorVerify
        onVerify={mockOnVerify}
        onCancel={mockOnCancel}
      />
    );

    await waitFor(() => {
      const inputs = screen.getAllByRole('textbox');
      expect(document.activeElement).toBe(inputs[0]);
    });
  });

  it('moves focus to next input on digit entry', async () => {
    const user = userEvent.setup();
    
    render(
      <TwoFactorVerify
        onVerify={mockOnVerify}
        onCancel={mockOnCancel}
      />
    );

    const inputs = screen.getAllByRole('textbox');
    
    await user.type(inputs[0], '1');
    expect(document.activeElement).toBe(inputs[1]);
  });

  it('shows error when code is incomplete', async () => {
    const user = userEvent.setup();
    mockOnVerify.mockResolvedValue({ success: false, error: 'Invalid code' });

    render(
      <TwoFactorVerify
        onVerify={mockOnVerify}
        onCancel={mockOnCancel}
      />
    );

    const verifyButton = screen.getByRole('button', { name: /verify/i });
    await user.click(verifyButton);

    // Should show error for incomplete code
    expect(await screen.findByText(/enter all 6 digits/i)).toBeInTheDocument();
  });

  it('calls onVerify with full code', async () => {
    const user = userEvent.setup();
    mockOnVerify.mockResolvedValue({ success: true });

    render(
      <TwoFactorVerify
        onVerify={mockOnVerify}
        onCancel={mockOnCancel}
      />
    );

    const inputs = screen.getAllByRole('textbox');
    
    // Enter full code
    await user.type(inputs[0], '123456');
    
    await waitFor(() => {
      expect(mockOnVerify).toHaveBeenCalledWith('123456', false);
    });
  });

  it('allows switching to backup code mode', async () => {
    const user = userEvent.setup();

    render(
      <TwoFactorVerify
        onVerify={mockOnVerify}
        onCancel={mockOnCancel}
      />
    );

    const backupLink = screen.getByText(/use a backup code/i);
    await user.click(backupLink);

    expect(screen.getByPlaceholderText(/enter backup code/i)).toBeInTheDocument();
  });

  it('calls onCancel when cancel is clicked', async () => {
    const user = userEvent.setup();

    render(
      <TwoFactorVerify
        onVerify={mockOnVerify}
        onCancel={mockOnCancel}
      />
    );

    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    await user.click(cancelButton);

    expect(mockOnCancel).toHaveBeenCalled();
  });

  it('handles paste of full code', async () => {
    const user = userEvent.setup();
    mockOnVerify.mockResolvedValue({ success: true });

    render(
      <TwoFactorVerify
        onVerify={mockOnVerify}
        onCancel={mockOnCancel}
      />
    );

    const inputs = screen.getAllByRole('textbox');
    
    // Simulate paste
    await user.click(inputs[0]);
    await user.paste('123456');

    await waitFor(() => {
      expect(mockOnVerify).toHaveBeenCalledWith('123456', false);
    });
  });

  it('shows loading state during verification', async () => {
    render(
      <TwoFactorVerify
        onVerify={mockOnVerify}
        onCancel={mockOnCancel}
        loading={true}
      />
    );

    const verifyButton = screen.getByRole('button', { name: /verify/i });
    // Ant Design loading buttons have the loading class rather than disabled attribute
    expect(verifyButton).toHaveClass('ant-btn-loading');
  });
});

describe('TwoFactorSetup', () => {
  const mockOnComplete = vi.fn();
  const mockOnCancel = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders setup screen with QR code', async () => {
    render(
      <TwoFactorSetup
        onComplete={mockOnComplete}
        onCancel={mockOnCancel}
      />
    );

    // Should show loading initially, then QR code
    await waitFor(() => {
      expect(screen.getByText('Set Up Authenticator')).toBeInTheDocument();
    });
  });

  it('displays secret for manual entry', async () => {
    render(
      <TwoFactorSetup
        onComplete={mockOnComplete}
        onCancel={mockOnCancel}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/can't scan/i)).toBeInTheDocument();
    });
  });

  it('progresses to verify step', async () => {
    const user = userEvent.setup();

    render(
      <TwoFactorSetup
        onComplete={mockOnComplete}
        onCancel={mockOnCancel}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Set Up Authenticator')).toBeInTheDocument();
    });

    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    expect(screen.getByText('Verify Setup')).toBeInTheDocument();
  });

  it('allows going back from verify step', async () => {
    const user = userEvent.setup();

    render(
      <TwoFactorSetup
        onComplete={mockOnComplete}
        onCancel={mockOnCancel}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Set Up Authenticator')).toBeInTheDocument();
    });

    // Go to verify step
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    // Go back
    const backButton = screen.getByRole('button', { name: /back/i });
    await user.click(backButton);

    expect(screen.getByText('Set Up Authenticator')).toBeInTheDocument();
  });

  it('calls onCancel when cancelled', async () => {
    const user = userEvent.setup();

    render(
      <TwoFactorSetup
        onComplete={mockOnComplete}
        onCancel={mockOnCancel}
      />
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
    });

    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    await user.click(cancelButton);

    expect(mockOnCancel).toHaveBeenCalled();
  });
});

describe('TwoFactorDisable', () => {
  const mockOnClose = vi.fn();
  const mockOnDisabled = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders disable modal when visible', () => {
    render(
      <TwoFactorDisable
        visible={true}
        onClose={mockOnClose}
        onDisabled={mockOnDisabled}
      />
    );

    expect(screen.getByText('Disable Two-Factor Authentication')).toBeInTheDocument();
  });

  it('does not render when not visible', () => {
    render(
      <TwoFactorDisable
        visible={false}
        onClose={mockOnClose}
        onDisabled={mockOnDisabled}
      />
    );

    expect(screen.queryByText('Disable Two-Factor Authentication')).not.toBeInTheDocument();
  });

  it('shows warning about reduced security', () => {
    render(
      <TwoFactorDisable
        visible={true}
        onClose={mockOnClose}
        onDisabled={mockOnDisabled}
      />
    );

    expect(screen.getByText(/less secure/i)).toBeInTheDocument();
  });

  it('requires verification code and password', () => {
    render(
      <TwoFactorDisable
        visible={true}
        onClose={mockOnClose}
        onDisabled={mockOnDisabled}
      />
    );

    expect(screen.getByPlaceholderText(/verification code/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/password/i)).toBeInTheDocument();
  });

  it('calls onClose when cancelled', async () => {
    const user = userEvent.setup();

    render(
      <TwoFactorDisable
        visible={true}
        onClose={mockOnClose}
        onDisabled={mockOnDisabled}
      />
    );

    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    await user.click(cancelButton);

    expect(mockOnClose).toHaveBeenCalled();
  });

  it('has a danger button to disable 2FA', () => {
    render(
      <TwoFactorDisable
        visible={true}
        onClose={mockOnClose}
        onDisabled={mockOnDisabled}
      />
    );

    const disableButton = screen.getByRole('button', { name: /disable 2fa/i });
    expect(disableButton).toBeInTheDocument();
    expect(disableButton).toHaveClass('ant-btn-dangerous');
  });
});

describe('Accessibility', () => {
  it('TwoFactorVerify has proper aria labels', () => {
    render(
      <TwoFactorVerify
        onVerify={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    const inputs = screen.getAllByRole('textbox');
    inputs.forEach((input, index) => {
      expect(input).toHaveAttribute('aria-label', `Digit ${index + 1}`);
    });
  });

  it('error messages are announced', async () => {
    const user = userEvent.setup();
    const mockOnVerify = vi.fn().mockResolvedValue({ success: false, error: 'Invalid code' });

    render(
      <TwoFactorVerify
        onVerify={mockOnVerify}
        onCancel={vi.fn()}
      />
    );

    const verifyButton = screen.getByRole('button', { name: /verify/i });
    await user.click(verifyButton);

    const alert = await screen.findByRole('alert');
    expect(alert).toBeInTheDocument();
  });
});
