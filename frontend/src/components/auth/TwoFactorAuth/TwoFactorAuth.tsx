/**
 * Two-Factor Authentication Components
 * 
 * Provides:
 * - 2FA verification during login
 * - 2FA setup with QR code
 * - Backup codes management
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Card,
  Input,
  Button,
  Typography,
  Space,
  Alert,
  Divider,
  Modal,
  List,
  message,
  Spin,
} from 'antd';
import {
  LockOutlined,
  SafetyOutlined,
  MobileOutlined,
  KeyOutlined,
  CopyOutlined,
  DownloadOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { apiService } from '../../../services/api';
import { twoFactorAuth } from '../../../services/security';
import './TwoFactorAuth.css';

const { Title, Text, Paragraph } = Typography;

// ============================================
// 2FA Verification Component (During Login)
// ============================================

interface TwoFactorVerifyProps {
  onVerify: (code: string, isBackupCode?: boolean) => Promise<{ success: boolean; error?: string }>;
  onCancel: () => void;
  onResend?: () => Promise<void>;
  loading?: boolean;
}

export const TwoFactorVerify: React.FC<TwoFactorVerifyProps> = ({
  onVerify,
  onCancel,
  onResend,
  loading = false,
}) => {
  const { t } = useTranslation();
  const [code, setCode] = useState(['', '', '', '', '', '']);
  const [error, setError] = useState<string | null>(null);
  const [isBackupMode, setIsBackupMode] = useState(false);
  const [backupCode, setBackupCode] = useState('');
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);

  // Focus first input on mount
  useEffect(() => {
    inputRefs.current[0]?.focus();
  }, []);

  // Handle digit input
  const handleDigitChange = (index: number, value: string) => {
    if (!/^\d*$/.test(value)) return;

    const newCode = [...code];
    newCode[index] = value.slice(-1);
    setCode(newCode);

    // Auto-focus next input
    if (value && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }

    // Auto-submit when complete
    if (index === 5 && value) {
      const fullCode = newCode.join('');
      if (fullCode.length === 6) {
        handleSubmit(fullCode);
      }
    }
  };

  // Handle backspace
  const handleKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === 'Backspace' && !code[index] && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }
  };

  // Handle paste
  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pastedData = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6);
    const newCode = [...code];
    
    for (let i = 0; i < pastedData.length; i++) {
      newCode[i] = pastedData[i];
    }
    
    setCode(newCode);
    
    if (pastedData.length === 6) {
      handleSubmit(pastedData);
    } else {
      inputRefs.current[pastedData.length]?.focus();
    }
  };

  // Submit verification
  const handleSubmit = async (codeStr?: string) => {
    const verifyCode = codeStr || code.join('');
    setError(null);

    if (verifyCode.length !== 6) {
      setError('Please enter all 6 digits');
      return;
    }

    const result = await onVerify(verifyCode, false);
    if (!result.success) {
      setError(result.error || 'Verification failed');
      setCode(['', '', '', '', '', '']);
      inputRefs.current[0]?.focus();
    }
  };

  // Submit backup code
  const handleBackupSubmit = async () => {
    if (!backupCode.trim()) {
      setError('Please enter your backup code');
      return;
    }

    const result = await onVerify(backupCode.trim(), true);
    if (!result.success) {
      setError(result.error || 'Invalid backup code');
    }
  };

  return (
    <Card className="two-factor-verify">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <div className="two-factor-header">
          <SafetyOutlined className="two-factor-icon" />
          <Title level={4}>{t('auth.2fa.title', 'Two-Factor Authentication')}</Title>
          <Text type="secondary">
            {isBackupMode
              ? t('auth.2fa.backup_prompt', 'Enter one of your backup codes')
              : t('auth.2fa.code_prompt', 'Enter the 6-digit code from your authenticator app')}
          </Text>
        </div>

        {error && (
          <Alert
            type="error"
            message={error}
            showIcon
            closable
            onClose={() => setError(null)}
          />
        )}

        {!isBackupMode ? (
          <>
            <div className="code-input-group" onPaste={handlePaste}>
              {code.map((digit, index) => (
                <Input
                  key={index}
                  ref={(el) => (inputRefs.current[index] = el?.input || null)}
                  value={digit}
                  onChange={(e) => handleDigitChange(index, e.target.value)}
                  onKeyDown={(e) => handleKeyDown(index, e)}
                  maxLength={1}
                  className="code-input"
                  disabled={loading}
                  aria-label={`Digit ${index + 1}`}
                />
              ))}
            </div>

            <Button
              type="primary"
              size="large"
              block
              loading={loading}
              onClick={() => handleSubmit()}
              icon={<LockOutlined />}
            >
              {t('auth.2fa.verify', 'Verify')}
            </Button>

            <div className="two-factor-links">
              <Button type="link" onClick={() => setIsBackupMode(true)}>
                {t('auth.2fa.use_backup', 'Use a backup code')}
              </Button>
              {onResend && (
                <Button type="link" onClick={onResend}>
                  {t('auth.2fa.resend', 'Resend code')}
                </Button>
              )}
            </div>
          </>
        ) : (
          <>
            <Input
              size="large"
              placeholder={t('auth.2fa.backup_placeholder', 'Enter backup code')}
              value={backupCode}
              onChange={(e) => setBackupCode(e.target.value)}
              prefix={<KeyOutlined />}
              disabled={loading}
            />

            <Button
              type="primary"
              size="large"
              block
              loading={loading}
              onClick={handleBackupSubmit}
              icon={<LockOutlined />}
            >
              {t('auth.2fa.verify_backup', 'Verify Backup Code')}
            </Button>

            <Button type="link" onClick={() => setIsBackupMode(false)}>
              {t('auth.2fa.use_app', 'Use authenticator app')}
            </Button>
          </>
        )}

        <Divider />

        <Button block onClick={onCancel}>
          {t('common.cancel', 'Cancel')}
        </Button>
      </Space>
    </Card>
  );
};

// ============================================
// 2FA Setup Component
// ============================================

interface TwoFactorSetupProps {
  onComplete: () => void;
  onCancel: () => void;
}

export const TwoFactorSetup: React.FC<TwoFactorSetupProps> = ({
  onComplete,
  onCancel,
}) => {
  const { t } = useTranslation();
  const [step, setStep] = useState<'qr' | 'verify' | 'backup'>('qr');
  const [setupData, setSetupData] = useState<{
    qrCode: string;
    secret: string;
    backupCodes: string[];
  } | null>(null);
  const [verifyCode, setVerifyCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch setup data
  useEffect(() => {
    const fetchSetup = async () => {
      try {
        setLoading(true);
        const response = await apiService.user.setup2FA();
        setSetupData({
          qrCode: response.data.qr_code,
          secret: response.data.secret,
          backupCodes: response.data.backup_codes,
        });
      } catch (error) {
        setError('Failed to initialize 2FA setup');
      } finally {
        setLoading(false);
      }
    };

    fetchSetup();
  }, []);

  // Verify and enable 2FA
  const handleVerify = async () => {
    if (!twoFactorAuth.validateCodeFormat(verifyCode)) {
      setError('Please enter a valid 6-digit code');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      await apiService.user.enable2FA(verifyCode);
      setStep('backup');
      message.success('Two-factor authentication enabled!');
    } catch (error: any) {
      setError(error.response?.data?.message || 'Invalid verification code');
    } finally {
      setLoading(false);
    }
  };

  // Copy secret to clipboard
  const copySecret = () => {
    navigator.clipboard.writeText(setupData?.secret || '');
    message.success('Secret copied to clipboard');
  };

  // Copy backup codes
  const copyBackupCodes = () => {
    const codes = setupData?.backupCodes.join('\n') || '';
    navigator.clipboard.writeText(codes);
    message.success('Backup codes copied to clipboard');
  };

  // Download backup codes
  const downloadBackupCodes = () => {
    const codes = setupData?.backupCodes.join('\n') || '';
    const blob = new Blob([codes], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'backup-codes.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading && !setupData) {
    return (
      <Card className="two-factor-setup">
        <div className="loading-state">
          <Spin size="large" />
          <Text>{t('auth.2fa.loading', 'Setting up two-factor authentication...')}</Text>
        </div>
      </Card>
    );
  }

  return (
    <Card className="two-factor-setup">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {step === 'qr' && (
          <>
            <div className="setup-header">
              <MobileOutlined className="setup-icon" />
              <Title level={4}>{t('auth.2fa.setup_title', 'Set Up Authenticator')}</Title>
              <Text type="secondary">
                {t('auth.2fa.setup_desc', 'Scan the QR code with your authenticator app (Google Authenticator, Authy, etc.)')}
              </Text>
            </div>

            {setupData?.qrCode && (
              <div className="qr-code-container">
                <img src={setupData.qrCode} alt="2FA QR Code" className="qr-code" />
              </div>
            )}

            <div className="manual-entry">
              <Text type="secondary">{t('auth.2fa.cant_scan', "Can't scan? Enter this code manually:")}</Text>
              <div className="secret-display">
                <code>{setupData?.secret}</code>
                <Button
                  type="text"
                  icon={<CopyOutlined />}
                  onClick={copySecret}
                  size="small"
                />
              </div>
            </div>

            <Button type="primary" block onClick={() => setStep('verify')}>
              {t('auth.2fa.next', 'Next: Verify Code')}
            </Button>
          </>
        )}

        {step === 'verify' && (
          <>
            <div className="setup-header">
              <SafetyOutlined className="setup-icon" />
              <Title level={4}>{t('auth.2fa.verify_title', 'Verify Setup')}</Title>
              <Text type="secondary">
                {t('auth.2fa.verify_desc', 'Enter the 6-digit code from your authenticator app to verify setup')}
              </Text>
            </div>

            {error && (
              <Alert type="error" message={error} showIcon closable onClose={() => setError(null)} />
            )}

            <Input
              size="large"
              placeholder="000000"
              value={verifyCode}
              onChange={(e) => setVerifyCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
              maxLength={6}
              style={{ textAlign: 'center', letterSpacing: '0.5em', fontSize: '1.5em' }}
            />

            <Space style={{ width: '100%' }}>
              <Button block onClick={() => setStep('qr')}>
                {t('common.back', 'Back')}
              </Button>
              <Button type="primary" block onClick={handleVerify} loading={loading}>
                {t('auth.2fa.enable', 'Enable 2FA')}
              </Button>
            </Space>
          </>
        )}

        {step === 'backup' && (
          <>
            <div className="setup-header">
              <CheckCircleOutlined className="setup-icon success" />
              <Title level={4}>{t('auth.2fa.backup_title', 'Save Your Backup Codes')}</Title>
              <Text type="secondary">
                {t('auth.2fa.backup_desc', 'Save these backup codes in a secure place. You can use them to access your account if you lose your authenticator device.')}
              </Text>
            </div>

            <Alert
              type="warning"
              message={t('auth.2fa.backup_warning', 'Each backup code can only be used once. Keep them safe!')}
              showIcon
            />

            <List
              className="backup-codes-list"
              bordered
              dataSource={setupData?.backupCodes || []}
              renderItem={(code) => (
                <List.Item>
                  <code>{twoFactorAuth.formatBackupCode(code)}</code>
                </List.Item>
              )}
            />

            <Space style={{ width: '100%' }}>
              <Button icon={<CopyOutlined />} onClick={copyBackupCodes} block>
                {t('common.copy', 'Copy')}
              </Button>
              <Button icon={<DownloadOutlined />} onClick={downloadBackupCodes} block>
                {t('common.download', 'Download')}
              </Button>
            </Space>

            <Button type="primary" block onClick={onComplete}>
              {t('auth.2fa.done', "I've Saved My Codes")}
            </Button>
          </>
        )}

        {step !== 'backup' && (
          <>
            <Divider />
            <Button block onClick={onCancel}>
              {t('common.cancel', 'Cancel')}
            </Button>
          </>
        )}
      </Space>
    </Card>
  );
};

// ============================================
// 2FA Disable Modal
// ============================================

interface TwoFactorDisableProps {
  visible: boolean;
  onClose: () => void;
  onDisabled: () => void;
}

export const TwoFactorDisable: React.FC<TwoFactorDisableProps> = ({
  visible,
  onClose,
  onDisabled,
}) => {
  const { t } = useTranslation();
  const [code, setCode] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDisable = async () => {
    if (!code || !password) {
      setError('Please enter both the verification code and your password');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      await apiService.user.disable2FA(code, password);
      message.success('Two-factor authentication disabled');
      onDisabled();
      onClose();
    } catch (error: any) {
      setError(error.response?.data?.message || 'Failed to disable 2FA');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      title={t('auth.2fa.disable_title', 'Disable Two-Factor Authentication')}
      open={visible}
      onCancel={onClose}
      footer={null}
    >
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <Alert
          type="warning"
          message={t('auth.2fa.disable_warning', 'This will make your account less secure. Are you sure?')}
          showIcon
        />

        {error && (
          <Alert type="error" message={error} showIcon closable onClose={() => setError(null)} />
        )}

        <Input
          placeholder={t('auth.2fa.code_placeholder', 'Verification code')}
          value={code}
          onChange={(e) => setCode(e.target.value)}
          prefix={<SafetyOutlined />}
        />

        <Input.Password
          placeholder={t('auth.password', 'Password')}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          prefix={<LockOutlined />}
        />

        <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
          <Button onClick={onClose}>{t('common.cancel', 'Cancel')}</Button>
          <Button type="primary" danger onClick={handleDisable} loading={loading}>
            {t('auth.2fa.disable', 'Disable 2FA')}
          </Button>
        </Space>
      </Space>
    </Modal>
  );
};

export default {
  TwoFactorVerify,
  TwoFactorSetup,
  TwoFactorDisable,
};
