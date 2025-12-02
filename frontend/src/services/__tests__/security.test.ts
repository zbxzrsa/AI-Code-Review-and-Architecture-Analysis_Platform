/**
 * Security Service Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  csrfManager,
  rateLimiter,
  twoFactorAuth,
  sessionSecurity,
  validatePasswordStrength,
  isValidEmail,
  sanitizeInput,
  TwoFactorState,
} from '../security';

describe('CSRF Manager', () => {
  beforeEach(() => {
    csrfManager.clearToken();
  });

  it('returns null when no token set', () => {
    expect(csrfManager.getToken()).toBeNull();
  });

  it('stores and retrieves token', () => {
    csrfManager.setToken('test-token-123');
    expect(csrfManager.getToken()).toBe('test-token-123');
  });

  it('clears token', () => {
    csrfManager.setToken('test-token-123');
    csrfManager.clearToken();
    expect(csrfManager.getToken()).toBeNull();
  });

  it('reports valid token status', () => {
    expect(csrfManager.hasValidToken()).toBe(false);
    csrfManager.setToken('test-token');
    expect(csrfManager.hasValidToken()).toBe(true);
  });
});

describe('Rate Limiter', () => {
  beforeEach(() => {
    rateLimiter.clearAll();
  });

  it('allows requests under limit', () => {
    const config = { maxRequests: 3, windowMs: 60000 };
    
    expect(rateLimiter.shouldLimit('test-key', config)).toBe(false);
    expect(rateLimiter.shouldLimit('test-key', config)).toBe(false);
    expect(rateLimiter.shouldLimit('test-key', config)).toBe(false);
  });

  it('blocks requests over limit', () => {
    const config = { maxRequests: 2, windowMs: 60000 };
    
    expect(rateLimiter.shouldLimit('test-key', config)).toBe(false);
    expect(rateLimiter.shouldLimit('test-key', config)).toBe(false);
    expect(rateLimiter.shouldLimit('test-key', config)).toBe(true);
  });

  it('returns correct remaining count', () => {
    const config = { maxRequests: 5, windowMs: 60000 };
    
    expect(rateLimiter.getRemaining('new-key', config)).toBe(5);
    rateLimiter.shouldLimit('new-key', config);
    expect(rateLimiter.getRemaining('new-key', config)).toBe(4);
  });

  it('clears specific key', () => {
    const config = { maxRequests: 2, windowMs: 60000 };
    
    rateLimiter.shouldLimit('key1', config);
    rateLimiter.shouldLimit('key2', config);
    
    rateLimiter.clear('key1');
    
    expect(rateLimiter.getRemaining('key1', config)).toBe(2);
    expect(rateLimiter.getRemaining('key2', config)).toBe(1);
  });

  it('clears all keys', () => {
    const config = { maxRequests: 5, windowMs: 60000 };
    
    rateLimiter.shouldLimit('key1', config);
    rateLimiter.shouldLimit('key2', config);
    
    rateLimiter.clearAll();
    
    expect(rateLimiter.getRemaining('key1', config)).toBe(5);
    expect(rateLimiter.getRemaining('key2', config)).toBe(5);
  });

  it('has pre-configured rate limits', () => {
    expect(rateLimiter.configs.login).toBeDefined();
    expect(rateLimiter.configs.register).toBeDefined();
    expect(rateLimiter.configs.passwordReset).toBeDefined();
    expect(rateLimiter.configs.twoFactor).toBeDefined();
  });
});

describe('Two-Factor Auth', () => {
  beforeEach(() => {
    twoFactorAuth.setState(TwoFactorState.NOT_ENABLED);
  });

  it('validates 6-digit code format', () => {
    expect(twoFactorAuth.validateCodeFormat('123456')).toBe(true);
    expect(twoFactorAuth.validateCodeFormat('12345')).toBe(false);
    expect(twoFactorAuth.validateCodeFormat('1234567')).toBe(false);
    expect(twoFactorAuth.validateCodeFormat('abcdef')).toBe(false);
    expect(twoFactorAuth.validateCodeFormat('12345a')).toBe(false);
  });

  it('validates backup code format', () => {
    expect(twoFactorAuth.validateBackupCodeFormat('ABCD1234')).toBe(true);
    expect(twoFactorAuth.validateBackupCodeFormat('ABCDE12345')).toBe(true);
    expect(twoFactorAuth.validateBackupCodeFormat('ABCD-1234')).toBe(true);
    expect(twoFactorAuth.validateBackupCodeFormat('ABC')).toBe(false);
  });

  it('formats backup codes', () => {
    expect(twoFactorAuth.formatBackupCode('ABCD1234')).toBe('ABCD-1234');
    expect(twoFactorAuth.formatBackupCode('abcde12345')).toBe('ABCDE-12345');
  });

  it('tracks 2FA state', () => {
    expect(twoFactorAuth.isEnabled()).toBe(false);
    expect(twoFactorAuth.isVerificationRequired()).toBe(false);

    twoFactorAuth.setState(TwoFactorState.VERIFICATION_REQUIRED);
    expect(twoFactorAuth.isVerificationRequired()).toBe(true);

    twoFactorAuth.setState(TwoFactorState.VERIFIED);
    expect(twoFactorAuth.isEnabled()).toBe(true);
  });
});

describe('Session Security', () => {
  beforeEach(() => {
    sessionSecurity.stopInactivityTimer();
  });

  afterEach(() => {
    sessionSecurity.stopInactivityTimer();
  });

  it('tracks last activity', () => {
    const before = Date.now();
    sessionSecurity.updateActivity();
    const after = Date.now();

    expect(sessionSecurity.lastActivity).toBeGreaterThanOrEqual(before);
    expect(sessionSecurity.lastActivity).toBeLessThanOrEqual(after);
  });

  it('reports active session', () => {
    sessionSecurity.updateActivity();
    expect(sessionSecurity.isActive()).toBe(true);
  });

  it('starts and stops inactivity timer', () => {
    const callback = vi.fn();
    sessionSecurity.startInactivityTimer(callback);
    
    expect(sessionSecurity.inactivityTimer).not.toBeNull();
    
    sessionSecurity.stopInactivityTimer();
    expect(sessionSecurity.inactivityTimer).toBeNull();
  });
});

describe('Password Validation', () => {
  it('rejects short passwords', () => {
    const result = validatePasswordStrength('short');
    expect(result.isStrong).toBe(false);
    expect(result.feedback).toContain('Password should be at least 8 characters');
  });

  it('requires mixed case', () => {
    const result = validatePasswordStrength('alllowercase123!');
    expect(result.feedback).toContain('Include both uppercase and lowercase letters');
  });

  it('requires numbers', () => {
    const result = validatePasswordStrength('NoNumbersHere!');
    expect(result.feedback).toContain('Include at least one number');
  });

  it('requires special characters', () => {
    const result = validatePasswordStrength('NoSpecial123');
    expect(result.feedback).toContain('Include at least one special character');
  });

  it('accepts strong passwords', () => {
    const result = validatePasswordStrength('SecureP@ss123!');
    expect(result.isStrong).toBe(true);
    expect(result.score).toBeGreaterThanOrEqual(3);
  });

  it('penalizes common patterns', () => {
    const result = validatePasswordStrength('password123!A');
    expect(result.feedback).toContain('Avoid common password patterns');
  });
});

describe('Email Validation', () => {
  it('accepts valid emails', () => {
    expect(isValidEmail('test@example.com')).toBe(true);
    expect(isValidEmail('user.name@domain.co.uk')).toBe(true);
    expect(isValidEmail('user+tag@example.org')).toBe(true);
  });

  it('rejects invalid emails', () => {
    expect(isValidEmail('')).toBe(false);
    expect(isValidEmail('invalid')).toBe(false);
    expect(isValidEmail('no@domain')).toBe(false);
    expect(isValidEmail('@nodomain.com')).toBe(false);
    expect(isValidEmail('spaces in@email.com')).toBe(false);
  });
});

describe('Input Sanitization', () => {
  it('escapes HTML entities', () => {
    expect(sanitizeInput('<script>alert("xss")</script>')).toBe(
      '&lt;script&gt;alert("xss")&lt;/script&gt;'
    );
  });

  it('escapes special characters', () => {
    expect(sanitizeInput('Hello & "World"')).toBe('Hello &amp; "World"');
  });

  it('preserves normal text', () => {
    expect(sanitizeInput('Hello World 123')).toBe('Hello World 123');
  });
});
