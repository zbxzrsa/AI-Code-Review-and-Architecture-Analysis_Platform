/**
 * Security Service
 * 
 * Comprehensive security utilities for:
 * - CSRF token management (stored in memory, not localStorage)
 * - Secure authentication with httpOnly cookies
 * - Rate limiting
 * - Security headers validation
 * - Two-factor authentication support
 */

import { api } from './api';

// ============================================
// CSRF Token Management
// ============================================

/**
 * CSRF token stored in memory only (not localStorage/sessionStorage)
 * This prevents XSS attacks from accessing the token
 */
let csrfToken: string | null = null;
let csrfTokenExpiry: number | null = null;
const CSRF_TOKEN_LIFETIME = 60 * 60 * 1000; // 1 hour

/**
 * CSRF Token Manager
 * 
 * Implements Double-Submit Cookie pattern:
 * 1. Server sends CSRF token in response header or body
 * 2. Client stores token in memory (not localStorage)
 * 3. Client sends token in X-CSRF-Token header on state-changing requests
 * 4. Server validates token matches the one stored server-side
 */
export const csrfManager = {
  /**
   * Get current CSRF token from memory
   */
  getToken(): string | null {
    // Check if token is expired
    if (csrfTokenExpiry && Date.now() > csrfTokenExpiry) {
      csrfToken = null;
      csrfTokenExpiry = null;
    }
    return csrfToken;
  },

  /**
   * Set CSRF token (called after login or token refresh)
   */
  setToken(token: string): void {
    csrfToken = token;
    csrfTokenExpiry = Date.now() + CSRF_TOKEN_LIFETIME;
  },

  /**
   * Clear CSRF token (called on logout)
   */
  clearToken(): void {
    csrfToken = null;
    csrfTokenExpiry = null;
  },

  /**
   * Fetch fresh CSRF token from server
   */
  async fetchToken(): Promise<string> {
    try {
      const response = await api.get('/csrf-token');
      const token = response.data.token || response.headers['x-csrf-token'];
      
      if (token) {
        this.setToken(token);
        return token;
      }
      throw new Error('No CSRF token received');
    } catch (error) {
      console.error('Failed to fetch CSRF token:', error);
      throw error;
    }
  },

  /**
   * Get token, fetching if necessary
   */
  async ensureToken(): Promise<string> {
    const token = this.getToken();
    if (token) return token;
    return this.fetchToken();
  },

  /**
   * Check if we have a valid token
   */
  hasValidToken(): boolean {
    return !!this.getToken();
  },
};

// ============================================
// Rate Limiting (Client-side)
// ============================================

interface RateLimitEntry {
  count: number;
  resetTime: number;
}

const rateLimitCache = new Map<string, RateLimitEntry>();

/**
 * Rate Limiter Configuration
 */
export interface RateLimitConfig {
  maxRequests: number;
  windowMs: number;
  keyGenerator?: (endpoint: string) => string;
}

const defaultRateLimitConfig: RateLimitConfig = {
  maxRequests: 100,
  windowMs: 60000, // 1 minute
};

/**
 * Client-side Rate Limiter
 * 
 * Prevents excessive requests before they hit the server.
 * Server-side rate limiting is still enforced.
 */
export const rateLimiter = {
  /**
   * Check if request should be rate limited
   */
  shouldLimit(key: string, config: RateLimitConfig = defaultRateLimitConfig): boolean {
    const now = Date.now();
    const entry = rateLimitCache.get(key);

    if (!entry || now > entry.resetTime) {
      rateLimitCache.set(key, {
        count: 1,
        resetTime: now + config.windowMs,
      });
      return false;
    }

    if (entry.count >= config.maxRequests) {
      return true;
    }

    entry.count++;
    return false;
  },

  /**
   * Get remaining requests for a key
   */
  getRemaining(key: string, config: RateLimitConfig = defaultRateLimitConfig): number {
    const entry = rateLimitCache.get(key);
    if (!entry || Date.now() > entry.resetTime) {
      return config.maxRequests;
    }
    return Math.max(0, config.maxRequests - entry.count);
  },

  /**
   * Get reset time for a key
   */
  getResetTime(key: string): number | null {
    const entry = rateLimitCache.get(key);
    if (!entry || Date.now() > entry.resetTime) {
      return null;
    }
    return entry.resetTime;
  },

  /**
   * Clear rate limit for a key
   */
  clear(key: string): void {
    rateLimitCache.delete(key);
  },

  /**
   * Clear all rate limits
   */
  clearAll(): void {
    rateLimitCache.clear();
  },

  /**
   * Rate limit configurations for different endpoints
   */
  configs: {
    login: { maxRequests: 5, windowMs: 60000 },       // 5 per minute
    register: { maxRequests: 3, windowMs: 60000 },    // 3 per minute
    passwordReset: { maxRequests: 3, windowMs: 300000 }, // 3 per 5 minutes
    twoFactor: { maxRequests: 5, windowMs: 60000 },   // 5 per minute
    api: { maxRequests: 100, windowMs: 60000 },       // 100 per minute
  },
};

// ============================================
// Security Headers Validation
// ============================================

/**
 * Security Headers Configuration
 */
export interface SecurityHeaders {
  'Content-Security-Policy'?: string;
  'X-Content-Type-Options'?: string;
  'X-Frame-Options'?: string;
  'X-XSS-Protection'?: string;
  'Referrer-Policy'?: string;
  'Strict-Transport-Security'?: string;
  'Permissions-Policy'?: string;
}

/**
 * Expected security headers for production
 */
export const expectedSecurityHeaders: SecurityHeaders = {
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'X-XSS-Protection': '1; mode=block',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
};

/**
 * Validate response has proper security headers
 */
export function validateSecurityHeaders(headers: Headers | Record<string, string>): {
  valid: boolean;
  missing: string[];
  warnings: string[];
} {
  const missing: string[] = [];
  const warnings: string[] = [];

  const getHeader = (name: string): string | null => {
    if (headers instanceof Headers) {
      return headers.get(name);
    }
    return headers[name] || headers[name.toLowerCase()] || null;
  };

  // Check expected headers
  for (const [header, expectedValue] of Object.entries(expectedSecurityHeaders)) {
    const value = getHeader(header);
    if (!value) {
      missing.push(header);
    } else if (value !== expectedValue) {
      warnings.push(`${header}: expected "${expectedValue}", got "${value}"`);
    }
  }

  // Check for HSTS in production
  if (typeof window !== 'undefined' && window.location.protocol === 'https:') {
    const hsts = getHeader('Strict-Transport-Security');
    if (!hsts) {
      missing.push('Strict-Transport-Security');
    }
  }

  return {
    valid: missing.length === 0,
    missing,
    warnings,
  };
}

// ============================================
// Two-Factor Authentication Helpers
// ============================================

/**
 * 2FA verification states
 */
export enum TwoFactorState {
  NOT_ENABLED = 'not_enabled',
  SETUP_REQUIRED = 'setup_required',
  VERIFICATION_REQUIRED = 'verification_required',
  VERIFIED = 'verified',
}

/**
 * 2FA setup data
 */
export interface TwoFactorSetupData {
  secret: string;
  qrCodeUrl: string;
  backupCodes: string[];
}

/**
 * Two-Factor Authentication Manager
 */
export const twoFactorAuth = {
  /**
   * Check current 2FA state
   */
  state: TwoFactorState.NOT_ENABLED,

  /**
   * Set 2FA state
   */
  setState(state: TwoFactorState): void {
    this.state = state;
  },

  /**
   * Check if 2FA verification is required
   */
  isVerificationRequired(): boolean {
    return this.state === TwoFactorState.VERIFICATION_REQUIRED;
  },

  /**
   * Check if 2FA is enabled
   */
  isEnabled(): boolean {
    return this.state !== TwoFactorState.NOT_ENABLED;
  },

  /**
   * Validate TOTP code format (6 digits)
   */
  validateCodeFormat(code: string): boolean {
    return /^\d{6}$/.test(code);
  },

  /**
   * Validate backup code format
   */
  validateBackupCodeFormat(code: string): boolean {
    // Backup codes are typically 8-10 alphanumeric characters
    return /^[A-Z0-9]{8,10}$/i.test(code.replace(/-/g, ''));
  },

  /**
   * Format backup code for display
   */
  formatBackupCode(code: string): string {
    // Format as XXXX-XXXX or XXXXX-XXXXX
    const clean = code.replace(/-/g, '').toUpperCase();
    const mid = Math.floor(clean.length / 2);
    return `${clean.slice(0, mid)}-${clean.slice(mid)}`;
  },
};

// ============================================
// Secure Session Management
// ============================================

/**
 * Session security utilities
 */
export const sessionSecurity = {
  /**
   * Session inactivity timeout (15 minutes)
   */
  INACTIVITY_TIMEOUT: 15 * 60 * 1000,

  /**
   * Last activity timestamp
   */
  lastActivity: Date.now(),

  /**
   * Inactivity timer reference
   */
  inactivityTimer: null as ReturnType<typeof setTimeout> | null,

  /**
   * Callback for session expiry
   */
  onSessionExpired: null as (() => void) | null,

  /**
   * Update last activity time
   */
  updateActivity(): void {
    this.lastActivity = Date.now();
    this.resetInactivityTimer();
  },

  /**
   * Start inactivity timer
   */
  startInactivityTimer(onExpired: () => void): void {
    this.onSessionExpired = onExpired;
    this.resetInactivityTimer();

    // Listen for user activity
    if (typeof window !== 'undefined') {
      const events = ['mousedown', 'keydown', 'scroll', 'touchstart'];
      events.forEach(event => {
        window.addEventListener(event, () => this.updateActivity(), { passive: true });
      });
    }
  },

  /**
   * Reset inactivity timer
   */
  resetInactivityTimer(): void {
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
    }
    this.inactivityTimer = setTimeout(() => {
      if (this.onSessionExpired) {
        this.onSessionExpired();
      }
    }, this.INACTIVITY_TIMEOUT);
  },

  /**
   * Stop inactivity timer
   */
  stopInactivityTimer(): void {
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }
  },

  /**
   * Check if session is still active
   */
  isActive(): boolean {
    return Date.now() - this.lastActivity < this.INACTIVITY_TIMEOUT;
  },
};

// ============================================
// Input Sanitization
// ============================================

/**
 * Sanitize user input to prevent XSS
 */
export function sanitizeInput(input: string): string {
  const div = document.createElement('div');
  div.textContent = input;
  return div.innerHTML;
}

/**
 * Validate email format
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Validate password strength
 */
export interface PasswordStrength {
  score: number; // 0-4
  feedback: string[];
  isStrong: boolean;
}

export function validatePasswordStrength(password: string): PasswordStrength {
  const feedback: string[] = [];
  let score = 0;

  if (password.length >= 8) score++;
  else feedback.push('Password should be at least 8 characters');

  if (password.length >= 12) score++;

  if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score++;
  else feedback.push('Include both uppercase and lowercase letters');

  if (/\d/.test(password)) score++;
  else feedback.push('Include at least one number');

  if (/[!@#$%^&*(),.?":{}|<>]/.test(password)) score++;
  else feedback.push('Include at least one special character');

  // Check for common patterns
  const commonPatterns = ['password', '123456', 'qwerty', 'abc123'];
  if (commonPatterns.some(p => password.toLowerCase().includes(p))) {
    score = Math.max(0, score - 2);
    feedback.push('Avoid common password patterns');
  }

  return {
    score: Math.min(4, score),
    feedback,
    isStrong: score >= 3,
  };
}

// ============================================
// Fingerprinting Detection (Anti-fraud)
// ============================================

/**
 * Generate device fingerprint for fraud detection
 * Note: This is for security purposes, respecting privacy
 */
export async function getDeviceFingerprint(): Promise<string> {
  const components: string[] = [];

  // Screen info
  if (typeof screen !== 'undefined') {
    components.push(`${screen.width}x${screen.height}x${screen.colorDepth}`);
  }

  // Timezone
  components.push(Intl.DateTimeFormat().resolvedOptions().timeZone);

  // Language
  components.push(navigator.language);

  // Platform
  components.push(navigator.platform);

  // Canvas fingerprint (generic)
  try {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.textBaseline = 'top';
      ctx.font = '14px Arial';
      ctx.fillText('fingerprint', 0, 0);
      components.push(canvas.toDataURL().slice(-50));
    }
  } catch {
    // Canvas fingerprinting blocked
  }

  // Create hash
  const data = components.join('|');
  const encoder = new TextEncoder();
  const buffer = await crypto.subtle.digest('SHA-256', encoder.encode(data));
  const hashArray = Array.from(new Uint8Array(buffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

// ============================================
// Secure Storage Wrapper
// ============================================

/**
 * Secure storage that encrypts sensitive data
 * Uses Web Crypto API when available
 */
export const secureStorage = {
  /**
   * Encryption key (derived from fingerprint)
   */
  encryptionKey: null as CryptoKey | null,

  /**
   * Initialize encryption key
   */
  async init(): Promise<void> {
    if (typeof crypto === 'undefined' || !crypto.subtle) {
      console.warn('Web Crypto API not available');
      return;
    }

    const fingerprint = await getDeviceFingerprint();
    const encoder = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
      'raw',
      encoder.encode(fingerprint),
      { name: 'PBKDF2' },
      false,
      ['deriveKey']
    );

    this.encryptionKey = await crypto.subtle.deriveKey(
      {
        name: 'PBKDF2',
        salt: encoder.encode('ai-code-review-salt'),
        iterations: 100000,
        hash: 'SHA-256',
      },
      keyMaterial,
      { name: 'AES-GCM', length: 256 },
      false,
      ['encrypt', 'decrypt']
    );
  },

  /**
   * Encrypt and store data
   */
  async setItem(key: string, value: string): Promise<void> {
    if (!this.encryptionKey) {
      sessionStorage.setItem(key, value);
      return;
    }

    const iv = crypto.getRandomValues(new Uint8Array(12));
    const encoder = new TextEncoder();
    const encrypted = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv },
      this.encryptionKey,
      encoder.encode(value)
    );

    const data = {
      iv: Array.from(iv),
      data: Array.from(new Uint8Array(encrypted)),
    };
    sessionStorage.setItem(key, JSON.stringify(data));
  },

  /**
   * Retrieve and decrypt data
   */
  async getItem(key: string): Promise<string | null> {
    const stored = sessionStorage.getItem(key);
    if (!stored) return null;

    if (!this.encryptionKey) {
      return stored;
    }

    try {
      const { iv, data } = JSON.parse(stored);
      const decrypted = await crypto.subtle.decrypt(
        { name: 'AES-GCM', iv: new Uint8Array(iv) },
        this.encryptionKey,
        new Uint8Array(data)
      );
      return new TextDecoder().decode(decrypted);
    } catch {
      return null;
    }
  },

  /**
   * Remove item
   */
  removeItem(key: string): void {
    sessionStorage.removeItem(key);
  },

  /**
   * Clear all items
   */
  clear(): void {
    sessionStorage.clear();
  },
};

// ============================================
// Content Security Policy Helpers
// ============================================

/**
 * Report CSP violations
 */
export function setupCSPReporting(): void {
  if (typeof document === 'undefined') return;

  document.addEventListener('securitypolicyviolation', (event) => {
    const report = {
      documentURI: event.documentURI,
      violatedDirective: event.violatedDirective,
      effectiveDirective: event.effectiveDirective,
      originalPolicy: event.originalPolicy,
      blockedURI: event.blockedURI,
      statusCode: event.statusCode,
      timestamp: new Date().toISOString(),
    };

    // Send to backend for logging
    api.post('/security/csp-report', report).catch(() => {
      // Silently fail - don't block on CSP reporting
    });

    if (process.env.NODE_ENV === 'development') {
      console.warn('CSP Violation:', report);
    }
  });
}

// ============================================
// Export all security utilities
// ============================================

export default {
  csrfManager,
  rateLimiter,
  twoFactorAuth,
  sessionSecurity,
  secureStorage,
  validateSecurityHeaders,
  sanitizeInput,
  isValidEmail,
  validatePasswordStrength,
  getDeviceFingerprint,
  setupCSPReporting,
};
