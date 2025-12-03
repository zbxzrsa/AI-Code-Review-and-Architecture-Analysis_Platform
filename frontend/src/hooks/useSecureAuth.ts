/**
 * Secure Authentication Hook
 *
 * Provides secure authentication functionality:
 * - Login with httpOnly cookies (no localStorage tokens)
 * - CSRF token management
 * - Two-factor authentication support
 * - Session management
 * - Rate limiting awareness
 */

import { useState, useCallback, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuthStore } from "../store/authStore";
import { apiService } from "../services/api";
import {
  csrfManager,
  sessionSecurity,
  twoFactorAuth,
  TwoFactorState,
  validatePasswordStrength,
  isValidEmail,
  rateLimiter,
} from "../services/security";
import { useNotification } from "../components/common/NotificationCenter";

/**
 * Login credentials
 */
interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
  invitationCode?: string;
}

/**
 * Login result
 */
interface LoginResult {
  success: boolean;
  requiresTwoFactor?: boolean;
  error?: string;
  user?: any;
}

/**
 * Two-factor verification data
 */
interface TwoFactorVerification {
  code: string;
  isBackupCode?: boolean;
}

/**
 * Registration data
 */
interface RegisterData {
  email: string;
  password: string;
  name: string;
  invitationCode: string;
}

/**
 * Secure Auth Hook
 */
export function useSecureAuth() {
  const navigate = useNavigate();
  const location = useLocation();
  const notify = useNotification();

  const {
    user,
    isAuthenticated,
    isLoading,
    error,
    twoFactor,
    setUser,
    setAuthenticated,
    setTwoFactorState,
    setLoading,
    setError,
    logout: storeLogout,
  } = useAuthStore();

  const [pendingTwoFactor, setPendingTwoFactor] = useState(false);

  /**
   * Initialize security on mount
   */
  useEffect(() => {
    // Setup session inactivity timer
    if (isAuthenticated) {
      sessionSecurity.startInactivityTimer(() => {
        notify.warning("Session expired due to inactivity");
        logout();
      });
    }

    return () => {
      sessionSecurity.stopInactivityTimer();
    };
    // Note: logout and notify are stable refs from stores
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated]);

  /**
   * Verify current session on mount
   */
  useEffect(() => {
    const verifySession = async () => {
      if (!isAuthenticated) return;

      try {
        setLoading(true);
        const response = await apiService.auth.me();
        setUser(response.data);

        // Fetch CSRF token
        await csrfManager.fetchToken();
      } catch (error) {
        // Session invalid - clear auth state
        storeLogout();
      } finally {
        setLoading(false);
      }
    };

    verifySession();
    // Intentionally run only on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /**
   * Login with email and password
   */
  const login = useCallback(
    async (credentials: LoginCredentials): Promise<LoginResult> => {
      const {
        email,
        password,
        rememberMe: _rememberMe,
        invitationCode,
      } = credentials;

      // Validate inputs
      if (!isValidEmail(email)) {
        return { success: false, error: "Invalid email address" };
      }

      if (!password) {
        return { success: false, error: "Password is required" };
      }

      // Check rate limiting
      if (rateLimiter.shouldLimit("/auth/login", rateLimiter.configs.login)) {
        const resetTime = rateLimiter.getResetTime("/auth/login");
        const waitSeconds = resetTime
          ? Math.ceil((resetTime - Date.now()) / 1000)
          : 60;
        return {
          success: false,
          error: `Too many login attempts. Please try again in ${waitSeconds} seconds.`,
        };
      }

      try {
        setLoading(true);
        setError(null);

        // Call login endpoint - server sets httpOnly cookie
        const response = await apiService.auth.login(
          email,
          password,
          invitationCode
        );
        const data = response.data;

        // Check if 2FA is required
        if (data.requires_two_factor) {
          setPendingTwoFactor(true);
          setTwoFactorState({ required: true, verified: false });
          twoFactorAuth.setState(TwoFactorState.VERIFICATION_REQUIRED);

          return { success: true, requiresTwoFactor: true };
        }

        // Login successful
        setUser(data.user);
        setAuthenticated(true);

        // Fetch CSRF token after login
        await csrfManager.fetchToken();

        // Start session timer
        sessionSecurity.startInactivityTimer(() => {
          notify.warning("Session expired due to inactivity");
          logout();
        });

        notify.success("Login successful");

        // Redirect to intended page or dashboard
        const returnUrl = new URLSearchParams(location.search).get("returnUrl");
        navigate(returnUrl || "/dashboard");

        return { success: true, user: data.user };
      } catch (error: any) {
        const message = error.response?.data?.message || "Login failed";
        setError(message);
        notify.error(message);
        return { success: false, error: message };
      } finally {
        setLoading(false);
      }
    },
    // Note: logout is excluded to avoid circular dependency - it's stable
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      location,
      navigate,
      notify,
      setUser,
      setAuthenticated,
      setTwoFactorState,
      setLoading,
      setError,
    ]
  );

  /**
   * Verify two-factor authentication code
   */
  const verifyTwoFactor = useCallback(
    async (verification: TwoFactorVerification): Promise<LoginResult> => {
      const { code, isBackupCode } = verification;

      // Validate code format
      if (!isBackupCode && !twoFactorAuth.validateCodeFormat(code)) {
        return {
          success: false,
          error: "Invalid code format. Enter 6 digits.",
        };
      }

      if (isBackupCode && !twoFactorAuth.validateBackupCodeFormat(code)) {
        return { success: false, error: "Invalid backup code format." };
      }

      // Check rate limiting
      if (
        rateLimiter.shouldLimit(
          "/auth/2fa/verify",
          rateLimiter.configs.twoFactor
        )
      ) {
        return {
          success: false,
          error: "Too many attempts. Please try again later.",
        };
      }

      try {
        setLoading(true);

        const response = await apiService.auth.verify2FA(code, isBackupCode);
        const data = response.data;

        // 2FA verified
        setPendingTwoFactor(false);
        setTwoFactorState({ required: false, verified: true });
        twoFactorAuth.setState(TwoFactorState.VERIFIED);

        setUser(data.user);
        setAuthenticated(true);

        // Fetch CSRF token
        await csrfManager.fetchToken();

        notify.success("Two-factor authentication verified");

        // Redirect
        const returnUrl = new URLSearchParams(location.search).get("returnUrl");
        navigate(returnUrl || "/dashboard");

        return { success: true, user: data.user };
      } catch (error: any) {
        const message = error.response?.data?.message || "Verification failed";
        notify.error(message);
        return { success: false, error: message };
      } finally {
        setLoading(false);
      }
    },
    [
      location,
      navigate,
      notify,
      setUser,
      setAuthenticated,
      setTwoFactorState,
      setLoading,
    ]
  );

  /**
   * Register new account
   */
  const register = useCallback(
    async (data: RegisterData): Promise<LoginResult> => {
      const { email, password, name, invitationCode } = data;

      // Validate inputs
      if (!isValidEmail(email)) {
        return { success: false, error: "Invalid email address" };
      }

      const passwordStrength = validatePasswordStrength(password);
      if (!passwordStrength.isStrong) {
        return { success: false, error: passwordStrength.feedback.join(". ") };
      }

      if (!name.trim()) {
        return { success: false, error: "Name is required" };
      }

      // Check rate limiting
      if (
        rateLimiter.shouldLimit("/auth/register", rateLimiter.configs.register)
      ) {
        return {
          success: false,
          error: "Too many registration attempts. Please try again later.",
        };
      }

      try {
        setLoading(true);
        setError(null);

        await apiService.auth.register({
          email,
          password,
          name,
          invitation_code: invitationCode,
        });

        notify.success(
          "Registration successful! Please check your email to verify your account."
        );

        return { success: true };
      } catch (error: any) {
        const message = error.response?.data?.message || "Registration failed";
        setError(message);
        notify.error(message);
        return { success: false, error: message };
      } finally {
        setLoading(false);
      }
    },
    [notify, setLoading, setError]
  );

  /**
   * Logout
   */
  const logout = useCallback(async () => {
    try {
      // Call logout endpoint to clear server-side session and cookies
      await apiService.auth.logout();
    } catch (error) {
      // Continue with client-side logout even if server call fails
      console.error("Logout API error:", error);
    } finally {
      // Clear client-side state
      storeLogout();
      csrfManager.clearToken();
      sessionSecurity.stopInactivityTimer();

      navigate("/login");
    }
  }, [navigate, storeLogout]);

  /**
   * Request password reset
   */
  const requestPasswordReset = useCallback(
    async (email: string): Promise<{ success: boolean; error?: string }> => {
      if (!isValidEmail(email)) {
        return { success: false, error: "Invalid email address" };
      }

      // Check rate limiting
      if (
        rateLimiter.shouldLimit(
          "/auth/password/reset",
          rateLimiter.configs.passwordReset
        )
      ) {
        return {
          success: false,
          error: "Too many requests. Please try again later.",
        };
      }

      try {
        setLoading(true);
        await apiService.user.requestPasswordReset(email);
        notify.success("Password reset email sent. Check your inbox.");
        return { success: true };
      } catch (error: any) {
        const message =
          error.response?.data?.message || "Failed to send reset email";
        notify.error(message);
        return { success: false, error: message };
      } finally {
        setLoading(false);
      }
    },
    [notify, setLoading]
  );

  /**
   * Reset password with token
   */
  const resetPassword = useCallback(
    async (
      token: string,
      newPassword: string
    ): Promise<{ success: boolean; error?: string }> => {
      const passwordStrength = validatePasswordStrength(newPassword);
      if (!passwordStrength.isStrong) {
        return { success: false, error: passwordStrength.feedback.join(". ") };
      }

      try {
        setLoading(true);
        await apiService.user.resetPassword(token, newPassword);
        notify.success("Password reset successful. You can now log in.");
        navigate("/login");
        return { success: true };
      } catch (error: any) {
        const message =
          error.response?.data?.message || "Password reset failed";
        notify.error(message);
        return { success: false, error: message };
      } finally {
        setLoading(false);
      }
    },
    [navigate, notify, setLoading]
  );

  return {
    // State
    user,
    isAuthenticated,
    isLoading,
    error,
    twoFactor,
    pendingTwoFactor,

    // Actions
    login,
    verifyTwoFactor,
    register,
    logout,
    requestPasswordReset,
    resetPassword,

    // Utilities
    isValidEmail,
    validatePasswordStrength,
  };
}

export default useSecureAuth;
