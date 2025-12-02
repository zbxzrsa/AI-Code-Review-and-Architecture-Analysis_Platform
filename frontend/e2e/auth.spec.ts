/**
 * E2E Tests - Authentication Flow
 */

import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test.describe('Login', () => {
    test('should display login page', async ({ page }) => {
      await page.goto('/login');
      
      await expect(page.getByRole('heading', { name: /login|sign in/i })).toBeVisible();
      await expect(page.getByLabel(/email/i)).toBeVisible();
      await expect(page.getByLabel(/password/i)).toBeVisible();
      await expect(page.getByRole('button', { name: /login|sign in/i })).toBeVisible();
    });

    test('should login with valid credentials', async ({ page }) => {
      await page.goto('/login');
      
      await page.getByLabel(/email/i).fill('test@example.com');
      await page.getByLabel(/password/i).fill('password123');
      await page.getByRole('button', { name: /login|sign in/i }).click();
      
      // Should redirect to dashboard
      await expect(page).toHaveURL(/dashboard/);
      await expect(page.getByText(/welcome|dashboard/i)).toBeVisible();
    });

    test('should show error for invalid credentials', async ({ page }) => {
      await page.goto('/login');
      
      await page.getByLabel(/email/i).fill('wrong@example.com');
      await page.getByLabel(/password/i).fill('wrongpassword');
      await page.getByRole('button', { name: /login|sign in/i }).click();
      
      await expect(page.getByText(/invalid|incorrect|failed/i)).toBeVisible();
    });

    test('should show validation errors for empty fields', async ({ page }) => {
      await page.goto('/login');
      
      await page.getByRole('button', { name: /login|sign in/i }).click();
      
      await expect(page.getByText(/required|please enter/i)).toBeVisible();
    });

    test('should navigate to forgot password', async ({ page }) => {
      await page.goto('/login');
      
      await page.getByRole('link', { name: /forgot password/i }).click();
      
      await expect(page).toHaveURL(/forgot-password|reset/);
    });

    test('should navigate to register', async ({ page }) => {
      await page.goto('/login');
      
      await page.getByRole('link', { name: /sign up|register|create account/i }).click();
      
      await expect(page).toHaveURL(/register|signup/);
    });
  });

  test.describe('Registration', () => {
    test('should display registration page', async ({ page }) => {
      await page.goto('/register');
      
      await expect(page.getByRole('heading', { name: /register|sign up|create/i })).toBeVisible();
      await expect(page.getByLabel(/name/i)).toBeVisible();
      await expect(page.getByLabel(/email/i)).toBeVisible();
      await expect(page.getByLabel(/password/i)).toBeVisible();
    });

    test('should register new user', async ({ page }) => {
      await page.goto('/register');
      
      const uniqueEmail = `test-${Date.now()}@example.com`;
      
      await page.getByLabel(/name/i).fill('Test User');
      await page.getByLabel(/email/i).fill(uniqueEmail);
      await page.getByLabel(/^password$/i).fill('SecurePass123!');
      await page.getByLabel(/confirm password/i).fill('SecurePass123!');
      
      // Accept terms if present
      const termsCheckbox = page.getByLabel(/terms|agree/i);
      if (await termsCheckbox.isVisible()) {
        await termsCheckbox.check();
      }
      
      await page.getByRole('button', { name: /register|sign up|create/i }).click();
      
      // Should redirect after successful registration
      await expect(page).toHaveURL(/dashboard|verify|welcome/);
    });

    test('should show error for existing email', async ({ page }) => {
      await page.goto('/register');
      
      await page.getByLabel(/name/i).fill('Test User');
      await page.getByLabel(/email/i).fill('existing@example.com');
      await page.getByLabel(/^password$/i).fill('SecurePass123!');
      await page.getByLabel(/confirm password/i).fill('SecurePass123!');
      await page.getByRole('button', { name: /register|sign up|create/i }).click();
      
      await expect(page.getByText(/already exists|already registered/i)).toBeVisible();
    });

    test('should validate password requirements', async ({ page }) => {
      await page.goto('/register');
      
      await page.getByLabel(/^password$/i).fill('weak');
      await page.getByLabel(/^password$/i).blur();
      
      await expect(page.getByText(/at least|minimum|weak|strong/i)).toBeVisible();
    });

    test('should validate password confirmation', async ({ page }) => {
      await page.goto('/register');
      
      await page.getByLabel(/^password$/i).fill('SecurePass123!');
      await page.getByLabel(/confirm password/i).fill('DifferentPass123!');
      await page.getByLabel(/confirm password/i).blur();
      
      await expect(page.getByText(/match|same/i)).toBeVisible();
    });
  });

  test.describe('Logout', () => {
    test.beforeEach(async ({ page }) => {
      // Login first
      await page.goto('/login');
      await page.getByLabel(/email/i).fill('test@example.com');
      await page.getByLabel(/password/i).fill('password123');
      await page.getByRole('button', { name: /login|sign in/i }).click();
      await page.waitForURL(/dashboard/);
    });

    test('should logout successfully', async ({ page }) => {
      // Find and click logout
      const userMenu = page.getByRole('button', { name: /user|profile|account/i });
      if (await userMenu.isVisible()) {
        await userMenu.click();
      }
      
      await page.getByRole('button', { name: /logout|sign out/i }).click();
      
      // Should redirect to login
      await expect(page).toHaveURL(/login/);
    });

    test('should clear session on logout', async ({ page }) => {
      // Logout
      const userMenu = page.getByRole('button', { name: /user|profile|account/i });
      if (await userMenu.isVisible()) {
        await userMenu.click();
      }
      await page.getByRole('button', { name: /logout|sign out/i }).click();
      
      // Try to access protected route
      await page.goto('/dashboard');
      
      // Should redirect to login
      await expect(page).toHaveURL(/login/);
    });
  });

  test.describe('Password Reset', () => {
    test('should display forgot password page', async ({ page }) => {
      await page.goto('/forgot-password');
      
      await expect(page.getByRole('heading', { name: /forgot|reset|password/i })).toBeVisible();
      await expect(page.getByLabel(/email/i)).toBeVisible();
    });

    test('should send reset email', async ({ page }) => {
      await page.goto('/forgot-password');
      
      await page.getByLabel(/email/i).fill('test@example.com');
      await page.getByRole('button', { name: /send|reset|submit/i }).click();
      
      await expect(page.getByText(/sent|check your email/i)).toBeVisible();
    });

    test('should show error for non-existent email', async ({ page }) => {
      await page.goto('/forgot-password');
      
      await page.getByLabel(/email/i).fill('nonexistent@example.com');
      await page.getByRole('button', { name: /send|reset|submit/i }).click();
      
      await expect(page.getByText(/not found|doesn't exist/i)).toBeVisible();
    });
  });

  test.describe('Two-Factor Authentication', () => {
    test('should prompt for 2FA code when enabled', async ({ page }) => {
      await page.goto('/login');
      
      // Login with 2FA-enabled account
      await page.getByLabel(/email/i).fill('2fa-user@example.com');
      await page.getByLabel(/password/i).fill('password123');
      await page.getByRole('button', { name: /login|sign in/i }).click();
      
      // Should show 2FA input
      await expect(page.getByText(/verification code|2fa|authenticator/i)).toBeVisible();
      await expect(page.getByLabel(/code/i)).toBeVisible();
    });

    test('should accept valid 2FA code', async ({ page }) => {
      await page.goto('/login');
      
      await page.getByLabel(/email/i).fill('2fa-user@example.com');
      await page.getByLabel(/password/i).fill('password123');
      await page.getByRole('button', { name: /login|sign in/i }).click();
      
      // Enter valid code (mock)
      await page.getByLabel(/code/i).fill('123456');
      await page.getByRole('button', { name: /verify|submit/i }).click();
      
      await expect(page).toHaveURL(/dashboard/);
    });
  });

  test.describe('Protected Routes', () => {
    test('should redirect unauthenticated users to login', async ({ page }) => {
      await page.goto('/dashboard');
      
      await expect(page).toHaveURL(/login/);
    });

    test('should redirect to original URL after login', async ({ page }) => {
      // Try to access protected route
      await page.goto('/projects/123');
      
      // Should redirect to login
      await expect(page).toHaveURL(/login/);
      
      // Login
      await page.getByLabel(/email/i).fill('test@example.com');
      await page.getByLabel(/password/i).fill('password123');
      await page.getByRole('button', { name: /login|sign in/i }).click();
      
      // Should redirect back to original URL
      await expect(page).toHaveURL(/projects\/123/);
    });
  });
});
