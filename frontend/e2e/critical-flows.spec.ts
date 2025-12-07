/**
 * Critical User Flows E2E Tests
 *
 * Tests the most critical user journeys to ensure core functionality works.
 * These tests are tagged as @critical and must pass before any deployment.
 */

import { test, expect, Page } from "@playwright/test";

// Test configuration
const TEST_USER = {
  email: "e2e-test@example.com",
  password: "TestPassword123!",
  name: "E2E Test User",
};

const ADMIN_USER = {
  email: "admin@example.com",
  password: "AdminPassword123!",
};

// ============================================================================
// Helper Functions
// ============================================================================

async function login(page: Page, email: string, password: string) {
  await page.goto("/login");
  await page.fill('[data-testid="email-input"]', email);
  await page.fill('[data-testid="password-input"]', password);
  await page.click('[data-testid="login-button"]');
  await page.waitForURL("**/dashboard");
}

async function logout(page: Page) {
  await page.click('[data-testid="user-menu"]');
  await page.click('[data-testid="logout-button"]');
  await page.waitForURL("**/login");
}

// ============================================================================
// Authentication Flow Tests
// ============================================================================

test.describe("Critical: Authentication Flow", () => {
  test.describe.configure({ tag: "@critical" });

  test("should login successfully with valid credentials", async ({ page }) => {
    await page.goto("/login");

    // Fill login form
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);

    // Submit
    await page.click('[data-testid="login-button"]');

    // Verify redirect to dashboard
    await expect(page).toHaveURL(/.*dashboard/);

    // Verify user name is displayed
    await expect(page.locator('[data-testid="user-name"]')).toContainText(
      TEST_USER.name
    );
  });

  test("should show error for invalid credentials", async ({ page }) => {
    await page.goto("/login");

    await page.fill('[data-testid="email-input"]', "invalid@example.com");
    await page.fill('[data-testid="password-input"]', "wrongpassword");
    await page.click('[data-testid="login-button"]');

    // Verify error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]')).toContainText(
      /invalid|incorrect/i
    );
  });

  test("should logout successfully", async ({ page }) => {
    // Login first
    await login(page, TEST_USER.email, TEST_USER.password);

    // Logout
    await logout(page);

    // Verify redirect to login
    await expect(page).toHaveURL(/.*login/);

    // Try to access protected route
    await page.goto("/dashboard");
    await expect(page).toHaveURL(/.*login/);
  });

  test("should refresh token automatically", async ({ page }) => {
    await login(page, TEST_USER.email, TEST_USER.password);

    // Wait for token to be near expiry (simulated)
    await page.evaluate(() => {
      // Clear access token to simulate expiry
      localStorage.setItem("access_token_expiry", Date.now().toString());
    });

    // Make API request - should trigger refresh
    await page.click('[data-testid="projects-link"]');

    // Should not be redirected to login
    await expect(page).not.toHaveURL(/.*login/);
  });
});

// ============================================================================
// Code Review Flow Tests
// ============================================================================

test.describe("Critical: Code Review Flow", () => {
  test.describe.configure({ tag: "@critical" });

  test.beforeEach(async ({ page }) => {
    await login(page, TEST_USER.email, TEST_USER.password);
  });

  test("should create a new project", async ({ page }) => {
    await page.goto("/projects");

    // Click create project button
    await page.click('[data-testid="create-project-button"]');

    // Fill project details
    await page.fill('[data-testid="project-name-input"]', "E2E Test Project");
    await page.fill(
      '[data-testid="project-description-input"]',
      "Created by E2E test"
    );
    await page.selectOption(
      '[data-testid="project-language-select"]',
      "python"
    );

    // Submit
    await page.click('[data-testid="submit-project-button"]');

    // Verify project is created
    await expect(page.locator('[data-testid="project-card"]')).toContainText(
      "E2E Test Project"
    );
  });

  test("should submit code for analysis", async ({ page }) => {
    await page.goto("/projects");

    // Click on first project
    await page.click('[data-testid="project-card"]:first-child');

    // Navigate to code review
    await page.click('[data-testid="new-review-button"]');

    // Enter code in editor
    const sampleCode = `
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
    `;

    await page.fill('[data-testid="code-editor"]', sampleCode);

    // Submit for analysis
    await page.click('[data-testid="analyze-button"]');

    // Wait for analysis to complete
    await expect(page.locator('[data-testid="analysis-status"]')).toHaveText(
      /completed/i,
      {
        timeout: 60000,
      }
    );

    // Verify issues are displayed
    await expect(page.locator('[data-testid="issues-panel"]')).toBeVisible();
  });

  test("should display analysis results", async ({ page }) => {
    await page.goto("/projects");

    // Click on first project
    await page.click('[data-testid="project-card"]:first-child');

    // Click on existing analysis
    await page.click('[data-testid="analysis-item"]:first-child');

    // Verify analysis details are displayed
    await expect(page.locator('[data-testid="issues-count"]')).toBeVisible();
    await expect(
      page.locator('[data-testid="suggestions-panel"]')
    ).toBeVisible();
    await expect(page.locator('[data-testid="metrics-panel"]')).toBeVisible();
  });

  test("should apply suggested fix", async ({ page }) => {
    await page.goto("/code-review");

    // Wait for issues to load
    await expect(
      page.locator('[data-testid="issue-item"]:first-child')
    ).toBeVisible();

    // Click on issue
    await page.click('[data-testid="issue-item"]:first-child');

    // Click apply fix button
    await page.click('[data-testid="apply-fix-button"]');

    // Confirm fix
    await page.click('[data-testid="confirm-fix-button"]');

    // Verify code is updated
    const editorContent = await page
      .locator('[data-testid="code-editor"]')
      .textContent();
    expect(editorContent).toBeDefined();
  });
});

// ============================================================================
// AI Chat Flow Tests
// ============================================================================

test.describe("Critical: AI Chat Flow", () => {
  test.describe.configure({ tag: "@critical" });

  test.beforeEach(async ({ page }) => {
    await login(page, TEST_USER.email, TEST_USER.password);
  });

  test("should send message and receive response", async ({ page }) => {
    await page.goto("/code-review");

    // Open chat panel
    await page.click('[data-testid="open-chat-button"]');

    // Type message
    await page.fill('[data-testid="chat-input"]', "Explain this code");

    // Send message
    await page.click('[data-testid="send-message-button"]');

    // Wait for AI response
    await expect(
      page.locator('[data-testid="ai-message"]:last-child')
    ).toBeVisible({
      timeout: 30000,
    });

    // Verify response is not empty
    const response = await page
      .locator('[data-testid="ai-message"]:last-child')
      .textContent();
    expect(response?.length).toBeGreaterThan(0);
  });

  test("should stream response in real-time", async ({ page }) => {
    await page.goto("/code-review");

    await page.click('[data-testid="open-chat-button"]');
    await page.fill(
      '[data-testid="chat-input"]',
      "Generate a detailed analysis"
    );
    await page.click('[data-testid="send-message-button"]');

    // Check for streaming indicator
    await expect(
      page.locator('[data-testid="streaming-indicator"]')
    ).toBeVisible();

    // Wait for completion
    await expect(
      page.locator('[data-testid="streaming-indicator"]')
    ).not.toBeVisible({
      timeout: 60000,
    });
  });
});

// ============================================================================
// Admin Flow Tests
// ============================================================================

test.describe("Critical: Admin Flow", () => {
  test.describe.configure({ tag: "@critical" });

  test.beforeEach(async ({ page }) => {
    await login(page, ADMIN_USER.email, ADMIN_USER.password);
  });

  test("should access admin dashboard", async ({ page }) => {
    await page.goto("/admin");

    // Verify admin components are visible
    await expect(page.locator('[data-testid="admin-sidebar"]')).toBeVisible();
    await expect(page.locator('[data-testid="system-metrics"]')).toBeVisible();
  });

  test("should view three-version status", async ({ page }) => {
    await page.goto("/admin/three-version");

    // Verify version status panels
    await expect(page.locator('[data-testid="v1-status"]')).toBeVisible();
    await expect(page.locator('[data-testid="v2-status"]')).toBeVisible();
    await expect(page.locator('[data-testid="v3-status"]')).toBeVisible();
  });

  test("should manage users", async ({ page }) => {
    await page.goto("/admin/users");

    // Verify user list is visible
    await expect(page.locator('[data-testid="users-table"]')).toBeVisible();

    // Verify can view user details
    await page.click('[data-testid="user-row"]:first-child');
    await expect(
      page.locator('[data-testid="user-details-modal"]')
    ).toBeVisible();
  });
});

// ============================================================================
// Error Handling Tests
// ============================================================================

test.describe("Critical: Error Handling", () => {
  test.describe.configure({ tag: "@critical" });

  test("should show error page for 404", async ({ page }) => {
    await page.goto("/nonexistent-page");

    await expect(page.locator('[data-testid="error-page"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-code"]')).toContainText(
      "404"
    );
  });

  test("should handle network errors gracefully", async ({ page }) => {
    await login(page, TEST_USER.email, TEST_USER.password);

    // Simulate offline
    await page.context().setOffline(true);

    // Try to navigate
    await page.click('[data-testid="projects-link"]');

    // Should show offline indicator
    await expect(
      page.locator('[data-testid="offline-indicator"]')
    ).toBeVisible();

    // Restore online
    await page.context().setOffline(false);
  });

  test("should handle API errors with retry", async ({ page }) => {
    await login(page, TEST_USER.email, TEST_USER.password);

    // Intercept API and fail first request
    let requestCount = 0;
    await page.route("**/api/projects", (route) => {
      requestCount++;
      if (requestCount === 1) {
        route.abort("failed");
      } else {
        route.continue();
      }
    });

    await page.goto("/projects");

    // Should eventually show projects (after retry)
    await expect(page.locator('[data-testid="project-list"]')).toBeVisible({
      timeout: 10000,
    });
  });
});

// ============================================================================
// Accessibility Tests
// ============================================================================

test.describe("Critical: Accessibility", () => {
  test.describe.configure({ tag: "@critical" });

  test("should be keyboard navigable", async ({ page }) => {
    await page.goto("/login");

    // Tab through form
    await page.keyboard.press("Tab");
    await expect(page.locator('[data-testid="email-input"]')).toBeFocused();

    await page.keyboard.press("Tab");
    await expect(page.locator('[data-testid="password-input"]')).toBeFocused();

    await page.keyboard.press("Tab");
    await expect(page.locator('[data-testid="login-button"]')).toBeFocused();

    // Can submit with Enter
    await page.keyboard.press("Enter");
  });

  test("should have proper ARIA labels", async ({ page }) => {
    await page.goto("/login");

    // Check ARIA labels
    const emailInput = page.locator('[data-testid="email-input"]');
    await expect(emailInput).toHaveAttribute("aria-label", /email/i);

    const passwordInput = page.locator('[data-testid="password-input"]');
    await expect(passwordInput).toHaveAttribute("aria-label", /password/i);
  });
});

// ============================================================================
// Performance Tests
// ============================================================================

test.describe("Critical: Performance", () => {
  test.describe.configure({ tag: "@critical" });

  test("should load dashboard within 3 seconds", async ({ page }) => {
    await login(page, TEST_USER.email, TEST_USER.password);

    const startTime = Date.now();
    await page.goto("/dashboard");
    await page.waitForLoadState("networkidle");
    const loadTime = Date.now() - startTime;

    expect(loadTime).toBeLessThan(3000);
  });

  test("should have no memory leaks on navigation", async ({ page }) => {
    await login(page, TEST_USER.email, TEST_USER.password);

    // Navigate multiple times
    for (let i = 0; i < 10; i++) {
      await page.goto("/dashboard");
      await page.goto("/projects");
      await page.goto("/code-review");
    }

    // Check JS heap size (if available)
    const metrics = await page.evaluate(() => {
      if (performance.memory) {
        return {
          usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
          jsHeapSizeLimit: (performance as any).memory.jsHeapSizeLimit,
        };
      }
      return null;
    });

    if (metrics) {
      // Should use less than 50% of heap limit
      expect(metrics.usedJSHeapSize).toBeLessThan(
        metrics.jsHeapSizeLimit * 0.5
      );
    }
  });
});
