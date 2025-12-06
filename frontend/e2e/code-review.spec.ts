/**
 * E2E Tests for Code Review Flow
 *
 * Tests cover the complete code review user journey:
 * - File upload and analysis
 * - AI-powered code suggestions
 * - Issue detection and navigation
 * - Fix application workflow
 */
import { test, expect, Page } from "@playwright/test";

// Test configuration
const BASE_URL = process.env.BASE_URL || "http://localhost:3000";
const API_URL = process.env.API_URL || "http://localhost:8000";

// Test data
const TEST_CODE = `
function processData(data) {
  var result = [];
  for (var i = 0; i < data.length; i++) {
    eval(data[i]); // Security issue: eval
    result.push(data[i]);
  }
  return result;
}

// SQL injection vulnerability
function getUser(id) {
  const query = "SELECT * FROM users WHERE id = " + id;
  return db.query(query);
}

// Hardcoded credentials
const API_KEY = "sk-1234567890abcdef";
const PASSWORD = "admin123";
`;

// Helper functions
async function login(
  page: Page,
  email: string = "test@example.com",
  password: string = "password123"
) {
  await page.goto(`${BASE_URL}/login`);
  await page.fill('[data-testid="email-input"]', email);
  await page.fill('[data-testid="password-input"]', password);
  await page.click('[data-testid="login-button"]');
  await page.waitForURL("**/dashboard");
}

async function navigateToCodeReview(page: Page) {
  await page.click('[data-testid="nav-code-review"]');
  await page.waitForURL("**/code-review");
}

// Test suite
test.describe("Code Review Flow", () => {
  test.beforeEach(async ({ page }) => {
    // Mock authentication
    await page.route(`${API_URL}/api/auth/login`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          access_token: "test-token",
          user: { id: "user1", email: "test@example.com", role: "user" },
        }),
      });
    });

    // Mock analysis endpoint
    await page.route(`${API_URL}/api/analyze/code`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          analysis_id: "analysis-123",
          status: "completed",
          issues: [
            {
              id: "issue-1",
              type: "security",
              severity: "critical",
              message: "Use of eval() is dangerous",
              line: 5,
              column: 5,
              suggestion: "Use JSON.parse() for data parsing",
            },
            {
              id: "issue-2",
              type: "security",
              severity: "high",
              message: "SQL injection vulnerability",
              line: 13,
              column: 17,
              suggestion: "Use parameterized queries",
            },
            {
              id: "issue-3",
              type: "security",
              severity: "critical",
              message: "Hardcoded API key detected",
              line: 17,
              column: 7,
              suggestion: "Use environment variables",
            },
          ],
          metrics: {
            total_issues: 3,
            critical: 2,
            high: 1,
            medium: 0,
            low: 0,
          },
        }),
      });
    });
  });

  test("should display code editor on code review page", async ({ page }) => {
    await login(page);
    await navigateToCodeReview(page);

    // Verify code editor is visible
    const editor = page.locator('[data-testid="code-editor"]');
    await expect(editor).toBeVisible();
  });

  test("should analyze pasted code", async ({ page }) => {
    await login(page);
    await navigateToCodeReview(page);

    // Paste code into editor
    const editor = page.locator(
      '[data-testid="code-editor"] textarea, .monaco-editor textarea'
    );
    await editor.fill(TEST_CODE);

    // Click analyze button
    await page.click('[data-testid="analyze-button"]');

    // Wait for analysis to complete
    await page.waitForSelector('[data-testid="analysis-results"]');

    // Verify issues are displayed
    const issuesList = page.locator('[data-testid="issues-list"]');
    await expect(issuesList).toBeVisible();
  });

  test("should display issue details on click", async ({ page }) => {
    await login(page);
    await navigateToCodeReview(page);

    // Trigger analysis
    const editor = page.locator(
      '[data-testid="code-editor"] textarea, .monaco-editor textarea'
    );
    await editor.fill(TEST_CODE);
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-results"]');

    // Click on first issue
    await page.click('[data-testid="issue-item"]:first-child');

    // Verify issue details panel is shown
    const detailsPanel = page.locator('[data-testid="issue-details"]');
    await expect(detailsPanel).toBeVisible();
    await expect(detailsPanel).toContainText("eval");
  });

  test("should filter issues by severity", async ({ page }) => {
    await login(page);
    await navigateToCodeReview(page);

    // Trigger analysis
    const editor = page.locator(
      '[data-testid="code-editor"] textarea, .monaco-editor textarea'
    );
    await editor.fill(TEST_CODE);
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-results"]');

    // Filter by critical severity
    await page.click('[data-testid="filter-critical"]');

    // Verify only critical issues are shown
    const issues = page.locator('[data-testid="issue-item"]');
    const count = await issues.count();
    expect(count).toBe(2); // 2 critical issues
  });

  test("should navigate to issue line in editor", async ({ page }) => {
    await login(page);
    await navigateToCodeReview(page);

    // Trigger analysis
    const editor = page.locator(
      '[data-testid="code-editor"] textarea, .monaco-editor textarea'
    );
    await editor.fill(TEST_CODE);
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-results"]');

    // Click "Go to line" on issue
    await page.click(
      '[data-testid="issue-item"]:first-child [data-testid="goto-line"]'
    );

    // Verify editor scrolled to line (check cursor position or highlight)
    const highlightedLine = page.locator(".line-highlight, .current-line");
    await expect(highlightedLine).toBeVisible();
  });

  test("should apply suggested fix", async ({ page }) => {
    // Mock fix application endpoint
    await page.route(`${API_URL}/api/fixes/apply`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          success: true,
          fixed_code: TEST_CODE.replace("eval(data[i])", "JSON.parse(data[i])"),
        }),
      });
    });

    await login(page);
    await navigateToCodeReview(page);

    // Trigger analysis
    const editor = page.locator(
      '[data-testid="code-editor"] textarea, .monaco-editor textarea'
    );
    await editor.fill(TEST_CODE);
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-results"]');

    // Click apply fix on first issue
    await page.click(
      '[data-testid="issue-item"]:first-child [data-testid="apply-fix"]'
    );

    // Verify fix was applied
    await page.waitForSelector('[data-testid="fix-applied-toast"]');
  });

  test("should show diff view for suggested fix", async ({ page }) => {
    await login(page);
    await navigateToCodeReview(page);

    // Trigger analysis
    const editor = page.locator(
      '[data-testid="code-editor"] textarea, .monaco-editor textarea'
    );
    await editor.fill(TEST_CODE);
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-results"]');

    // Click preview fix
    await page.click(
      '[data-testid="issue-item"]:first-child [data-testid="preview-fix"]'
    );

    // Verify diff view is shown
    const diffView = page.locator('[data-testid="diff-viewer"]');
    await expect(diffView).toBeVisible();
  });

  test("should export analysis report", async ({ page }) => {
    await login(page);
    await navigateToCodeReview(page);

    // Trigger analysis
    const editor = page.locator(
      '[data-testid="code-editor"] textarea, .monaco-editor textarea'
    );
    await editor.fill(TEST_CODE);
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-results"]');

    // Mock download
    const downloadPromise = page.waitForEvent("download");
    await page.click('[data-testid="export-report"]');
    const download = await downloadPromise;

    // Verify download
    expect(download.suggestedFilename()).toContain("analysis-report");
  });

  test("should display analysis metrics", async ({ page }) => {
    await login(page);
    await navigateToCodeReview(page);

    // Trigger analysis
    const editor = page.locator(
      '[data-testid="code-editor"] textarea, .monaco-editor textarea'
    );
    await editor.fill(TEST_CODE);
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-results"]');

    // Verify metrics are displayed
    const metricsPanel = page.locator('[data-testid="analysis-metrics"]');
    await expect(metricsPanel).toBeVisible();
    await expect(metricsPanel).toContainText("3"); // Total issues
  });

  test("should handle analysis error gracefully", async ({ page }) => {
    // Mock error response
    await page.route(`${API_URL}/api/analyze/code`, async (route) => {
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ error: "Analysis service unavailable" }),
      });
    });

    await login(page);
    await navigateToCodeReview(page);

    // Trigger analysis
    const editor = page.locator(
      '[data-testid="code-editor"] textarea, .monaco-editor textarea'
    );
    await editor.fill(TEST_CODE);
    await page.click('[data-testid="analyze-button"]');

    // Verify error message is shown
    const errorMessage = page.locator('[data-testid="error-message"]');
    await expect(errorMessage).toBeVisible();
    await expect(errorMessage).toContainText("error");
  });

  test("should support file upload for analysis", async ({ page }) => {
    await login(page);
    await navigateToCodeReview(page);

    // Upload file
    const fileInput = page.locator(
      '[data-testid="file-upload"] input[type="file"]'
    );
    await fileInput.setInputFiles({
      name: "test.js",
      mimeType: "text/javascript",
      buffer: Buffer.from(TEST_CODE),
    });

    // Wait for file to be loaded in editor
    await page.waitForTimeout(500);

    // Verify code is loaded
    const editor = page.locator('[data-testid="code-editor"]');
    await expect(editor).toContainText("processData");
  });

  test("should maintain analysis history", async ({ page }) => {
    // Mock history endpoint
    await page.route(`${API_URL}/api/analyze/history`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          history: [
            {
              id: "analysis-1",
              timestamp: new Date().toISOString(),
              issues_count: 5,
            },
            {
              id: "analysis-2",
              timestamp: new Date().toISOString(),
              issues_count: 3,
            },
          ],
        }),
      });
    });

    await login(page);
    await navigateToCodeReview(page);

    // Open history panel
    await page.click('[data-testid="history-button"]');

    // Verify history is displayed
    const historyPanel = page.locator('[data-testid="analysis-history"]');
    await expect(historyPanel).toBeVisible();
  });

  test("should support AI chat for code questions", async ({ page }) => {
    // Mock chat endpoint
    await page.route(`${API_URL}/api/chat/message`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          response:
            "The eval() function is dangerous because it executes arbitrary code.",
        }),
      });
    });

    await login(page);
    await navigateToCodeReview(page);

    // Open AI chat
    await page.click('[data-testid="ai-chat-toggle"]');

    // Send message
    const chatInput = page.locator('[data-testid="chat-input"]');
    await chatInput.fill("What is dangerous about eval?");
    await page.click('[data-testid="send-message"]');

    // Verify response
    const chatResponse = page.locator(
      '[data-testid="chat-message"]:last-child'
    );
    await expect(chatResponse).toContainText("eval");
  });
});

// Performance tests
test.describe("Code Review Performance", () => {
  test("should load code review page within 2 seconds", async ({ page }) => {
    await login(page);

    const startTime = Date.now();
    await navigateToCodeReview(page);
    const loadTime = Date.now() - startTime;

    expect(loadTime).toBeLessThan(2000);
  });

  test("should analyze code within 5 seconds", async ({ page }) => {
    await login(page);
    await navigateToCodeReview(page);

    const editor = page.locator(
      '[data-testid="code-editor"] textarea, .monaco-editor textarea'
    );
    await editor.fill(TEST_CODE);

    const startTime = Date.now();
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-results"]');
    const analysisTime = Date.now() - startTime;

    expect(analysisTime).toBeLessThan(5000);
  });
});

// Accessibility tests
test.describe("Code Review Accessibility", () => {
  test("should be keyboard navigable", async ({ page }) => {
    await login(page);
    await navigateToCodeReview(page);

    // Tab to analyze button
    await page.keyboard.press("Tab");
    await page.keyboard.press("Tab");
    await page.keyboard.press("Tab");

    // Verify focus is visible
    const focusedElement = page.locator(":focus");
    await expect(focusedElement).toBeVisible();
  });

  test("should have proper ARIA labels", async ({ page }) => {
    await login(page);
    await navigateToCodeReview(page);

    // Check ARIA labels
    const analyzeButton = page.locator('[data-testid="analyze-button"]');
    await expect(analyzeButton).toHaveAttribute("aria-label", /.+/);
  });
});
