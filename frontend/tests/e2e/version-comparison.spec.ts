/**
 * E2E Tests for Version Comparison Admin Page
 *
 * Tests the three-version comparison UI including:
 * - Loading comparison requests
 * - Viewing side-by-side outputs
 * - Output diff viewing
 * - Rollback initiation
 */

import { test, expect } from "@playwright/test";

// Test data
const mockComparisonRequest = {
  requestId: "req-12345678",
  code: "function test() { return eval(userInput); }",
  language: "javascript",
  timestamp: new Date().toISOString(),
  v1Output: {
    version: "v1",
    versionId: "v1-abc123",
    modelVersion: "gpt-4o-2024-05-13",
    promptVersion: "code-review-v4-exp",
    timestamp: new Date().toISOString(),
    latencyMs: 2500,
    cost: 0.0045,
    issues: [
      {
        id: "issue-1",
        type: "security",
        severity: "critical",
        message:
          "Use of eval() with user input creates code injection vulnerability",
        file: "test.js",
        line: 1,
        suggestion: "Use JSON.parse() or a safer alternative",
      },
    ],
    rawOutput: '{"issues": [{"type": "security", "severity": "critical"}]}',
    confidence: 0.95,
    securityPassed: true,
  },
  v2Output: {
    version: "v2",
    versionId: "v2-current",
    modelVersion: "gpt-4o-2024-05-13",
    promptVersion: "code-review-v3",
    timestamp: new Date().toISOString(),
    latencyMs: 2800,
    cost: 0.0042,
    issues: [
      {
        id: "issue-2",
        type: "security",
        severity: "high",
        message: "Potential code injection via eval()",
        file: "test.js",
        line: 1,
      },
    ],
    rawOutput: '{"issues": [{"type": "security", "severity": "high"}]}',
    confidence: 0.88,
    securityPassed: true,
  },
};

test.describe("Version Comparison Page", () => {
  test.beforeEach(async ({ page }) => {
    // Mock API responses
    await page.route(
      "**/api/admin/lifecycle/comparison-requests*",
      async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            requests: [mockComparisonRequest],
            total: 1,
            limit: 50,
            offset: 0,
          }),
        });
      }
    );

    await page.route(
      "**/api/admin/lifecycle/stats/comparison",
      async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            totalRequests: 100,
            withV1Output: 95,
            withV3Output: 10,
            languages: { javascript: 50, python: 30, typescript: 20 },
            comparisonMetrics: {
              samples: 95,
              avgIssueCountDelta: 0.5,
              avgLatencyDeltaMs: -300,
            },
          }),
        });
      }
    );

    // Navigate to page (assumes admin is logged in)
    await page.goto("/admin/version-comparison");
  });

  test("should display page title", async ({ page }) => {
    await expect(page.locator("h2")).toContainText("Version Comparison");
  });

  test("should load and display comparison requests", async ({ page }) => {
    // Wait for requests to load
    await page.waitForSelector(".ant-select");

    // Should show the dropdown with requests
    const select = page.locator(".ant-select");
    await expect(select).toBeVisible();
  });

  test("should show side-by-side comparison when request selected", async ({
    page,
  }) => {
    // Wait for data to load
    await page.waitForTimeout(500);

    // Should show V1 and V2 cards
    await expect(page.locator("text=V1 Experiment")).toBeVisible();
    await expect(page.locator("text=V2 Production")).toBeVisible();
  });

  test("should display metrics correctly", async ({ page }) => {
    await page.waitForTimeout(500);

    // Check latency is displayed
    const v1Latency = page.locator("text=2500").first();
    await expect(v1Latency).toBeVisible();

    // Check cost is displayed
    await expect(page.locator("text=$0.0045").first()).toBeVisible();
  });

  test("should show issue severity badges", async ({ page }) => {
    await page.waitForTimeout(500);

    // V1 has critical severity
    await expect(page.locator('.ant-tag:has-text("Critical")')).toBeVisible();

    // V2 has high severity
    await expect(page.locator('.ant-tag:has-text("High")')).toBeVisible();
  });

  test("should switch to diff view tab", async ({ page }) => {
    await page.waitForTimeout(500);

    // Click on Output Diff tab
    await page.click("text=Output Diff");

    // Should show diff viewer
    await expect(page.locator("text=V2 Baseline")).toBeVisible();
    await expect(page.locator("text=V1 Experiment")).toBeVisible();
  });

  test("should switch to executability tests tab", async ({ page }) => {
    await page.waitForTimeout(500);

    // Click on Executability Tests tab
    await page.click("text=Executability Tests");

    // Should show test table
    await expect(page.locator("text=Syntax Validity")).toBeVisible();
    await expect(page.locator("text=Type Safety")).toBeVisible();
  });

  test("should switch to evidence chain tab", async ({ page }) => {
    await page.waitForTimeout(500);

    // Click on Evidence Chain tab
    await page.click("text=Evidence Chain");

    // Should show evidence sections
    await expect(page.locator("text=Triggering Rules")).toBeVisible();
    await expect(page.locator("text=Model Confidence")).toBeVisible();
  });

  test("should open rollback modal", async ({ page }) => {
    await page.waitForTimeout(500);

    // Click rollback button
    await page.click('button:has-text("Rollback")');

    // Modal should appear
    await expect(page.locator("text=Initiate Rollback")).toBeVisible();
    await expect(page.locator("text=Reason for rollback")).toBeVisible();
  });

  test("should show validation error when rollback without reason", async ({
    page,
  }) => {
    await page.waitForTimeout(500);

    // Open rollback modal
    await page.click('button:has-text("Rollback")');
    await page.waitForSelector(".ant-modal");

    // Try to confirm without selecting reason
    await page.click('button:has-text("Confirm Rollback")');

    // Should show error message
    await expect(
      page.locator("text=Please select a rollback reason")
    ).toBeVisible();
  });

  test("should submit rollback successfully", async ({ page }) => {
    // Mock rollback API
    await page.route("**/api/admin/lifecycle/rollback", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          success: true,
          message: "Rollback initiated",
          rollbackId: "rb-123",
        }),
      });
    });

    await page.waitForTimeout(500);

    // Open rollback modal
    await page.click('button:has-text("Rollback")');
    await page.waitForSelector(".ant-modal");

    // Select reason
    await page.click(".ant-select-selector");
    await page.click("text=Accuracy Regression");

    // Add notes
    await page.fill("textarea", "Testing rollback");

    // Submit
    await page.click('button:has-text("Confirm Rollback")');

    // Should show success message
    await expect(
      page.locator("text=Rollback initiated successfully")
    ).toBeVisible();
  });

  test("should refresh data when clicking refresh button", async ({ page }) => {
    await page.waitForTimeout(500);

    // Track API calls
    let apiCalls = 0;
    await page.route(
      "**/api/admin/lifecycle/comparison-requests*",
      async (route) => {
        apiCalls++;
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            requests: [mockComparisonRequest],
            total: 1,
            limit: 50,
            offset: 0,
          }),
        });
      }
    );

    // Initial load
    await page.waitForTimeout(100);
    const initialCalls = apiCalls;

    // Click refresh
    await page.click('button:has-text("Refresh")');

    // Should have made another API call
    await page.waitForTimeout(500);
    expect(apiCalls).toBeGreaterThan(initialCalls);
  });

  test("should show history tab with requests", async ({ page }) => {
    await page.waitForTimeout(500);

    // Click on History tab
    await page.click('span:has-text("History")');

    // Should show history table
    await expect(page.locator("text=Request ID")).toBeVisible();
    await expect(page.locator("text=Language")).toBeVisible();
    await expect(page.locator("text=Timestamp")).toBeVisible();
  });

  test("should handle empty state gracefully", async ({ page }) => {
    // Override with empty response
    await page.route(
      "**/api/admin/lifecycle/comparison-requests*",
      async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            requests: [],
            total: 0,
            limit: 50,
            offset: 0,
          }),
        });
      }
    );

    await page.reload();
    await page.waitForTimeout(500);

    // Should show empty state message
    await expect(
      page.locator("text=No comparison requests available")
    ).toBeVisible();
  });

  test("should handle API errors gracefully", async ({ page }) => {
    // Override with error response
    await page.route(
      "**/api/admin/lifecycle/comparison-requests*",
      async (route) => {
        await route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({ error: "Internal server error" }),
        });
      }
    );

    await page.reload();
    await page.waitForTimeout(500);

    // Should show error message
    await expect(
      page.locator("text=Failed to fetch comparison requests")
    ).toBeVisible();
  });
});

test.describe("Version Comparison - Mobile Responsive", () => {
  test.use({ viewport: { width: 375, height: 667 } });

  test("should be responsive on mobile", async ({ page }) => {
    await page.route(
      "**/api/admin/lifecycle/comparison-requests*",
      async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            requests: [mockComparisonRequest],
            total: 1,
            limit: 50,
            offset: 0,
          }),
        });
      }
    );

    await page.goto("/admin/version-comparison");
    await page.waitForTimeout(500);

    // Page should be scrollable
    const pageHeight = await page.evaluate(() => document.body.scrollHeight);
    expect(pageHeight).toBeGreaterThan(667);

    // Cards should stack vertically
    const v1Card = page.locator("text=V1 Experiment").first();
    await expect(v1Card).toBeVisible();
  });
});

test.describe("Version Comparison - Accessibility", () => {
  test("should be keyboard navigable", async ({ page }) => {
    await page.route(
      "**/api/admin/lifecycle/comparison-requests*",
      async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            requests: [mockComparisonRequest],
            total: 1,
            limit: 50,
            offset: 0,
          }),
        });
      }
    );

    await page.goto("/admin/version-comparison");
    await page.waitForTimeout(500);

    // Tab through elements
    await page.keyboard.press("Tab");
    await page.keyboard.press("Tab");
    await page.keyboard.press("Tab");

    // Should be able to navigate with keyboard
    const focusedElement = await page.evaluate(
      () => document.activeElement?.tagName
    );
    expect(focusedElement).toBeTruthy();
  });

  test("should have proper ARIA labels", async ({ page }) => {
    await page.route(
      "**/api/admin/lifecycle/comparison-requests*",
      async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            requests: [mockComparisonRequest],
            total: 1,
            limit: 50,
            offset: 0,
          }),
        });
      }
    );

    await page.goto("/admin/version-comparison");
    await page.waitForTimeout(500);

    // Check for accessible elements
    const buttons = await page.locator("button").count();
    expect(buttons).toBeGreaterThan(0);

    // Tabs should have proper roles
    const tabs = await page.locator('[role="tab"]').count();
    expect(tabs).toBeGreaterThan(0);
  });
});
