/**
 * E2E Tests for Admin Functionality
 *
 * Tests cover admin-specific features:
 * - User management
 * - System configuration
 * - Experiment management (V1/V2/V3)
 * - Audit log viewing
 * - Multi-signature approvals
 */
import { test, expect, Page, Route } from "@playwright/test";

const BASE_URL = "http://localhost:3000";
const API_URL = "http://localhost:8000";

// Helper functions
async function loginAsAdmin(page: Page): Promise<void> {
  await page.route(`${API_URL}/api/auth/login`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        access_token: "admin-token",
        user: { id: "admin1", email: "admin@example.com", role: "admin" },
      }),
    });
  });

  await page.goto(`${BASE_URL}/login`);
  await page.fill('[data-testid="email-input"]', "admin@example.com");
  await page.fill('[data-testid="password-input"]', "adminpass");
  await page.click('[data-testid="login-button"]');
  await page.waitForURL("**/dashboard");
}

// User Management Tests
test.describe("Admin - User Management", () => {
  test.beforeEach(async ({ page }) => {
    // Mock users list
    await page.route(`${API_URL}/api/admin/users`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          users: [
            {
              id: "user1",
              email: "user1@example.com",
              role: "user",
              status: "active",
              created_at: "2024-01-01",
            },
            {
              id: "user2",
              email: "user2@example.com",
              role: "user",
              status: "suspended",
              created_at: "2024-01-02",
            },
            {
              id: "admin1",
              email: "admin@example.com",
              role: "admin",
              status: "active",
              created_at: "2024-01-01",
            },
          ],
          total: 3,
        }),
      });
    });

    await loginAsAdmin(page);
  });

  test("should display user list", async ({ page }) => {
    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-users"]');

    const userTable = page.locator('[data-testid="users-table"]');
    await expect(userTable).toBeVisible();

    const rows = page.locator('[data-testid="user-row"]');
    expect(await rows.count()).toBe(3);
  });

  test("should suspend user", async ({ page }) => {
    await page.route(`${API_URL}/api/admin/users/*/suspend`, async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({ success: true }),
      });
    });

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-users"]');

    // Click suspend on first active user
    await page.click(
      '[data-testid="user-row"]:first-child [data-testid="suspend-user"]'
    );

    // Confirm
    await page.click('[data-testid="confirm-dialog-yes"]');

    // Verify success message
    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();
  });

  test("should reactivate suspended user", async ({ page }) => {
    await page.route(
      `${API_URL}/api/admin/users/*/reactivate`,
      async (route) => {
        await route.fulfill({
          status: 200,
          body: JSON.stringify({ success: true }),
        });
      }
    );

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-users"]');

    // Find suspended user and reactivate
    await page.click(
      '[data-testid="user-row"]:nth-child(2) [data-testid="reactivate-user"]'
    );
    await page.click('[data-testid="confirm-dialog-yes"]');

    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();
  });

  test("should reset user password", async ({ page }) => {
    await page.route(
      `${API_URL}/api/admin/users/*/reset-password`,
      async (route) => {
        await route.fulfill({
          status: 200,
          body: JSON.stringify({ temporary_password: "temp123" }),
        });
      }
    );

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-users"]');

    await page.click(
      '[data-testid="user-row"]:first-child [data-testid="reset-password"]'
    );
    await page.click('[data-testid="confirm-dialog-yes"]');

    // Verify temp password is shown
    await expect(
      page.locator('[data-testid="temp-password-dialog"]')
    ).toBeVisible();
  });

  test("should filter users by status", async ({ page }) => {
    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-users"]');

    // Filter by suspended
    await page.selectOption('[data-testid="status-filter"]', "suspended");

    // Should show only suspended users
    const rows = page.locator('[data-testid="user-row"]');
    expect(await rows.count()).toBe(1);
  });

  test("should search users by email", async ({ page }) => {
    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-users"]');

    await page.fill('[data-testid="search-users"]', "user1");

    // Should filter results
    const rows = page.locator('[data-testid="user-row"]');
    await expect(rows.first()).toContainText("user1@example.com");
  });
});

// Three-Version Control Tests
test.describe("Admin - Three-Version Control", () => {
  test.beforeEach(async ({ page }) => {
    // Mock evolution status
    await page.route(`${API_URL}/api/three-version/status`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          cycle_running: true,
          current_phase: "monitoring",
          v1: { status: "experimenting", experiments: 3, accuracy: 0.82 },
          v2: { status: "production", requests_24h: 15000, accuracy: 0.91 },
          v3: { status: "quarantine", archived: 5 },
        }),
      });
    });

    await loginAsAdmin(page);
  });

  test("should display version status dashboard", async ({ page }) => {
    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-three-version"]');

    // Verify all three version cards are visible
    await expect(page.locator('[data-testid="v1-status"]')).toBeVisible();
    await expect(page.locator('[data-testid="v2-status"]')).toBeVisible();
    await expect(page.locator('[data-testid="v3-status"]')).toBeVisible();
  });

  test("should display V1 experiments", async ({ page }) => {
    await page.route(
      `${API_URL}/api/three-version/experiments`,
      async (route) => {
        await route.fulfill({
          status: 200,
          body: JSON.stringify({
            experiments: [
              {
                id: "exp1",
                name: "GPT-4 Turbo Test",
                status: "running",
                accuracy: 0.85,
              },
              {
                id: "exp2",
                name: "Claude-3 Opus",
                status: "completed",
                accuracy: 0.88,
              },
            ],
          }),
        });
      }
    );

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-three-version"]');
    await page.click('[data-testid="view-experiments"]');

    const experimentTable = page.locator('[data-testid="experiments-table"]');
    await expect(experimentTable).toBeVisible();
  });

  test("should promote experiment from V1 to V2", async ({ page }) => {
    await page.route(`${API_URL}/api/three-version/promote`, async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({ success: true }),
      });
    });

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-three-version"]');
    await page.click('[data-testid="promote-button"]');

    // Fill promotion form
    await page.fill(
      '[data-testid="promotion-reason"]',
      "Accuracy threshold met"
    );
    await page.click('[data-testid="confirm-promote"]');

    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();
  });

  test("should demote V2 to V3 (quarantine)", async ({ page }) => {
    await page.route(`${API_URL}/api/three-version/demote`, async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({ success: true }),
      });
    });

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-three-version"]');
    await page.click('[data-testid="demote-button"]');

    await page.fill(
      '[data-testid="demotion-reason"]',
      "Error rate exceeded threshold"
    );
    await page.click('[data-testid="confirm-demote"]');

    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();
  });

  test("should re-evaluate V3 experiment", async ({ page }) => {
    await page.route(
      `${API_URL}/api/three-version/reevaluate`,
      async (route) => {
        await route.fulfill({
          status: 200,
          body: JSON.stringify({ success: true }),
        });
      }
    );

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-three-version"]');
    await page.click(
      '[data-testid="v3-status"] [data-testid="reevaluate-button"]'
    );

    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();
  });

  test("should show evolution metrics", async ({ page }) => {
    await page.route(`${API_URL}/api/three-version/metrics`, async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({
          promotions_30d: 5,
          demotions_30d: 2,
          avg_experiment_duration: "3.5 days",
          current_accuracy: 0.91,
        }),
      });
    });

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-three-version"]');
    await page.click('[data-testid="view-metrics"]');

    const metricsPanel = page.locator('[data-testid="evolution-metrics"]');
    await expect(metricsPanel).toBeVisible();
  });
});

// Audit Log Tests
test.describe("Admin - Audit Logs", () => {
  test.beforeEach(async ({ page }) => {
    await page.route(`${API_URL}/api/admin/audit/logs`, async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({
          logs: [
            {
              id: "log1",
              entity: "version",
              action: "promote",
              actor: "admin@example.com",
              timestamp: "2024-01-15T10:30:00Z",
            },
            {
              id: "log2",
              entity: "user",
              action: "create",
              actor: "admin@example.com",
              timestamp: "2024-01-15T09:00:00Z",
            },
            {
              id: "log3",
              entity: "experiment",
              action: "quarantine",
              actor: "system",
              timestamp: "2024-01-14T22:00:00Z",
            },
          ],
          total: 3,
        }),
      });
    });

    await loginAsAdmin(page);
  });

  test("should display audit log list", async ({ page }) => {
    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-audit"]');

    const auditTable = page.locator('[data-testid="audit-table"]');
    await expect(auditTable).toBeVisible();
  });

  test("should filter audit logs by entity", async ({ page }) => {
    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-audit"]');

    await page.selectOption('[data-testid="entity-filter"]', "version");

    const rows = page.locator('[data-testid="audit-row"]');
    expect(await rows.count()).toBeGreaterThan(0);
  });

  test("should filter audit logs by date range", async ({ page }) => {
    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-audit"]');

    await page.fill('[data-testid="date-from"]', "2024-01-01");
    await page.fill('[data-testid="date-to"]', "2024-01-31");
    await page.click('[data-testid="apply-filters"]');

    const auditTable = page.locator('[data-testid="audit-table"]');
    await expect(auditTable).toBeVisible();
  });

  test("should export audit logs", async ({ page }) => {
    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-audit"]');

    const downloadPromise = page.waitForEvent("download");
    await page.click('[data-testid="export-audit-logs"]');
    const download = await downloadPromise;

    expect(download.suggestedFilename()).toContain("audit");
  });

  test("should verify audit log integrity", async ({ page }) => {
    await page.route(`${API_URL}/api/admin/audit/verify`, async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({
          valid: true,
          verified_count: 100,
          tampered_count: 0,
        }),
      });
    });

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-audit"]');
    await page.click('[data-testid="verify-integrity"]');

    await expect(
      page.locator('[data-testid="integrity-result"]')
    ).toContainText("valid");
  });
});

// Multi-Signature Approval Tests
test.describe("Admin - Multi-Signature Approvals", () => {
  test.beforeEach(async ({ page }) => {
    await page.route(`${API_URL}/api/admin/multisig/pending`, async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({
          requests: [
            {
              id: "req1",
              operation: "version_promotion",
              requester: "admin1@example.com",
              signatures: 1,
              required: 2,
              expires_at: new Date(Date.now() + 86400000).toISOString(),
            },
          ],
        }),
      });
    });

    await loginAsAdmin(page);
  });

  test("should display pending approval requests", async ({ page }) => {
    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-multisig"]');

    const requestsTable = page.locator('[data-testid="multisig-requests"]');
    await expect(requestsTable).toBeVisible();
  });

  test("should sign pending request", async ({ page }) => {
    await page.route(`${API_URL}/api/admin/multisig/sign/*`, async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({ success: true }),
      });
    });

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-multisig"]');
    await page.click('[data-testid="sign-request"]');

    // Enter signing password/key
    await page.fill('[data-testid="signing-password"]', "signingkey123");
    await page.click('[data-testid="confirm-sign"]');

    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();
  });

  test("should reject pending request", async ({ page }) => {
    await page.route(
      `${API_URL}/api/admin/multisig/reject/*`,
      async (route) => {
        await route.fulfill({
          status: 200,
          body: JSON.stringify({ success: true }),
        });
      }
    );

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-multisig"]');
    await page.click('[data-testid="reject-request"]');

    await page.fill(
      '[data-testid="rejection-reason"]',
      "Not needed at this time"
    );
    await page.click('[data-testid="confirm-reject"]');

    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();
  });
});

// System Configuration Tests
test.describe("Admin - System Configuration", () => {
  test.beforeEach(async ({ page }) => {
    await page.route(`${API_URL}/api/admin/config`, async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({
          ai_provider: "openai",
          model: "gpt-4",
          max_tokens: 4096,
          temperature: 0.7,
          rate_limit_per_user: 100,
          promotion_threshold: 0.85,
        }),
      });
    });

    await loginAsAdmin(page);
  });

  test("should display system configuration", async ({ page }) => {
    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-config"]');

    const configForm = page.locator('[data-testid="config-form"]');
    await expect(configForm).toBeVisible();
  });

  test("should update AI provider settings", async ({ page }) => {
    await page.route(`${API_URL}/api/admin/config`, async (route) => {
      if (route.request().method() === "PUT") {
        await route.fulfill({
          status: 200,
          body: JSON.stringify({ success: true }),
        });
      }
    });

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-config"]');

    await page.selectOption('[data-testid="ai-provider-select"]', "anthropic");
    await page.click('[data-testid="save-config"]');

    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();
  });

  test("should update promotion threshold", async ({ page }) => {
    await page.route(`${API_URL}/api/admin/config`, async (route) => {
      if (route.request().method() === "PUT") {
        await route.fulfill({
          status: 200,
          body: JSON.stringify({ success: true }),
        });
      }
    });

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-config"]');

    await page.fill('[data-testid="promotion-threshold"]', "0.90");
    await page.click('[data-testid="save-config"]');

    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();
  });
});

// Provider Health Tests
test.describe("Admin - Provider Health", () => {
  test.beforeEach(async ({ page }) => {
    await page.route(`${API_URL}/api/admin/providers/health`, async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({
          providers: [
            {
              name: "OpenAI",
              status: "healthy",
              latency_ms: 150,
              success_rate: 0.99,
            },
            {
              name: "Anthropic",
              status: "degraded",
              latency_ms: 500,
              success_rate: 0.85,
            },
            {
              name: "Local Model",
              status: "healthy",
              latency_ms: 50,
              success_rate: 1.0,
            },
          ],
        }),
      });
    });

    await loginAsAdmin(page);
  });

  test("should display provider health status", async ({ page }) => {
    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-providers"]');

    const healthTable = page.locator('[data-testid="provider-health-table"]');
    await expect(healthTable).toBeVisible();
  });

  test("should trigger manual health check", async ({ page }) => {
    await page.route(`${API_URL}/api/admin/providers/check`, async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({ success: true }),
      });
    });

    await page.click('[data-testid="nav-admin"]');
    await page.click('[data-testid="admin-providers"]');
    await page.click('[data-testid="check-health-button"]');

    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();
  });
});
