/**
 * E2E Tests - Projects Management
 */

import { test, expect } from '@playwright/test';

test.describe('Projects Management', () => {
  // Login before each test
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.getByLabel(/email/i).fill('test@example.com');
    await page.getByLabel(/password/i).fill('password123');
    await page.getByRole('button', { name: /login|sign in/i }).click();
    await page.waitForURL(/dashboard/);
  });

  test.describe('Projects List', () => {
    test('should display projects list', async ({ page }) => {
      await page.goto('/projects');
      
      await expect(page.getByRole('heading', { name: /projects/i })).toBeVisible();
    });

    test('should show create project button', async ({ page }) => {
      await page.goto('/projects');
      
      await expect(page.getByRole('button', { name: /create|new project/i })).toBeVisible();
    });

    test('should display project cards', async ({ page }) => {
      await page.goto('/projects');
      
      // Wait for projects to load
      await page.waitForSelector('[data-testid="project-card"], .project-card');
      
      const projectCards = page.locator('[data-testid="project-card"], .project-card');
      await expect(projectCards.first()).toBeVisible();
    });

    test('should filter projects by language', async ({ page }) => {
      await page.goto('/projects');
      
      // Select language filter
      await page.getByLabel(/language/i).click();
      await page.getByRole('option', { name: /python/i }).click();
      
      // Verify filter applied
      await expect(page).toHaveURL(/language=python/);
    });

    test('should search projects', async ({ page }) => {
      await page.goto('/projects');
      
      await page.getByPlaceholder(/search/i).fill('test project');
      await page.keyboard.press('Enter');
      
      // Verify search applied
      await expect(page).toHaveURL(/search=test/);
    });

    test('should sort projects', async ({ page }) => {
      await page.goto('/projects');
      
      // Click sort dropdown
      await page.getByRole('button', { name: /sort/i }).click();
      await page.getByRole('option', { name: /newest|created/i }).click();
      
      // Verify sort applied
      await expect(page).toHaveURL(/sort=/);
    });

    test('should paginate projects', async ({ page }) => {
      await page.goto('/projects');
      
      // Click next page if available
      const nextButton = page.getByRole('button', { name: /next|>|→/i });
      if (await nextButton.isEnabled()) {
        await nextButton.click();
        await expect(page).toHaveURL(/page=2/);
      }
    });
  });

  test.describe('Create Project', () => {
    test('should navigate to create project page', async ({ page }) => {
      await page.goto('/projects');
      
      await page.getByRole('button', { name: /create|new project/i }).click();
      
      await expect(page).toHaveURL(/projects\/new|create/);
    });

    test('should display project form', async ({ page }) => {
      await page.goto('/projects/new');
      
      await expect(page.getByLabel(/project name/i)).toBeVisible();
      await expect(page.getByLabel(/description/i)).toBeVisible();
      await expect(page.getByLabel(/language/i)).toBeVisible();
    });

    test('should create a new project', async ({ page }) => {
      await page.goto('/projects/new');
      
      const projectName = `Test Project ${Date.now()}`;
      
      await page.getByLabel(/project name/i).fill(projectName);
      await page.getByLabel(/description/i).fill('E2E test project description');
      
      // Select language
      await page.getByLabel(/language/i).click();
      await page.getByRole('option', { name: /python/i }).click();
      
      // Submit form
      await page.getByRole('button', { name: /create|submit/i }).click();
      
      // Should redirect to project page
      await expect(page).toHaveURL(/projects\/[a-zA-Z0-9-]+$/);
      await expect(page.getByText(projectName)).toBeVisible();
    });

    test('should show validation errors', async ({ page }) => {
      await page.goto('/projects/new');
      
      // Try to submit empty form
      await page.getByRole('button', { name: /create|submit/i }).click();
      
      await expect(page.getByText(/required|please enter/i)).toBeVisible();
    });

    test('should cancel creation', async ({ page }) => {
      await page.goto('/projects/new');
      
      await page.getByRole('button', { name: /cancel/i }).click();
      
      await expect(page).toHaveURL(/projects$/);
    });
  });

  test.describe('Project Details', () => {
    test('should display project details', async ({ page }) => {
      await page.goto('/projects');
      
      // Click on first project
      await page.locator('[data-testid="project-card"], .project-card').first().click();
      
      // Should show project details
      await expect(page.getByRole('heading')).toBeVisible();
    });

    test('should show project files', async ({ page }) => {
      await page.goto('/projects');
      await page.locator('[data-testid="project-card"], .project-card').first().click();
      
      // Navigate to files tab
      await page.getByRole('tab', { name: /files/i }).click();
      
      await expect(page.getByText(/file tree|files/i)).toBeVisible();
    });

    test('should show project settings', async ({ page }) => {
      await page.goto('/projects');
      await page.locator('[data-testid="project-card"], .project-card').first().click();
      
      // Navigate to settings
      await page.getByRole('tab', { name: /settings/i }).click();
      
      await expect(page.getByText(/project settings/i)).toBeVisible();
    });
  });

  test.describe('Edit Project', () => {
    test('should edit project details', async ({ page }) => {
      await page.goto('/projects');
      await page.locator('[data-testid="project-card"], .project-card').first().click();
      
      // Go to settings
      await page.getByRole('tab', { name: /settings/i }).click();
      
      // Edit name
      const nameInput = page.getByLabel(/project name/i);
      await nameInput.clear();
      await nameInput.fill('Updated Project Name');
      
      // Save
      await page.getByRole('button', { name: /save|update/i }).click();
      
      await expect(page.getByText(/saved|updated/i)).toBeVisible();
    });

    test('should update project description', async ({ page }) => {
      await page.goto('/projects');
      await page.locator('[data-testid="project-card"], .project-card').first().click();
      await page.getByRole('tab', { name: /settings/i }).click();
      
      const descInput = page.getByLabel(/description/i);
      await descInput.clear();
      await descInput.fill('Updated description');
      
      await page.getByRole('button', { name: /save|update/i }).click();
      
      await expect(page.getByText(/saved|updated/i)).toBeVisible();
    });
  });

  test.describe('Delete Project', () => {
    test('should show delete confirmation', async ({ page }) => {
      await page.goto('/projects');
      await page.locator('[data-testid="project-card"], .project-card').first().click();
      await page.getByRole('tab', { name: /settings/i }).click();
      
      // Click delete button
      await page.getByRole('button', { name: /delete/i }).click();
      
      // Should show confirmation dialog
      await expect(page.getByText(/are you sure|confirm/i)).toBeVisible();
    });

    test('should cancel deletion', async ({ page }) => {
      await page.goto('/projects');
      await page.locator('[data-testid="project-card"], .project-card').first().click();
      await page.getByRole('tab', { name: /settings/i }).click();
      
      await page.getByRole('button', { name: /delete/i }).click();
      await page.getByRole('button', { name: /cancel|no/i }).click();
      
      // Dialog should close
      await expect(page.getByText(/are you sure|confirm/i)).not.toBeVisible();
    });

    test('should delete project', async ({ page }) => {
      // Create a project to delete
      await page.goto('/projects/new');
      const projectName = `Delete Test ${Date.now()}`;
      await page.getByLabel(/project name/i).fill(projectName);
      await page.getByLabel(/description/i).fill('To be deleted');
      await page.getByLabel(/language/i).click();
      await page.getByRole('option', { name: /python/i }).click();
      await page.getByRole('button', { name: /create|submit/i }).click();
      await page.waitForURL(/projects\/[a-zA-Z0-9-]+$/);
      
      // Delete the project
      await page.getByRole('tab', { name: /settings/i }).click();
      await page.getByRole('button', { name: /delete/i }).click();
      await page.getByRole('button', { name: /confirm|yes|delete/i }).click();
      
      // Should redirect to projects list
      await expect(page).toHaveURL(/projects$/);
    });
  });

  test.describe('Project Members', () => {
    test('should show team members', async ({ page }) => {
      await page.goto('/projects');
      await page.locator('[data-testid="project-card"], .project-card').first().click();
      
      // Navigate to team tab
      await page.getByRole('tab', { name: /team|members/i }).click();
      
      await expect(page.getByText(/members|team/i)).toBeVisible();
    });

    test('should invite team member', async ({ page }) => {
      await page.goto('/projects');
      await page.locator('[data-testid="project-card"], .project-card').first().click();
      await page.getByRole('tab', { name: /team|members/i }).click();
      
      // Click invite button
      await page.getByRole('button', { name: /invite|add member/i }).click();
      
      // Fill invite form
      await page.getByLabel(/email/i).fill('newmember@example.com');
      await page.getByRole('button', { name: /send|invite/i }).click();
      
      await expect(page.getByText(/invitation sent|invited/i)).toBeVisible();
    });
  });

  test.describe('Project Activity', () => {
    test('should show project activity', async ({ page }) => {
      await page.goto('/projects');
      await page.locator('[data-testid="project-card"], .project-card').first().click();
      
      // Navigate to activity tab
      await page.getByRole('tab', { name: /activity|history/i }).click();
      
      await expect(page.getByText(/activity|history/i)).toBeVisible();
    });
  });
});
