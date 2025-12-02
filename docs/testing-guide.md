# Comprehensive Testing Guide

## Overview

This guide covers the testing infrastructure for the AI Code Review Platform, following the testing pyramid approach.

## Testing Pyramid

```
        /\
       /  \      E2E Tests (10%)
      /────\     User workflow tests
     /      \
    /────────\   Integration Tests (20%)
   /          \  API and database tests
  /────────────\
 /              \ Unit Tests (70%)
/________________\ Component and function tests
```

| Level       | Coverage Target | Framework    |
| ----------- | --------------- | ------------ |
| Unit        | 50%+            | Jest, pytest |
| Integration | 20%             | pytest-httpx |
| E2E         | 10%             | Playwright   |

---

## Frontend Testing

### Framework Stack

- **Test Runner**: Jest / Vitest
- **Testing Library**: React Testing Library
- **E2E**: Playwright
- **Mocking**: MSW (Mock Service Worker)

### Directory Structure

```
frontend/
├── tests/
│   ├── setup.ts           # Test setup and mocks
│   ├── __mocks__/         # Module mocks
│   ├── unit/
│   │   ├── components/    # Component tests
│   │   ├── hooks/         # Hook tests
│   │   └── stores/        # Store tests
│   └── integration/
│       ├── auth.test.tsx
│       └── projects.test.tsx
├── e2e/
│   ├── auth.spec.ts
│   ├── projects.spec.ts
│   └── analysis.spec.ts
└── jest.config.js
```

### Running Frontend Tests

```bash
# Run all unit tests
npm test

# Run with coverage
npm run test:coverage

# Run specific test file
npm test -- Sidebar.test.tsx

# Watch mode
npm run test:watch

# Run E2E tests
npx playwright test

# Run E2E with UI
npx playwright test --ui

# Run E2E in specific browser
npx playwright test --project=chromium
```

### Writing Component Tests

```tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MyComponent } from "./MyComponent";

describe("MyComponent", () => {
  it("renders correctly", () => {
    render(<MyComponent />);
    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("handles user interaction", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    render(<MyComponent onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText("Name"), "Test");
    await user.click(screen.getByRole("button", { name: /submit/i }));

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith({ name: "Test" });
    });
  });
});
```

### Writing Hook Tests

```typescript
import { renderHook, act, waitFor } from "@testing-library/react";
import { useMyHook } from "./useMyHook";

describe("useMyHook", () => {
  it("returns initial state", () => {
    const { result } = renderHook(() => useMyHook());
    expect(result.current.data).toBeNull();
  });

  it("fetches data", async () => {
    const { result } = renderHook(() => useMyHook());

    await waitFor(() => {
      expect(result.current.data).not.toBeNull();
    });
  });
});
```

### Writing Store Tests

```typescript
import { renderHook, act } from "@testing-library/react";
import { useMyStore } from "./myStore";

describe("myStore", () => {
  beforeEach(() => {
    // Reset store state
    useMyStore.setState({ items: [] });
  });

  it("adds item", () => {
    const { result } = renderHook(() => useMyStore());

    act(() => {
      result.current.addItem({ id: "1", name: "Test" });
    });

    expect(result.current.items).toHaveLength(1);
  });
});
```

---

## Backend Testing

### Framework Stack

- **Test Runner**: pytest
- **Async Support**: pytest-asyncio
- **Coverage**: pytest-cov
- **Mocking**: unittest.mock, pytest-mock
- **HTTP Testing**: httpx

### Directory Structure

```
backend/
├── tests/
│   ├── conftest.py        # Fixtures
│   ├── factories.py       # Test data factories
│   ├── unit/
│   │   ├── services/
│   │   │   ├── test_auth_service.py
│   │   │   └── test_project_service.py
│   │   ├── models/
│   │   └── utils/
│   └── integration/
│       ├── test_auth_api.py
│       └── test_project_api.py
├── pytest.ini
└── requirements-test.txt
```

### Running Backend Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run specific test
pytest tests/unit/services/test_auth_service.py

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run marked tests
pytest -m "not slow"
```

### Writing Service Tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestAuthService:
    @pytest.fixture
    def auth_service(self):
        service = AuthService()
        service.db = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_authenticate_user(self, auth_service):
        mock_user = User(email="test@example.com")
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )

        result = await auth_service.authenticate("test@example.com", "password")

        assert result is not None
        assert result.email == "test@example.com"
```

### Writing API Tests

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_project(client: AsyncClient, auth_headers):
    response = await client.post(
        "/api/projects",
        json={"name": "Test Project", "language": "python"},
        headers=auth_headers
    )

    assert response.status_code == 201
    assert response.json()["name"] == "Test Project"
```

### Test Fixtures

```python
# conftest.py
import pytest
from httpx import AsyncClient

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
async def auth_headers(client):
    response = await client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "password"
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def user_factory():
    def _create(**kwargs):
        return User(
            id=kwargs.get("id", "user-123"),
            email=kwargs.get("email", "test@example.com"),
            **kwargs
        )
    return _create
```

---

## E2E Testing

### Playwright Configuration

```typescript
// playwright.config.ts
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "chromium", use: devices["Desktop Chrome"] },
    { name: "firefox", use: devices["Desktop Firefox"] },
    { name: "Mobile Safari", use: devices["iPhone 12"] },
  ],
});
```

### Writing E2E Tests

```typescript
import { test, expect } from "@playwright/test";

test.describe("Projects", () => {
  test.beforeEach(async ({ page }) => {
    // Login
    await page.goto("/login");
    await page.fill('[name="email"]', "test@example.com");
    await page.fill('[name="password"]', "password");
    await page.click('button[type="submit"]');
    await page.waitForURL("/dashboard");
  });

  test("should create project", async ({ page }) => {
    await page.goto("/projects/new");
    await page.fill('[name="name"]', "Test Project");
    await page.click('button[type="submit"]');

    await expect(page).toHaveURL(/\/projects\/.+/);
    await expect(page.getByText("Test Project")).toBeVisible();
  });
});
```

---

## Test Data Factories

### Frontend Factory

```typescript
// factories/user.ts
export const createUser = (overrides = {}) => ({
  id: "user-123",
  email: "test@example.com",
  name: "Test User",
  role: "user",
  createdAt: new Date().toISOString(),
  ...overrides,
});

export const createProject = (overrides = {}) => ({
  id: "project-123",
  name: "Test Project",
  language: "python",
  ownerId: "user-123",
  ...overrides,
});
```

### Backend Factory

```python
# factories.py
import factory
from factory.alchemy import SQLAlchemyModelFactory
from app.models import User, Project

class UserFactory(SQLAlchemyModelFactory):
    class Meta:
        model = User

    id = factory.Faker('uuid4')
    email = factory.Faker('email')
    name = factory.Faker('name')
    role = 'user'

class ProjectFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Project

    id = factory.Faker('uuid4')
    name = factory.Faker('company')
    language = 'python'
    owner = factory.SubFactory(UserFactory)
```

---

## Mocking

### Frontend Mocking with MSW

```typescript
// mocks/handlers.ts
import { rest } from "msw";

export const handlers = [
  rest.get("/api/projects", (req, res, ctx) => {
    return res(
      ctx.json({
        items: [
          { id: "1", name: "Project 1" },
          { id: "2", name: "Project 2" },
        ],
        total: 2,
      })
    );
  }),

  rest.post("/api/projects", async (req, res, ctx) => {
    const body = await req.json();
    return res(
      ctx.status(201),
      ctx.json({
        id: "3",
        ...body,
      })
    );
  }),
];
```

### Backend Mocking

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mock():
    with patch('app.services.external_api.fetch') as mock_fetch:
        mock_fetch.return_value = {'data': 'mocked'}

        result = await my_function()

        assert result['data'] == 'mocked'
        mock_fetch.assert_called_once()
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Tests
on: [push, pull_request]

jobs:
  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci && npm test

  backend-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r requirements-test.txt
      - run: pytest --cov=app

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npx playwright install
      - run: npx playwright test
```

---

## Best Practices

### 1. Test Naming

```typescript
// Good
it("should display error message when login fails");
it("renders project list with pagination");

// Bad
it("test login");
it("works");
```

### 2. AAA Pattern

```typescript
test("adds item to cart", () => {
  // Arrange
  const cart = new Cart();
  const item = { id: "1", price: 10 };

  // Act
  cart.add(item);

  // Assert
  expect(cart.items).toContain(item);
});
```

### 3. Isolation

```python
@pytest.fixture(autouse=True)
def reset_database(db):
    yield
    db.rollback()
```

### 4. Descriptive Assertions

```typescript
// Good
expect(user.email).toBe("test@example.com");

// Better (custom matchers)
expect(user).toBeValidUser();
```

### 5. Avoid Test Interdependence

```typescript
// Bad - tests depend on order
test('creates user', () => { ... });
test('uses created user', () => { ... }); // Depends on previous

// Good - independent tests
test('creates user', () => { ... });
test('fetches user', async () => {
  await createTestUser(); // Setup in test
  // ...
});
```

---

## Troubleshooting

### Common Issues

| Issue            | Solution                                 |
| ---------------- | ---------------------------------------- |
| Tests timeout    | Increase timeout, check async operations |
| Flaky tests      | Add proper waits, check race conditions  |
| Mock not working | Verify mock path matches import path     |
| Database state   | Ensure proper cleanup between tests      |
| Port conflicts   | Use random ports or cleanup properly     |

### Debug Commands

```bash
# Frontend debug
DEBUG=pw:api npx playwright test

# Backend debug
pytest -v --tb=long --capture=no

# Run single test with output
pytest tests/unit/test_auth.py::test_login -v -s
```
