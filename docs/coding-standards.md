# Coding Standards Guide

This document defines coding standards for the AI Code Review Platform to ensure consistency, maintainability, and quality.

## Table of Contents

1. [TypeScript/JavaScript Standards](#typescriptjavascript-standards)
2. [Python Standards](#python-standards)
3. [Documentation Standards](#documentation-standards)
4. [Testing Standards](#testing-standards)
5. [Logging Standards](#logging-standards)
6. [Configuration Standards](#configuration-standards)

---

## TypeScript/JavaScript Standards

### File Organization

```typescript
// 1. Imports (external, then internal)
import React from "react";
import axios from "axios";

import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/Button";

// 2. Types and interfaces
interface UserProps {
  id: string;
  name: string;
}

// 3. Constants
const MAX_RETRIES = 3;

// 4. Component/Function
export function UserCard({ id, name }: UserProps) {
  // Implementation
}

// 5. Default export (if applicable)
export default UserCard;
```

### Naming Conventions

| Type               | Convention           | Example         |
| ------------------ | -------------------- | --------------- |
| Files (components) | PascalCase           | `UserCard.tsx`  |
| Files (utilities)  | camelCase            | `formatDate.ts` |
| Variables          | camelCase            | `userName`      |
| Constants          | SCREAMING_SNAKE      | `MAX_RETRIES`   |
| Types/Interfaces   | PascalCase           | `UserProps`     |
| Functions          | camelCase            | `getUserById`   |
| React components   | PascalCase           | `UserCard`      |
| Hooks              | camelCase with `use` | `useAuth`       |

### JSDoc Comments

All public functions and components must have JSDoc comments:

````typescript
/**
 * Fetches user data from the API.
 *
 * @param userId - The unique identifier of the user
 * @param options - Optional fetch configuration
 * @param options.includeProjects - Whether to include user's projects
 * @returns Promise resolving to the user data
 *
 * @example
 * ```typescript
 * const user = await getUser('user-123', { includeProjects: true });
 * console.log(user.name);
 * ```
 *
 * @throws {NotFoundError} When user doesn't exist
 * @throws {NetworkError} When request fails
 *
 * @see {@link updateUser} for updating user data
 */
async function getUser(
  userId: string,
  options?: GetUserOptions
): Promise<User> {
  // Implementation
}
````

### React Component Documentation

````typescript
/**
 * Displays a user card with avatar and basic information.
 *
 * @component
 * @example
 * ```tsx
 * <UserCard
 *   user={{ id: '1', name: 'John', email: 'john@example.com' }}
 *   onEdit={(user) => console.log('Edit', user)}
 * />
 * ```
 */
interface UserCardProps {
  /** The user object to display */
  user: User;
  /** Callback when edit button is clicked */
  onEdit?: (user: User) => void;
  /** Whether to show the edit button */
  showEditButton?: boolean;
}

export function UserCard({
  user,
  onEdit,
  showEditButton = true,
}: UserCardProps) {
  // Implementation
}
````

### Error Handling

```typescript
// ✅ Good: Specific error handling with context
try {
  const user = await getUser(userId);
  return user;
} catch (error) {
  logger.error("Failed to fetch user", {
    userId,
    error: error instanceof Error ? error.message : "Unknown error",
  });

  if (error instanceof NotFoundError) {
    throw new UserNotFoundError(userId);
  }

  throw new ServiceError("User service unavailable", { cause: error });
}

// ❌ Bad: Generic catch without context
try {
  const user = await getUser(userId);
  return user;
} catch (error) {
  console.log(error);
  throw error;
}
```

---

## Python Standards

### File Organization

```python
"""
Module docstring describing the purpose.

This module provides user management functionality.
"""
# 1. Standard library imports
import os
import json
from typing import Dict, List, Optional

# 2. Third-party imports
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# 3. Local imports
from backend.shared.config import config
from backend.shared.logging import logger

# 4. Constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# 5. Type definitions
class UserCreate(BaseModel):
    """Schema for creating a user."""
    name: str
    email: str

# 6. Functions/Classes
async def create_user(user_data: UserCreate) -> User:
    """Create a new user."""
    pass
```

### Naming Conventions

| Type      | Convention         | Example            |
| --------- | ------------------ | ------------------ |
| Files     | snake_case         | `user_service.py`  |
| Variables | snake_case         | `user_name`        |
| Constants | SCREAMING_SNAKE    | `MAX_RETRIES`      |
| Classes   | PascalCase         | `UserService`      |
| Functions | snake_case         | `get_user_by_id`   |
| Private   | leading underscore | `_internal_method` |

### Docstrings (Google Style)

```python
def get_user(
    user_id: str,
    include_projects: bool = False
) -> User:
    """
    Fetch user data from the database.

    Retrieves user information including optional related data.
    Uses caching to improve performance for repeated requests.

    Args:
        user_id: The unique identifier of the user.
        include_projects: Whether to include the user's projects.
            Defaults to False.

    Returns:
        User object containing the user's data. If include_projects
        is True, the projects field will be populated.

    Raises:
        UserNotFoundError: If the user doesn't exist.
        DatabaseError: If the database connection fails.

    Example:
        >>> user = await get_user("user-123", include_projects=True)
        >>> print(user.name)
        "John Doe"
        >>> len(user.projects)
        5

    Note:
        This function caches results for 5 minutes.

    See Also:
        update_user: For updating user data.
        delete_user: For removing users.
    """
    pass
```

### Class Documentation

```python
class UserService:
    """
    Service for managing user operations.

    Provides methods for CRUD operations on users with
    caching, validation, and audit logging.

    Attributes:
        db: Database connection pool.
        cache: Redis cache client.
        logger: Logger instance for this service.

    Example:
        >>> service = UserService(db_pool, redis_client)
        >>> user = await service.create_user({"name": "John"})
        >>> print(user.id)
        "user-abc123"
    """

    def __init__(self, db, cache):
        """
        Initialize the UserService.

        Args:
            db: Database connection pool.
            cache: Redis cache client.
        """
        self.db = db
        self.cache = cache
        self.logger = get_logger(__name__)
```

---

## Documentation Standards

### When to Document

| Scenario                    | Required |
| --------------------------- | -------- |
| Public functions/methods    | ✅ Yes   |
| Public classes              | ✅ Yes   |
| Complex algorithms          | ✅ Yes   |
| API endpoints               | ✅ Yes   |
| Private functions (complex) | ✅ Yes   |
| Private functions (simple)  | Optional |
| Self-explanatory code       | Optional |

### Inline Comments

```typescript
// ✅ Good: Explains WHY, not WHAT
// Use exponential backoff to prevent overwhelming the server
// during high traffic periods
const delay = Math.pow(2, attempt) * 1000;

// ❌ Bad: Explains what the code obviously does
// Add 1 to counter
counter += 1;
```

### Complex Logic Documentation

For complex algorithms, include:

1. **Overview**: What the algorithm does
2. **Inputs/Outputs**: What goes in and comes out
3. **Steps**: High-level description of steps
4. **Time/Space complexity**: Big O notation
5. **Example**: Concrete example

```typescript
/**
 * Calculates the optimal provider routing based on health scores.
 *
 * Algorithm Overview:
 * This uses a weighted scoring system that considers:
 * - Provider health (40% weight)
 * - Response latency (30% weight)
 * - Cost per request (20% weight)
 * - Success rate (10% weight)
 *
 * Steps:
 * 1. Filter out unhealthy providers (health < 0.5)
 * 2. Calculate weighted score for each provider
 * 3. Sort by score descending
 * 4. Return top N providers
 *
 * Time Complexity: O(n log n) where n = number of providers
 * Space Complexity: O(n) for the sorted array
 *
 * @example
 * Input providers:
 * - OpenAI: health=0.9, latency=100ms, cost=$0.01, success=0.95
 * - Anthropic: health=0.85, latency=150ms, cost=$0.02, success=0.90
 *
 * Scores:
 * - OpenAI: 0.9*0.4 + 0.9*0.3 + 0.8*0.2 + 0.95*0.1 = 0.875
 * - Anthropic: 0.85*0.4 + 0.8*0.3 + 0.6*0.2 + 0.9*0.1 = 0.79
 *
 * Result: [OpenAI, Anthropic]
 */
function selectOptimalProviders(
  providers: Provider[],
  count: number = 3
): Provider[] {
  // Implementation
}
```

---

## Testing Standards

### Test File Organization

```typescript
// __tests__/userService.test.ts

describe("UserService", () => {
  // Setup and teardown
  beforeEach(() => {
    // Reset mocks
  });

  afterEach(() => {
    // Cleanup
  });

  describe("getUser", () => {
    it("should return user when found", async () => {
      // Arrange
      const mockUser = { id: "123", name: "John" };
      mockApi.get.mockResolvedValue({ data: mockUser });

      // Act
      const result = await userService.getUser("123");

      // Assert
      expect(result).toEqual(mockUser);
      expect(mockApi.get).toHaveBeenCalledWith("/users/123");
    });

    it("should throw NotFoundError when user not found", async () => {
      // Arrange
      mockApi.get.mockRejectedValue(new Error("Not found"));

      // Act & Assert
      await expect(userService.getUser("999")).rejects.toThrow(NotFoundError);
    });
  });
});
```

### Test Coverage Requirements

| Category          | Minimum Coverage |
| ----------------- | ---------------- |
| Critical paths    | 90%              |
| Business logic    | 85%              |
| API endpoints     | 80%              |
| Utility functions | 80%              |
| UI components     | 70%              |

---

## Logging Standards

### Log Levels Usage

| Level   | When to Use                          | Example                    |
| ------- | ------------------------------------ | -------------------------- |
| `error` | Errors requiring immediate attention | Database connection failed |
| `warn`  | Potential issues to investigate      | API rate limit approaching |
| `info`  | Normal operation milestones          | User logged in             |
| `http`  | HTTP request/response                | GET /api/users 200 45ms    |
| `debug` | Development debugging info           | Cache hit for key xyz      |

### Logging Best Practices

```typescript
// ✅ Good: Structured logging with context
logger.info("User logged in", {
  userId: user.id,
  method: "oauth",
  provider: "github",
  ip: request.ip,
});

// ✅ Good: Error with full context
logger.error("Failed to process payment", {
  userId: user.id,
  orderId: order.id,
  amount: order.total,
  error: error.message,
  stack: error.stack,
});

// ❌ Bad: No context
logger.info("User logged in");

// ❌ Bad: Sensitive data in logs
logger.info("Login", { password: user.password }); // NEVER DO THIS
```

---

## Configuration Standards

### Environment Variables

```bash
# ✅ Good: Descriptive names with prefixes
DATABASE_URL=postgresql://...
REDIS_HOST=localhost
OPENAI_API_KEY=sk-...
VITE_API_URL=/api

# ❌ Bad: Ambiguous names
URL=postgresql://...
KEY=sk-...
```

### Configuration Validation

```typescript
// Always validate configuration at startup
const errors = validateConfig();
if (errors.length > 0) {
  console.error("Configuration errors:", errors);
  process.exit(1);
}
```

---

## Code Review Checklist

Before submitting a PR, ensure:

- [ ] All public methods have JSDoc/docstrings
- [ ] Complex logic has explanatory comments
- [ ] Unit tests are included (80%+ coverage)
- [ ] Configuration uses environment variables
- [ ] Proper logging with context is used
- [ ] Error handling is comprehensive
- [ ] Types are properly defined
- [ ] No hardcoded values
- [ ] No sensitive data in logs
