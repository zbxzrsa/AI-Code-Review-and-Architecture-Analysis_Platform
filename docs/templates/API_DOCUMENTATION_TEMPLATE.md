# API Documentation: [API Name]

| **Document Information** |                            |
| ------------------------ | -------------------------- |
| **API Version**          | v1.0.0                     |
| **OpenAPI Version**      | 3.0.3                      |
| **Base URL**             | https://api.example.com/v1 |
| **Last Updated**         | YYYY-MM-DD                 |

---

## Change History

| Version | Date       | Author        | Description         |
| ------- | ---------- | ------------- | ------------------- |
| 1.0.0   | YYYY-MM-DD | [Author Name] | Initial API release |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Authentication](#2-authentication)
3. [Rate Limiting](#3-rate-limiting)
4. [Error Handling](#4-error-handling)
5. [API Endpoints](#5-api-endpoints)
6. [Data Models](#6-data-models)
7. [Examples](#7-examples)
8. [SDKs and Tools](#8-sdks-and-tools)
9. [Changelog](#9-changelog)

---

## 1. Introduction

### 1.1 Overview

[Brief description of what this API does and its main use cases.]

### 1.2 Base URL

```
Production: https://api.example.com/v1
Staging:    https://api-staging.example.com/v1
```

### 1.3 Request Format

All requests must:

- Use HTTPS
- Include `Content-Type: application/json` header
- Include authentication headers (see Authentication section)

### 1.4 Response Format

All responses are in JSON format with the following structure:

**Success Response:**

```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "request_id": "req-uuid",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

**Error Response:**

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": { ... }
  },
  "meta": {
    "request_id": "req-uuid",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

---

## 2. Authentication

### 2.1 API Key Authentication

Include your API key in the request header:

```http
Authorization: Bearer YOUR_API_KEY
```

### 2.2 Obtaining API Keys

1. Log in to the [Developer Portal](https://developer.example.com)
2. Navigate to Settings > API Keys
3. Click "Generate New Key"

### 2.3 API Key Best Practices

- Never expose API keys in client-side code
- Rotate keys regularly
- Use different keys for different environments
- Set appropriate scopes for each key

---

## 3. Rate Limiting

### 3.1 Rate Limits

| Plan       | Requests/Minute | Requests/Day |
| ---------- | --------------- | ------------ |
| Free       | 60              | 1,000        |
| Pro        | 300             | 50,000       |
| Enterprise | Custom          | Custom       |

### 3.2 Rate Limit Headers

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1704067200
```

### 3.3 Handling Rate Limits

When rate limited, you'll receive a `429 Too Many Requests` response:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 60 seconds.",
    "retry_after": 60
  }
}
```

---

## 4. Error Handling

### 4.1 HTTP Status Codes

| Status Code | Description                             |
| ----------- | --------------------------------------- |
| 200         | Success                                 |
| 201         | Created                                 |
| 400         | Bad Request - Invalid parameters        |
| 401         | Unauthorized - Invalid/missing auth     |
| 403         | Forbidden - Insufficient permissions    |
| 404         | Not Found - Resource doesn't exist      |
| 422         | Unprocessable Entity - Validation error |
| 429         | Too Many Requests - Rate limited        |
| 500         | Internal Server Error                   |

### 4.2 Error Codes

| Error Code             | Description                  |
| ---------------------- | ---------------------------- |
| `INVALID_REQUEST`      | Request body is malformed    |
| `VALIDATION_ERROR`     | Request validation failed    |
| `AUTHENTICATION_ERROR` | Authentication failed        |
| `AUTHORIZATION_ERROR`  | Insufficient permissions     |
| `RESOURCE_NOT_FOUND`   | Requested resource not found |
| `RATE_LIMIT_EXCEEDED`  | Rate limit exceeded          |
| `INTERNAL_ERROR`       | Internal server error        |

---

## 5. API Endpoints

### 5.1 [Resource Name]

#### 5.1.1 List [Resources]

```http
GET /api/v1/resources
```

**Query Parameters:**

| Parameter | Type    | Required | Description                     |
| --------- | ------- | -------- | ------------------------------- |
| page      | integer | No       | Page number (default: 1)        |
| limit     | integer | No       | Items per page (default: 20)    |
| sort      | string  | No       | Sort field (e.g., "created_at") |
| order     | string  | No       | Sort order: "asc" or "desc"     |

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "res-123",
        "name": "Resource Name",
        "created_at": "2024-01-01T00:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 100,
      "total_pages": 5
    }
  }
}
```

#### 5.1.2 Get [Resource]

```http
GET /api/v1/resources/{id}
```

**Path Parameters:**

| Parameter | Type   | Required | Description         |
| --------- | ------ | -------- | ------------------- |
| id        | string | Yes      | Resource identifier |

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "id": "res-123",
    "name": "Resource Name",
    "description": "Resource description",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-02T00:00:00Z"
  }
}
```

#### 5.1.3 Create [Resource]

```http
POST /api/v1/resources
```

**Request Body:**

```json
{
  "name": "New Resource",
  "description": "Resource description"
}
```

**Request Body Schema:**

| Field       | Type   | Required | Description             |
| ----------- | ------ | -------- | ----------------------- |
| name        | string | Yes      | Resource name (max 255) |
| description | string | No       | Resource description    |

**Response (201 Created):**

```json
{
  "success": true,
  "data": {
    "id": "res-456",
    "name": "New Resource",
    "description": "Resource description",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

#### 5.1.4 Update [Resource]

```http
PUT /api/v1/resources/{id}
```

**Request Body:**

```json
{
  "name": "Updated Name",
  "description": "Updated description"
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "data": {
    "id": "res-123",
    "name": "Updated Name",
    "description": "Updated description",
    "updated_at": "2024-01-02T00:00:00Z"
  }
}
```

#### 5.1.5 Delete [Resource]

```http
DELETE /api/v1/resources/{id}
```

**Response (204 No Content):**

No response body.

---

## 6. Data Models

### 6.1 Resource

```typescript
interface Resource {
  id: string; // Unique identifier
  name: string; // Resource name
  description?: string; // Optional description
  status: ResourceStatus;
  created_at: string; // ISO 8601 datetime
  updated_at: string; // ISO 8601 datetime
}

enum ResourceStatus {
  ACTIVE = "active",
  INACTIVE = "inactive",
  DELETED = "deleted",
}
```

### 6.2 Pagination

```typescript
interface Pagination {
  page: number; // Current page
  limit: number; // Items per page
  total: number; // Total items
  total_pages: number; // Total pages
}
```

---

## 7. Examples

### 7.1 cURL Examples

**List Resources:**

```bash
curl -X GET "https://api.example.com/v1/resources" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json"
```

**Create Resource:**

```bash
curl -X POST "https://api.example.com/v1/resources" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Resource",
    "description": "Description here"
  }'
```

### 7.2 Python Example

```python
import requests

API_KEY = "your-api-key"
BASE_URL = "https://api.example.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# List resources
response = requests.get(f"{BASE_URL}/resources", headers=headers)
data = response.json()

# Create resource
new_resource = {
    "name": "New Resource",
    "description": "Created via Python"
}
response = requests.post(
    f"{BASE_URL}/resources",
    headers=headers,
    json=new_resource
)
```

### 7.3 JavaScript Example

```javascript
const API_KEY = "your-api-key";
const BASE_URL = "https://api.example.com/v1";

// List resources
const listResources = async () => {
  const response = await fetch(`${BASE_URL}/resources`, {
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/json",
    },
  });
  return response.json();
};

// Create resource
const createResource = async (data) => {
  const response = await fetch(`${BASE_URL}/resources`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
  return response.json();
};
```

---

## 8. SDKs and Tools

### 8.1 Official SDKs

| Language   | Repository                            | Status |
| ---------- | ------------------------------------- | ------ |
| Python     | [github.com/example/sdk-python](link) | Stable |
| JavaScript | [github.com/example/sdk-js](link)     | Stable |
| Go         | [github.com/example/sdk-go](link)     | Beta   |

### 8.2 Postman Collection

Import our Postman collection: [Download Collection](link)

### 8.3 OpenAPI Specification

Download the OpenAPI spec: [openapi.yaml](link)

---

## 9. Changelog

### v1.0.0 (YYYY-MM-DD)

**Added:**

- Initial API release
- Resource CRUD operations
- Authentication via API keys

**Changed:**

- N/A

**Deprecated:**

- N/A

**Removed:**

- N/A

---

## Support

- **Documentation:** https://docs.example.com
- **API Status:** https://status.example.com
- **Support Email:** api-support@example.com
- **Community Forum:** https://community.example.com
