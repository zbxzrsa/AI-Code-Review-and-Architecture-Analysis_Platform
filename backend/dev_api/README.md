# Dev API Server - Modular Architecture

> **P0 Optimization**: File splitting and reorganization completed  
> **Original**: `dev-api-server.py` (4,492 lines)  
> **Result**: Modular structure with 20+ files, each < 500 lines

## Directory Structure

```
dev_api/
├── __init__.py          # Package initialization
├── app.py               # FastAPI application factory (132 lines)
├── config.py            # Configuration and constants (100 lines)
├── models.py            # Pydantic models (150 lines)
├── mock_data.py         # Mock data for development (300 lines)
├── middleware.py        # Custom middleware (50 lines)
├── README.md            # This file
│
├── core/                # Core infrastructure
│   ├── __init__.py      # Core exports
│   ├── config.py        # Pydantic Settings (150 lines)
│   ├── dependencies.py  # Dependency injection (200 lines)
│   └── middleware.py    # Middleware implementations (200 lines)
│
├── routes/              # API route modules
│   ├── __init__.py      # Route exports
│   ├── admin.py         # Admin endpoints (200 lines)
│   ├── analysis.py      # Code analysis (150 lines)
│   ├── auth.py          # Authentication (350 lines)
│   ├── dashboard.py     # Dashboard metrics (50 lines)
│   ├── oauth.py         # OAuth integration (200 lines)
│   ├── projects.py      # Project management (200 lines)
│   ├── reports.py       # Reports and backups (150 lines)
│   ├── security.py      # Security endpoints (180 lines)
│   ├── three_version.py # Three-version evolution (150 lines)
│   ├── users.py         # User management (120 lines)
│   └── vulnerabilities.py # Vulnerability scanning (350 lines)
│
└── services/            # Business logic services
    ├── __init__.py      # Service exports
    ├── code_review_service.py    # Code review logic (250 lines)
    ├── vulnerability_service.py  # Vulnerability handling (200 lines)
    └── analytics_service.py      # Analytics logic (180 lines)
```

## Quality Metrics

| Metric             | Requirement | Actual            |
| ------------------ | ----------- | ----------------- |
| Max lines per file | < 2,000     | ✅ All < 400      |
| Module count       | -           | 20 files          |
| Test coverage      | > 80%       | 🔄 In progress    |
| API docs           | Complete    | ✅ Auto-generated |

## Module Boundaries

### Routes (API Layer)

- Handle HTTP requests/responses
- Input validation
- Response formatting
- No business logic

### Services (Business Layer)

- Business logic implementation
- Data processing
- External service integration
- Reusable across routes

### Core (Infrastructure Layer)

- Configuration management
- Dependency injection
- Middleware
- Cross-cutting concerns

## API Endpoints

### Authentication (`/api/auth`)

| Endpoint           | Method | Description       |
| ------------------ | ------ | ----------------- |
| `/login`           | POST   | User login        |
| `/register`        | POST   | User registration |
| `/logout`          | POST   | User logout       |
| `/refresh`         | POST   | Refresh token     |
| `/me`              | GET    | Current user      |
| `/password/reset`  | POST   | Password reset    |
| `/password/change` | POST   | Change password   |
| `/sessions`        | GET    | List sessions     |

### Vulnerabilities (`/api/vulnerabilities`)

| Endpoint            | Method | Description          |
| ------------------- | ------ | -------------------- |
| `/`                 | GET    | List vulnerabilities |
| `/{id}`             | GET    | Get vulnerability    |
| `/{id}/status`      | PATCH  | Update status        |
| `/stats`            | GET    | Statistics           |
| `/scan`             | POST   | Trigger scan         |
| `/{id}/fixes`       | GET    | Get auto-fixes       |
| `/fixes/{id}/apply` | POST   | Apply fix            |

### Projects (`/api/projects`)

| Endpoint         | Method         | Description |
| ---------------- | -------------- | ----------- |
| `/`              | GET/POST       | List/Create |
| `/{id}`          | GET/PUT/DELETE | CRUD        |
| `/{id}/settings` | GET/PUT        | Settings    |
| `/{id}/members`  | GET/POST       | Members     |

### Analysis (`/api/analysis`)

| Endpoint        | Method | Description  |
| --------------- | ------ | ------------ |
| `/code`         | POST   | Analyze code |
| `/{id}/results` | GET    | Get results  |
| `/history`      | GET    | History      |

## Usage

### Running the Server

```bash
# From backend directory
python dev-api-server.py

# Or with uvicorn
uvicorn dev_api:app --reload --host 0.0.0.0 --port 8000
```

### Importing Modules

```python
# Import the app
from dev_api import app

# Use services
from dev_api.services import CodeReviewService, VulnerabilityService
review_service = CodeReviewService()
result = await review_service.analyze_code(code, "python")

# Use dependencies
from dev_api.core import get_current_user, require_admin
```

### Adding New Routes

1. Create new file in `routes/`:

```python
# routes/my_feature.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/my-feature", tags=["My Feature"])

@router.get("/")
async def list_items():
    return {"items": []}
```

2. Register in `routes/__init__.py`:

```python
from .my_feature import router as my_feature_router
```

3. Add to `app.py`:

```python
application.include_router(my_feature_router)
```

## Migration Notes

### From Old Structure

The original `dev-api-server.py` was split into:

| Old Section             | New Location                |
| ----------------------- | --------------------------- |
| Auth endpoints          | `routes/auth.py`            |
| Project endpoints       | `routes/projects.py`        |
| Analysis endpoints      | `routes/analysis.py`        |
| Vulnerability endpoints | `routes/vulnerabilities.py` |
| Admin endpoints         | `routes/admin.py`           |
| Configuration           | `core/config.py`            |
| Dependencies            | `core/dependencies.py`      |
| Business logic          | `services/`                 |

### Backward Compatibility

- Entry point remains `dev-api-server.py`
- All API endpoints unchanged
- Import paths maintained via re-exports

## Version History

| Version | Date       | Changes                                     |
| ------- | ---------- | ------------------------------------------- |
| 2.1.0   | 2024-12-07 | Added auth, vulnerabilities, services, core |
| 2.0.0   | 2024-12-06 | Initial modular split                       |
| 1.0.0   | -          | Original monolithic file                    |
