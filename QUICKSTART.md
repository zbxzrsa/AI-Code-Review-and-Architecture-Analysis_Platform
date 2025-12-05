# 🚀 Quick Start Guide

Get the AI Code Review Platform running in 5 minutes!

## ✅ Prerequisites

| Requirement    | Version | Notes                               |
| -------------- | ------- | ----------------------------------- |
| Docker         | 20.10+  | `docker --version`                  |
| Docker Compose | 2.0+    | `docker compose version`            |
| Node.js        | 20+     | For frontend dev: `node --version`  |
| Python         | 3.10+   | For backend dev: `python --version` |
| RAM            | 8GB+    | Minimum for all services            |
| Disk           | 10GB+   | For Docker images                   |

---

## 🎯 Option 1: Quick Demo (No API Keys Required)

This runs the platform in **mock mode** - no real AI providers needed.

### Step 1: Setup Environment

```bash
# Clone and enter project
cd AI-Code-Review-and-Architecture-Analysis_Platform

# Copy environment file (mock mode enabled by default)
cp .env.example .env
```

### Step 2: Start Services

```bash
# Start all Docker services
docker compose up -d

# Wait for services to be healthy (~60 seconds)
docker compose ps
```

### Step 3: Start Development Servers

```bash
# Terminal 1: Start backend API server
cd backend
python dev-api-server.py

# Terminal 2: Start frontend
cd frontend
npm install
npm run dev
```

### Step 4: Access the Platform

| Service      | URL                        | Credentials                |
| ------------ | -------------------------- | -------------------------- |
| **Frontend** | http://localhost:5173      | demo@example.com / demo123 |
| **API Docs** | http://localhost:8000/docs | -                          |
| **Grafana**  | http://localhost:3002      | admin / admin              |

---

## 🔑 Option 2: Full Mode (With AI Providers)

For real AI-powered code review functionality.

### Step 1: Get API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **HuggingFace** (optional): https://huggingface.co/settings/tokens

### Step 2: Configure Environment

```bash
cp .env.example .env

# Edit .env file:
# MOCK_MODE=false
# OPENAI_API_KEY=sk-your-real-key
# ANTHROPIC_API_KEY=sk-ant-your-real-key
```

### Step 3: Start Services

Same as Option 1, Steps 2-4.

---

## 🔧 Port Reference

| Service         | Port | Description              |
| --------------- | ---- | ------------------------ |
| Frontend (Vite) | 5173 | React development server |
| Backend API     | 8000 | FastAPI with auto-reload |
| PostgreSQL      | 5432 | Database                 |
| Redis           | 6379 | Cache & sessions         |
| Neo4j           | 7687 | Graph database           |
| MinIO           | 9000 | Object storage           |
| Grafana         | 3002 | Monitoring dashboards    |
| Prometheus      | 9090 | Metrics collection       |

---

## ✅ Verify Installation

### Health Checks

```bash
# Backend API
curl http://localhost:8000/health
# Expected: {"status": "healthy"}

# Projects API
curl http://localhost:8000/api/projects
# Expected: {"items": [...], "total": 3, ...}

# Docker services
docker compose ps
# All services should show "Up (healthy)"
```

### Run Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

---

## 🔥 Quick Commands

```bash
# Start everything
docker compose up -d && cd backend && python dev-api-server.py &

# Stop everything
docker compose down

# View logs
docker compose logs -f

# Restart a service
docker compose restart platform-postgres

# Clean restart
docker compose down -v && docker compose up -d
```

---

## ❓ Troubleshooting

### Port Already in Use

```bash
# Find and kill process on port 8000
lsof -i :8000
kill -9 <PID>

# Or change port in .env
API_PORT=8001
```

### Database Connection Failed

```bash
# Wait for PostgreSQL to be ready
docker compose logs platform-postgres

# Restart database
docker compose restart platform-postgres
```

### Frontend Can't Connect to API

1. Check backend is running: `curl http://localhost:8000/health`
2. Check Vite proxy config in `frontend/vite.config.ts`
3. Ensure CORS is enabled in backend

### Docker Out of Space

```bash
# Clean up Docker
docker system prune -a
docker volume prune
```

---

## 📚 Next Steps

1. **Explore the UI**: Navigate through Dashboard, Projects, Code Review
2. **Create a Project**: Add your first repository
3. **Run Analysis**: Submit code for AI review
4. **Check Docs**: Read `docs/` folder for detailed guides

## 🆘 Getting Help

- **Issues**: GitHub Issues
- **Docs**: `/docs` folder
- **API Reference**: http://localhost:8000/docs
