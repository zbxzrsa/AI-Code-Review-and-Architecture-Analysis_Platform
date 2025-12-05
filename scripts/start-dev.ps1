# Development Start Script for Windows (PowerShell)
# Usage: .\scripts\start-dev.ps1

Write-Host "Starting AI Code Review Platform (Development Mode)" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# Check Docker
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Docker is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Start infrastructure services
Write-Host "`nStarting infrastructure services..." -ForegroundColor Yellow
docker-compose up -d postgres redis

# Wait for PostgreSQL to be ready
Write-Host "Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
do {
    $attempt++
    $result = docker exec platform-postgres pg_isready 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PostgreSQL is ready!" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 2
} while ($attempt -lt $maxAttempts)

if ($attempt -ge $maxAttempts) {
    Write-Host "Warning: PostgreSQL may not be ready yet" -ForegroundColor Yellow
}

# Start Ollama (if available)
Write-Host "`nStarting Ollama AI service..." -ForegroundColor Yellow
docker-compose up -d ollama 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Ollama started. Pull a model with: docker exec platform-ollama ollama pull codellama:7b" -ForegroundColor Green
} else {
    Write-Host "Ollama service not available (optional)" -ForegroundColor Yellow
}

# Start remaining services
Write-Host "`nStarting application services..." -ForegroundColor Yellow
docker-compose up -d

# Show status
Write-Host "`n=================================================" -ForegroundColor Cyan
Write-Host "Services Status:" -ForegroundColor Cyan
docker-compose ps

Write-Host "`n=================================================" -ForegroundColor Cyan
Write-Host "Access URLs:" -ForegroundColor Green
Write-Host "  Frontend:    http://localhost:3000" -ForegroundColor White
Write-Host "  API:         http://localhost:8000" -ForegroundColor White
Write-Host "  Grafana:     http://localhost:3002 (admin/admin)" -ForegroundColor White
Write-Host "  Prometheus:  http://localhost:9090" -ForegroundColor White
Write-Host "=================================================" -ForegroundColor Cyan

Write-Host "`nTo stop all services: docker-compose down" -ForegroundColor Yellow
Write-Host "To view logs: docker-compose logs -f [service-name]" -ForegroundColor Yellow

Write-Host "`n=================================================" -ForegroundColor Cyan
Write-Host "Quick Start Commands:" -ForegroundColor Green
Write-Host "  Start modular backend:  cd backend && uvicorn app.main:app --reload --port 8000" -ForegroundColor White
Write-Host "  Start frontend dev:     cd frontend && npm run dev" -ForegroundColor White
Write-Host "  Run backend tests:      cd backend && pytest app/tests/ -v" -ForegroundColor White
Write-Host "  Run frontend tests:     cd frontend && npm test" -ForegroundColor White
Write-Host "=================================================" -ForegroundColor Cyan
