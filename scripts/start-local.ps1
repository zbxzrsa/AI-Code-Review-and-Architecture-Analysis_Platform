# Local Development Start Script (No Docker)
# AI 代码审查平台 - 本地开发启动脚本
# Usage: .\scripts\start-local.ps1

param(
    [switch]$Backend,
    [switch]$Frontend,
    [switch]$Both
)

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "AI Code Review Platform - Local Development" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Set environment
$env:MOCK_MODE = "true"
$env:ENVIRONMENT = "development"

function Start-Backend {
    Write-Host "`nStarting Backend API Server..." -ForegroundColor Yellow
    
    # Check Python
    if (!(Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "Error: Python is not installed" -ForegroundColor Red
        return
    }
    
    # Check if venv exists
    $venvPath = "backend\.venv"
    if (!(Test-Path $venvPath)) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv $venvPath
    }
    
    # Activate and install deps
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    & "$venvPath\Scripts\pip" install -q fastapi uvicorn pydantic python-dotenv
    
    # Start server
    Write-Host "Starting uvicorn server on port 8000..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; .\.venv\Scripts\activate; uvicorn app.main:app --reload --port 8000"
}

function Start-Frontend {
    Write-Host "`nStarting Frontend Dev Server..." -ForegroundColor Yellow
    
    # Check Node
    if (!(Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Host "Error: Node.js is not installed" -ForegroundColor Red
        return
    }
    
    # Check if node_modules exists
    if (!(Test-Path "frontend\node_modules")) {
        Write-Host "Installing npm dependencies..." -ForegroundColor Yellow
        Set-Location frontend
        npm install
        Set-Location ..
    }
    
    # Start dev server
    Write-Host "Starting Vite dev server on port 5173..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"
}

# Main logic
if ($Backend -or $Both -or (!$Backend -and !$Frontend)) {
    Start-Backend
}

if ($Frontend -or $Both -or (!$Backend -and !$Frontend)) {
    Start-Frontend
}

Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "Services Starting..." -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Backend API:  http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs:     http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Frontend:     http://localhost:5173" -ForegroundColor White
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "`nMode: MOCK_MODE=true (No external APIs needed)" -ForegroundColor Yellow
Write-Host "To stop: Close the terminal windows" -ForegroundColor Yellow
