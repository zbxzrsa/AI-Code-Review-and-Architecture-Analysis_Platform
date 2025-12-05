#!/bin/bash
# Development Start Script for Linux/Mac
# Usage: ./scripts/start-dev.sh

set -e

echo -e "\033[36mStarting AI Code Review Platform (Development Mode)\033[0m"
echo "================================================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "\033[31mError: Docker is not installed or not in PATH\033[0m"
    exit 1
fi

# Start infrastructure services
echo -e "\n\033[33mStarting infrastructure services...\033[0m"
docker-compose up -d postgres redis

# Wait for PostgreSQL to be ready
echo -e "\033[33mWaiting for PostgreSQL to be ready...\033[0m"
max_attempts=30
attempt=0
until docker exec platform-postgres pg_isready &> /dev/null || [[ $attempt -ge $max_attempts ]]; do
    attempt=$((attempt + 1))
    sleep 2
done

if [[ $attempt -lt $max_attempts ]]; then
    echo -e "\033[32mPostgreSQL is ready!\033[0m"
else
    echo -e "\033[33mWarning: PostgreSQL may not be ready yet\033[0m"
fi

# Start Ollama (if available)
echo -e "\n\033[33mStarting Ollama AI service...\033[0m"
if docker-compose up -d ollama 2>/dev/null; then
    echo -e "\033[32mOllama started. Pull a model with: docker exec platform-ollama ollama pull codellama:7b\033[0m"
else
    echo -e "\033[33mOllama service not available (optional)\033[0m"
fi

# Start remaining services
echo -e "\n\033[33mStarting application services...\033[0m"
docker-compose up -d

# Show status
echo -e "\n================================================="
echo -e "\033[36mServices Status:\033[0m"
docker-compose ps

echo -e "\n================================================="
echo -e "\033[32mAccess URLs:\033[0m"
echo -e "  Frontend:    http://localhost:3000"
echo -e "  API:         http://localhost:8000"
echo -e "  Grafana:     http://localhost:3002 (admin/admin)"
echo -e "  Prometheus:  http://localhost:9090"
echo "================================================="

echo -e "\n\033[33mTo stop all services: docker-compose down\033[0m"
echo -e "\033[33mTo view logs: docker-compose logs -f [service-name]\033[0m"
