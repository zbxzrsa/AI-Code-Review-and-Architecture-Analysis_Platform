# AI Code Review Platform - Makefile
# Common operations for development, testing, and deployment

.PHONY: help dev prod test lint build deploy clean

# Default target
help:
	@echo "AI Code Review Platform - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  make dev           - Start development environment"
	@echo "  make dev-frontend  - Start frontend dev server"
	@echo "  make dev-backend   - Start backend services"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests"
	@echo "  make test-unit     - Run unit tests"
	@echo "  make test-e2e      - Run end-to-end tests"
	@echo "  make lint          - Run linters"
	@echo ""
	@echo "Build:"
	@echo "  make build         - Build all services"
	@echo "  make build-frontend - Build frontend"
	@echo "  make build-images  - Build Docker images"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-v1     - Deploy to V1 experiment"
	@echo "  make deploy-v2     - Deploy to V2 production"
	@echo "  make deploy-offline - Deploy offline mode"
	@echo "  make rollback      - Rollback V2 to previous version"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-apply     - Apply Kubernetes manifests"
	@echo "  make k8s-status    - Show deployment status"
	@echo "  make k8s-logs      - Show service logs"
	@echo ""
	@echo "Three-Version Evolution:"
	@echo "  make verify-three-version     - Verify three-version implementation"
	@echo "  make verify-three-version-api - Verify with API tests (requires running service)"
	@echo "  make test-three-version       - Run three-version unit tests"
	@echo "  make start-three-version      - Start three-version service"
	@echo "  make stop-three-version       - Stop three-version service"
	@echo "  make logs-three-version       - View three-version service logs"
	@echo ""
	@echo "Networked Learning:"
	@echo "  make verify-networked-learning - Verify all networked learning modules"
	@echo "  make test-networked-learning   - Run all networked learning tests"
	@echo "  make test-learning-system      - Test V1/V3 learning system"
	@echo "  make test-cleansing-pipeline   - Test data cleansing pipeline"
	@echo "  make test-lifecycle-manager    - Test data lifecycle management"
	@echo "  make test-tech-elimination     - Test technology elimination"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make seal-secrets  - Seal secrets with kubeseal"
	@echo "  make gold-set-eval - Run gold-set evaluation"

# ============================================================
# Variables
# ============================================================

VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
REGISTRY ?= gcr.io/coderev-platform
NAMESPACE_V1 ?= platform-v1-exp
NAMESPACE_V2 ?= platform-v2-stable
NAMESPACE_V3 ?= platform-v3-legacy

# ============================================================
# Development
# ============================================================

dev:
	docker-compose up -d

dev-frontend:
	cd frontend && npm run dev

dev-backend:
	docker-compose up -d postgres redis
	cd backend && uvicorn main:app --reload --port 8000

dev-stop:
	docker-compose down

dev-logs:
	docker-compose logs -f

# ============================================================
# Testing
# ============================================================

test: test-unit test-integration

test-unit:
	cd backend && pytest tests/unit -v
	cd frontend && npm test

test-integration:
	cd backend && pytest tests/integration -v

test-e2e:
	cd frontend && npm run test:e2e

test-security:
	cd backend && bandit -r . -ll
	cd frontend && npm audit

test-gold-set:
	python scripts/statistical_tests.py \
		--version $(VERSION) \
		--baseline v2-current \
		--output results/gold-set-$(VERSION).json

lint:
	cd backend && ruff check .
	cd backend && black --check .
	cd frontend && npm run lint

lint-fix:
	cd backend && ruff check --fix .
	cd backend && black .
	cd frontend && npm run lint:fix

# ============================================================
# Verification
# ============================================================

verify-three-version:
	python scripts/verify_three_version.py

verify-three-version-api:
	python scripts/verify_three_version.py --api-test

test-three-version:
	cd tests && pytest backend/test_three_version_cycle.py -v

start-three-version:
	docker-compose up -d three-version-service

stop-three-version:
	docker-compose stop three-version-service

logs-three-version:
	docker-compose logs -f three-version-service

# ============================================================
# Networked Learning Tests
# ============================================================

verify-networked-learning:
	python scripts/verify_networked_learning.py

test-networked-learning:
	python -m pytest tests/unit/test_auto_network_learning.py -v

test-learning-system:
	python -m pytest tests/unit/test_auto_network_learning.py::TestV1V3AutoLearningSystem -v

test-cleansing-pipeline:
	python -m pytest tests/unit/test_auto_network_learning.py::TestDataCleansingPipeline -v

test-lifecycle-manager:
	python -m pytest tests/unit/test_auto_network_learning.py::TestDataLifecycleManager -v

test-tech-elimination:
	python -m pytest tests/unit/test_auto_network_learning.py::TestTechEliminationManager -v

test-learning-integration:
	python -m pytest tests/unit/test_auto_network_learning.py::TestIntegration -v

test-self-evolution:
	python -m pytest tests/integration/test_self_evolution_cycle.py -v

# ============================================================
# Build
# ============================================================

build: build-frontend build-backend

build-frontend:
	cd frontend && npm ci && npm run build

build-backend:
	cd backend && pip install -r requirements.txt

build-images:
	docker build -t $(REGISTRY)/vcai:$(VERSION) -f backend/Dockerfile.vcai .
	docker build -t $(REGISTRY)/crai:$(VERSION) -f backend/Dockerfile.crai .
	docker build -t $(REGISTRY)/lifecycle-controller:$(VERSION) -f services/lifecycle-controller/Dockerfile .
	docker build -t $(REGISTRY)/evaluation-pipeline:$(VERSION) -f services/evaluation-pipeline/Dockerfile .
	docker build -t $(REGISTRY)/three-version-service:$(VERSION) -f backend/services/three-version-service/Dockerfile backend/services/three-version-service
	docker build -t $(REGISTRY)/frontend:$(VERSION) -f frontend/Dockerfile .

push-images:
	docker push $(REGISTRY)/vcai:$(VERSION)
	docker push $(REGISTRY)/crai:$(VERSION)
	docker push $(REGISTRY)/lifecycle-controller:$(VERSION)
	docker push $(REGISTRY)/evaluation-pipeline:$(VERSION)
	docker push $(REGISTRY)/three-version-service:$(VERSION)
	docker push $(REGISTRY)/frontend:$(VERSION)

sign-images:
	cosign sign --yes $(REGISTRY)/vcai:$(VERSION)
	cosign sign --yes $(REGISTRY)/crai:$(VERSION)
	cosign sign --yes $(REGISTRY)/lifecycle-controller:$(VERSION)
	cosign sign --yes $(REGISTRY)/evaluation-pipeline:$(VERSION)
	cosign sign --yes $(REGISTRY)/three-version-service:$(VERSION)
	cosign sign --yes $(REGISTRY)/frontend:$(VERSION)

# ============================================================
# Kubernetes Deployment
# ============================================================

k8s-apply:
	kubectl apply -k kubernetes/base/
	kubectl apply -k kubernetes/overlays/v1-exp/
	kubectl apply -k kubernetes/overlays/v2-stable/
	kubectl apply -k kubernetes/overlays/v3-legacy/

k8s-apply-v1:
	kubectl apply -k kubernetes/overlays/v1-exp/

k8s-apply-v2:
	kubectl apply -k kubernetes/overlays/v2-stable/

k8s-apply-offline:
	kubectl apply -k kubernetes/overlays/offline/

k8s-status:
	@echo "=== V1 Experiment ==="
	kubectl get pods -n $(NAMESPACE_V1) -o wide
	@echo ""
	@echo "=== V2 Stable ==="
	kubectl get pods -n $(NAMESPACE_V2) -o wide
	kubectl argo rollouts get rollout vcai-rollout -n $(NAMESPACE_V2) 2>/dev/null || true
	@echo ""
	@echo "=== V3 Legacy ==="
	kubectl get pods -n $(NAMESPACE_V3) -o wide

k8s-logs:
	kubectl logs -n $(NAMESPACE_V2) -l app=vcai --tail=100 -f

k8s-logs-v1:
	kubectl logs -n $(NAMESPACE_V1) -l app=vcai --tail=100 -f

# ============================================================
# Deployment Operations
# ============================================================

deploy-v1:
	@echo "Deploying $(VERSION) to V1 Experiment..."
	kubectl set image deployment/vcai-service vcai=$(REGISTRY)/vcai:$(VERSION) -n $(NAMESPACE_V1)
	kubectl set image deployment/crai-service crai=$(REGISTRY)/crai:$(VERSION) -n $(NAMESPACE_V1)
	kubectl rollout status deployment/vcai-service -n $(NAMESPACE_V1) --timeout=300s

deploy-v2:
	@echo "Deploying $(VERSION) to V2 Production via Argo Rollouts..."
	kubectl argo rollouts set image vcai-rollout vcai=$(REGISTRY)/vcai:$(VERSION) -n $(NAMESPACE_V2)
	@echo "Rollout started. Monitor with: make rollout-status"

deploy-offline:
	@echo "Deploying offline mode..."
	kubectl apply -k kubernetes/overlays/offline/
	kubectl rollout status deployment/vcai-service -n platform-offline --timeout=600s

rollout-status:
	kubectl argo rollouts get rollout vcai-rollout -n $(NAMESPACE_V2) -w

rollout-promote:
	kubectl argo rollouts promote vcai-rollout -n $(NAMESPACE_V2)

rollout-abort:
	kubectl argo rollouts abort vcai-rollout -n $(NAMESPACE_V2)

rollback:
	@echo "Rolling back V2 to previous version..."
	kubectl argo rollouts undo vcai-rollout -n $(NAMESPACE_V2)
	kubectl argo rollouts get rollout vcai-rollout -n $(NAMESPACE_V2) -w

# ============================================================
# Evaluation & Monitoring
# ============================================================

gold-set-eval:
	@echo "Running gold-set evaluation for $(VERSION)..."
	curl -X POST http://localhost:8080/evaluate/gold-set \
		-H "Content-Type: application/json" \
		-d '{"version_id": "$(VERSION)", "model_version": "gpt-4o", "prompt_version": "code-review-v3"}'

check-health:
	./scripts/check_rollout_health.sh vcai-rollout $(NAMESPACE_V2)

stats:
	@echo "=== Prometheus Metrics ==="
	curl -s "http://localhost:9090/api/v1/query?query=slo:v2:error_rate:ratio_rate5m" | jq '.data.result[0].value[1]'
	@echo ""
	@echo "=== Comparison Stats ==="
	curl -s http://localhost:8080/stats/comparison | jq

# ============================================================
# Secrets Management
# ============================================================

seal-secrets:
	@echo "Sealing secrets for V1..."
	kubeseal --format yaml < kubernetes/overlays/v1-exp/secrets.yaml > kubernetes/overlays/v1-exp/sealed-secrets.yaml
	@echo "Sealing secrets for V2..."
	kubeseal --format yaml < kubernetes/overlays/v2-stable/secrets.yaml > kubernetes/overlays/v2-stable/sealed-secrets.yaml
	@echo "Done. Commit the sealed-secrets.yaml files."

# ============================================================
# Policy Validation
# ============================================================

validate-policies:
	conftest test kubernetes/ --policy policies/kubernetes/

validate-opa:
	opa check services/lifecycle-controller/policies/

# ============================================================
# Cleanup
# ============================================================

clean:
	rm -rf frontend/build frontend/dist
	rm -rf backend/__pycache__ backend/.pytest_cache
	rm -rf results/*.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-docker:
	docker system prune -f
	docker image prune -f

clean-k8s-v1:
	kubectl delete -k kubernetes/overlays/v1-exp/ --ignore-not-found

# ============================================================
# Helm Operations
# ============================================================

HELM_RELEASE ?= coderev
HELM_CHART ?= ./charts/coderev-platform
HELM_NAMESPACE ?= coderev

helm-deps:
	helm repo add bitnami https://charts.bitnami.com/bitnami
	helm repo add argo https://argoproj.github.io/argo-helm
	helm repo update

helm-lint:
	helm lint $(HELM_CHART)

helm-template:
	helm template $(HELM_RELEASE) $(HELM_CHART) --debug

helm-install:
	helm install $(HELM_RELEASE) $(HELM_CHART) \
		--namespace $(HELM_NAMESPACE) --create-namespace

helm-install-dev:
	helm install $(HELM_RELEASE)-dev $(HELM_CHART) \
		-f $(HELM_CHART)/values-development.yaml \
		--namespace $(HELM_NAMESPACE)-dev --create-namespace

helm-install-prod:
	helm install $(HELM_RELEASE) $(HELM_CHART) \
		-f $(HELM_CHART)/values-production.yaml \
		--namespace $(HELM_NAMESPACE) --create-namespace

helm-install-hipaa:
	helm install $(HELM_RELEASE) $(HELM_CHART) \
		-f $(HELM_CHART)/values-hipaa.yaml \
		--namespace $(HELM_NAMESPACE)-hipaa --create-namespace

helm-install-offline:
	helm install $(HELM_RELEASE) $(HELM_CHART) \
		-f $(HELM_CHART)/values-offline.yaml \
		--namespace $(HELM_NAMESPACE)-offline --create-namespace

helm-upgrade:
	helm upgrade $(HELM_RELEASE) $(HELM_CHART) \
		--namespace $(HELM_NAMESPACE)

helm-uninstall:
	helm uninstall $(HELM_RELEASE) --namespace $(HELM_NAMESPACE)

helm-status:
	helm status $(HELM_RELEASE) --namespace $(HELM_NAMESPACE)

helm-history:
	helm history $(HELM_RELEASE) --namespace $(HELM_NAMESPACE)

helm-rollback:
	helm rollback $(HELM_RELEASE) --namespace $(HELM_NAMESPACE)

helm-package:
	helm package $(HELM_CHART) -d ./dist

# ============================================================
# Load Testing
# ============================================================

load-test:
	python scripts/load_test.py --url http://localhost --requests 100 --concurrency 10

load-test-heavy:
	python scripts/load_test.py --url http://localhost --requests 1000 --concurrency 50

load-test-duration:
	python scripts/load_test.py --url http://localhost --duration 60 --concurrency 20

# ============================================================
# Health & Validation
# ============================================================

health-check:
	python scripts/health_check.py --verbose

api-test:
	python scripts/api_test.py --base-url http://localhost

validate-system:
	pytest tests/integration/test_system_validation.py -v

# ============================================================
# Quick Start (New Optimized Commands)
# ============================================================

# Validate environment setup
validate-env:
	python scripts/validate_env.py

# Start everything for local development (demo mode)
start-demo:
	@echo "Starting AI Code Review Platform (Demo Mode)..."
	@cp -n .env.example .env 2>/dev/null || true
	docker compose up -d
	@echo "Waiting for services to start..."
	@sleep 5
	@echo ""
	@echo "Starting backend API server..."
	cd backend && python dev-api-server.py &
	@sleep 3
	@echo ""
	@echo "==================================================="
	@echo "Platform is ready!"
	@echo "==================================================="
	@echo "Frontend: Run 'cd frontend && npm run dev' in another terminal"
	@echo "API Docs: http://localhost:8000/docs"
	@echo "==================================================="

# Start full development environment
start-full:
	@echo "Starting full development environment..."
	docker compose up -d
	cd backend && python dev-api-server.py &
	cd frontend && npm run dev

# Stop all services
stop-all:
	docker compose down
	@pkill -f "dev-api-server.py" 2>/dev/null || true
	@echo "All services stopped"

# Quick API health test
quick-test:
	@echo "Testing API endpoints..."
	@curl -s http://localhost:8000/health | python -m json.tool
	@echo ""
	@curl -s http://localhost:8000/api/projects | python -m json.tool | head -20
	@echo ""
	@echo "API is working!"

# View API docs
api-docs:
	@echo "Opening API documentation..."
	@python -m webbrowser "http://localhost:8000/docs" 2>/dev/null || echo "Open http://localhost:8000/docs in your browser"

# Seed demo data
seed-demo:
	@echo "Seeding demo data..."
	curl -X POST http://localhost:8000/api/seed/demo
	@echo "Demo data seeded!"

# ============================================================
# Documentation
# ============================================================

docs:
	@echo "Opening documentation..."
	@open docs/three-version-architecture.md 2>/dev/null || xdg-open docs/three-version-architecture.md 2>/dev/null || echo "Open docs/three-version-architecture.md manually"

docs-serve:
	cd docs && python -m http.server 8000

# ============================================================
# Quick Reference
# ============================================================

.PHONY: quick-help
quick-help:
	@echo ""
	@echo "Quick Start Commands:"
	@echo "  make validate-env   - Check environment setup"
	@echo "  make start-demo     - Start platform (demo mode)"
	@echo "  make stop-all       - Stop all services"
	@echo "  make quick-test     - Test API endpoints"
	@echo "  make api-docs       - Open API documentation"
	@echo ""
	@echo "URLs:"
	@echo "  Frontend:    http://localhost:5173"
	@echo "  API:         http://localhost:8000"
	@echo "  API Docs:    http://localhost:8000/docs"
	@echo "  Grafana:     http://localhost:3002"
	@echo ""

# ============================================================
# Project Optimization
# ============================================================

.PHONY: optimize optimize-report optimize-deps optimize-build clean-cache

# Run full project optimization analysis
optimize:
	@echo "Running project optimization analysis..."
	python scripts/optimize_project.py --report --path .
	@echo "Report saved to OPTIMIZATION_REPORT.md"

# Generate optimization report only (no changes)
optimize-report:
	@echo "Generating optimization report..."
	python scripts/optimize_project.py --dry-run --report --path .

# Clean up Python cache and build artifacts
clean-cache:
	@echo "Cleaning Python cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaning frontend cache..."
	rm -rf frontend/node_modules/.cache 2>/dev/null || true
	rm -rf frontend/dist 2>/dev/null || true
	@echo "Cache cleaned!"

# Optimize Python dependencies
optimize-deps:
	@echo "Analyzing Python dependencies..."
	pip install pipdeptree 2>/dev/null || true
	pipdeptree --warn silence | grep -E "^\w+" | sort | uniq > deps-analysis.txt
	@echo "Dependency analysis saved to deps-analysis.txt"

# Optimize frontend build
optimize-build:
	@echo "Building optimized frontend..."
	cd frontend && npm run build -- --mode production
	@echo "Analyzing bundle size..."
	du -sh frontend/dist/*
	@echo "Build complete!"

# Clean all artifacts
clean-all: clean clean-cache
	@echo "Removing additional artifacts..."
	rm -rf .coverage htmlcov 2>/dev/null || true
	rm -rf *.log 2>/dev/null || true
	@echo "All artifacts cleaned!"

# Show project size stats
size-stats:
	@echo "Project size analysis:"
	@echo ""
	@echo "Total size (excluding node_modules, .git):"
	@du -sh --exclude=node_modules --exclude=.git . 2>/dev/null || du -sh . 2>/dev/null
	@echo ""
	@echo "By directory:"
	@du -sh */ 2>/dev/null | sort -hr | head -15
	@echo ""
	@echo "Largest files:"
	@find . -type f -not -path "./node_modules/*" -not -path "./.git/*" -exec du -h {} + 2>/dev/null | sort -hr | head -10
