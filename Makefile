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
	docker build -t $(REGISTRY)/frontend:$(VERSION) -f frontend/Dockerfile .

push-images:
	docker push $(REGISTRY)/vcai:$(VERSION)
	docker push $(REGISTRY)/crai:$(VERSION)
	docker push $(REGISTRY)/lifecycle-controller:$(VERSION)
	docker push $(REGISTRY)/evaluation-pipeline:$(VERSION)
	docker push $(REGISTRY)/frontend:$(VERSION)

sign-images:
	cosign sign --yes $(REGISTRY)/vcai:$(VERSION)
	cosign sign --yes $(REGISTRY)/crai:$(VERSION)
	cosign sign --yes $(REGISTRY)/lifecycle-controller:$(VERSION)
	cosign sign --yes $(REGISTRY)/evaluation-pipeline:$(VERSION)
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
# Documentation
# ============================================================

docs:
	@echo "Opening documentation..."
	@open docs/three-version-architecture.md 2>/dev/null || xdg-open docs/three-version-architecture.md 2>/dev/null || echo "Open docs/three-version-architecture.md manually"

docs-serve:
	cd docs && python -m http.server 8000
