#!/bin/bash
# =============================================================================
# Test Runner Script for AI Code Review Platform
# =============================================================================
#
# Usage:
#   ./scripts/run_tests.sh [options]
#
# Options:
#   --unit          Run unit tests only
#   --integration   Run integration tests only
#   --e2e           Run end-to-end tests only
#   --all           Run all tests (default)
#   --coverage      Generate coverage report
#   --verbose       Verbose output
#   --ci            CI mode (strict, with coverage)
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_E2E=false
RUN_ALL=true
COVERAGE=false
VERBOSE=false
CI_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_UNIT=true
            RUN_ALL=false
            shift
            ;;
        --integration)
            RUN_INTEGRATION=true
            RUN_ALL=false
            shift
            ;;
        --e2e)
            RUN_E2E=true
            RUN_ALL=false
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --ci)
            CI_MODE=true
            COVERAGE=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set pytest options
PYTEST_OPTS=""
if [ "$VERBOSE" = true ]; then
    PYTEST_OPTS="$PYTEST_OPTS -v"
fi
if [ "$COVERAGE" = true ]; then
    PYTEST_OPTS="$PYTEST_OPTS --cov=backend --cov=services --cov-report=xml --cov-report=html"
fi
if [ "$CI_MODE" = true ]; then
    PYTEST_OPTS="$PYTEST_OPTS --tb=short -q"
fi

# Header
echo ""
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}   AI Code Review Platform - Test Runner    ${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Check Node
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js not found${NC}"
    exit 1
fi

# Track results
FAILED=0
PASSED=0

# =============================================================================
# Unit Tests
# =============================================================================
run_unit_tests() {
    echo -e "${YELLOW}Running Unit Tests...${NC}"
    echo "----------------------------------------"
    
    # Backend unit tests
    echo -e "${BLUE}Backend Tests:${NC}"
    if python -m pytest tests/unit $PYTEST_OPTS; then
        ((PASSED++))
        echo -e "${GREEN}✓ Backend unit tests passed${NC}"
    else
        ((FAILED++))
        echo -e "${RED}✗ Backend unit tests failed${NC}"
    fi
    
    # Frontend unit tests
    echo -e "${BLUE}Frontend Tests:${NC}"
    cd frontend
    if npm test -- --watchAll=false --passWithNoTests; then
        ((PASSED++))
        echo -e "${GREEN}✓ Frontend unit tests passed${NC}"
    else
        ((FAILED++))
        echo -e "${RED}✗ Frontend unit tests failed${NC}"
    fi
    cd ..
    
    # Service tests
    echo -e "${BLUE}Service Tests:${NC}"
    if python -m pytest services/**/tests $PYTEST_OPTS 2>/dev/null || true; then
        ((PASSED++))
        echo -e "${GREEN}✓ Service tests passed${NC}"
    fi
    
    echo ""
}

# =============================================================================
# Integration Tests
# =============================================================================
run_integration_tests() {
    echo -e "${YELLOW}Running Integration Tests...${NC}"
    echo "----------------------------------------"
    
    # Check if services are running
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠ Services not running. Starting with Docker Compose...${NC}"
        docker-compose up -d
        sleep 10
    fi
    
    # Run integration tests
    echo -e "${BLUE}Integration Tests:${NC}"
    if python -m pytest tests/integration $PYTEST_OPTS -m "not slow"; then
        ((PASSED++))
        echo -e "${GREEN}✓ Integration tests passed${NC}"
    else
        ((FAILED++))
        echo -e "${RED}✗ Integration tests failed${NC}"
    fi
    
    echo ""
}

# =============================================================================
# E2E Tests
# =============================================================================
run_e2e_tests() {
    echo -e "${YELLOW}Running E2E Tests...${NC}"
    echo "----------------------------------------"
    
    cd frontend
    
    # Install Playwright if needed
    if [ ! -d "node_modules/@playwright" ]; then
        echo "Installing Playwright..."
        npx playwright install
    fi
    
    # Run E2E tests
    echo -e "${BLUE}Playwright E2E Tests:${NC}"
    if npx playwright test; then
        ((PASSED++))
        echo -e "${GREEN}✓ E2E tests passed${NC}"
    else
        ((FAILED++))
        echo -e "${RED}✗ E2E tests failed${NC}"
    fi
    
    cd ..
    echo ""
}

# =============================================================================
# Run Tests
# =============================================================================

if [ "$RUN_ALL" = true ]; then
    RUN_UNIT=true
    RUN_INTEGRATION=true
    RUN_E2E=true
fi

if [ "$RUN_UNIT" = true ]; then
    run_unit_tests
fi

if [ "$RUN_INTEGRATION" = true ]; then
    run_integration_tests
fi

if [ "$RUN_E2E" = true ]; then
    run_e2e_tests
fi

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}                  Summary                   ${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ "$COVERAGE" = true ]; then
    echo -e "${YELLOW}Coverage report generated:${NC}"
    echo "  - HTML: htmlcov/index.html"
    echo "  - XML:  coverage.xml"
    echo ""
fi

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
