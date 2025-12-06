#!/usr/bin/env python3
"""
Three-Version Evolution System Verification Script

Verifies the implementation of the three-version self-evolution system:
- File structure validation
- Import verification
- API endpoint testing
- Component integration checks

Usage:
    python scripts/verify_three_version.py [--api-test]
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Constants for repeated messages
MSG_FILE_NOT_FOUND = "File not found"
MSG_CLASS_IMPORT = "Class import"
MSG_MODULE_EXISTS = "Module exists"


def print_header(title: str):
    """Print section header."""
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{title}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")


def print_result(name: str, success: bool, message: str = ""):
    """Print test result."""
    status = f"{GREEN}✓ PASS{RESET}" if success else f"{RED}✗ FAIL{RESET}"
    print(f"  {status} {name}")
    if message and not success:
        print(f"       {YELLOW}{message}{RESET}")


def get_project_root() -> Path:
    """Get project root directory."""
    script_path = Path(__file__).resolve()
    return script_path.parent.parent


def check_file_contains(
    file_path: Path,
    patterns: List[str],
    name: str,
    any_pattern: bool = False
) -> Tuple[bool, int, int]:
    """
    Check if file exists and contains required patterns.
    
    Args:
        file_path: Path to file
        patterns: List of patterns to search for
        name: Display name for result
        any_pattern: If True, any pattern match is success; if False, all must match
    
    Returns:
        Tuple of (exists, passed_count, failed_count)
    """
    if not file_path.exists():
        print_result(f"{name} exists", False, MSG_FILE_NOT_FOUND)
        return False, 0, 1
    
    content = file_path.read_text(encoding='utf-8')
    
    if any_pattern:
        has_pattern = any(p in content for p in patterns)
    else:
        has_pattern = all(p in content for p in patterns)
    
    print_result(name, has_pattern)
    return True, (1 if has_pattern else 0), (0 if has_pattern else 1)


# =============================================================================
# File Structure Verification
# =============================================================================

def verify_file_structure() -> Tuple[int, int]:
    """Verify all required files exist."""
    print_header("File Structure Verification")
    
    root = get_project_root()
    
    required_files = [
        # AI Core
        "ai_core/three_version_cycle/__init__.py",
        "ai_core/three_version_cycle/cross_version_feedback.py",
        "ai_core/three_version_cycle/v3_comparison_engine.py",
        "ai_core/three_version_cycle/dual_ai_coordinator.py",
        "ai_core/three_version_cycle/spiral_evolution_manager.py",
        "ai_core/three_version_cycle/version_manager.py",
        "ai_core/three_version_cycle/version_ai_engine.py",
        "ai_core/three_version_cycle/experiment_framework.py",
        "ai_core/three_version_cycle/self_evolution_cycle.py",
        
        # Backend Service
        "backend/services/three-version-service/api.py",
        "backend/services/three-version-service/main.py",
        "backend/services/three-version-service/metrics.py",
        "backend/services/three-version-service/requirements.txt",
        "backend/services/three-version-service/Dockerfile",
        "backend/services/three-version-service/README.md",
        
        # Frontend
        "frontend/src/pages/admin/ThreeVersionControl.tsx",
        "frontend/src/services/threeVersionService.ts",
        
        # Monitoring
        "monitoring/grafana/provisioning/dashboards/three-version-evolution.json",
        "monitoring/prometheus/rules/three-version-alerts.yml",
        
        # Kubernetes
        "kubernetes/deployments/three-version-service.yaml",
        
        # Helm
        "charts/coderev-platform/templates/three-version-service.yaml",
        
        # Documentation
        "docs/three-version-evolution.md",
        
        # Tests
        "tests/backend/test_three_version_cycle.py",
    ]
    
    passed = 0
    failed = 0
    
    for file_path in required_files:
        full_path = root / file_path
        exists = full_path.exists()
        print_result(file_path, exists, MSG_FILE_NOT_FOUND if not exists else "")
        if exists:
            passed += 1
        else:
            failed += 1
    
    return passed, failed


# =============================================================================
# Import Verification
# =============================================================================

def verify_imports() -> Tuple[int, int]:
    """Verify Python imports work correctly."""
    print_header("Import Verification")
    
    root = get_project_root()
    sys.path.insert(0, str(root))
    
    imports_to_test = [
        ("ai_core.three_version_cycle", MSG_MODULE_EXISTS),
        ("ai_core.three_version_cycle.VersionManager", MSG_CLASS_IMPORT),
        ("ai_core.three_version_cycle.DualAICoordinator", MSG_CLASS_IMPORT),
        ("ai_core.three_version_cycle.CrossVersionFeedbackSystem", MSG_CLASS_IMPORT),
        ("ai_core.three_version_cycle.V3ComparisonEngine", MSG_CLASS_IMPORT),
        ("ai_core.three_version_cycle.SpiralEvolutionManager", MSG_CLASS_IMPORT),
        ("ai_core.three_version_cycle.EnhancedSelfEvolutionCycle", MSG_CLASS_IMPORT),
    ]
    
    passed = 0
    failed = 0
    
    for import_path, description in imports_to_test:
        try:
            parts = import_path.split(".")
            if len(parts) > 2:
                module = importlib.import_module(".".join(parts[:-1]))
                getattr(module, parts[-1])
            else:
                importlib.import_module(import_path)
            print_result(f"{import_path} ({description})", True)
            passed += 1
        except Exception as e:
            print_result(f"{import_path} ({description})", False, str(e))
            failed += 1
    
    return passed, failed


# =============================================================================
# Configuration Verification
# =============================================================================

def verify_configurations() -> Tuple[int, int]:
    """Verify configuration files contain required entries."""
    print_header("Configuration Verification")
    
    root = get_project_root()
    passed = 0
    failed = 0
    
    # Define checks: (path, patterns, name, any_pattern)
    checks = [
        (root / "docker-compose.yml", ["three-version-service"], 
         "docker-compose.yml has three-version-service", False),
        (root / "gateway" / "nginx.conf", ["evolution_service", "/api/v1/evolution"], 
         "nginx.conf has evolution route", True),
        (root / "observability" / "prometheus.yml", ["three-version-service"], 
         "prometheus.yml has scrape config", False),
        (root / ".github" / "workflows" / "ci-cd.yml", ["three-version-service"], 
         "ci-cd.yml has three-version-service build", False),
        (root / "charts" / "coderev-platform" / "values.yaml", ["threeVersionService"], 
         "Helm values.yaml has threeVersionService config", False),
    ]
    
    for file_path, patterns, name, any_pattern in checks:
        _, p, f = check_file_contains(file_path, patterns, name, any_pattern)
        passed += p
        failed += f
    
    return passed, failed


# =============================================================================
# Frontend Verification
# =============================================================================

def verify_frontend() -> Tuple[int, int]:
    """Verify frontend files contain required entries."""
    print_header("Frontend Verification")
    
    root = get_project_root()
    frontend = root / "frontend"
    passed = 0
    failed = 0
    
    # Define checks: (path, patterns, name, any_pattern)
    checks = [
        (frontend / "src" / "App.tsx", 
         ["ThreeVersionControl", "/admin/three-version"],
         "App.tsx has ThreeVersionControl route", False),
        (frontend / "src" / "components" / "layout" / "Sidebar" / "Sidebar.tsx",
         ["three-version", "three_version"],
         "Sidebar.tsx has three-version nav item", True),
        (frontend / "src" / "services" / "index.ts",
         ["threeVersionService"],
         "services/index.ts exports threeVersionService", False),
        (frontend / "public" / "locales" / "en" / "translation.json",
         ["three_version"],
         "English translation has three_version keys", False),
        (frontend / "public" / "locales" / "zh-CN" / "translation.json",
         ["three_version"],
         "Chinese translation has three_version keys", False),
    ]
    
    for file_path, patterns, name, any_pattern in checks:
        _, p, f = check_file_contains(file_path, patterns, name, any_pattern)
        passed += p
        failed += f
    
    return passed, failed


# =============================================================================
# API Test (Optional)
# =============================================================================

def test_api() -> Tuple[int, int]:
    """Test API endpoints if service is running."""
    print_header("API Endpoint Tests")
    
    try:
        import httpx
    except ImportError:
        print(f"  {YELLOW}Skipping: httpx not installed{RESET}")
        return 0, 0
    
    base_url = "http://localhost:8010"
    passed = 0
    failed = 0
    
    endpoints = [
        ("GET", "/api/v1/evolution/health", 200),
        ("GET", "/api/v1/evolution/status", 200),
        ("GET", "/api/v1/evolution/ai/status", 200),
    ]
    
    for method, path, expected_status in endpoints:
        try:
            with httpx.Client(timeout=5) as client:
                response = client.request(method, f"{base_url}{path}")
                success = response.status_code == expected_status
                print_result(f"{method} {path}", success, 
                           f"Expected {expected_status}, got {response.status_code}" if not success else "")
                passed += 1 if success else 0
                failed += 0 if success else 1
        except Exception as e:
            print_result(f"{method} {path}", False, f"Connection error: {e}")
            failed += 1
    
    return passed, failed


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Verify three-version evolution system")
    parser.add_argument("--api-test", action="store_true", help="Test API endpoints (requires running service)")
    args = parser.parse_args()
    
    print(f"\n{BOLD}Three-Version Evolution System Verification{RESET}")
    print(f"{'='*60}")
    
    total_passed = 0
    total_failed = 0
    
    # File structure
    passed, failed = verify_file_structure()
    total_passed += passed
    total_failed += failed
    
    # Imports
    passed, failed = verify_imports()
    total_passed += passed
    total_failed += failed
    
    # Configurations
    passed, failed = verify_configurations()
    total_passed += passed
    total_failed += failed
    
    # Frontend
    passed, failed = verify_frontend()
    total_passed += passed
    total_failed += failed
    
    # API tests
    if args.api_test:
        passed, failed = test_api()
        total_passed += passed
        total_failed += failed
    
    # Summary
    print_header("Summary")
    print(f"  {GREEN}Passed: {total_passed}{RESET}")
    print(f"  {RED}Failed: {total_failed}{RESET}")
    
    if total_failed == 0:
        print(f"\n{GREEN}{BOLD}✓ All verifications passed!{RESET}\n")
        return 0
    else:
        print(f"\n{RED}{BOLD}✗ Some verifications failed.{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
