#!/usr/bin/env python3
"""
Environment Validation Script
环境验证脚本

Validates that all required environment variables and services are configured correctly.
验证所有必需的环境变量和服务配置是否正确。

Usage:
    python scripts/validate_env.py [--strict]
    
Options:
    --strict    Fail if optional variables are missing
"""

import os
import sys
import socket
import urllib.request
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")

def print_success(text: str):
    print(f"  {Colors.GREEN}✓{Colors.RESET} {text}")

def print_warning(text: str):
    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {text}")

def print_error(text: str):
    print(f"  {Colors.RED}✗{Colors.RESET} {text}")

def print_info(text: str):
    print(f"  {Colors.BLUE}ℹ{Colors.RESET} {text}")

# Required environment variables
REQUIRED_VARS = {
    'database': [
        ('POSTGRES_HOST', 'localhost', 'PostgreSQL host'),
        ('POSTGRES_PORT', '5432', 'PostgreSQL port'),
        ('POSTGRES_USER', 'coderev', 'PostgreSQL username'),
        ('POSTGRES_PASSWORD', None, 'PostgreSQL password'),
        ('POSTGRES_DB', 'code_review_platform', 'PostgreSQL database name'),
    ],
    'redis': [
        ('REDIS_HOST', 'localhost', 'Redis host'),
        ('REDIS_PORT', '6379', 'Redis port'),
    ],
    'security': [
        ('JWT_SECRET_KEY', None, 'JWT secret key for token signing'),
    ],
}

# Optional environment variables
OPTIONAL_VARS = {
    'ai_providers': [
        ('OPENAI_API_KEY', 'OpenAI API key'),
        ('ANTHROPIC_API_KEY', 'Anthropic API key'),
        ('HUGGINGFACE_TOKEN', 'HuggingFace token'),
    ],
    'oauth': [
        ('GITHUB_CLIENT_ID', 'GitHub OAuth client ID'),
        ('GITHUB_CLIENT_SECRET', 'GitHub OAuth client secret'),
        ('GITLAB_CLIENT_ID', 'GitLab OAuth client ID'),
        ('GITLAB_CLIENT_SECRET', 'GitLab OAuth client secret'),
    ],
    'monitoring': [
        ('PROMETHEUS_URL', 'Prometheus URL'),
        ('GRAFANA_URL', 'Grafana URL'),
    ],
}

# Services to check
SERVICES = [
    ('Backend API', 'localhost', 8000, '/health'),
    ('PostgreSQL', 'localhost', 5432, None),
    ('Redis', 'localhost', 6379, None),
    ('Neo4j', 'localhost', 7687, None),
    ('MinIO', 'localhost', 9000, None),
    ('Grafana', 'localhost', 3002, None),
    ('Prometheus', 'localhost', 9090, None),
]

def load_env_file(env_path: Path) -> Dict[str, str]:
    """Load environment variables from .env file"""
    env_vars = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

def _mask_sensitive_value(var_name: str, value: str) -> str:
    """Mask sensitive values for display."""
    sensitive_keywords = ('password', 'secret', 'key', 'token')
    is_sensitive = any(kw in var_name.lower() for kw in sensitive_keywords)
    if is_sensitive and len(value) > 4:
        return '***' + value[-4:]
    elif is_sensitive:
        return '****'
    return value


def _is_placeholder_value(value: str) -> bool:
    """Check if value is a placeholder."""
    return value.startswith('your_') or value.startswith('sk-your')


def _check_required_vars(env_vars: Dict[str, str]) -> Tuple[int, int]:
    """Check required environment variables."""
    errors = 0
    success = 0
    
    print(f"\n{Colors.BOLD}Required Variables:{Colors.RESET}")
    for category, vars_list in REQUIRED_VARS.items():
        for var_info in vars_list:
            var_name, default, description = var_info
            value = env_vars.get(var_name) or os.getenv(var_name) or default
            
            if value:
                display_value = _mask_sensitive_value(var_name, value)
                print_success(f"{var_name}: {display_value}")
                success += 1
            else:
                print_error(f"{var_name}: NOT SET - {description}")
                errors += 1
    
    return errors, success


def _check_optional_vars(env_vars: Dict[str, str], mock_mode: bool) -> Tuple[int, int]:
    """Check optional environment variables."""
    warnings = 0
    success = 0
    
    print(f"\n{Colors.BOLD}Optional Variables:{Colors.RESET}")
    for category, vars_list in OPTIONAL_VARS.items():
        for var_info in vars_list:
            var_name, description = var_info
            value = env_vars.get(var_name) or os.getenv(var_name)
            
            if value and not _is_placeholder_value(value):
                display_value = _mask_sensitive_value(var_name, value)
                print_success(f"{var_name}: {display_value}")
                success += 1
            elif mock_mode and category == 'ai_providers':
                print_info(f"{var_name}: Not set (OK in mock mode)")
            else:
                print_warning(f"{var_name}: Not set - {description}")
                warnings += 1
    
    return warnings, success


def check_env_vars(env_vars: Dict[str, str]) -> Tuple[int, int, int]:
    """Check required and optional environment variables."""
    print_header("Environment Variables Check")
    
    # Check MOCK_MODE
    mock_mode = env_vars.get('MOCK_MODE', os.getenv('MOCK_MODE', 'true')).lower() == 'true'
    if mock_mode:
        print_info("MOCK_MODE is enabled - AI provider keys are optional")
    
    # Check required and optional variables
    req_errors, req_success = _check_required_vars(env_vars)
    opt_warnings, opt_success = _check_optional_vars(env_vars, mock_mode)
    
    return req_errors, opt_warnings, req_success + opt_success

def check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def check_http_endpoint(host: str, port: int, path: str, timeout: float = 5.0) -> Tuple[bool, Optional[str]]:
    """Check if an HTTP endpoint is responding"""
    try:
        url = f"http://{host}:{port}{path}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read().decode('utf-8')
            return True, data
    except Exception as e:
        return False, str(e)

def check_services() -> Tuple[int, int]:
    """Check if required services are running"""
    print_header("Service Health Check")
    
    running = 0
    not_running = 0
    
    for service_name, host, port, http_path in SERVICES:
        port_env = os.getenv(f"{service_name.upper().replace(' ', '_')}_PORT")
        if port_env:
            port = int(port_env)
        
        if check_port(host, port):
            if http_path:
                success, _ = check_http_endpoint(host, port, http_path)
                if success:
                    print_success(f"{service_name} ({host}:{port}) - HTTP OK")
                    running += 1
                else:
                    print_warning(f"{service_name} ({host}:{port}) - Port open but HTTP failed")
                    running += 1
            else:
                print_success(f"{service_name} ({host}:{port}) - Port open")
                running += 1
        else:
            print_error(f"{service_name} ({host}:{port}) - Not responding")
            not_running += 1
    
    return running, not_running

def check_files() -> Tuple[int, int]:
    """Check if required files exist"""
    print_header("Required Files Check")
    
    project_root = Path(__file__).parent.parent
    
    required_files = [
        '.env',
        'docker-compose.yml',
        'backend/dev-api-server.py',
        'frontend/package.json',
        'frontend/vite.config.ts',
    ]
    
    found = 0
    missing = 0
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print_success(f"{file_path}")
            found += 1
        else:
            if file_path == '.env':
                print_warning(f"{file_path} - Copy from .env.example")
            else:
                print_error(f"{file_path} - Missing")
            missing += 1
    
    return found, missing

def check_node_python_versions():
    """Check Node.js and Python versions"""
    print_header("Runtime Versions")
    
    # Check Python version
    py_version = sys.version_info
    if py_version >= (3, 10):
        print_success(f"Python {py_version.major}.{py_version.minor}.{py_version.micro} (>=3.10 required)")
    else:
        print_error(f"Python {py_version.major}.{py_version.minor}.{py_version.micro} (>=3.10 required)")
    
    # Check Node.js version
    try:
        import subprocess
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        node_version = result.stdout.strip()
        major_version = int(node_version.replace('v', '').split('.')[0])
        if major_version >= 20:
            print_success(f"Node.js {node_version} (>=20 required)")
        else:
            print_warning(f"Node.js {node_version} (>=20 recommended)")
    except Exception:
        print_warning("Node.js - Could not detect version")
    
    # Check Docker
    try:
        import subprocess
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        docker_version = result.stdout.strip()
        print_success(f"Docker - {docker_version}")
    except Exception:
        print_error("Docker - Not found or not running")

def main():
    print(f"\n{Colors.BOLD}🔍 AI Code Review Platform - Environment Validation{Colors.RESET}")
    print("=" * 60)
    
    strict_mode = '--strict' in sys.argv
    
    # Find project root
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    env_example_path = project_root / '.env.example'
    
    # Load environment variables
    env_vars = {}
    if env_path.exists():
        env_vars = load_env_file(env_path)
        print_info(f"Loaded .env from {env_path}")
    elif env_example_path.exists():
        env_vars = load_env_file(env_example_path)
        print_warning("Using .env.example (copy to .env for production)")
    
    # Run checks
    check_node_python_versions()
    
    env_errors, env_warnings, env_success = check_env_vars(env_vars)
    
    services_running, services_not_running = check_services()
    
    files_found, files_missing = check_files()
    
    # Summary
    print_header("Summary")
    
    total_errors = env_errors + files_missing
    total_warnings = env_warnings + services_not_running
    
    print(f"\n  Environment Variables: {env_success} OK, {env_warnings} warnings, {env_errors} errors")
    print(f"  Services: {services_running} running, {services_not_running} not running")
    print(f"  Files: {files_found} found, {files_missing} missing")
    
    if total_errors == 0 and (total_warnings == 0 or not strict_mode):
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Environment is ready!{Colors.RESET}")
        print("\n  Start the platform with:")
        print("    1. docker compose up -d")
        print("    2. cd backend && python dev-api-server.py")
        print("    3. cd frontend && npm run dev")
        print("\n  Access:")
        print("    - Frontend: http://localhost:5173")
        print("    - API Docs: http://localhost:8000/docs")
        return 0
    elif total_errors > 0:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Environment has errors that must be fixed{Colors.RESET}")
        return 1
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ Environment has warnings (non-critical){Colors.RESET}")
        return 0 if not strict_mode else 1

if __name__ == '__main__':
    sys.exit(main())
