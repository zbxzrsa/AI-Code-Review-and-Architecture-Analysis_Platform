#!/usr/bin/env python3
"""
API Test Script for AI Code Review Platform

Quick smoke tests for all platform APIs.

Usage:
    python scripts/api_test.py [--base-url URL] [--verbose]
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import httpx


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    response: Optional[dict] = None


# Test configurations
TESTS = [
    # Health checks
    {"name": "Gateway Health", "method": "GET", "path": "/health"},
    {"name": "VCAI Health", "method": "GET", "path": "/api/v2/health"},
    {"name": "Lifecycle Controller Health", "method": "GET", "path": "/api/admin/lifecycle/health"},
    
    # API endpoints
    {"name": "List Versions", "method": "GET", "path": "/api/admin/lifecycle/versions"},
    {"name": "Comparison Stats", "method": "GET", "path": "/api/admin/lifecycle/stats/comparison"},
    {"name": "Rollback History", "method": "GET", "path": "/api/admin/lifecycle/rollback/history?limit=5"},
    
    # Code analysis
    {
        "name": "Code Analysis",
        "method": "POST",
        "path": "/api/v2/analyze",
        "body": {
            "code": "def hello(): print('world')",
            "language": "python"
        }
    },
]


async def run_test(
    client: httpx.AsyncClient,
    test: dict,
    verbose: bool = False
) -> TestResult:
    """Run a single API test."""
    name = test["name"]
    method = test["method"]
    path = test["path"]
    body = test.get("body")
    
    try:
        start = datetime.now()
        
        if method == "GET":
            response = await client.get(path, timeout=30.0)
        elif method == "POST":
            response = await client.post(path, json=body, timeout=30.0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        duration = (datetime.now() - start).total_seconds() * 1000
        
        if response.status_code in [200, 201, 202]:
            try:
                data = response.json()
            except (ValueError, TypeError):
                data = {"raw": response.text[:100]}
            
            return TestResult(
                name=name,
                passed=True,
                duration_ms=round(duration, 2),
                response=data if verbose else None
            )
        else:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=round(duration, 2),
                error=f"HTTP {response.status_code}: {response.text[:100]}"
            )
    
    except httpx.ConnectError:
        return TestResult(
            name=name,
            passed=False,
            duration_ms=0,
            error="Connection refused"
        )
    except httpx.TimeoutException:
        return TestResult(
            name=name,
            passed=False,
            duration_ms=30000,
            error="Request timeout"
        )
    except Exception as e:
        return TestResult(
            name=name,
            passed=False,
            duration_ms=0,
            error=str(e)
        )


async def run_all_tests(base_url: str, verbose: bool = False) -> list[TestResult]:
    """Run all API tests."""
    results = []
    
    async with httpx.AsyncClient(base_url=base_url) as client:
        for test in TESTS:
            result = await run_test(client, test, verbose)
            results.append(result)
            
            # Print progress
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.name} ({result.duration_ms}ms)")
            
            if not result.passed and result.error:
                print(f"      Error: {result.error}")
            
            if verbose and result.response:
                print(f"      Response: {json.dumps(result.response, indent=2)[:200]}")
    
    return results


def print_summary(results: list[TestResult]):
    """Print test summary."""
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_time = sum(r.duration_ms for r in results)
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {len(results)}")
    print(f"  Time:   {total_time:.0f}ms")
    print("=" * 50)
    
    if failed > 0:
        print("\nFailed Tests:")
        for r in results:
            if not r.passed:
                print(f"  ❌ {r.name}: {r.error}")


async def main():
    parser = argparse.ArgumentParser(
        description="API tests for AI Code Review Platform"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost",
        help="Base URL for API calls"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 50)
    print("AI Code Review Platform - API Tests")
    print("=" * 50)
    print(f"Base URL: {args.base_url}")
    print(f"Time: {datetime.now().isoformat()}")
    print("-" * 50 + "\n")
    
    results = await run_all_tests(args.base_url, args.verbose)
    print_summary(results)
    
    # Exit code
    failed = sum(1 for r in results if not r.passed)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
