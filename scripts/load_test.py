#!/usr/bin/env python3
"""
Load Test Script for AI Code Review Platform

Generates concurrent requests to test system performance under load.

Usage:
    python scripts/load_test.py [options]

Options:
    --url URL           Base URL (default: http://localhost)
    --requests N        Total requests (default: 100)
    --concurrency N     Concurrent requests (default: 10)
    --duration SECS     Duration in seconds (alternative to --requests)
"""

import argparse
import asyncio
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import httpx


@dataclass
class RequestResult:
    success: bool
    status_code: int
    duration_ms: float
    error: str = ""


@dataclass
class LoadTestResults:
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    durations: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: float = 0
    end_time: float = 0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.successful / self.total_requests * 100

    @property
    def avg_duration(self) -> float:
        if not self.durations:
            return 0
        return statistics.mean(self.durations)

    @property
    def p50_duration(self) -> float:
        if not self.durations:
            return 0
        return statistics.median(self.durations)

    @property
    def p95_duration(self) -> float:
        if not self.durations:
            return 0
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.95)
        return sorted_durations[idx]

    @property
    def p99_duration(self) -> float:
        if not self.durations:
            return 0
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.99)
        return sorted_durations[min(idx, len(sorted_durations) - 1)]

    @property
    def requests_per_second(self) -> float:
        duration = self.end_time - self.start_time
        if duration == 0:
            return 0
        return self.total_requests / duration


# Sample code snippets for testing
SAMPLE_CODES = [
    {
        "code": "def hello(): print('world')",
        "language": "python"
    },
    {
        "code": "function test() { return 1 + 1; }",
        "language": "javascript"
    },
    {
        "code": """
def process_data(items):
    result = []
    for item in items:
        if item.valid:
            result.append(transform(item))
    return result
""",
        "language": "python"
    },
    {
        "code": """
async function fetchData(url) {
    const response = await fetch(url);
    const data = await response.json();
    return data;
}
""",
        "language": "javascript"
    },
    {
        "code": """
class UserService:
    def __init__(self, db):
        self.db = db
    
    def get_user(self, user_id):
        return self.db.query(User).get(user_id)
""",
        "language": "python"
    },
]


async def make_request(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore
) -> RequestResult:
    """Make a single API request."""
    async with semaphore:
        code_sample = random.choice(SAMPLE_CODES)
        
        try:
            start = time.time()
            response = await client.post(
                f"{url}/api/v2/analyze",
                json=code_sample,
                timeout=30.0
            )
            duration = (time.time() - start) * 1000
            
            return RequestResult(
                success=response.status_code == 200,
                status_code=response.status_code,
                duration_ms=duration
            )
        except httpx.TimeoutException:
            return RequestResult(
                success=False,
                status_code=0,
                duration_ms=30000,
                error="Timeout"
            )
        except httpx.ConnectError:
            return RequestResult(
                success=False,
                status_code=0,
                duration_ms=0,
                error="Connection refused"
            )
        except Exception as e:
            return RequestResult(
                success=False,
                status_code=0,
                duration_ms=0,
                error=str(e)
            )


async def run_load_test(
    url: str,
    total_requests: int,
    concurrency: int,
    duration: int = 0
) -> LoadTestResults:
    """Run the load test."""
    results = LoadTestResults()
    semaphore = asyncio.Semaphore(concurrency)
    
    async with httpx.AsyncClient() as client:
        results.start_time = time.time()
        
        if duration > 0:
            # Duration-based test
            end_time = time.time() + duration
            tasks = []
            
            while time.time() < end_time:
                task = asyncio.create_task(
                    make_request(client, url, semaphore)
                )
                tasks.append(task)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
            
            request_results = await asyncio.gather(*tasks)
        else:
            # Request count-based test
            tasks = [
                make_request(client, url, semaphore)
                for _ in range(total_requests)
            ]
            request_results = await asyncio.gather(*tasks)
        
        results.end_time = time.time()
    
    # Process results
    for result in request_results:
        results.total_requests += 1
        if result.success:
            results.successful += 1
            results.durations.append(result.duration_ms)
        else:
            results.failed += 1
            if result.error:
                results.errors.append(result.error)
    
    return results


def print_results(results: LoadTestResults):
    """Print load test results."""
    print("\n" + "=" * 60)
    print("   Load Test Results")
    print("=" * 60)
    
    print("\n📊 Summary")
    print(f"   Total Requests:    {results.total_requests}")
    print(f"   Successful:        {results.successful}")
    print(f"   Failed:            {results.failed}")
    print(f"   Success Rate:      {results.success_rate:.1f}%")
    
    print("\n⏱️  Latency")
    print(f"   Average:           {results.avg_duration:.0f}ms")
    print(f"   P50:               {results.p50_duration:.0f}ms")
    print(f"   P95:               {results.p95_duration:.0f}ms")
    print(f"   P99:               {results.p99_duration:.0f}ms")
    
    print("\n🚀 Throughput")
    print(f"   Requests/sec:      {results.requests_per_second:.1f}")
    print(f"   Duration:          {results.end_time - results.start_time:.1f}s")
    
    if results.errors:
        print(f"\n❌ Errors ({len(results.errors)})")
        error_counts = {}
        for e in results.errors:
            error_counts[e] = error_counts.get(e, 0) + 1
        for error, count in error_counts.items():
            print(f"   {error}: {count}")
    
    print("\n" + "=" * 60)
    
    # SLO Check
    print("\n📋 SLO Check")
    p95_ok = results.p95_duration <= 3000
    error_ok = (results.failed / max(results.total_requests, 1)) <= 0.02
    
    print(f"   P95 Latency < 3s:  {'✅ PASS' if p95_ok else '❌ FAIL'} ({results.p95_duration:.0f}ms)")
    print(f"   Error Rate < 2%:   {'✅ PASS' if error_ok else '❌ FAIL'} ({100 - results.success_rate:.1f}%)")
    
    if p95_ok and error_ok:
        print("\n✅ All SLOs met!")
    else:
        print("\n⚠️  SLO violations detected!")


async def main():
    parser = argparse.ArgumentParser(
        description="Load test for AI Code Review Platform"
    )
    parser.add_argument(
        "--url",
        default="http://localhost",
        help="Base URL"
    )
    parser.add_argument(
        "--requests", "-n",
        type=int,
        default=100,
        help="Total requests"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Concurrent requests"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=0,
        help="Duration in seconds (overrides --requests)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("   AI Code Review Platform - Load Test")
    print("=" * 60)
    print(f"\nURL:         {args.url}")
    print(f"Requests:    {args.requests if args.duration == 0 else 'unlimited'}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Duration:    {args.duration}s" if args.duration > 0 else "")
    print(f"\nStarting at: {datetime.now().isoformat()}")
    print("-" * 60)
    
    results = await run_load_test(
        args.url,
        args.requests,
        args.concurrency,
        args.duration
    )
    
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
