#!/usr/bin/env python3
"""
Health Check Script for AI Code Review Platform

Checks the health of all platform services and reports status.

Usage:
    python scripts/health_check.py [--json] [--verbose]
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx


@dataclass
class ServiceHealth:
    name: str
    url: str
    status: str  # healthy, degraded, unhealthy, unknown
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict] = None


@dataclass
class HealthReport:
    timestamp: str
    overall_status: str
    services: List[ServiceHealth]
    summary: Dict[str, int]


# Service endpoints to check
SERVICES = [
    {"name": "API Gateway", "url": "http://localhost/health"},
    {"name": "VCAI V2", "url": "http://localhost:8001/health"},
    {"name": "CRAI V2", "url": "http://localhost:8002/health"},
    {"name": "Lifecycle Controller", "url": "http://localhost:8003/health"},
    {"name": "Evaluation Pipeline", "url": "http://localhost:8004/health"},
    {"name": "Prometheus", "url": "http://localhost:9090/-/healthy"},
    {"name": "Grafana", "url": "http://localhost:3001/api/health"},
    {"name": "OPA", "url": "http://localhost:8181/health"},
    {"name": "Redis", "url": "http://localhost:6379", "type": "tcp"},
    {"name": "PostgreSQL", "url": "http://localhost:5432", "type": "tcp"},
]


async def check_http_health(
    client: httpx.AsyncClient,
    name: str,
    url: str,
    timeout: float = 5.0
) -> ServiceHealth:
    """Check health of an HTTP service."""
    try:
        start = datetime.now()
        response = await client.get(url, timeout=timeout)
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        if response.status_code == 200:
            try:
                details = response.json()
            except:
                details = {"raw": response.text[:100]}
            
            return ServiceHealth(
                name=name,
                url=url,
                status="healthy",
                response_time_ms=round(elapsed, 2),
                details=details
            )
        else:
            return ServiceHealth(
                name=name,
                url=url,
                status="degraded",
                response_time_ms=round(elapsed, 2),
                error=f"HTTP {response.status_code}"
            )
    except httpx.ConnectError:
        return ServiceHealth(
            name=name,
            url=url,
            status="unhealthy",
            error="Connection refused"
        )
    except httpx.TimeoutException:
        return ServiceHealth(
            name=name,
            url=url,
            status="unhealthy",
            error="Timeout"
        )
    except Exception as e:
        return ServiceHealth(
            name=name,
            url=url,
            status="unknown",
            error=str(e)
        )


async def check_tcp_health(
    name: str,
    host: str,
    port: int,
    timeout: float = 5.0
) -> ServiceHealth:
    """Check health of a TCP service."""
    try:
        start = datetime.now()
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )
        elapsed = (datetime.now() - start).total_seconds() * 1000
        writer.close()
        await writer.wait_closed()
        
        return ServiceHealth(
            name=name,
            url=f"tcp://{host}:{port}",
            status="healthy",
            response_time_ms=round(elapsed, 2)
        )
    except asyncio.TimeoutError:
        return ServiceHealth(
            name=name,
            url=f"tcp://{host}:{port}",
            status="unhealthy",
            error="Connection timeout"
        )
    except ConnectionRefusedError:
        return ServiceHealth(
            name=name,
            url=f"tcp://{host}:{port}",
            status="unhealthy",
            error="Connection refused"
        )
    except Exception as e:
        return ServiceHealth(
            name=name,
            url=f"tcp://{host}:{port}",
            status="unknown",
            error=str(e)
        )


async def run_health_checks(verbose: bool = False) -> HealthReport:
    """Run health checks on all services."""
    results: List[ServiceHealth] = []
    
    async with httpx.AsyncClient() as client:
        tasks = []
        
        for service in SERVICES:
            if service.get("type") == "tcp":
                # Parse TCP URL
                url = service["url"]
                if "://" in url:
                    url = url.split("://")[1]
                host, port = url.split(":")
                tasks.append(check_tcp_health(
                    service["name"],
                    host,
                    int(port)
                ))
            else:
                tasks.append(check_http_health(
                    client,
                    service["name"],
                    service["url"]
                ))
        
        results = await asyncio.gather(*tasks)
    
    # Calculate summary
    summary = {"healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    
    # Determine overall status
    if summary["unhealthy"] > 0:
        overall = "unhealthy"
    elif summary["degraded"] > 0:
        overall = "degraded"
    elif summary["unknown"] > 0:
        overall = "unknown"
    else:
        overall = "healthy"
    
    return HealthReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_status=overall,
        services=results,
        summary=summary
    )


def print_report(report: HealthReport, verbose: bool = False):
    """Print health report in human-readable format."""
    # Status colors
    status_symbols = {
        "healthy": "✅",
        "degraded": "⚠️",
        "unhealthy": "❌",
        "unknown": "❓"
    }
    
    print("\n" + "=" * 60)
    print("   AI Code Review Platform - Health Check")
    print("=" * 60)
    print(f"\nTimestamp: {report.timestamp}")
    print(f"Overall Status: {status_symbols[report.overall_status]} {report.overall_status.upper()}")
    print("\n" + "-" * 60)
    print("Services:")
    print("-" * 60)
    
    for service in report.services:
        symbol = status_symbols[service.status]
        line = f"  {symbol} {service.name:<25} {service.status:<10}"
        
        if service.response_time_ms:
            line += f" ({service.response_time_ms}ms)"
        
        print(line)
        
        if verbose and service.error:
            print(f"      Error: {service.error}")
        
        if verbose and service.details:
            print(f"      Details: {json.dumps(service.details, indent=2)[:100]}")
    
    print("\n" + "-" * 60)
    print("Summary:")
    print("-" * 60)
    print(f"  Healthy:   {report.summary['healthy']}")
    print(f"  Degraded:  {report.summary['degraded']}")
    print(f"  Unhealthy: {report.summary['unhealthy']}")
    print(f"  Unknown:   {report.summary['unknown']}")
    print("=" * 60 + "\n")


def print_json_report(report: HealthReport):
    """Print health report as JSON."""
    data = {
        "timestamp": report.timestamp,
        "overall_status": report.overall_status,
        "summary": report.summary,
        "services": [asdict(s) for s in report.services]
    }
    print(json.dumps(data, indent=2))


async def main():
    parser = argparse.ArgumentParser(
        description="Health check for AI Code Review Platform"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    report = await run_health_checks(verbose=args.verbose)
    
    if args.json:
        print_json_report(report)
    else:
        print_report(report, verbose=args.verbose)
    
    # Exit code based on overall status
    if report.overall_status == "healthy":
        sys.exit(0)
    elif report.overall_status == "degraded":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
