"""
Module Integration Example

Demonstrates how to use the versioned modules together.
"""

import asyncio
from typing import Dict, Any


async def main():
    """
    Example integration of multiple modules working together.
    """

    # ===========================================
    # 1. Import modules using the central loader
    # ===========================================

    from modules import (
        get_module,
        get_production_module,
        get_code_reviewer,
        get_auth_manager,
    )

    # Get production modules
    auth_module = get_production_module("Authentication")
    cache_module = get_production_module("Caching")
    monitoring_module = get_production_module("Monitoring")

    # Or get specific versions
    experimental_auth = get_module("Authentication", "V1")

    print("✓ Modules loaded successfully")

    # ===========================================
    # 2. Set up Authentication
    # ===========================================

    from modules.Authentication_V2.src.auth_manager import AuthManager
    from modules.Authentication_V2.src.session_manager import SessionManager

    auth = AuthManager()
    sessions = SessionManager()

    # Register a user
    result = await auth.register("user@example.com", "password123")
    print(f"✓ User registered: {result.success}")

    # Login
    login_result = await auth.login("user@example.com", "password123")
    print(f"✓ User logged in: {login_result.success}")

    # ===========================================
    # 3. Set up Caching
    # ===========================================

    from modules.Caching_V1.src.cache_manager import CacheManager
    from modules.Caching_V1.src.semantic_cache import SemanticCache

    cache = CacheManager()
    semantic_cache = SemanticCache()

    # Cache some data
    cache.set("user:123", {"name": "Test User"}, level="l2")
    cached = cache.get_cascading("user:123")
    print(f"✓ Cache working: {cached is not None}")

    # ===========================================
    # 4. Set up Monitoring
    # ===========================================

    from modules.Monitoring_V1.src.metrics_collector import MetricsCollector
    from modules.Monitoring_V2.src.slo_tracker import SLOTracker, SLOType

    metrics = MetricsCollector(prefix="app")
    metrics.register_counter("requests_total", "Total requests")
    metrics.register_histogram("request_duration", "Request duration")

    # Record metrics
    metrics.inc("requests_total")
    with metrics.timer("request_duration"):
        await asyncio.sleep(0.01)  # Simulate work

    print(f"✓ Metrics recorded: {metrics.get_value('requests_total')}")

    # SLO Tracking (V2)
    slo = SLOTracker()
    slo.define_slo("availability", SLOType.AVAILABILITY, 99.9)
    slo.record_event("availability", is_good=True)

    status = slo.get_status("availability")
    print(f"✓ SLO status: {status.budget_status.value}")

    # ===========================================
    # 5. Set up Self-Healing
    # ===========================================

    from modules.SelfHealing_V1.src.health_monitor import HealthMonitor
    from modules.SelfHealing_V2.src.predictive_healer import PredictiveHealer

    health = HealthMonitor()

    async def check_api():
        return True

    health.register_service("api", check_api)
    result = await health.check_service("api")
    print(f"✓ Health check: {result.status.value}")

    # Predictive healing (V2)
    healer = PredictiveHealer()
    for i in range(20):
        await healer.record_metric("api", "cpu_usage", 30 + i)

    prediction = await healer.predict_failures("api")
    print(f"✓ Prediction risk: {prediction.risk_level.value}")

    # ===========================================
    # 6. Set up AI Orchestration
    # ===========================================

    from modules.AIOrchestration_V1.src.orchestrator import Orchestrator, AITask, MockProvider
    from modules.AIOrchestration_V2.src.circuit_breaker import CircuitBreaker

    orchestrator = Orchestrator()
    orchestrator.register_provider(MockProvider("mock", "gpt-4"))

    task = AITask(task_id="test", prompt="Hello", model="gpt-4")
    result = await orchestrator.execute(task)
    print(f"✓ AI task completed: {result.status.value}")

    # Circuit breaker (V2)
    breaker = CircuitBreaker()

    async def api_call():
        return "success"

    _result = await breaker.execute(api_call)  # Result available for use
    print(f"✓ Circuit breaker: {breaker.state.value}")

    # ===========================================
    # Summary
    # ===========================================

    print("\n" + "="*50)
    print("Integration Example Complete!")
    print("="*50)
    print("""
Modules Used:
- Authentication V2: MFA, OAuth, Sessions
- Caching V1/V2: Multi-level, Semantic
- Monitoring V1/V2: Metrics, SLO Tracking
- SelfHealing V1/V2: Health Monitor, Predictive
- AIOrchestration V1/V2: Orchestrator, Circuit Breaker
""")


if __name__ == "__main__":
    asyncio.run(main())
