"""
Unit Tests for Enhanced Circuit Breaker Pattern

Tests cover:
- Dynamic threshold calculation
- State transitions
- Fallback mechanisms
- Provider-specific circuit breakers
- Real-time monitoring
"""
import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, 'd:/Desktop/AI-Code-Review-and-Architecture-Analysis_Platform')

from backend.shared.patterns.circuit_breaker.enhanced_circuit_breaker import (
    EnhancedCircuitBreaker, DynamicThresholdConfig, CircuitState,
    CircuitOpenError, FailureType, RequestMetric, CircuitBreakerMetrics,
    circuit_breaker,
)
from backend.shared.patterns.circuit_breaker.provider_circuit_breakers import (
    ProviderCircuitBreakerManager, ProviderConfig, ProviderType,
    ProviderHealth, create_default_provider_manager, get_provider_manager,
)
from backend.shared.patterns.circuit_breaker.monitoring import (
    CircuitBreakerMonitor, AlertSeverity, AlertType, Alert,
)


# =============================================================================
# DynamicThresholdConfig Tests
# =============================================================================

class TestDynamicThresholdConfig:
    """Tests for DynamicThresholdConfig."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        config = DynamicThresholdConfig()
        
        assert config.failure_rate_threshold == 0.50
        assert config.window_seconds == 30
        assert config.minimum_requests == 10
        assert config.recovery_timeout_seconds == 30.0
    
    def test_custom_values(self):
        """Should accept custom values."""
        config = DynamicThresholdConfig(
            failure_rate_threshold=0.30,
            window_seconds=60,
            minimum_requests=20
        )
        
        assert config.failure_rate_threshold == 0.30
        assert config.window_seconds == 60
        assert config.minimum_requests == 20


# =============================================================================
# CircuitBreakerMetrics Tests
# =============================================================================

class TestCircuitBreakerMetrics:
    """Tests for CircuitBreakerMetrics."""
    
    def test_initial_values(self):
        """Should have zero initial values."""
        metrics = CircuitBreakerMetrics()
        
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.current_failure_rate == 0.0
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = CircuitBreakerMetrics()
        metrics.total_requests = 100
        metrics.failed_requests = 10
        
        data = metrics.to_dict()
        
        assert data["total_requests"] == 100
        assert data["failed_requests"] == 10
        assert "state" in data


# =============================================================================
# EnhancedCircuitBreaker Tests
# =============================================================================

class TestEnhancedCircuitBreaker:
    """Tests for EnhancedCircuitBreaker."""
    
    @pytest.fixture
    def breaker(self):
        return EnhancedCircuitBreaker(
            name="test_breaker",
            config=DynamicThresholdConfig(
                failure_rate_threshold=0.50,
                window_seconds=30,
                minimum_requests=5,
                consecutive_failure_threshold=3
            )
        )
    
    def test_initial_state_is_closed(self, breaker):
        """Initial state should be CLOSED."""
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_record_success_increments_counters(self, breaker):
        """Recording success should increment counters."""
        await breaker.record_success(100.0)
        
        assert breaker.metrics.total_requests == 1
        assert breaker.metrics.successful_requests == 1
        assert breaker.metrics.consecutive_successes == 1
    
    @pytest.mark.asyncio
    async def test_record_failure_increments_counters(self, breaker):
        """Recording failure should increment counters."""
        await breaker.record_failure(100.0, FailureType.ERROR, "Test error")
        
        assert breaker.metrics.total_requests == 1
        assert breaker.metrics.failed_requests == 1
        assert breaker.metrics.consecutive_failures == 1
    
    @pytest.mark.asyncio
    async def test_opens_on_consecutive_failures(self, breaker):
        """Should open after consecutive failure threshold."""
        for _ in range(3):  # consecutive_failure_threshold = 3
            await breaker.record_failure(100.0, FailureType.ERROR)
        
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_opens_on_high_failure_rate(self, breaker):
        """Should open when failure rate exceeds threshold."""
        # Add enough requests to meet minimum
        for _ in range(3):
            await breaker.record_success(100.0)
        
        # Now add failures to exceed 50% rate
        for _ in range(5):
            await breaker.record_failure(100.0, FailureType.ERROR)
        
        # Force recalculation
        await breaker.record_failure(100.0, FailureType.ERROR)
        
        # Circuit should be open due to either consecutive failures or rate
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_rejects_when_open(self, breaker):
        """Should reject requests when circuit is open."""
        # Force open state
        breaker._state = CircuitState.OPEN
        breaker._metrics.circuit_opened_at = datetime.now(timezone.utc)
        
        async def dummy_func():
            return "success"
        
        with pytest.raises(CircuitOpenError):
            await breaker.execute(dummy_func)
    
    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self, breaker):
        """Should transition to HALF_OPEN after recovery timeout."""
        breaker._state = CircuitState.OPEN
        breaker._metrics.circuit_opened_at = datetime.now(timezone.utc) - timedelta(seconds=60)
        
        # Check state update
        await breaker._check_and_update_state()
        
        assert breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_closes_after_successful_recovery(self, breaker):
        """Should close after successful requests in HALF_OPEN state."""
        breaker._state = CircuitState.HALF_OPEN
        breaker.config.recovery_success_threshold = 2
        
        await breaker.record_success(100.0)
        await breaker.record_success(100.0)
        
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_reopens_on_failure_in_half_open(self, breaker):
        """Should reopen on failure in HALF_OPEN state."""
        breaker._state = CircuitState.HALF_OPEN
        
        await breaker.record_failure(100.0, FailureType.ERROR)
        
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_execute_with_successful_function(self, breaker):
        """Should execute function and record success."""
        async def success_func():
            return "result"
        
        result = await breaker.execute(success_func)
        
        assert result == "result"
        assert breaker.metrics.successful_requests == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_failing_function(self, breaker):
        """Should record failure when function raises."""
        async def fail_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await breaker.execute(fail_func)
        
        assert breaker.metrics.failed_requests == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback(self):
        """Should use fallback when circuit is open."""
        async def fallback_func(*args, **kwargs):
            return "fallback_result"
        
        breaker = EnhancedCircuitBreaker(
            name="test",
            fallback=fallback_func
        )
        breaker._state = CircuitState.OPEN
        breaker._metrics.circuit_opened_at = datetime.now(timezone.utc)
        
        async def main_func():
            return "main_result"
        
        result = await breaker.execute(main_func)
        
        assert result == "fallback_result"
    
    @pytest.mark.asyncio
    async def test_reset(self, breaker):
        """Should reset to initial state."""
        await breaker.record_failure(100.0, FailureType.ERROR)
        await breaker.record_failure(100.0, FailureType.ERROR)
        
        await breaker.reset()
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.metrics.total_requests == 0
    
    def test_get_status(self, breaker):
        """Should return comprehensive status."""
        status = breaker.get_status()
        
        assert status["name"] == "test_breaker"
        assert "state" in status
        assert "metrics" in status
        assert "config" in status
    
    def test_interception_delay_target(self, breaker):
        """Should have interception delay target defined."""
        assert breaker.INTERCEPTION_DELAY_TARGET_MS == 100


class TestCircuitBreakerDecorator:
    """Tests for circuit_breaker decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_creates_breaker(self):
        """Decorator should create circuit breaker."""
        @circuit_breaker("test_service")
        async def my_service():
            return "result"
        
        assert hasattr(my_service, "circuit_breaker")
        assert my_service.circuit_breaker.name == "test_service"
    
    @pytest.mark.asyncio
    async def test_decorator_executes_function(self):
        """Decorator should execute wrapped function."""
        @circuit_breaker("test_service")
        async def my_service():
            return "result"
        
        result = await my_service()
        assert result == "result"


# =============================================================================
# ProviderCircuitBreakerManager Tests
# =============================================================================

class TestProviderCircuitBreakerManager:
    """Tests for ProviderCircuitBreakerManager."""
    
    @pytest.fixture
    def manager(self):
        return ProviderCircuitBreakerManager()
    
    def test_register_provider(self, manager):
        """Should register provider with circuit breaker."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="openai_test",
            endpoint="https://api.openai.com"
        )
        
        manager.register_provider(config)
        
        assert "openai_test" in manager._providers
        assert "openai_test" in manager._breakers
    
    def test_unregister_provider(self, manager):
        """Should remove provider."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="openai_test",
            endpoint="https://api.openai.com"
        )
        
        manager.register_provider(config)
        manager.unregister_provider("openai_test")
        
        assert "openai_test" not in manager._providers
    
    def test_get_available_providers(self, manager):
        """Should return providers with non-open circuits."""
        config1 = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="provider1",
            endpoint="https://api.openai.com"
        )
        config2 = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            name="provider2",
            endpoint="https://api.anthropic.com"
        )
        
        manager.register_provider(config1)
        manager.register_provider(config2)
        
        available = manager.get_available_providers()
        
        assert len(available) == 2
    
    def test_select_provider_returns_healthy(self, manager):
        """Should select healthy provider."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="provider1",
            endpoint="https://api.openai.com",
            priority=1
        )
        
        manager.register_provider(config)
        
        selected = manager.select_provider()
        
        assert selected == "provider1"
    
    def test_select_provider_prefers_preferred(self, manager):
        """Should prefer the preferred provider if healthy."""
        config1 = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="provider1",
            endpoint="https://api.openai.com",
            priority=1
        )
        config2 = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            name="provider2",
            endpoint="https://api.anthropic.com",
            priority=2
        )
        
        manager.register_provider(config1)
        manager.register_provider(config2)
        
        selected = manager.select_provider(preferred="provider2")
        
        assert selected == "provider2"
    
    @pytest.mark.asyncio
    async def test_execute_through_provider(self, manager):
        """Should execute function through provider's circuit breaker."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="provider1",
            endpoint="https://api.openai.com"
        )
        manager.register_provider(config)
        
        async def test_func():
            return "result"
        
        result = await manager.execute("provider1", test_func)
        
        assert result == "result"
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, manager):
        """Should fallback to next provider on failure."""
        config1 = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="provider1",
            endpoint="https://api.openai.com"
        )
        config2 = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            name="provider2",
            endpoint="https://api.anthropic.com"
        )
        
        manager.register_provider(config1, fallback_providers=["provider2"])
        manager.register_provider(config2)
        
        # Force provider1 to be open
        manager._breakers["provider1"]._state = CircuitState.OPEN
        manager._breakers["provider1"]._metrics.circuit_opened_at = datetime.now(timezone.utc)
        
        call_count = 0
        async def test_func(provider_name):
            nonlocal call_count
            call_count += 1
            return f"result_from_{provider_name}"
        
        result = await manager.execute_with_fallback(
            test_func,
            preferred_provider="provider1"
        )
        
        assert "provider2" in result
    
    def test_get_provider_health(self, manager):
        """Should return provider health status."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="provider1",
            endpoint="https://api.openai.com"
        )
        manager.register_provider(config)
        
        health = manager.get_provider_health("provider1")
        
        assert health is not None
        assert health.provider_name == "provider1"
        assert health.circuit_state == CircuitState.CLOSED
    
    def test_get_metrics(self, manager):
        """Should return manager metrics."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="provider1",
            endpoint="https://api.openai.com"
        )
        manager.register_provider(config)
        
        metrics = manager.get_metrics()
        
        assert "total_providers" in metrics
        assert "healthy_providers" in metrics
        assert "fault_isolation_rate" in metrics
        assert metrics["target_isolation_rate"] == 0.999


class TestDefaultProviderManager:
    """Tests for default provider manager creation."""
    
    def test_create_default_manager(self):
        """Should create manager with default providers."""
        manager = create_default_provider_manager()
        
        assert "openai_primary" in manager._providers
        assert "anthropic_primary" in manager._providers
        assert "local_llm" in manager._providers


# =============================================================================
# CircuitBreakerMonitor Tests
# =============================================================================

class TestCircuitBreakerMonitor:
    """Tests for CircuitBreakerMonitor."""
    
    @pytest.fixture
    def monitor(self):
        manager = ProviderCircuitBreakerManager()
        manager.register_provider(ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="test_provider",
            endpoint="https://api.openai.com"
        ))
        return CircuitBreakerMonitor(manager)
    
    def test_get_current_status(self, monitor):
        """Should return current status."""
        status = monitor.get_current_status()
        
        assert "providers" in status
        assert "metrics" in status
        assert "healthy_count" in status
    
    def test_get_active_alerts_initially_empty(self, monitor):
        """Should have no alerts initially."""
        alerts = monitor.get_active_alerts()
        assert len(alerts) == 0
    
    def test_acknowledge_alert(self, monitor):
        """Should acknowledge alert."""
        # Create a test alert
        monitor._alerts.append(Alert(
            alert_id="alert_1",
            alert_type=AlertType.CIRCUIT_OPENED,
            severity=AlertSeverity.ERROR,
            provider_name="test_provider",
            message="Test alert",
            details={}
        ))
        
        result = monitor.acknowledge_alert("alert_1")
        
        assert result is True
        assert monitor._alerts[0].acknowledged is True
    
    def test_subscribe_returns_queue(self, monitor):
        """Should return queue for subscriptions."""
        queue = monitor.subscribe()
        
        assert queue is not None
        assert isinstance(queue, asyncio.Queue)
    
    def test_unsubscribe(self, monitor):
        """Should remove subscriber."""
        queue = monitor.subscribe()
        assert queue in monitor._subscribers
        
        monitor.unsubscribe(queue)
        assert queue not in monitor._subscribers
    
    def test_get_dashboard_data(self, monitor):
        """Should return dashboard data."""
        data = monitor.get_dashboard_data()
        
        assert "current_status" in data
        assert "active_alerts" in data
        assert "metrics_summary" in data
        assert "timestamp" in data


class TestAlertTypes:
    """Tests for alert types and severity."""
    
    def test_alert_types_defined(self):
        """All alert types should be defined."""
        assert AlertType.CIRCUIT_OPENED.value == "circuit_opened"
        assert AlertType.HIGH_FAILURE_RATE.value == "high_failure_rate"
        assert AlertType.ALL_PROVIDERS_DOWN.value == "all_providers_down"
    
    def test_alert_severity_defined(self):
        """All severity levels should be defined."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"
    
    def test_alert_to_dict(self):
        """Alert should convert to dictionary."""
        alert = Alert(
            alert_id="alert_1",
            alert_type=AlertType.CIRCUIT_OPENED,
            severity=AlertSeverity.ERROR,
            provider_name="test",
            message="Test alert",
            details={"key": "value"}
        )
        
        data = alert.to_dict()
        
        assert data["alert_id"] == "alert_1"
        assert data["alert_type"] == "circuit_opened"
        assert data["severity"] == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
