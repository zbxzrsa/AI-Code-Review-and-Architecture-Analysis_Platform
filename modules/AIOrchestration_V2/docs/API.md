# AIOrchestration_V2 API Reference

## Overview

Production AI orchestration with load balancing and circuit breaker.

## Classes

### Orchestrator

Main orchestration class for AI task execution.

```python
from modules.AIOrchestration_V2 import Orchestrator

orchestrator = Orchestrator(
    default_timeout=30,
    max_concurrent=10
)
```

#### Methods

- `register_provider(provider)` - Register AI provider
- `execute(task)` - Execute single task
- `execute_batch(tasks)` - Execute multiple tasks
- `get_metrics()` - Get orchestration metrics

### LoadBalancer

Health-aware load balancing across providers.

```python
from modules.AIOrchestration_V2.src.load_balancer import LoadBalancer, LoadBalancingStrategy

balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME)
balancer.add_endpoint("openai", weight=2, max_connections=100)
```

#### Strategies

- `ROUND_ROBIN` - Rotate through endpoints
- `WEIGHTED_ROUND_ROBIN` - Weight-based rotation
- `LEAST_CONNECTIONS` - Select least busy
- `LEAST_RESPONSE_TIME` - Select fastest
- `RANDOM` - Random selection

### CircuitBreaker

Fault tolerance with three-state circuit.

```python
from modules.AIOrchestration_V2.src.circuit_breaker import CircuitBreaker, CircuitConfig

config = CircuitConfig(failure_threshold=5, timeout_seconds=30)
breaker = CircuitBreaker(config)

result = await breaker.execute(async_function)
```

#### States

- `CLOSED` - Normal operation
- `OPEN` - Failing, reject requests
- `HALF_OPEN` - Testing recovery

## Configuration

See `config/orchestration_config.yaml`
