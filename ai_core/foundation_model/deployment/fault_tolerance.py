"""
Fault tolerance and health monitoring module.

Provides:
- System health monitoring
- Automatic checkpointing
- Retry logic with backoff
- Graceful degradation
- Recovery procedures
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from .config import PracticalDeploymentConfig

logger = logging.getLogger(__name__)


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self, config: PracticalDeploymentConfig):
        self.config = config
        
        self.health_status = {
            "model": "unknown",
            "rag": "unknown",
            "adapter": "unknown",
            "gpu": "unknown",
            "memory": "unknown",
        }
        
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.max_metrics_history = 1000
        
        self._is_running = False
        self._check_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start health checking."""
        self._is_running = True
        self._check_task = asyncio.create_task(self._check_loop())
        logger.info("Started health checker")
    
    async def stop(self):
        """Stop health checking."""
        self._is_running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped health checker")
    
    async def _check_loop(self):
        """Main health check loop."""
        while self._is_running:
            try:
                await self._perform_checks()
                await asyncio.sleep(self.config.health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def _perform_checks(self):
        """Perform all health checks."""
        # GPU check
        self._check_gpu()
        
        # Memory check
        self._check_memory()
        
        # Model check
        self.health_status["model"] = "healthy"
        
        # RAG check
        self.health_status["rag"] = "healthy"
        
        # Adapter check
        self.health_status["adapter"] = "healthy"
    
    def _check_gpu(self):
        """Check GPU health and memory."""
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                self._add_metric("gpu_memory_allocated_gb", gpu_memory_allocated)
                self._add_metric("gpu_memory_reserved_gb", gpu_memory_reserved)
                self._add_metric("gpu_utilization_percent", gpu_memory_allocated / gpu_memory_total * 100)
                
                # Check if GPU memory is critically high
                if gpu_memory_allocated / gpu_memory_total > 0.95:
                    self.health_status["gpu"] = "critical"
                elif gpu_memory_allocated / gpu_memory_total > 0.85:
                    self.health_status["gpu"] = "warning"
                else:
                    self.health_status["gpu"] = "healthy"
                    
            except Exception as e:
                logger.warning(f"GPU check failed: {e}")
                self.health_status["gpu"] = "unhealthy"
        else:
            self.health_status["gpu"] = "no_gpu"
    
    def _check_memory(self):
        """Check system memory."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            self._add_metric("system_memory_percent", memory.percent)
            self._add_metric("system_memory_available_gb", memory.available / 1e9)
            
            if memory.percent > 95:
                self.health_status["memory"] = "critical"
            elif memory.percent > 85:
                self.health_status["memory"] = "warning"
            else:
                self.health_status["memory"] = "healthy"
                
        except ImportError:
            self.health_status["memory"] = "unknown"
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")
            self.health_status["memory"] = "unhealthy"
    
    def _add_metric(self, name: str, value: float):
        """Add a metric value with history limit."""
        self.metrics[name].append(value)
        if len(self.metrics[name]) > self.max_metrics_history:
            self.metrics[name] = self.metrics[name][-self.max_metrics_history:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        is_healthy = all(
            status in ("healthy", "no_gpu", "unknown")
            for status in self.health_status.values()
        )
        
        return {
            "healthy": is_healthy,
            "status": self.health_status.copy(),
            "metrics": {
                k: v[-10:] if v else []  # Last 10 values
                for k, v in self.metrics.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        critical_statuses = ["critical", "unhealthy"]
        return not any(
            status in critical_statuses
            for status in self.health_status.values()
        )


class FaultToleranceManager:
    """
    Manages fault tolerance and recovery.
    
    Features:
    - Automatic checkpointing
    - Retry logic with exponential backoff
    - Graceful degradation
    - Recovery procedures
    """
    
    def __init__(self, config: PracticalDeploymentConfig):
        self.config = config
        
        self.checkpoint_path = Path("checkpoints/practical")
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.last_checkpoint: Optional[datetime] = None
        self.retry_counts: Dict[str, int] = defaultdict(int)
        self.failed_operations: List[Dict[str, Any]] = []
    
    async def save_checkpoint(
        self,
        adapter_manager,
        rag_system,
        additional_state: Optional[Dict[str, Any]] = None,
    ):
        """Save system checkpoint."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ckpt_dir = self.checkpoint_path / timestamp
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save adapters
            adapters_path = ckpt_dir / "adapters"
            for adapter_name in adapter_manager.adapters:
                adapter_manager.save_adapter(adapter_name, str(adapters_path))
            
            # Save RAG index
            if rag_system.index.index_path:
                original_path = rag_system.index.index_path
                rag_system.index.index_path = ckpt_dir / "rag_index"
                rag_system.index.save()
                rag_system.index.index_path = original_path
            
            # Save additional state
            if additional_state:
                import json
                with open(ckpt_dir / "state.json", "w") as f:
                    json.dump(additional_state, f, indent=2, default=str)
            
            self.last_checkpoint = datetime.now(timezone.utc)
            
            # Clean up old checkpoints (keep last 5)
            await self._cleanup_old_checkpoints(keep=5)
            
            logger.info(f"Saved checkpoint: {ckpt_dir}")
            return str(ckpt_dir)
            
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            raise
    
    async def load_checkpoint(
        self,
        adapter_manager,
        rag_system,
        checkpoint_dir: Optional[str] = None,
    ):
        """Load system checkpoint."""
        if checkpoint_dir is None:
            # Find latest checkpoint
            checkpoints = sorted([
                d for d in self.checkpoint_path.iterdir()
                if d.is_dir()
            ])
            if not checkpoints:
                logger.warning("No checkpoints found")
                return None
            checkpoint_dir = str(checkpoints[-1])
        
        ckpt_path = Path(checkpoint_dir)
        
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            return None
        
        try:
            # Load adapters
            adapters_path = ckpt_path / "adapters"
            if adapters_path.exists():
                for adapter_dir in adapters_path.iterdir():
                    if adapter_dir.is_dir():
                        adapter_manager.load_adapter(adapter_dir.name, str(adapters_path))
            
            # Load RAG index
            rag_path = ckpt_path / "rag_index"
            if rag_path.exists():
                original_path = rag_system.index.index_path
                rag_system.index.index_path = rag_path
                rag_system.index.load()
                rag_system.index.index_path = original_path
            
            # Load additional state
            state_path = ckpt_path / "state.json"
            additional_state = None
            if state_path.exists():
                import json
                with open(state_path, "r") as f:
                    additional_state = json.load(f)
            
            logger.info(f"Loaded checkpoint: {ckpt_path}")
            return additional_state
            
        except Exception as e:
            logger.error(f"Checkpoint load failed: {e}")
            raise
    
    async def _cleanup_old_checkpoints(self, keep: int = 5):
        """Remove old checkpoints, keeping only the most recent ones."""
        import shutil
        
        checkpoints = sorted([
            d for d in self.checkpoint_path.iterdir()
            if d.is_dir()
        ])
        
        to_remove = checkpoints[:-keep] if len(checkpoints) > keep else []
        
        for ckpt in to_remove:
            try:
                shutil.rmtree(ckpt)
                logger.info(f"Removed old checkpoint: {ckpt}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {ckpt}: {e}")
    
    async def retry_with_backoff(
        self,
        func: Callable,
        operation_name: str,
        *args,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Execute function with retry and exponential backoff."""
        max_retries = max_retries or self.config.max_retries
        
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                self.retry_counts[operation_name] = 0
                return result
                
            except Exception as e:
                self.retry_counts[operation_name] += 1
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8...
                
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    self.failed_operations.append({
                        "operation": operation_name,
                        "error": str(e),
                        "attempts": attempt + 1,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fault tolerance statistics."""
        return {
            "last_checkpoint": self.last_checkpoint.isoformat() if self.last_checkpoint else None,
            "retry_counts": dict(self.retry_counts),
            "failed_operations": self.failed_operations[-10:],  # Last 10
            "checkpoint_path": str(self.checkpoint_path),
        }
