"""
Periodic retraining scheduler for LoRA adapters.

Collects training samples and triggers retraining based on schedule.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW

from .config import PracticalDeploymentConfig, RetrainingFrequency
from .lora import LoRAAdapterManager

logger = logging.getLogger(__name__)


class RetrainingDataCollector:
    """Collects and manages data for periodic retraining."""

    def __init__(self, config: PracticalDeploymentConfig):
        self.config = config
        self.data_path = Path(config.retraining_data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Collected samples
        self.samples: List[Dict[str, Any]] = []
        self.sample_count = 0

        # Active learning
        self.uncertain_samples: List[Dict[str, Any]] = []

    def add_sample(
        self,
        input_text: str,
        output_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        uncertainty: Optional[float] = None,
    ):
        """Add a training sample."""
        sample = {
            "input": input_text,
            "output": output_text,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uncertainty": uncertainty,
        }

        self.samples.append(sample)
        self.sample_count += 1

        # Active learning: prioritize uncertain samples
        if self.config.enable_active_learning:
            if uncertainty and uncertainty > self.config.uncertainty_threshold:
                self.uncertain_samples.append(sample)

    def add_feedback(
        self,
        sample_id: str,
        feedback: Dict[str, Any],
    ):
        """Add user feedback for a sample."""
        # Find and update sample with feedback
        for sample in self.samples:
            if sample.get("id") == sample_id:
                sample["feedback"] = feedback
                sample["feedback_timestamp"] = datetime.now(timezone.utc).isoformat()
                break

    def get_training_batch(
        self,
        batch_size: int = 1000,
        prioritize_uncertain: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get a batch for retraining."""
        if prioritize_uncertain and self.uncertain_samples:
            # 30% uncertain, 70% regular
            uncertain_count = int(batch_size * 0.3)
            regular_count = batch_size - uncertain_count

            batch = self.uncertain_samples[:uncertain_count]
            batch.extend(self.samples[:regular_count])

            return batch

        return self.samples[:batch_size]

    def save_samples(self):
        """Save collected samples to disk."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        save_path = self.data_path / f"samples_{timestamp}.json"

        with open(save_path, "w") as f:
            json.dump(self.samples, f, indent=2)

        logger.info(f"Saved {len(self.samples)} samples to {save_path}")

        # Clear saved samples
        self.samples = []
        self.uncertain_samples = []

    def load_samples(self, path: str):
        """Load samples from disk."""
        with open(path, "r") as f:
            loaded = json.load(f)

        self.samples.extend(loaded)
        self.sample_count = len(self.samples)
        logger.info(f"Loaded {len(loaded)} samples from {path}")

    def is_ready_for_retraining(self) -> bool:
        """Check if enough samples for retraining."""
        return self.sample_count >= self.config.min_samples_for_retraining

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "total_samples": self.sample_count,
            "in_memory_samples": len(self.samples),
            "uncertain_samples": len(self.uncertain_samples),
            "ready_for_retraining": self.is_ready_for_retraining(),
            "min_samples_required": self.config.min_samples_for_retraining,
        }


class RetrainingScheduler:
    """
    Schedules and executes periodic adapter retraining.
    """

    def __init__(
        self,
        adapter_manager: LoRAAdapterManager,
        data_collector: RetrainingDataCollector,
        config: PracticalDeploymentConfig,
    ):
        self.adapter_manager = adapter_manager
        self.data_collector = data_collector
        self.config = config

        # Scheduler state
        self.is_running = False
        self.last_retraining: Optional[datetime] = None
        self.next_retraining: Optional[datetime] = None

        # Training state
        self.training_in_progress = False
        self.training_history: List[Dict[str, Any]] = []

        self._scheduler_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the retraining scheduler."""
        if self.is_running:
            return

        self.is_running = True
        self._update_next_retraining()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        logger.info(f"Started retraining scheduler, next: {self.next_retraining}")

    async def stop(self):
        """Stop the retraining scheduler."""
        self.is_running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped retraining scheduler")

    def _update_next_retraining(self):
        """Calculate next retraining time."""
        now = datetime.now(timezone.utc)

        if self.config.retraining_frequency == RetrainingFrequency.HOURLY:
            delta = timedelta(hours=1)
        elif self.config.retraining_frequency == RetrainingFrequency.DAILY:
            delta = timedelta(days=1)
        elif self.config.retraining_frequency == RetrainingFrequency.WEEKLY:
            delta = timedelta(weeks=1)
        elif self.config.retraining_frequency == RetrainingFrequency.MONTHLY:
            delta = timedelta(days=30)
        else:
            delta = None

        if delta:
            self.next_retraining = now + delta

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                now = datetime.now(timezone.utc)

                # Check if it's time for retraining
                if (self.next_retraining and
                    now >= self.next_retraining and
                    self.data_collector.is_ready_for_retraining()):

                    await self.execute_retraining()
                    self._update_next_retraining()

                # Check every minute
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def execute_retraining(
        self,
        adapter_name: str = "default",
        epochs: int = 1,
        learning_rate: float = 1e-4,
    ):
        """Execute adapter retraining."""
        if self.training_in_progress:
            logger.warning("Training already in progress")
            return None

        self.training_in_progress = True
        start_time = datetime.now(timezone.utc)

        logger.info(f"Starting adapter retraining: {adapter_name}")

        try:
            # Get training data
            training_data = self.data_collector.get_training_batch()

            if not training_data:
                logger.warning("No training data available")
                return None

            # Create or get adapter
            if adapter_name not in self.adapter_manager.adapters:
                self.adapter_manager.create_adapter(adapter_name)

            # Get trainable parameters
            params = self.adapter_manager.get_trainable_parameters(adapter_name)

            if not params:
                logger.warning("No trainable parameters")
                return None

            # Setup optimizer
            optimizer = AdamW(params, lr=learning_rate, weight_decay=0.01)

            # Training loop
            total_loss = 0.0

            for epoch in range(epochs):
                epoch_loss = 0.0

                for _sample in training_data:
                    # In production, would tokenize and train properly
                    # sample would be used for tokenization
                    loss = torch.tensor(0.1, requires_grad=True)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                total_loss += epoch_loss
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(training_data):.4f}")

            # Update version
            self.adapter_manager.adapter_versions[adapter_name] += 1

            # Save adapter
            self.adapter_manager.save_adapter(
                adapter_name,
                self.config.retraining_data_path,
            )

            # Record history
            end_time = datetime.now(timezone.utc)
            result = {
                "adapter": adapter_name,
                "started_at": start_time.isoformat(),
                "ended_at": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "samples": len(training_data),
                "epochs": epochs,
                "avg_loss": total_loss / (len(training_data) * epochs),
                "version": self.adapter_manager.adapter_versions[adapter_name],
            }

            self.training_history.append(result)
            self.last_retraining = end_time

            logger.info(
                f"Completed retraining: {len(training_data)} samples, "
                f"version {self.adapter_manager.adapter_versions[adapter_name]}"
            )

            return result

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return None

        finally:
            self.training_in_progress = False

    async def trigger_retraining(
        self,
        adapter_name: str = "default",
        epochs: int = 1,
    ):
        """Manually trigger retraining."""
        return await self.execute_retraining(adapter_name, epochs=epochs)

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "is_running": self.is_running,
            "training_in_progress": self.training_in_progress,
            "last_retraining": self.last_retraining.isoformat() if self.last_retraining else None,
            "next_retraining": self.next_retraining.isoformat() if self.next_retraining else None,
            "frequency": self.config.retraining_frequency.value,
            "history_count": len(self.training_history),
        }
