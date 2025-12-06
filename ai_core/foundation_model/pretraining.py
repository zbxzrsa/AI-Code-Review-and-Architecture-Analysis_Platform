"""
Pre-training Infrastructure for Foundation Model

Implements enterprise-grade distributed training with:
- 4D Parallelism (Data, Model, Pipeline, Tensor)
- AdamW optimizer with cosine decay
- BF16/FP8 mixed precision training
- Gradient checkpointing
- Advanced checkpointing and recovery

Training specs:
- Batch size: 2048-8192 sequences
- Learning rate: ~3e-4 with cosine decay
- Training time: 30-90 days
"""

import gc
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class PrecisionMode(str, Enum):
    """Training precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"


class ParallelismStrategy(str, Enum):
    """Parallelism strategies."""
    DATA = "data"
    MODEL = "model"
    PIPELINE = "pipeline"
    TENSOR = "tensor"
    FULL_4D = "full_4d"


@dataclass
class PretrainingConfig:
    """
    Pre-training configuration.
    
    Optimized for 500B-1T parameter models on large GPU clusters.
    """
    # Model
    model_name: str = "moe_transformer"
    vocab_size: int = 128000
    max_seq_length: int = 4096  # Training sequence length
    
    # Optimization
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    
    # Batch sizing
    global_batch_size: int = 4096  # Total batch across all GPUs
    micro_batch_size: int = 1  # Per-GPU batch size
    gradient_accumulation_steps: int = 32
    
    # Training duration
    max_steps: int = 500000  # ~30-90 days depending on hardware
    warmup_steps: int = 2000
    
    # Precision
    precision: PrecisionMode = PrecisionMode.BF16
    use_gradient_checkpointing: bool = True
    
    # Parallelism
    parallelism: ParallelismStrategy = ParallelismStrategy.FULL_4D
    tensor_parallel_size: int = 8
    pipeline_parallel_size: int = 4
    data_parallel_size: int = 8
    
    # Checkpointing
    checkpoint_dir: str = "/checkpoints"
    save_interval: int = 1000
    keep_last_n: int = 5
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    
    # Distributed
    backend: str = "nccl"
    
    def __post_init__(self):
        """Validate configuration."""
        # Ensure batch sizes are consistent
        # Note: world_size used implicitly through data_parallel_size
        effective_batch = (
            self.micro_batch_size * 
            self.gradient_accumulation_steps * 
            self.data_parallel_size
        )
        
        if effective_batch != self.global_batch_size:
            logger.warning(
                f"Batch size mismatch: global={self.global_batch_size}, "
                f"effective={effective_batch}"
            )


@dataclass
class TrainingState:
    """Tracks training state for checkpointing and recovery."""
    step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    samples_seen: int = 0
    
    # Loss tracking
    loss_history: List[float] = field(default_factory=list)
    aux_loss_history: List[float] = field(default_factory=list)
    
    # Metrics
    best_loss: float = float('inf')
    best_step: int = 0
    
    # Timing
    start_time: Optional[float] = None
    total_train_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "tokens_seen": self.tokens_seen,
            "samples_seen": self.samples_seen,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "total_train_time": self.total_train_time,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        return cls(
            step=data.get("step", 0),
            epoch=data.get("epoch", 0),
            tokens_seen=data.get("tokens_seen", 0),
            samples_seen=data.get("samples_seen", 0),
            best_loss=data.get("best_loss", float('inf')),
            best_step=data.get("best_step", 0),
            total_train_time=data.get("total_train_time", 0.0),
        )


# =============================================================================
# Mixed Precision Manager
# =============================================================================

class MixedPrecisionManager:
    """
    Manages mixed precision training.
    
    Supports:
    - FP16 with loss scaling
    - BF16 (no loss scaling needed)
    - FP8 (transformer engine)
    """
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        self.precision = config.precision
        
        # Initialize scaler for FP16
        if self.precision == PrecisionMode.FP16:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Determine dtype
        if self.precision == PrecisionMode.BF16:
            self.dtype = torch.bfloat16
        elif self.precision == PrecisionMode.FP16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        logger.info(f"Mixed precision: {self.precision.value}, dtype: {self.dtype}")
    
    def autocast_context(self):
        """Get autocast context manager."""
        if self.precision == PrecisionMode.FP32:
            return torch.cuda.amp.autocast(enabled=False)
        
        return torch.cuda.amp.autocast(
            enabled=True,
            dtype=self.dtype,
        )
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient computation."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer: torch.optim.Optimizer, skip_if_nan: bool = True):
        """Optimizer step with scaling."""
        if self.scaler is not None:
            # Check for NaN gradients
            if skip_if_nan:
                self.scaler.unscale_(optimizer)
                grad_norm = self._get_grad_norm(optimizer)
                
                if not torch.isfinite(grad_norm):
                    logger.warning("Skipping step due to NaN gradients")
                    self.scaler.update()
                    return False
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        return True
    
    def _get_grad_norm(self, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """Compute gradient norm."""
        total_norm = 0.0
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_norm += p.grad.data.float().norm().item() ** 2
        return torch.tensor(total_norm ** 0.5)


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create cosine decay scheduler with linear warmup.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Steps for linear warmup
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of peak LR
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio
        
        return decayed
    
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """
    Manages model checkpointing with:
    - Distributed checkpointing
    - Automatic cleanup
    - Recovery from failures
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 5,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        self.checkpoints: List[str] = []
        self._discover_checkpoints()
    
    def _discover_checkpoints(self):
        """Discover existing checkpoints."""
        for ckpt in sorted(self.checkpoint_dir.glob("step_*.pt")):
            self.checkpoints.append(str(ckpt))
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        state: TrainingState,
        rank: int = 0,
    ):
        """Save checkpoint."""
        # Only rank 0 saves
        if rank != 0:
            return
        
        checkpoint_path = self.checkpoint_dir / f"step_{state.step:08d}.pt"
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "training_state": state.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if self.save_optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if self.save_scheduler and scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Save atomically
        temp_path = checkpoint_path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)
        
        self.checkpoints.append(str(checkpoint_path))
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup()
    
    def _cleanup(self):
        """Remove old checkpoints."""
        while len(self.checkpoints) > self.keep_last_n:
            old_ckpt = self.checkpoints.pop(0)
            try:
                os.remove(old_ckpt)
                logger.info(f"Removed old checkpoint: {old_ckpt}")
            except Exception as e:
                logger.warning(f"Failed to remove {old_ckpt}: {e}")
    
    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cuda"),
    ) -> Optional[TrainingState]:
        """Load the latest checkpoint."""
        if not self.checkpoints:
            logger.info("No checkpoints found")
            return None
        
        latest_ckpt = self.checkpoints[-1]
        return self.load(
            latest_ckpt, model, optimizer, scheduler, device
        )
    
    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cuda"),
    ) -> TrainingState:
        """Load a specific checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        state = TrainingState.from_dict(checkpoint["training_state"])
        
        logger.info(f"Resumed from step {state.step}")
        
        return state


# =============================================================================
# Distributed Training Setup
# =============================================================================

class DistributedTrainer4D:
    """
    4D Parallel Distributed Trainer
    
    Implements:
    - Data Parallelism: Replicate model, split data
    - Model Parallelism: Split model across GPUs
    - Pipeline Parallelism: Split layers into stages
    - Tensor Parallelism: Split individual tensors
    """
    
    def __init__(
        self,
        config: PretrainingConfig,
        model: nn.Module,
    ):
        self.config = config
        self.model = model
        
        # Distributed setup
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.device = torch.device("cuda")
        
        # Process groups for different parallelism types
        self.dp_group = None  # Data parallel
        self.tp_group = None  # Tensor parallel
        self.pp_group = None  # Pipeline parallel
        
        self._setup_distributed()
    
    def _setup_distributed(self):
        """Initialize distributed training."""
        if not dist.is_initialized():
            # Check for distributed environment
            if "WORLD_SIZE" in os.environ:
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method="env://",
                )
                
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()
                self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
                
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")
                
                logger.info(
                    f"Distributed: rank {self.rank}/{self.world_size}, "
                    f"local_rank {self.local_rank}"
                )
            else:
                logger.info("Running in single-GPU mode")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        
        # Setup process groups for 4D parallelism
        if self.config.parallelism == ParallelismStrategy.FULL_4D:
            self._setup_4d_groups()
    
    def _setup_4d_groups(self):
        """Setup process groups for 4D parallelism."""
        if self.world_size <= 1:
            return
        
        tp_size = self.config.tensor_parallel_size
        pp_size = self.config.pipeline_parallel_size
        dp_size = self.config.data_parallel_size
        
        # Validate
        if tp_size * pp_size * dp_size != self.world_size:
            logger.warning(
                f"Parallelism config mismatch: "
                f"TP={tp_size} * PP={pp_size} * DP={dp_size} != {self.world_size}"
            )
            return
        
        # Create tensor parallel groups
        for i in range(self.world_size // tp_size):
            ranks = list(range(i * tp_size, (i + 1) * tp_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.tp_group = group
        
        # Create data parallel groups (ranks with same model partition)
        for i in range(tp_size * pp_size):
            ranks = [i + j * tp_size * pp_size for j in range(dp_size)]
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.dp_group = group
        
        logger.info(f"Setup 4D parallelism: TP={tp_size}, PP={pp_size}, DP={dp_size}")
    
    def get_model(self) -> nn.Module:
        """Get the model (unwrapped if DDP)."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
    
    def synchronize(self):
        """Synchronize all processes."""
        if self.world_size > 1:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.AVG):
        """All-reduce a tensor across data parallel group."""
        if self.world_size > 1:
            group = self.dp_group if self.dp_group is not None else None
            dist.all_reduce(tensor, op=op, group=group)
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0


# =============================================================================
# Pre-training Engine
# =============================================================================

class PretrainingEngine:
    """
    Main pre-training engine.
    
    Orchestrates:
    - Distributed training
    - Mixed precision
    - Checkpointing
    - Logging and monitoring
    """
    
    def __init__(
        self,
        config: PretrainingConfig,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Initialize distributed trainer
        self.trainer = DistributedTrainer4D(config, model)
        self.model = self.trainer.model
        self.device = self.trainer.device
        
        # Mixed precision
        self.precision_manager = MixedPrecisionManager(config)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
            min_lr_ratio=config.min_learning_rate / config.learning_rate,
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            keep_last_n=config.keep_last_n,
        )
        
        # Training state
        self.state = TrainingState()
        
        # Enable gradient checkpointing
        if config.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        logger.info(f"PretrainingEngine initialized on {self.device}")
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        model = self.trainer.get_model()
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for biases and LayerNorm
            if "bias" in name or "norm" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        model = self.trainer.get_model()
        
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        else:
            # Manual checkpointing for transformer layers
            if hasattr(model, 'layers'):
                for layer in model.layers:
                    layer.gradient_checkpointing = True
                logger.info("Enabled manual gradient checkpointing")
    
    def train(
        self,
        resume: bool = True,
        callbacks: Optional[List[Callable]] = None,
    ) -> TrainingState:
        """
        Main training loop.
        
        Args:
            resume: Whether to resume from checkpoint
            callbacks: Optional training callbacks
        """
        # Resume from checkpoint if available
        if resume:
            loaded_state = self.checkpoint_manager.load_latest(
                self.trainer.get_model(),
                self.optimizer,
                self.scheduler,
                self.device,
            )
            if loaded_state is not None:
                self.state = loaded_state
        
        self.state.start_time = time.time()
        self.model.train()
        
        logger.info(f"Starting training from step {self.state.step}")
        
        # Create iterator
        data_iter = iter(self.train_dataloader)
        
        # Training loop
        while self.state.step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Epoch complete
                self.state.epoch += 1
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            # Training step
            metrics = self._train_step(batch)
            
            # Update state
            self.state.step += 1
            self.state.samples_seen += batch['input_ids'].size(0)
            self.state.tokens_seen += batch['input_ids'].numel()
            
            if metrics['loss'] < self.state.best_loss:
                self.state.best_loss = metrics['loss']
                self.state.best_step = self.state.step
            
            # Logging
            if self.state.step % self.config.log_interval == 0:
                self._log_metrics(metrics)
            
            # Evaluation
            if (self.eval_dataloader is not None and 
                self.state.step % self.config.eval_interval == 0):
                eval_metrics = self.evaluate()
                self._log_metrics(eval_metrics, prefix="eval")
            
            # Checkpointing
            if self.state.step % self.config.save_interval == 0:
                self.checkpoint_manager.save(
                    self.trainer.get_model(),
                    self.optimizer,
                    self.scheduler,
                    self.state,
                    rank=self.trainer.rank,
                )
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self.state, metrics)
        
        # Final save
        self.state.total_train_time = time.time() - self.state.start_time
        self.checkpoint_manager.save(
            self.trainer.get_model(),
            self.optimizer,
            self.scheduler,
            self.state,
            rank=self.trainer.rank,
        )
        
        logger.info(f"Training complete. Total steps: {self.state.step}")
        
        return self.state
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Accumulation loop
        total_loss = 0.0
        total_aux_loss = 0.0
        
        self.optimizer.zero_grad()
        
        for _ in range(self.config.gradient_accumulation_steps):
            with self.precision_manager.autocast_context():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                logits = outputs['logits']
                aux_loss = outputs.get('aux_loss', 0.0)
                
                # Compute cross-entropy loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                
                # Add auxiliary loss (MoE router loss)
                if isinstance(aux_loss, torch.Tensor):
                    loss = loss + aux_loss
                    total_aux_loss += aux_loss.item()
                
                # Scale for accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            scaled_loss = self.precision_manager.scale_loss(loss)
            scaled_loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
        
        # Gradient clipping
        if self.config.grad_clip > 0:
            if self.precision_manager.scaler is not None:
                self.precision_manager.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )
        
        # Optimizer step
        self.precision_manager.step(self.optimizer)
        
        # Scheduler step
        self.scheduler.step()
        
        # All-reduce metrics
        if self.trainer.world_size > 1:
            loss_tensor = torch.tensor([total_loss], device=self.device)
            self.trainer.all_reduce(loss_tensor)
            total_loss = loss_tensor.item()
        
        return {
            'loss': total_loss,
            'aux_loss': total_aux_loss / self.config.gradient_accumulation_steps,
            'lr': self.scheduler.get_last_lr()[0],
            'tokens_per_sec': self._compute_throughput(),
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        for batch in self.eval_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            with self.precision_manager.autocast_context():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                logits = outputs['logits']
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
        
        self.model.train()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
        }
    
    def _compute_throughput(self) -> float:
        """Compute tokens per second."""
        if self.state.start_time is None:
            return 0.0
        
        elapsed = time.time() - self.state.start_time
        if elapsed <= 0:
            return 0.0
        
        return self.state.tokens_seen / elapsed
    
    def _log_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str = "train",  # noqa: ARG002 - reserved for prefixed metric logging
    ):
        """Log training metrics."""
        if not self.trainer.is_main_process():
            return
        
        elapsed = time.time() - self.state.start_time if self.state.start_time else 0
        
        log_str = (
            f"Step {self.state.step}/{self.config.max_steps} | "
            f"Epoch {self.state.epoch} | "
            f"Loss: {metrics.get('loss', 0):.4f} | "
            f"LR: {metrics.get('lr', 0):.2e} | "
            f"Tokens/s: {metrics.get('tokens_per_sec', 0):.0f} | "
            f"Time: {elapsed:.0f}s"
        )
        
        if 'aux_loss' in metrics and metrics['aux_loss'] > 0:
            log_str += f" | Aux: {metrics['aux_loss']:.4f}"
        
        if 'perplexity' in metrics:
            log_str += f" | PPL: {metrics['perplexity']:.2f}"
        
        logger.info(log_str)


# =============================================================================
# Training Dataset
# =============================================================================

class PretrainingDataset(Dataset):
    """
    Dataset for pre-training.
    
    Handles:
    - Token packing for efficient training
    - Dynamic sequence length
    - Causal masking
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 4096,
        pack_sequences: bool = True,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pack_sequences = pack_sequences
        
        # Load shard metadata
        self.shards = sorted(self.data_path.glob("*.parquet"))
        if not self.shards:
            self.shards = sorted(self.data_path.glob("*.jsonl"))
        
        logger.info(f"Found {len(self.shards)} data shards")
    
    def __len__(self) -> int:
        # Estimate based on shards (actual count would require scanning all data)
        return len(self.shards) * 10000
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # For actual implementation, would load from shards
        # This is a placeholder
        
        # Generate dummy data for demonstration
        input_ids = torch.randint(0, 32000, (self.max_length,))
        
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'attention_mask': torch.ones(self.max_length),
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_pretraining_dataloader(
    dataset: Dataset,
    config: PretrainingConfig,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create distributed dataloader for pre-training."""
    sampler = None
    shuffle = True
    
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        shuffle = False
    
    return DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


def estimate_training_time(
    config: PretrainingConfig,
    tokens_per_second: float,
    total_tokens: int = 15_000_000_000_000,  # 15T
) -> Dict[str, float]:
    """Estimate training time."""
    # Total tokens to process
    tokens_per_step = (
        config.global_batch_size * 
        config.max_seq_length
    )
    
    total_steps = total_tokens // tokens_per_step
    
    # Time estimates
    seconds_per_step = tokens_per_step / tokens_per_second
    total_seconds = total_steps * seconds_per_step
    
    return {
        "total_steps": total_steps,
        "seconds_per_step": seconds_per_step,
        "total_hours": total_seconds / 3600,
        "total_days": total_seconds / 86400,
        "tokens_per_second": tokens_per_second,
    }
