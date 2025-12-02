"""
Distributed Training and Model Parallelism
Supports multi-GPU and multi-node training
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = 'nccl'
    master_addr: str = 'localhost'
    master_port: str = '12355'
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True


class DistributedTrainer:
    """
    Distributed Training System
    
    Features:
    - Data parallel training
    - Automatic mixed precision
    - Gradient accumulation
    - Checkpoint management
    - Dynamic batch sizing
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DistributedConfig] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        """
        Initialize Distributed Trainer
        
        Args:
            model: Model to train
            config: Distributed configuration
            use_amp: Whether to use automatic mixed precision
            gradient_accumulation_steps: Steps to accumulate gradients
        """
        self.config = config or DistributedConfig()
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.is_distributed = self.config.world_size > 1
        self.is_main_process = self.config.rank == 0
        
        # Initialize distributed if needed
        if self.is_distributed:
            self._setup_distributed()
        
        # Setup device
        if torch.cuda.is_available():
            if self.is_distributed:
                self.device = torch.device(f'cuda:{self.config.local_rank}')
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=self.config.find_unused_parameters,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view
            )
        
        # AMP scaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def _setup_distributed(self) -> None:
        """Setup distributed training environment"""
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        
        dist.init_process_group(
            backend=self.config.backend,
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        logger.info(
            f"Initialized distributed training: "
            f"rank={self.config.rank}, world_size={self.config.world_size}"
        )
    
    def get_data_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """Get distributed data loader"""
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=shuffle
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        scheduler: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            loss_fn: Loss function
            scheduler: Optional learning rate scheduler
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Set epoch for distributed sampler
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(self.epoch)
        
        optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
                
                self.global_step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
        
        self.epoch += 1
        
        avg_loss = total_loss / num_batches
        
        # Reduce loss across all processes
        if self.is_distributed:
            loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        return {
            'loss': avg_loss,
            'epoch': self.epoch,
            'global_step': self.global_step
        }
    
    def evaluate(
        self,
        eval_loader: DataLoader,
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            eval_loader: Evaluation data loader
            metric_fn: Optional metric function
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                if metric_fn:
                    # Custom metric
                    correct = metric_fn(outputs, targets)
                else:
                    # Default accuracy
                    _, predicted = outputs.max(1)
                    correct = (predicted == targets).sum().item()
                
                total_correct += correct
                total_samples += targets.size(0)
        
        accuracy = total_correct / total_samples
        
        # Reduce across processes
        if self.is_distributed:
            acc_tensor = torch.tensor([total_correct, total_samples]).to(self.device)
            dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
            accuracy = acc_tensor[0].item() / acc_tensor[1].item()
        
        return {'accuracy': accuracy}
    
    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict] = None
    ) -> None:
        """Save training checkpoint"""
        if not self.is_main_process:
            return
        
        # Get model state (unwrap DDP if needed)
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'epoch': self.epoch,
            'global_step': self.global_step
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if extra_state:
            checkpoint.update(extra_state)
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        model = self.model.module if self.is_distributed else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {path}")
        
        return checkpoint
    
    def cleanup(self) -> None:
        """Cleanup distributed resources"""
        if self.is_distributed:
            dist.destroy_process_group()


class ModelParallel:
    """
    Model Parallelism for Large Models
    
    Features:
    - Pipeline parallelism
    - Tensor parallelism
    - Memory-efficient training
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_gpus: int = 2,
        split_size: int = 2
    ):
        """
        Initialize Model Parallel
        
        Args:
            model: Model to parallelize
            num_gpus: Number of GPUs to use
            split_size: Micro-batch size for pipeline
        """
        self.num_gpus = num_gpus
        self.split_size = split_size
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for model parallelism")
        
        self.devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
        
        # Split model across devices
        self.model_parts = self._split_model(model)
    
    def _split_model(self, model: nn.Module) -> List[nn.Module]:
        """Split model into parts for each device"""
        # This is a simplified implementation
        # Real implementation would need model-specific splitting
        
        layers = list(model.children())
        n_layers = len(layers)
        layers_per_device = n_layers // self.num_gpus
        
        parts = []
        for i in range(self.num_gpus):
            start = i * layers_per_device
            end = start + layers_per_device if i < self.num_gpus - 1 else n_layers
            
            part = nn.Sequential(*layers[start:end])
            part.to(self.devices[i])
            parts.append(part)
        
        return parts
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pipeline parallelism
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        splits = x.split(self.split_size, dim=0)
        outputs = []
        
        for split in splits:
            # Move through pipeline
            h = split.to(self.devices[0])
            
            for i, part in enumerate(self.model_parts):
                h = part(h)
                if i < len(self.model_parts) - 1:
                    h = h.to(self.devices[i + 1])
            
            outputs.append(h)
        
        # Concatenate outputs on first device
        return torch.cat([o.to(self.devices[0]) for o in outputs], dim=0)
    
    def parameters(self):
        """Get all model parameters"""
        for part in self.model_parts:
            yield from part.parameters()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get combined state dict"""
        state = {}
        for i, part in enumerate(self.model_parts):
            for name, param in part.state_dict().items():
                state[f'part_{i}.{name}'] = param.cpu()
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load combined state dict"""
        for i, part in enumerate(self.model_parts):
            part_state = {
                name.replace(f'part_{i}.', ''): param.to(self.devices[i])
                for name, param in state.items()
                if name.startswith(f'part_{i}.')
            }
            part.load_state_dict(part_state)


def launch_distributed_training(
    train_fn: Callable,
    world_size: int,
    args: Tuple = ()
) -> None:
    """
    Launch distributed training across multiple processes
    
    Args:
        train_fn: Training function to run
        world_size: Number of processes/GPUs
        args: Arguments to pass to train_fn
    """
    mp.spawn(
        _distributed_worker,
        args=(train_fn, world_size, args),
        nprocs=world_size,
        join=True
    )


def _distributed_worker(
    rank: int,
    train_fn: Callable,
    world_size: int,
    args: Tuple
) -> None:
    """Worker function for distributed training"""
    config = DistributedConfig(
        world_size=world_size,
        rank=rank,
        local_rank=rank
    )
    
    # Run training function
    train_fn(config, *args)


@contextmanager
def distributed_context(config: DistributedConfig):
    """Context manager for distributed training"""
    if config.world_size > 1:
        os.environ['MASTER_ADDR'] = config.master_addr
        os.environ['MASTER_PORT'] = config.master_port
        
        dist.init_process_group(
            backend=config.backend,
            world_size=config.world_size,
            rank=config.rank
        )
    
    try:
        yield
    finally:
        if config.world_size > 1:
            dist.destroy_process_group()
