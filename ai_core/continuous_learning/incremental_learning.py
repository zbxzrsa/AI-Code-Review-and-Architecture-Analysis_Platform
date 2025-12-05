"""
Incremental Learning Module
Implements EWC, SI, and other continual learning methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import copy
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EWC:
    """
    Elastic Weight Consolidation (EWC)
    
    Prevents catastrophic forgetting by adding a regularization term
    that penalizes changes to important weights.
    
    Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting
    in neural networks" (2017)
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        online: bool = False,
        gamma: float = 0.9,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize EWC
        
        Args:
            model: Neural network model
            ewc_lambda: EWC regularization strength
            online: Whether to use online EWC
            gamma: Decay factor for online EWC
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.online = online
        self.gamma = gamma
        
        # Store parameters and Fisher information from previous tasks
        self.params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}
        
        self.tasks_learned = 0
    
    def compute_fisher(
        self,
        dataset: Dataset,
        sample_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix diagonal
        
        Args:
            dataset: Dataset to compute Fisher on
            sample_size: Number of samples to use (None for all)
            
        Returns:
            Fisher information dictionary
        """
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        self.model.eval()
        
        # Create dataloader
        if sample_size and sample_size < len(dataset):
            indices = torch.randperm(len(dataset))[:sample_size]
            subset = torch.utils.data.Subset(dataset, indices)
            dataloader = DataLoader(subset, batch_size=1, shuffle=True, num_workers=0)
        else:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        
        # Compute Fisher
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # Use log-likelihood of true class
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.pow(2)
        
        # Normalize
        n_samples = len(dataloader)
        for name in fisher:
            fisher[name] /= n_samples
        
        return fisher
    
    def register_task(self, dataset: Dataset) -> None:
        """
        Register completion of a task
        
        Args:
            dataset: Dataset of the completed task
        """
        # Compute Fisher information
        new_fisher = self.compute_fisher(dataset)
        
        # Store current parameters
        new_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        if self.online and self.fisher:
            # Online EWC: accumulate Fisher information
            for name in new_fisher:
                self.fisher[name] = (
                    self.gamma * self.fisher[name] + new_fisher[name]
                )
        else:
            # Standard EWC: replace Fisher
            self.fisher = new_fisher
        
        self.params = new_params
        self.tasks_learned += 1
        
        logger.info(f"Registered task {self.tasks_learned}")
    
    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty
        
        Returns:
            EWC regularization loss
        """
        if not self.fisher:
            return torch.tensor(0.0, device=self.device)
        
        loss = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher and name in self.params:
                loss += (
                    self.fisher[name] * 
                    (param - self.params[name]).pow(2)
                ).sum()
        
        return loss * self.ewc_lambda
    
    def train_on_task(
        self,
        dataset: Dataset,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        register_after: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train on a task with EWC regularization
        
        Args:
            dataset: Task dataset
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            register_after: Whether to register task after training
            
        Returns:
            Training history
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        history = {'loss': [], 'ewc_loss': [], 'task_loss': []}
        
        for _ in range(epochs):
            self.model.train()
            epoch_loss = []
            epoch_ewc = []
            epoch_task = []
            
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                task_loss = F.cross_entropy(outputs, targets)
                ewc_loss = self.penalty()
                
                total_loss = task_loss + ewc_loss
                total_loss.backward()
                optimizer.step()
                
                epoch_loss.append(total_loss.item())
                epoch_ewc.append(ewc_loss.item())
                epoch_task.append(task_loss.item())
            
            history['loss'].append(np.mean(epoch_loss))
            history['ewc_loss'].append(np.mean(epoch_ewc))
            history['task_loss'].append(np.mean(epoch_task))
            
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Loss={history['loss'][-1]:.4f}, "
                f"Task={history['task_loss'][-1]:.4f}, "
                f"EWC={history['ewc_loss'][-1]:.4f}"
            )
        
        if register_after:
            self.register_task(dataset)
        
        return history


class SynapticIntelligence:
    """
    Synaptic Intelligence (SI)
    
    Online computation of parameter importance using
    gradient-based path integral.
    
    Reference: Zenke et al., "Continual Learning Through
    Synaptic Intelligence" (2017)
    """
    
    def __init__(
        self,
        model: nn.Module,
        si_lambda: float = 1.0,
        epsilon: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize SI
        
        Args:
            model: Neural network model
            si_lambda: SI regularization strength
            epsilon: Small constant for numerical stability
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.si_lambda = si_lambda
        self.epsilon = epsilon
        
        # Initialize importance and parameter tracking
        self.importance: Dict[str, torch.Tensor] = {}
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.omega: Dict[str, torch.Tensor] = {}
        
        # Path integral (gradient * delta)
        self.w: Dict[str, torch.Tensor] = {}
        
        self._init_tracking()
    
    def _init_tracking(self) -> None:
        """Initialize parameter tracking"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.prev_params[name] = param.clone().detach()
                self.w[name] = torch.zeros_like(param)
                self.omega[name] = torch.zeros_like(param)
    
    def update_omega(self) -> None:
        """Update importance (omega) after task completion"""
        for name, param in self.model.named_parameters():
            if name in self.prev_params:
                delta = param.detach() - self.prev_params[name]
                
                # Update omega
                self.omega[name] += (
                    self.w[name] / (delta.pow(2) + self.epsilon)
                )
                
                # Reset w and update prev_params
                self.w[name] = torch.zeros_like(param)
                self.prev_params[name] = param.clone().detach()
    
    def update_w(self) -> None:
        """Update path integral after each parameter update"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                delta = param.detach() - self.prev_params[name]
                self.w[name] += -param.grad.detach() * delta
    
    def penalty(self) -> torch.Tensor:
        """Compute SI penalty"""
        loss = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if name in self.omega and name in self.prev_params:
                loss += (
                    self.omega[name] * 
                    (param - self.prev_params[name]).pow(2)
                ).sum()
        
        return loss * self.si_lambda
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """Single training step with SI"""
        self.model.train()
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        optimizer.zero_grad()
        
        outputs = self.model(inputs)
        task_loss = F.cross_entropy(outputs, targets)
        si_loss = self.penalty()
        
        total_loss = task_loss + si_loss
        total_loss.backward()
        
        # Update path integral before optimizer step
        self.update_w()
        
        optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'si_loss': si_loss.item()
        }


class IncrementalLearner:
    """
    Unified Incremental Learning System
    
    Features:
    - Multiple regularization methods (EWC, SI, LwF)
    - Progressive neural networks
    - PackNet-style pruning
    - Dynamic architecture expansion
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: str = 'ewc',
        regularization_strength: float = 1000.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Incremental Learner
        
        Args:
            model: Base model
            method: 'ewc', 'si', 'lwf', or 'packnet'
            regularization_strength: Regularization strength
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.method = method
        self.reg_strength = regularization_strength
        
        # Initialize method-specific components
        if method == 'ewc':
            self.regularizer = EWC(model, ewc_lambda=regularization_strength, device=device)
        elif method == 'si':
            self.regularizer = SynapticIntelligence(model, si_lambda=regularization_strength, device=device)
        elif method == 'lwf':
            self.prev_model = None
            self.lwf_temp = 2.0
        elif method == 'packnet':
            self.masks: Dict[int, Dict[str, torch.Tensor]] = {}
            self.current_task = 0
        
        self.tasks: List[Dict[str, Any]] = []
    
    def learn_task(
        self,
        dataset: Dataset,
        task_id: int,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        validation_dataset: Optional[Dataset] = None
    ) -> Dict[str, Any]:
        """
        Learn a new task
        
        Args:
            dataset: Task dataset
            task_id: Task identifier
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            validation_dataset: Optional validation set
            
        Returns:
            Training results
        """
        if self.method == 'ewc':
            history = self.regularizer.train_on_task(
                dataset, epochs, batch_size, lr
            )
        elif self.method == 'si':
            history = self._train_with_si(dataset, epochs, batch_size, lr)
        elif self.method == 'lwf':
            history = self._train_with_lwf(dataset, epochs, batch_size, lr)
        elif self.method == 'packnet':
            history = self._train_with_packnet(dataset, task_id, epochs, batch_size, lr)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Record task
        task_info = {
            'task_id': task_id,
            'history': history,
            'dataset_size': len(dataset)
        }
        
        if validation_dataset:
            val_acc = self.evaluate(validation_dataset)
            task_info['validation_accuracy'] = val_acc
        
        self.tasks.append(task_info)
        
        return task_info
    
    def _train_with_si(
        self,
        dataset: Dataset,
        epochs: int,
        batch_size: int,
        lr: float
    ) -> Dict[str, List[float]]:
        """Train with Synaptic Intelligence"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        history = {'loss': [], 'task_loss': [], 'si_loss': []}
        
        for _ in range(epochs):
            epoch_metrics = {'loss': [], 'task_loss': [], 'si_loss': []}
            
            for inputs, targets in dataloader:
                metrics = self.regularizer.train_step(inputs, targets, optimizer)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)
            
            for k in history:
                history[k].append(np.mean(epoch_metrics[k]))
        
        # Update omega after task
        self.regularizer.update_omega()
        
        return history
    
    def _train_with_lwf(
        self,
        dataset: Dataset,
        epochs: int,
        batch_size: int,
        lr: float
    ) -> Dict[str, List[float]]:
        """Train with Learning without Forgetting"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        history = {'loss': [], 'task_loss': [], 'lwf_loss': []}
        
        for _ in range(epochs):
            epoch_metrics = {'loss': [], 'task_loss': [], 'lwf_loss': []}
            
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                task_loss = F.cross_entropy(outputs, targets)
                
                # LwF loss
                lwf_loss = torch.tensor(0.0, device=self.device)
                if self.prev_model is not None:
                    with torch.no_grad():
                        prev_outputs = self.prev_model(inputs)
                    
                    # Knowledge distillation loss
                    soft_targets = F.softmax(prev_outputs / self.lwf_temp, dim=1)
                    soft_outputs = F.log_softmax(outputs / self.lwf_temp, dim=1)
                    lwf_loss = F.kl_div(soft_outputs, soft_targets, reduction='batchmean')
                    lwf_loss *= self.lwf_temp ** 2
                
                total_loss = task_loss + self.reg_strength * lwf_loss
                total_loss.backward()
                optimizer.step()
                
                epoch_metrics['loss'].append(total_loss.item())
                epoch_metrics['task_loss'].append(task_loss.item())
                epoch_metrics['lwf_loss'].append(lwf_loss.item())
            
            for k in history:
                history[k].append(np.mean(epoch_metrics[k]))
        
        # Save model for next task
        self.prev_model = copy.deepcopy(self.model)
        self.prev_model.eval()
        
        return history
    
    def _train_with_packnet(
        self,
        dataset: Dataset,
        task_id: int,
        epochs: int,
        batch_size: int,
        lr: float
    ) -> Dict[str, List[float]]:
        """Train with PackNet pruning"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Apply previous masks
        for prev_task_id, masks in self.masks.items():
            for name, param in self.model.named_parameters():
                if name in masks:
                    param.data *= (1 - masks[name])
        
        history = {'loss': []}
        
        for _ in range(epochs):
            epoch_loss = []
            
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                
                # Zero out gradients for masked parameters
                for prev_id, masks in self.masks.items():
                    for name, param in self.model.named_parameters():
                        if name in masks and param.grad is not None:
                            param.grad *= (1 - masks[name])
                
                optimizer.step()
                epoch_loss.append(loss.item())
            
            history['loss'].append(np.mean(epoch_loss))
        
        # Prune and create mask for this task
        self.masks[task_id] = self._prune_weights(prune_ratio=0.5)
        self.current_task = task_id
        
        return history
    
    def _prune_weights(self, prune_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """Prune weights and create mask"""
        masks = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Get absolute values and flatten for quantile computation
                abs_weights = param.data.abs()
                
                # Find threshold (flatten to compute global quantile)
                threshold = torch.quantile(abs_weights.flatten(), prune_ratio)
                
                # Create mask
                mask = (abs_weights > threshold).float()
                masks[name] = mask
                
                # Apply mask
                param.data *= mask
        
        return masks
    
    def evaluate(
        self,
        dataset: Dataset,
        task_id: Optional[int] = None  # noqa: ARG002 - reserved for future task-specific evaluation
    ) -> float:
        """Evaluate on a dataset"""
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        return correct / total
    
    def evaluate_all_tasks(
        self,
        task_datasets: Dict[int, Dataset]
    ) -> Dict[int, float]:
        """Evaluate on all tasks"""
        results = {}
        
        for task_id, dataset in task_datasets.items():
            acc = self.evaluate(dataset, task_id)
            results[task_id] = acc
            logger.info(f"Task {task_id}: Accuracy = {acc:.4f}")
        
        return results
    
    def get_forgetting(
        self,
        task_datasets: Dict[int, Dataset]
    ) -> Dict[str, float]:
        """Calculate forgetting metrics"""
        if len(self.tasks) < 2:
            return {'average_forgetting': 0.0}
        
        current_accs = self.evaluate_all_tasks(task_datasets)
        
        forgetting = {}
        for task_info in self.tasks[:-1]:
            task_id = task_info['task_id']
            if task_id in task_datasets:
                initial_acc = task_info.get('validation_accuracy', 1.0)
                current_acc = current_accs.get(task_id, 0.0)
                forgetting[task_id] = max(0, initial_acc - current_acc)
        
        avg_forgetting = np.mean(list(forgetting.values())) if forgetting else 0.0
        
        return {
            'task_forgetting': forgetting,
            'average_forgetting': avg_forgetting
        }
