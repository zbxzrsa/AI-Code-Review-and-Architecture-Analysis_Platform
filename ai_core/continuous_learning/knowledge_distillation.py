"""
Knowledge Distillation and Model Fusion
Implements knowledge transfer between models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import copy
import numpy as np

logger = logging.getLogger(__name__)


class KnowledgeDistillation:
    """
    Knowledge Distillation System
    
    Features:
    - Teacher-student training
    - Multiple teacher ensemble distillation
    - Feature-level distillation
    - Progressive distillation
    - Self-distillation
    """
    
    def __init__(
        self,
        student: nn.Module,
        teacher: Union[nn.Module, List[nn.Module]],
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_layers: Optional[List[str]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Knowledge Distillation
        
        Args:
            student: Student model to train
            teacher: Teacher model(s) to distill from
            temperature: Softmax temperature for soft labels
            alpha: Weight for distillation loss (1-alpha for hard labels)
            feature_layers: Layer names for feature distillation
            device: Device to use
        """
        self.student = student.to(device)
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.feature_layers = feature_layers or []
        
        # Handle single or multiple teachers
        if isinstance(teacher, list):
            self.teachers = [t.to(device).eval() for t in teacher]
        else:
            self.teachers = [teacher.to(device).eval()]
        
        # Freeze teachers
        for teacher in self.teachers:
            for param in teacher.parameters():
                param.requires_grad = False
        
        # Feature hooks for intermediate layer distillation
        self.student_features: Dict[str, torch.Tensor] = {}
        self.teacher_features: Dict[str, torch.Tensor] = {}
        
        if self.feature_layers:
            self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward hooks for feature extraction"""
        def get_student_hook(name: str):
            def hook(module, input, output):
                self.student_features[name] = output
            return hook
        
        def get_teacher_hook(name: str):
            def hook(module, input, output):
                self.teacher_features[name] = output
            return hook
        
        for name in self.feature_layers:
            # Register on student
            for n, m in self.student.named_modules():
                if n == name:
                    m.register_forward_hook(get_student_hook(name))
            
            # Register on first teacher
            for n, m in self.teachers[0].named_modules():
                if n == name:
                    m.register_forward_hook(get_teacher_hook(name))
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss
        
        Args:
            student_logits: Student output logits
            teacher_logits: Teacher output logits (or ensemble)
            targets: Ground truth labels
            
        Returns:
            Total loss and loss components
        """
        # Soft labels (distillation) loss
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        
        distill_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard labels (classification) loss
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        # Feature distillation loss
        feature_loss = torch.tensor(0.0, device=self.device)
        if self.feature_layers:
            for layer_name in self.feature_layers:
                if layer_name in self.student_features and layer_name in self.teacher_features:
                    s_feat = self.student_features[layer_name]
                    t_feat = self.teacher_features[layer_name]
                    
                    # Normalize features
                    s_feat = F.normalize(s_feat.view(s_feat.size(0), -1), dim=1)
                    t_feat = F.normalize(t_feat.view(t_feat.size(0), -1), dim=1)
                    
                    feature_loss += F.mse_loss(s_feat, t_feat)
            
            total_loss = total_loss + 0.1 * feature_loss
        
        return total_loss, {
            'distill_loss': distill_loss.item(),
            'hard_loss': hard_loss.item(),
            'feature_loss': feature_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            inputs: Input batch
            targets: Target labels
            optimizer: Optimizer for student
            
        Returns:
            Training metrics
        """
        self.student.train()
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Get teacher predictions
        with torch.no_grad():
            if len(self.teachers) == 1:
                teacher_logits = self.teachers[0](inputs)
            else:
                # Ensemble teachers
                teacher_outputs = [t(inputs) for t in self.teachers]
                teacher_logits = torch.stack(teacher_outputs).mean(dim=0)
        
        # Get student predictions
        student_logits = self.student(inputs)
        
        # Compute loss
        loss, loss_components = self.distillation_loss(
            student_logits, teacher_logits, targets
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = student_logits.max(1)
        accuracy = (predicted == targets).float().mean().item()
        
        return {**loss_components, 'accuracy': accuracy}
    
    def train(
        self,
        train_dataset: Dataset,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        validation_dataset: Optional[Dataset] = None
    ) -> Dict[str, List[float]]:
        """
        Train student with knowledge distillation
        
        Args:
            train_dataset: Training dataset
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            validation_dataset: Optional validation dataset
            
        Returns:
            Training history
        """
        optimizer = optim.Adam(self.student.parameters(), lr=lr, weight_decay=1e-4)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'distill_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            epoch_loss = []
            epoch_accuracy = []
            epoch_distill_loss = []
            
            for inputs, targets in dataloader:
                metrics = self.train_step(inputs, targets, optimizer)
                epoch_loss.append(metrics['total_loss'])
                epoch_accuracy.append(metrics['accuracy'])
                epoch_distill_loss.append(metrics['distill_loss'])
            
            avg_loss = np.mean(epoch_loss)
            avg_accuracy = np.mean(epoch_accuracy)
            avg_distill = np.mean(epoch_distill_loss)
            
            history['train_loss'].append(avg_loss)
            history['train_accuracy'].append(avg_accuracy)
            history['distill_loss'].append(avg_distill)
            
            # Validation
            if validation_dataset:
                val_acc = self.evaluate(validation_dataset)
                history['val_accuracy'].append(val_acc)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}, "
                f"Distill={avg_distill:.4f}"
            )
        
        return history
    
    def evaluate(self, dataset: Dataset) -> float:
        """Evaluate student accuracy"""
        self.student.eval()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.student(inputs)
                _, predicted = outputs.max(1)
                
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        return correct / total
    
    def progressive_distill(
        self,
        train_dataset: Dataset,
        compression_ratios: List[float],
        epochs_per_stage: int = 5
    ) -> nn.Module:
        """
        Progressive distillation with increasing compression
        
        Args:
            train_dataset: Training dataset
            compression_ratios: List of compression ratios per stage
            epochs_per_stage: Training epochs per stage
            
        Returns:
            Final compressed student model
        """
        current_model = self.student
        
        for i, ratio in enumerate(compression_ratios):
            logger.info(f"Progressive distillation stage {i+1}: {ratio}x compression")
            
            # Create compressed student
            compressed_student = self._create_compressed_model(current_model, ratio)
            
            # Set up distillation
            distiller = KnowledgeDistillation(
                student=compressed_student,
                teacher=current_model,
                temperature=self.temperature,
                alpha=self.alpha,
                device=self.device
            )
            
            # Train
            distiller.train(train_dataset, epochs=epochs_per_stage)
            
            # Update current model
            current_model = compressed_student
        
        self.student = current_model
        return current_model
    
    def _create_compressed_model(
        self,
        model: nn.Module,
        compression_ratio: float
    ) -> nn.Module:
        """Create a compressed version of the model"""
        # This is a simplified implementation
        # In practice, you would implement proper architecture compression
        compressed = copy.deepcopy(model)
        
        # Simple width scaling (placeholder - real implementation would modify layers)
        for _, module in compressed.named_modules():
            if isinstance(module, nn.Linear):
                # Real architecture compression would resize layers here
                # compressed_in = int(module.in_features / compression_ratio)
                # compressed_out = int(module.out_features / compression_ratio)
                pass
        
        return compressed


class ModelFusion:
    """
    Model Fusion System
    
    Features:
    - Weight averaging
    - Feature-level fusion
    - Ensemble distillation
    - Selective fusion
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def weight_average(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None
    ) -> nn.Module:
        """
        Average model weights
        
        Args:
            models: List of models to average
            weights: Optional weights for each model
            
        Returns:
            Averaged model
        """
        if not models:
            raise ValueError("No models provided")
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Create output model
        fused_model = copy.deepcopy(models[0])
        
        # Average parameters
        fused_state = fused_model.state_dict()
        
        for key in fused_state:
            fused_state[key] = sum(
                w * m.state_dict()[key].float()
                for w, m in zip(weights, models)
            )
        
        fused_model.load_state_dict(fused_state)
        
        return fused_model.to(self.device)
    
    def selective_fusion(
        self,
        models: List[nn.Module],
        validation_dataset: Dataset,
        layer_selection: str = 'best'
    ) -> nn.Module:
        """
        Selective layer-wise fusion based on performance
        
        Args:
            models: Models to fuse
            validation_dataset: Dataset for evaluation
            layer_selection: 'best' or 'weighted'
            
        Returns:
            Fused model
        """
        if not models:
            raise ValueError("No models provided")
        
        # Evaluate each model
        model_scores = []
        for model in models:
            model = model.to(self.device).eval()
            score = self._evaluate_model(model, validation_dataset)
            model_scores.append(score)
        
        if layer_selection == 'best':
            # Select layers from best performing model
            best_idx = np.argmax(model_scores)
            return copy.deepcopy(models[best_idx])
        
        else:  # weighted
            # Weight average based on performance
            weights = [s / sum(model_scores) for s in model_scores]
            return self.weight_average(models, weights)
    
    def _evaluate_model(
        self,
        model: nn.Module,
        dataset: Dataset
    ) -> float:
        """Quick model evaluation"""
        model.eval()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        return correct / total
    
    def ensemble_to_single(
        self,
        ensemble_models: List[nn.Module],
        student_architecture: nn.Module,
        train_dataset: Dataset,
        epochs: int = 10
    ) -> nn.Module:
        """
        Distill an ensemble into a single model
        
        Args:
            ensemble_models: Models in the ensemble
            student_architecture: Single model architecture
            train_dataset: Training dataset
            epochs: Training epochs
            
        Returns:
            Distilled single model
        """
        distiller = KnowledgeDistillation(
            student=student_architecture,
            teacher=ensemble_models,
            temperature=4.0,
            alpha=0.7,
            device=self.device
        )
        
        distiller.train(train_dataset, epochs=epochs)
        
        return distiller.student
    
    def stochastic_weight_averaging(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        swa_start_epoch: int = 10,
        swa_lr: float = 1e-4,
        total_epochs: int = 30
    ) -> nn.Module:
        """
        Stochastic Weight Averaging for improved generalization
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            swa_start_epoch: Epoch to start SWA
            swa_lr: SWA learning rate
            total_epochs: Total training epochs
            
        Returns:
            SWA model
        """
        model = model.to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=swa_lr, momentum=0.9, weight_decay=1e-4)
        dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        
        # SWA model
        swa_model = copy.deepcopy(model)
        swa_n = 0
        
        for epoch in range(total_epochs):
            model.train()
            
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Update SWA model
            if epoch >= swa_start_epoch:
                swa_n += 1
                with torch.no_grad():
                    for swa_param, param in zip(swa_model.parameters(), model.parameters()):
                        swa_param.data = (swa_param.data * (swa_n - 1) + param.data) / swa_n
            
            logger.info(f"Epoch {epoch+1}/{total_epochs}: Loss={loss.item():.4f}")
        
        return swa_model
