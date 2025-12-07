"""
Model distillation for creating smaller, faster models.

Transfers knowledge from large teacher model to smaller student model.

Features:
- Dynamic student architecture configuration
- Multiple layer reduction strategies
- Weight initialization from teacher
- Architecture presets (small, medium, large)
"""

import copy
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from .config import PracticalDeploymentConfig

logger = logging.getLogger(__name__)


class StudentSizePreset(str, Enum):
    """Predefined student model size presets."""
    TINY = "tiny"      # ~10% of teacher
    SMALL = "small"    # ~25% of teacher
    MEDIUM = "medium"  # ~50% of teacher
    LARGE = "large"    # ~75% of teacher


@dataclass
class StudentModelConfig:
    """Configuration for student model architecture.
    
    Attributes:
        hidden_size: Hidden dimension (None to derive from teacher)
        num_layers: Number of transformer layers (None to derive)
        num_attention_heads: Number of attention heads (None to derive)
        intermediate_size: FFN intermediate size (None to derive)
        size_preset: Predefined size preset (overrides individual settings)
        layer_selection: Strategy for selecting which teacher layers to keep
        initialize_from_teacher: Whether to initialize student weights from teacher
    """
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    size_preset: Optional[StudentSizePreset] = None
    layer_selection: str = "uniform"  # "uniform", "first", "last", "every_n"
    initialize_from_teacher: bool = True
    vocab_size: Optional[int] = None


class StudentModelBuilder:
    """
    Builds student models with configurable architecture.
    
    Supports:
    - Layer reduction (skip layers)
    - Dimension reduction
    - Weight initialization from teacher
    """
    
    # Size preset ratios (relative to teacher)
    SIZE_RATIOS = {
        StudentSizePreset.TINY: {"layers": 0.1, "hidden": 0.5, "heads": 0.5},
        StudentSizePreset.SMALL: {"layers": 0.25, "hidden": 0.75, "heads": 0.75},
        StudentSizePreset.MEDIUM: {"layers": 0.5, "hidden": 1.0, "heads": 1.0},
        StudentSizePreset.LARGE: {"layers": 0.75, "hidden": 1.0, "heads": 1.0},
    }
    
    def __init__(self, teacher_model: nn.Module):
        self.teacher = teacher_model
        self.teacher_config = self._extract_teacher_config()
    
    def _extract_teacher_config(self) -> Dict[str, Any]:
        """Extract configuration from teacher model."""
        config = {
            "num_layers": 0,
            "hidden_size": 0,
            "num_attention_heads": 0,
            "intermediate_size": 0,
            "vocab_size": 0,
        }
        
        # Try to get config from model
        if hasattr(self.teacher, 'config'):
            tc = self.teacher.config
            config["num_layers"] = getattr(tc, 'num_hidden_layers', getattr(tc, 'n_layer', 12))
            config["hidden_size"] = getattr(tc, 'hidden_size', getattr(tc, 'n_embd', 768))
            config["num_attention_heads"] = getattr(tc, 'num_attention_heads', getattr(tc, 'n_head', 12))
            config["intermediate_size"] = getattr(tc, 'intermediate_size', config["hidden_size"] * 4)
            config["vocab_size"] = getattr(tc, 'vocab_size', 50257)
        else:
            # Infer from model structure
            for name, module in self.teacher.named_modules():
                if isinstance(module, nn.Linear):
                    if config["hidden_size"] == 0:
                        config["hidden_size"] = module.in_features
                if 'layer' in name.lower() or 'block' in name.lower():
                    config["num_layers"] += 1
            
            config["num_layers"] = max(config["num_layers"] // 2, 1)  # Rough estimate
            config["num_attention_heads"] = config["hidden_size"] // 64
            config["intermediate_size"] = config["hidden_size"] * 4
        
        logger.info(f"Extracted teacher config: {config}")
        return config
    
    def build(self, student_config: StudentModelConfig) -> nn.Module:
        """Build student model based on configuration."""
        # Apply size preset if specified
        if student_config.size_preset:
            student_config = self._apply_preset(student_config)
        
        # Fill in missing values from teacher
        final_config = self._finalize_config(student_config)
        
        # Build the student model
        student = self._create_student_architecture(final_config)
        
        # Initialize from teacher if requested
        if student_config.initialize_from_teacher:
            self._initialize_from_teacher(student, final_config)
        
        return student
    
    def _apply_preset(self, config: StudentModelConfig) -> StudentModelConfig:
        """Apply size preset to configuration."""
        ratios = self.SIZE_RATIOS.get(config.size_preset, self.SIZE_RATIOS[StudentSizePreset.SMALL])
        
        new_config = StudentModelConfig(
            num_layers=config.num_layers or int(self.teacher_config["num_layers"] * ratios["layers"]),
            hidden_size=config.hidden_size or int(self.teacher_config["hidden_size"] * ratios["hidden"]),
            num_attention_heads=config.num_attention_heads or int(self.teacher_config["num_attention_heads"] * ratios["heads"]),
            intermediate_size=config.intermediate_size,
            layer_selection=config.layer_selection,
            initialize_from_teacher=config.initialize_from_teacher,
            vocab_size=config.vocab_size or self.teacher_config.get("vocab_size"),
        )
        
        # Ensure valid values
        new_config.num_layers = max(1, new_config.num_layers)
        new_config.num_attention_heads = max(1, new_config.num_attention_heads)
        
        # Hidden size must be divisible by num_heads
        if new_config.hidden_size % new_config.num_attention_heads != 0:
            new_config.hidden_size = (new_config.hidden_size // new_config.num_attention_heads) * new_config.num_attention_heads
        
        return new_config
    
    def _finalize_config(self, config: StudentModelConfig) -> Dict[str, Any]:
        """Finalize configuration with defaults from teacher."""
        return {
            "num_layers": config.num_layers or self.teacher_config["num_layers"],
            "hidden_size": config.hidden_size or self.teacher_config["hidden_size"],
            "num_attention_heads": config.num_attention_heads or self.teacher_config["num_attention_heads"],
            "intermediate_size": config.intermediate_size or (config.hidden_size or self.teacher_config["hidden_size"]) * 4,
            "vocab_size": config.vocab_size or self.teacher_config.get("vocab_size", 50257),
            "layer_selection": config.layer_selection,
        }
    
    def _create_student_architecture(self, config: Dict[str, Any]) -> nn.Module:
        """Create student model architecture."""
        # Try to use the same architecture class as teacher
        if hasattr(self.teacher, 'config') and hasattr(self.teacher, '__class__'):
            return self._create_from_teacher_class(config)
        
        # Fall back to generic transformer
        return self._create_generic_transformer(config)
    
    def _create_from_teacher_class(self, config: Dict[str, Any]) -> nn.Module:
        """Create student using teacher's architecture class."""
        try:
            # Copy and modify teacher's config
            student_config = copy.deepcopy(self.teacher.config)
            
            # Update with student dimensions
            if hasattr(student_config, 'num_hidden_layers'):
                student_config.num_hidden_layers = config["num_layers"]
            elif hasattr(student_config, 'n_layer'):
                student_config.n_layer = config["num_layers"]
            
            if hasattr(student_config, 'hidden_size'):
                student_config.hidden_size = config["hidden_size"]
            elif hasattr(student_config, 'n_embd'):
                student_config.n_embd = config["hidden_size"]
            
            if hasattr(student_config, 'num_attention_heads'):
                student_config.num_attention_heads = config["num_attention_heads"]
            elif hasattr(student_config, 'n_head'):
                student_config.n_head = config["num_attention_heads"]
            
            if hasattr(student_config, 'intermediate_size'):
                student_config.intermediate_size = config["intermediate_size"]
            
            # Create new model with modified config
            student = self.teacher.__class__(student_config)
            
            logger.info(f"Created student model using teacher's architecture class")
            return student
            
        except Exception as e:
            logger.warning(f"Could not create from teacher class: {e}, using generic transformer")
            return self._create_generic_transformer(config)
    
    def _create_generic_transformer(self, config: Dict[str, Any]) -> nn.Module:
        """Create a generic transformer model."""
        return GenericTransformer(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
        )
    
    def _initialize_from_teacher(self, student: nn.Module, config: Dict[str, Any]) -> None:
        """Initialize student weights from teacher."""
        layer_selection = config.get("layer_selection", "uniform")
        teacher_layers = self.teacher_config["num_layers"]
        student_layers = config["num_layers"]
        
        # Determine which teacher layers to copy
        if layer_selection == "uniform":
            # Uniformly spaced layers
            step = teacher_layers / student_layers
            selected_layers = [int(i * step) for i in range(student_layers)]
        elif layer_selection == "first":
            # First N layers
            selected_layers = list(range(student_layers))
        elif layer_selection == "last":
            # Last N layers
            selected_layers = list(range(teacher_layers - student_layers, teacher_layers))
        else:
            selected_layers = list(range(student_layers))
        
        logger.info(f"Initializing student from teacher layers: {selected_layers}")
        
        # Copy embeddings
        self._copy_embeddings(student)
        
        # Copy selected layers
        self._copy_layers(student, selected_layers, config)
    
    def _copy_embeddings(self, student: nn.Module) -> None:
        """Copy embedding weights from teacher to student."""
        teacher_emb = None
        student_emb = None
        
        for name, module in self.teacher.named_modules():
            if 'embed' in name.lower() and isinstance(module, nn.Embedding):
                teacher_emb = module
                break
        
        for name, module in student.named_modules():
            if 'embed' in name.lower() and isinstance(module, nn.Embedding):
                student_emb = module
                break
        
        if teacher_emb and student_emb:
            # Copy weights (truncate if necessary)
            min_vocab = min(teacher_emb.num_embeddings, student_emb.num_embeddings)
            min_dim = min(teacher_emb.embedding_dim, student_emb.embedding_dim)
            
            with torch.no_grad():
                student_emb.weight[:min_vocab, :min_dim] = teacher_emb.weight[:min_vocab, :min_dim]
            
            logger.info(f"Copied embeddings: {min_vocab} tokens, {min_dim} dimensions")
    
    def _copy_layers(self, student: nn.Module, selected_layers: List[int], config: Dict[str, Any]) -> None:
        """Copy transformer layer weights from teacher to student."""
        # Find layer modules
        teacher_layers = []
        student_layers = []
        
        for name, module in self.teacher.named_modules():
            if self._is_transformer_layer(name, module):
                teacher_layers.append((name, module))
        
        for name, module in student.named_modules():
            if self._is_transformer_layer(name, module):
                student_layers.append((name, module))
        
        # Copy selected layers
        for i, layer_idx in enumerate(selected_layers):
            if i < len(student_layers) and layer_idx < len(teacher_layers):
                self._copy_layer_weights(
                    teacher_layers[layer_idx][1],
                    student_layers[i][1],
                    config
                )
    
    def _is_transformer_layer(self, name: str, module: nn.Module) -> bool:
        """Check if module is a transformer layer."""
        layer_keywords = ['layer', 'block', 'h.']
        return any(kw in name.lower() for kw in layer_keywords) and hasattr(module, 'parameters')
    
    def _copy_layer_weights(self, teacher_layer: nn.Module, student_layer: nn.Module, config: Dict[str, Any]) -> None:
        """Copy weights from teacher layer to student layer."""
        teacher_dict = dict(teacher_layer.named_parameters())
        
        with torch.no_grad():
            for name, param in student_layer.named_parameters():
                if name in teacher_dict:
                    teacher_param = teacher_dict[name]
                    
                    # Handle dimension mismatch by truncating
                    if param.shape == teacher_param.shape:
                        param.copy_(teacher_param)
                    else:
                        # Truncate to smaller dimensions
                        slices = tuple(slice(0, min(p, t)) for p, t in zip(param.shape, teacher_param.shape))
                        param[slices] = teacher_param[slices]


class GenericTransformer(nn.Module):
    """Generic transformer model for distillation when teacher class unavailable."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(2048, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.transformer(x)
        logits = self.output_projection(x)
        
        return {"logits": logits}


class ModelDistiller:
    """
    Distills large teacher model to smaller student model.
    
    Useful for:
    - Edge deployment
    - Faster inference
    - Lower cost
    
    Uses knowledge distillation with soft targets.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        config: PracticalDeploymentConfig,
    ):
        self.teacher_model = teacher_model
        self.config = config
        
        self.device = next(teacher_model.parameters()).device
        
        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # Student model (to be created)
        self.student_model: Optional[nn.Module] = None
    
    @classmethod
    def create_teacher_model(
        cls,
        model_name_or_path: str,
        config: "PracticalDeploymentConfig",
        device_map: Optional[str] = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "ModelDistiller":
        """
        Create a ModelDistiller with teacher model loaded from path or hub.
        
        ⚠️ PLACEHOLDER: Full implementation needs HuggingFace integration.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            config: Deployment configuration
            device_map: Device mapping strategy ("auto", "cuda:0", etc.)
            torch_dtype: Model dtype (torch.float16, torch.bfloat16, etc.)
            
        Returns:
            ModelDistiller instance with teacher model loaded
            
        Technical Design:
            1. Support loading from HuggingFace Hub
            2. Support loading from local checkpoint
            3. Automatic model architecture detection
            4. Memory-efficient loading with device_map
            5. Support for sharded models
            
        Target Version: v2.1.5
        
        Example:
            ```python
            distiller = ModelDistiller.create_teacher_model(
                "meta-llama/Llama-2-7b-hf",
                config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            ```
        """
        logger.info(f"Loading teacher model from: {model_name_or_path}")
        
        # Placeholder implementation
        logger.warning(
            "create_teacher_model is a placeholder. "
            "Full implementation with HuggingFace support coming in v2.1.5. "
            "Please load your model manually and pass to ModelDistiller()."
        )
        
        # TODO: Implement proper model loading
        # try:
        #     from transformers import AutoModelForCausalLM
        #     teacher = AutoModelForCausalLM.from_pretrained(
        #         model_name_or_path,
        #         device_map=device_map,
        #         torch_dtype=torch_dtype,
        #     )
        #     return cls(teacher, config)
        # except ImportError:
        #     raise ImportError("transformers required: pip install transformers")
        
        raise NotImplementedError(
            "create_teacher_model() is not yet implemented. "
            "Please load your model manually: "
            "distiller = ModelDistiller(your_loaded_model, config)"
        )
    
    def create_student_model(
        self,
        student_config: Optional[Dict[str, Any]] = None,
        size_preset: Optional[StudentSizePreset] = None,
    ) -> nn.Module:
        """
        Create a smaller student model with configurable architecture.
        
        Args:
            student_config: Configuration dictionary with:
                - hidden_size: Hidden dimension (None to derive from preset/teacher)
                - num_layers: Number of layers (None to derive)
                - num_attention_heads: Number of attention heads (None to derive)
                - intermediate_size: FFN intermediate size (None to derive)
                - layer_selection: "uniform", "first", "last" (default: "uniform")
                - initialize_from_teacher: bool (default: True)
            size_preset: Predefined size preset (overrides student_config if set)
                - StudentSizePreset.TINY: ~10% of teacher
                - StudentSizePreset.SMALL: ~25% of teacher
                - StudentSizePreset.MEDIUM: ~50% of teacher
                - StudentSizePreset.LARGE: ~75% of teacher
                
        Returns:
            Student model instance, ready for distillation
            
        Example:
            ```python
            # Using preset
            student = distiller.create_student_model(size_preset=StudentSizePreset.SMALL)
            
            # Using custom config
            student = distiller.create_student_model({
                "num_layers": 6,
                "hidden_size": 512,
                "num_attention_heads": 8,
            })
            ```
        """
        # Determine configuration
        if size_preset is None:
            size_preset_str = self.config.student_model_size
            size_preset_map = {
                "tiny": StudentSizePreset.TINY,
                "small": StudentSizePreset.SMALL,
                "medium": StudentSizePreset.MEDIUM,
                "large": StudentSizePreset.LARGE,
            }
            size_preset = size_preset_map.get(size_preset_str.lower(), StudentSizePreset.SMALL)
        
        # Build StudentModelConfig
        config_dict = student_config or {}
        model_config = StudentModelConfig(
            hidden_size=config_dict.get("hidden_size"),
            num_layers=config_dict.get("num_layers"),
            num_attention_heads=config_dict.get("num_attention_heads"),
            intermediate_size=config_dict.get("intermediate_size"),
            size_preset=size_preset if not config_dict else None,
            layer_selection=config_dict.get("layer_selection", "uniform"),
            initialize_from_teacher=config_dict.get("initialize_from_teacher", True),
            vocab_size=config_dict.get("vocab_size"),
        )
        
        # Use StudentModelBuilder to create the student
        builder = StudentModelBuilder(self.teacher_model)
        self.student_model = builder.build(model_config)
        
        # Move to same device as teacher
        self.student_model = self.student_model.to(self.device)
        
        # Log model info
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        compression = teacher_params / student_params if student_params > 0 else 1.0
        
        logger.info(
            f"Created student model: "
            f"{student_params:,} params ({student_params/1e6:.1f}M), "
            f"{compression:.1f}x smaller than teacher ({teacher_params:,} params)"
        )
        
        return self.student_model
    
    def create_student_from_preset(
        self,
        preset: StudentSizePreset = StudentSizePreset.SMALL,
    ) -> nn.Module:
        """
        Create student model using a predefined size preset.
        
        Args:
            preset: One of TINY, SMALL, MEDIUM, LARGE
            
        Returns:
            Student model instance
        """
        return self.create_student_model(size_preset=preset)
    
    def distill(
        self,
        train_data: List[Dict[str, Any]],
        epochs: int = 3,
        temperature: float = 2.0,
        alpha: float = 0.5,
        learning_rate: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Distill knowledge from teacher to student.
        
        Loss = α * hard_loss + (1-α) * soft_loss * T²
        
        Args:
            train_data: Training samples
            epochs: Number of training epochs
            temperature: Softmax temperature (higher = softer distribution)
            alpha: Weight for hard loss vs soft loss
            learning_rate: Learning rate for student optimizer
            
        Returns:
            Training results including final loss and metrics
        """
        if self.student_model is None:
            raise ValueError("Student model not created")
        
        optimizer = AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        logger.info(f"Starting distillation: {epochs} epochs, T={temperature}, α={alpha}")
        
        results = {
            "epochs": [],
            "final_loss": 0.0,
            "total_samples": len(train_data) * epochs,
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_hard_loss = 0.0
            epoch_soft_loss = 0.0
            
            self.student_model.train()
            
            for sample in train_data:
                input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                
                # Get teacher logits (no gradient)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(input_ids=input_ids)
                    teacher_logits = teacher_outputs['logits']
                
                # Get student logits
                student_outputs = self.student_model(input_ids=input_ids)
                student_logits = student_outputs['logits']
                
                # Soft loss (KL divergence with temperature)
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction='batchmean',
                ) * (temperature ** 2)
                
                # Hard loss (cross entropy with true labels)
                labels = sample.get('labels', sample['input_ids']).to(self.device)
                hard_loss = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                
                # Combined loss
                loss = alpha * hard_loss + (1 - alpha) * soft_loss
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_hard_loss += hard_loss.item()
                epoch_soft_loss += soft_loss.item()
            
            avg_loss = epoch_loss / len(train_data)
            avg_hard = epoch_hard_loss / len(train_data)
            avg_soft = epoch_soft_loss / len(train_data)
            
            results["epochs"].append({
                "epoch": epoch + 1,
                "total_loss": avg_loss,
                "hard_loss": avg_hard,
                "soft_loss": avg_soft,
            })
            
            logger.info(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Loss: {avg_loss:.4f} (hard: {avg_hard:.4f}, soft: {avg_soft:.4f})"
            )
        
        results["final_loss"] = results["epochs"][-1]["total_loss"]
        
        return results
    
    def get_student_model(self) -> Optional[nn.Module]:
        """Get the trained student model."""
        return self.student_model
    
    def compare_models(
        self,
        test_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare teacher and student model outputs."""
        if self.student_model is None:
            raise ValueError("Student model not created")
        
        self.teacher_model.eval()
        self.student_model.eval()
        
        metrics = {
            "agreement_rate": 0.0,
            "kl_divergence": 0.0,
            "samples_evaluated": len(test_data),
        }
        
        agreements = 0
        total_kl = 0.0
        
        with torch.no_grad():
            for sample in test_data:
                input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                
                teacher_outputs = self.teacher_model(input_ids=input_ids)
                student_outputs = self.student_model(input_ids=input_ids)
                
                teacher_pred = teacher_outputs['logits'].argmax(dim=-1)
                student_pred = student_outputs['logits'].argmax(dim=-1)
                
                # Check agreement
                if torch.equal(teacher_pred, student_pred):
                    agreements += 1
                
                # Calculate KL divergence
                kl = F.kl_div(
                    F.log_softmax(student_outputs['logits'], dim=-1),
                    F.softmax(teacher_outputs['logits'], dim=-1),
                    reduction='batchmean',
                )
                total_kl += kl.item()
        
        metrics["agreement_rate"] = agreements / len(test_data)
        metrics["kl_divergence"] = total_kl / len(test_data)
        
        return metrics
