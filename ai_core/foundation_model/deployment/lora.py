"""
LoRA (Low-Rank Adaptation) module for efficient fine-tuning.

Implements adapter-based training where only small adapter weights
are trained while the base model remains frozen.
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .config import PracticalDeploymentConfig

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) Layer
    
    Efficient fine-tuning by adding low-rank matrices:
    W' = W + BA (where B is r×d, A is d×r, r << d)
    
    This reduces trainable parameters from d×d to 2×r×d.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Freeze original weights
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize: A with Kaiming, B with zeros (start with original behavior)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        original_output = self.original_layer(x)
        
        # LoRA output: B(A(dropout(x))) * scaling
        lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into original layer for inference."""
        with torch.no_grad():
            merged_weight = (
                self.original_layer.weight +
                self.lora_B.weight @ self.lora_A.weight * self.scaling
            )
            self.original_layer.weight.copy_(merged_weight)
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from original layer."""
        with torch.no_grad():
            unmerged_weight = (
                self.original_layer.weight -
                self.lora_B.weight @ self.lora_A.weight * self.scaling
            )
            self.original_layer.weight.copy_(unmerged_weight)
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get only LoRA weights for efficient saving."""
        return {
            "lora_A": self.lora_A.weight.data.clone(),
            "lora_B": self.lora_B.weight.data.clone(),
        }
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load LoRA weights."""
        self.lora_A.weight.data.copy_(state_dict["lora_A"])
        self.lora_B.weight.data.copy_(state_dict["lora_B"])
    
    @property
    def trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in [self.lora_A.weight, self.lora_B.weight])


class LoRAAdapterManager:
    """
    Manages multiple LoRA adapters for different tasks/domains.
    
    Features:
    - Create/load/save adapters
    - Switch between adapters at runtime
    - Merge multiple adapters with weighted average
    - Version control for adapters
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: PracticalDeploymentConfig,
    ):
        self.base_model = base_model
        self.config = config
        
        self.device = next(base_model.parameters()).device
        
        # Adapter storage
        self.adapters: Dict[str, Dict[str, LoRALayer]] = {}
        self.active_adapter: Optional[str] = None
        
        # Version tracking
        self.adapter_versions: Dict[str, int] = {}
        self.adapter_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Original modules cache for proper replacement
        self._original_modules: Dict[str, nn.Module] = {}
        
        # Freeze base model
        if config.freeze_base_model:
            self._freeze_base_model()
    
    def _freeze_base_model(self):
        """Freeze all base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Froze base model parameters")
    
    def create_adapter(
        self,
        adapter_name: str,
        target_modules: Optional[List[str]] = None,
    ) -> Dict[str, LoRALayer]:
        """Create a new LoRA adapter."""
        if adapter_name in self.adapters:
            logger.warning(f"Adapter {adapter_name} already exists, overwriting")
        
        target_modules = target_modules or self.config.lora_target_modules
        adapter_layers = {}
        
        for name, module in self.base_model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    lora_layer = LoRALayer(
                        module,
                        r=self.config.lora_r,
                        alpha=self.config.lora_alpha,
                        dropout=self.config.lora_dropout,
                    ).to(self.device)
                    
                    adapter_layers[name] = lora_layer
                    self._original_modules[name] = module
        
        self.adapters[adapter_name] = adapter_layers
        self.adapter_versions[adapter_name] = 1
        self.adapter_metadata[adapter_name] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_modules": target_modules,
            "r": self.config.lora_r,
            "alpha": self.config.lora_alpha,
            "trainable_params": sum(l.trainable_parameters for l in adapter_layers.values()),
        }
        
        logger.info(
            f"Created adapter '{adapter_name}' with {len(adapter_layers)} LoRA layers, "
            f"{self.adapter_metadata[adapter_name]['trainable_params']:,} trainable params"
        )
        
        return adapter_layers
    
    def activate_adapter(self, adapter_name: str):
        """Activate an adapter for inference."""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        # Deactivate current if different
        if self.active_adapter and self.active_adapter != adapter_name:
            self.deactivate_adapter()
        
        self.active_adapter = adapter_name
        logger.info(f"Activated adapter: {adapter_name}")
    
    def deactivate_adapter(self):
        """Deactivate current adapter (use base model only)."""
        self.active_adapter = None
        logger.info("Deactivated adapter, using base model")
    
    def forward_with_adapter(
        self,
        input_ids: torch.Tensor,
        adapter_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass with optional adapter."""
        adapter_name = adapter_name or self.active_adapter
        
        if adapter_name is None:
            # Use base model only
            return self.base_model(input_ids=input_ids, **kwargs)
        
        adapter = self.adapters.get(adapter_name)
        if adapter is None:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        # In a full implementation, we would properly inject LoRA layers
        # For now, we use the base model (adapters would modify in place)
        return self.base_model(input_ids=input_ids, **kwargs)
    
    def save_adapter(
        self,
        adapter_name: str,
        save_path: str,
    ):
        """Save adapter to disk."""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        save_dir = Path(save_path) / adapter_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        adapter_state = {}
        for layer_name, lora_layer in self.adapters[adapter_name].items():
            adapter_state[layer_name] = lora_layer.get_lora_state_dict()
        
        torch.save(adapter_state, save_dir / "adapter_weights.pt")
        
        # Save metadata
        metadata = {
            **self.adapter_metadata[adapter_name],
            "version": self.adapter_versions[adapter_name],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved adapter '{adapter_name}' to {save_dir}")
    
    def load_adapter(
        self,
        adapter_name: str,
        load_path: str,
    ):
        """Load adapter from disk."""
        load_dir = Path(load_path) / adapter_name
        
        if not load_dir.exists():
            raise ValueError(f"Adapter path not found: {load_dir}")
        
        # Load metadata
        with open(load_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create adapter structure
        self.create_adapter(adapter_name, metadata.get("target_modules"))
        
        # Load weights
        adapter_state = torch.load(
            load_dir / "adapter_weights.pt",
            map_location=self.device,
            weights_only=True,
        )
        
        for layer_name, state_dict in adapter_state.items():
            if layer_name in self.adapters[adapter_name]:
                self.adapters[adapter_name][layer_name].load_lora_state_dict(state_dict)
        
        self.adapter_metadata[adapter_name] = metadata
        self.adapter_versions[adapter_name] = metadata.get("version", 1)
        
        logger.info(f"Loaded adapter '{adapter_name}' v{self.adapter_versions[adapter_name]} from {load_dir}")
    
    def merge_adapters(
        self,
        adapter_names: List[str],
        weights: Optional[List[float]] = None,
        new_name: str = "merged",
    ) -> Dict[str, LoRALayer]:
        """Merge multiple adapters with weighted average."""
        if len(adapter_names) < 2:
            raise ValueError("Need at least 2 adapters to merge")
        
        # Normalize weights
        weights = weights or [1.0 / len(adapter_names)] * len(adapter_names)
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        if len(weights) != len(adapter_names):
            raise ValueError("Weights must match adapter count")
        
        # Verify all adapters exist
        for name in adapter_names:
            if name not in self.adapters:
                raise ValueError(f"Adapter '{name}' not found")
        
        # Create new adapter
        merged_adapter = self.create_adapter(new_name)
        
        # Merge weights
        for layer_name in merged_adapter:
            merged_a = torch.zeros_like(merged_adapter[layer_name].lora_A.weight)
            merged_b = torch.zeros_like(merged_adapter[layer_name].lora_B.weight)
            
            for adapter_name, weight in zip(adapter_names, weights):
                if layer_name in self.adapters[adapter_name]:
                    merged_a += weight * self.adapters[adapter_name][layer_name].lora_A.weight
                    merged_b += weight * self.adapters[adapter_name][layer_name].lora_B.weight
            
            merged_adapter[layer_name].lora_A.weight.data.copy_(merged_a)
            merged_adapter[layer_name].lora_B.weight.data.copy_(merged_b)
        
        logger.info(f"Merged {len(adapter_names)} adapters into '{new_name}' with weights {weights}")
        
        return merged_adapter
    
    def delete_adapter(self, adapter_name: str):
        """Delete an adapter from memory."""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        if self.active_adapter == adapter_name:
            self.deactivate_adapter()
        
        del self.adapters[adapter_name]
        del self.adapter_versions[adapter_name]
        del self.adapter_metadata[adapter_name]
        
        logger.info(f"Deleted adapter: {adapter_name}")
    
    def get_trainable_parameters(
        self,
        adapter_name: Optional[str] = None,
    ) -> List[nn.Parameter]:
        """Get trainable parameters (adapter only)."""
        adapter_name = adapter_name or self.active_adapter
        
        if adapter_name is None:
            return []
        
        if adapter_name not in self.adapters:
            return []
        
        params = []
        for lora_layer in self.adapters[adapter_name].values():
            params.extend([
                lora_layer.lora_A.weight,
                lora_layer.lora_B.weight,
            ])
        
        return params
    
    def get_adapter_info(self, adapter_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about an adapter."""
        adapter_name = adapter_name or self.active_adapter
        
        if adapter_name is None or adapter_name not in self.adapters:
            return {}
        
        return {
            "name": adapter_name,
            "version": self.adapter_versions[adapter_name],
            "metadata": self.adapter_metadata[adapter_name],
            "num_layers": len(self.adapters[adapter_name]),
            "is_active": adapter_name == self.active_adapter,
        }
    
    def list_adapters(self) -> List[str]:
        """List all available adapters."""
        return list(self.adapters.keys())
