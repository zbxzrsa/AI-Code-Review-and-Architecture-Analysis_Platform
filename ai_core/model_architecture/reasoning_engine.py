"""
Reasoning Engine with Dynamic Path Selection
Implements multi-path reasoning and adaptive inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReasoningPath:
    """A reasoning path configuration"""
    name: str
    modules: List[str]
    complexity: float
    estimated_latency: float
    accuracy_score: float


class ReasoningModule(nn.Module):
    """Base class for reasoning modules"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor:
        raise NotImplementedError


class DirectReasoningModule(ReasoningModule):
    """Direct reasoning - simple forward pass"""
    
    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)
        self.transform = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor:
        return self.activation(self.transform(x))


class ChainOfThoughtModule(ReasoningModule):
    """Chain-of-thought reasoning"""
    
    def __init__(self, hidden_dim: int, num_steps: int = 3):
        super().__init__(hidden_dim)
        self.num_steps = num_steps
        
        self.steps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            for _ in range(num_steps)
        ])
        
        self.combiner = nn.Linear(hidden_dim * num_steps, hidden_dim)
    
    def forward(self, x: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor:
        thoughts = []
        h = x
        
        for step in self.steps:
            h = step(h)
            thoughts.append(h)
        
        # Combine all thoughts
        combined = torch.cat(thoughts, dim=-1)
        return self.combiner(combined)


class AnalogicalReasoningModule(ReasoningModule):
    """Analogical reasoning using similarity matching"""
    
    def __init__(self, hidden_dim: int, num_exemplars: int = 64):
        super().__init__(hidden_dim)
        self.num_exemplars = num_exemplars
        
        # Learnable exemplar memory
        self.exemplars = nn.Parameter(torch.randn(num_exemplars, hidden_dim))
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor:
        # Project input as query
        q = self.query_proj(x)
        
        # Project exemplars as keys and values
        k = self.key_proj(self.exemplars)
        v = self.value_proj(self.exemplars)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        attn = F.softmax(attn, dim=-1)
        
        # Aggregate values
        out = torch.matmul(attn, v)
        
        return self.output_proj(out + x)


class AbstractionModule(ReasoningModule):
    """Abstraction reasoning - compress and decompress"""
    
    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64):
        super().__init__(hidden_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor, context: Optional[Dict] = None) -> torch.Tensor:
        # Compress to abstract representation
        abstract = self.encoder(x)
        
        # Decompress back
        reconstructed = self.decoder(abstract)
        
        return reconstructed + x  # Residual connection


class DynamicRouter(nn.Module):
    """
    Dynamic Path Router
    
    Selects the best reasoning path based on input complexity
    """
    
    def __init__(
        self,
        input_dim: int,
        num_paths: int,
        temperature: float = 1.0,
        hard_routing: bool = False
    ):
        super().__init__()
        
        self.num_paths = num_paths
        self.temperature = temperature
        self.hard_routing = hard_routing
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_paths)
        )
        
        # Load balancing
        self.path_usage = torch.zeros(num_paths)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Route input to best path
        
        Args:
            x: Input tensor
            
        Returns:
            Routing weights and selected path index
        """
        # Compute routing scores
        scores = self.router(x.mean(dim=1) if x.dim() > 2 else x)
        
        if self.hard_routing:
            # Hard routing - select single path
            selected = scores.argmax(dim=-1)
            weights = F.one_hot(selected, self.num_paths).float()
        else:
            # Soft routing - weighted combination
            weights = F.softmax(scores / self.temperature, dim=-1)
            selected = weights.argmax(dim=-1)
        
        # Update usage statistics
        self.path_usage += weights.sum(dim=0).detach()
        
        return weights, selected.item() if selected.dim() == 0 else selected[0].item()
    
    def get_load_balance_loss(self) -> torch.Tensor:
        """Compute load balancing loss"""
        if self.path_usage.sum() == 0:
            return torch.tensor(0.0)
        
        # Encourage uniform usage
        usage = self.path_usage / self.path_usage.sum()
        target = torch.ones_like(usage) / self.num_paths
        
        return F.mse_loss(usage, target)


class ReasoningEngine(nn.Module):
    """
    Multi-Path Reasoning Engine
    
    Features:
    - Multiple reasoning modules
    - Dynamic path selection
    - Confidence-based routing
    - Ensemble reasoning
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        routing_type: str = 'dynamic',
        use_ensemble: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.routing_type = routing_type
        self.use_ensemble = use_ensemble
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Reasoning modules
        self.reasoning_modules = nn.ModuleDict({
            'direct': DirectReasoningModule(hidden_dim),
            'chain_of_thought': ChainOfThoughtModule(hidden_dim, num_steps=3),
            'analogical': AnalogicalReasoningModule(hidden_dim),
            'abstraction': AbstractionModule(hidden_dim)
        })
        
        # Router
        if routing_type == 'dynamic':
            self.router = DynamicRouter(
                hidden_dim,
                len(self.reasoning_modules),
                hard_routing=not use_ensemble
            )
        
        # Output projection
        if use_ensemble:
            self.output_proj = nn.Linear(hidden_dim * len(self.reasoning_modules), output_dim)
        else:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Statistics
        self.routing_stats: Dict[str, int] = {name: 0 for name in self.reasoning_modules}
    
    def forward(
        self,
        x: torch.Tensor,
        return_confidence: bool = False,
        force_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with dynamic reasoning
        
        Args:
            x: Input tensor
            return_confidence: Whether to return confidence score
            force_path: Force a specific reasoning path
            
        Returns:
            Output tensor and metadata
        """
        # Project input
        h = self.input_proj(x)
        
        metadata = {}
        module_names = list(self.reasoning_modules.keys())
        
        if force_path:
            # Use specified path
            if force_path not in self.reasoning_modules:
                raise ValueError(f"Unknown reasoning path: {force_path}")
            
            h = self.reasoning_modules[force_path](h)
            selected_path = force_path
            weights = None
            
        elif self.routing_type == 'dynamic':
            # Dynamic routing
            weights, selected_idx = self.router(h)
            
            if self.use_ensemble:
                # Ensemble all paths with weights
                outputs = []
                for i, (name, module) in enumerate(self.reasoning_modules.items()):
                    out = module(h)
                    outputs.append(out * weights[:, i:i+1])
                h = torch.cat(outputs, dim=-1)
            else:
                # Single path
                selected_path = module_names[selected_idx]
                h = self.reasoning_modules[selected_path](h)
            
            metadata['routing_weights'] = weights.detach().cpu().numpy()
            
        else:
            # Default: chain of thought
            h = self.reasoning_modules['chain_of_thought'](h)
            selected_path = 'chain_of_thought'
        
        # Output projection
        output = self.output_proj(h)
        
        # Confidence
        if return_confidence:
            confidence = self.confidence_head(h if not self.use_ensemble else h[:, :self.hidden_dim])
            metadata['confidence'] = confidence
        
        # Update statistics
        if not self.use_ensemble and 'selected_path' in locals():
            self.routing_stats[selected_path] += 1
            metadata['selected_path'] = selected_path
        
        return output, metadata
    
    def reason_with_backtrack(
        self,
        x: torch.Tensor,
        max_attempts: int = 3,
        confidence_threshold: float = 0.7
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Reasoning with backtracking on low confidence
        
        Args:
            x: Input tensor
            max_attempts: Maximum reasoning attempts
            confidence_threshold: Minimum acceptable confidence
            
        Returns:
            Best output and metadata
        """
        best_output = None
        best_confidence = 0.0
        best_path = None
        attempts = []
        
        for attempt in range(max_attempts):
            output, metadata = self.forward(x, return_confidence=True)
            confidence = metadata['confidence'].mean().item()
            
            attempts.append({
                'attempt': attempt,
                'confidence': confidence,
                'path': metadata.get('selected_path', 'ensemble')
            })
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_output = output
                best_path = metadata.get('selected_path', 'ensemble')
            
            if confidence >= confidence_threshold:
                break
        
        return best_output, {
            'best_confidence': best_confidence,
            'best_path': best_path,
            'attempts': attempts
        }
    
    def get_routing_stats(self) -> Dict[str, float]:
        """Get routing statistics"""
        total = sum(self.routing_stats.values())
        if total == 0:
            return {name: 0.0 for name in self.routing_stats}
        
        return {name: count / total for name, count in self.routing_stats.items()}
    
    def reset_stats(self) -> None:
        """Reset routing statistics"""
        self.routing_stats = {name: 0 for name in self.reasoning_modules}
        if hasattr(self, 'router'):
            self.router.path_usage.zero_()


class HierarchicalReasoningEngine(ReasoningEngine):
    """
    Hierarchical Reasoning with Multiple Levels
    
    Features:
    - Coarse-to-fine reasoning
    - Progressive refinement
    - Level-specific experts
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_levels: int = 3
    ):
        super().__init__(input_dim, hidden_dim, output_dim)
        
        self.num_levels = num_levels
        
        # Level-specific reasoning
        self.level_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_levels)
        ])
        
        # Level gates
        self.level_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            for _ in range(num_levels)
        ])
        
        # Early exit heads
        self.early_exit_heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim)
            for _ in range(num_levels)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        early_exit_threshold: float = 0.9,
        return_all_levels: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Hierarchical forward pass with early exit
        
        Args:
            x: Input tensor
            early_exit_threshold: Confidence threshold for early exit
            return_all_levels: Whether to return outputs from all levels
            
        Returns:
            Output tensor and metadata
        """
        h = self.input_proj(x)
        
        level_outputs = []
        exit_level = self.num_levels - 1
        
        for level in range(self.num_levels):
            # Apply level-specific reasoning
            h = self.level_modules[level](h) + h  # Residual
            
            # Check gate / confidence
            gate = self.level_gates[level](h)
            output = self.early_exit_heads[level](h)
            
            level_outputs.append({
                'output': output,
                'gate': gate.mean().item()
            })
            
            # Early exit if confident enough
            if gate.mean().item() >= early_exit_threshold:
                exit_level = level
                break
        
        final_output = level_outputs[exit_level]['output']
        
        metadata = {
            'exit_level': exit_level,
            'total_levels': self.num_levels,
            'level_gates': [lo['gate'] for lo in level_outputs]
        }
        
        if return_all_levels:
            metadata['all_outputs'] = [lo['output'] for lo in level_outputs]
        
        return final_output, metadata
