"""
模块化 AI 架构 (Modular AI Architecture)

模块功能描述:
    支持插件式功能扩展的模块化 AI 架构。

主要功能:
    - 插件动态加载和管理
    - 模块化组件设计
    - 配置验证和异常处理
    - 热插拔支持

主要组件:
    - PluginManager: 插件管理器
    - ModulePlugin: 模块插件抽象基类
    - ModularAIArchitecture: 模块化AI架构主类

最后修改日期: 2024-12-07
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Tuple, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import importlib
import inspect
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """
    插件元数据数据类
    
    功能描述:
        描述插件的基本信息和依赖关系。
    
    属性说明:
        - name: 插件名称
        - version: 版本号
        - description: 描述
        - author: 作者
        - dependencies: 依赖列表
    """
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)


class ModulePlugin(ABC):
    """
    模块插件抽象基类
    
    功能描述:
        定义模块插件的基本接口。
    
    抽象方法:
        - metadata: 返回插件元数据
        - create_module(): 创建模块实例
    """
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    @abstractmethod
    def create_module(self, config: Dict[str, Any]) -> nn.Module:
        """Create the module instance"""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        return True
    
    def on_load(self) -> None:
        """Called when plugin is loaded"""
        pass
    
    def on_unload(self) -> None:
        """Called when plugin is unloaded"""
        pass


class PluginManager:
    """
    Plugin Manager for Dynamic Module Loading
    
    Features:
    - Dynamic plugin discovery
    - Hot reloading
    - Dependency resolution
    - Version management
    """
    
    def __init__(self, plugin_dir: Optional[str] = None):
        self.plugin_dir = Path(plugin_dir) if plugin_dir else None
        self.plugins: Dict[str, ModulePlugin] = {}
        self.loaded_modules: Dict[str, nn.Module] = {}
        
        if self.plugin_dir:
            self._discover_plugins()
    
    def _discover_plugins(self) -> None:
        """Discover plugins in plugin directory"""
        if not self.plugin_dir or not self.plugin_dir.exists():
            return
        
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            try:
                spec = importlib.util.spec_from_file_location(
                    plugin_file.stem, plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find plugin classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, ModulePlugin) and 
                        obj != ModulePlugin):
                        plugin = obj()
                        self.register_plugin(plugin)
                        
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")
    
    def register_plugin(self, plugin: ModulePlugin) -> None:
        """Register a plugin"""
        name = plugin.metadata.name
        
        if name in self.plugins:
            logger.warning(f"Plugin {name} already registered, replacing")
        
        self.plugins[name] = plugin
        plugin.on_load()
        logger.info(f"Registered plugin: {name} v{plugin.metadata.version}")
    
    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin"""
        if name not in self.plugins:
            return False
        
        plugin = self.plugins[name]
        plugin.on_unload()
        del self.plugins[name]
        
        logger.info(f"Unregistered plugin: {name}")
        return True
    
    def get_plugin(self, name: str) -> Optional[ModulePlugin]:
        """Get a plugin by name"""
        return self.plugins.get(name)
    
    def list_plugins(self) -> List[PluginMetadata]:
        """List all registered plugins"""
        return [p.metadata for p in self.plugins.values()]
    
    def create_module(
        self,
        plugin_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """Create a module from a plugin"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} not found")
        
        plugin = self.plugins[plugin_name]
        config = config or {}
        
        if not plugin.validate_config(config):
            raise ValueError(f"Invalid configuration for plugin {plugin_name}")
        
        module = plugin.create_module(config)
        self.loaded_modules[plugin_name] = module
        
        return module


class ModularBlock(nn.Module):
    """A modular block that can be dynamically configured"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_type: str = 'linear',
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = False
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Build block based on type
        layers = []
        
        if block_type == 'linear':
            layers.append(nn.Linear(in_features, out_features))
        elif block_type == 'conv1d':
            layers.append(nn.Conv1d(in_features, out_features, kernel_size=3, padding=1))
        elif block_type == 'attention':
            layers.append(nn.MultiheadAttention(in_features, num_heads=8, batch_first=True))
        
        # Batch normalization
        if batch_norm and block_type == 'linear':
            layers.append(nn.BatchNorm1d(out_features))
        
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.block = nn.Sequential(*layers) if block_type != 'attention' else layers[0]
        self.block_type = block_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_type == 'attention':
            x, _ = self.block(x, x, x)
            return x
        return self.block(x)


class ModularAIArchitecture(nn.Module):
    """
    Modular AI Architecture
    
    Features:
    - Dynamic layer configuration
    - Hot-swappable modules
    - Multi-head outputs
    - Residual connections
    - Feature routing
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        block_configs: Optional[List[Dict[str, Any]]] = None,
        use_residual: bool = True,
        multi_head: bool = False,
        num_heads: int = 1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        self.multi_head = multi_head
        
        # Default block configs
        if block_configs is None:
            block_configs = [
                {'block_type': 'linear', 'activation': 'relu', 'dropout': 0.1}
                for _ in hidden_dims
            ]
        
        # Build encoder
        self.encoder = self._build_encoder(input_dim, hidden_dims, block_configs)
        
        # Build output heads
        if multi_head:
            self.heads = nn.ModuleDict({
                f'head_{i}': nn.Linear(hidden_dims[-1], output_dim)
                for i in range(num_heads)
            })
        else:
            self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Optional residual projections
        if use_residual:
            self.residual_projections = nn.ModuleList()
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                if prev_dim != hidden_dim:
                    self.residual_projections.append(
                        nn.Linear(prev_dim, hidden_dim)
                    )
                else:
                    self.residual_projections.append(nn.Identity())
                prev_dim = hidden_dim
        
        # Plugin slots
        self.plugins: Dict[str, nn.Module] = {}
    
    def _build_encoder(
        self,
        input_dim: int,
        hidden_dims: List[int],
        block_configs: List[Dict[str, Any]]
    ) -> nn.ModuleList:
        """Build encoder layers"""
        layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, (hidden_dim, config) in enumerate(zip(hidden_dims, block_configs)):
            block = ModularBlock(
                in_features=prev_dim,
                out_features=hidden_dim,
                **config
            )
            layers.append(block)
            prev_dim = hidden_dim
        
        return layers
    
    def forward(
        self,
        x: torch.Tensor,
        head_name: Optional[str] = None
    ) -> torch.Tensor:
        """Forward pass"""
        # Encode
        h = x
        for i, layer in enumerate(self.encoder):
            new_h = layer(h)
            
            # Apply residual
            if self.use_residual:
                if hasattr(self, 'residual_projections'):
                    residual = self.residual_projections[i](h)
                    new_h = new_h + residual
            
            h = new_h
        
        # Apply plugins
        for name, plugin in self.plugins.items():
            h = plugin(h)
        
        # Output
        if self.multi_head:
            if head_name:
                return self.heads[head_name](h)
            else:
                return {name: head(h) for name, head in self.heads.items()}
        else:
            return self.output_layer(h)
    
    def add_plugin(self, name: str, module: nn.Module) -> None:
        """Add a plugin module"""
        self.plugins[name] = module
        logger.info(f"Added plugin: {name}")
    
    def remove_plugin(self, name: str) -> bool:
        """Remove a plugin module"""
        if name in self.plugins:
            del self.plugins[name]
            logger.info(f"Removed plugin: {name}")
            return True
        return False
    
    def add_head(self, name: str, output_dim: Optional[int] = None) -> None:
        """Add an output head"""
        if not self.multi_head:
            raise ValueError("Multi-head not enabled")
        
        output_dim = output_dim or self.output_dim
        hidden_dim = list(self.encoder)[-1].out_features
        
        self.heads[name] = nn.Linear(hidden_dim, output_dim)
        logger.info(f"Added head: {name}")
    
    def remove_head(self, name: str) -> bool:
        """Remove an output head"""
        if not self.multi_head:
            return False
        
        if name in self.heads:
            del self.heads[name]
            logger.info(f"Removed head: {name}")
            return True
        return False
    
    def freeze_encoder(self) -> None:
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_feature_dim(self) -> int:
        """Get the feature dimension before output"""
        return list(self.encoder)[-1].out_features
    
    @property
    def feature_dim(self) -> int:
        """Feature dimension property for compatibility"""
        return self.get_feature_dim()
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before output layer"""
        h = x
        for layer in self.encoder:
            h = layer(h)
        return h


class AdaptiveArchitecture(ModularAIArchitecture):
    """
    Self-Adapting Architecture
    
    Features:
    - Dynamic layer addition/removal
    - Automatic architecture search
    - Complexity-aware scaling
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initial_hidden: int = 256,
        max_layers: int = 10,
        growth_factor: float = 1.5
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=[initial_hidden],
            output_dim=output_dim
        )
        
        self.max_layers = max_layers
        self.growth_factor = growth_factor
        self.current_layers = 1
    
    def grow(self) -> bool:
        """Add a new layer"""
        if self.current_layers >= self.max_layers:
            return False
        
        # Get current hidden dim
        current_dim = list(self.encoder)[-1].out_features
        new_dim = int(current_dim * self.growth_factor)
        
        # Add new layer
        new_block = ModularBlock(
            in_features=current_dim,
            out_features=new_dim,
            block_type='linear',
            activation='relu',
            dropout=0.1
        )
        
        self.encoder.append(new_block)
        
        # Update output layer
        if hasattr(self, 'output_layer'):
            self.output_layer = nn.Linear(new_dim, self.output_dim)
        
        # Update residual projections
        if self.use_residual and hasattr(self, 'residual_projections'):
            self.residual_projections.append(
                nn.Linear(current_dim, new_dim) if current_dim != new_dim else nn.Identity()
            )
        
        self.current_layers += 1
        logger.info(f"Architecture grew to {self.current_layers} layers")
        
        return True
    
    def shrink(self) -> bool:
        """Remove the last layer"""
        if self.current_layers <= 1:
            return False
        
        # Remove last layer
        self.encoder = self.encoder[:-1]
        
        # Update output layer
        new_dim = list(self.encoder)[-1].out_features
        if hasattr(self, 'output_layer'):
            self.output_layer = nn.Linear(new_dim, self.output_dim)
        
        # Update residual projections
        if self.use_residual and hasattr(self, 'residual_projections'):
            self.residual_projections = self.residual_projections[:-1]
        
        self.current_layers -= 1
        logger.info(f"Architecture shrunk to {self.current_layers} layers")
        
        return True
    
    def adapt(
        self,
        performance_history: List[float],
        threshold: float = 0.01
    ) -> str:
        """
        Adapt architecture based on performance
        
        Args:
            performance_history: Recent performance scores
            threshold: Improvement threshold
            
        Returns:
            Action taken: 'grow', 'shrink', or 'none'
        """
        if len(performance_history) < 2:
            return 'none'
        
        recent_improvement = performance_history[-1] - performance_history[-2]
        
        if recent_improvement > threshold:
            # Good progress, try growing
            if self.grow():
                return 'grow'
        elif recent_improvement < -threshold:
            # Degradation, try shrinking
            if self.shrink():
                return 'shrink'
        
        return 'none'
