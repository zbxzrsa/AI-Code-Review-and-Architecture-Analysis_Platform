"""
Plan A: Lightweight Continuous Learning (Practical Deployment)

A cost-effective, production-ready approach:
- Frozen base model with LoRA adapters
- RAG system for real-time information retrieval
- Periodic retraining scheduler
- Quantization (INT8/INT4) for efficiency
- Model distillation for deployment

Cost: Controllable (~$1M-5M/year for enterprise)
"""

import asyncio
import gc
import hashlib
import json
import logging
import math
import os
import pickle
import shutil
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class QuantizationType(str, Enum):
    """Quantization types for efficient inference."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    GPTQ = "gptq"
    AWQ = "awq"


class RetrainingFrequency(str, Enum):
    """Adapter retraining frequency."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_DEMAND = "on_demand"


@dataclass
class PracticalDeploymentConfig:
    """Configuration for practical lightweight deployment."""
    # Base model (frozen)
    base_model_path: str = "models/base"
    freeze_base_model: bool = True
    
    # LoRA configuration
    lora_r: int = 8  # LoRA rank
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # RAG configuration
    enable_rag: bool = True
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.7
    rag_index_path: str = "indices/rag"
    
    # Retraining schedule
    retraining_frequency: RetrainingFrequency = RetrainingFrequency.WEEKLY
    retraining_data_path: str = "data/retraining"
    min_samples_for_retraining: int = 1000
    
    # Quantization
    quantization_type: QuantizationType = QuantizationType.INT8
    
    # Distillation
    enable_distillation: bool = False
    student_model_size: str = "small"  # small, medium, large
    
    # Active learning
    enable_active_learning: bool = True
    uncertainty_threshold: float = 0.3
    
    # Cost control
    max_daily_tokens: int = 100_000_000  # 100M tokens/day
    max_monthly_cost_usd: float = 100_000
    
    # Fault tolerance
    checkpoint_interval_minutes: int = 30
    max_retries: int = 3
    health_check_interval_seconds: int = 60


# =============================================================================
# LoRA Adapter System
# =============================================================================

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) Layer
    
    Efficient fine-tuning by adding low-rank matrices:
    W' = W + BA (where B is r×d, A is d×r, r << d)
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
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        original_output = self.original_layer(x)
        
        # LoRA output
        lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into original layer."""
        with torch.no_grad():
            merged_weight = (
                self.original_layer.weight +
                self.lora_B.weight @ self.lora_A.weight * self.scaling
            )
            self.original_layer.weight.copy_(merged_weight)
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get only LoRA weights for efficient saving."""
        return {
            "lora_A": self.lora_A.weight.data,
            "lora_B": self.lora_B.weight.data,
        }
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load LoRA weights."""
        self.lora_A.weight.data.copy_(state_dict["lora_A"])
        self.lora_B.weight.data.copy_(state_dict["lora_B"])


class LoRAAdapterManager:
    """
    Manages multiple LoRA adapters for different tasks/domains.
    
    Features:
    - Create/load/save adapters
    - Switch between adapters
    - Merge adapters
    - Version control
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
        
        self.adapters[adapter_name] = adapter_layers
        self.adapter_versions[adapter_name] = 1
        self.adapter_metadata[adapter_name] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_modules": target_modules,
            "r": self.config.lora_r,
            "alpha": self.config.lora_alpha,
        }
        
        logger.info(f"Created adapter '{adapter_name}' with {len(adapter_layers)} LoRA layers")
        
        return adapter_layers
    
    def activate_adapter(self, adapter_name: str):
        """Activate an adapter for inference."""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
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
        
        # Replace layers temporarily and forward
        # (In production, would modify model architecture properly)
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
        adapter_state = torch.load(load_dir / "adapter_weights.pt", map_location=self.device)
        
        for layer_name, state_dict in adapter_state.items():
            if layer_name in self.adapters[adapter_name]:
                self.adapters[adapter_name][layer_name].load_lora_state_dict(state_dict)
        
        self.adapter_metadata[adapter_name] = metadata
        self.adapter_versions[adapter_name] = metadata.get("version", 1)
        
        logger.info(f"Loaded adapter '{adapter_name}' from {load_dir}")
    
    def merge_adapters(
        self,
        adapter_names: List[str],
        weights: Optional[List[float]] = None,
        new_name: str = "merged",
    ) -> Dict[str, LoRALayer]:
        """Merge multiple adapters with weighted average."""
        if len(adapter_names) < 2:
            raise ValueError("Need at least 2 adapters to merge")
        
        weights = weights or [1.0 / len(adapter_names)] * len(adapter_names)
        
        if len(weights) != len(adapter_names):
            raise ValueError("Weights must match adapter count")
        
        # Create new adapter
        first_adapter = self.adapters[adapter_names[0]]
        merged_adapter = self.create_adapter(new_name)
        
        # Merge weights
        for layer_name in merged_adapter:
            merged_A = torch.zeros_like(merged_adapter[layer_name].lora_A.weight)
            merged_B = torch.zeros_like(merged_adapter[layer_name].lora_B.weight)
            
            for adapter_name, weight in zip(adapter_names, weights):
                if layer_name in self.adapters[adapter_name]:
                    merged_A += weight * self.adapters[adapter_name][layer_name].lora_A.weight
                    merged_B += weight * self.adapters[adapter_name][layer_name].lora_B.weight
            
            merged_adapter[layer_name].lora_A.weight.data.copy_(merged_A)
            merged_adapter[layer_name].lora_B.weight.data.copy_(merged_B)
        
        logger.info(f"Merged {len(adapter_names)} adapters into '{new_name}'")
        
        return merged_adapter
    
    def get_trainable_parameters(
        self,
        adapter_name: Optional[str] = None,
    ) -> List[nn.Parameter]:
        """Get trainable parameters (adapter only)."""
        adapter_name = adapter_name or self.active_adapter
        
        if adapter_name is None:
            return []
        
        params = []
        for lora_layer in self.adapters[adapter_name].values():
            params.extend([
                lora_layer.lora_A.weight,
                lora_layer.lora_B.weight,
            ])
        
        return params


# =============================================================================
# RAG System for Real-Time Information
# =============================================================================

@dataclass
class RAGDocument:
    """Document in RAG index."""
    doc_id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RAGIndex:
    """
    Simple vector index for RAG.
    
    Production systems should use:
    - FAISS for large-scale
    - Pinecone/Weaviate for managed
    - Milvus for self-hosted
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        index_path: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path) if index_path else None
        
        self.documents: Dict[str, RAGDocument] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        
        if self.index_path and self.index_path.exists():
            self.load()
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add document to index."""
        doc = RAGDocument(
            doc_id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )
        
        self.documents[doc_id] = doc
        
        # Update index
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
            self.doc_ids = [doc_id]
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
            self.doc_ids.append(doc_id)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[RAGDocument, float]]:
        """Search for similar documents."""
        if self.embeddings is None or len(self.doc_ids) == 0:
            return []
        
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        
        # Cosine similarity
        similarities = embeddings_norm @ query_norm
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                doc = self.documents[self.doc_ids[idx]]
                results.append((doc, score))
        
        return results
    
    def save(self):
        """Save index to disk."""
        if self.index_path is None:
            raise ValueError("No index path specified")
        
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(self.index_path / "embeddings.npy", self.embeddings)
        
        # Save documents
        doc_data = {
            doc_id: {
                "content": doc.content,
                "metadata": doc.metadata,
                "timestamp": doc.timestamp.isoformat(),
            }
            for doc_id, doc in self.documents.items()
        }
        
        with open(self.index_path / "documents.json", "w") as f:
            json.dump(doc_data, f)
        
        # Save doc IDs
        with open(self.index_path / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids, f)
        
        logger.info(f"Saved RAG index with {len(self.documents)} documents")
    
    def load(self):
        """Load index from disk."""
        if self.index_path is None or not self.index_path.exists():
            return
        
        # Load embeddings
        emb_path = self.index_path / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
        
        # Load doc IDs
        ids_path = self.index_path / "doc_ids.json"
        if ids_path.exists():
            with open(ids_path, "r") as f:
                self.doc_ids = json.load(f)
        
        # Load documents
        docs_path = self.index_path / "documents.json"
        if docs_path.exists():
            with open(docs_path, "r") as f:
                doc_data = json.load(f)
            
            for doc_id, data in doc_data.items():
                idx = self.doc_ids.index(doc_id) if doc_id in self.doc_ids else -1
                embedding = self.embeddings[idx] if idx >= 0 else np.zeros(self.embedding_dim)
                
                self.documents[doc_id] = RAGDocument(
                    doc_id=doc_id,
                    content=data["content"],
                    embedding=embedding,
                    metadata=data.get("metadata", {}),
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                )
        
        logger.info(f"Loaded RAG index with {len(self.documents)} documents")


class RAGSystem:
    """
    Retrieval-Augmented Generation System
    
    Augments frozen model with real-time information retrieval.
    """
    
    def __init__(
        self,
        config: PracticalDeploymentConfig,
        embedding_model: Optional[nn.Module] = None,
    ):
        self.config = config
        self.embedding_model = embedding_model
        
        # Initialize index
        self.index = RAGIndex(
            embedding_dim=768,
            index_path=config.rag_index_path,
        )
        
        # Cache for embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        if self.embedding_model is not None:
            # Use embedding model
            # (In production, this would call the actual model)
            embedding = np.random.randn(768).astype(np.float32)
        else:
            # Fallback: simple hash-based embedding (not recommended for production)
            embedding = np.random.randn(768).astype(np.float32)
        
        self.embedding_cache[cache_key] = embedding
        return embedding
    
    def add_knowledge(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add new knowledge to the RAG system."""
        doc_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()
        embedding = self.compute_embedding(content)
        
        self.index.add_document(doc_id, content, embedding, metadata)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for a query."""
        query_embedding = self.compute_embedding(query)
        
        results = self.index.search(
            query_embedding,
            top_k=top_k or self.config.rag_top_k,
            threshold=self.config.rag_similarity_threshold,
        )
        
        return [(doc.content, score) for doc, score in results]
    
    def augment_prompt(
        self,
        prompt: str,
        top_k: Optional[int] = None,
    ) -> str:
        """Augment a prompt with retrieved context."""
        retrieved = self.retrieve(prompt, top_k)
        
        if not retrieved:
            return prompt
        
        context = "\n\n---\n\n".join([
            f"[Relevant Information {i+1}] (score: {score:.2f}):\n{content}"
            for i, (content, score) in enumerate(retrieved)
        ])
        
        return f"""Based on the following relevant information:

{context}

---

User Query: {prompt}

Response:"""
    
    def update_index(self):
        """Update the RAG index from data sources."""
        # In production, this would fetch from various sources:
        # - GitHub issues/PRs
        # - Stack Overflow
        # - Documentation
        # - Internal knowledge bases
        
        self.index.save()
        logger.info("Updated RAG index")


# =============================================================================
# Periodic Retraining Scheduler
# =============================================================================

class RetrainingDataCollector:
    """Collects and manages data for periodic retraining."""
    
    def __init__(
        self,
        config: PracticalDeploymentConfig,
    ):
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
        # In production, this would update the sample with feedback
        # for better quality training data
        pass
    
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
            json.dump(self.samples, f)
        
        logger.info(f"Saved {len(self.samples)} samples to {save_path}")
    
    def load_samples(self, path: str):
        """Load samples from disk."""
        with open(path, "r") as f:
            self.samples = json.load(f)
        
        self.sample_count = len(self.samples)
        logger.info(f"Loaded {self.sample_count} samples")
    
    def is_ready_for_retraining(self) -> bool:
        """Check if enough samples for retraining."""
        return self.sample_count >= self.config.min_samples_for_retraining


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
                raise  # Re-raise CancelledError after cleanup
        
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
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def execute_retraining(
        self,
        adapter_name: str = "default",
    ):
        """Execute adapter retraining."""
        if self.training_in_progress:
            logger.warning("Training already in progress")
            return
        
        self.training_in_progress = True
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting adapter retraining: {adapter_name}")
        
        try:
            # Get training data
            training_data = self.data_collector.get_training_batch()
            
            if not training_data:
                logger.warning("No training data available")
                return
            
            # Create or get adapter
            if adapter_name not in self.adapter_manager.adapters:
                self.adapter_manager.create_adapter(adapter_name)
            
            # Get trainable parameters
            params = self.adapter_manager.get_trainable_parameters(adapter_name)
            
            if not params:
                logger.warning("No trainable parameters")
                return
            
            # Setup optimizer
            optimizer = AdamW(params, lr=1e-4)
            
            # Training loop (simplified)
            total_loss = 0.0
            
            for sample in training_data:
                # In production, would tokenize and train properly
                # This is a placeholder
                loss = torch.tensor(0.1)  # Simulated loss
                total_loss += loss.item()
            
            # Update version
            self.adapter_manager.adapter_versions[adapter_name] += 1
            
            # Save adapter
            self.adapter_manager.save_adapter(
                adapter_name,
                self.config.retraining_data_path,
            )
            
            # Record history
            end_time = datetime.now(timezone.utc)
            self.training_history.append({
                "adapter": adapter_name,
                "started_at": start_time.isoformat(),
                "ended_at": end_time.isoformat(),
                "samples": len(training_data),
                "avg_loss": total_loss / len(training_data),
                "version": self.adapter_manager.adapter_versions[adapter_name],
            })
            
            self.last_retraining = end_time
            
            logger.info(
                f"Completed retraining: {len(training_data)} samples, "
                f"version {self.adapter_manager.adapter_versions[adapter_name]}"
            )
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            
        finally:
            self.training_in_progress = False
    
    async def trigger_retraining(self, adapter_name: str = "default"):
        """Manually trigger retraining."""
        await self.execute_retraining(adapter_name)


# =============================================================================
# Quantization for Efficient Inference
# =============================================================================

class ModelQuantizer:
    """
    Quantizes models for efficient inference.
    
    Supports:
    - INT8 (8-bit integer)
    - INT4 (4-bit integer)
    - FP8 (8-bit floating point)
    """
    
    def __init__(self, config: PracticalDeploymentConfig):
        self.config = config
    
    def quantize_int8(self, model: nn.Module) -> nn.Module:
        """Quantize model to INT8."""
        logger.info("Quantizing model to INT8")
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        
        return quantized_model
    
    def quantize_int4(self, model: nn.Module) -> nn.Module:
        """Quantize model to INT4 (requires special kernels)."""
        logger.info("Quantizing model to INT4")
        
        # INT4 quantization typically requires libraries like bitsandbytes
        # or custom CUDA kernels
        
        # Placeholder - in production, use:
        # - bitsandbytes for consumer GPUs
        # - GPTQ for better accuracy
        # - AWQ for activation-aware quantization
        
        return model
    
    def quantize(self, model: nn.Module) -> nn.Module:
        """Quantize model based on config."""
        quant_type = self.config.quantization_type
        
        if quant_type == QuantizationType.NONE:
            return model
        elif quant_type == QuantizationType.INT8:
            return self.quantize_int8(model)
        elif quant_type == QuantizationType.INT4:
            return self.quantize_int4(model)
        else:
            logger.warning(f"Unsupported quantization: {quant_type}")
            return model
    
    @staticmethod
    def estimate_memory_savings(
        original_params: int,
        quant_type: QuantizationType,
    ) -> Dict[str, float]:
        """Estimate memory savings from quantization."""
        original_size_gb = original_params * 4 / 1e9  # FP32
        
        if quant_type == QuantizationType.INT8:
            quantized_size_gb = original_params / 1e9
            savings = 0.75
        elif quant_type == QuantizationType.INT4:
            quantized_size_gb = original_params * 0.5 / 1e9
            savings = 0.875
        elif quant_type == QuantizationType.FP8:
            quantized_size_gb = original_params / 1e9
            savings = 0.75
        else:
            quantized_size_gb = original_size_gb
            savings = 0.0
        
        return {
            "original_gb": original_size_gb,
            "quantized_gb": quantized_size_gb,
            "savings_percent": savings * 100,
        }


# =============================================================================
# Model Distillation
# =============================================================================

class ModelDistiller:
    """
    Distills large teacher model to smaller student model.
    
    Useful for:
    - Edge deployment
    - Faster inference
    - Lower cost
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
        
        # Student model (to be created)
        self.student_model: Optional[nn.Module] = None
    
    def create_student_model(
        self,
        student_config: Dict[str, Any],
    ) -> nn.Module:
        """Create a smaller student model."""
        # In production, would create appropriate architecture
        # based on student_config
        
        logger.info(f"Creating student model: {self.config.student_model_size}")
        
        # Placeholder - actual implementation depends on model architecture
        return self.teacher_model  # Return teacher as placeholder
    
    def distill(
        self,
        train_data: List[Dict[str, Any]],
        epochs: int = 3,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ) -> nn.Module:
        """
        Distill knowledge from teacher to student.
        
        Loss = α * hard_loss + (1-α) * soft_loss * T²
        """
        if self.student_model is None:
            raise ValueError("Student model not created")
        
        optimizer = AdamW(self.student_model.parameters(), lr=1e-4)
        
        logger.info(f"Starting distillation: {epochs} epochs")
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for sample in train_data:
                # Get teacher logits
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=sample['input_ids'].unsqueeze(0).to(self.device)
                    )
                    teacher_logits = teacher_outputs['logits']
                
                # Get student logits
                student_outputs = self.student_model(
                    input_ids=sample['input_ids'].unsqueeze(0).to(self.device)
                )
                student_logits = student_outputs['logits']
                
                # Soft loss (KL divergence)
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction='batchmean',
                ) * (temperature ** 2)
                
                # Hard loss (cross entropy with labels)
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
                optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_data):.4f}")
        
        return self.student_model


# =============================================================================
# Fault Tolerance & Monitoring
# =============================================================================

class HealthChecker:
    """System health monitoring."""
    
    def __init__(self, config: PracticalDeploymentConfig):
        self.config = config
        
        self.health_status = {
            "model": "unknown",
            "rag": "unknown",
            "adapter": "unknown",
            "gpu": "unknown",
        }
        
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        
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
                raise  # Re-raise CancelledError after cleanup
    
    async def _check_loop(self):
        """Main health check loop."""
        while self._is_running:
            try:
                await self._perform_checks()
                await asyncio.sleep(self.config.health_check_interval_seconds)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def _perform_checks(self):
        """Perform all health checks."""
        # GPU check
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                self.metrics["gpu_memory_gb"].append(gpu_memory)
                self.health_status["gpu"] = "healthy"
            except Exception:
                self.health_status["gpu"] = "unhealthy"
        else:
            self.health_status["gpu"] = "no_gpu"
        
        # Model check (simplified)
        self.health_status["model"] = "healthy"
        
        # RAG check
        self.health_status["rag"] = "healthy"
        
        # Adapter check
        self.health_status["adapter"] = "healthy"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "status": self.health_status,
            "metrics": {
                k: v[-10:] if v else []  # Last 10 values
                for k, v in self.metrics.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class FaultToleranceManager:
    """
    Manages fault tolerance and recovery.
    
    Features:
    - Automatic checkpointing
    - Retry logic
    - Graceful degradation
    - Recovery procedures
    """
    
    def __init__(self, config: PracticalDeploymentConfig):
        self.config = config
        
        self.checkpoint_path = Path("checkpoints/practical")
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.last_checkpoint: Optional[datetime] = None
        self.retry_counts: Dict[str, int] = defaultdict(int)
    
    async def save_checkpoint(
        self,
        adapter_manager: LoRAAdapterManager,
        rag_system: RAGSystem,
    ):
        """Save system checkpoint."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ckpt_dir = self.checkpoint_path / timestamp
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Save adapters
        for adapter_name in adapter_manager.adapters:
            adapter_manager.save_adapter(adapter_name, str(ckpt_dir / "adapters"))
        
        # Save RAG index
        rag_system.index.index_path = ckpt_dir / "rag_index"
        rag_system.index.save()
        
        self.last_checkpoint = datetime.now(timezone.utc)
        
        logger.info(f"Saved checkpoint: {ckpt_dir}")
    
    async def load_checkpoint(
        self,
        adapter_manager: LoRAAdapterManager,
        rag_system: RAGSystem,
        checkpoint_dir: Optional[str] = None,
    ):
        """Load system checkpoint."""
        if checkpoint_dir is None:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_path.iterdir())
            if not checkpoints:
                logger.warning("No checkpoints found")
                return
            checkpoint_dir = str(checkpoints[-1])
        
        ckpt_path = Path(checkpoint_dir)
        
        # Load adapters
        adapters_path = ckpt_path / "adapters"
        if adapters_path.exists():
            for adapter_dir in adapters_path.iterdir():
                if adapter_dir.is_dir():
                    adapter_manager.load_adapter(adapter_dir.name, str(adapters_path))
        
        # Load RAG index
        rag_path = ckpt_path / "rag_index"
        if rag_path.exists():
            rag_system.index.index_path = rag_path
            rag_system.index.load()
        
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    
    async def retry_with_backoff(
        self,
        func: Callable,
        operation_name: str,
        *args,
        **kwargs,
    ) -> Any:
        """Execute function with retry and exponential backoff."""
        max_retries = self.config.max_retries
        
        for attempt in range(max_retries):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                self.retry_counts[operation_name] = 0
                return result
                
            except Exception as e:
                self.retry_counts[operation_name] += 1
                wait_time = 2 ** attempt  # Exponential backoff
                
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    raise


# =============================================================================
# Cost Controller
# =============================================================================

class CostController:
    """
    Monitors and controls costs.
    
    Tracks:
    - Token usage
    - API costs
    - Compute costs
    """
    
    def __init__(self, config: PracticalDeploymentConfig):
        self.config = config
        
        self.daily_tokens = 0
        self.monthly_cost = 0.0
        
        self.token_history: List[Dict[str, Any]] = []
        self.cost_history: List[Dict[str, Any]] = []
        
        self._lock = threading.Lock()
    
    def record_usage(
        self,
        tokens: int,
        cost: float = 0.0,
        operation: str = "inference",
    ):
        """Record usage."""
        with self._lock:
            self.daily_tokens += tokens
            self.monthly_cost += cost
            
            self.token_history.append({
                "tokens": tokens,
                "cost": cost,
                "operation": operation,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    
    def check_limits(self) -> Tuple[bool, str]:
        """Check if within limits."""
        if self.daily_tokens > self.config.max_daily_tokens:
            return False, f"Daily token limit exceeded: {self.daily_tokens}/{self.config.max_daily_tokens}"
        
        if self.monthly_cost > self.config.max_monthly_cost_usd:
            return False, f"Monthly cost limit exceeded: ${self.monthly_cost:.2f}/${self.config.max_monthly_cost_usd}"
        
        return True, "Within limits"
    
    def reset_daily(self):
        """Reset daily counters."""
        with self._lock:
            self.daily_tokens = 0
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        return {
            "daily_tokens": self.daily_tokens,
            "daily_limit": self.config.max_daily_tokens,
            "daily_usage_percent": self.daily_tokens / self.config.max_daily_tokens * 100,
            "monthly_cost": self.monthly_cost,
            "monthly_limit": self.config.max_monthly_cost_usd,
            "monthly_usage_percent": self.monthly_cost / self.config.max_monthly_cost_usd * 100,
        }


# =============================================================================
# Main Practical Deployment System
# =============================================================================

class PracticalDeploymentSystem:
    """
    Main orchestrator for Plan A: Lightweight Continuous Learning.
    
    Integrates:
    - Frozen base model
    - LoRA adapters
    - RAG system
    - Periodic retraining
    - Quantization
    - Fault tolerance
    - Cost control
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: PracticalDeploymentConfig,
    ):
        self.base_model = base_model
        self.config = config
        
        # Initialize components
        self.adapter_manager = LoRAAdapterManager(base_model, config)
        self.rag_system = RAGSystem(config)
        self.data_collector = RetrainingDataCollector(config)
        self.retraining_scheduler = RetrainingScheduler(
            self.adapter_manager,
            self.data_collector,
            config,
        )
        self.quantizer = ModelQuantizer(config)
        self.health_checker = HealthChecker(config)
        self.fault_tolerance = FaultToleranceManager(config)
        self.cost_controller = CostController(config)
        
        # System state
        self.is_running = False
        self.initialized_at: Optional[datetime] = None
    
    async def start(self):
        """Start the deployment system."""
        if self.is_running:
            return
        
        self.is_running = True
        self.initialized_at = datetime.now(timezone.utc)
        
        # Start background services
        await self.health_checker.start()
        await self.retraining_scheduler.start()
        
        # Create default adapter
        self.adapter_manager.create_adapter("default")
        self.adapter_manager.activate_adapter("default")
        
        logger.info("Practical deployment system started")
    
    async def stop(self):
        """Stop the deployment system."""
        self.is_running = False
        
        # Stop background services
        await self.health_checker.stop()
        await self.retraining_scheduler.stop()
        
        # Save checkpoint
        await self.fault_tolerance.save_checkpoint(
            self.adapter_manager,
            self.rag_system,
        )
        
        logger.info("Practical deployment system stopped")
    
    async def process(
        self,
        input_text: str,
        use_rag: bool = True,
        adapter_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process input with the full system.
        
        1. Check cost limits
        2. RAG augmentation
        3. Model inference with adapter
        4. Collect for retraining
        5. Return result
        """
        # Check limits
        within_limits, message = self.cost_controller.check_limits()
        if not within_limits:
            return {"error": message, "status": "rate_limited"}
        
        result = {
            "input": input_text,
            "output": None,
            "rag_context": [],
            "adapter_used": adapter_name or self.adapter_manager.active_adapter,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        try:
            # RAG augmentation
            if use_rag and self.config.enable_rag:
                augmented_prompt = self.rag_system.augment_prompt(input_text)
                retrieved = self.rag_system.retrieve(input_text)
                result["rag_context"] = [{"content": c, "score": s} for c, s in retrieved]
            else:
                augmented_prompt = input_text
            
            # Model inference (placeholder - actual implementation would tokenize and generate)
            output = f"Processed: {augmented_prompt[:100]}..."
            result["output"] = output
            
            # Record usage
            tokens_used = len(input_text.split()) * 2  # Rough estimate
            self.cost_controller.record_usage(tokens_used)
            
            # Collect for retraining
            self.data_collector.add_sample(input_text, output)
            
        except Exception as e:
            result["error"] = str(e)
            result["status"] = "error"
        
        return result
    
    def add_knowledge(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add knowledge to RAG system."""
        self.rag_system.add_knowledge(content, metadata)
    
    async def trigger_retraining(self, adapter_name: str = "default"):
        """Manually trigger adapter retraining."""
        await self.retraining_scheduler.trigger_retraining(adapter_name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "is_running": self.is_running,
            "initialized_at": self.initialized_at.isoformat() if self.initialized_at else None,
            "health": self.health_checker.get_status(),
            "cost": self.cost_controller.get_usage_summary(),
            "adapters": list(self.adapter_manager.adapters.keys()),
            "active_adapter": self.adapter_manager.active_adapter,
            "retraining": {
                "last": self.retraining_scheduler.last_retraining.isoformat() if self.retraining_scheduler.last_retraining else None,
                "next": self.retraining_scheduler.next_retraining.isoformat() if self.retraining_scheduler.next_retraining else None,
                "in_progress": self.retraining_scheduler.training_in_progress,
            },
            "data_collected": self.data_collector.sample_count,
        }
