"""
RAG (Retrieval-Augmented Generation) module.

Augments frozen model with real-time information retrieval
for up-to-date knowledge without retraining.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn

from .config import PracticalDeploymentConfig

logger = logging.getLogger(__name__)


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
    Vector index for RAG with optional FAISS support.
    
    For production, uses FAISS for efficient similarity search.
    Falls back to NumPy for development/testing.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        index_path: Optional[str] = None,
        use_faiss: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path) if index_path else None
        self.use_faiss = use_faiss
        
        self.documents: Dict[str, RAGDocument] = {}
        self.doc_ids: List[str] = []
        
        # FAISS index (if available)
        self._faiss_index = None
        self._faiss_available = False
        
        # NumPy fallback
        self.embeddings: Optional[np.ndarray] = None
        
        # Initialize FAISS if requested
        if use_faiss:
            self._init_faiss()
        
        if self.index_path and self.index_path.exists():
            self.load()
    
    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            import faiss
            
            # Use IVF index for large-scale search
            # Start with flat index, can be upgraded later
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine after L2 norm)
            self._faiss_available = True
            logger.info("FAISS index initialized successfully")
            
        except ImportError:
            logger.warning("FAISS not available, falling back to NumPy-based search")
            self._faiss_available = False
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add document to index."""
        # Normalize embedding for cosine similarity
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        doc = RAGDocument(
            doc_id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )
        
        self.documents[doc_id] = doc
        
        # Add to index
        if self._faiss_available and self._faiss_index is not None:
            self._faiss_index.add(embedding.reshape(1, -1))
            self.doc_ids.append(doc_id)
        else:
            # NumPy fallback
            if self.embeddings is None:
                self.embeddings = embedding.reshape(1, -1)
                self.doc_ids = [doc_id]
            else:
                self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
                self.doc_ids.append(doc_id)
    
    def add_documents_batch(
        self,
        documents: List[Tuple[str, str, np.ndarray, Optional[Dict[str, Any]]]],
    ):
        """Add multiple documents efficiently."""
        if not documents:
            return
        
        embeddings_batch = []
        
        for doc_id, content, embedding, metadata in documents:
            # Normalize
            embedding = embedding.astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            doc = RAGDocument(
                doc_id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata or {},
            )
            
            self.documents[doc_id] = doc
            self.doc_ids.append(doc_id)
            embeddings_batch.append(embedding)
        
        embeddings_array = np.vstack(embeddings_batch)
        
        if self._faiss_available and self._faiss_index is not None:
            self._faiss_index.add(embeddings_array)
        else:
            if self.embeddings is None:
                self.embeddings = embeddings_array
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings_array])
        
        logger.info(f"Added {len(documents)} documents to index")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[RAGDocument, float]]:
        """Search for similar documents."""
        if len(self.doc_ids) == 0:
            return []
        
        # Normalize query
        query_embedding = query_embedding.astype(np.float32)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        if self._faiss_available and self._faiss_index is not None:
            return self._search_faiss(query_embedding, top_k, threshold)
        else:
            return self._search_numpy(query_embedding, top_k, threshold)
    
    def _search_faiss(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        threshold: float,
    ) -> List[Tuple[RAGDocument, float]]:
        """Search using FAISS."""
        query = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self._faiss_index.search(query, min(top_k, len(self.doc_ids)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            if score >= threshold:
                doc_id = self.doc_ids[idx]
                if doc_id in self.documents:
                    results.append((self.documents[doc_id], float(score)))
        
        return results
    
    def _search_numpy(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        threshold: float,
    ) -> List[Tuple[RAGDocument, float]]:
        """Search using NumPy (fallback)."""
        if self.embeddings is None:
            return []
        
        # Cosine similarity (embeddings are already normalized)
        similarities = self.embeddings @ query_embedding
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                doc = self.documents[self.doc_ids[idx]]
                results.append((doc, score))
        
        return results
    
    def remove_document(self, doc_id: str):
        """Remove a document from the index."""
        if doc_id not in self.documents:
            return
        
        # Remove from documents
        del self.documents[doc_id]
        
        # Rebuild index without the document
        # (FAISS doesn't support efficient removal)
        idx = self.doc_ids.index(doc_id)
        self.doc_ids.pop(idx)
        
        if self._faiss_available:
            # Rebuild FAISS index
            self._rebuild_faiss_index()
        else:
            # Remove from NumPy array
            if self.embeddings is not None:
                self.embeddings = np.delete(self.embeddings, idx, axis=0)
                if len(self.embeddings) == 0:
                    self.embeddings = None
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from documents."""
        if not self._faiss_available:
            return
        
        import faiss
        
        self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        if self.doc_ids:
            embeddings = np.vstack([
                self.documents[doc_id].embedding.reshape(1, -1)
                for doc_id in self.doc_ids
            ])
            self._faiss_index.add(embeddings)
    
    def save(self):
        """Save index to disk."""
        if self.index_path is None:
            raise ValueError("No index path specified")
        
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self._faiss_available and self._faiss_index is not None:
            import faiss
            faiss.write_index(
                self._faiss_index,
                str(self.index_path / "faiss.index")
            )
        elif self.embeddings is not None:
            np.save(self.index_path / "embeddings.npy", self.embeddings)
        
        # Save documents (without embeddings, they're in the index)
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
        
        # Save config
        config = {
            "embedding_dim": self.embedding_dim,
            "use_faiss": self._faiss_available,
            "num_documents": len(self.documents),
        }
        
        with open(self.index_path / "config.json", "w") as f:
            json.dump(config, f)
        
        logger.info(f"Saved RAG index with {len(self.documents)} documents")
    
    def load(self):
        """Load index from disk."""
        if self.index_path is None or not self.index_path.exists():
            return
        
        # Load config
        config_path = self.index_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            self.embedding_dim = config.get("embedding_dim", self.embedding_dim)
        
        # Load doc IDs
        ids_path = self.index_path / "doc_ids.json"
        if ids_path.exists():
            with open(ids_path, "r") as f:
                self.doc_ids = json.load(f)
        
        # Load FAISS index or embeddings
        faiss_path = self.index_path / "faiss.index"
        if faiss_path.exists() and self._faiss_available:
            import faiss
            self._faiss_index = faiss.read_index(str(faiss_path))
        else:
            emb_path = self.index_path / "embeddings.npy"
            if emb_path.exists():
                self.embeddings = np.load(emb_path)
        
        # Load documents
        docs_path = self.index_path / "documents.json"
        if docs_path.exists():
            with open(docs_path, "r") as f:
                doc_data = json.load(f)
            
            for doc_id, data in doc_data.items():
                # Get embedding from index
                if doc_id in self.doc_ids:
                    idx = self.doc_ids.index(doc_id)
                    if self._faiss_available and self._faiss_index is not None:
                        embedding = self._faiss_index.reconstruct(idx)
                    elif self.embeddings is not None:
                        embedding = self.embeddings[idx]
                    else:
                        embedding = np.zeros(self.embedding_dim)
                else:
                    embedding = np.zeros(self.embedding_dim)
                
                self.documents[doc_id] = RAGDocument(
                    doc_id=doc_id,
                    content=data["content"],
                    embedding=embedding,
                    metadata=data.get("metadata", {}),
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                )
        
        logger.info(f"Loaded RAG index with {len(self.documents)} documents")
    
    def __len__(self) -> int:
        return len(self.documents)


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
            use_faiss=config.rag_use_faiss,
        )
        
        # Cache for embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_max_size = 10000
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        if self.embedding_model is not None:
            # Use embedding model
            import torch
            
            with torch.no_grad():
                # Tokenize and encode
                # This is a placeholder - actual implementation depends on model
                embedding = np.random.randn(768).astype(np.float32)
        else:
            # Fallback: deterministic hash-based embedding
            rng = np.random.default_rng(hash(text) % (2**32))
            embedding = rng.standard_normal(768).astype(np.float32)
        
        # Cache management
        if len(self.embedding_cache) >= self.cache_max_size:
            # Remove oldest entries
            keys_to_remove = list(self.embedding_cache.keys())[:1000]
            for key in keys_to_remove:
                del self.embedding_cache[key]
        
        self.embedding_cache[cache_key] = embedding
        return embedding
    
    def add_knowledge(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add new knowledge to the RAG system. Returns doc_id."""
        doc_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()
        embedding = self.compute_embedding(content)
        
        self.index.add_document(doc_id, content, embedding, metadata)
        return doc_id
    
    def add_knowledge_batch(
        self,
        documents: List[Tuple[str, Optional[Dict[str, Any]]]],
    ) -> List[str]:
        """Add multiple documents efficiently."""
        batch = []
        doc_ids = []
        
        for content, metadata in documents:
            doc_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()
            embedding = self.compute_embedding(content)
            batch.append((doc_id, content, embedding, metadata))
            doc_ids.append(doc_id)
        
        self.index.add_documents_batch(batch)
        return doc_ids
    
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
        self.index.save()
        logger.info("Updated RAG index")
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Cleared embedding cache")
