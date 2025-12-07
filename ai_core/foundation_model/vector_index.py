"""
Production Vector Indexing Module

Provides high-performance vector indexing with:
- FAISS: For high-performance CPU/GPU vector search
- Milvus: For distributed, scalable vector database
- NumPy fallback: For development/testing

Optimized for:
- Batch operations
- Memory efficiency
- Production scalability
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    """Supported vector index types."""
    NUMPY = "numpy"      # Simple NumPy (development)
    FAISS_FLAT = "faiss_flat"    # FAISS flat index (exact search)
    FAISS_IVF = "faiss_ivf"      # FAISS IVF index (approximate)
    FAISS_HNSW = "faiss_hnsw"    # FAISS HNSW index (fast approximate)
    MILVUS = "milvus"    # Milvus distributed index


@dataclass
class IndexConfig:
    """Configuration for vector index."""
    index_type: IndexType = IndexType.FAISS_FLAT
    embedding_dim: int = 768
    metric: str = "cosine"  # "cosine", "l2", "ip"
    
    # FAISS IVF settings
    nlist: int = 100  # Number of clusters
    nprobe: int = 10  # Clusters to search
    
    # FAISS HNSW settings
    M: int = 32  # Number of connections per layer
    ef_construction: int = 200  # Construction quality
    ef_search: int = 64  # Search quality
    
    # Milvus settings
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "vectors"
    
    # Batch settings
    batch_size: int = 1000
    num_threads: int = 4


class BaseVectorIndex(ABC):
    """Abstract base class for vector indices."""
    
    @abstractmethod
    def add(self, embeddings: np.ndarray, ids: Optional[List[str]] = None) -> List[str]:
        """Add embeddings to index. Returns list of IDs."""
        pass
    
    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors. Returns list of (id, score)."""
        pass
    
    @abstractmethod
    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """Batch search. Returns list of results per query."""
        pass
    
    @abstractmethod
    def remove(self, ids: List[str]):
        """Remove vectors by ID."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load index from disk."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return number of vectors in index."""
        pass


class NumpyVectorIndex(BaseVectorIndex):
    """Simple NumPy-based vector index for development."""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.embeddings: Optional[np.ndarray] = None
        self.ids: List[str] = []
        self._id_counter = 0
    
    def add(
        self,
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add embeddings to index."""
        embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity
        if self.config.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            embeddings = embeddings / norms
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"vec_{self._id_counter + i}" for i in range(len(embeddings))]
        self._id_counter += len(embeddings)
        
        # Add to storage
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.ids.extend(ids)
        return ids
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        if self.embeddings is None or len(self.ids) == 0:
            return []
        
        query = query.astype(np.float32).flatten()
        
        # Normalize query
        if self.config.metric == "cosine":
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm
        
        # Compute similarity
        if self.config.metric in ("cosine", "ip"):
            scores = self.embeddings @ query
        else:  # L2
            scores = -np.linalg.norm(self.embeddings - query, axis=1)
        
        # Get top-k
        k = min(k, len(self.ids))
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold:
                results.append((self.ids[idx], score))
        
        return results
    
    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """Batch search for similar vectors."""
        if self.embeddings is None or len(self.ids) == 0:
            return [[] for _ in range(len(queries))]
        
        queries = queries.astype(np.float32)
        
        # Normalize queries
        if self.config.metric == "cosine":
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            queries = queries / norms
        
        # Batch similarity computation
        if self.config.metric in ("cosine", "ip"):
            all_scores = queries @ self.embeddings.T
        else:  # L2
            # Using broadcasting for efficient L2 computation
            all_scores = -np.sqrt(
                np.sum(queries**2, axis=1, keepdims=True) +
                np.sum(self.embeddings**2, axis=1) -
                2 * queries @ self.embeddings.T
            )
        
        results = []
        k = min(k, len(self.ids))
        
        for scores in all_scores:
            top_indices = np.argsort(scores)[-k:][::-1]
            query_results = [
                (self.ids[idx], float(scores[idx]))
                for idx in top_indices
            ]
            results.append(query_results)
        
        return results
    
    def remove(self, ids: List[str]):
        """Remove vectors by ID."""
        if not ids or self.embeddings is None:
            return
        
        ids_set = set(ids)
        keep_mask = [i for i, vid in enumerate(self.ids) if vid not in ids_set]
        
        if keep_mask:
            self.embeddings = self.embeddings[keep_mask]
            self.ids = [self.ids[i] for i in keep_mask]
        else:
            self.embeddings = None
            self.ids = []
    
    def save(self, path: str):
        """Save index to disk."""
        import json
        from pathlib import Path
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.embeddings is not None:
            np.save(save_path / "embeddings.npy", self.embeddings)
        
        with open(save_path / "ids.json", "w") as f:
            json.dump({"ids": self.ids, "counter": self._id_counter}, f)
    
    def load(self, path: str):
        """Load index from disk."""
        import json
        from pathlib import Path
        
        load_path = Path(path)
        
        emb_path = load_path / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
        
        ids_path = load_path / "ids.json"
        if ids_path.exists():
            with open(ids_path, "r") as f:
                data = json.load(f)
            self.ids = data.get("ids", [])
            self._id_counter = data.get("counter", len(self.ids))
    
    def __len__(self) -> int:
        return len(self.ids)


class FAISSVectorIndex(BaseVectorIndex):
    """FAISS-based vector index for production."""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.ids: List[str] = []
        self._id_counter = 0
        self._index = None
        self._faiss = None
        
        self._init_faiss()
    
    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            import faiss
            self._faiss = faiss
            
            dim = self.config.embedding_dim
            
            if self.config.index_type == IndexType.FAISS_FLAT:
                # Flat index (exact search)
                if self.config.metric == "l2":
                    self._index = faiss.IndexFlatL2(dim)
                else:  # cosine or IP
                    self._index = faiss.IndexFlatIP(dim)
                    
            elif self.config.index_type == IndexType.FAISS_IVF:
                # IVF index (approximate)
                quantizer = faiss.IndexFlatIP(dim) if self.config.metric != "l2" else faiss.IndexFlatL2(dim)
                self._index = faiss.IndexIVFFlat(
                    quantizer,
                    dim,
                    self.config.nlist,
                )
                self._index.nprobe = self.config.nprobe
                
            elif self.config.index_type == IndexType.FAISS_HNSW:
                # HNSW index (fast approximate)
                self._index = faiss.IndexHNSWFlat(dim, self.config.M)
                self._index.hnsw.efConstruction = self.config.ef_construction
                self._index.hnsw.efSearch = self.config.ef_search
            
            # Use multiple threads
            faiss.omp_set_num_threads(self.config.num_threads)
            
            logger.info(f"Initialized FAISS {self.config.index_type.value} index")
            
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")
            raise
    
    def add(
        self,
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add embeddings to index."""
        embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity
        if self.config.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            embeddings = embeddings / norms
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"vec_{self._id_counter + i}" for i in range(len(embeddings))]
        self._id_counter += len(embeddings)
        
        # Train IVF if needed
        if self.config.index_type == IndexType.FAISS_IVF:
            if not self._index.is_trained:
                self._index.train(embeddings)
        
        # Add to index
        self._index.add(embeddings)
        self.ids.extend(ids)
        
        return ids
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        results = self.search_batch(query.reshape(1, -1), k)
        
        if results and results[0]:
            return [(vid, score) for vid, score in results[0] if score >= threshold]
        return []
    
    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """Batch search for similar vectors."""
        if len(self.ids) == 0:
            return [[] for _ in range(len(queries))]
        
        queries = queries.astype(np.float32)
        
        # Normalize queries
        if self.config.metric == "cosine":
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            queries = queries / norms
        
        k = min(k, len(self.ids))
        scores, indices = self._index.search(queries, k)
        
        results = []
        for batch_scores, batch_indices in zip(scores, indices):
            query_results = []
            for score, idx in zip(batch_scores, batch_indices):
                if idx >= 0 and idx < len(self.ids):
                    query_results.append((self.ids[idx], float(score)))
            results.append(query_results)
        
        return results
    
    def remove(self, ids: List[str]):
        """Remove vectors by ID - requires rebuilding index."""
        if not ids:
            return
        
        ids_set = set(ids)
        keep_mask = [i for i, vid in enumerate(self.ids) if vid not in ids_set]
        
        if not keep_mask:
            self._init_faiss()
            self.ids = []
            return
        
        # Get embeddings to keep
        old_embeddings = self._index.reconstruct_n(0, len(self.ids))
        new_embeddings = old_embeddings[keep_mask]
        new_ids = [self.ids[i] for i in keep_mask]
        
        # Reinitialize and add
        self._init_faiss()
        self.ids = []
        self.add(new_embeddings, new_ids)
    
    def save(self, path: str):
        """Save index to disk."""
        import json
        from pathlib import Path
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self._faiss.write_index(self._index, str(save_path / "faiss.index"))
        
        with open(save_path / "ids.json", "w") as f:
            json.dump({"ids": self.ids, "counter": self._id_counter}, f)
    
    def load(self, path: str):
        """Load index from disk."""
        import json
        from pathlib import Path
        
        load_path = Path(path)
        
        index_path = load_path / "faiss.index"
        if index_path.exists():
            self._index = self._faiss.read_index(str(index_path))
        
        ids_path = load_path / "ids.json"
        if ids_path.exists():
            with open(ids_path, "r") as f:
                data = json.load(f)
            self.ids = data.get("ids", [])
            self._id_counter = data.get("counter", len(self.ids))
    
    def __len__(self) -> int:
        return len(self.ids)


class MilvusVectorIndex(BaseVectorIndex):
    """Milvus-based vector index for distributed production."""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.ids: List[str] = []
        self._id_counter = 0
        self._collection = None
        
        self._init_milvus()
    
    def _init_milvus(self):
        """Initialize Milvus connection and collection."""
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
            
            # Connect to Milvus
            connections.connect(
                "default",
                host=self.config.milvus_host,
                port=self.config.milvus_port,
            )
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.embedding_dim),
            ]
            schema = CollectionSchema(fields, f"Vector collection for {self.config.collection_name}")
            
            # Create or get collection
            if utility.has_collection(self.config.collection_name):
                self._collection = Collection(self.config.collection_name)
            else:
                self._collection = Collection(self.config.collection_name, schema)
                
                # Create index
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "IP" if self.config.metric != "l2" else "L2",
                    "params": {"nlist": self.config.nlist},
                }
                self._collection.create_index("embedding", index_params)
            
            # Load collection
            self._collection.load()
            
            logger.info(f"Connected to Milvus collection: {self.config.collection_name}")
            
        except ImportError:
            logger.error("pymilvus not installed. Install with: pip install pymilvus")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def add(
        self,
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add embeddings to Milvus."""
        embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity
        if self.config.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            embeddings = embeddings / norms
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"vec_{self._id_counter + i}" for i in range(len(embeddings))]
        self._id_counter += len(embeddings)
        
        # Insert into Milvus
        data = [ids, embeddings.tolist()]
        self._collection.insert(data)
        self._collection.flush()
        
        self.ids.extend(ids)
        return ids
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Search in Milvus."""
        results = self.search_batch(query.reshape(1, -1), k)
        
        if results and results[0]:
            return [(vid, score) for vid, score in results[0] if score >= threshold]
        return []
    
    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """Batch search in Milvus."""
        queries = queries.astype(np.float32)
        
        # Normalize queries
        if self.config.metric == "cosine":
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            queries = queries / norms
        
        search_params = {"metric_type": "IP", "params": {"nprobe": self.config.nprobe}}
        
        results = self._collection.search(
            queries.tolist(),
            "embedding",
            search_params,
            limit=k,
            output_fields=["id"],
        )
        
        return [
            [(hit.id, hit.score) for hit in hits]
            for hits in results
        ]
    
    def remove(self, ids: List[str]):
        """Remove vectors from Milvus."""
        if not ids:
            return
        
        expr = f'id in {ids}'
        self._collection.delete(expr)
        self._collection.flush()
        
        ids_set = set(ids)
        self.ids = [vid for vid in self.ids if vid not in ids_set]
    
    def save(self, path: str):
        """Save is handled by Milvus automatically."""
        self._collection.flush()
        logger.info("Milvus collection flushed")
    
    def load(self, path: str):
        """Load collection (Milvus handles persistence)."""
        self._collection.load()
    
    def __len__(self) -> int:
        return self._collection.num_entities


def create_vector_index(config: IndexConfig) -> BaseVectorIndex:
    """Factory function to create the appropriate vector index."""
    if config.index_type == IndexType.NUMPY:
        return NumpyVectorIndex(config)
    
    elif config.index_type in (IndexType.FAISS_FLAT, IndexType.FAISS_IVF, IndexType.FAISS_HNSW):
        try:
            return FAISSVectorIndex(config)
        except ImportError:
            logger.warning("FAISS not available, falling back to NumPy")
            return NumpyVectorIndex(config)
    
    elif config.index_type == IndexType.MILVUS:
        try:
            return MilvusVectorIndex(config)
        except (ImportError, Exception) as e:
            logger.warning(f"Milvus not available ({e}), falling back to NumPy")
            return NumpyVectorIndex(config)
    
    else:
        logger.warning(f"Unknown index type: {config.index_type}, using NumPy")
        return NumpyVectorIndex(config)
