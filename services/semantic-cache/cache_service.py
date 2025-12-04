"""
Semantic Cache Service

Provides semantic similarity-based caching for code analysis results.
Useful in offline mode to reduce redundant model calls by finding
similar previously-analyzed code.

Features:
- Embedding-based similarity search
- Configurable similarity threshold
- TTL-based expiration
- Cache warming from common patterns
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached analysis result"""
    code_hash: str
    embedding: List[float]
    language: str
    analysis_result: Dict[str, Any]
    model_version: str
    prompt_version: str
    created_at: str
    ttl_hours: int


class CacheRequest(BaseModel):
    """Request to check cache"""
    code: str
    language: str
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None


class CacheResponse(BaseModel):
    """Cache lookup response"""
    hit: bool
    result: Optional[Dict[str, Any]] = None
    similarity: Optional[float] = None
    cache_key: Optional[str] = None


class StoreRequest(BaseModel):
    """Request to store in cache"""
    code: str
    language: str
    analysis_result: Dict[str, Any]
    model_version: str
    prompt_version: str
    ttl_hours: int = 168  # 1 week default


class SemanticCacheService:
    """
    Semantic similarity-based cache for code analysis.
    
    Uses embeddings to find similar code that has already been analyzed,
    reducing redundant model calls especially in offline deployments.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        embedding_url: str = "http://localhost:8102",
        similarity_threshold: float = 0.92,
        max_cache_size: int = 100000
    ):
        self.redis_url = redis_url
        self.embedding_url = embedding_url
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        
        self.redis_client: Optional[redis.Redis] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # In-memory index for fast similarity search
        self.embedding_index: Dict[str, np.ndarray] = {}
        self.cache_keys: List[str] = []
    
    async def initialize(self):
        """Initialize connections"""
        self.redis_client = redis.from_url(self.redis_url)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Load existing embeddings into memory
        await self._load_embedding_index()
        
        logger.info(f"Semantic cache initialized with {len(self.cache_keys)} entries")
    
    async def close(self):
        """Close connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.http_client:
            await self.http_client.aclose()
    
    async def _load_embedding_index(self):
        """Load embeddings from Redis into memory index"""
        try:
            keys = await self.redis_client.keys("semantic:*")
            for key in keys[:self.max_cache_size]:
                key_str = key.decode() if isinstance(key, bytes) else key
                data = await self.redis_client.hgetall(key_str)
                if data and b'embedding' in data:
                    embedding_bytes = data[b'embedding']
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    self.embedding_index[key_str] = embedding
                    self.cache_keys.append(key_str)
        except Exception as e:
            logger.error(f"Failed to load embedding index: {e}")
    
    async def get_embedding(self, code: str) -> np.ndarray:
        """Get embedding for code from embedding server"""
        try:
            response = await self.http_client.post(
                f"{self.embedding_url}/embed",
                json={"inputs": code}
            )
            if response.status_code == 200:
                data = response.json()
                # Handle different response formats
                if isinstance(data, list):
                    embedding = data[0] if isinstance(data[0], list) else data
                else:
                    embedding = data.get('embeddings', [[]])[0]
                return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
        
        # Fallback: use hash-based pseudo-embedding
        return self._hash_to_embedding(code)
    
    def _hash_to_embedding(self, code: str, dim: int = 384) -> np.ndarray:
        """Generate pseudo-embedding from hash (fallback)"""
        code_hash = hashlib.sha256(code.encode()).digest()
        # Expand hash to embedding dimension
        np.random.seed(int.from_bytes(code_hash[:4], 'big'))
        return np.random.randn(dim).astype(np.float32)
    
    def _compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between embeddings"""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    async def lookup(
        self,
        code: str,
        language: str,
        model_version: Optional[str] = None,
        prompt_version: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict], float, Optional[str]]:
        """
        Look up similar code in cache.
        
        Returns:
            (hit, result, similarity, cache_key)
        """
        # Get embedding for query code
        query_embedding = await self.get_embedding(code)
        
        # Find most similar cached entry
        best_similarity = 0.0
        best_key = None
        
        for key, cached_embedding in self.embedding_index.items():
            similarity = self._compute_similarity(query_embedding, cached_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key
        
        # Check if similarity meets threshold
        if best_similarity >= self.similarity_threshold and best_key:
            # Get cached result from Redis
            data = await self.redis_client.hgetall(best_key)
            if data:
                # Check language match
                cached_language = data.get(b'language', b'').decode()
                if cached_language != language:
                    return False, None, best_similarity, None
                
                # Optional: check model/prompt version
                if model_version:
                    cached_model = data.get(b'model_version', b'').decode()
                    if cached_model != model_version:
                        return False, None, best_similarity, None
                
                if prompt_version:
                    cached_prompt = data.get(b'prompt_version', b'').decode()
                    if cached_prompt != prompt_version:
                        return False, None, best_similarity, None
                
                # Return cached result
                result_json = data.get(b'analysis_result', b'{}').decode()
                result = json.loads(result_json)
                
                # Update access time
                await self.redis_client.hset(best_key, 'last_accessed', datetime.utcnow().isoformat())
                
                logger.info(f"Cache hit: similarity={best_similarity:.4f}, key={best_key}")
                return True, result, best_similarity, best_key
        
        logger.debug(f"Cache miss: best_similarity={best_similarity:.4f}")
        return False, None, best_similarity, None
    
    async def store(
        self,
        code: str,
        language: str,
        analysis_result: Dict[str, Any],
        model_version: str,
        prompt_version: str,
        ttl_hours: int = 168
    ) -> str:
        """
        Store analysis result in cache.
        
        Returns:
            Cache key
        """
        # Compute code hash and embedding
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        embedding = await self.get_embedding(code)
        
        # Create cache key
        cache_key = f"semantic:{code_hash[:16]}"
        
        # Store in Redis
        cache_data = {
            'code_hash': code_hash,
            'embedding': embedding.tobytes(),
            'language': language,
            'analysis_result': json.dumps(analysis_result),
            'model_version': model_version,
            'prompt_version': prompt_version,
            'created_at': datetime.utcnow().isoformat(),
            'ttl_hours': ttl_hours
        }
        
        await self.redis_client.hset(cache_key, mapping=cache_data)
        await self.redis_client.expire(cache_key, ttl_hours * 3600)
        
        # Update in-memory index
        self.embedding_index[cache_key] = embedding
        if cache_key not in self.cache_keys:
            self.cache_keys.append(cache_key)
        
        # Evict oldest entries if over limit
        if len(self.cache_keys) > self.max_cache_size:
            await self._evict_oldest(len(self.cache_keys) - self.max_cache_size)
        
        logger.info(f"Cached result: key={cache_key}")
        return cache_key
    
    async def _evict_oldest(self, count: int):
        """Evict oldest cache entries"""
        to_evict = self.cache_keys[:count]
        for key in to_evict:
            await self.redis_client.delete(key)
            if key in self.embedding_index:
                del self.embedding_index[key]
            self.cache_keys.remove(key)
        logger.info(f"Evicted {count} old cache entries")
    
    async def warm_cache(self, patterns_dir: str):
        """
        Warm cache with common code patterns.
        
        Pre-analyzes common patterns to improve cache hit rate.
        """
        import os
        
        if not os.path.exists(patterns_dir):
            logger.warning(f"Patterns directory not found: {patterns_dir}")
            return
        
        count = 0
        for filename in os.listdir(patterns_dir):
            filepath = os.path.join(patterns_dir, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r') as f:
                        code = f.read()
                    
                    # Get embedding and store placeholder
                    embedding = await self.get_embedding(code)
                    code_hash = hashlib.sha256(code.encode()).hexdigest()
                    cache_key = f"semantic:{code_hash[:16]}"
                    
                    # Store embedding only (result will be filled on first hit)
                    await self.redis_client.hset(cache_key, mapping={
                        'code_hash': code_hash,
                        'embedding': embedding.tobytes(),
                        'language': self._detect_language(filename),
                        'warmed_at': datetime.utcnow().isoformat()
                    })
                    
                    self.embedding_index[cache_key] = embedding
                    self.cache_keys.append(cache_key)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to warm cache for {filename}: {e}")
        
        logger.info(f"Warmed cache with {count} patterns")
    
    def _detect_language(self, filename: str) -> str:
        """Detect language from filename extension"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.rb': 'ruby',
            '.php': 'php',
        }
        for ext, lang in ext_map.items():
            if filename.endswith(ext):
                return lang
        return 'unknown'
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache_keys),
            'index_size_mb': sum(e.nbytes for e in self.embedding_index.values()) / 1024 / 1024,
            'similarity_threshold': self.similarity_threshold,
            'max_cache_size': self.max_cache_size
        }
    
    async def clear(self):
        """Clear all cache entries"""
        for key in self.cache_keys:
            await self.redis_client.delete(key)
        self.embedding_index.clear()
        self.cache_keys.clear()
        logger.info("Cache cleared")


# FastAPI Application
app = FastAPI(
    title="Semantic Cache Service",
    description="Embedding-based semantic cache for code analysis",
    version="1.0.0"
)

cache_service: Optional[SemanticCacheService] = None


@app.on_event("startup")
async def startup():
    global cache_service
    cache_service = SemanticCacheService()
    await cache_service.initialize()


@app.on_event("shutdown")
async def shutdown():
    if cache_service:
        await cache_service.close()


@app.get("/health")
async def health():
    return {"status": "healthy", "entries": len(cache_service.cache_keys)}


@app.post("/lookup", response_model=CacheResponse)
async def lookup(request: CacheRequest):
    """Look up similar code in cache"""
    hit, result, similarity, cache_key = await cache_service.lookup(
        request.code,
        request.language,
        request.model_version,
        request.prompt_version
    )
    return CacheResponse(
        hit=hit,
        result=result,
        similarity=similarity,
        cache_key=cache_key
    )


@app.post("/store")
async def store(request: StoreRequest):
    """Store analysis result in cache"""
    cache_key = await cache_service.store(
        request.code,
        request.language,
        request.analysis_result,
        request.model_version,
        request.prompt_version,
        request.ttl_hours
    )
    return {"status": "stored", "cache_key": cache_key}


@app.get("/stats")
async def stats():
    """Get cache statistics"""
    return await cache_service.get_stats()


@app.post("/warm")
async def warm_cache(patterns_dir: str = "/data/patterns"):
    """Warm cache with common patterns"""
    await cache_service.warm_cache(patterns_dir)
    return {"status": "warmed", "entries": len(cache_service.cache_keys)}


@app.delete("/clear")
async def clear_cache():
    """Clear all cache entries"""
    await cache_service.clear()
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8103)
