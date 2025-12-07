"""
Semantic Memory Module

Stores structured knowledge and concepts. Supports:
- Concept storage with embeddings
- Relationship tracking
- Semantic search
- Knowledge consolidation
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """A concept in semantic memory."""
    concept_id: str
    name: str
    description: str
    embedding: Optional[np.ndarray] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "description": self.description,
            "attributes": self.attributes,
            "access_count": self.access_count,
        }


@dataclass
class Relationship:
    """A relationship between concepts."""
    source_id: str
    target_id: str
    relation_type: str
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticMemory:
    """
    Semantic memory for structured knowledge storage.
    
    Features:
    - Concept storage with embeddings
    - Relationship graph between concepts
    - Semantic similarity search
    - Concept activation spreading
    
    Usage:
        memory = SemanticMemory()
        
        # Store concept
        memory.store_concept(
            name="Python",
            description="A programming language",
            attributes={"type": "language", "paradigm": "multi-paradigm"}
        )
        
        # Add relationship
        memory.add_relationship("python", "programming", "is_a")
        
        # Search
        results = memory.search_concepts(query_embedding, k=5)
    """
    
    def __init__(
        self,
        max_concepts: int = 50000,
        embedding_dim: int = 768,
    ):
        """
        Initialize semantic memory.
        
        Args:
            max_concepts: Maximum number of concepts
            embedding_dim: Dimension of concept embeddings
        """
        self.max_concepts = max_concepts
        self.embedding_dim = embedding_dim
        
        # Concept storage
        self.concepts: Dict[str, Concept] = {}
        
        # Relationship graph (adjacency list)
        self.relationships: Dict[str, List[Relationship]] = defaultdict(list)
        self.reverse_relationships: Dict[str, List[Relationship]] = defaultdict(list)
        
        # Name to ID mapping
        self.name_to_id: Dict[str, str] = {}
        
        # Statistics
        self.total_stored = 0
        self.total_queries = 0
    
    def store_concept(
        self,
        name: str,
        description: str,
        embedding: Optional[np.ndarray] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a concept in semantic memory.
        
        Args:
            name: Concept name
            description: Concept description
            embedding: Optional embedding vector
            attributes: Additional attributes
            
        Returns:
            Concept ID
        """
        # Generate concept ID
        concept_id = self._generate_concept_id(name)
        
        # Check if exists
        if concept_id in self.concepts:
            # Update existing
            concept = self.concepts[concept_id]
            concept.description = description
            if embedding is not None:
                concept.embedding = embedding
            if attributes:
                concept.attributes.update(attributes)
            return concept_id
        
        # Check capacity
        if len(self.concepts) >= self.max_concepts:
            self._evict_least_accessed()
        
        # Create concept
        concept = Concept(
            concept_id=concept_id,
            name=name,
            description=description,
            embedding=embedding,
            attributes=attributes or {},
        )
        
        self.concepts[concept_id] = concept
        self.name_to_id[name.lower()] = concept_id
        self.total_stored += 1
        
        return concept_id
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID."""
        concept = self.concepts.get(concept_id)
        if concept:
            concept.access_count += 1
            concept.last_accessed = datetime.now(timezone.utc)
        return concept
    
    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Get a concept by name."""
        concept_id = self.name_to_id.get(name.lower())
        if concept_id:
            return self.get_concept(concept_id)
        return None
    
    def add_relationship(
        self,
        source_name: str,
        target_name: str,
        relation_type: str,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a relationship between concepts.
        
        Args:
            source_name: Source concept name
            target_name: Target concept name
            relation_type: Type of relationship (e.g., "is_a", "has_a", "related_to")
            strength: Relationship strength (0-1)
            metadata: Additional metadata
            
        Returns:
            True if relationship was added
        """
        source_id = self.name_to_id.get(source_name.lower())
        target_id = self.name_to_id.get(target_name.lower())
        
        if not source_id or not target_id:
            return False
        
        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            metadata=metadata or {},
        )
        
        self.relationships[source_id].append(relationship)
        self.reverse_relationships[target_id].append(relationship)
        
        return True
    
    def get_related_concepts(
        self,
        concept_id: str,
        relation_type: Optional[str] = None,
        max_depth: int = 1,
    ) -> List[Tuple[Concept, str, float]]:
        """
        Get concepts related to the given concept.
        
        Args:
            concept_id: Source concept ID
            relation_type: Optional filter by relation type
            max_depth: Maximum depth to traverse
            
        Returns:
            List of (concept, relation_type, strength) tuples
        """
        results = []
        visited: Set[str] = {concept_id}
        queue = [(concept_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            for rel in self.relationships.get(current_id, []):
                if relation_type and rel.relation_type != relation_type:
                    continue
                
                if rel.target_id not in visited:
                    visited.add(rel.target_id)
                    concept = self.concepts.get(rel.target_id)
                    if concept:
                        results.append((concept, rel.relation_type, rel.strength))
                        queue.append((rel.target_id, depth + 1))
        
        return results
    
    def search_concepts(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Tuple[Concept, float]]:
        """
        Search concepts by embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (concept, similarity) tuples
        """
        self.total_queries += 1
        
        similarities = []
        
        for concept in self.concepts.values():
            if concept.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, concept.embedding)
                if similarity >= min_similarity:
                    similarities.append((concept, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update access counts
        for concept, _ in similarities[:k]:
            concept.access_count += 1
            concept.last_accessed = datetime.now(timezone.utc)
        
        return similarities[:k]
    
    def spreading_activation(
        self,
        start_concepts: List[str],
        decay: float = 0.5,
        max_iterations: int = 3,
    ) -> Dict[str, float]:
        """
        Perform spreading activation from start concepts.
        
        Args:
            start_concepts: List of starting concept IDs
            decay: Activation decay per hop
            max_iterations: Maximum iterations
            
        Returns:
            Dict of concept_id -> activation level
        """
        activation: Dict[str, float] = {}
        
        # Initialize start concepts
        for concept_id in start_concepts:
            if concept_id in self.concepts:
                activation[concept_id] = 1.0
        
        # Spread activation
        for _ in range(max_iterations):
            new_activation: Dict[str, float] = {}
            
            for concept_id, level in activation.items():
                for rel in self.relationships.get(concept_id, []):
                    spread = level * decay * rel.strength
                    target_id = rel.target_id
                    new_activation[target_id] = max(
                        new_activation.get(target_id, 0),
                        spread,
                    )
            
            # Update activation
            for concept_id, level in new_activation.items():
                activation[concept_id] = max(activation.get(concept_id, 0), level)
        
        return activation
    
    def _evict_least_accessed(self):
        """Evict the least accessed concept."""
        if not self.concepts:
            return
        
        # Find least accessed
        min_concept = min(
            self.concepts.values(),
            key=lambda c: (c.access_count, c.created_at),
        )
        
        self._remove_concept(min_concept.concept_id)
    
    def _remove_concept(self, concept_id: str):
        """Remove a concept and its relationships."""
        concept = self.concepts.pop(concept_id, None)
        if concept:
            self.name_to_id.pop(concept.name.lower(), None)
        
        # Remove relationships
        self.relationships.pop(concept_id, None)
        self.reverse_relationships.pop(concept_id, None)
    
    def _generate_concept_id(self, name: str) -> str:
        """Generate concept ID from name."""
        return name.lower().replace(" ", "_")[:50]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_concepts": len(self.concepts),
            "max_concepts": self.max_concepts,
            "total_relationships": sum(len(rels) for rels in self.relationships.values()),
            "total_stored": self.total_stored,
            "total_queries": self.total_queries,
        }
    
    def clear(self):
        """Clear all memory."""
        self.concepts.clear()
        self.relationships.clear()
        self.reverse_relationships.clear()
        self.name_to_id.clear()
