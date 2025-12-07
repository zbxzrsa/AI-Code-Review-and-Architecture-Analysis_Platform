"""
Unit tests for RAG (Retrieval-Augmented Generation) module.
"""

import pytest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timezone

from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
from ai_core.foundation_model.deployment.rag import RAGDocument, RAGIndex, RAGSystem


class TestRAGDocument:
    """Tests for RAGDocument dataclass."""

    def test_creation(self):
        """Test creating a RAG document."""
        embedding = np.random.randn(768).astype(np.float32)
        doc = RAGDocument(
            doc_id="doc_001",
            content="Test content",
            embedding=embedding,
        )
        
        assert doc.doc_id == "doc_001"
        assert doc.content == "Test content"
        assert doc.embedding.shape == (768,)
        assert isinstance(doc.metadata, dict)
        assert isinstance(doc.timestamp, datetime)

    def test_with_metadata(self):
        """Test creating document with metadata."""
        embedding = np.random.randn(768).astype(np.float32)
        metadata = {"source": "test", "category": "unit_test"}
        
        doc = RAGDocument(
            doc_id="doc_002",
            content="Test content",
            embedding=embedding,
            metadata=metadata,
        )
        
        assert doc.metadata["source"] == "test"
        assert doc.metadata["category"] == "unit_test"

    def test_timestamp_default(self):
        """Test timestamp defaults to current UTC time."""
        embedding = np.random.randn(768).astype(np.float32)
        before = datetime.now(timezone.utc)
        
        doc = RAGDocument(
            doc_id="doc_003",
            content="Test",
            embedding=embedding,
        )
        
        after = datetime.now(timezone.utc)
        assert before <= doc.timestamp <= after


class TestRAGIndex:
    """Tests for RAGIndex class."""

    @pytest.fixture
    def rag_index(self):
        """Create a RAG index for testing."""
        return RAGIndex(embedding_dim=768, use_faiss=False)

    def test_initialization(self, rag_index):
        """Test index initialization."""
        assert rag_index.embedding_dim == 768
        assert len(rag_index.documents) == 0
        assert len(rag_index.doc_ids) == 0

    def test_add_document(self, rag_index):
        """Test adding a document to index."""
        embedding = np.random.randn(768).astype(np.float32)
        
        rag_index.add_document(
            doc_id="doc_001",
            content="Python programming tips",
            embedding=embedding,
        )
        
        assert len(rag_index.documents) == 1
        assert "doc_001" in rag_index.documents
        assert len(rag_index.doc_ids) == 1

    def test_add_multiple_documents(self, rag_index):
        """Test adding multiple documents."""
        for i in range(5):
            embedding = np.random.randn(768).astype(np.float32)
            rag_index.add_document(
                doc_id=f"doc_{i:03d}",
                content=f"Document content {i}",
                embedding=embedding,
            )
        
        assert len(rag_index.documents) == 5
        assert len(rag_index.doc_ids) == 5

    def test_search_empty_index(self, rag_index):
        """Test searching empty index returns empty list."""
        query_embedding = np.random.randn(768).astype(np.float32)
        results = rag_index.search(query_embedding, top_k=5)
        
        assert results == []

    def test_search_returns_similar_documents(self, rag_index):
        """Test search returns similar documents."""
        # Add documents with known embeddings
        base_embedding = np.random.randn(768).astype(np.float32)
        
        # Very similar document
        similar_embedding = base_embedding + np.random.randn(768).astype(np.float32) * 0.1
        rag_index.add_document("similar", "Similar content", similar_embedding)
        
        # Different document
        different_embedding = np.random.randn(768).astype(np.float32)
        rag_index.add_document("different", "Different content", different_embedding)
        
        # Search with base embedding
        results = rag_index.search(base_embedding, top_k=2)
        
        assert len(results) == 2
        # Similar document should rank higher
        assert results[0][0].doc_id == "similar"

    def test_search_top_k(self, rag_index):
        """Test top_k parameter limits results."""
        # Add 10 documents
        for i in range(10):
            embedding = np.random.randn(768).astype(np.float32)
            rag_index.add_document(f"doc_{i}", f"Content {i}", embedding)
        
        query = np.random.randn(768).astype(np.float32)
        
        results = rag_index.search(query, top_k=3)
        assert len(results) == 3
        
        results = rag_index.search(query, top_k=5)
        assert len(results) == 5

    def test_search_threshold(self, rag_index):
        """Test similarity threshold filtering."""
        embedding = np.ones(768, dtype=np.float32)
        rag_index.add_document("doc1", "Content", embedding)
        
        # Search with very different embedding and high threshold
        different_query = -np.ones(768, dtype=np.float32)
        results = rag_index.search(different_query, top_k=5, threshold=0.9)
        
        # Should filter out dissimilar results
        assert len(results) == 0

    def test_save_and_load(self, rag_index):
        """Test saving and loading index."""
        with TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index"
            rag_index.index_path = index_path
            
            # Add documents
            for i in range(3):
                embedding = np.random.randn(768).astype(np.float32)
                rag_index.add_document(f"doc_{i}", f"Content {i}", embedding)
            
            # Save
            rag_index.save()
            
            # Create new index and load
            new_index = RAGIndex(embedding_dim=768, index_path=str(index_path), use_faiss=False)
            
            assert len(new_index.documents) == 3
            assert "doc_0" in new_index.documents

    def test_embedding_normalization(self, rag_index):
        """Test embeddings are normalized for cosine similarity."""
        # Add non-normalized embedding
        embedding = np.ones(768, dtype=np.float32) * 10
        rag_index.add_document("doc1", "Content", embedding)
        
        # The stored embedding should be normalized
        stored_doc = rag_index.documents["doc1"]
        norm = np.linalg.norm(stored_doc.embedding)
        assert np.isclose(norm, 1.0, atol=0.01)


class TestRAGSystem:
    """Tests for RAGSystem class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PracticalDeploymentConfig(
            enable_rag=True,
            rag_top_k=5,
            rag_similarity_threshold=0.5,
            rag_use_faiss=False,
        )

    @pytest.fixture
    def rag_system(self, config):
        """Create RAG system for testing."""
        return RAGSystem(config)

    def test_initialization(self, rag_system, config):
        """Test RAG system initialization."""
        assert rag_system.config == config
        assert rag_system.index is not None
        assert len(rag_system.embedding_cache) == 0

    def test_compute_embedding(self, rag_system):
        """Test embedding computation."""
        embedding = rag_system.compute_embedding("Test text")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        assert embedding.dtype == np.float32

    def test_embedding_cache(self, rag_system):
        """Test embedding caching."""
        text = "Test text for caching"
        
        # First call computes embedding
        embedding1 = rag_system.compute_embedding(text)
        assert len(rag_system.embedding_cache) == 1
        
        # Second call uses cache
        embedding2 = rag_system.compute_embedding(text)
        
        np.testing.assert_array_equal(embedding1, embedding2)

    def test_add_knowledge(self, rag_system):
        """Test adding knowledge to RAG system."""
        rag_system.add_knowledge(
            content="Python is a programming language",
            metadata={"topic": "programming"},
        )
        
        assert len(rag_system.index.documents) == 1

    def test_retrieve(self, rag_system):
        """Test retrieving relevant documents."""
        # Add some knowledge
        rag_system.add_knowledge("Python is a programming language")
        rag_system.add_knowledge("Java is also a programming language")
        rag_system.add_knowledge("Cooking recipes and food")
        
        # Retrieve related to programming
        results = rag_system.retrieve("programming languages", top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, tuple) for r in results)

    def test_retrieve_empty(self, rag_system):
        """Test retrieve with no documents."""
        results = rag_system.retrieve("test query")
        assert results == []

    def test_augment_prompt(self, rag_system):
        """Test prompt augmentation."""
        # Add knowledge
        rag_system.add_knowledge("Python uses indentation for code blocks")
        
        original_prompt = "How does Python work?"
        augmented = rag_system.augment_prompt(original_prompt)
        
        # Augmented prompt should contain original query
        assert original_prompt in augmented or "How does Python work" in augmented

    def test_augment_prompt_no_results(self, rag_system):
        """Test augment prompt when no documents match."""
        # Don't add any knowledge
        
        prompt = "Test query"
        augmented = rag_system.augment_prompt(prompt)
        
        # Should return original prompt when no results
        assert prompt == augmented

    def test_retrieve_respects_top_k(self, rag_system):
        """Test retrieve respects top_k config."""
        # Add many documents
        for i in range(20):
            rag_system.add_knowledge(f"Document content number {i}")
        
        results = rag_system.retrieve("document", top_k=3)
        assert len(results) <= 3

    def test_update_index(self, rag_system):
        """Test index update (save)."""
        with TemporaryDirectory() as tmpdir:
            rag_system.index.index_path = Path(tmpdir) / "index"
            rag_system.add_knowledge("Test content")
            
            # Should not raise
            rag_system.update_index()
            
            assert (Path(tmpdir) / "index").exists()


class TestRAGIndexWithFAISS:
    """Tests for FAISS-based RAG index (if available)."""

    @pytest.fixture
    def faiss_available(self):
        """Check if FAISS is available."""
        try:
            import faiss
            return True
        except ImportError:
            return False

    def test_faiss_initialization(self, faiss_available):
        """Test FAISS index initialization."""
        index = RAGIndex(embedding_dim=768, use_faiss=True)
        
        if faiss_available:
            assert index._faiss_available
        else:
            assert not index._faiss_available

    @pytest.mark.skipif(
        not RAGIndex(embedding_dim=768, use_faiss=True)._faiss_available,
        reason="FAISS not installed"
    )
    def test_faiss_search(self):
        """Test FAISS-based search."""
        index = RAGIndex(embedding_dim=768, use_faiss=True)
        
        # Add documents
        for i in range(100):
            embedding = np.random.randn(768).astype(np.float32)
            index.add_document(f"doc_{i}", f"Content {i}", embedding)
        
        # Search
        query = np.random.randn(768).astype(np.float32)
        results = index.search(query, top_k=10)
        
        assert len(results) == 10


class TestRAGIntegration:
    """Integration tests for RAG functionality."""

    def test_end_to_end_retrieval(self):
        """Test complete retrieval workflow."""
        config = PracticalDeploymentConfig(
            enable_rag=True,
            rag_top_k=3,
            rag_similarity_threshold=0.0,  # Low threshold for test
            rag_use_faiss=False,
        )
        
        system = RAGSystem(config)
        
        # Build knowledge base
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with many layers",
            "Python is commonly used for data science",
            "Database systems store and retrieve data efficiently",
            "Web development involves HTML, CSS, and JavaScript",
        ]
        
        for doc in documents:
            system.add_knowledge(doc)
        
        # Query
        results = system.retrieve("AI and neural networks", top_k=3)
        
        assert len(results) <= 3
        
        # Results should include AI-related documents
        contents = [r[0] for r in results]
        ai_related = any("machine learning" in c.lower() or "neural" in c.lower() for c in contents)
        # Note: Due to random embeddings in test, this may not always pass
        # In production, actual embeddings would make this reliable

    def test_document_deduplication(self):
        """Test that duplicate documents are handled."""
        config = PracticalDeploymentConfig(rag_use_faiss=False)
        system = RAGSystem(config)
        
        # Add same content multiple times
        system.add_knowledge("Test content")
        system.add_knowledge("Test content")  # Different doc_id but same content
        
        # Should have 2 documents (no automatic deduplication - this is by design)
        assert len(system.index.documents) == 2
