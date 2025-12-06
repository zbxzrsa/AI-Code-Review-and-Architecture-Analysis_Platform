"""
Unit tests for Distributed Verification System.

Tests cover:
- Node management and health checking
- Consensus algorithm
- Verification workflow
- Performance requirements (<500ms)
"""
import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.shared.security.distributed_verification import (
    DistributedVerificationService,
    VerificationNode,
    NodeVerificationResult,
    ConsensusResult,
    NodeHealth,
    VerificationStatus,
    VerificationNodeServer
)


class TestVerificationNode:
    """Tests for VerificationNode dataclass."""
    
    def test_node_creation(self):
        """Test creating a verification node."""
        node = VerificationNode(
            node_id="node1",
            url="http://localhost:8001",
            region="us-east-1"
        )
        
        assert node.node_id == "node1"
        assert node.health == NodeHealth.HEALTHY
        assert node.success_rate == 1.0
    
    def test_node_to_dict(self):
        """Test node serialization."""
        node = VerificationNode(
            node_id="node1",
            url="http://localhost:8001",
            region="us-east-1",
            priority=5,
            health=NodeHealth.DEGRADED
        )
        
        d = node.to_dict()
        assert d["node_id"] == "node1"
        assert d["health"] == "degraded"
        assert d["priority"] == 5


class TestNodeVerificationResult:
    """Tests for NodeVerificationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a verification result."""
        result = NodeVerificationResult(
            node_id="node1",
            is_valid=True,
            hash_verified="abc123",
            signature_verified=True,
            chain_verified=True,
            response_time_ms=50.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert result.is_valid
        assert result.error is None
    
    def test_result_with_error(self):
        """Test verification result with error."""
        result = NodeVerificationResult(
            node_id="node1",
            is_valid=False,
            hash_verified="",
            signature_verified=False,
            chain_verified=False,
            response_time_ms=100.0,
            timestamp=datetime.now(timezone.utc),
            error="Connection timeout"
        )
        
        assert not result.is_valid
        assert result.error == "Connection timeout"
    
    def test_result_to_dict(self):
        """Test result serialization."""
        now = datetime.now(timezone.utc)
        result = NodeVerificationResult(
            node_id="node1",
            is_valid=True,
            hash_verified="abc123",
            signature_verified=True,
            chain_verified=True,
            response_time_ms=50.0,
            timestamp=now
        )
        
        d = result.to_dict()
        assert d["node_id"] == "node1"
        assert d["is_valid"] is True


class TestConsensusResult:
    """Tests for ConsensusResult dataclass."""
    
    def test_consensus_creation(self):
        """Test creating a consensus result."""
        result = ConsensusResult(
            request_id="req123",
            log_id="log456",
            is_valid=True,
            consensus_reached=True,
            consensus_ratio=0.8,
            total_nodes=5,
            responding_nodes=4,
            agreeing_nodes=4,
            status=VerificationStatus.VERIFIED,
            node_results=[],
            total_time_ms=250.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert result.consensus_reached
        assert result.consensus_ratio == 0.8
    
    def test_consensus_to_dict(self):
        """Test consensus serialization."""
        result = ConsensusResult(
            request_id="req123",
            log_id="log456",
            is_valid=True,
            consensus_reached=True,
            consensus_ratio=0.8,
            total_nodes=5,
            responding_nodes=4,
            agreeing_nodes=4,
            status=VerificationStatus.VERIFIED,
            node_results=[],
            total_time_ms=250.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        d = result.to_dict()
        assert d["request_id"] == "req123"
        assert d["status"] == "verified"


class TestDistributedVerificationService:
    """Tests for DistributedVerificationService class."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database client."""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.fetch = AsyncMock(return_value=[])
        db.fetchone = AsyncMock(return_value=None)
        return db
    
    @pytest.fixture
    def node_configs(self):
        """Create node configurations."""
        return [
            {"node_id": "node1", "url": "http://localhost:8001", "region": "us-east-1"},
            {"node_id": "node2", "url": "http://localhost:8002", "region": "us-west-1"},
            {"node_id": "node3", "url": "http://localhost:8003", "region": "eu-west-1"},
        ]
    
    def test_service_initialization(self, mock_db, node_configs):
        """Test service initialization."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs,
            consensus_threshold=0.67,
            timeout_ms=500
        )
        
        assert len(service.nodes) == 3
        assert service.consensus_threshold == 0.67
        assert service.timeout_ms == 500
    
    @pytest.mark.asyncio
    async def test_service_start_stop(self, mock_db, node_configs):
        """Test service startup and shutdown."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs
        )
        
        await service.start()
        assert service._running is True
        assert service._session is not None
        
        await service.stop()
        assert service._running is False
    
    def test_get_available_nodes(self, mock_db, node_configs):
        """Test getting available nodes."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs
        )
        
        # All nodes should be healthy initially
        available = service._get_available_nodes()
        assert len(available) == 3
        
        # Mark one as unhealthy
        service.nodes["node1"].health = NodeHealth.UNHEALTHY
        available = service._get_available_nodes()
        assert len(available) == 2
    
    def test_node_sorting_by_priority(self, mock_db, node_configs):
        """Test that nodes are sorted by priority."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs
        )
        
        service.nodes["node1"].priority = 1
        service.nodes["node2"].priority = 3
        service.nodes["node3"].priority = 2
        
        available = service._get_available_nodes()
        assert available[0].node_id == "node2"  # Highest priority
        assert available[1].node_id == "node3"
        assert available[2].node_id == "node1"
    
    def test_handle_node_failure(self, mock_db, node_configs):
        """Test handling node failures."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs
        )
        
        node = service.nodes["node1"]
        
        # First failure
        service._handle_node_failure(node, "test error")
        assert node.consecutive_failures == 1
        assert node.health == NodeHealth.DEGRADED
        
        # More failures
        service._handle_node_failure(node, "test error")
        service._handle_node_failure(node, "test error")
        assert node.consecutive_failures == 3
        assert node.health == NodeHealth.UNHEALTHY
    
    def test_calculate_consensus_no_responses(self, mock_db, node_configs):
        """Test consensus calculation with no responses."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs
        )
        
        result = service._calculate_consensus(
            request_id="req1",
            log_id="log1",
            queried_nodes=[service.nodes["node1"]],
            results=[],
            start_time=time.perf_counter()
        )
        
        assert result.status == VerificationStatus.TIMEOUT
        assert not result.consensus_reached
    
    def test_calculate_consensus_majority_valid(self, mock_db, node_configs):
        """Test consensus with majority valid responses."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs,
            consensus_threshold=0.67
        )
        
        now = datetime.now(timezone.utc)
        results = [
            NodeVerificationResult(
                node_id="node1",
                is_valid=True,
                hash_verified="hash",
                signature_verified=True,
                chain_verified=True,
                response_time_ms=50,
                timestamp=now
            ),
            NodeVerificationResult(
                node_id="node2",
                is_valid=True,
                hash_verified="hash",
                signature_verified=True,
                chain_verified=True,
                response_time_ms=60,
                timestamp=now
            ),
            NodeVerificationResult(
                node_id="node3",
                is_valid=False,
                hash_verified="hash",
                signature_verified=False,
                chain_verified=True,
                response_time_ms=70,
                timestamp=now
            )
        ]
        
        consensus = service._calculate_consensus(
            request_id="req1",
            log_id="log1",
            queried_nodes=list(service.nodes.values()),
            results=results,
            start_time=time.perf_counter()
        )
        
        assert consensus.is_valid
        assert consensus.consensus_reached
        assert consensus.agreeing_nodes == 2
    
    def test_calculate_consensus_majority_invalid(self, mock_db, node_configs):
        """Test consensus with majority invalid responses."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs,
            consensus_threshold=0.67
        )
        
        now = datetime.now(timezone.utc)
        results = [
            NodeVerificationResult(
                node_id="node1",
                is_valid=False,
                hash_verified="hash",
                signature_verified=False,
                chain_verified=True,
                response_time_ms=50,
                timestamp=now
            ),
            NodeVerificationResult(
                node_id="node2",
                is_valid=False,
                hash_verified="hash",
                signature_verified=False,
                chain_verified=True,
                response_time_ms=60,
                timestamp=now
            ),
            NodeVerificationResult(
                node_id="node3",
                is_valid=True,
                hash_verified="hash",
                signature_verified=True,
                chain_verified=True,
                response_time_ms=70,
                timestamp=now
            )
        ]
        
        consensus = service._calculate_consensus(
            request_id="req1",
            log_id="log1",
            queried_nodes=list(service.nodes.values()),
            results=results,
            start_time=time.perf_counter()
        )
        
        assert not consensus.is_valid
        assert consensus.consensus_reached
    
    def test_calculate_consensus_no_consensus(self, mock_db, node_configs):
        """Test consensus calculation when threshold not met."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs,
            consensus_threshold=0.8  # High threshold
        )
        
        now = datetime.now(timezone.utc)
        results = [
            NodeVerificationResult(
                node_id="node1",
                is_valid=True,
                hash_verified="hash",
                signature_verified=True,
                chain_verified=True,
                response_time_ms=50,
                timestamp=now
            ),
            NodeVerificationResult(
                node_id="node2",
                is_valid=False,
                hash_verified="hash",
                signature_verified=False,
                chain_verified=True,
                response_time_ms=60,
                timestamp=now
            ),
            NodeVerificationResult(
                node_id="node3",
                is_valid=True,
                hash_verified="hash",
                signature_verified=True,
                chain_verified=True,
                response_time_ms=70,
                timestamp=now
            )
        ]
        
        consensus = service._calculate_consensus(
            request_id="req1",
            log_id="log1",
            queried_nodes=list(service.nodes.values()),
            results=results,
            start_time=time.perf_counter()
        )
        
        # 2/3 = 0.67 < 0.8 threshold
        assert not consensus.consensus_reached
        assert consensus.status == VerificationStatus.INCONSISTENT
    
    @pytest.mark.asyncio
    async def test_verify_log_insufficient_nodes(self, mock_db, node_configs):
        """Test verification with insufficient nodes."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs,
            min_nodes=5  # More than available
        )
        
        await service.start()
        
        result = await service.verify_log(
            log_id="log1",
            signature="sig123",
            expected_hash="hash456"
        )
        
        assert not result.consensus_reached
        assert result.status == VerificationStatus.FAILED
        
        await service.stop()
    
    def test_get_node_status(self, mock_db, node_configs):
        """Test getting node status."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs
        )
        
        status = service.get_node_status()
        
        assert status["total_nodes"] == 3
        assert status["healthy_nodes"] == 3
        assert "nodes" in status
    
    def test_get_metrics(self, mock_db, node_configs):
        """Test getting verification metrics."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs
        )
        
        metrics = service.get_metrics()
        
        assert metrics["total_verifications"] == 0
        assert metrics["consensus_threshold"] == 0.67
        assert metrics["timeout_ms"] == 500
    
    @pytest.mark.asyncio
    async def test_add_node(self, mock_db, node_configs):
        """Test adding a new node."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs
        )
        
        await service.start()
        
        new_node = {
            "node_id": "node4",
            "url": "http://localhost:8004",
            "region": "ap-northeast-1"
        }
        
        await service.add_node(new_node)
        assert "node4" in service.nodes
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_remove_node(self, mock_db, node_configs):
        """Test removing a node."""
        service = DistributedVerificationService(
            db_client=mock_db,
            nodes=node_configs
        )
        
        await service.remove_node("node1")
        assert "node1" not in service.nodes
        assert len(service.nodes) == 2


class TestVerificationNodeServer:
    """Tests for VerificationNodeServer class."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database client."""
        db = AsyncMock()
        db.fetchone = AsyncMock(return_value={"signature": "prev_sig"})
        return db
    
    @pytest.fixture
    def public_key_pem(self):
        """Generate test public key."""
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        return private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_db, public_key_pem):
        """Test health check when healthy."""
        server = VerificationNodeServer(
            node_id="node1",
            db_client=mock_db,
            public_key_pem=public_key_pem
        )
        
        result = await server.health_check()
        
        assert result["status"] == "healthy"
        assert result["node_id"] == "node1"
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_db, public_key_pem):
        """Test health check when unhealthy."""
        mock_db.fetchone = AsyncMock(side_effect=Exception("DB error"))
        
        server = VerificationNodeServer(
            node_id="node1",
            db_client=mock_db,
            public_key_pem=public_key_pem
        )
        
        result = await server.health_check()
        
        assert result["status"] == "unhealthy"
        assert "error" in result


class TestPerformanceRequirements:
    """Tests for performance requirements (< 500ms verification delay)."""
    
    def test_timeout_configuration(self):
        """Test that default timeout is 500ms."""
        db = AsyncMock()
        service = DistributedVerificationService(
            db_client=db,
            nodes=[]
        )
        
        assert service.timeout_ms == 500
    
    def test_consensus_calculation_performance(self):
        """Test that consensus calculation is fast."""
        db = AsyncMock()
        nodes = [
            {"node_id": f"node{i}", "url": f"http://localhost:800{i}", "region": "test"}
            for i in range(100)
        ]
        service = DistributedVerificationService(
            db_client=db,
            nodes=nodes
        )
        
        now = datetime.now(timezone.utc)
        results = [
            NodeVerificationResult(
                node_id=f"node{i}",
                is_valid=i % 3 != 0,  # Mixed results
                hash_verified="hash",
                signature_verified=True,
                chain_verified=True,
                response_time_ms=50,
                timestamp=now
            )
            for i in range(50)
        ]
        
        start = time.perf_counter()
        consensus = service._calculate_consensus(
            request_id="req1",
            log_id="log1",
            queried_nodes=list(service.nodes.values())[:50],
            results=results,
            start_time=start
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Consensus calculation should be very fast (< 10ms)
        assert elapsed_ms < 10


class TestNodeHealth:
    """Tests for NodeHealth enum."""
    
    def test_health_values(self):
        """Test health enum values."""
        assert NodeHealth.HEALTHY.value == "healthy"
        assert NodeHealth.DEGRADED.value == "degraded"
        assert NodeHealth.UNHEALTHY.value == "unhealthy"
        assert NodeHealth.OFFLINE.value == "offline"


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""
    
    def test_status_values(self):
        """Test status enum values."""
        assert VerificationStatus.PENDING.value == "pending"
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.FAILED.value == "failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
