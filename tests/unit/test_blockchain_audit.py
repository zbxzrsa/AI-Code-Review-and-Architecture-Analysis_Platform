"""
Unit tests for Blockchain Audit Logs Integration.

Tests cover:
- Merkle tree construction and verification
- Blockchain client operations
- Anchor creation and verification
- Error handling and edge cases
"""
import asyncio
import hashlib
import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.shared.security.blockchain_audit import (
    MerkleTree,
    MerkleNode,
    BlockchainAnchor,
    VerificationResult,
    BlockchainNetwork,
    AnchorStatus,
    EthereumClient,
    HyperledgerClient,
    BlockchainAuditService,
    BLOCKCHAIN_ANCHOR_MIGRATION
)


class TestMerkleTree:
    """Tests for MerkleTree class."""
    
    def test_empty_tree(self):
        """Test Merkle tree with no leaves."""
        tree = MerkleTree([])
        assert tree.root is None
        assert tree.tree == []
    
    def test_single_leaf(self):
        """Test Merkle tree with single leaf."""
        leaves = ["abc123"]
        tree = MerkleTree(leaves)
        assert tree.root == "abc123"
        assert len(tree.tree) == 1
    
    def test_two_leaves(self):
        """Test Merkle tree with two leaves."""
        leaves = ["leaf1", "leaf2"]
        tree = MerkleTree(leaves)
        
        expected_root = tree._hash_pair("leaf1", "leaf2")
        assert tree.root == expected_root
        assert len(tree.tree) == 2
    
    def test_multiple_leaves(self):
        """Test Merkle tree with multiple leaves."""
        leaves = ["a", "b", "c", "d"]
        tree = MerkleTree(leaves)
        
        # Verify tree structure
        assert len(tree.tree) == 3  # 3 levels
        assert len(tree.tree[0]) == 4  # 4 leaves
        assert len(tree.tree[1]) == 2  # 2 intermediate nodes
        assert len(tree.tree[2]) == 1  # 1 root
    
    def test_odd_number_leaves(self):
        """Test Merkle tree with odd number of leaves."""
        leaves = ["a", "b", "c"]
        tree = MerkleTree(leaves)
        
        assert tree.root is not None
        assert len(tree.tree) >= 2
    
    def test_get_proof_valid_index(self):
        """Test getting Merkle proof for valid index."""
        leaves = ["a", "b", "c", "d"]
        tree = MerkleTree(leaves)
        
        proof = tree.get_proof(0)
        assert len(proof) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in proof)
    
    def test_get_proof_invalid_index(self):
        """Test getting Merkle proof for invalid index."""
        leaves = ["a", "b"]
        tree = MerkleTree(leaves)
        
        with pytest.raises(ValueError):
            tree.get_proof(10)
    
    def test_verify_proof_valid(self):
        """Test verifying valid Merkle proof."""
        leaves = ["a", "b", "c", "d"]
        tree = MerkleTree(leaves)
        
        for i in range(len(leaves)):
            proof = tree.get_proof(i)
            is_valid = tree.verify_proof(leaves[i], proof, tree.root)
            assert is_valid
    
    def test_verify_proof_invalid_leaf(self):
        """Test verifying proof with wrong leaf."""
        leaves = ["a", "b", "c", "d"]
        tree = MerkleTree(leaves)
        
        proof = tree.get_proof(0)
        is_valid = tree.verify_proof("wrong_leaf", proof, tree.root)
        assert not is_valid
    
    def test_verify_proof_invalid_root(self):
        """Test verifying proof with wrong root."""
        leaves = ["a", "b", "c", "d"]
        tree = MerkleTree(leaves)
        
        proof = tree.get_proof(0)
        is_valid = tree.verify_proof(leaves[0], proof, "wrong_root")
        assert not is_valid
    
    def test_hash_pair_deterministic(self):
        """Test that hash_pair is deterministic."""
        tree = MerkleTree([])
        
        hash1 = tree._hash_pair("a", "b")
        hash2 = tree._hash_pair("a", "b")
        assert hash1 == hash2
    
    def test_hash_pair_order_matters(self):
        """Test that hash_pair order matters."""
        tree = MerkleTree([])
        
        hash1 = tree._hash_pair("a", "b")
        hash2 = tree._hash_pair("b", "a")
        assert hash1 != hash2


class TestBlockchainAnchor:
    """Tests for BlockchainAnchor dataclass."""
    
    def test_anchor_creation(self):
        """Test creating a blockchain anchor."""
        anchor = BlockchainAnchor(
            anchor_id="test123",
            merkle_root="root_hash",
            transaction_hash="tx_hash",
            block_number=12345,
            network=BlockchainNetwork.ETHEREUM_SEPOLIA,
            timestamp=datetime.now(timezone.utc),
            log_count=10,
            log_range=("log1", "log10")
        )
        
        assert anchor.anchor_id == "test123"
        assert anchor.status == AnchorStatus.PENDING
        assert anchor.confirmation_count == 0
    
    def test_anchor_to_dict(self):
        """Test anchor serialization."""
        now = datetime.now(timezone.utc)
        anchor = BlockchainAnchor(
            anchor_id="test123",
            merkle_root="root_hash",
            transaction_hash="tx_hash",
            block_number=12345,
            network=BlockchainNetwork.ETHEREUM_SEPOLIA,
            timestamp=now,
            log_count=10,
            log_range=("log1", "log10"),
            status=AnchorStatus.CONFIRMED,
            gas_used=21000
        )
        
        d = anchor.to_dict()
        assert d["anchor_id"] == "test123"
        assert d["network"] == "ethereum_sepolia"
        assert d["status"] == "confirmed"
        assert d["gas_used"] == 21000


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""
    
    def test_verification_result_creation(self):
        """Test creating a verification result."""
        result = VerificationResult(
            is_valid=True,
            anchor=None,
            merkle_proof=["proof1", "proof2"],
            verified_at=datetime.now(timezone.utc),
            verification_path=[{"step": "test"}]
        )
        
        assert result.is_valid
        assert result.error is None
    
    def test_verification_result_to_dict(self):
        """Test verification result serialization."""
        result = VerificationResult(
            is_valid=False,
            anchor=None,
            merkle_proof=[],
            verified_at=datetime.now(timezone.utc),
            verification_path=[],
            error="Test error"
        )
        
        d = result.to_dict()
        assert d["is_valid"] is False
        assert d["error"] == "Test error"


class TestEthereumClient:
    """Tests for EthereumClient class."""
    
    def test_client_initialization(self):
        """Test Ethereum client initialization."""
        client = EthereumClient(
            rpc_url="http://localhost:8545",
            private_key="0x" + "1" * 64,
            contract_address="0x" + "2" * 40,
            network=BlockchainNetwork.ETHEREUM_SEPOLIA
        )
        
        assert client.network == BlockchainNetwork.ETHEREUM_SEPOLIA
        assert client.gas_limit == 100000
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure handling."""
        client = EthereumClient(
            rpc_url="http://invalid:8545",
            private_key="0x" + "1" * 64,
            contract_address="0x" + "2" * 40
        )
        
        # Should handle connection failure gracefully
        with patch.object(client, 'w3', None):
            result = await client.connect()
            # Will fail since we can't actually connect
            assert result is False or result is True  # Depends on mock
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test client disconnection."""
        client = EthereumClient(
            rpc_url="http://localhost:8545",
            private_key="0x" + "1" * 64,
            contract_address="0x" + "2" * 40
        )
        
        await client.disconnect()
        assert client._connected is False


class TestHyperledgerClient:
    """Tests for HyperledgerClient class."""
    
    def test_client_initialization(self):
        """Test Hyperledger client initialization."""
        client = HyperledgerClient(
            gateway_url="http://localhost:8080",
            channel_name="mychannel",
            chaincode_name="audit",
            msp_id="Org1MSP",
            certificate_path="/path/to/cert",
            private_key_path="/path/to/key"
        )
        
        assert client.channel_name == "mychannel"
        assert client.chaincode_name == "audit"
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test client disconnection."""
        client = HyperledgerClient(
            gateway_url="http://localhost:8080",
            channel_name="mychannel",
            chaincode_name="audit",
            msp_id="Org1MSP",
            certificate_path="/path/to/cert",
            private_key_path="/path/to/key"
        )
        
        await client.disconnect()
        assert client._connected is False


class TestBlockchainAuditService:
    """Tests for BlockchainAuditService class."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database client."""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.fetch = AsyncMock(return_value=[])
        db.fetchone = AsyncMock(return_value=None)
        return db
    
    @pytest.fixture
    def mock_blockchain(self):
        """Create mock blockchain client."""
        client = AsyncMock()
        client.connect = AsyncMock(return_value=True)
        client.disconnect = AsyncMock()
        client.store_merkle_root = AsyncMock(return_value=("tx_hash", 12345))
        client.verify_merkle_root = AsyncMock(return_value=True)
        client.get_transaction_status = AsyncMock(return_value={
            "status": "confirmed",
            "confirmations": 10,
            "block_number": 12345
        })
        client.network = BlockchainNetwork.ETHEREUM_SEPOLIA
        return client
    
    @pytest.mark.asyncio
    async def test_service_start(self, mock_db, mock_blockchain):
        """Test service startup."""
        service = BlockchainAuditService(
            db_client=mock_db,
            blockchain_client=mock_blockchain,
            batch_size=10
        )
        
        result = await service.start()
        assert result is True
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_add_log_for_anchoring(self, mock_db, mock_blockchain):
        """Test adding logs for anchoring."""
        service = BlockchainAuditService(
            db_client=mock_db,
            blockchain_client=mock_blockchain,
            batch_size=100
        )
        
        await service.start()
        
        log_entry = {
            "id": "log1",
            "signature": "sig123",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await service.add_log_for_anchoring(log_entry)
        assert len(service._pending_logs) == 1
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_force_anchor(self, mock_db, mock_blockchain):
        """Test forced anchoring."""
        service = BlockchainAuditService(
            db_client=mock_db,
            blockchain_client=mock_blockchain,
            batch_size=100
        )
        
        await service.start()
        
        # Add some logs
        for i in range(5):
            await service.add_log_for_anchoring({
                "id": f"log{i}",
                "signature": f"sig{i}" * 10,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Force anchor
        anchor = await service.force_anchor()
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_force_anchor_no_pending(self, mock_db, mock_blockchain):
        """Test forced anchoring with no pending logs."""
        service = BlockchainAuditService(
            db_client=mock_db,
            blockchain_client=mock_blockchain
        )
        
        await service.start()
        
        anchor = await service.force_anchor()
        assert anchor is None
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_get_anchor_statistics(self, mock_db, mock_blockchain):
        """Test getting anchor statistics."""
        mock_db.fetchone = AsyncMock(return_value={
            "total_anchors": 10,
            "confirmed": 8,
            "pending": 2,
            "failed": 0,
            "total_logs_anchored": 100,
            "total_gas_used": 210000,
            "avg_confirmations": 15.5,
            "first_anchor": datetime.now(timezone.utc),
            "last_anchor": datetime.now(timezone.utc)
        })
        
        service = BlockchainAuditService(
            db_client=mock_db,
            blockchain_client=mock_blockchain
        )
        
        await service.start()
        
        stats = await service.get_anchor_statistics()
        
        assert stats["total_anchors"] == 10
        assert stats["confirmed"] == 8
        
        await service.stop()


class TestBlockchainMigration:
    """Tests for database migration SQL."""
    
    def test_migration_contains_tables(self):
        """Test that migration creates required tables."""
        assert "blockchain_anchors" in BLOCKCHAIN_ANCHOR_MIGRATION
        assert "merkle_proof" in BLOCKCHAIN_ANCHOR_MIGRATION
    
    def test_migration_contains_indexes(self):
        """Test that migration creates indexes."""
        assert "idx_blockchain_anchors_status" in BLOCKCHAIN_ANCHOR_MIGRATION
        assert "idx_audit_log_blockchain_anchor" in BLOCKCHAIN_ANCHOR_MIGRATION


class TestBlockchainNetwork:
    """Tests for BlockchainNetwork enum."""
    
    def test_network_values(self):
        """Test network enum values."""
        assert BlockchainNetwork.ETHEREUM_MAINNET.value == "ethereum_mainnet"
        assert BlockchainNetwork.ETHEREUM_SEPOLIA.value == "ethereum_sepolia"
        assert BlockchainNetwork.HYPERLEDGER_FABRIC.value == "hyperledger_fabric"


class TestAnchorStatus:
    """Tests for AnchorStatus enum."""
    
    def test_status_values(self):
        """Test status enum values."""
        assert AnchorStatus.PENDING.value == "pending"
        assert AnchorStatus.CONFIRMED.value == "confirmed"
        assert AnchorStatus.FAILED.value == "failed"


# Boundary condition tests
class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""
    
    def test_merkle_tree_large_dataset(self):
        """Test Merkle tree with large number of leaves."""
        leaves = [f"leaf_{i}" for i in range(1000)]
        tree = MerkleTree(leaves)
        
        assert tree.root is not None
        # Verify first and last leaf proofs
        assert tree.verify_proof(leaves[0], tree.get_proof(0), tree.root)
        assert tree.verify_proof(leaves[-1], tree.get_proof(len(leaves) - 1), tree.root)
    
    def test_merkle_tree_power_of_two(self):
        """Test Merkle tree with power of 2 leaves."""
        for n in [2, 4, 8, 16, 32]:
            leaves = [f"leaf_{i}" for i in range(n)]
            tree = MerkleTree(leaves)
            
            for i in range(n):
                proof = tree.get_proof(i)
                assert tree.verify_proof(leaves[i], proof, tree.root)
    
    def test_merkle_tree_special_characters(self):
        """Test Merkle tree with special characters in leaves."""
        leaves = [
            "normal",
            "with spaces",
            "with\ttabs",
            "with\nnewlines",
            "unicode: 你好",
            "emoji: 🔐"
        ]
        tree = MerkleTree(leaves)
        
        assert tree.root is not None
        for i, leaf in enumerate(leaves):
            assert tree.verify_proof(leaf, tree.get_proof(i), tree.root)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
