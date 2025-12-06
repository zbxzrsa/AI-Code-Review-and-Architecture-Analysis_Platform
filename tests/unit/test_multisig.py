"""
Unit tests for Multi-signature Support System.

Tests cover:
- Key management
- Signature verification
- Approval workflow
- Policy enforcement
"""
import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import base64

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.shared.security.multisig import (
    MultiSigService,
    KeyManager,
    SignatureVerifier,
    Signer,
    Signature,
    ApprovalPolicy,
    ApprovalRequest,
    SignatureAlgorithm,
    ApprovalStatus,
    OperationType,
    MULTISIG_MIGRATION
)


class TestSignatureVerifier:
    """Tests for SignatureVerifier class."""
    
    @pytest.fixture
    def rsa_key_pair(self):
        """Generate RSA key pair for testing."""
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    @pytest.fixture
    def ecdsa_key_pair(self):
        """Generate ECDSA key pair for testing."""
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        
        private_key = ec.generate_private_key(
            ec.SECP256R1(),
            backend=default_backend()
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def test_rsa_sign_and_verify(self, rsa_key_pair):
        """Test RSA signature and verification."""
        private_pem, public_pem = rsa_key_pair
        message = b"test message"
        
        signature = SignatureVerifier.sign(
            private_pem,
            message,
            SignatureAlgorithm.RSA_PSS
        )
        
        is_valid = SignatureVerifier.verify(
            public_pem,
            message,
            signature,
            SignatureAlgorithm.RSA_PSS
        )
        
        assert is_valid
    
    def test_ecdsa_sign_and_verify(self, ecdsa_key_pair):
        """Test ECDSA signature and verification."""
        private_pem, public_pem = ecdsa_key_pair
        message = b"test message"
        
        signature = SignatureVerifier.sign(
            private_pem,
            message,
            SignatureAlgorithm.ECDSA
        )
        
        is_valid = SignatureVerifier.verify(
            public_pem,
            message,
            signature,
            SignatureAlgorithm.ECDSA
        )
        
        assert is_valid
    
    def test_invalid_signature(self, rsa_key_pair):
        """Test verification of invalid signature."""
        _, public_pem = rsa_key_pair
        message = b"test message"
        fake_signature = b"invalid_signature" * 32
        
        is_valid = SignatureVerifier.verify(
            public_pem,
            message,
            fake_signature,
            SignatureAlgorithm.RSA_PSS
        )
        
        assert not is_valid
    
    def test_wrong_message(self, rsa_key_pair):
        """Test verification with wrong message."""
        private_pem, public_pem = rsa_key_pair
        
        signature = SignatureVerifier.sign(
            private_pem,
            b"original message",
            SignatureAlgorithm.RSA_PSS
        )
        
        is_valid = SignatureVerifier.verify(
            public_pem,
            b"different message",
            signature,
            SignatureAlgorithm.RSA_PSS
        )
        
        assert not is_valid


class TestKeyManager:
    """Tests for KeyManager class."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database client."""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.fetch = AsyncMock(return_value=[])
        return db
    
    def test_generate_rsa_key_pair(self):
        """Test RSA key pair generation."""
        private_pem, public_pem = KeyManager.generate_key_pair(
            SignatureAlgorithm.RSA_PSS,
            key_size=2048
        )
        
        assert b"BEGIN PRIVATE KEY" in private_pem
        assert b"BEGIN PUBLIC KEY" in public_pem
    
    def test_generate_ecdsa_key_pair(self):
        """Test ECDSA key pair generation."""
        private_pem, public_pem = KeyManager.generate_key_pair(
            SignatureAlgorithm.ECDSA
        )
        
        assert b"BEGIN PRIVATE KEY" in private_pem
        assert b"BEGIN PUBLIC KEY" in public_pem
    
    @pytest.mark.asyncio
    async def test_add_signer(self, mock_db):
        """Test adding a signer."""
        _, public_pem = KeyManager.generate_key_pair(SignatureAlgorithm.RSA_PSS)
        
        km = KeyManager(db_client=mock_db)
        
        signer = Signer(
            signer_id="signer1",
            name="Test User",
            email="test@example.com",
            public_key_pem=public_pem,
            algorithm=SignatureAlgorithm.RSA_PSS,
            role="admin"
        )
        
        result = await km.add_signer(signer)
        assert result is True
        assert km.get_signer("signer1") is not None
    
    @pytest.mark.asyncio
    async def test_remove_signer(self, mock_db):
        """Test removing a signer."""
        _, public_pem = KeyManager.generate_key_pair(SignatureAlgorithm.RSA_PSS)
        
        km = KeyManager(db_client=mock_db)
        
        signer = Signer(
            signer_id="signer1",
            name="Test User",
            email="test@example.com",
            public_key_pem=public_pem,
            algorithm=SignatureAlgorithm.RSA_PSS,
            role="admin"
        )
        
        await km.add_signer(signer)
        result = await km.remove_signer("signer1")
        
        assert result is True
        assert km.get_signer("signer1") is None
    
    def test_get_signers_by_role(self, mock_db):
        """Test getting signers by role."""
        _, public_pem = KeyManager.generate_key_pair(SignatureAlgorithm.RSA_PSS)
        
        km = KeyManager(db_client=mock_db)
        
        km._signers = {
            "admin1": Signer(
                signer_id="admin1",
                name="Admin",
                email="admin@example.com",
                public_key_pem=public_pem,
                algorithm=SignatureAlgorithm.RSA_PSS,
                role="admin"
            ),
            "user1": Signer(
                signer_id="user1",
                name="User",
                email="user@example.com",
                public_key_pem=public_pem,
                algorithm=SignatureAlgorithm.RSA_PSS,
                role="user"
            )
        }
        
        admins = km.get_signers_by_role("admin")
        assert len(admins) == 1
        assert admins[0].signer_id == "admin1"


class TestSigner:
    """Tests for Signer dataclass."""
    
    def test_signer_creation(self):
        """Test creating a signer."""
        _, public_pem = KeyManager.generate_key_pair(SignatureAlgorithm.RSA_PSS)
        
        signer = Signer(
            signer_id="signer1",
            name="Test User",
            email="test@example.com",
            public_key_pem=public_pem,
            algorithm=SignatureAlgorithm.RSA_PSS,
            role="admin",
            weight=2
        )
        
        assert signer.weight == 2
        assert signer.enabled is True
    
    def test_signer_to_dict(self):
        """Test signer serialization."""
        _, public_pem = KeyManager.generate_key_pair(SignatureAlgorithm.RSA_PSS)
        
        signer = Signer(
            signer_id="signer1",
            name="Test User",
            email="test@example.com",
            public_key_pem=public_pem,
            algorithm=SignatureAlgorithm.RSA_PSS,
            role="admin"
        )
        
        d = signer.to_dict()
        assert d["signer_id"] == "signer1"
        assert "public_key_fingerprint" in d


class TestApprovalPolicy:
    """Tests for ApprovalPolicy dataclass."""
    
    def test_policy_creation(self):
        """Test creating a policy."""
        policy = ApprovalPolicy(
            policy_id="P001",
            name="Test Policy",
            description="Test description",
            operation_types=[OperationType.VERSION_PROMOTION],
            required_signatures=2,
            required_roles=["admin"]
        )
        
        assert policy.timeout_hours == 24  # Default
        assert policy.enabled is True
    
    def test_policy_to_dict(self):
        """Test policy serialization."""
        policy = ApprovalPolicy(
            policy_id="P001",
            name="Test Policy",
            description="Test",
            operation_types=[OperationType.VERSION_PROMOTION, OperationType.CONFIG_CHANGE],
            required_signatures=3
        )
        
        d = policy.to_dict()
        assert d["policy_id"] == "P001"
        assert len(d["operation_types"]) == 2


class TestApprovalRequest:
    """Tests for ApprovalRequest dataclass."""
    
    def test_request_creation(self):
        """Test creating an approval request."""
        request = ApprovalRequest(
            request_id="req1",
            operation_type=OperationType.VERSION_PROMOTION,
            policy_id="P001",
            requester_id="user1",
            payload={"version": "v1.0"},
            payload_hash="hash123",
            status=ApprovalStatus.PENDING,
            signatures=[],
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
        )
        
        assert request.executed_at is None
    
    def test_request_to_dict(self):
        """Test request serialization."""
        request = ApprovalRequest(
            request_id="req1",
            operation_type=OperationType.VERSION_PROMOTION,
            policy_id="P001",
            requester_id="user1",
            payload={"version": "v1.0"},
            payload_hash="hash123",
            status=ApprovalStatus.PENDING,
            signatures=[],
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
        )
        
        d = request.to_dict()
        assert d["request_id"] == "req1"
        assert d["status"] == "pending"


class TestMultiSigService:
    """Tests for MultiSigService class."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database client."""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.fetch = AsyncMock(return_value=[])
        db.fetchone = AsyncMock(return_value=None)
        return db
    
    @pytest.fixture
    def key_manager(self, mock_db):
        """Create key manager with test signers."""
        km = KeyManager(db_client=mock_db)
        
        _, public_pem = KeyManager.generate_key_pair(SignatureAlgorithm.RSA_PSS)
        
        km._signers = {
            "admin1": Signer(
                signer_id="admin1",
                name="Admin 1",
                email="admin1@example.com",
                public_key_pem=public_pem,
                algorithm=SignatureAlgorithm.RSA_PSS,
                role="admin"
            ),
            "admin2": Signer(
                signer_id="admin2",
                name="Admin 2",
                email="admin2@example.com",
                public_key_pem=public_pem,
                algorithm=SignatureAlgorithm.RSA_PSS,
                role="admin"
            )
        }
        
        return km
    
    def test_service_initialization(self, mock_db, key_manager):
        """Test service initialization."""
        service = MultiSigService(
            db_client=mock_db,
            key_manager=key_manager
        )
        
        assert len(service.policies) > 0  # Default policies
    
    def test_get_policy_for_operation(self, mock_db, key_manager):
        """Test getting policy for operation type."""
        service = MultiSigService(
            db_client=mock_db,
            key_manager=key_manager
        )
        
        policy = service.get_policy_for_operation(OperationType.VERSION_PROMOTION)
        assert policy is not None
        assert OperationType.VERSION_PROMOTION in policy.operation_types
    
    @pytest.mark.asyncio
    async def test_create_request(self, mock_db, key_manager):
        """Test creating approval request."""
        service = MultiSigService(
            db_client=mock_db,
            key_manager=key_manager
        )
        
        request = await service.create_request(
            operation_type=OperationType.VERSION_PROMOTION,
            requester_id="user1",
            payload={"version": "v1.0"}
        )
        
        assert request is not None
        assert request.status == ApprovalStatus.PENDING
        assert request.request_id in service._requests
    
    @pytest.mark.asyncio
    async def test_create_request_no_policy(self, mock_db, key_manager):
        """Test creating request with no applicable policy."""
        service = MultiSigService(
            db_client=mock_db,
            key_manager=key_manager
        )
        
        # Clear all policies
        service.policies.clear()
        
        with pytest.raises(ValueError):
            await service.create_request(
                operation_type=OperationType.VERSION_PROMOTION,
                requester_id="user1",
                payload={}
            )
    
    @pytest.mark.asyncio
    async def test_reject_request(self, mock_db, key_manager):
        """Test rejecting request."""
        service = MultiSigService(
            db_client=mock_db,
            key_manager=key_manager
        )
        
        request = await service.create_request(
            operation_type=OperationType.VERSION_PROMOTION,
            requester_id="user1",
            payload={}
        )
        
        result = await service.reject_request(
            request.request_id,
            "admin1",
            "Not needed"
        )
        
        assert result is True
        assert request.status == ApprovalStatus.REJECTED
    
    def test_get_pending_requests(self, mock_db, key_manager):
        """Test getting pending requests."""
        service = MultiSigService(
            db_client=mock_db,
            key_manager=key_manager
        )
        
        # Add some requests
        service._requests = {
            "req1": ApprovalRequest(
                request_id="req1",
                operation_type=OperationType.VERSION_PROMOTION,
                policy_id="P001",
                requester_id="user1",
                payload={},
                payload_hash="hash1",
                status=ApprovalStatus.PENDING,
                signatures=[],
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
            ),
            "req2": ApprovalRequest(
                request_id="req2",
                operation_type=OperationType.CONFIG_CHANGE,
                policy_id="P003",
                requester_id="user2",
                payload={},
                payload_hash="hash2",
                status=ApprovalStatus.APPROVED,
                signatures=[],
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
            )
        }
        
        pending = service.get_pending_requests()
        assert len(pending) == 1
        assert pending[0].request_id == "req1"
    
    def test_get_status(self, mock_db, key_manager):
        """Test getting service status."""
        service = MultiSigService(
            db_client=mock_db,
            key_manager=key_manager
        )
        
        status = service.get_status()
        
        assert "total_signers" in status
        assert "total_policies" in status
        assert "pending_requests" in status
    
    def test_add_policy(self, mock_db, key_manager):
        """Test adding a custom policy."""
        service = MultiSigService(
            db_client=mock_db,
            key_manager=key_manager
        )
        
        new_policy = ApprovalPolicy(
            policy_id="CUSTOM001",
            name="Custom Policy",
            description="Custom test policy",
            operation_types=[OperationType.DATA_DELETION],
            required_signatures=5
        )
        
        service.add_policy(new_policy)
        assert "CUSTOM001" in service.policies
    
    def test_check_approval_simple(self, mock_db, key_manager):
        """Test approval checking with simple signature count."""
        service = MultiSigService(
            db_client=mock_db,
            key_manager=key_manager
        )
        
        policy = ApprovalPolicy(
            policy_id="test",
            name="Test",
            description="Test",
            operation_types=[OperationType.VERSION_PROMOTION],
            required_signatures=2
        )
        
        request = ApprovalRequest(
            request_id="req1",
            operation_type=OperationType.VERSION_PROMOTION,
            policy_id="test",
            requester_id="user1",
            payload={},
            payload_hash="hash",
            status=ApprovalStatus.PENDING,
            signatures=[
                Signature(signer_id="admin1", signature=b"sig1", timestamp=datetime.now(timezone.utc))
            ],
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
        )
        
        # Not enough signatures
        assert not service._check_approval(request, policy)
        
        # Add another signature
        request.signatures.append(
            Signature(signer_id="admin2", signature=b"sig2", timestamp=datetime.now(timezone.utc))
        )
        
        # Now should be approved
        assert service._check_approval(request, policy)
    
    def test_check_approval_with_weight(self, mock_db, key_manager):
        """Test approval checking with weighted signatures."""
        service = MultiSigService(
            db_client=mock_db,
            key_manager=key_manager
        )
        
        # Update signers with weights
        key_manager._signers["admin1"].weight = 2
        key_manager._signers["admin2"].weight = 1
        
        policy = ApprovalPolicy(
            policy_id="test",
            name="Test",
            description="Test",
            operation_types=[OperationType.VERSION_PROMOTION],
            required_signatures=1,
            required_weight=3
        )
        
        request = ApprovalRequest(
            request_id="req1",
            operation_type=OperationType.VERSION_PROMOTION,
            policy_id="test",
            requester_id="user1",
            payload={},
            payload_hash="hash",
            status=ApprovalStatus.PENDING,
            signatures=[
                Signature(signer_id="admin1", signature=b"sig1", timestamp=datetime.now(timezone.utc))
            ],
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
        )
        
        # Weight 2 < required 3
        assert not service._check_approval(request, policy)
        
        # Add admin2 (weight 1), total weight = 3
        request.signatures.append(
            Signature(signer_id="admin2", signature=b"sig2", timestamp=datetime.now(timezone.utc))
        )
        
        assert service._check_approval(request, policy)


class TestMultiSigEnums:
    """Tests for enum types."""
    
    def test_signature_algorithm_values(self):
        """Test signature algorithm enum values."""
        assert SignatureAlgorithm.RSA_PSS.value == "rsa_pss"
        assert SignatureAlgorithm.ECDSA.value == "ecdsa"
    
    def test_approval_status_values(self):
        """Test approval status enum values."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
    
    def test_operation_type_values(self):
        """Test operation type enum values."""
        assert OperationType.VERSION_PROMOTION.value == "version_promotion"
        assert OperationType.KEY_ROTATION.value == "key_rotation"


class TestMultiSigMigration:
    """Tests for database migration SQL."""
    
    def test_migration_contains_tables(self):
        """Test that migration creates required tables."""
        assert "multisig_signers" in MULTISIG_MIGRATION
        assert "multisig_requests" in MULTISIG_MIGRATION
    
    def test_migration_contains_indexes(self):
        """Test that migration creates indexes."""
        assert "idx_multisig_requests_status" in MULTISIG_MIGRATION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
