"""
Multi-signature Support for Enhanced Audit Security

Implements multi-signature approval process for sensitive operations.

Features:
- Multi-signature approval workflow
- Flexible key management scheme
- Configurable policy support
- Operation audit logging
"""
import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa, ec
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
import base64

logger = logging.getLogger(__name__)


class SignatureAlgorithm(str, Enum):
    """Supported signature algorithms."""
    RSA_PSS = "rsa_pss"
    ECDSA = "ecdsa"
    ED25519 = "ed25519"


class ApprovalStatus(str, Enum):
    """Approval request status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"


class OperationType(str, Enum):
    """Types of operations requiring multi-sig."""
    VERSION_PROMOTION = "version_promotion"
    VERSION_QUARANTINE = "version_quarantine"
    CONFIG_CHANGE = "config_change"
    USER_ADMIN = "user_admin"
    KEY_ROTATION = "key_rotation"
    POLICY_CHANGE = "policy_change"
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"


@dataclass
class Signer:
    """Multi-sig signer information."""
    signer_id: str
    name: str
    email: str
    public_key_pem: bytes
    algorithm: SignatureAlgorithm
    role: str
    weight: int = 1  # For weighted multi-sig
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signer_id": self.signer_id,
            "name": self.name,
            "email": self.email,
            "public_key_fingerprint": self._get_fingerprint(),
            "algorithm": self.algorithm.value,
            "role": self.role,
            "weight": self.weight,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None
        }

    def _get_fingerprint(self) -> str:
        """Get public key fingerprint."""
        return hashlib.sha256(self.public_key_pem).hexdigest()[:16]


@dataclass
class Signature:
    """Individual signature on an approval request."""
    signer_id: str
    signature: bytes
    timestamp: datetime
    comment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signer_id": self.signer_id,
            "signature": base64.b64encode(self.signature).decode(),
            "timestamp": self.timestamp.isoformat(),
            "comment": self.comment,
            "metadata": self.metadata
        }


@dataclass
class ApprovalPolicy:
    """Multi-sig approval policy."""
    policy_id: str
    name: str
    description: str
    operation_types: List[OperationType]
    required_signatures: int
    required_weight: Optional[int] = None  # For weighted multi-sig
    required_roles: List[str] = field(default_factory=list)
    timeout_hours: int = 24
    allow_self_approval: bool = False
    require_different_signers: bool = True
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "operation_types": [op.value for op in self.operation_types],
            "required_signatures": self.required_signatures,
            "required_weight": self.required_weight,
            "required_roles": self.required_roles,
            "timeout_hours": self.timeout_hours,
            "allow_self_approval": self.allow_self_approval,
            "require_different_signers": self.require_different_signers,
            "enabled": self.enabled
        }


@dataclass
class ApprovalRequest:
    """Multi-sig approval request."""
    request_id: str
    operation_type: OperationType
    policy_id: str
    requester_id: str
    payload: Dict[str, Any]
    payload_hash: str
    status: ApprovalStatus
    signatures: List[Signature]
    created_at: datetime
    expires_at: datetime
    executed_at: Optional[datetime] = None
    execution_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "operation_type": self.operation_type.value,
            "policy_id": self.policy_id,
            "requester_id": self.requester_id,
            "payload_hash": self.payload_hash,
            "status": self.status.value,
            "signatures": [s.to_dict() for s in self.signatures],
            "signature_count": len(self.signatures),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "execution_result": self.execution_result
        }


class SignatureVerifier:
    """Verifies cryptographic signatures."""

    @staticmethod
    def verify(
        public_key_pem: bytes,
        message: bytes,
        signature: bytes,
        algorithm: SignatureAlgorithm
    ) -> bool:
        """Verify a signature.

        Args:
            public_key_pem: Public key in PEM format
            message: Original message
            signature: Signature to verify
            algorithm: Signature algorithm

        Returns:
            True if signature is valid
        """
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem,
                backend=default_backend()
            )

            if algorithm == SignatureAlgorithm.RSA_PSS:
                public_key.verify(
                    signature,
                    message,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
            elif algorithm == SignatureAlgorithm.ECDSA:
                public_key.verify(
                    signature,
                    message,
                    ec.ECDSA(hashes.SHA256())
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            return True

        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False

    @staticmethod
    def sign(
        private_key_pem: bytes,
        message: bytes,
        algorithm: SignatureAlgorithm,
        password: Optional[bytes] = None
    ) -> bytes:
        """Sign a message.

        Args:
            private_key_pem: Private key in PEM format
            message: Message to sign
            algorithm: Signature algorithm
            password: Optional key password

        Returns:
            Signature bytes
        """
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=password,
            backend=default_backend()
        )

        if algorithm == SignatureAlgorithm.RSA_PSS:
            signature = private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        elif algorithm == SignatureAlgorithm.ECDSA:
            signature = private_key.sign(
                message,
                ec.ECDSA(hashes.SHA256())
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return signature


class KeyManager:
    """Manages multi-sig keys."""

    def __init__(self, db_client, encryption_key: bytes = None):
        """Initialize key manager.

        Args:
            db_client: Database client
            encryption_key: Optional key for encrypting stored keys
        """
        self.db = db_client
        self.encryption_key = encryption_key
        self._signers: Dict[str, Signer] = {}

    async def load_signers(self):
        """Load signers from database."""
        try:
            records = await self.db.fetch(
                "SELECT * FROM audits.multisig_signers WHERE enabled = true"
            )

            for record in records:
                signer = Signer(
                    signer_id=record["signer_id"],
                    name=record["name"],
                    email=record["email"],
                    public_key_pem=record["public_key_pem"].encode(),
                    algorithm=SignatureAlgorithm(record["algorithm"]),
                    role=record["role"],
                    weight=record["weight"],
                    enabled=record["enabled"],
                    created_at=record["created_at"],
                    last_used=record.get("last_used")
                )
                self._signers[signer.signer_id] = signer

            logger.info(f"Loaded {len(self._signers)} signers")

        except Exception as e:
            logger.error(f"Failed to load signers: {e}")

    async def add_signer(self, signer: Signer) -> bool:
        """Add a new signer.

        Args:
            signer: Signer to add

        Returns:
            True if successful
        """
        try:
            await self.db.execute(
                """
                INSERT INTO audits.multisig_signers
                (signer_id, name, email, public_key_pem, algorithm, role, weight, enabled, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                signer.signer_id,
                signer.name,
                signer.email,
                signer.public_key_pem.decode(),
                signer.algorithm.value,
                signer.role,
                signer.weight,
                signer.enabled,
                signer.created_at
            )

            self._signers[signer.signer_id] = signer
            logger.info(f"Added signer: {signer.signer_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add signer: {e}")
            return False

    async def remove_signer(self, signer_id: str) -> bool:
        """Remove a signer.

        Args:
            signer_id: ID of signer to remove

        Returns:
            True if successful
        """
        try:
            await self.db.execute(
                """
                UPDATE audits.multisig_signers
                SET enabled = false
                WHERE signer_id = $1
                """,
                signer_id
            )

            if signer_id in self._signers:
                del self._signers[signer_id]

            logger.info(f"Removed signer: {signer_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove signer: {e}")
            return False

    def get_signer(self, signer_id: str) -> Optional[Signer]:
        """Get a signer by ID."""
        return self._signers.get(signer_id)

    def get_signers_by_role(self, role: str) -> List[Signer]:
        """Get all signers with a specific role."""
        return [s for s in self._signers.values() if s.role == role and s.enabled]

    def get_all_signers(self) -> List[Signer]:
        """Get all enabled signers."""
        return [s for s in self._signers.values() if s.enabled]

    @staticmethod
    def generate_key_pair(
        algorithm: SignatureAlgorithm,
        key_size: int = 4096
    ) -> Tuple[bytes, bytes]:
        """Generate a new key pair.

        Args:
            algorithm: Signature algorithm
            key_size: Key size (for RSA)

        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        if algorithm == SignatureAlgorithm.RSA_PSS:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
        elif algorithm == SignatureAlgorithm.ECDSA:
            private_key = ec.generate_private_key(
                ec.SECP256R1(),
                backend=default_backend()
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

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


class MultiSigService:
    """Multi-signature approval service."""

    # Default policies
    DEFAULT_POLICIES = [
        ApprovalPolicy(
            policy_id="P001",
            name="Version Promotion",
            description="Requires 2 admin signatures for version promotion",
            operation_types=[OperationType.VERSION_PROMOTION],
            required_signatures=2,
            required_roles=["admin"],
            timeout_hours=24
        ),
        ApprovalPolicy(
            policy_id="P002",
            name="Version Quarantine",
            description="Requires 2 signatures for quarantine",
            operation_types=[OperationType.VERSION_QUARANTINE],
            required_signatures=2,
            timeout_hours=12
        ),
        ApprovalPolicy(
            policy_id="P003",
            name="Config Change",
            description="Requires 2 admin signatures for config changes",
            operation_types=[OperationType.CONFIG_CHANGE],
            required_signatures=2,
            required_roles=["admin"],
            timeout_hours=48
        ),
        ApprovalPolicy(
            policy_id="P004",
            name="Key Rotation",
            description="Requires 3 signatures for key rotation",
            operation_types=[OperationType.KEY_ROTATION],
            required_signatures=3,
            required_roles=["admin", "security"],
            timeout_hours=72
        ),
        ApprovalPolicy(
            policy_id="P005",
            name="Data Operations",
            description="Requires 2 admin signatures for data export/deletion",
            operation_types=[OperationType.DATA_EXPORT, OperationType.DATA_DELETION],
            required_signatures=2,
            required_roles=["admin"],
            timeout_hours=24,
            allow_self_approval=False
        )
    ]

    def __init__(
        self,
        db_client,
        key_manager: KeyManager,
        audit_logger=None
    ):
        """Initialize multi-sig service.

        Args:
            db_client: Database client
            key_manager: Key manager instance
            audit_logger: Optional audit logger for operation logging
        """
        self.db = db_client
        self.key_manager = key_manager
        self.audit_logger = audit_logger

        # Policies
        self.policies: Dict[str, ApprovalPolicy] = {
            p.policy_id: p for p in self.DEFAULT_POLICIES
        }

        # Pending requests
        self._requests: Dict[str, ApprovalRequest] = {}

        # Execution handlers
        self._handlers: Dict[OperationType, callable] = {}

        # Background task
        self._running = False
        self._expiry_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the multi-sig service."""
        await self.key_manager.load_signers()
        await self._load_pending_requests()

        self._running = True
        self._expiry_task = asyncio.create_task(self._expiry_loop())

        logger.info("Multi-sig service started")

    async def stop(self):
        """Stop the multi-sig service."""
        self._running = False

        if self._expiry_task:
            self._expiry_task.cancel()
            try:
                await self._expiry_task
            except asyncio.CancelledError:
                # Intentionally not re-raised: we initiated the cancellation
                # during shutdown, so propagation is not needed
                logger.debug("Expiry task cancelled during shutdown")

        logger.info("Multi-sig service stopped")

    async def _load_pending_requests(self):
        """Load pending requests from database."""
        try:
            records = await self.db.fetch(
                """
                SELECT * FROM audits.multisig_requests
                WHERE status = 'pending' AND expires_at > NOW()
                """
            )

            for record in records:
                signatures = json.loads(record["signatures"]) if record["signatures"] else []

                request = ApprovalRequest(
                    request_id=record["request_id"],
                    operation_type=OperationType(record["operation_type"]),
                    policy_id=record["policy_id"],
                    requester_id=record["requester_id"],
                    payload=json.loads(record["payload"]),
                    payload_hash=record["payload_hash"],
                    status=ApprovalStatus(record["status"]),
                    signatures=[
                        Signature(
                            signer_id=s["signer_id"],
                            signature=base64.b64decode(s["signature"]),
                            timestamp=datetime.fromisoformat(s["timestamp"]),
                            comment=s.get("comment")
                        )
                        for s in signatures
                    ],
                    created_at=record["created_at"],
                    expires_at=record["expires_at"]
                )
                self._requests[request.request_id] = request

            logger.info(f"Loaded {len(self._requests)} pending requests")

        except Exception as e:
            logger.error(f"Failed to load pending requests: {e}")

    async def _expiry_loop(self):
        """Background loop to expire old requests."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.now(timezone.utc)
                expired = []

                for request_id, request in self._requests.items():
                    if request.status == ApprovalStatus.PENDING and request.expires_at < now:
                        request.status = ApprovalStatus.EXPIRED
                        expired.append(request_id)

                for request_id in expired:
                    await self._update_request_status(request_id, ApprovalStatus.EXPIRED)
                    logger.info(f"Request {request_id} expired")

            except asyncio.CancelledError:
                logger.debug("Expiry loop cancelled")
                raise
            except Exception as e:
                logger.error(f"Expiry loop error: {e}")

    def register_handler(self, operation_type: OperationType, handler: callable):
        """Register an execution handler for an operation type.

        Args:
            operation_type: Type of operation
            handler: Async function to execute when approved
        """
        self._handlers[operation_type] = handler
        logger.info(f"Registered handler for {operation_type.value}")

    def add_policy(self, policy: ApprovalPolicy):
        """Add or update a policy."""
        self.policies[policy.policy_id] = policy
        logger.info(f"Added policy: {policy.policy_id}")

    def get_policy_for_operation(self, operation_type: OperationType) -> Optional[ApprovalPolicy]:
        """Get the policy for an operation type."""
        for policy in self.policies.values():
            if operation_type in policy.operation_types and policy.enabled:
                return policy
        return None

    async def create_request(
        self,
        operation_type: OperationType,
        requester_id: str,
        payload: Dict[str, Any]
    ) -> ApprovalRequest:
        """Create a new approval request.

        Args:
            operation_type: Type of operation
            requester_id: ID of the requester
            payload: Operation payload

        Returns:
            Created approval request
        """
        # Find applicable policy
        policy = self.get_policy_for_operation(operation_type)
        if not policy:
            raise ValueError(f"No policy found for operation: {operation_type}")

        # Create request
        request_id = hashlib.sha256(
            f"{operation_type.value}:{requester_id}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        payload_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

        now = datetime.now(timezone.utc)

        request = ApprovalRequest(
            request_id=request_id,
            operation_type=operation_type,
            policy_id=policy.policy_id,
            requester_id=requester_id,
            payload=payload,
            payload_hash=payload_hash,
            status=ApprovalStatus.PENDING,
            signatures=[],
            created_at=now,
            expires_at=now + timedelta(hours=policy.timeout_hours)
        )

        # Store in database
        await self.db.execute(
            """
            INSERT INTO audits.multisig_requests
            (request_id, operation_type, policy_id, requester_id, payload, payload_hash,
             status, signatures, created_at, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            request.request_id,
            request.operation_type.value,
            request.policy_id,
            request.requester_id,
            json.dumps(request.payload),
            request.payload_hash,
            request.status.value,
            json.dumps([]),
            request.created_at,
            request.expires_at
        )

        self._requests[request.request_id] = request

        # Log event
        if self.audit_logger:
            await self.audit_logger.log_event(
                entity="multisig",
                action="create_request",
                actor_id=requester_id,
                resource_id=request_id,
                payload={
                    "operation_type": operation_type.value,
                    "policy_id": policy.policy_id
                }
            )

        logger.info(f"Created approval request: {request_id}")
        return request

    async def add_signature(
        self,
        request_id: str,
        signer_id: str,
        signature: bytes,
        comment: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Add a signature to an approval request.

        Args:
            request_id: Request ID
            signer_id: Signer ID
            signature: Cryptographic signature
            comment: Optional comment

        Returns:
            Tuple of (success, message)
        """
        # Get request
        request = self._requests.get(request_id)
        if not request:
            return False, "Request not found"

        if request.status != ApprovalStatus.PENDING:
            return False, f"Request is {request.status.value}"

        if request.expires_at < datetime.now(timezone.utc):
            request.status = ApprovalStatus.EXPIRED
            await self._update_request_status(request_id, ApprovalStatus.EXPIRED)
            return False, "Request has expired"

        # Get signer
        signer = self.key_manager.get_signer(signer_id)
        if not signer:
            return False, "Signer not found"

        # Get policy
        policy = self.policies.get(request.policy_id)
        if not policy:
            return False, "Policy not found"

        # Check if signer already signed
        if policy.require_different_signers:
            existing_signers = {s.signer_id for s in request.signatures}
            if signer_id in existing_signers:
                return False, "Signer has already signed this request"

        # Check self-approval
        if not policy.allow_self_approval and signer_id == request.requester_id:
            return False, "Self-approval not allowed"

        # Check role requirement
        if policy.required_roles and signer.role not in policy.required_roles:
            return False, f"Signer role '{signer.role}' not in required roles"

        # Verify signature
        message = request.payload_hash.encode()
        if not SignatureVerifier.verify(
            signer.public_key_pem,
            message,
            signature,
            signer.algorithm
        ):
            return False, "Invalid signature"

        # Add signature
        sig = Signature(
            signer_id=signer_id,
            signature=signature,
            timestamp=datetime.now(timezone.utc),
            comment=comment
        )
        request.signatures.append(sig)

        # Update signer last used
        signer.last_used = datetime.now(timezone.utc)

        # Update database
        await self.db.execute(
            """
            UPDATE audits.multisig_requests
            SET signatures = $1
            WHERE request_id = $2
            """,
            json.dumps([s.to_dict() for s in request.signatures]),
            request_id
        )

        # Log event
        if self.audit_logger:
            await self.audit_logger.log_event(
                entity="multisig",
                action="add_signature",
                actor_id=signer_id,
                resource_id=request_id,
                payload={"comment": comment}
            )

        logger.info(f"Added signature from {signer_id} to request {request_id}")

        # Check if approved
        if self._check_approval(request, policy):
            request.status = ApprovalStatus.APPROVED
            await self._update_request_status(request_id, ApprovalStatus.APPROVED)

            # Execute if handler registered
            if request.operation_type in self._handlers:
                await self._execute_request(request)

            return True, "Request approved"

        return True, f"Signature added ({len(request.signatures)}/{policy.required_signatures})"

    def _check_approval(self, request: ApprovalRequest, policy: ApprovalPolicy) -> bool:
        """Check if request has enough signatures for approval."""
        # Count valid signatures
        valid_signatures = len(request.signatures)

        if valid_signatures < policy.required_signatures:
            return False

        # Check weight if required
        if policy.required_weight:
            total_weight = 0
            for sig in request.signatures:
                signer = self.key_manager.get_signer(sig.signer_id)
                if signer:
                    total_weight += signer.weight

            if total_weight < policy.required_weight:
                return False

        return True

    async def _execute_request(self, request: ApprovalRequest):
        """Execute an approved request."""
        handler = self._handlers.get(request.operation_type)
        if not handler:
            logger.warning(f"No handler for {request.operation_type}")
            return

        try:
            result = await handler(request.payload)

            request.status = ApprovalStatus.EXECUTED
            request.executed_at = datetime.now(timezone.utc)
            request.execution_result = result

            await self.db.execute(
                """
                UPDATE audits.multisig_requests
                SET status = $1, executed_at = $2, execution_result = $3
                WHERE request_id = $4
                """,
                request.status.value,
                request.executed_at,
                json.dumps(result) if result else None,
                request.request_id
            )

            # Log event
            if self.audit_logger:
                await self.audit_logger.log_event(
                    entity="multisig",
                    action="execute_request",
                    actor_id="system",
                    resource_id=request.request_id,
                    payload={"result": result}
                )

            logger.info(f"Executed request {request.request_id}")

        except Exception as e:
            logger.error(f"Failed to execute request {request.request_id}: {e}")

            if self.audit_logger:
                await self.audit_logger.log_event(
                    entity="multisig",
                    action="execute_request",
                    actor_id="system",
                    resource_id=request.request_id,
                    payload={"error": str(e)},
                    status="failure"
                )

    async def reject_request(
        self,
        request_id: str,
        rejector_id: str,
        reason: str
    ) -> bool:
        """Reject an approval request.

        Args:
            request_id: Request ID
            rejector_id: ID of the rejector
            reason: Rejection reason

        Returns:
            True if successful
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        if request.status != ApprovalStatus.PENDING:
            return False

        request.status = ApprovalStatus.REJECTED

        await self.db.execute(
            """
            UPDATE audits.multisig_requests
            SET status = $1, execution_result = $2
            WHERE request_id = $3
            """,
            request.status.value,
            json.dumps({"rejected_by": rejector_id, "reason": reason}),
            request_id
        )

        # Log event
        if self.audit_logger:
            await self.audit_logger.log_event(
                entity="multisig",
                action="reject_request",
                actor_id=rejector_id,
                resource_id=request_id,
                payload={"reason": reason}
            )

        logger.info(f"Request {request_id} rejected by {rejector_id}")
        return True

    async def _update_request_status(self, request_id: str, status: ApprovalStatus):
        """Update request status in database."""
        await self.db.execute(
            """
            UPDATE audits.multisig_requests
            SET status = $1
            WHERE request_id = $2
            """,
            status.value,
            request_id
        )

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a request by ID."""
        return self._requests.get(request_id)

    def get_pending_requests(
        self,
        operation_type: OperationType = None,
        requester_id: str = None
    ) -> List[ApprovalRequest]:
        """Get pending requests."""
        requests = [
            r for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING
        ]

        if operation_type:
            requests = [r for r in requests if r.operation_type == operation_type]

        if requester_id:
            requests = [r for r in requests if r.requester_id == requester_id]

        return sorted(requests, key=lambda r: r.created_at, reverse=True)

    def get_requests_needing_signature(self, signer_id: str) -> List[ApprovalRequest]:
        """Get requests that need signature from a specific signer."""
        signer = self.key_manager.get_signer(signer_id)
        if not signer:
            return []

        result = []
        for request in self._requests.values():
            if request.status != ApprovalStatus.PENDING:
                continue

            # Check if already signed
            existing_signers = {s.signer_id for s in request.signatures}
            if signer_id in existing_signers:
                continue

            # Check policy
            policy = self.policies.get(request.policy_id)
            if not policy:
                continue

            # Check role
            if policy.required_roles and signer.role not in policy.required_roles:
                continue

            # Check self-approval
            if not policy.allow_self_approval and signer_id == request.requester_id:
                continue

            result.append(request)

        return sorted(result, key=lambda r: r.expires_at)

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "total_signers": len(self.key_manager.get_all_signers()),
            "total_policies": len(self.policies),
            "pending_requests": len([
                r for r in self._requests.values()
                if r.status == ApprovalStatus.PENDING
            ]),
            "approved_requests": len([
                r for r in self._requests.values()
                if r.status == ApprovalStatus.APPROVED
            ]),
            "rejected_requests": len([
                r for r in self._requests.values()
                if r.status == ApprovalStatus.REJECTED
            ]),
            "executed_requests": len([
                r for r in self._requests.values()
                if r.status == ApprovalStatus.EXECUTED
            ])
        }


# Database migration for multi-sig tables
MULTISIG_MIGRATION = """
-- Multi-sig signers table
CREATE TABLE IF NOT EXISTS audits.multisig_signers (
    signer_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    public_key_pem TEXT NOT NULL,
    algorithm VARCHAR(20) NOT NULL,
    role VARCHAR(50) NOT NULL,
    weight INTEGER NOT NULL DEFAULT 1,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used TIMESTAMPTZ
);

-- Multi-sig approval requests table
CREATE TABLE IF NOT EXISTS audits.multisig_requests (
    request_id VARCHAR(64) PRIMARY KEY,
    operation_type VARCHAR(50) NOT NULL,
    policy_id VARCHAR(64) NOT NULL,
    requester_id VARCHAR(64) NOT NULL,
    payload JSONB NOT NULL,
    payload_hash VARCHAR(64) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    signatures JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    executed_at TIMESTAMPTZ,
    execution_result JSONB,
    CONSTRAINT valid_status CHECK (status IN ('pending', 'approved', 'rejected', 'expired', 'executed'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_multisig_requests_status
    ON audits.multisig_requests(status);
CREATE INDEX IF NOT EXISTS idx_multisig_requests_operation
    ON audits.multisig_requests(operation_type);
CREATE INDEX IF NOT EXISTS idx_multisig_requests_requester
    ON audits.multisig_requests(requester_id);
CREATE INDEX IF NOT EXISTS idx_multisig_requests_expires
    ON audits.multisig_requests(expires_at);
"""
