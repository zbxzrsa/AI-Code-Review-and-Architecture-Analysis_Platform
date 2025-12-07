"""
Tamper-proof audit logger with cryptographic signatures and chain verification.

Implements immutable audit trail with SHA256 hashing, RSA signatures, and chain validation.
"""
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from enum import Enum

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

# SQL query constants
AUDIT_BASE_QUERY = "SELECT * FROM audits.audit_log WHERE 1=1"
AUDIT_TS_GTE_PARAM = " AND ts >= $"
AUDIT_TS_LTE_PARAM = " AND ts <= $"
AUDIT_ENTITY_PARAM = " AND entity = $"


class AuditAction(str, Enum):
    """Audit actions."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    PROMOTE = "promote"
    QUARANTINE = "quarantine"
    EXECUTE = "execute"
    LOGIN = "login"
    LOGOUT = "logout"
    PERMISSION_DENIED = "permission_denied"
    QUOTA_EXCEEDED = "quota_exceeded"
    PROVIDER_FAILURE = "provider_failure"


class AuditEntity(str, Enum):
    """Audit entities."""
    VERSION = "version"
    EXPERIMENT = "experiment"
    PROJECT = "project"
    USER = "user"
    API_KEY = "api_key"
    PROVIDER = "provider"
    ANALYSIS = "analysis"
    SESSION = "session"


class TamperProofAuditLogger:
    """Tamper-proof audit logger with cryptographic signatures."""

    def __init__(
        self,
        db_client,
        private_key_path: str = None,
        private_key_password: bytes = None
    ):
        """Initialize audit logger."""
        self.db = db_client
        self.private_key = self._load_private_key(private_key_path, private_key_password)
        self.public_key = self.private_key.public_key()

    def _load_private_key(
        self,
        key_path: str = None,
        password: bytes = None
    ) -> rsa.RSAPrivateKey:
        """Load private key from file or generate new one."""
        try:
            if key_path:
                with open(key_path, "rb") as key_file:
                    private_key = serialization.load_pem_private_key(
                        key_file.read(),
                        password=password,
                        backend=default_backend()
                    )
                    logger.info(f"Private key loaded from {key_path}")
                    return private_key
            else:
                # Generate new key if not provided
                logger.warning("Generating new RSA key pair (not recommended for production)")
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=4096,
                    backend=default_backend()
                )
                return private_key
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")
            raise

    def _compute_log_hash(self, log_entry: Dict[str, Any]) -> bytes:
        """Compute SHA256 hash of log entry."""
        try:
            # Serialize with sorted keys for consistency
            log_bytes = json.dumps(log_entry, sort_keys=True, default=str).encode()
            digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
            digest.update(log_bytes)
            return digest.finalize()
        except Exception as e:
            logger.error(f"Failed to compute log hash: {e}")
            raise

    def _sign_hash(self, log_hash: bytes) -> bytes:
        """Sign hash with private key using RSA-PSS."""
        try:
            signature = self.private_key.sign(
                log_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature
        except Exception as e:
            logger.error(f"Failed to sign hash: {e}")
            raise

    async def _get_previous_signature(self) -> Optional[str]:
        """Get signature of previous log entry for chain."""
        try:
            result = await self.db.fetchone(
                """
                SELECT signature FROM audits.audit_log
                ORDER BY ts DESC, id DESC
                LIMIT 1
                """
            )
            return result["signature"] if result else None
        except Exception as e:
            logger.error(f"Failed to get previous signature: {e}")
            return None

    async def log_event(
        self,
        entity: str,
        action: str,
        actor_id: str,
        payload: Dict[str, Any],
        resource_id: str = None,
        status: str = "success"
    ) -> str:
        """Log event with cryptographic signature."""
        try:
            # Get previous signature for chain
            prev_signature = await self._get_previous_signature()
            prev_hash = prev_signature if prev_signature else "0" * 64

            # Construct log entry
            log_entry = {
                "entity": entity,
                "action": action,
                "actor_id": actor_id,
                "resource_id": resource_id,
                "payload": payload,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prev_hash": prev_hash
            }

            # Compute hash
            log_hash = self._compute_log_hash(log_entry)
            _log_hash_hex = log_hash.hex()  # Available for debug logging

            # Sign hash
            signature = self._sign_hash(log_hash)
            signature_hex = signature.hex()

            # Store in database
            await self.db.execute(
                """
                INSERT INTO audits.audit_log
                (entity, action, actor_id, resource_id, payload, signature, prev_hash, ts, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                entity,
                action,
                actor_id,
                resource_id,
                json.dumps(payload, default=str),
                signature_hex,
                prev_hash,
                log_entry["timestamp"],
                status
            )

            logger.info(
                f"Audit event logged: {entity}:{action} by {actor_id} "
                f"(signature: {signature_hex[:16]}...)"
            )

            return signature_hex
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise

    async def verify_integrity(
        self,
        from_ts: datetime = None,
        to_ts: datetime = None,
        entity: str = None
    ) -> Dict[str, Any]:
        """Verify audit log chain integrity."""
        try:
            # Build query
            query = AUDIT_BASE_QUERY
            params = []

            if from_ts:
                query += AUDIT_TS_GTE_PARAM + str(len(params) + 1)
                params.append(from_ts)

            if to_ts:
                query += AUDIT_TS_LTE_PARAM + str(len(params) + 1)
                params.append(to_ts)

            if entity:
                query += AUDIT_ENTITY_PARAM + str(len(params) + 1)
                params.append(entity)

            query += " ORDER BY ts ASC, id ASC"

            logs = await self.db.fetch(query, *params)

            if not logs:
                return {"valid": True, "verified_count": 0, "message": "No logs to verify"}

            tampered_logs = []
            broken_chains = []

            for i, log in enumerate(logs):
                # Reconstruct log entry
                log_entry = {
                    "entity": log["entity"],
                    "action": log["action"],
                    "actor_id": log["actor_id"],
                    "resource_id": log["resource_id"],
                    "payload": json.loads(log["payload"]),
                    "status": log["status"],
                    "timestamp": log["ts"].isoformat(),
                    "prev_hash": log["prev_hash"]
                }

                # Compute hash
                log_hash = self._compute_log_hash(log_entry)

                # Verify signature
                try:
                    self.public_key.verify(
                        bytes.fromhex(log["signature"]),
                        log_hash,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )
                except Exception as e:
                    tampered_logs.append({
                        "id": log["id"],
                        "timestamp": log["ts"],
                        "error": str(e)
                    })
                    logger.error(f"Signature verification failed for log {log['id']}: {e}")
                    continue

                # Verify chain
                if i > 0:
                    expected_prev_hash = logs[i - 1]["signature"]
                    if log["prev_hash"] != expected_prev_hash:
                        broken_chains.append({
                            "id": log["id"],
                            "timestamp": log["ts"],
                            "expected_prev_hash": expected_prev_hash,
                            "actual_prev_hash": log["prev_hash"]
                        })
                        logger.error(f"Chain broken at log {log['id']}")

            # Return results
            is_valid = len(tampered_logs) == 0 and len(broken_chains) == 0

            result = {
                "valid": is_valid,
                "verified_count": len(logs),
                "tampered_logs": tampered_logs,
                "broken_chains": broken_chains,
                "verification_time": datetime.now(timezone.utc).isoformat()
            }

            if is_valid:
                logger.info(f"Audit log integrity verified: {len(logs)} entries")
            else:
                logger.warning(
                    f"Audit log integrity issues found: "
                    f"{len(tampered_logs)} tampered, {len(broken_chains)} broken chains"
                )

            return result
        except Exception as e:
            logger.error(f"Failed to verify audit integrity: {e}")
            raise

    async def get_audit_trail(
        self,
        actor_id: str = None,
        entity: str = None,
        action: str = None,
        from_ts: datetime = None,
        to_ts: datetime = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit trail for specific criteria."""
        try:
            query = "SELECT * FROM audits.audit_log WHERE 1=1"
            params = []

            if actor_id:
                query += " AND actor_id = $" + str(len(params) + 1)
                params.append(actor_id)

            if entity:
                query += " AND entity = $" + str(len(params) + 1)
                params.append(entity)

            if action:
                query += " AND action = $" + str(len(params) + 1)
                params.append(action)

            if from_ts:
                query += " AND ts >= $" + str(len(params) + 1)
                params.append(from_ts)

            if to_ts:
                query += " AND ts <= $" + str(len(params) + 1)
                params.append(to_ts)

            query += " ORDER BY ts DESC, id DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)

            logs = await self.db.fetch(query, *params)

            return [
                {
                    "id": log["id"],
                    "entity": log["entity"],
                    "action": log["action"],
                    "actor_id": log["actor_id"],
                    "resource_id": log["resource_id"],
                    "payload": json.loads(log["payload"]),
                    "status": log["status"],
                    "timestamp": log["ts"].isoformat(),
                    "signature": log["signature"][:16] + "..."  # Truncate for display
                }
                for log in logs
            ]
        except Exception as e:
            logger.error(f"Failed to get audit trail: {e}")
            raise

    async def export_audit_log(
        self,
        from_ts: datetime,
        to_ts: datetime,
        format: str = "json"
    ) -> str:
        """Export audit log for compliance."""
        try:
            logs = await self.get_audit_trail(from_ts=from_ts, to_ts=to_ts, limit=10000)

            if format == "json":
                return json.dumps(logs, indent=2, default=str)
            elif format == "csv":
                import csv
                import io

                output = io.StringIO()
                writer = csv.DictWriter(
                    output,
                    fieldnames=[
                        "id", "entity", "action", "actor_id", "resource_id",
                        "status", "timestamp"
                    ]
                )
                writer.writeheader()

                for log in logs:
                    writer.writerow({
                        "id": log.get("id"),
                        "entity": log.get("entity"),
                        "action": log.get("action"),
                        "actor_id": log.get("actor_id"),
                        "resource_id": log.get("resource_id"),
                        "status": log.get("status"),
                        "timestamp": log.get("timestamp")
                    })

                return output.getvalue()
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logger.error(f"Failed to export audit log: {e}")
            raise

    async def get_statistics(
        self,
        from_ts: datetime = None,
        to_ts: datetime = None
    ) -> Dict[str, Any]:
        """Get audit log statistics."""
        try:
            query = "SELECT * FROM audits.audit_log WHERE 1=1"
            params = []

            if from_ts:
                query += " AND ts >= $" + str(len(params) + 1)
                params.append(from_ts)

            if to_ts:
                query += " AND ts <= $" + str(len(params) + 1)
                params.append(to_ts)

            logs = await self.db.fetch(query, *params)

            # Calculate statistics
            total_entries = len(logs)
            entities = {}
            actions = {}
            actors = {}
            statuses = {}

            for log in logs:
                # Count by entity
                entities[log["entity"]] = entities.get(log["entity"], 0) + 1

                # Count by action
                actions[log["action"]] = actions.get(log["action"], 0) + 1

                # Count by actor
                actors[log["actor_id"]] = actors.get(log["actor_id"], 0) + 1

                # Count by status
                statuses[log["status"]] = statuses.get(log["status"], 0) + 1

            return {
                "total_entries": total_entries,
                "by_entity": entities,
                "by_action": actions,
                "by_actor": actors,
                "by_status": statuses,
                "period": {
                    "from": from_ts.isoformat() if from_ts else None,
                    "to": to_ts.isoformat() if to_ts else None
                }
            }
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            raise

    def export_public_key(self, output_path: str) -> bool:
        """Export public key for verification."""
        try:
            public_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            with open(output_path, "wb") as f:
                f.write(public_pem)

            logger.info(f"Public key exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export public key: {e}")
            return False

    def export_private_key(
        self,
        output_path: str,
        password: bytes = None
    ) -> bool:
        """Export private key (for backup only)."""
        try:
            if password:
                encryption = serialization.BestAvailableEncryption(password)
            else:
                encryption = serialization.NoEncryption()

            private_pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=encryption
            )

            with open(output_path, "wb") as f:
                f.write(private_pem)

            logger.warning(f"Private key exported to {output_path} (SECURE THIS FILE)")
            return True
        except Exception as e:
            logger.error(f"Failed to export private key: {e}")
            return False
