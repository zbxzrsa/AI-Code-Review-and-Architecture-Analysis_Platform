"""
Distributed Verification System

Cross-node audit verification mechanism with consensus-based validation.

Features:
- Multi-node verification protocol
- Consensus algorithm for result consistency
- Exception handling with automatic retry
- Performance target: verification delay < 500ms
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import aiohttp
from collections import Counter

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    """Verification status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    FAILED = "failed"
    INCONSISTENT = "inconsistent"
    TIMEOUT = "timeout"


class NodeHealth(str, Enum):
    """Node health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class VerificationNode:
    """Verification node configuration."""
    node_id: str
    url: str
    region: str
    priority: int = 0
    health: NodeHealth = NodeHealth.HEALTHY
    last_heartbeat: Optional[datetime] = None
    avg_response_time_ms: float = 0.0
    success_rate: float = 1.0
    consecutive_failures: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "url": self.url,
            "region": self.region,
            "priority": self.priority,
            "health": self.health.value,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "avg_response_time_ms": self.avg_response_time_ms,
            "success_rate": self.success_rate,
            "consecutive_failures": self.consecutive_failures
        }


@dataclass
class NodeVerificationResult:
    """Verification result from a single node."""
    node_id: str
    is_valid: bool
    hash_verified: str
    signature_verified: bool
    chain_verified: bool
    response_time_ms: float
    timestamp: datetime
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "is_valid": self.is_valid,
            "hash_verified": self.hash_verified,
            "signature_verified": self.signature_verified,
            "chain_verified": self.chain_verified,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error
        }


@dataclass
class ConsensusResult:
    """Consensus result from distributed verification."""
    request_id: str
    log_id: str
    is_valid: bool
    consensus_reached: bool
    consensus_ratio: float
    total_nodes: int
    responding_nodes: int
    agreeing_nodes: int
    status: VerificationStatus
    node_results: List[NodeVerificationResult]
    total_time_ms: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "log_id": self.log_id,
            "is_valid": self.is_valid,
            "consensus_reached": self.consensus_reached,
            "consensus_ratio": self.consensus_ratio,
            "total_nodes": self.total_nodes,
            "responding_nodes": self.responding_nodes,
            "agreeing_nodes": self.agreeing_nodes,
            "status": self.status.value,
            "node_results": [r.to_dict() for r in self.node_results],
            "total_time_ms": self.total_time_ms,
            "timestamp": self.timestamp.isoformat()
        }


class DistributedVerificationService:
    """Distributed verification service for audit logs."""
    
    def __init__(
        self,
        db_client,
        nodes: List[Dict[str, Any]],
        consensus_threshold: float = 0.67,  # 2/3 majority
        timeout_ms: int = 500,  # Performance target
        min_nodes: int = 3,
        max_retries: int = 2,
        healthcheck_interval_seconds: int = 30
    ):
        """Initialize distributed verification service.
        
        Args:
            db_client: Database client
            nodes: List of verification node configurations
            consensus_threshold: Required ratio for consensus (0.67 = 2/3 majority)
            timeout_ms: Maximum verification time in milliseconds
            min_nodes: Minimum nodes required for verification
            max_retries: Maximum retry attempts per node
            healthcheck_interval_seconds: Health check interval
        """
        self.db = db_client
        self.consensus_threshold = consensus_threshold
        self.timeout_ms = timeout_ms
        self.min_nodes = min_nodes
        self.max_retries = max_retries
        self.healthcheck_interval = healthcheck_interval_seconds
        
        # Initialize nodes
        self.nodes: Dict[str, VerificationNode] = {}
        for node_config in nodes:
            node = VerificationNode(
                node_id=node_config["node_id"],
                url=node_config["url"],
                region=node_config.get("region", "unknown"),
                priority=node_config.get("priority", 0)
            )
            self.nodes[node.node_id] = node
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._healthcheck_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._total_verifications = 0
        self._successful_verifications = 0
        self._failed_verifications = 0
        self._avg_verification_time_ms = 0.0
    
    async def start(self):
        """Start the distributed verification service."""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=10,
            keepalive_timeout=30
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout_ms / 1000)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        self._running = True
        self._healthcheck_task = asyncio.create_task(self._healthcheck_loop())
        
        logger.info(
            f"Distributed verification service started with {len(self.nodes)} nodes"
        )
    
    async def stop(self):
        """Stop the distributed verification service."""
        self._running = False
        
        if self._healthcheck_task:
            self._healthcheck_task.cancel()
            try:
                await self._healthcheck_task
            except asyncio.CancelledError:
                logger.debug("Healthcheck task cancelled")
        
        if self._session:
            await self._session.close()
        
        logger.info("Distributed verification service stopped")
    
    async def _healthcheck_loop(self):
        """Background health check loop."""
        while self._running:
            try:
                await asyncio.sleep(self.healthcheck_interval)
                await self._check_all_nodes_health()
            except asyncio.CancelledError:
                logger.debug("Healthcheck loop cancelled")
                raise
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _check_all_nodes_health(self):
        """Check health of all verification nodes."""
        tasks = []
        for node_id, node in self.nodes.items():
            tasks.append(self._check_node_health(node))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_node_health(self, node: VerificationNode):
        """Check health of a single node."""
        if not self._session:
            return
        
        try:
            start_time = time.perf_counter()
            
            async with self._session.get(
                f"{node.url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                if response.status == 200:
                    node.last_heartbeat = datetime.now(timezone.utc)
                    node.consecutive_failures = 0
                    
                    # Update average response time
                    if node.avg_response_time_ms == 0:
                        node.avg_response_time_ms = elapsed_ms
                    else:
                        # Exponential moving average
                        node.avg_response_time_ms = (
                            0.9 * node.avg_response_time_ms + 0.1 * elapsed_ms
                        )
                    
                    # Update health status
                    if elapsed_ms < 100:
                        node.health = NodeHealth.HEALTHY
                    elif elapsed_ms < 300:
                        node.health = NodeHealth.DEGRADED
                    else:
                        node.health = NodeHealth.DEGRADED
                else:
                    self._handle_node_failure(node)
                    
        except asyncio.TimeoutError:
            self._handle_node_failure(node, "timeout")
        except Exception as e:
            self._handle_node_failure(node, str(e))
    
    def _handle_node_failure(self, node: VerificationNode, error: str = "unknown"):
        """Handle node failure."""
        node.consecutive_failures += 1
        
        if node.consecutive_failures >= 3:
            node.health = NodeHealth.UNHEALTHY
        elif node.consecutive_failures >= 5:
            node.health = NodeHealth.OFFLINE
        else:
            node.health = NodeHealth.DEGRADED
        
        # Update success rate
        total = max(1, self._total_verifications)
        node.success_rate = max(0, node.success_rate - (1 / total))
        
        logger.warning(
            f"Node {node.node_id} failure ({node.consecutive_failures}): {error}"
        )
    
    def _get_available_nodes(self) -> List[VerificationNode]:
        """Get list of available (healthy) nodes."""
        available = [
            node for node in self.nodes.values()
            if node.health in (NodeHealth.HEALTHY, NodeHealth.DEGRADED)
        ]
        
        # Sort by priority, then by response time
        available.sort(key=lambda n: (-n.priority, n.avg_response_time_ms))
        
        return available
    
    async def verify_log(
        self,
        log_id: str,
        signature: str,
        expected_hash: str,
        prev_hash: Optional[str] = None
    ) -> ConsensusResult:
        """Verify an audit log entry across multiple nodes.
        
        Args:
            log_id: ID of the audit log
            signature: Cryptographic signature to verify
            expected_hash: Expected hash of the log entry
            prev_hash: Previous log entry hash for chain verification
            
        Returns:
            ConsensusResult with verification details
        """
        request_id = hashlib.sha256(
            f"{log_id}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]
        
        start_time = time.perf_counter()
        
        # Get available nodes
        available_nodes = self._get_available_nodes()
        
        if len(available_nodes) < self.min_nodes:
            logger.warning(
                f"Insufficient nodes: {len(available_nodes)} < {self.min_nodes}"
            )
            return ConsensusResult(
                request_id=request_id,
                log_id=log_id,
                is_valid=False,
                consensus_reached=False,
                consensus_ratio=0.0,
                total_nodes=len(self.nodes),
                responding_nodes=0,
                agreeing_nodes=0,
                status=VerificationStatus.FAILED,
                node_results=[],
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                timestamp=datetime.now(timezone.utc)
            )
        
        # Prepare verification request
        verification_request = {
            "request_id": request_id,
            "log_id": log_id,
            "signature": signature,
            "expected_hash": expected_hash,
            "prev_hash": prev_hash
        }
        
        # Send verification requests to all nodes concurrently
        tasks = []
        for node in available_nodes:
            task = asyncio.create_task(
                self._verify_on_node(node, verification_request)
            )
            tasks.append(task)
        
        # Wait for all responses with timeout
        try:
            node_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_ms / 1000
            )
        except asyncio.TimeoutError:
            # Collect whatever results we have
            node_results = []
            for task in tasks:
                if task.done():
                    try:
                        node_results.append(task.result())
                    except Exception:
                        pass
        
        # Process results
        successful_results: List[NodeVerificationResult] = []
        for result in node_results:
            if isinstance(result, NodeVerificationResult):
                successful_results.append(result)
        
        # Calculate consensus
        consensus_result = self._calculate_consensus(
            request_id,
            log_id,
            available_nodes,
            successful_results,
            start_time
        )
        
        # Update metrics
        self._total_verifications += 1
        if consensus_result.is_valid and consensus_result.consensus_reached:
            self._successful_verifications += 1
        else:
            self._failed_verifications += 1
        
        # Update average time
        self._avg_verification_time_ms = (
            0.95 * self._avg_verification_time_ms + 
            0.05 * consensus_result.total_time_ms
        )
        
        # Log result
        if consensus_result.consensus_reached:
            logger.info(
                f"Verification complete: log={log_id}, valid={consensus_result.is_valid}, "
                f"consensus={consensus_result.consensus_ratio:.2%}, "
                f"time={consensus_result.total_time_ms:.1f}ms"
            )
        else:
            logger.warning(
                f"Verification failed: log={log_id}, "
                f"consensus={consensus_result.consensus_ratio:.2%} "
                f"(required: {self.consensus_threshold:.2%})"
            )
        
        return consensus_result
    
    async def _verify_on_node(
        self,
        node: VerificationNode,
        request: Dict[str, Any]
    ) -> NodeVerificationResult:
        """Send verification request to a single node."""
        if not self._session:
            raise RuntimeError("Service not started")
        
        start_time = time.perf_counter()
        
        for retry in range(self.max_retries + 1):
            try:
                async with self._session.post(
                    f"{node.url}/verify",
                    json=request
                ) as response:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update node metrics
                        node.consecutive_failures = 0
                        node.avg_response_time_ms = (
                            0.9 * node.avg_response_time_ms + 0.1 * elapsed_ms
                        )
                        
                        return NodeVerificationResult(
                            node_id=node.node_id,
                            is_valid=result.get("is_valid", False),
                            hash_verified=result.get("hash_verified", ""),
                            signature_verified=result.get("signature_verified", False),
                            chain_verified=result.get("chain_verified", False),
                            response_time_ms=elapsed_ms,
                            timestamp=datetime.now(timezone.utc)
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                        
            except asyncio.TimeoutError:
                if retry == self.max_retries:
                    self._handle_node_failure(node, "timeout")
                    return NodeVerificationResult(
                        node_id=node.node_id,
                        is_valid=False,
                        hash_verified="",
                        signature_verified=False,
                        chain_verified=False,
                        response_time_ms=(time.perf_counter() - start_time) * 1000,
                        timestamp=datetime.now(timezone.utc),
                        error="timeout"
                    )
                continue
                
            except Exception as e:
                if retry == self.max_retries:
                    self._handle_node_failure(node, str(e))
                    return NodeVerificationResult(
                        node_id=node.node_id,
                        is_valid=False,
                        hash_verified="",
                        signature_verified=False,
                        chain_verified=False,
                        response_time_ms=(time.perf_counter() - start_time) * 1000,
                        timestamp=datetime.now(timezone.utc),
                        error=str(e)
                    )
                await asyncio.sleep(0.05 * (retry + 1))  # Backoff
        
        # Should not reach here
        return NodeVerificationResult(
            node_id=node.node_id,
            is_valid=False,
            hash_verified="",
            signature_verified=False,
            chain_verified=False,
            response_time_ms=(time.perf_counter() - start_time) * 1000,
            timestamp=datetime.now(timezone.utc),
            error="max_retries_exceeded"
        )
    
    def _calculate_consensus(
        self,
        request_id: str,
        log_id: str,
        queried_nodes: List[VerificationNode],
        results: List[NodeVerificationResult],
        start_time: float
    ) -> ConsensusResult:
        """Calculate consensus from node results."""
        total_time_ms = (time.perf_counter() - start_time) * 1000
        
        responding_nodes = len(results)
        
        if responding_nodes == 0:
            return ConsensusResult(
                request_id=request_id,
                log_id=log_id,
                is_valid=False,
                consensus_reached=False,
                consensus_ratio=0.0,
                total_nodes=len(queried_nodes),
                responding_nodes=0,
                agreeing_nodes=0,
                status=VerificationStatus.TIMEOUT,
                node_results=[],
                total_time_ms=total_time_ms,
                timestamp=datetime.now(timezone.utc)
            )
        
        # Count valid/invalid votes
        valid_votes = sum(1 for r in results if r.is_valid and not r.error)
        invalid_votes = sum(1 for r in results if not r.is_valid and not r.error)
        error_votes = sum(1 for r in results if r.error)
        
        # Exclude error votes from consensus calculation
        effective_votes = responding_nodes - error_votes
        
        if effective_votes == 0:
            return ConsensusResult(
                request_id=request_id,
                log_id=log_id,
                is_valid=False,
                consensus_reached=False,
                consensus_ratio=0.0,
                total_nodes=len(queried_nodes),
                responding_nodes=responding_nodes,
                agreeing_nodes=0,
                status=VerificationStatus.FAILED,
                node_results=results,
                total_time_ms=total_time_ms,
                timestamp=datetime.now(timezone.utc)
            )
        
        # Determine majority opinion
        majority_is_valid = valid_votes > invalid_votes
        agreeing_nodes = valid_votes if majority_is_valid else invalid_votes
        consensus_ratio = agreeing_nodes / effective_votes
        
        # Check if consensus threshold is met
        consensus_reached = consensus_ratio >= self.consensus_threshold
        
        # Determine status
        if consensus_reached:
            status = VerificationStatus.VERIFIED
        elif consensus_ratio > 0.5:
            status = VerificationStatus.INCONSISTENT
        else:
            status = VerificationStatus.FAILED
        
        return ConsensusResult(
            request_id=request_id,
            log_id=log_id,
            is_valid=majority_is_valid if consensus_reached else False,
            consensus_reached=consensus_reached,
            consensus_ratio=consensus_ratio,
            total_nodes=len(queried_nodes),
            responding_nodes=responding_nodes,
            agreeing_nodes=agreeing_nodes,
            status=status,
            node_results=results,
            total_time_ms=total_time_ms,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def verify_batch(
        self,
        log_entries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify a batch of audit log entries.
        
        Args:
            log_entries: List of log entries with id, signature, hash, prev_hash
            
        Returns:
            Batch verification result
        """
        results = []
        verified_count = 0
        failed_count = 0
        total_time_ms = 0
        
        for entry in log_entries:
            result = await self.verify_log(
                log_id=entry["id"],
                signature=entry["signature"],
                expected_hash=entry["hash"],
                prev_hash=entry.get("prev_hash")
            )
            
            results.append(result.to_dict())
            total_time_ms += result.total_time_ms
            
            if result.is_valid and result.consensus_reached:
                verified_count += 1
            else:
                failed_count += 1
        
        return {
            "total": len(log_entries),
            "verified": verified_count,
            "failed": failed_count,
            "avg_time_ms": total_time_ms / len(log_entries) if log_entries else 0,
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get status of all verification nodes."""
        return {
            "total_nodes": len(self.nodes),
            "healthy_nodes": sum(
                1 for n in self.nodes.values() 
                if n.health == NodeHealth.HEALTHY
            ),
            "degraded_nodes": sum(
                1 for n in self.nodes.values() 
                if n.health == NodeHealth.DEGRADED
            ),
            "unhealthy_nodes": sum(
                1 for n in self.nodes.values() 
                if n.health in (NodeHealth.UNHEALTHY, NodeHealth.OFFLINE)
            ),
            "nodes": {
                node_id: node.to_dict() 
                for node_id, node in self.nodes.items()
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get verification metrics."""
        return {
            "total_verifications": self._total_verifications,
            "successful_verifications": self._successful_verifications,
            "failed_verifications": self._failed_verifications,
            "success_rate": (
                self._successful_verifications / self._total_verifications
                if self._total_verifications > 0 else 0
            ),
            "avg_verification_time_ms": self._avg_verification_time_ms,
            "consensus_threshold": self.consensus_threshold,
            "timeout_ms": self.timeout_ms,
            "min_nodes": self.min_nodes
        }
    
    async def add_node(self, node_config: Dict[str, Any]):
        """Add a new verification node.
        
        Args:
            node_config: Node configuration with node_id, url, region
        """
        node = VerificationNode(
            node_id=node_config["node_id"],
            url=node_config["url"],
            region=node_config.get("region", "unknown"),
            priority=node_config.get("priority", 0)
        )
        
        # Perform initial health check
        await self._check_node_health(node)
        
        self.nodes[node.node_id] = node
        logger.info(f"Added verification node: {node.node_id} ({node.url})")
    
    async def remove_node(self, node_id: str):
        """Remove a verification node.
        
        Args:
            node_id: ID of the node to remove
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Removed verification node: {node_id}")
        else:
            logger.warning(f"Node not found: {node_id}")


class VerificationNodeServer:
    """Verification node server implementation.
    
    This is the server-side implementation that runs on each verification node.
    """
    
    def __init__(
        self,
        node_id: str,
        db_client,
        public_key_pem: bytes
    ):
        """Initialize verification node server.
        
        Args:
            node_id: Unique identifier for this node
            db_client: Database client for local audit log access
            public_key_pem: Public key for signature verification
        """
        self.node_id = node_id
        self.db = db_client
        
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        
        self.public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )
    
    async def verify(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process verification request.
        
        Args:
            request: Verification request with log details
            
        Returns:
            Verification result
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        
        try:
            log_id = request["log_id"]
            signature_hex = request["signature"]
            expected_hash = request["expected_hash"]
            prev_hash = request.get("prev_hash")
            
            # Compute hash of expected_hash
            computed_hash = hashlib.sha256(expected_hash.encode()).hexdigest()
            hash_verified = computed_hash
            
            # Verify signature
            signature_bytes = bytes.fromhex(signature_hex)
            hash_bytes = bytes.fromhex(expected_hash)
            
            try:
                self.public_key.verify(
                    signature_bytes,
                    hash_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                signature_verified = True
            except Exception:
                signature_verified = False
            
            # Verify chain (if prev_hash provided)
            chain_verified = True
            if prev_hash:
                # Get previous log entry from local database
                prev_log = await self.db.fetchone(
                    """
                    SELECT signature FROM audits.audit_log
                    WHERE id = (
                        SELECT id FROM audits.audit_log
                        WHERE id < $1
                        ORDER BY id DESC
                        LIMIT 1
                    )
                    """,
                    log_id
                )
                
                if prev_log:
                    chain_verified = prev_log["signature"] == prev_hash
            
            # Overall validity
            is_valid = signature_verified and chain_verified
            
            return {
                "is_valid": is_valid,
                "hash_verified": hash_verified,
                "signature_verified": signature_verified,
                "chain_verified": chain_verified,
                "node_id": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Verification error on node {self.node_id}: {e}")
            return {
                "is_valid": False,
                "hash_verified": "",
                "signature_verified": False,
                "chain_verified": False,
                "node_id": self.node_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Check database connectivity
            await self.db.fetchone("SELECT 1")
            
            return {
                "status": "healthy",
                "node_id": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "node_id": self.node_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
