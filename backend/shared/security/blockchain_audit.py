"""
Blockchain Audit Logs Integration

Implements immutable audit log storage on blockchain with support for
multiple blockchain networks (Ethereum, Hyperledger Fabric, private chains).

Features:
- Multi-chain support (Ethereum, Hyperledger, Private)
- Batch transaction optimization
- Merkle tree for efficient verification
- Anchor points for efficient queries
- Complete API documentation
"""
import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import aiohttp
from eth_account import Account
from web3 import Web3, AsyncWeb3
from web3.middleware import construct_sign_and_send_raw_middleware

logger = logging.getLogger(__name__)


class BlockchainNetwork(str, Enum):
    """Supported blockchain networks."""
    ETHEREUM_MAINNET = "ethereum_mainnet"
    ETHEREUM_SEPOLIA = "ethereum_sepolia"
    POLYGON = "polygon"
    HYPERLEDGER_FABRIC = "hyperledger_fabric"
    PRIVATE_CHAIN = "private_chain"


class AnchorStatus(str, Enum):
    """Anchor transaction status."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"


@dataclass
class MerkleNode:
    """Merkle tree node."""
    hash: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None


@dataclass
class BlockchainAnchor:
    """Blockchain anchor point for audit logs."""
    anchor_id: str
    merkle_root: str
    transaction_hash: str
    block_number: int
    network: BlockchainNetwork
    timestamp: datetime
    log_count: int
    log_range: Tuple[str, str]  # (first_log_id, last_log_id)
    status: AnchorStatus = AnchorStatus.PENDING
    gas_used: Optional[int] = None
    confirmation_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "merkle_root": self.merkle_root,
            "transaction_hash": self.transaction_hash,
            "block_number": self.block_number,
            "network": self.network.value,
            "timestamp": self.timestamp.isoformat(),
            "log_count": self.log_count,
            "log_range": self.log_range,
            "status": self.status.value,
            "gas_used": self.gas_used,
            "confirmation_count": self.confirmation_count
        }


@dataclass
class VerificationResult:
    """Blockchain verification result."""
    is_valid: bool
    anchor: Optional[BlockchainAnchor]
    merkle_proof: List[str]
    verified_at: datetime
    verification_path: List[Dict[str, Any]]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "anchor": self.anchor.to_dict() if self.anchor else None,
            "merkle_proof": self.merkle_proof,
            "verified_at": self.verified_at.isoformat(),
            "verification_path": self.verification_path,
            "error": self.error
        }


class MerkleTree:
    """Merkle tree implementation for audit log hashing."""
    
    def __init__(self, leaves: List[str]):
        """Initialize Merkle tree with leaf hashes."""
        self.leaves = leaves
        self.tree: List[List[str]] = []
        self._build_tree()
    
    def _hash_pair(self, left: str, right: str) -> str:
        """Hash a pair of nodes."""
        combined = (left + right).encode()
        return hashlib.sha256(combined).hexdigest()
    
    def _build_tree(self):
        """Build the Merkle tree from leaves."""
        if not self.leaves:
            return
        
        # Add leaves as first level
        current_level = self.leaves.copy()
        self.tree.append(current_level)
        
        # Build tree bottom-up
        while len(current_level) > 1:
            next_level = []
            
            # Pad with duplicate if odd number
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])
            
            # Hash pairs
            for i in range(0, len(current_level), 2):
                parent_hash = self._hash_pair(current_level[i], current_level[i + 1])
                next_level.append(parent_hash)
            
            self.tree.append(next_level)
            current_level = next_level
    
    @property
    def root(self) -> Optional[str]:
        """Get the Merkle root."""
        if not self.tree:
            return None
        return self.tree[-1][0] if self.tree[-1] else None
    
    def get_proof(self, leaf_index: int) -> List[Tuple[str, str]]:
        """Get Merkle proof for a leaf.
        
        Returns list of (hash, position) tuples where position is 'left' or 'right'.
        """
        if leaf_index >= len(self.leaves):
            raise ValueError(f"Leaf index {leaf_index} out of range")
        
        proof = []
        index = leaf_index
        
        for level in range(len(self.tree) - 1):
            level_nodes = self.tree[level]
            
            # Find sibling
            if index % 2 == 0:
                # Sibling is on the right
                sibling_index = index + 1
                if sibling_index < len(level_nodes):
                    proof.append((level_nodes[sibling_index], "right"))
            else:
                # Sibling is on the left
                sibling_index = index - 1
                proof.append((level_nodes[sibling_index], "left"))
            
            # Move to parent index
            index = index // 2
        
        return proof
    
    def verify_proof(self, leaf_hash: str, proof: List[Tuple[str, str]], root: str) -> bool:
        """Verify a Merkle proof."""
        current = leaf_hash
        
        for sibling_hash, position in proof:
            if position == "left":
                current = self._hash_pair(sibling_hash, current)
            else:
                current = self._hash_pair(current, sibling_hash)
        
        return current == root


class BlockchainClient(ABC):
    """Abstract base class for blockchain clients."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to blockchain network."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from blockchain network."""
        pass
    
    @abstractmethod
    async def store_merkle_root(
        self,
        merkle_root: str,
        metadata: Dict[str, Any]
    ) -> Tuple[str, int]:
        """Store Merkle root on blockchain.
        
        Returns (transaction_hash, block_number).
        """
        pass
    
    @abstractmethod
    async def verify_merkle_root(
        self,
        transaction_hash: str,
        expected_root: str
    ) -> bool:
        """Verify stored Merkle root."""
        pass
    
    @abstractmethod
    async def get_transaction_status(
        self,
        transaction_hash: str
    ) -> Dict[str, Any]:
        """Get transaction status and confirmations."""
        pass


class EthereumClient(BlockchainClient):
    """Ethereum blockchain client."""
    
    # ABI for simple storage contract
    CONTRACT_ABI = [
        {
            "inputs": [
                {"name": "merkleRoot", "type": "bytes32"},
                {"name": "logCount", "type": "uint256"},
                {"name": "timestamp", "type": "uint256"}
            ],
            "name": "storeAuditAnchor",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [{"name": "transactionId", "type": "bytes32"}],
            "name": "getAuditAnchor",
            "outputs": [
                {"name": "merkleRoot", "type": "bytes32"},
                {"name": "logCount", "type": "uint256"},
                {"name": "timestamp", "type": "uint256"}
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "anonymous": False,
            "inputs": [
                {"indexed": True, "name": "anchorId", "type": "bytes32"},
                {"indexed": False, "name": "merkleRoot", "type": "bytes32"},
                {"indexed": False, "name": "logCount", "type": "uint256"},
                {"indexed": False, "name": "timestamp", "type": "uint256"}
            ],
            "name": "AuditAnchorStored",
            "type": "event"
        }
    ]
    
    def __init__(
        self,
        rpc_url: str,
        private_key: str,
        contract_address: str,
        network: BlockchainNetwork = BlockchainNetwork.ETHEREUM_SEPOLIA,
        gas_limit: int = 100000,
        gas_price_gwei: Optional[int] = None
    ):
        self.rpc_url = rpc_url
        self.private_key = private_key
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.network = network
        self.gas_limit = gas_limit
        self.gas_price_gwei = gas_price_gwei
        
        self.w3: Optional[Web3] = None
        self.contract = None
        self.account = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Ethereum network."""
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            
            if not self.w3.is_connected():
                logger.error("Failed to connect to Ethereum network")
                return False
            
            # Setup account
            self.account = Account.from_key(self.private_key)
            self.w3.middleware_onion.add(
                construct_sign_and_send_raw_middleware(self.account)
            )
            self.w3.eth.default_account = self.account.address
            
            # Setup contract
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.CONTRACT_ABI
            )
            
            self._connected = True
            logger.info(
                f"Connected to {self.network.value} "
                f"(chain_id: {self.w3.eth.chain_id})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Ethereum: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Ethereum network."""
        self._connected = False
        self.w3 = None
        self.contract = None
        logger.info("Disconnected from Ethereum network")
    
    async def store_merkle_root(
        self,
        merkle_root: str,
        metadata: Dict[str, Any]
    ) -> Tuple[str, int]:
        """Store Merkle root on Ethereum."""
        if not self._connected:
            raise ConnectionError("Not connected to Ethereum network")
        
        try:
            # Convert merkle root to bytes32
            merkle_root_bytes = bytes.fromhex(
                merkle_root[2:] if merkle_root.startswith("0x") else merkle_root
            )
            
            log_count = metadata.get("log_count", 0)
            timestamp = int(datetime.now(timezone.utc).timestamp())
            
            # Build transaction
            gas_price = self.w3.eth.gas_price
            if self.gas_price_gwei:
                gas_price = Web3.to_wei(self.gas_price_gwei, 'gwei')
            
            tx = self.contract.functions.storeAuditAnchor(
                merkle_root_bytes,
                log_count,
                timestamp
            ).build_transaction({
                'from': self.account.address,
                'gas': self.gas_limit,
                'gasPrice': gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(
                tx,
                private_key=self.private_key
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            logger.info(
                f"Stored Merkle root on Ethereum: tx={tx_hash.hex()}, "
                f"block={receipt['blockNumber']}, gas={receipt['gasUsed']}"
            )
            
            return tx_hash.hex(), receipt['blockNumber']
            
        except Exception as e:
            logger.error(f"Failed to store Merkle root: {e}")
            raise
    
    async def verify_merkle_root(
        self,
        transaction_hash: str,
        expected_root: str
    ) -> bool:
        """Verify stored Merkle root on Ethereum."""
        if not self._connected:
            raise ConnectionError("Not connected to Ethereum network")
        
        try:
            # Get transaction receipt
            tx_hash = bytes.fromhex(
                transaction_hash[2:] if transaction_hash.startswith("0x") 
                else transaction_hash
            )
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            
            if not receipt:
                logger.warning(f"Transaction {transaction_hash} not found")
                return False
            
            # Parse event logs
            logs = self.contract.events.AuditAnchorStored().process_receipt(receipt)
            
            if not logs:
                logger.warning(f"No AuditAnchorStored events in {transaction_hash}")
                return False
            
            # Compare merkle roots
            stored_root = logs[0]['args']['merkleRoot'].hex()
            expected = expected_root[2:] if expected_root.startswith("0x") else expected_root
            
            return stored_root == expected
            
        except Exception as e:
            logger.error(f"Failed to verify Merkle root: {e}")
            return False
    
    async def get_transaction_status(
        self,
        transaction_hash: str
    ) -> Dict[str, Any]:
        """Get transaction status and confirmations."""
        if not self._connected:
            raise ConnectionError("Not connected to Ethereum network")
        
        try:
            tx_hash = bytes.fromhex(
                transaction_hash[2:] if transaction_hash.startswith("0x") 
                else transaction_hash
            )
            
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            
            if not receipt:
                return {
                    "status": "pending",
                    "confirmations": 0,
                    "block_number": None
                }
            
            current_block = self.w3.eth.block_number
            confirmations = current_block - receipt['blockNumber']
            
            return {
                "status": "confirmed" if receipt['status'] == 1 else "failed",
                "confirmations": confirmations,
                "block_number": receipt['blockNumber'],
                "gas_used": receipt['gasUsed'],
                "effective_gas_price": receipt.get('effectiveGasPrice')
            }
            
        except Exception as e:
            logger.error(f"Failed to get transaction status: {e}")
            raise


class HyperledgerClient(BlockchainClient):
    """Hyperledger Fabric blockchain client."""
    
    def __init__(
        self,
        gateway_url: str,
        channel_name: str,
        chaincode_name: str,
        msp_id: str,
        certificate_path: str,
        private_key_path: str
    ):
        self.gateway_url = gateway_url
        self.channel_name = channel_name
        self.chaincode_name = chaincode_name
        self.msp_id = msp_id
        self.certificate_path = certificate_path
        self.private_key_path = private_key_path
        
        self._connected = False
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self) -> bool:
        """Connect to Hyperledger Fabric network."""
        try:
            self._session = aiohttp.ClientSession()
            
            # Test connection
            async with self._session.get(
                f"{self.gateway_url}/health"
            ) as response:
                if response.status == 200:
                    self._connected = True
                    logger.info(f"Connected to Hyperledger Fabric: {self.channel_name}")
                    return True
            
            logger.error("Hyperledger Fabric health check failed")
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to Hyperledger: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Hyperledger Fabric network."""
        if self._session:
            await self._session.close()
        self._connected = False
        logger.info("Disconnected from Hyperledger Fabric")
    
    async def store_merkle_root(
        self,
        merkle_root: str,
        metadata: Dict[str, Any]
    ) -> Tuple[str, int]:
        """Store Merkle root on Hyperledger Fabric."""
        if not self._connected or not self._session:
            raise ConnectionError("Not connected to Hyperledger Fabric")
        
        try:
            payload = {
                "channel": self.channel_name,
                "chaincode": self.chaincode_name,
                "function": "storeAuditAnchor",
                "args": [
                    merkle_root,
                    json.dumps(metadata)
                ]
            }
            
            async with self._session.post(
                f"{self.gateway_url}/invoke",
                json=payload
            ) as response:
                result = await response.json()
                
                if response.status != 200:
                    raise Exception(f"Hyperledger invoke failed: {result}")
                
                tx_id = result.get("transactionId")
                block_num = result.get("blockNumber", 0)
                
                logger.info(
                    f"Stored Merkle root on Hyperledger: tx={tx_id}, "
                    f"block={block_num}"
                )
                
                return tx_id, block_num
                
        except Exception as e:
            logger.error(f"Failed to store on Hyperledger: {e}")
            raise
    
    async def verify_merkle_root(
        self,
        transaction_hash: str,
        expected_root: str
    ) -> bool:
        """Verify stored Merkle root on Hyperledger Fabric."""
        if not self._connected or not self._session:
            raise ConnectionError("Not connected to Hyperledger Fabric")
        
        try:
            payload = {
                "channel": self.channel_name,
                "chaincode": self.chaincode_name,
                "function": "getAuditAnchor",
                "args": [transaction_hash]
            }
            
            async with self._session.post(
                f"{self.gateway_url}/query",
                json=payload
            ) as response:
                result = await response.json()
                
                if response.status != 200:
                    return False
                
                stored_root = result.get("merkleRoot")
                return stored_root == expected_root
                
        except Exception as e:
            logger.error(f"Failed to verify on Hyperledger: {e}")
            return False
    
    async def get_transaction_status(
        self,
        transaction_hash: str
    ) -> Dict[str, Any]:
        """Get transaction status from Hyperledger Fabric."""
        if not self._connected or not self._session:
            raise ConnectionError("Not connected to Hyperledger Fabric")
        
        try:
            async with self._session.get(
                f"{self.gateway_url}/transaction/{transaction_hash}"
            ) as response:
                result = await response.json()
                
                return {
                    "status": result.get("validationCode", "unknown"),
                    "confirmations": 1,  # Hyperledger is immediate finality
                    "block_number": result.get("blockNumber")
                }
                
        except Exception as e:
            logger.error(f"Failed to get status from Hyperledger: {e}")
            raise


class BlockchainAuditService:
    """Main service for blockchain audit log integration."""
    
    def __init__(
        self,
        db_client,
        blockchain_client: BlockchainClient,
        batch_size: int = 100,
        batch_interval_seconds: int = 300,
        required_confirmations: int = 6
    ):
        """Initialize blockchain audit service.
        
        Args:
            db_client: Database client for audit logs
            blockchain_client: Blockchain client implementation
            batch_size: Number of logs to batch before anchoring
            batch_interval_seconds: Max time between anchors
            required_confirmations: Required block confirmations
        """
        self.db = db_client
        self.blockchain = blockchain_client
        self.batch_size = batch_size
        self.batch_interval = batch_interval_seconds
        self.required_confirmations = required_confirmations
        
        self._pending_logs: List[Dict[str, Any]] = []
        self._last_anchor_time: Optional[datetime] = None
        self._running = False
        self._background_task: Optional[asyncio.Task] = None
    
    async def start(self) -> bool:
        """Start the blockchain audit service."""
        try:
            # Connect to blockchain
            if not await self.blockchain.connect():
                return False
            
            # Start background anchoring task
            self._running = True
            self._background_task = asyncio.create_task(self._anchoring_loop())
            
            logger.info("Blockchain audit service started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start blockchain audit service: {e}")
            return False
    
    async def stop(self):
        """Stop the blockchain audit service."""
        self._running = False
        
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                logger.debug("Background task cancelled")
        
        await self.blockchain.disconnect()
        logger.info("Blockchain audit service stopped")
    
    async def _anchoring_loop(self):
        """Background loop for automatic anchoring."""
        while self._running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                should_anchor = False
                
                # Check batch size
                if len(self._pending_logs) >= self.batch_size:
                    should_anchor = True
                    logger.info(f"Batch size reached ({self.batch_size}), triggering anchor")
                
                # Check time interval
                elif self._last_anchor_time:
                    elapsed = (datetime.now(timezone.utc) - self._last_anchor_time).seconds
                    if elapsed >= self.batch_interval and self._pending_logs:
                        should_anchor = True
                        logger.info(f"Time interval reached ({elapsed}s), triggering anchor")
                
                if should_anchor:
                    await self._create_anchor()
                
                # Check pending anchor confirmations
                await self._check_pending_anchors()
                
            except asyncio.CancelledError:
                logger.debug("Anchoring loop cancelled")
                raise
            except Exception as e:
                logger.error(f"Error in anchoring loop: {e}")
    
    async def add_log_for_anchoring(self, log_entry: Dict[str, Any]):
        """Add an audit log entry for blockchain anchoring.
        
        Args:
            log_entry: Audit log entry with id, signature, timestamp
        """
        self._pending_logs.append(log_entry)
        
        # Check if we need to anchor immediately
        if len(self._pending_logs) >= self.batch_size:
            await self._create_anchor()
    
    async def _create_anchor(self) -> Optional[BlockchainAnchor]:
        """Create a blockchain anchor for pending logs."""
        if not self._pending_logs:
            return None
        
        try:
            logs_to_anchor = self._pending_logs.copy()
            self._pending_logs.clear()
            
            # Build Merkle tree from log signatures
            leaf_hashes = [log["signature"] for log in logs_to_anchor]
            merkle_tree = MerkleTree(leaf_hashes)
            
            if not merkle_tree.root:
                logger.error("Failed to build Merkle tree")
                return None
            
            # Store on blockchain
            metadata = {
                "log_count": len(logs_to_anchor),
                "first_log_id": logs_to_anchor[0]["id"],
                "last_log_id": logs_to_anchor[-1]["id"],
                "first_timestamp": logs_to_anchor[0]["timestamp"],
                "last_timestamp": logs_to_anchor[-1]["timestamp"]
            }
            
            tx_hash, block_number = await self.blockchain.store_merkle_root(
                merkle_tree.root,
                metadata
            )
            
            # Create anchor record
            anchor = BlockchainAnchor(
                anchor_id=hashlib.sha256(tx_hash.encode()).hexdigest()[:16],
                merkle_root=merkle_tree.root,
                transaction_hash=tx_hash,
                block_number=block_number,
                network=getattr(self.blockchain, 'network', BlockchainNetwork.PRIVATE_CHAIN),
                timestamp=datetime.now(timezone.utc),
                log_count=len(logs_to_anchor),
                log_range=(logs_to_anchor[0]["id"], logs_to_anchor[-1]["id"]),
                status=AnchorStatus.PENDING
            )
            
            # Store anchor in database
            await self._store_anchor(anchor, logs_to_anchor, leaf_hashes)
            
            self._last_anchor_time = datetime.now(timezone.utc)
            
            logger.info(
                f"Created blockchain anchor: {anchor.anchor_id} "
                f"({anchor.log_count} logs, merkle_root={anchor.merkle_root[:16]}...)"
            )
            
            return anchor
            
        except Exception as e:
            logger.error(f"Failed to create anchor: {e}")
            # Re-add logs for retry
            self._pending_logs.extend(logs_to_anchor)
            return None
    
    async def _store_anchor(
        self,
        anchor: BlockchainAnchor,
        logs: List[Dict[str, Any]],
        leaf_hashes: List[str]
    ):
        """Store anchor and update log entries in database."""
        try:
            # Store anchor
            await self.db.execute(
                """
                INSERT INTO audits.blockchain_anchors
                (anchor_id, merkle_root, transaction_hash, block_number, network,
                 timestamp, log_count, first_log_id, last_log_id, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                anchor.anchor_id,
                anchor.merkle_root,
                anchor.transaction_hash,
                anchor.block_number,
                anchor.network.value,
                anchor.timestamp,
                anchor.log_count,
                anchor.log_range[0],
                anchor.log_range[1],
                anchor.status.value
            )
            
            # Build Merkle tree for proofs
            merkle_tree = MerkleTree(leaf_hashes)
            
            # Update logs with anchor reference and Merkle proof
            for i, log in enumerate(logs):
                proof = merkle_tree.get_proof(i)
                proof_json = json.dumps([(h, p) for h, p in proof])
                
                await self.db.execute(
                    """
                    UPDATE audits.audit_log
                    SET blockchain_anchor_id = $1, merkle_proof = $2
                    WHERE id = $3
                    """,
                    anchor.anchor_id,
                    proof_json,
                    log["id"]
                )
            
        except Exception as e:
            logger.error(f"Failed to store anchor: {e}")
            raise
    
    async def _check_pending_anchors(self):
        """Check and update status of pending anchors."""
        try:
            pending = await self.db.fetch(
                """
                SELECT * FROM audits.blockchain_anchors
                WHERE status = 'pending'
                """
            )
            
            for anchor_record in pending:
                status = await self.blockchain.get_transaction_status(
                    anchor_record["transaction_hash"]
                )
                
                if status["status"] == "confirmed":
                    confirmations = status.get("confirmations", 0)
                    
                    if confirmations >= self.required_confirmations:
                        await self.db.execute(
                            """
                            UPDATE audits.blockchain_anchors
                            SET status = 'confirmed',
                                confirmation_count = $1,
                                gas_used = $2
                            WHERE anchor_id = $3
                            """,
                            confirmations,
                            status.get("gas_used"),
                            anchor_record["anchor_id"]
                        )
                        logger.info(
                            f"Anchor {anchor_record['anchor_id']} confirmed "
                            f"({confirmations} confirmations)"
                        )
                    else:
                        await self.db.execute(
                            """
                            UPDATE audits.blockchain_anchors
                            SET confirmation_count = $1
                            WHERE anchor_id = $2
                            """,
                            confirmations,
                            anchor_record["anchor_id"]
                        )
                
                elif status["status"] == "failed":
                    await self.db.execute(
                        """
                        UPDATE audits.blockchain_anchors
                        SET status = 'failed'
                        WHERE anchor_id = $1
                        """,
                        anchor_record["anchor_id"]
                    )
                    logger.error(f"Anchor {anchor_record['anchor_id']} failed")
                    
        except Exception as e:
            logger.error(f"Error checking pending anchors: {e}")
    
    async def verify_log(self, log_id: str) -> VerificationResult:
        """Verify a single audit log against blockchain anchor.
        
        Args:
            log_id: ID of the audit log to verify
            
        Returns:
            VerificationResult with verification details
        """
        verification_path = []
        
        try:
            # Get log entry with anchor reference
            log = await self.db.fetchone(
                """
                SELECT al.*, ba.merkle_root, ba.transaction_hash, ba.block_number,
                       ba.network, ba.status as anchor_status, ba.confirmation_count
                FROM audits.audit_log al
                LEFT JOIN audits.blockchain_anchors ba 
                    ON al.blockchain_anchor_id = ba.anchor_id
                WHERE al.id = $1
                """,
                log_id
            )
            
            if not log:
                return VerificationResult(
                    is_valid=False,
                    anchor=None,
                    merkle_proof=[],
                    verified_at=datetime.now(timezone.utc),
                    verification_path=[],
                    error="Log entry not found"
                )
            
            verification_path.append({
                "step": "log_found",
                "log_id": log_id,
                "signature": log["signature"][:16] + "..."
            })
            
            # Check if log has blockchain anchor
            if not log.get("blockchain_anchor_id"):
                return VerificationResult(
                    is_valid=False,
                    anchor=None,
                    merkle_proof=[],
                    verified_at=datetime.now(timezone.utc),
                    verification_path=verification_path,
                    error="Log not yet anchored to blockchain"
                )
            
            # Build anchor object
            anchor = BlockchainAnchor(
                anchor_id=log["blockchain_anchor_id"],
                merkle_root=log["merkle_root"],
                transaction_hash=log["transaction_hash"],
                block_number=log["block_number"],
                network=BlockchainNetwork(log["network"]),
                timestamp=log["ts"],
                log_count=0,
                log_range=("", ""),
                status=AnchorStatus(log["anchor_status"]),
                confirmation_count=log["confirmation_count"]
            )
            
            verification_path.append({
                "step": "anchor_found",
                "anchor_id": anchor.anchor_id,
                "status": anchor.status.value,
                "confirmations": anchor.confirmation_count
            })
            
            # Verify anchor status
            if anchor.status != AnchorStatus.CONFIRMED:
                return VerificationResult(
                    is_valid=False,
                    anchor=anchor,
                    merkle_proof=[],
                    verified_at=datetime.now(timezone.utc),
                    verification_path=verification_path,
                    error=f"Anchor not yet confirmed (status: {anchor.status.value})"
                )
            
            # Get and verify Merkle proof
            merkle_proof = json.loads(log["merkle_proof"]) if log.get("merkle_proof") else []
            
            if not merkle_proof:
                return VerificationResult(
                    is_valid=False,
                    anchor=anchor,
                    merkle_proof=[],
                    verified_at=datetime.now(timezone.utc),
                    verification_path=verification_path,
                    error="Merkle proof not found"
                )
            
            verification_path.append({
                "step": "merkle_proof_found",
                "proof_length": len(merkle_proof)
            })
            
            # Verify local Merkle proof
            leaf_hash = log["signature"]
            merkle_tree = MerkleTree([])  # Empty tree for verification
            
            is_valid_local = merkle_tree.verify_proof(
                leaf_hash,
                merkle_proof,
                anchor.merkle_root
            )
            
            verification_path.append({
                "step": "local_merkle_verification",
                "is_valid": is_valid_local
            })
            
            if not is_valid_local:
                return VerificationResult(
                    is_valid=False,
                    anchor=anchor,
                    merkle_proof=[h for h, _ in merkle_proof],
                    verified_at=datetime.now(timezone.utc),
                    verification_path=verification_path,
                    error="Merkle proof verification failed"
                )
            
            # Verify on blockchain
            is_valid_blockchain = await self.blockchain.verify_merkle_root(
                anchor.transaction_hash,
                anchor.merkle_root
            )
            
            verification_path.append({
                "step": "blockchain_verification",
                "is_valid": is_valid_blockchain
            })
            
            return VerificationResult(
                is_valid=is_valid_blockchain,
                anchor=anchor,
                merkle_proof=[h for h, _ in merkle_proof],
                verified_at=datetime.now(timezone.utc),
                verification_path=verification_path,
                error=None if is_valid_blockchain else "Blockchain verification failed"
            )
            
        except Exception as e:
            logger.error(f"Log verification failed: {e}")
            return VerificationResult(
                is_valid=False,
                anchor=None,
                merkle_proof=[],
                verified_at=datetime.now(timezone.utc),
                verification_path=verification_path,
                error=str(e)
            )
    
    async def verify_range(
        self,
        from_ts: datetime,
        to_ts: datetime
    ) -> Dict[str, Any]:
        """Verify all audit logs in a time range.
        
        Args:
            from_ts: Start timestamp
            to_ts: End timestamp
            
        Returns:
            Verification summary with details
        """
        try:
            # Get anchors in range
            anchors = await self.db.fetch(
                """
                SELECT * FROM audits.blockchain_anchors
                WHERE timestamp >= $1 AND timestamp <= $2
                ORDER BY timestamp ASC
                """,
                from_ts,
                to_ts
            )
            
            results = {
                "total_anchors": len(anchors),
                "verified_anchors": 0,
                "failed_anchors": 0,
                "pending_anchors": 0,
                "total_logs": 0,
                "verified_logs": 0,
                "details": []
            }
            
            for anchor_record in anchors:
                anchor_id = anchor_record["anchor_id"]
                
                # Count logs in anchor
                log_count = await self.db.fetchone(
                    """
                    SELECT COUNT(*) as count FROM audits.audit_log
                    WHERE blockchain_anchor_id = $1
                    """,
                    anchor_id
                )
                
                results["total_logs"] += log_count["count"]
                
                if anchor_record["status"] == "confirmed":
                    # Verify on blockchain
                    is_valid = await self.blockchain.verify_merkle_root(
                        anchor_record["transaction_hash"],
                        anchor_record["merkle_root"]
                    )
                    
                    if is_valid:
                        results["verified_anchors"] += 1
                        results["verified_logs"] += log_count["count"]
                    else:
                        results["failed_anchors"] += 1
                    
                    results["details"].append({
                        "anchor_id": anchor_id,
                        "status": "verified" if is_valid else "failed",
                        "log_count": log_count["count"],
                        "confirmations": anchor_record["confirmation_count"]
                    })
                    
                elif anchor_record["status"] == "pending":
                    results["pending_anchors"] += 1
                    results["details"].append({
                        "anchor_id": anchor_id,
                        "status": "pending",
                        "log_count": log_count["count"]
                    })
                else:
                    results["failed_anchors"] += 1
                    results["details"].append({
                        "anchor_id": anchor_id,
                        "status": "failed",
                        "log_count": log_count["count"]
                    })
            
            results["verification_time"] = datetime.now(timezone.utc).isoformat()
            results["is_valid"] = (
                results["failed_anchors"] == 0 and 
                results["pending_anchors"] == 0
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Range verification failed: {e}")
            raise
    
    async def get_anchor_statistics(self) -> Dict[str, Any]:
        """Get blockchain anchoring statistics."""
        try:
            stats = await self.db.fetchone(
                """
                SELECT 
                    COUNT(*) as total_anchors,
                    COUNT(*) FILTER (WHERE status = 'confirmed') as confirmed,
                    COUNT(*) FILTER (WHERE status = 'pending') as pending,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed,
                    SUM(log_count) as total_logs_anchored,
                    SUM(gas_used) as total_gas_used,
                    AVG(confirmation_count) as avg_confirmations,
                    MIN(timestamp) as first_anchor,
                    MAX(timestamp) as last_anchor
                FROM audits.blockchain_anchors
                """
            )
            
            return {
                "total_anchors": stats["total_anchors"] or 0,
                "confirmed": stats["confirmed"] or 0,
                "pending": stats["pending"] or 0,
                "failed": stats["failed"] or 0,
                "total_logs_anchored": stats["total_logs_anchored"] or 0,
                "total_gas_used": stats["total_gas_used"] or 0,
                "avg_confirmations": float(stats["avg_confirmations"] or 0),
                "first_anchor": stats["first_anchor"].isoformat() if stats["first_anchor"] else None,
                "last_anchor": stats["last_anchor"].isoformat() if stats["last_anchor"] else None,
                "pending_logs": len(self._pending_logs)
            }
            
        except Exception as e:
            logger.error(f"Failed to get anchor statistics: {e}")
            raise
    
    async def force_anchor(self) -> Optional[BlockchainAnchor]:
        """Force immediate anchoring of pending logs.
        
        Returns:
            Created anchor or None if no pending logs
        """
        if not self._pending_logs:
            logger.info("No pending logs to anchor")
            return None
        
        return await self._create_anchor()


# Database migration for blockchain anchors
BLOCKCHAIN_ANCHOR_MIGRATION = """
-- Blockchain anchor storage table
CREATE TABLE IF NOT EXISTS audits.blockchain_anchors (
    anchor_id VARCHAR(64) PRIMARY KEY,
    merkle_root VARCHAR(64) NOT NULL,
    transaction_hash VARCHAR(128) NOT NULL UNIQUE,
    block_number BIGINT NOT NULL,
    network VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    log_count INTEGER NOT NULL,
    first_log_id BIGINT NOT NULL,
    last_log_id BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    gas_used BIGINT,
    confirmation_count INTEGER DEFAULT 0,
    CONSTRAINT valid_status CHECK (status IN ('pending', 'confirmed', 'failed'))
);

-- Add blockchain reference columns to audit_log
ALTER TABLE audits.audit_log 
ADD COLUMN IF NOT EXISTS blockchain_anchor_id VARCHAR(64) REFERENCES audits.blockchain_anchors(anchor_id),
ADD COLUMN IF NOT EXISTS merkle_proof TEXT;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_blockchain_anchors_status 
    ON audits.blockchain_anchors(status);
CREATE INDEX IF NOT EXISTS idx_blockchain_anchors_timestamp 
    ON audits.blockchain_anchors(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_blockchain_anchor 
    ON audits.audit_log(blockchain_anchor_id);
"""
