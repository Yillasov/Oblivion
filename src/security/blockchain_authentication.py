"""
Blockchain Mission Authentication System.
Ensures secure and tamper-proof mission data for UCAV operations.
"""

import hashlib
import time
import json
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import uuid
import hmac
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Block:
    """Represents a block in the mission authentication blockchain."""
    index: int
    timestamp: float
    mission_data: Dict[str, Any]
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate the hash of this block."""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "mission_data": self.mission_data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()


class MissionBlockchain:
    """Blockchain implementation for mission authentication."""
    
    def __init__(self, difficulty: int = 2):
        """Initialize the blockchain with a genesis block."""
        self.chain: List[Block] = []
        self.difficulty = difficulty
        self.pending_missions: List[Dict[str, Any]] = []
        self.create_genesis_block()
        self.node_id = str(uuid.uuid4()).replace('-', '')
        # Generate a secure key for HMAC
        self.secret_key = os.urandom(32)
        
    def create_genesis_block(self) -> None:
        """Create the genesis block of the blockchain."""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            mission_data={"message": "Genesis Block", "mission_id": "genesis"},
            previous_hash="0"
        )
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
        logger.info("Genesis block created")
    
    def get_latest_block(self) -> Block:
        """Get the latest block in the blockchain."""
        return self.chain[-1]
    
    def add_mission(self, mission_data: Dict[str, Any]) -> None:
        """Add a new mission to pending missions."""
        # Add timestamp and unique ID if not present
        if "timestamp" not in mission_data:
            mission_data["timestamp"] = time.time()
        if "mission_id" not in mission_data:
            mission_data["mission_id"] = str(uuid.uuid4())
            
        self.pending_missions.append(mission_data)
        logger.info(f"Mission {mission_data['mission_id']} added to pending missions")
    
    def mine_pending_missions(self) -> Optional[Block]:
        """Mine pending missions into a new block."""
        if not self.pending_missions:
            logger.warning("No pending missions to mine")
            return None
            
        last_block = self.get_latest_block()
        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            mission_data=self.pending_missions[0],  # Mine one mission at a time
            previous_hash=last_block.hash
        )
        
        new_block.hash = self.proof_of_work(new_block)
        self.chain.append(new_block)
        self.pending_missions.pop(0)
        
        logger.info(f"Block #{new_block.index} mined with hash {new_block.hash}")
        return new_block
    
    def proof_of_work(self, block: Block) -> str:
        """
        Perform proof of work to find a hash with the required difficulty.
        """
        block.nonce = 0
        computed_hash = block.calculate_hash()
        
        while not computed_hash.startswith('0' * self.difficulty):
            block.nonce += 1
            computed_hash = block.calculate_hash()
            
        return computed_hash
    
    def is_chain_valid(self) -> bool:
        """Validate the integrity of the blockchain."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check if the current block's hash is valid
            if current_block.hash != current_block.calculate_hash():
                logger.error(f"Block #{current_block.index} has invalid hash")
                return False
                
            # Check if the previous hash reference is correct
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Block #{current_block.index} has invalid previous hash reference")
                return False
                
        return True
    
    def get_mission_history(self, mission_id: str) -> List[Dict[str, Any]]:
        """Get the history of a specific mission from the blockchain."""
        mission_history = []
        
        for block in self.chain:
            if block.index == 0:  # Skip genesis block
                continue
                
            if block.mission_data.get("mission_id") == mission_id:
                mission_history.append({
                    "block_index": block.index,
                    "timestamp": block.timestamp,
                    "mission_data": block.mission_data,
                    "hash": block.hash
                })
                
        return mission_history
    
    def create_authentication_token(self, mission_id: str) -> str:
        """Create an HMAC authentication token for a mission."""
        timestamp = str(int(time.time()))
        message = f"{mission_id}:{timestamp}".encode()
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        
        return f"{mission_id}:{timestamp}:{signature}"
    
    def verify_authentication_token(self, token: str) -> bool:
        """Verify an authentication token."""
        try:
            mission_id, timestamp, signature = token.split(":")
            message = f"{mission_id}:{timestamp}".encode()
            expected_signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
            
            # Check if the signature matches
            if not hmac.compare_digest(signature, expected_signature):
                logger.warning(f"Invalid signature for mission {mission_id}")
                return False
                
            # Check if the token is not expired (valid for 1 hour)
            current_time = int(time.time())
            token_time = int(timestamp)
            if current_time - token_time > 3600:
                logger.warning(f"Expired token for mission {mission_id}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            return False


class MissionAuthenticator:
    """Interface for mission authentication using blockchain."""
    
    def __init__(self, difficulty: int = 2):
        """Initialize the mission authenticator."""
        self.blockchain = MissionBlockchain(difficulty)
        logger.info("Mission authenticator initialized")
    
    def register_mission(self, mission_data: Dict[str, Any]) -> str:
        """
        Register a new mission in the blockchain.
        
        Args:
            mission_data: Mission data to register
            
        Returns:
            str: Mission ID
        """
        # Ensure mission has an ID
        if "mission_id" not in mission_data:
            mission_data["mission_id"] = str(uuid.uuid4())
            
        # Add mission to blockchain
        self.blockchain.add_mission(mission_data)
        
        # Mine the block to add it to the chain
        self.blockchain.mine_pending_missions()
        
        # Create authentication token
        token = self.blockchain.create_authentication_token(mission_data["mission_id"])
        
        logger.info(f"Mission {mission_data['mission_id']} registered successfully")
        return mission_data["mission_id"]
    
    def verify_mission(self, mission_id: str, token: str) -> bool:
        """
        Verify a mission's authenticity.
        
        Args:
            mission_id: Mission ID to verify
            token: Authentication token
            
        Returns:
            bool: True if mission is authentic, False otherwise
        """
        # Verify token
        if not self.blockchain.verify_authentication_token(token):
            return False
            
        # Check if mission exists in blockchain
        mission_history = self.blockchain.get_mission_history(mission_id)
        if not mission_history:
            logger.warning(f"Mission {mission_id} not found in blockchain")
            return False
            
        # Verify blockchain integrity
        if not self.blockchain.is_chain_valid():
            logger.error("Blockchain integrity check failed")
            return False
            
        logger.info(f"Mission {mission_id} verified successfully")
        return True
    
    def get_mission_audit_trail(self, mission_id: str) -> List[Dict[str, Any]]:
        """
        Get the audit trail for a mission.
        
        Args:
            mission_id: Mission ID to get audit trail for
            
        Returns:
            List[Dict[str, Any]]: Audit trail entries
        """
        mission_history = self.blockchain.get_mission_history(mission_id)
        
        # Format the audit trail
        audit_trail = []
        for entry in mission_history:
            audit_trail.append({
                "timestamp": datetime.fromtimestamp(entry["timestamp"]).isoformat(),
                "block_index": entry["block_index"],
                "mission_data": entry["mission_data"],
                "hash": entry["hash"]
            })
            
        return audit_trail


# Example usage
if __name__ == "__main__":
    # Create mission authenticator
    authenticator = MissionAuthenticator(difficulty=2)
    
    # Register a mission
    mission_data = {
        "name": "Reconnaissance Alpha",
        "type": "surveillance",
        "waypoints": [[0, 0, 100], [10, 20, 100], [30, 40, 100]],
        "authorized_by": "Command Center",
        "priority": "high"
    }
    
    mission_id = authenticator.register_mission(mission_data)
    
    # Create authentication token
    token = authenticator.blockchain.create_authentication_token(mission_id)
    
    # Verify mission
    is_authentic = authenticator.verify_mission(mission_id, token)
    
    # Get audit trail
    audit_trail = authenticator.get_mission_audit_trail(mission_id)
    
    logger.info(f"Mission authenticated: {is_authentic}")
    logger.info(f"Audit trail: {json.dumps(audit_trail, indent=2)}")