"""
Communication protocols for UCAV swarm coordination.
Enables information sharing and collective decision making.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from enum import Enum

from src.core.integration.neuromorphic_system import NeuromorphicSystem

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages exchanged between swarm agents."""
    POSITION = "position"
    THREAT = "threat"
    OBJECTIVE = "objective"
    DECISION = "decision"
    STATUS = "status"
    COMMAND = "command"

class SwarmMessage:
    """Message exchanged between swarm agents."""
    
    def __init__(self, 
                 sender_id: str, 
                 msg_type: MessageType, 
                 content: Dict[str, Any],
                 priority: int = 1,
                 timestamp: Optional[float] = None):
        self.sender_id = sender_id
        self.msg_type = msg_type
        self.content = content
        self.priority = priority  # 1-5, with 5 being highest
        self.timestamp = timestamp or time.time()
        self.msg_id = f"{sender_id}_{self.timestamp}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for transmission."""
        return {
            "sender_id": self.sender_id,
            "msg_type": self.msg_type.value,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "msg_id": self.msg_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SwarmMessage':
        """Create message from dictionary."""
        return cls(
            sender_id=data["sender_id"],
            msg_type=MessageType(data["msg_type"]),
            content=data["content"],
            priority=data["priority"],
            timestamp=data["timestamp"]
        )

class SwarmCommunicationProtocol:
    """Protocol for communication between swarm agents."""
    
    def __init__(self, agent_id: str, max_range: float = 10000.0):
        self.agent_id = agent_id
        self.max_range = max_range
        self.message_queue: List[SwarmMessage] = []
        self.received_messages: Dict[str, SwarmMessage] = {}
        self.message_handlers: Dict[MessageType, Any] = {}
        self.last_broadcast_time = 0.0
        self.broadcast_interval = 0.2  # seconds
    
    def register_handler(self, msg_type: MessageType, handler_func: Any) -> None:
        """Register a handler function for a specific message type."""
        self.message_handlers[msg_type] = handler_func
    
    def create_message(self, msg_type: MessageType, content: Dict[str, Any], priority: int = 1) -> SwarmMessage:
        """Create a new message to be sent."""
        return SwarmMessage(self.agent_id, msg_type, content, priority)
    
    def queue_message(self, message: SwarmMessage) -> None:
        """Add message to outgoing queue."""
        self.message_queue.append(message)
    
    def broadcast_position(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Broadcast agent position to nearby agents."""
        msg = self.create_message(
            MessageType.POSITION,
            {
                "position": position.tolist(),
                "velocity": velocity.tolist()
            }
        )
        self.queue_message(msg)
    
    def broadcast_threat(self, threat_position: np.ndarray, threat_type: str, confidence: float) -> None:
        """Broadcast threat information to nearby agents."""
        msg = self.create_message(
            MessageType.THREAT,
            {
                "position": threat_position.tolist(),
                "type": threat_type,
                "confidence": confidence
            },
            priority=4  # High priority for threats
        )
        self.queue_message(msg)
    
    def broadcast_decision(self, decision_type: str, parameters: Dict[str, Any]) -> None:
        """Broadcast a decision to nearby agents."""
        msg = self.create_message(
            MessageType.DECISION,
            {
                "type": decision_type,
                "parameters": parameters
            },
            priority=3
        )
        self.queue_message(msg)
    
    def receive_message(self, message_data: Dict[str, Any], sender_distance: float) -> bool:
        """Process an incoming message from another agent."""
        # Check if sender is within range
        if sender_distance > self.max_range:
            return False
            
        # Convert to message object
        message = SwarmMessage.from_dict(message_data)
        
        # Store message
        self.received_messages[message.msg_id] = message
        
        # Handle message if handler exists
        if message.msg_type in self.message_handlers:
            try:
                self.message_handlers[message.msg_type](message)
            except Exception as e:
                logger.error(f"Error handling message: {str(e)}")
                return False
                
        return True
    
    def update(self, nearby_agents: Dict[str, Dict[str, Any]]) -> None:
        """Update communication, sending queued messages to nearby agents."""
        current_time = time.time()
        
        # Only broadcast at specified intervals
        if current_time - self.last_broadcast_time < self.broadcast_interval:
            return
            
        # Send queued messages to nearby agents
        for msg in sorted(self.message_queue, key=lambda m: m.priority, reverse=True):
            msg_data = msg.to_dict()
            
            for agent_id, agent_info in nearby_agents.items():
                # In a real system, this would use the actual communication system
                # For simulation, we directly pass the message
                if hasattr(agent_info["agent"], "comm_protocol"):
                    agent_info["agent"].comm_protocol.receive_message(
                        msg_data, agent_info["distance"]
                    )
            
        # Clear queue after broadcasting
        self.message_queue = []
        self.last_broadcast_time = current_time

class CollectiveDecisionMaking:
    """Implements collective decision making for swarm agents."""
    
    def __init__(self, agent_id: str, decision_threshold: float = 0.7):
        self.agent_id = agent_id
        self.decision_threshold = decision_threshold
        self.decision_votes: Dict[str, Dict[str, Any]] = {}
        self.local_decisions: Dict[str, Dict[str, Any]] = {}
        self.decision_history: List[Dict[str, Any]] = []
    
    def propose_decision(self, decision_id: str, decision_type: str, 
                        parameters: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Propose a new decision to the swarm."""
        decision = {
            "id": decision_id,
            "type": decision_type,
            "parameters": parameters,
            "confidence": confidence,
            "votes": {self.agent_id: confidence},
            "timestamp": time.time()
        }
        
        self.decision_votes[decision_id] = decision
        return decision
    
    def vote_on_decision(self, decision_id: str, agent_id: str, 
                        confidence: float) -> Optional[Dict[str, Any]]:
        """Vote on an existing decision proposal."""
        if decision_id not in self.decision_votes:
            return None
            
        # Add vote
        self.decision_votes[decision_id]["votes"][agent_id] = confidence
        
        # Check if decision has reached threshold
        decision = self.decision_votes[decision_id]
        vote_count = len(decision["votes"])
        avg_confidence = sum(decision["votes"].values()) / vote_count
        
        if vote_count >= 3 and avg_confidence >= self.decision_threshold:
            # Decision reached consensus
            decision["status"] = "accepted"
            decision["final_confidence"] = avg_confidence
            self.local_decisions[decision_id] = decision
            self.decision_history.append(decision)
            return decision
            
        return self.decision_votes[decision_id]
    
    def get_active_decisions(self) -> List[Dict[str, Any]]:
        """Get list of active decisions that have reached consensus."""
        return list(self.local_decisions.values())