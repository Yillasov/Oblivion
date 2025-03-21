"""
Swarm Coordinator for UCAV platforms.
Provides centralized management for swarm operations while enabling decentralized execution.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional
import logging
import threading

# Fix the import path for NeuromorphicSystem
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.swarm.intelligence import SwarmAgent, SwarmIntelligence
from src.swarm.communication import MessageType

logger = logging.getLogger(__name__)

class SwarmCoordinator:
    """Coordinates swarm operations for UCAV platforms."""
    
    def __init__(self, config: Dict[str, Any], neuromorphic_system: Optional[NeuromorphicSystem] = None):
        self.config = config
        self.neuromorphic_system = neuromorphic_system
        self.swarm_intelligence = SwarmIntelligence(config)
        
        # Swarm state
        self.mission_state = "initializing"  # initializing, active, completed, aborted
        self.environment_data = {"obstacles": [], "threats": []}
        
        # Coordination
        self.coordination_strategy = config.get("coordination_strategy", "consensus")
        self.role_assignments: Dict[str, str] = {}
        
        # Start coordination thread
        self.active = True
        self.coord_thread = threading.Thread(target=self._coordination_loop)
        self.coord_thread.daemon = True
        self.coord_thread.start()
        
        logger.info(f"Initialized swarm coordinator with {self.coordination_strategy} strategy")
    
    def add_agent(self, agent: SwarmAgent) -> bool:
        """Add an agent to the swarm."""
        # Add to swarm intelligence
        if not self.swarm_intelligence.add_agent(agent):
            return False
            
        # Assign initial role
        self._assign_role(agent.agent_id)
        
        return True
    
    def _assign_role(self, agent_id: str) -> None:
        """Assign a role to an agent."""
        # Simple role assignment based on current roles
        current_roles = list(self.role_assignments.values())
        
        # Count occurrences of each role
        role_counts = {}
        for role in current_roles:
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # Assign role based on what's needed
        if "scout" not in role_counts or role_counts.get("scout", 0) < 2:
            # Need scouts
            self.role_assignments[agent_id] = "scout"
        elif "defender" not in role_counts or role_counts.get("defender", 0) < 2:
            # Need defenders
            self.role_assignments[agent_id] = "defender"
        else:
            # Default to attacker
            self.role_assignments[agent_id] = "attacker"
            
        logger.info(f"Assigned role '{self.role_assignments[agent_id]}' to agent {agent_id}")
    
    def _coordination_loop(self) -> None:
        """Main coordination loop."""
        update_interval = 0.1  # seconds
        
        while self.active:
            if self.mission_state == "active":
                # Update swarm based on current environment
                self.swarm_intelligence.update_swarm(update_interval, self.environment_data)
            
            # Sleep briefly
            time.sleep(update_interval)
    
    def update_environment(self, environment_data: Dict[str, Any]) -> None:
        """Update environment data."""
        self.environment_data = environment_data
        
        # Check for threats and broadcast to swarm
        if "threats" in environment_data:
            for threat in environment_data["threats"]:
                self._broadcast_threat(threat)
    
    def _broadcast_threat(self, threat: Dict[str, Any]) -> None:
        """Broadcast threat information to all agents."""
        threat_position = np.array(threat["position"])
        threat_type = threat.get("type", "unknown")
        threat_level = threat.get("threat_level", 0.5)
        
        # Have each agent broadcast the threat to nearby agents
        for agent_id, agent in self.swarm_intelligence.agents.items():
            # Calculate distance to threat
            distance = np.linalg.norm(agent.position - threat_position)
            
            # Only agents that can detect the threat will broadcast it
            if distance <= agent.perception_radius:
                agent.comm_protocol.broadcast_threat(
                    threat_position, threat_type, threat_level
                )
    
    def get_swarm_decisions(self) -> List[Dict[str, Any]]:
        """Get collective decisions made by the swarm."""
        all_decisions = []
        
        for agent_id, agent in self.swarm_intelligence.agents.items():
            agent_decisions = agent.decision_making.get_active_decisions()
            all_decisions.extend(agent_decisions)
        
        # Remove duplicates based on decision ID
        unique_decisions = {}
        for decision in all_decisions:
            if decision["id"] not in unique_decisions:
                unique_decisions[decision["id"]] = decision
        
        return list(unique_decisions.values())
    
    def start_mission(self) -> bool:
        """Start the swarm mission."""
        if self.mission_state != "initializing":
            logger.warning(f"Cannot start mission in state {self.mission_state}")
            return False
            
        self.mission_state = "active"
        logger.info("Started swarm mission")
        return True
    
    def abort_mission(self) -> bool:
        """Abort the swarm mission."""
        self.mission_state = "aborted"
        logger.info("Aborted swarm mission")
        return True
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current status of the swarm."""
        agent_positions = {agent_id: agent.position.tolist() for agent_id, agent in self.swarm_intelligence.agents.items()}
        
        return {
            "mission_state": self.mission_state,
            "agent_count": len(self.swarm_intelligence.agents),
            "role_assignments": self.role_assignments,
            "agent_positions": agent_positions,
            "coordination_strategy": self.coordination_strategy
        }