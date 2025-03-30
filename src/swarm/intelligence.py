"""
Swarm Intelligence Algorithms for UCAV coordination.
Provides decentralized control and coordination for multiple UCAVs.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional
import logging

# Fix the import path for NeuromorphicSystem
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.swarm.communication import SwarmCommunicationProtocol, CollectiveDecisionMaking, MessageType

# Replace standard logging with the project's custom logging framework
from src.core.utils.logging_framework import get_logger

# Use the custom logger
logger = get_logger("swarm_intelligence")

class SwarmAgent:
    """Represents a single UCAV agent in a swarm."""
    
    def __init__(self, agent_id: str, position: np.ndarray, 
                 capabilities: Dict[str, Any], neuromorphic_system: Optional[NeuromorphicSystem] = None):
        self.agent_id = agent_id
        self.position = position
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.capabilities = capabilities
        self.neuromorphic_system = neuromorphic_system
        self.neighbors = {}
        self.state = "active"
        
        # Agent parameters
        self.max_speed = capabilities.get("max_speed", 100.0)  # m/s
        self.perception_radius = capabilities.get("perception_radius", 5000.0)  # meters
        
        # Add communication protocol
        comm_range = capabilities.get("communication_range", 8000.0)  # meters
        self.comm_protocol = SwarmCommunicationProtocol(agent_id, comm_range)
        
        # Add collective decision making
        self.decision_making = CollectiveDecisionMaking(agent_id)
        
        # Register message handlers
        self._register_message_handlers()
        
        logger.info(f"Initialized swarm agent {agent_id}")
    
    def _register_message_handlers(self):
        """Register handlers for different message types."""
        self.comm_protocol.register_handler(MessageType.POSITION, self._handle_position_message)
        self.comm_protocol.register_handler(MessageType.THREAT, self._handle_threat_message)
        self.comm_protocol.register_handler(MessageType.DECISION, self._handle_decision_message)
    
    def _handle_position_message(self, message):
        """Handle position update from another agent."""
        # Position updates are handled automatically by the swarm intelligence
        pass
    
    def _handle_threat_message(self, message):
        """Handle threat information from another agent."""
        # Could trigger evasive maneuvers or tactical responses
        threat_pos = np.array(message.content["position"])
        threat_type = message.content["type"]
        
        # Calculate distance to threat
        distance = np.linalg.norm(self.position - threat_pos)
        
        # React based on threat type and distance
        if distance < 2000.0:  # If threat is close
            # Propose evasive action to swarm
            self.decision_making.propose_decision(
                f"evade_{message.msg_id}",
                "evasion",
                {"threat_position": threat_pos.tolist(), "threat_type": threat_type},
                0.9  # High confidence for close threats
            )
    
    def _handle_decision_message(self, message):
        """Handle decision proposals or votes."""
        decision_type = message.content.get("type")
        parameters = message.content.get("parameters", {})
        
        if "vote" in message.content:
            # This is a vote on an existing decision
            decision_id = message.content["decision_id"]
            confidence = message.content["vote"]
            self.decision_making.vote_on_decision(decision_id, message.sender_id, confidence)
        else:
            # This is a new decision proposal
            decision_id = f"{message.sender_id}_{int(message.timestamp)}"
            confidence = message.content.get("confidence", 0.7)
            
            # Vote on the proposed decision
            self.decision_making.vote_on_decision(
                decision_id, message.sender_id, confidence
            )
    
    def update(self, delta_time: float) -> None:
        """Update agent position based on current velocity and acceleration."""
        # Add error handling for position updates
        try:
            self.velocity += self.acceleration * delta_time
            
            # Limit speed to max_speed
            speed = np.linalg.norm(self.velocity)
            if speed > self.max_speed:
                self.velocity = self.velocity * (self.max_speed / speed)
                
            self.position += self.velocity * delta_time
            self.acceleration = np.zeros(3)
            
            # Broadcast position to neighbors
            self.comm_protocol.broadcast_position(self.position, self.velocity)
            
            # Update communication protocol
            self.comm_protocol.update(self.neighbors)
        except Exception as e:
            logger.error(f"Error updating agent {self.agent_id}: {str(e)}")


class SwarmIntelligence:
    """Implements swarm intelligence algorithms for UCAV coordination."""
    
    def __init__(self, config: Dict[str, Any]):
        self.agents: Dict[str, SwarmAgent] = {}
        self.mission_objective = config.get("mission_objective", np.zeros(3))
        
        # Swarm behavior weights
        self.separation_weight = config.get("separation_weight", 1.5)
        self.alignment_weight = config.get("alignment_weight", 1.0)
        self.cohesion_weight = config.get("cohesion_weight", 1.0)
        self.objective_weight = config.get("objective_weight", 2.0)
        
        logger.info("Initialized swarm intelligence")
    
    def add_agent(self, agent: SwarmAgent) -> bool:
        """Add an agent to the swarm."""
        if agent.agent_id in self.agents:
            return False
        self.agents[agent.agent_id] = agent
        return True
    
    def update_swarm(self, delta_time: float, environment_data: Dict[str, Any]) -> None:
        """Update the entire swarm based on swarm intelligence algorithms."""
        # Update neighbor information
        self._update_neighbor_information()
        
        # Apply swarm intelligence algorithms to each agent
        for agent_id, agent in self.agents.items():
            if agent.state != "active":
                continue
                
            # Calculate forces
            separation = self._calculate_separation(agent) * self.separation_weight
            alignment = self._calculate_alignment(agent) * self.alignment_weight
            cohesion = self._calculate_cohesion(agent) * self.cohesion_weight
            objective = self._calculate_objective_force(agent) * self.objective_weight
            
            # Combine forces
            combined_force = separation + alignment + cohesion + objective
            
            # Apply force to agent
            agent.acceleration = combined_force
            
            # Update agent position
            agent.update(delta_time)
    
    def _update_neighbor_information(self) -> None:
        """Update neighbor information for all agents."""
        for agent_id, agent in self.agents.items():
            agent.neighbors = {}
            
            for other_id, other_agent in self.agents.items():
                if other_id == agent_id:
                    continue
                    
                # Calculate distance between agents
                distance = np.linalg.norm(agent.position - other_agent.position)
                
                # Check if within perception radius
                if distance <= agent.perception_radius:
                    agent.neighbors[other_id] = {
                        "agent": other_agent,
                        "distance": distance,
                        "relative_position": other_agent.position - agent.position
                    }
    
    def _calculate_separation(self, agent: SwarmAgent) -> np.ndarray:
        """Calculate separation force to avoid collisions with neighbors."""
        if not agent.neighbors:
            return np.zeros(3)
            
        separation_force = np.zeros(3)
        
        for neighbor_info in agent.neighbors.values():
            # Get relative position and distance
            relative_pos = neighbor_info["relative_position"]
            distance = neighbor_info["distance"]
            
            # Avoid division by zero
            if distance < 0.1:
                distance = 0.1
                
            # Force is inversely proportional to distance
            repulsion = relative_pos / (distance ** 2)
            separation_force -= repulsion
            
        return separation_force
    
    def _calculate_alignment(self, agent: SwarmAgent) -> np.ndarray:
        """Calculate alignment force to align with neighbors' heading."""
        if not agent.neighbors:
            return np.zeros(3)
            
        avg_velocity = np.zeros(3)
        
        for neighbor_info in agent.neighbors.values():
            neighbor = neighbor_info["agent"]
            avg_velocity += neighbor.velocity
            
        avg_velocity /= len(agent.neighbors)
        
        # Calculate steering force towards average velocity
        alignment_force = avg_velocity - agent.velocity
        
        return alignment_force
    
    def _calculate_cohesion(self, agent: SwarmAgent) -> np.ndarray:
        """Calculate cohesion force to move toward center of neighbors."""
        if not agent.neighbors:
            return np.zeros(3)
            
        center_of_mass = np.zeros(3)
        
        for neighbor_info in agent.neighbors.values():
            neighbor = neighbor_info["agent"]
            center_of_mass += neighbor.position
            
        center_of_mass /= len(agent.neighbors)
        
        # Calculate steering force towards center of mass
        desired_velocity = center_of_mass - agent.position
        
        # Scale to max_speed
        if np.linalg.norm(desired_velocity) > 0:
            desired_velocity = desired_velocity / np.linalg.norm(desired_velocity) * agent.max_speed
            
        cohesion_force = desired_velocity - agent.velocity
        
        return cohesion_force
    
    def _calculate_objective_force(self, agent: SwarmAgent) -> np.ndarray:
        """Calculate force to move toward mission objective."""
        # Calculate steering force towards objective
        desired_velocity = self.mission_objective - agent.position
        
        # Scale to max_speed
        if np.linalg.norm(desired_velocity) > 0:
            desired_velocity = desired_velocity / np.linalg.norm(desired_velocity) * agent.max_speed
            
        objective_force = desired_velocity - agent.velocity
        
        return objective_force


# Add factory method to make the classes more accessible
def create_swarm_intelligence(config: Dict[str, Any]) -> SwarmIntelligence:
    """
    Factory method to create a SwarmIntelligence instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SwarmIntelligence: Configured swarm intelligence instance
    """
    return SwarmIntelligence(config)

def create_swarm_agent(agent_id: str, position: np.ndarray, 
                      capabilities: Dict[str, Any], 
                      neuromorphic_system: Optional[NeuromorphicSystem] = None) -> SwarmAgent:
    """
    Factory method to create a SwarmAgent instance.
    
    Args:
        agent_id: Unique agent identifier
        position: Initial position
        capabilities: Agent capabilities
        neuromorphic_system: Optional neuromorphic system
        
    Returns:
        SwarmAgent: Configured swarm agent
    """
    return SwarmAgent(agent_id, position, capabilities, neuromorphic_system)