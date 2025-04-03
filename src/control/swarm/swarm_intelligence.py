#!/usr/bin/env python3
"""
Simple Swarm Intelligence Framework

Implements basic swarm behaviors and collective intelligence algorithms
inspired by natural systems like bird flocks, fish schools, and insect colonies.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import time

from src.core.utils.logging_framework import get_logger

logger = get_logger("swarm_intelligence")


@dataclass
class SwarmAgent:
    """Individual agent in a swarm."""
    
    id: int                      # Unique identifier
    position: np.ndarray         # Position vector [x, y, z]
    velocity: np.ndarray         # Velocity vector [vx, vy, vz]
    max_speed: float = 10.0      # Maximum speed
    perception_radius: float = 50.0  # How far the agent can perceive
    
    # Agent state and capabilities
    energy: float = 100.0        # Energy level
    role: str = "worker"         # Role in swarm
    data: Dict = field(default_factory=dict)  # Custom data storage
    
    def __post_init__(self):
        """Initialize after creation."""
        if self.data is None:
            self.data = {}


class SwarmBehavior:
    """Base class for swarm behaviors."""
    
    def __init__(self, weight: float = 1.0):
        """Initialize behavior."""
        self.weight = weight
    
    def compute(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> np.ndarray:
        """
        Compute behavior vector.
        
        Args:
            agent: The agent to compute behavior for
            neighbors: List of neighboring agents
            
        Returns:
            np.ndarray: Behavior force vector
        """
        return np.zeros(3)


class SeparationBehavior(SwarmBehavior):
    """Separation behavior - avoid crowding neighbors."""
    
    def __init__(self, min_distance: float = 5.0, weight: float = 1.0):
        """Initialize separation behavior."""
        super().__init__(weight)
        self.min_distance = min_distance
    
    def compute(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> np.ndarray:
        """Compute separation force."""
        if not neighbors:
            return np.zeros(3)
        
        separation_force = np.zeros(3)
        
        for neighbor in neighbors:
            # Vector from neighbor to agent
            offset = agent.position - neighbor.position
            distance = np.linalg.norm(offset)
            
            # Apply separation only if too close
            if 0 < distance < self.min_distance:
                # Weighted by closeness (closer = stronger)
                repulsion = offset / (distance * distance)
                separation_force += repulsion
        
        # Normalize if non-zero
        if np.linalg.norm(separation_force) > 0:
            separation_force = separation_force / np.linalg.norm(separation_force)
            
        return separation_force * self.weight


class AlignmentBehavior(SwarmBehavior):
    """Alignment behavior - steer towards average heading of neighbors."""
    
    def compute(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> np.ndarray:
        """Compute alignment force."""
        if not neighbors:
            return np.zeros(3)
        
        # Calculate average velocity of neighbors
        avg_velocity = np.zeros(3)
        for neighbor in neighbors:
            avg_velocity += neighbor.velocity
        
        avg_velocity /= len(neighbors)
        
        # Normalize if non-zero
        if np.linalg.norm(avg_velocity) > 0:
            avg_velocity = avg_velocity / np.linalg.norm(avg_velocity)
            
        return avg_velocity * self.weight


class CohesionBehavior(SwarmBehavior):
    """Cohesion behavior - steer towards center of neighbors."""
    
    def compute(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> np.ndarray:
        """Compute cohesion force."""
        if not neighbors:
            return np.zeros(3)
        
        # Calculate center of mass
        center = np.zeros(3)
        for neighbor in neighbors:
            center += neighbor.position
        
        center /= len(neighbors)
        
        # Direction to center
        to_center = center - agent.position
        
        # Normalize if non-zero
        if np.linalg.norm(to_center) > 0:
            to_center = to_center / np.linalg.norm(to_center)
            
        return to_center * self.weight


class GoalSeekBehavior(SwarmBehavior):
    """Goal-seeking behavior - move toward a target."""
    
    def __init__(self, goal_position: np.ndarray, weight: float = 1.0):
        """Initialize goal-seeking behavior."""
        super().__init__(weight)
        self.goal_position = goal_position
    
    def compute(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> np.ndarray:
        """Compute goal-seeking force."""
        # Direction to goal
        to_goal = self.goal_position - agent.position
        
        # Normalize if non-zero
        if np.linalg.norm(to_goal) > 0:
            to_goal = to_goal / np.linalg.norm(to_goal)
            
        return to_goal * self.weight


class SwarmIntelligence:
    """Simple swarm intelligence framework."""
    
    def __init__(self):
        """Initialize swarm intelligence system."""
        self.agents: List[SwarmAgent] = []
        self.behaviors: List[SwarmBehavior] = []
        self.next_id = 0
        
        logger.info("Initialized swarm intelligence framework")
    
    def add_agent(self, position: List[float], velocity: List[float], 
                role: str = "worker") -> SwarmAgent:
        """Add agent to swarm."""
        agent = SwarmAgent(
            id=self.next_id,
            position=np.array(position),
            velocity=np.array(velocity),
            role=role
        )
        self.agents.append(agent)
        self.next_id += 1
        
        logger.debug(f"Added agent {agent.id} at position {position}")
        return agent
    
    def add_behavior(self, behavior: SwarmBehavior) -> None:
        """Add behavior to swarm."""
        self.behaviors.append(behavior)
        
        behavior_name = behavior.__class__.__name__
        logger.debug(f"Added behavior {behavior_name} with weight {behavior.weight}")
    
    def get_neighbors(self, agent: SwarmAgent) -> List[SwarmAgent]:
        """Get neighbors within perception radius."""
        neighbors = []
        
        for other in self.agents:
            if other.id == agent.id:
                continue
                
            distance = np.linalg.norm(agent.position - other.position)
            if distance <= agent.perception_radius:
                neighbors.append(other)
        
        return neighbors
    
    def update(self, dt: float) -> None:
        """Update all agents in the swarm."""
        for agent in self.agents:
            # Get neighbors
            neighbors = self.get_neighbors(agent)
            
            # Calculate combined force from all behaviors
            combined_force = np.zeros(3)
            for behavior in self.behaviors:
                force = behavior.compute(agent, neighbors)
                combined_force += force
            
            # Normalize if too strong
            force_magnitude = np.linalg.norm(combined_force)
            if force_magnitude > 1.0:
                combined_force = combined_force / force_magnitude
            
            # Update velocity
            agent.velocity += combined_force * dt
            
            # Limit speed
            speed = np.linalg.norm(agent.velocity)
            if speed > agent.max_speed:
                agent.velocity = (agent.velocity / speed) * agent.max_speed
            
            # Update position
            agent.position += agent.velocity * dt
        
        logger.debug(f"Updated {len(self.agents)} agents")
    
    def create_formation(self, formation_type: str, center: List[float], 
                       num_agents: int, spacing: float = 10.0) -> None:
        """Create agents in a specific formation."""
        center_pos = np.array(center)
        
        if formation_type == "line":
            for i in range(num_agents):
                pos = center_pos + np.array([i * spacing - (num_agents-1) * spacing/2, 0, 0])
                self.add_agent(pos.tolist(), [0, 0, 0])
                
        elif formation_type == "circle":
            for i in range(num_agents):
                angle = 2 * np.pi * i / num_agents
                pos = center_pos + np.array([np.cos(angle) * spacing, np.sin(angle) * spacing, 0])
                # Velocity tangent to circle
                vel = np.array([-np.sin(angle), np.cos(angle), 0]) * 2.0
                self.add_agent(pos.tolist(), vel.tolist())
                
        elif formation_type == "grid":
            side = int(np.ceil(np.sqrt(num_agents)))
            for i in range(min(side * side, num_agents)):
                row, col = i // side, i % side
                pos = center_pos + np.array(
                    [(col - side/2) * spacing, (row - side/2) * spacing, 0])
                self.add_agent(pos.tolist(), [0, 0, 0])
        
        logger.info(f"Created {formation_type} formation with {num_agents} agents")


# Example usage
def create_example_swarm() -> SwarmIntelligence:
    """Create an example swarm with basic behaviors."""
    swarm = SwarmIntelligence()
    
    # Create a circular formation
    swarm.create_formation("circle", [0, 0, 0], 10, spacing=20.0)
    
    # Add behaviors
    swarm.add_behavior(SeparationBehavior(min_distance=15.0, weight=1.5))
    swarm.add_behavior(AlignmentBehavior(weight=1.0))
    swarm.add_behavior(CohesionBehavior(weight=1.0))
    swarm.add_behavior(GoalSeekBehavior(np.array([100, 100, 0]), weight=0.5))
    
    return swarm