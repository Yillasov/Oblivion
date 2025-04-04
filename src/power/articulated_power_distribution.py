"""
Efficient power distribution for articulated structures in UCAV platforms.

This module provides specialized power distribution capabilities for
articulated components like control surfaces, landing gear, and weapon bays.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time

from src.power.power_distribution import PowerDistributor, PowerDistributionConfig
from src.power.resource_management import PowerResourceManager
from src.core.utils.logging_framework import get_logger

logger = get_logger("articulated_power")

class ArticulatedPowerDistributor(PowerDistributor):
    """Power distributor specialized for articulated structures."""
    
    def __init__(self, 
                resource_manager: PowerResourceManager,
                config: Optional[PowerDistributionConfig] = None):
        """Initialize articulated power distributor."""
        super().__init__(resource_manager, config)
        
        # Articulation-specific attributes
        self.joint_power_map: Dict[str, Dict[str, float]] = {}
        self.articulation_states: Dict[str, Dict[str, Any]] = {}
        self.power_routing_efficiency: Dict[str, float] = {}
        
        # Default efficiency for power routing through joints
        self.default_joint_efficiency = 0.92  # 92% efficiency
        
    def register_articulated_system(self, 
                                   system_id: str,
                                   joints: List[str],
                                   power_requirements: Dict[str, float],
                                   articulation_range: Dict[str, Tuple[float, float]]) -> bool:
        """
        Register an articulated system for power distribution.
        
        Args:
            system_id: System identifier
            joints: List of joint identifiers
            power_requirements: Power requirements by mode
            articulation_range: Range of motion for each joint
            
        Returns:
            Success status
        """
        # Register with base distributor
        if not super().register_system(system_id, power_requirements):
            return False
        
        # Store joint information
        self.joint_power_map[system_id] = {joint: 0.0 for joint in joints}
        
        # Initialize articulation states
        self.articulation_states[system_id] = {
            "joints": {joint: {"position": 0.0, "velocity": 0.0} for joint in joints},
            "range": articulation_range,
            "last_update": time.time()
        }
        
        # Initialize power routing efficiency for each joint
        self.power_routing_efficiency[system_id] = {
            joint: self.default_joint_efficiency for joint in joints
        }
        
        logger.info(f"Registered articulated system '{system_id}' with {len(joints)} joints")
        return True
    
    def update_articulation_state(self, 
                                 system_id: str, 
                                 joint_states: Dict[str, Dict[str, float]]) -> bool:
        """
        Update the articulation state of a system.
        
        Args:
            system_id: System identifier
            joint_states: Current joint positions and velocities
            
        Returns:
            Success status
        """
        if system_id not in self.articulation_states:
            logger.error(f"Articulated system '{system_id}' not registered")
            return False
        
        # Update joint states
        current_time = time.time()
        time_delta = current_time - self.articulation_states[system_id]["last_update"]
        
        for joint, state in joint_states.items():
            if joint in self.articulation_states[system_id]["joints"]:
                self.articulation_states[system_id]["joints"][joint] = state
                
                # Calculate power routing efficiency based on joint position
                # Joints at extreme positions may have reduced efficiency
                position = state["position"]
                joint_range = self.articulation_states[system_id]["range"].get(
                    joint, (-1.0, 1.0))  # Default range
                
                # Normalize position within range
                range_min, range_max = joint_range
                normalized_pos = (position - range_min) / (range_max - range_min)
                
                # Efficiency curve: highest at center, lower at extremes
                # Simple parabolic model: f(x) = a*(x-0.5)^2 + b
                efficiency = -0.1 * ((normalized_pos - 0.5) ** 2) + self.default_joint_efficiency
                
                # Additional efficiency reduction based on velocity (dynamic losses)
                velocity_factor = 1.0 - min(0.05, abs(state["velocity"]) * 0.01)
                
                self.power_routing_efficiency[system_id][joint] = efficiency * velocity_factor
        
        # Update timestamp
        self.articulation_states[system_id]["last_update"] = current_time
        
        # Trigger redistribution if needed
        if time_delta > 0.1:  # Only redistribute if significant time has passed
            self._redistribute_joint_power(system_id)
            
        return True
    
    def _redistribute_joint_power(self, system_id: str) -> None:
        """
        Redistribute power across joints based on current articulation state.
        
        Args:
            system_id: System identifier
        """
        if system_id not in self.distribution_map:
            return
            
        total_allocation = sum(self.distribution_map[system_id].values())
        joints = self.articulation_states[system_id]["joints"]
        
        # Calculate power needs based on joint velocity
        joint_power_needs = {}
        total_needs = 0.0
        
        for joint, state in joints.items():
            # Power need increases with velocity
            power_need = 1.0 + abs(state["velocity"]) * 2.0
            joint_power_needs[joint] = power_need
            total_needs += power_need
        
        # Distribute power proportionally to needs
        if total_needs > 0:
            for joint, need in joint_power_needs.items():
                # Allocate power proportionally
                allocation = (need / total_needs) * total_allocation
                
                # Apply efficiency loss
                efficiency = self.power_routing_efficiency[system_id][joint]
                effective_power = allocation * efficiency
                
                self.joint_power_map[system_id][joint] = effective_power
    
    def distribute_power(self, flight_conditions: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Distribute power with articulation-aware logic.
        
        Args:
            flight_conditions: Current flight conditions
            
        Returns:
            Power distribution map
        """
        # Get base distribution
        distribution = super().distribute_power(flight_conditions)
        
        # Apply articulation-specific adjustments
        for system_id in self.articulation_states:
            if system_id in distribution:
                self._redistribute_joint_power(system_id)
                
                # Log joint power distribution
                logger.debug(f"Joint power for '{system_id}': {self.joint_power_map[system_id]}")
        
        return distribution
    
    def get_joint_power(self, system_id: str, joint_id: str) -> float:
        """
        Get power allocation for a specific joint.
        
        Args:
            system_id: System identifier
            joint_id: Joint identifier
            
        Returns:
            Power allocation in kW
        """
        if system_id not in self.joint_power_map or joint_id not in self.joint_power_map[system_id]:
            return 0.0
            
        return self.joint_power_map[system_id][joint_id]
    
    def calculate_power_losses(self, system_id: str) -> Dict[str, float]:
        """
        Calculate power losses due to articulation.
        
        Args:
            system_id: System identifier
            
        Returns:
            Power losses by joint
        """
        if system_id not in self.joint_power_map:
            return {}
            
        losses = {}
        for joint, power in self.joint_power_map[system_id].items():
            efficiency = self.power_routing_efficiency[system_id][joint]
            losses[joint] = power * (1.0 - efficiency)
            
        return losses
    
    def _balance_load(self) -> None:
        """Balance load across power resources with articulation awareness."""
        # Identify systems with high articulation activity
        active_articulation = {}
        
        for system_id, states in self.articulation_states.items():
            # Calculate average joint velocity as activity indicator
            avg_velocity = sum(
                abs(state["velocity"]) 
                for state in states["joints"].values()
            ) / max(1, len(states["joints"]))
            
            active_articulation[system_id] = avg_velocity
        
        # Prioritize systems with high articulation activity
        if active_articulation:
            # Sort systems by activity level
            sorted_systems = sorted(
                active_articulation.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Adjust priorities temporarily for high-activity systems
            for system_id, activity in sorted_systems:
                if activity > 0.5 and system_id in self.system_priorities:
                    # Temporarily boost priority for systems with high activity
                    original_priority = self.system_priorities[system_id]
                    # This would need to be implemented in the base class
                    # self._boost_priority(system_id, activity)
                    
                    logger.debug(f"Boosting priority for active articulated system '{system_id}'")