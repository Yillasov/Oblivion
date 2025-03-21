"""
Integration of power systems with propulsion and payload subsystems.

This module provides classes and utilities for integrating power management
with propulsion and payload systems in the UCAV platform.
"""

from typing import Dict, List, Any, Optional, Tuple
import time

from src.power.integration import PowerIntegrator
from src.power.resource_management import PowerResourceManager, PowerPriority
from src.power.power_distribution import PowerDistributor, PowerDistributionConfig
from src.propulsion.base import PropulsionInterface
from src.payload.base import PayloadInterface
from src.core.utils.logging_framework import get_logger

logger = get_logger("power_system_integration")


class PowerSystemIntegrator:
    """Integrates power systems with propulsion and payload subsystems."""
    
    def __init__(self, 
                power_integrator: PowerIntegrator,
                resource_manager: PowerResourceManager,
                power_distributor: Optional[PowerDistributor] = None):
        """
        Initialize the power system integrator.
        
        Args:
            power_integrator: Power integrator
            resource_manager: Power resource manager
            power_distributor: Power distributor
        """
        self.power_integrator = power_integrator
        self.resource_manager = resource_manager
        self.power_distributor = power_distributor or PowerDistributor(
            resource_manager, PowerDistributionConfig()
        )
        
        # Track integrated systems
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        self.payload_systems: Dict[str, PayloadInterface] = {}
        self.power_requirements: Dict[str, Dict[str, float]] = {}
        self.system_priorities: Dict[str, PowerPriority] = {}
        
        logger.info("Power system integrator initialized")
    
    def integrate_propulsion_system(self, 
                                  system_id: str, 
                                  propulsion_system: PropulsionInterface,
                                  power_requirements: Dict[str, float],
                                  priority: PowerPriority = PowerPriority.HIGH) -> bool:
        """
        Integrate a propulsion system with power management.
        
        Args:
            system_id: System identifier
            propulsion_system: Propulsion system
            power_requirements: Power requirements by resource ID
            priority: System priority
            
        Returns:
            Success status
        """
        if system_id in self.propulsion_systems:
            logger.warning(f"Propulsion system '{system_id}' already integrated")
            return False
        
        try:
            # Register with power distributor
            if not self.power_distributor.register_system(system_id, power_requirements, priority):
                logger.error(f"Failed to register propulsion system '{system_id}' with power distributor")
                return False
            
            # Store system
            self.propulsion_systems[system_id] = propulsion_system
            self.power_requirements[system_id] = power_requirements
            self.system_priorities[system_id] = priority
            
            # Connect to power integrator
            for resource_id in power_requirements:
                self.power_integrator.connect_to_system(resource_id, f"propulsion_{system_id}")
            
            logger.info(f"Integrated propulsion system '{system_id}' with power management")
            return True
            
        except Exception as e:
            logger.error(f"Error integrating propulsion system: {str(e)}")
            return False
    
    def integrate_payload_system(self, 
                               system_id: str, 
                               payload_system: PayloadInterface,
                               power_requirements: Dict[str, float],
                               priority: PowerPriority = PowerPriority.MEDIUM) -> bool:
        """
        Integrate a payload system with power management.
        
        Args:
            system_id: System identifier
            payload_system: Payload system
            power_requirements: Power requirements by resource ID
            priority: System priority
            
        Returns:
            Success status
        """
        if system_id in self.payload_systems:
            logger.warning(f"Payload system '{system_id}' already integrated")
            return False
        
        try:
            # Register with power distributor
            if not self.power_distributor.register_system(system_id, power_requirements, priority):
                logger.error(f"Failed to register payload system '{system_id}' with power distributor")
                return False
            
            # Store system
            self.payload_systems[system_id] = payload_system
            self.power_requirements[system_id] = power_requirements
            self.system_priorities[system_id] = priority
            
            # Connect to power integrator
            for resource_id in power_requirements:
                self.power_integrator.connect_to_system(resource_id, f"payload_{system_id}")
            
            logger.info(f"Integrated payload system '{system_id}' with power management")
            return True
            
        except Exception as e:
            logger.error(f"Error integrating payload system: {str(e)}")
            return False
    
    def update_power_distribution(self, flight_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
        """
        Update power distribution based on current conditions.
        
        Args:
            flight_conditions: Current flight conditions
            
        Returns:
            Power distribution map
        """
        # Distribute power
        distribution = self.power_distributor.distribute_power(flight_conditions)
        
        # Apply distribution to systems
        self._apply_power_distribution(distribution)
        
        return distribution
    
    def _apply_power_distribution(self, distribution: Dict[str, Dict[str, float]]) -> None:
        """
        Apply power distribution to systems.
        
        Args:
            distribution: Power distribution map
        """
        # Apply to propulsion systems
        for system_id, propulsion in self.propulsion_systems.items():
            if system_id in distribution:
                total_power = sum(distribution[system_id].values())
                max_power = sum(self.power_requirements[system_id].values())
                power_level = (total_power / max_power * 100) if max_power > 0 else 0
                
                # Set power level on propulsion system
                try:
                    # Check which method the propulsion system implements
                    if hasattr(propulsion, 'set_power_level'):
                        propulsion.set_power_level(min(power_level, 100.0))
                    elif hasattr(propulsion, 'set_power_state'):
                        propulsion.set_power_state({"power_level": min(power_level, 100.0)})
                    else:
                        logger.warning(f"Propulsion system '{system_id}' has no power control method")
                except Exception as e:
                    logger.warning(f"Failed to set power level on propulsion system '{system_id}': {str(e)}")
        
        # Apply to payload systems
        for system_id, payload in self.payload_systems.items():
            if system_id in distribution:
                total_power = sum(distribution[system_id].values())
                max_power = sum(self.power_requirements[system_id].values())
                power_level = (total_power / max_power * 100) if max_power > 0 else 0
                
                # Set power level on payload system
                try:
                    # Payload systems might have different methods for power control
                    if hasattr(payload, 'set_power_level'):
                        payload.set_power_level(min(power_level, 100.0))
                    elif hasattr(payload, 'set_power'):
                        payload.set_power(min(power_level / 100.0, 1.0))  # Normalized to 0-1
                except Exception as e:
                    logger.warning(f"Failed to set power level on payload system '{system_id}': {str(e)}")
    
    def get_system_power_status(self) -> Dict[str, Any]:
        """
        Get power status for all integrated systems.
        
        Returns:
            Power status information
        """
        status = {
            "propulsion": {},
            "payload": {},
            "total_allocated": 0.0,
            "timestamp": time.time()
        }
        
        # Get distribution from distributor
        for system_id in self.propulsion_systems:
            if system_id in self.power_distributor.distribution_map:
                allocation = self.power_distributor.get_system_power(system_id)
                status["propulsion"][system_id] = allocation
                status["total_allocated"] += allocation["total_allocation"]
        
        for system_id in self.payload_systems:
            if system_id in self.power_distributor.distribution_map:
                allocation = self.power_distributor.get_system_power(system_id)
                status["payload"][system_id] = allocation
                status["total_allocated"] += allocation["total_allocation"]
        
        return status
    
    def handle_power_emergency(self, emergency_type: str) -> Dict[str, Any]:
        """
        Handle power emergency situations.
        
        Args:
            emergency_type: Type of emergency
            
        Returns:
            Emergency response information
        """
        response = {
            "emergency_type": emergency_type,
            "actions_taken": [],
            "systems_affected": [],
            "timestamp": time.time()
        }
        
        try:
            if emergency_type == "power_loss":
                # Prioritize critical systems
                self.power_distributor.resource_manager.distribution_strategy = "priority_based"
                
                # Update distribution
                distribution = self.power_distributor.distribute_power({"emergency": True})
                self._apply_power_distribution(distribution)
                
                # Identify affected systems
                for system_id, allocations in distribution.items():
                    if sum(allocations.values()) < sum(self.power_requirements.get(system_id, {}).values()):
                        response["systems_affected"].append(system_id)
                
                response["actions_taken"].append("switched_to_priority_distribution")
                response["actions_taken"].append("reduced_non_critical_power")
                
            elif emergency_type == "overload":
                # Switch to efficiency-based distribution
                self.power_distributor.resource_manager.distribution_strategy = "efficiency_based"
                
                # Optimize power usage
                optimization_result = self.power_distributor.optimize_distribution()
                
                # Update distribution
                distribution = self.power_distributor.distribute_power({"emergency": True})
                self._apply_power_distribution(distribution)
                
                response["actions_taken"].append("switched_to_efficiency_distribution")
                response["actions_taken"].append("optimized_power_usage")
                
            else:
                logger.warning(f"Unknown emergency type: {emergency_type}")
                response["actions_taken"].append("no_action_taken")
            
            logger.info(f"Handled power emergency '{emergency_type}'")
            return response
            
        except Exception as e:
            logger.error(f"Error handling power emergency: {str(e)}")
            response["actions_taken"].append("error_during_handling")
            return response