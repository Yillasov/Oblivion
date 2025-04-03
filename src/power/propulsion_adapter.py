"""
Adapter for propulsion-power integration.

This module provides adapters for integrating propulsion systems with power management.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional
import time

from src.power.system_integration import PowerSystemIntegrator
from src.propulsion.base import PropulsionInterface
from src.propulsion.integration import PropulsionIntegrator
from src.power.resource_management import PowerPriority
from src.core.utils.logging_framework import get_logger

logger = get_logger("propulsion_power_adapter")


class PropulsionPowerAdapter:
    """Adapter for propulsion-power integration."""
    
    def __init__(self, power_integrator: PowerSystemIntegrator):
        """
        Initialize propulsion power adapter.
        
        Args:
            power_integrator: Power system integrator
        """
        self.power_integrator = power_integrator
        self.propulsion_integrators: Dict[str, PropulsionIntegrator] = {}
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        self.power_modes: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        logger.info("Propulsion power adapter initialized")
    
    def register_propulsion_integrator(self, 
                                     integrator_id: str, 
                                     propulsion_integrator: PropulsionIntegrator) -> bool:
        """
        Register a propulsion integrator.
        
        Args:
            integrator_id: Integrator identifier
            propulsion_integrator: Propulsion integrator
            
        Returns:
            Success status
        """
        if integrator_id in self.propulsion_integrators:
            logger.warning(f"Propulsion integrator '{integrator_id}' already registered")
            return False
        
        self.propulsion_integrators[integrator_id] = propulsion_integrator
        
        # Register all propulsion systems from this integrator
        for system_id, system in propulsion_integrator.propulsion_systems.items():
            full_id = f"{integrator_id}_{system_id}"
            self.propulsion_systems[full_id] = system
            
            # Create power modes for this system
            self._create_power_modes(full_id, system)
            
            # Integrate with power system
            self.power_integrator.integrate_propulsion_system(
                full_id, 
                system, 
                self.power_modes[full_id]["normal"],
                PowerPriority.HIGH
            )
        
        logger.info(f"Registered propulsion integrator '{integrator_id}' with {len(propulsion_integrator.propulsion_systems)} systems")
        return True
    
    def register_propulsion_system(self, 
                                 system_id: str, 
                                 propulsion_system: PropulsionInterface) -> bool:
        """
        Register an individual propulsion system.
        
        Args:
            system_id: System identifier
            propulsion_system: Propulsion system
            
        Returns:
            Success status
        """
        if system_id in self.propulsion_systems:
            logger.warning(f"Propulsion system '{system_id}' already registered")
            return False
        
        self.propulsion_systems[system_id] = propulsion_system
        
        # Create power modes for this system
        self._create_power_modes(system_id, propulsion_system)
        
        # Integrate with power system
        self.power_integrator.integrate_propulsion_system(
            system_id, 
            propulsion_system, 
            self.power_modes[system_id]["normal"],
            PowerPriority.HIGH
        )
        
        logger.info(f"Registered propulsion system '{system_id}'")
        return True
    
    def _create_power_modes(self, system_id: str, system: PropulsionInterface) -> None:
        """
        Create power modes for a propulsion system.
        
        Args:
            system_id: System identifier
            system: Propulsion system
        """
        try:
            specs = system.get_specifications()
            
            # Default power values
            idle_power = 0.5  # Default 0.5 kW
            nominal_power = 2.0  # Default 2.0 kW
            max_power = 5.0  # Default 5.0 kW
            
            # Try to find any attribute that might contain power information
            power_found = False
            for attr_name in dir(specs):
                # Skip private attributes and methods
                if attr_name.startswith('_') or callable(getattr(specs, attr_name)):
                    continue
                
                # Look for attributes that might contain power information
                if 'power' in attr_name.lower() or 'energy' in attr_name.lower():
                    attr_value = getattr(specs, attr_name)
                    # Check if it's a number
                    if isinstance(attr_value, (int, float)) and attr_value > 0:
                        nominal_power = attr_value
                        idle_power = nominal_power * 0.25
                        max_power = nominal_power * 2.0
                        power_found = True
                        logger.debug(f"Found power attribute '{attr_name}' with value {attr_value}")
                        break
            
            # If no power attribute found, try to use weight or thrust as a proxy
            if not power_found:
                if hasattr(specs, 'weight') and isinstance(specs.weight, (int, float)) and specs.weight > 0:
                    # Rough estimate: 1 kW per 10 kg
                    nominal_power = specs.weight / 10.0
                    idle_power = nominal_power * 0.25
                    max_power = nominal_power * 2.0
                    logger.debug(f"Estimated power from weight: {nominal_power} kW")
                # Check for thrust-related attributes with different possible names
                elif any(hasattr(specs, attr) for attr in ['thrust', 'max_thrust', 'thrust_capacity', 'thrust_rating']):
                    # Find the first available thrust attribute
                    for thrust_attr in ['thrust', 'max_thrust', 'thrust_capacity', 'thrust_rating']:
                        if hasattr(specs, thrust_attr):
                            thrust_value = getattr(specs, thrust_attr)
                            if isinstance(thrust_value, (int, float)) and thrust_value > 0:
                                # Rough estimate: 1 kW per 50 N of thrust
                                nominal_power = thrust_value / 50.0
                                idle_power = nominal_power * 0.25
                                max_power = nominal_power * 2.0
                                logger.debug(f"Estimated power from {thrust_attr}: {nominal_power} kW")
                                break
                # If no specific attributes found, use default values
                else:
                    logger.debug(f"Using default power values for system '{system_id}'")
            
            # Ensure power values are positive
            idle_power = max(0.1, idle_power)
            nominal_power = max(idle_power, nominal_power)
            max_power = max(nominal_power, max_power)
            
            # Create power modes
            self.power_modes[system_id] = {
                "idle": {"main_power": idle_power},
                "normal": {"main_power": nominal_power},
                "max": {"main_power": max_power},
                "emergency": {"main_power": idle_power * 0.8}  # 80% of idle
            }
            
            logger.debug(f"Created power modes for system '{system_id}': idle={idle_power}kW, normal={nominal_power}kW, max={max_power}kW")
            
        except Exception as e:
            logger.error(f"Error creating power modes for system '{system_id}': {str(e)}")
            # Create fallback power modes
            self.power_modes[system_id] = {
                "idle": {"main_power": 0.5},
                "normal": {"main_power": 2.0},
                "max": {"main_power": 5.0},
                "emergency": {"main_power": 0.4}
            }
    
    def set_power_mode(self, system_id: str, mode: str) -> bool:
        """
        Set power mode for a propulsion system.
        
        Args:
            system_id: System identifier
            mode: Power mode
            
        Returns:
            Success status
        """
        if system_id not in self.propulsion_systems:
            logger.warning(f"Propulsion system '{system_id}' not found")
            return False
        
        if mode not in self.power_modes[system_id]:
            logger.warning(f"Power mode '{mode}' not defined for system '{system_id}'")
            return False
        
        try:
            # Update power requirements with the power integrator
            requirements = self.power_modes[system_id][mode]
            
            # Get current priority or use default
            current_priority = self.power_integrator.system_priorities.get(
                system_id, PowerPriority.HIGH
            )
            
            # Re-integrate with updated requirements
            integration_success = self.power_integrator.integrate_propulsion_system(
                system_id,
                self.propulsion_systems[system_id],
                requirements,
                current_priority
            )
            
            if not integration_success:
                logger.warning(f"Failed to re-integrate propulsion system '{system_id}' with new power mode")
                return False
            
            # Update power distribution
            self.power_integrator.update_power_distribution()
            
            # Verify the system received the expected power
            power_status = self.power_integrator.get_system_power_status()
            if system_id in power_status.get("propulsion", {}):
                logger.info(f"Set power mode '{mode}' for propulsion system '{system_id}'")
                return True
            else:
                logger.warning(f"Power mode '{mode}' set for system '{system_id}' but no power allocated")
                return False
            
        except Exception as e:
            logger.error(f"Error setting power mode: {str(e)}")
            return False
    
    def handle_propulsion_event(self, system_id: str, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        Handle propulsion system events.
        
        Args:
            system_id: System identifier
            event_type: Event type
            event_data: Event data
            
        Returns:
            Success status
        """
        if system_id not in self.propulsion_systems:
            logger.warning(f"Propulsion system '{system_id}' not found")
            return False
        
        try:
            if event_type == "startup":
                # Set to normal power mode
                return self.set_power_mode(system_id, "normal")
                
            elif event_type == "shutdown":
                # Set to idle power mode
                return self.set_power_mode(system_id, "idle")
                
            elif event_type == "emergency":
                # Set to emergency power mode
                return self.set_power_mode(system_id, "emergency")
                
            elif event_type == "max_thrust":
                # Set to max power mode
                return self.set_power_mode(system_id, "max")
                
            else:
                logger.warning(f"Unknown propulsion event type: {event_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error handling propulsion event: {str(e)}")
            return False
    
    def get_propulsion_power_status(self, system_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get power status for propulsion systems.
        
        Args:
            system_id: Optional system identifier to get status for a specific system
            
        Returns:
            Power status information
        """
        status = {
            "timestamp": time.time(),
            "systems": {}
        }
        
        # Get overall power status
        power_status = self.power_integrator.get_system_power_status()
        
        # Filter for propulsion systems
        if system_id:
            # Get status for specific system
            if system_id in self.propulsion_systems:
                if system_id in power_status.get("propulsion", {}):
                    allocation = power_status["propulsion"][system_id]
                    current_mode = "unknown"
                    
                    # Determine current mode based on power allocation
                    for mode, requirements in self.power_modes.get(system_id, {}).items():
                        if abs(sum(requirements.values()) - allocation["total_allocation"]) < 0.1:
                            current_mode = mode
                            break
                    
                    status["systems"][system_id] = {
                        "allocation": allocation,
                        "current_mode": current_mode,
                        "available_modes": list(self.power_modes.get(system_id, {}).keys())
                    }
                else:
                    status["systems"][system_id] = {
                        "error": "System registered but not receiving power"
                    }
            else:
                status["systems"][system_id] = {
                    "error": "System not registered"
                }
        else:
            # Get status for all systems
            for sys_id in self.propulsion_systems:
                if sys_id in power_status.get("propulsion", {}):
                    status["systems"][sys_id] = {
                        "allocation": power_status["propulsion"][sys_id],
                        "available_modes": list(self.power_modes.get(sys_id, {}).keys())
                    }
        
        return status