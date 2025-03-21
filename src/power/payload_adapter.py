"""
Adapter for payload-power integration.

This module provides adapters for integrating payload systems with power management.
"""

from typing import Dict, List, Any, Optional
import time

from src.power.system_integration import PowerSystemIntegrator
from src.payload.base import PayloadInterface
from src.payload.integration import PayloadIntegrator
from src.power.resource_management import PowerPriority
from src.core.utils.logging_framework import get_logger

logger = get_logger("payload_power_adapter")


class PayloadPowerAdapter:
    """Adapter for payload-power integration."""
    
    def __init__(self, power_integrator: PowerSystemIntegrator):
        """
        Initialize payload power adapter.
        
        Args:
            power_integrator: Power system integrator
        """
        self.power_integrator = power_integrator
        self.payload_integrators: Dict[str, PayloadIntegrator] = {}
        self.payload_systems: Dict[str, PayloadInterface] = {}
        self.power_modes: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        logger.info("Payload power adapter initialized")
    
    def register_payload_integrator(self, 
                                  integrator_id: str, 
                                  payload_integrator: PayloadIntegrator) -> bool:
        """
        Register a payload integrator.
        
        Args:
            integrator_id: Integrator identifier
            payload_integrator: Payload integrator
            
        Returns:
            Success status
        """
        if integrator_id in self.payload_integrators:
            logger.warning(f"Payload integrator '{integrator_id}' already registered")
            return False
        
        self.payload_integrators[integrator_id] = payload_integrator
        
        # Register all payload systems from this integrator
        # Use the payloads attribute instead of payload_systems
        systems = {}
        for payload_entry in payload_integrator.payloads:
            # Each entry in payloads is a dict with 'payload', 'mount_type', and 'mount_location'
            payload = payload_entry.get("payload")
            if payload:
                # Use a unique ID based on the payload specifications
                specs = payload.get_specifications()
                system_id = f"{specs.__class__.__name__}_{id(payload)}"
                systems[system_id] = payload
        
        if not systems:
            logger.warning(f"No payload systems found in integrator '{integrator_id}'")
            return True
            
        for system_id, system in systems.items():
            full_id = f"{integrator_id}_{system_id}"
            self.payload_systems[full_id] = system
            
            # Create power modes for this system
            self._create_power_modes(full_id, system)
            
            # Integrate with power system
            self.power_integrator.integrate_payload_system(
                full_id, 
                system, 
                self.power_modes[full_id]["normal"],
                PowerPriority.MEDIUM
            )
        
        logger.info(f"Registered payload integrator '{integrator_id}' with {len(systems)} systems")
        return True
    
    def register_payload_system(self, 
                              system_id: str, 
                              payload_system: PayloadInterface) -> bool:
        """
        Register an individual payload system.
        
        Args:
            system_id: System identifier
            payload_system: Payload system
            
        Returns:
            Success status
        """
        if system_id in self.payload_systems:
            logger.warning(f"Payload system '{system_id}' already registered")
            return False
        
        self.payload_systems[system_id] = payload_system
        
        # Create power modes for this system
        self._create_power_modes(system_id, payload_system)
        
        # Integrate with power system
        integration_success = self.power_integrator.integrate_payload_system(
            system_id, 
            payload_system, 
            self.power_modes[system_id]["normal"],
            PowerPriority.MEDIUM
        )
        
        if not integration_success:
            logger.warning(f"Failed to integrate payload system '{system_id}' with power system")
            return False
            
        logger.info(f"Registered payload system '{system_id}'")
        return True
    
    def _create_power_modes(self, system_id: str, system: PayloadInterface) -> None:
        """
        Create power modes for a payload system.
        
        Args:
            system_id: System identifier
            system: Payload system
        """
        try:
            specs = system.get_specifications()
            
            # Default power value
            nominal_power = 0.5  # 500W default
            
            # Try to extract power information from specs
            power_found = False
            
            # First check for direct power requirement attribute
            if hasattr(specs, 'power_requirements') and isinstance(specs.power_requirements, (int, float)):
                nominal_power = specs.power_requirements
                power_found = True
                logger.debug(f"Found power_requirements: {nominal_power}kW")
            
            # If not found, try other common attribute names
            if not power_found:
                for attr_name in ['power_consumption', 'power_rating', 'power_draw', 'power']:
                    if hasattr(specs, attr_name):
                        attr_value = getattr(specs, attr_name)
                        if isinstance(attr_value, (int, float)) and attr_value > 0:
                            nominal_power = attr_value
                            power_found = True
                            logger.debug(f"Found {attr_name}: {nominal_power}kW")
                            break
            
            # Ensure power value is positive
            nominal_power = max(0.1, nominal_power)
            
            # Create power modes with appropriate scaling
            self.power_modes[system_id] = {
                "standby": {"payload_power": nominal_power * 0.2},  # 20% of nominal
                "normal": {"payload_power": nominal_power},
                "active": {"payload_power": nominal_power * 1.5},  # 150% of nominal
                "low_power": {"payload_power": nominal_power * 0.1}  # 10% of nominal
            }
            
            logger.debug(f"Created power modes for system '{system_id}': standby={nominal_power*0.2}kW, normal={nominal_power}kW, active={nominal_power*1.5}kW")
            
        except Exception as e:
            logger.error(f"Error creating power modes for system '{system_id}': {str(e)}")
            # Create fallback power modes
            self.power_modes[system_id] = {
                "standby": {"payload_power": 0.1},
                "normal": {"payload_power": 0.5},
                "active": {"payload_power": 0.75},
                "low_power": {"payload_power": 0.05}
            }
    
    def set_power_mode(self, system_id: str, mode: str) -> bool:
        """
        Set power mode for a payload system.
        
        Args:
            system_id: System identifier
            mode: Power mode
            
        Returns:
            Success status
        """
        if system_id not in self.payload_systems:
            logger.warning(f"Payload system '{system_id}' not found")
            return False
        
        if mode not in self.power_modes[system_id]:
            logger.warning(f"Power mode '{mode}' not defined for system '{system_id}'")
            return False
        
        try:
            # Update power requirements with the power integrator
            requirements = self.power_modes[system_id][mode]
            
            # Get current priority or use default
            current_priority = self.power_integrator.system_priorities.get(
                system_id, PowerPriority.MEDIUM
            )
            
            # Re-integrate with updated requirements
            integration_success = self.power_integrator.integrate_payload_system(
                system_id,
                self.payload_systems[system_id],
                requirements,
                current_priority
            )
            
            if not integration_success:
                logger.warning(f"Failed to re-integrate payload system '{system_id}' with new power mode")
                return False
            
            # Update power distribution
            self.power_integrator.update_power_distribution()
            
            # Verify the system received the expected power
            power_status = self.power_integrator.get_system_power_status()
            if system_id in power_status.get("payload", {}):
                logger.info(f"Set power mode '{mode}' for payload system '{system_id}'")
                return True
            else:
                logger.warning(f"Power mode '{mode}' set for system '{system_id}' but no power allocated")
                return False
            
        except Exception as e:
            logger.error(f"Error setting power mode: {str(e)}")
            return False
    
    def handle_payload_event(self, system_id: str, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        Handle payload system events.
        
        Args:
            system_id: System identifier
            event_type: Event type
            event_data: Event data
            
        Returns:
            Success status
        """
        if system_id not in self.payload_systems:
            logger.warning(f"Payload system '{system_id}' not found")
            return False
        
        try:
            if event_type == "activate":
                # Set to active power mode
                return self.set_power_mode(system_id, "active")
                
            elif event_type == "deactivate":
                # Set to standby power mode
                return self.set_power_mode(system_id, "standby")
                
            elif event_type == "low_power_mode":
                # Set to low power mode
                return self.set_power_mode(system_id, "low_power")
                
            elif event_type == "normal_operation":
                # Set to normal power mode
                return self.set_power_mode(system_id, "normal")
                
            else:
                logger.warning(f"Unknown payload event type: {event_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error handling payload event: {str(e)}")
            return False
    
    def get_payload_power_status(self, system_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get power status for payload systems.
        
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
        
        # Filter for payload systems
        if system_id:
            # Get status for specific system
            if system_id in self.payload_systems:
                if system_id in power_status.get("payload", {}):
                    allocation = power_status["payload"][system_id]
                    current_mode = "unknown"
                    
                    # Determine current mode based on power allocation
                    for mode, requirements in self.power_modes.get(system_id, {}).items():
                        if abs(requirements.get("payload_power", 0) - allocation.get("payload_power", 0)) < 0.01:
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
            for sys_id in self.payload_systems:
                if sys_id in power_status.get("payload", {}):
                    allocation = power_status["payload"][sys_id]
                    current_mode = "unknown"
                    
                    # Determine current mode based on power allocation
                    for mode, requirements in self.power_modes.get(sys_id, {}).items():
                        if abs(requirements.get("payload_power", 0) - allocation.get("payload_power", 0)) < 0.01:
                            current_mode = mode
                            break
                    
                    status["systems"][sys_id] = {
                        "allocation": allocation,
                        "current_mode": current_mode,
                        "available_modes": list(self.power_modes.get(sys_id, {}).keys())
                    }
                else:
                    status["systems"][sys_id] = {
                        "error": "System registered but not receiving power"
                    }
        
        return status
