"""
Integration between propulsion systems and stealth technologies.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional
from src.propulsion.base import PropulsionInterface
from src.stealth.low_observable_nozzle import LowObservableNozzle


class PropulsionStealthIntegrator:
    """Integrates propulsion systems with stealth technologies."""
    
    def __init__(self):
        """Initialize propulsion-stealth integrator."""
        self.propulsion_systems: Dict[str, PropulsionInterface] = {}
        self.stealth_systems: Dict[str, Any] = {}
        self.integrations: Dict[str, Dict[str, str]] = {}
        
    def register_propulsion_system(self, system_id: str, system: PropulsionInterface) -> bool:
        """Register a propulsion system."""
        if system_id in self.propulsion_systems:
            return False
            
        self.propulsion_systems[system_id] = system
        return True
        
    def register_stealth_system(self, system_id: str, system: Any) -> bool:
        """Register a stealth system."""
        if system_id in self.stealth_systems:
            return False
            
        self.stealth_systems[system_id] = system
        return True
        
    def integrate_systems(self, propulsion_id: str, stealth_id: str) -> bool:
        """
        Integrate a propulsion system with a stealth system.
        
        Args:
            propulsion_id: ID of the propulsion system
            stealth_id: ID of the stealth system
            
        Returns:
            Success status
        """
        if (propulsion_id not in self.propulsion_systems or 
            stealth_id not in self.stealth_systems):
            return False
            
        propulsion_system = self.propulsion_systems[propulsion_id]
        stealth_system = self.stealth_systems[stealth_id]
        
        # Check if stealth system is a low-observable nozzle
        if isinstance(stealth_system, LowObservableNozzle):
            # Connect the nozzle to the propulsion system
            success = stealth_system.connect_propulsion_system(propulsion_system)
            
            if success:
                # Record the integration
                if propulsion_id not in self.integrations:
                    self.integrations[propulsion_id] = {}
                    
                self.integrations[propulsion_id][stealth_id] = "low_observable_nozzle"
                
                return True
                
        return False
        
    def update_all_integrations(self) -> Dict[str, Any]:
        """
        Update all integrated systems.
        
        Returns:
            Status of all updates
        """
        results = {}
        
        for propulsion_id, stealth_dict in self.integrations.items():
            results[propulsion_id] = {}
            
            for stealth_id, integration_type in stealth_dict.items():
                stealth_system = self.stealth_systems[stealth_id]
                
                if integration_type == "low_observable_nozzle":
                    # Update the nozzle from propulsion
                    success = stealth_system.update_from_propulsion()
                    results[propulsion_id][stealth_id] = success
                    
        return results
        
    def get_propulsion_impact(self, propulsion_id: str) -> Dict[str, Any]:
        """
        Get the impact of stealth systems on a propulsion system.
        
        Args:
            propulsion_id: ID of the propulsion system
            
        Returns:
            Dictionary of impact metrics
        """
        if propulsion_id not in self.integrations:
            return {"has_stealth_integration": False}
            
        impact = {
            "has_stealth_integration": True,
            "systems": {},
            "total_thrust_loss": 0.0,
            "total_fuel_efficiency_impact": 0.0
        }
        
        for stealth_id, integration_type in self.integrations[propulsion_id].items():
            stealth_system = self.stealth_systems[stealth_id]
            
            if integration_type == "low_observable_nozzle":
                # Get impact from nozzle
                nozzle_impact = stealth_system.get_propulsion_impact()
                impact["systems"][stealth_id] = nozzle_impact
                
                # Add to totals
                impact["total_thrust_loss"] += nozzle_impact.get("thrust_loss", 0.0)
                impact["total_fuel_efficiency_impact"] += nozzle_impact.get("fuel_efficiency_impact", 0.0)
                
        return impact
        
    def adjust_stealth_for_mission(self, mission_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust stealth systems based on mission profile.
        
        Args:
            mission_profile: Mission profile data
            
        Returns:
            Adjustment results
        """
        results = {}
        mission_type = mission_profile.get("type", "standard")
        stealth_priority = mission_profile.get("stealth_priority", 0.5)  # 0.0-1.0
        
        for propulsion_id, stealth_dict in self.integrations.items():
            results[propulsion_id] = {}
            
            for stealth_id, integration_type in stealth_dict.items():
                stealth_system = self.stealth_systems[stealth_id]
                
                if integration_type == "low_observable_nozzle":
                    # Adjust nozzle based on mission
                    if mission_type == "stealth":
                        # Maximum stealth
                        params = {
                            "power_level": 1.0,
                            "ir_suppression_active": True,
                            "shape_adaptation_active": True,
                            "cooling_active": True
                        }
                    elif mission_type == "combat":
                        # Balance stealth and performance
                        params = {
                            "power_level": 0.7,
                            "ir_suppression_active": True,
                            "shape_adaptation_active": True,
                            "cooling_active": True
                        }
                    elif mission_type == "cruise":
                        # Moderate stealth, focus on efficiency
                        params = {
                            "power_level": 0.5,
                            "ir_suppression_active": True,
                            "shape_adaptation_active": True,
                            "cooling_active": False
                        }
                    else:
                        # Default settings
                        params = {
                            "power_level": stealth_priority,
                            "ir_suppression_active": stealth_priority > 0.3,
                            "shape_adaptation_active": stealth_priority > 0.2,
                            "cooling_active": stealth_priority > 0.6
                        }
                    
                    # Apply parameters
                    success = stealth_system.adjust_parameters(params)
                    results[propulsion_id][stealth_id] = {
                        "success": success,
                        "params": params
                    }
                    
        return results