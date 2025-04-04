"""
Cross-module validation for biomimetic designs.

Ensures compatibility and consistency across biomimetic modules.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Set, Union

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.biomimetic.design.principles import BiomimeticDesignFramework, BiomimeticPrinciple

logger = get_logger("biomimetic_validator")

class BiomimeticValidator:
    """Validator for biomimetic designs and integrations."""
    
    def __init__(self, design_framework: BiomimeticDesignFramework):
        """
        Initialize the biomimetic validator.
        
        Args:
            design_framework: Biomimetic design framework
        """
        self.design_framework = design_framework
        self.active_principles: Set[BiomimeticPrinciple] = set()
        
        # Define validation rules
        self.validation_rules = {
            "power_requirements": self._validate_power_requirements,
            "material_compatibility": self._validate_material_compatibility,
            "sensor_integration": self._validate_sensor_integration,
            "principle_compatibility": self._validate_principle_compatibility
        }
        
        logger.info("Biomimetic validator initialized")
    
    def validate_integration(self, integration_framework) -> Dict[str, Any]:
        """
        Validate the integration framework.
        
        Args:
            integration_framework: Biomimetic integration framework
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Check module dependencies
        for module in integration_framework.active_modules:
            if module in integration_framework.module_dependencies:
                dependencies = integration_framework.module_dependencies[module]
                for dependency in dependencies:
                    if dependency not in integration_framework.active_modules:
                        errors.append(f"Module '{module}' depends on '{dependency}' which is not active")
        
        # Run validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_result = rule_func(integration_framework)
                if not rule_result["success"]:
                    errors.extend(rule_result["errors"])
                    warnings.extend(rule_result.get("warnings", []))
            except Exception as e:
                errors.append(f"Error in validation rule '{rule_name}': {e}")
        
        # Determine active principles
        self._determine_active_principles(integration_framework)
        
        return {
            "success": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "active_principles": list(self.active_principles)
        }
    
    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Validate power configuration
        if "power" in config:
            power_config = config["power"]
            
            # Validate power state
            if "state" in power_config:
                valid_states = ["high_performance", "balanced", "efficiency", "stealth", "emergency"]
                if power_config["state"] not in valid_states:
                    errors.append(f"Invalid power state: {power_config['state']}")
        
        # Validate hardware configuration
        if "hardware" in config:
            hardware_config = config["hardware"]
            
            # Validate wing flapping configuration
            if "wing_flapping" in hardware_config:
                wing_config = hardware_config["wing_flapping"]
                
                if "frequency" in wing_config and (wing_config["frequency"] < 0.1 or wing_config["frequency"] > 10.0):
                    errors.append(f"Wing flapping frequency out of range: {wing_config['frequency']}")
                
                if "amplitude" in wing_config and (wing_config["amplitude"] < 0.0 or wing_config["amplitude"] > 1.0):
                    errors.append(f"Wing flapping amplitude out of range: {wing_config['amplitude']}")
        
        return {
            "success": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_power_requirements(self, integration_framework) -> Dict[str, Any]:
        """Validate power requirements."""
        errors = []
        
        if not hasattr(integration_framework, "power_integrator") or not integration_framework.power_integrator:
            return {"success": True}  # Skip if power integrator not available
        
        # Check if hardware power requirements are registered
        if hasattr(integration_framework, "hardware_integration") and integration_framework.hardware_integration:
            for group_id in getattr(integration_framework.hardware_integration.actuator_controller, "actuator_groups", {}):
                if group_id not in integration_framework.power_integrator.biomimetic_systems:
                    errors.append(f"Actuator group '{group_id}' not registered with power integrator")
        
        return {
            "success": len(errors) == 0,
            "errors": errors
        }
    
    def _validate_material_compatibility(self, integration_framework) -> Dict[str, Any]:
        """Validate material compatibility."""
        # This would check if selected materials are compatible with the design principles
        return {"success": True}  # Simplified for brevity
    
    def _validate_sensor_integration(self, integration_framework) -> Dict[str, Any]:
        """Validate sensor integration."""
        errors = []
        
        if not hasattr(integration_framework, "sensor_interface"):
            return {"success": True}  # Skip if sensor interface not available
        
        # Check if sensors are properly mapped
        if len(integration_framework.sensor_interface.mappings) == 0:
            errors.append("No sensor mappings defined")
        
        return {
            "success": len(errors) == 0,
            "errors": errors
        }
    
    def _validate_principle_compatibility(self, integration_framework) -> Dict[str, Any]:
        """Validate compatibility between active principles."""
        # This would check if the active principles are compatible with each other
        return {"success": True}  # Simplified for brevity
    
    def _determine_active_principles(self, integration_framework) -> None:
        """Determine active biomimetic principles."""
        self.active_principles = set()
        
        # Check power integrator for energy efficiency
        if hasattr(integration_framework, "power_integrator") and integration_framework.power_integrator:
            if integration_framework.power_integrator.energy_harvesting_enabled:
                self.active_principles.add(BiomimeticPrinciple.ENERGY_EFFICIENCY)
        
        # Check hardware integration for adaptive morphology
        if hasattr(integration_framework, "hardware_integration") and integration_framework.hardware_integration:
            if hasattr(integration_framework.hardware_integration, "cpg_controller") and integration_framework.hardware_integration.cpg_controller:
                self.active_principles.add(BiomimeticPrinciple.ADAPTIVE_MORPHOLOGY)
        
        # Check sensor interface for sensory integration
        if hasattr(integration_framework, "sensor_interface") and integration_framework.sensor_interface:
            if len(integration_framework.sensor_interface.sensors) > 0:
                self.active_principles.add(BiomimeticPrinciple.SENSORY_INTEGRATION)
    
    def get_active_principles(self) -> List[BiomimeticPrinciple]:
        """
        Get the active biomimetic principles.
        
        Returns:
            List of active principles
        """
        return list(self.active_principles)