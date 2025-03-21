"""
Power requirement calculation system.

This module calculates power requirements for various systems
based on specifications, operational modes, and environmental conditions.
"""

from typing import Dict, Any
import math

class PowerRequirementCalculator:
    """Calculates power requirements for systems."""
    
    @staticmethod
    def calculate_base_requirements(specs: Dict[str, Any]) -> float:
        """
        Calculate base power requirements from specifications.
        
        Args:
            specs: System specifications
            
        Returns:
            Base power requirement in kW
        """
        # Example calculation using weight and volume
        weight_factor = specs.get("weight", 0) * 0.01
        volume_factor = sum(specs.get("volume", {}).values()) * 0.05
        return weight_factor + volume_factor
    
    @staticmethod
    def adjust_for_mode(base_power: float, mode: str) -> float:
        """
        Adjust power requirements based on operational mode.
        
        Args:
            base_power: Base power requirement in kW
            mode: Operational mode
            
        Returns:
            Adjusted power requirement in kW
        """
        mode_multipliers = {
            "standby": 0.5,
            "normal": 1.0,
            "active": 1.5,
            "emergency": 2.0
        }
        return base_power * mode_multipliers.get(mode, 1.0)
    
    @staticmethod
    def adjust_for_environment(base_power: float, conditions: Dict[str, float]) -> float:
        """
        Adjust power requirements based on environmental conditions.
        
        Args:
            base_power: Base power requirement in kW
            conditions: Environmental conditions
            
        Returns:
            Adjusted power requirement in kW
        """
        temperature = conditions.get("temperature", 25.0)
        altitude = conditions.get("altitude", 0.0)
        
        # Increase power requirement at high altitude and extreme temperatures
        altitude_factor = 1.0 + (altitude / 10000.0) * 0.1
        temperature_factor = 1.0 + max(0.0, (temperature - 25.0) / 100.0)
        
        return base_power * altitude_factor * temperature_factor
    
    @staticmethod
    def calculate_total_requirements(specs: Dict[str, Any], mode: str, conditions: Dict[str, float]) -> float:
        """
        Calculate total power requirements.
        
        Args:
            specs: System specifications
            mode: Operational mode
            conditions: Environmental conditions
            
        Returns:
            Total power requirement in kW
        """
        base_power = PowerRequirementCalculator.calculate_base_requirements(specs)
        mode_adjusted_power = PowerRequirementCalculator.adjust_for_mode(base_power, mode)
        total_power = PowerRequirementCalculator.adjust_for_environment(mode_adjusted_power, conditions)
        return total_power