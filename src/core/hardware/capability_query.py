"""
Hardware Capability Query System

Provides a simple interface to query hardware capabilities across different
neuromorphic hardware platforms.
"""

from typing import Dict, List, Any, Optional, Set, Union
import operator
from functools import reduce

from src.core.utils.logging_framework import get_logger
from src.core.hardware.hardware_registry import hardware_registry
from src.core.hardware.exceptions import NeuromorphicHardwareError

logger = get_logger("capability_query")


class CapabilityQuery:
    """
    Query system for hardware capabilities.
    
    Allows filtering and comparing hardware based on their capabilities.
    """
    
    def __init__(self):
        """Initialize capability query system."""
        self.registry = hardware_registry
    
    def find_hardware_with_capability(self, capability: str, min_value: Any = None) -> List[str]:
        """
        Find hardware types that have a specific capability.
        
        Args:
            capability: Capability name (e.g., "on_chip_learning")
            min_value: Minimum value for the capability (optional)
            
        Returns:
            List[str]: List of hardware types with the capability
        """
        result = []
        
        for hw_type in self.registry.get_available_hardware_types():
            hw = self.registry.get_hardware(hw_type)
            if not hw:
                continue
                
            capabilities = hw.get_capabilities()
            
            if capability in capabilities:
                if min_value is not None:
                    # Check if capability meets minimum value
                    if self._compare_values(capabilities[capability], min_value):
                        result.append(hw_type)
                else:
                    result.append(hw_type)
        
        return result
    
    def find_best_hardware_for_capabilities(self, required_capabilities: Dict[str, Any]) -> Optional[str]:
        """
        Find best hardware that meets all required capabilities.
        
        Args:
            required_capabilities: Dictionary of required capabilities
            
        Returns:
            Optional[str]: Best hardware type or None if none found
        """
        matching_hardware = []
        
        for hw_type in self.registry.get_available_hardware_types():
            hw = self.registry.get_hardware(hw_type)
            if not hw:
                continue
                
            capabilities = hw.get_capabilities()
            meets_requirements = True
            
            for cap_name, cap_value in required_capabilities.items():
                if cap_name not in capabilities:
                    meets_requirements = False
                    break
                    
                if not self._compare_values(capabilities[cap_name], cap_value):
                    meets_requirements = False
                    break
            
            if meets_requirements:
                matching_hardware.append(hw_type)
        
        if not matching_hardware:
            return None
            
        # Return the first matching hardware for simplicity
        # In a real implementation, you might want to rank them
        return matching_hardware[0]
    
    def get_capability(self, hardware_type: str, capability: str) -> Any:
        """
        Get specific capability value for a hardware type.
        
        Args:
            hardware_type: Hardware type
            capability: Capability name
            
        Returns:
            Any: Capability value or None if not found
        """
        hw = self.registry.get_hardware(hardware_type)
        if not hw:
            return None
            
        capabilities = hw.get_capabilities()
        return capabilities.get(capability)
    
    def compare_hardware(self, hw_type1: str, hw_type2: str, 
                        capabilities: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare capabilities between two hardware types.
        
        Args:
            hw_type1: First hardware type
            hw_type2: Second hardware type
            capabilities: List of capabilities to compare
            
        Returns:
            Dict[str, Dict[str, Any]]: Comparison results
        """
        hw1 = self.registry.get_hardware(hw_type1)
        hw2 = self.registry.get_hardware(hw_type2)
        
        if not hw1 or not hw2:
            return {}
            
        caps1 = hw1.get_capabilities()
        caps2 = hw2.get_capabilities()
        
        result = {}
        
        for cap in capabilities:
            if cap in caps1 and cap in caps2:
                result[cap] = {
                    hw_type1: caps1[cap],
                    hw_type2: caps2[cap],
                    "difference": self._calculate_difference(caps1[cap], caps2[cap])
                }
        
        return result
    
    def _compare_values(self, actual_value: Any, required_value: Any) -> bool:
        """Compare capability values."""
        if isinstance(actual_value, (int, float)) and isinstance(required_value, (int, float)):
            return actual_value >= required_value
        elif isinstance(actual_value, bool) and isinstance(required_value, bool):
            return actual_value == required_value
        elif isinstance(actual_value, list) and isinstance(required_value, list):
            return all(item in actual_value for item in required_value)
        else:
            return actual_value == required_value
    
    def _calculate_difference(self, value1: Any, value2: Any) -> Any:
        """Calculate difference between capability values."""
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            return value1 - value2
        elif isinstance(value1, bool) and isinstance(value2, bool):
            return value1 == value2
        elif isinstance(value1, list) and isinstance(value2, list):
            return {
                "only_in_first": [item for item in value1 if item not in value2],
                "only_in_second": [item for item in value2 if item not in value1],
                "common": [item for item in value1 if item in value2]
            }
        else:
            return "different" if value1 != value2 else "same"


# Create global instance
capability_query = CapabilityQuery()


def find_hardware_with_capability(capability: str, min_value: Any = None) -> List[str]:
    """
    Find hardware types that have a specific capability.
    
    Args:
        capability: Capability name (e.g., "on_chip_learning")
        min_value: Minimum value for the capability (optional)
        
    Returns:
        List[str]: List of hardware types with the capability
    """
    return capability_query.find_hardware_with_capability(capability, min_value)


def find_best_hardware_for_capabilities(required_capabilities: Dict[str, Any]) -> Optional[str]:
    """
    Find best hardware that meets all required capabilities.
    
    Args:
        required_capabilities: Dictionary of required capabilities
        
    Returns:
        Optional[str]: Best hardware type or None if none found
    """
    return capability_query.find_best_hardware_for_capabilities(required_capabilities)


def get_capability(hardware_type: str, capability: str) -> Any:
    """
    Get specific capability value for a hardware type.
    
    Args:
        hardware_type: Hardware type
        capability: Capability name
        
    Returns:
        Any: Capability value or None if not found
    """
    return capability_query.get_capability(hardware_type, capability)


def compare_hardware(hw_type1: str, hw_type2: str, capabilities: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Compare capabilities between two hardware types.
    
    Args:
        hw_type1: First hardware type
        hw_type2: Second hardware type
        capabilities: List of capabilities to compare
        
    Returns:
        Dict[str, Dict[str, Any]]: Comparison results
    """
    return capability_query.compare_hardware(hw_type1, hw_type2, capabilities)