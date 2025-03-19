"""
Hardware Capability Negotiation System

Provides automatic detection of hardware capabilities and negotiation of features
to ensure optimal utilization of available neuromorphic hardware.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import logging

from src.core.utils.logging_framework import get_logger
from src.core.hardware.hardware_capabilities import HardwareCapabilitiesDiscovery
from src.core.hardware.exceptions import UnsupportedFeatureError

logger = get_logger("capability_negotiator")

class CapabilityNegotiator:
    """
    Negotiates features based on hardware capabilities.
    
    Automatically detects hardware capabilities and negotiates features
    to ensure optimal utilization of available neuromorphic hardware.
    """
    
    def __init__(self, hardware_type: str, hardware_info: Dict[str, Any]):
        """
        Initialize capability negotiator.
        
        Args:
            hardware_type: Type of hardware ('loihi', 'spinnaker', 'truenorth')
            hardware_info: Hardware information dictionary
        """
        self.hardware_type = hardware_type
        self.hardware_info = hardware_info
        self.discovery = HardwareCapabilitiesDiscovery()
        self.capabilities = self.discovery.discover_capabilities(hardware_info)
        
        # Feature requirements for different operations
        self.feature_requirements = {
            "on_chip_learning": {
                "required": ["on_chip_learning"],
                "optional": ["supports_stdp"]
            },
            "real_time_simulation": {
                "required": ["real_time_capable"],
                "optional": ["hardware_timestep_us"]
            },
            "multi_compartment": {
                "required": ["supports_multi_compartment"],
                "optional": ["max_compartments_per_neuron"]
            }
        }
    
    def negotiate_features(self, requested_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Negotiate features based on hardware capabilities.
        
        Args:
            requested_features: Dictionary of requested features and parameters
            
        Returns:
            Dict[str, Any]: Dictionary of negotiated features and parameters
        """
        negotiated = {}
        unsupported = []
        
        for feature, params in requested_features.items():
            if self._is_feature_supported(feature):
                negotiated[feature] = self._negotiate_parameters(feature, params)
            else:
                unsupported.append(feature)
                
        if unsupported:
            logger.warning(f"Unsupported features on {self.hardware_type}: {', '.join(unsupported)}")
            
        return negotiated
    
    def _is_feature_supported(self, feature: str) -> bool:
        """Check if a feature is supported by the hardware."""
        if feature not in self.feature_requirements:
            return False
            
        required = self.feature_requirements[feature]["required"]
        return all(req in self.capabilities and self.capabilities[req] for req in required)
    
    def _negotiate_parameters(self, feature: str, requested_params: Any) -> Any:
        """Negotiate parameters for a feature based on hardware capabilities."""
        if not isinstance(requested_params, dict):
            return requested_params
            
        negotiated_params = {}
        
        for param, value in requested_params.items():
            if param == "precision" and "precision" in self.capabilities:
                # Negotiate precision (e.g., weight precision)
                hw_precision = self._parse_precision(self.capabilities["precision"])
                req_precision = self._parse_precision(value)
                negotiated_params[param] = min(hw_precision, req_precision)
                
            elif param == "neuron_count" and "neurons_per_core" in self.capabilities:
                # Negotiate neuron count to be efficient on this hardware
                neurons_per_core = self.capabilities["neurons_per_core"]
                negotiated_params[param] = ((value + neurons_per_core - 1) // neurons_per_core) * neurons_per_core
                
            elif param == "learning_rate" and "on_chip_learning" in self.capabilities:
                # Adjust learning rate based on hardware capabilities
                if self.hardware_type == "loihi":
                    # Loihi has specific learning rate requirements
                    negotiated_params[param] = max(0.001, min(0.1, value))
                else:
                    negotiated_params[param] = value
                    
            else:
                # Pass through other parameters
                negotiated_params[param] = value
                
        return negotiated_params
    
    def _parse_precision(self, precision_str: str) -> int:
        """Parse precision string (e.g., '8-bit') to integer."""
        try:
            return int(precision_str.split('-')[0])
        except (ValueError, AttributeError, IndexError):
            return 32  # Default to 32-bit if parsing fails
    
    def get_supported_features(self) -> List[str]:
        """Get list of supported features on this hardware."""
        return [
            feature for feature in self.feature_requirements
            if self._is_feature_supported(feature)
        ]
    
    def check_feature_compatibility(self, network_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if a network configuration is compatible with the hardware.
        
        Args:
            network_config: Neural network configuration
            
        Returns:
            Tuple[bool, List[str]]: (is_compatible, incompatibility_reasons)
        """
        incompatibilities = []
        
        # Check neuron model compatibility
        if "neuron_model" in network_config:
            model = network_config["neuron_model"]
            if "neuron_models" in self.capabilities:
                if model not in self.capabilities["neuron_models"]:
                    incompatibilities.append(f"Neuron model '{model}' not supported")
        
        # Check learning rule compatibility
        if "learning_rule" in network_config:
            rule = network_config["learning_rule"]
            if not self.capabilities.get("on_chip_learning", False):
                incompatibilities.append("On-chip learning not supported")
            elif "learning_rules" in self.capabilities:
                if rule not in self.capabilities["learning_rules"]:
                    incompatibilities.append(f"Learning rule '{rule}' not supported")
        
        # Check network size compatibility
        if "neurons" in network_config:
            neuron_count = len(network_config["neurons"])
            max_neurons = self.capabilities.get("total_neurons", float('inf'))
            if neuron_count > max_neurons:
                incompatibilities.append(f"Network too large: {neuron_count} neurons > {max_neurons}")
        
        return len(incompatibilities) == 0, incompatibilities


def create_capability_negotiator(hardware_type: str, hardware_info: Dict[str, Any]) -> CapabilityNegotiator:
    """
    Create a capability negotiator for the specified hardware.
    
    Args:
        hardware_type: Type of hardware ('loihi', 'spinnaker', 'truenorth')
        hardware_info: Hardware information dictionary
        
    Returns:
        CapabilityNegotiator: Hardware capability negotiator
    """
    return CapabilityNegotiator(hardware_type, hardware_info)