#!/usr/bin/env python3
"""
Automated validation tool for neuromorphic hardware compatibility.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import sys
from typing import Dict, Tuple, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.core.hardware.hardware_capabilities import HardwareCapabilitiesDiscovery
from src.core.utils.logging_framework import get_logger

logger = get_logger("hardware_compatibility_validator")

class HardwareCompatibilityValidator:
    def __init__(self, neuromorphic_system: NeuromorphicSystem):
        self.system = neuromorphic_system
        self.capabilities_discovery = HardwareCapabilitiesDiscovery()

    def validate_model_compatibility(self, model_requirements: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate if a neural model is compatible with the current hardware.
        
        Args:
            model_requirements: Requirements of the neural model
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_compatible, details)
        """
        try:
            hardware_info = self.system.get_hardware_info()
            capabilities = self.capabilities_discovery.discover_capabilities(hardware_info)
            
            compatible = True
            details = {}

            # Check neuron count
            if "neuron_count" in model_requirements:
                req_neurons = model_requirements["neuron_count"]
                max_neurons = capabilities.get("neurons_per_core", 0) * capabilities.get("cores_per_chip", 0)
                if req_neurons > max_neurons:
                    compatible = False
                    details["neuron_count"] = f"Required: {req_neurons}, Available: {max_neurons}"
                    # Attempt graceful degradation
                    req_neurons = max_neurons
                    details["neuron_count_adjusted"] = f"Adjusted to: {req_neurons}"

            # Check synapse count
            if "synapse_count" in model_requirements:
                req_synapses = model_requirements["synapse_count"]
                max_synapses = capabilities.get("max_fan_in", 0) * capabilities.get("max_fan_out", 0)
                if req_synapses > max_synapses:
                    compatible = False
                    details["synapse_count"] = f"Required: {req_synapses}, Available: {max_synapses}"
                    # Attempt graceful degradation
                    req_synapses = max_synapses
                    details["synapse_count_adjusted"] = f"Adjusted to: {req_synapses}"

            # Check learning capabilities
            if model_requirements.get("requires_learning", False) and not capabilities.get("on_chip_learning", False):
                compatible = False
                details["learning"] = "On-chip learning not supported"

            logger.info(f"Model compatibility check completed: {'compatible' if compatible else 'incompatible'}")
            return compatible, details

        except KeyError as e:
            logger.error(f"Missing key in hardware capabilities: {str(e)}")
            return False, {"error": f"Missing key in hardware capabilities: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error during compatibility validation: {str(e)}")
            return False, {"error": f"Unexpected error: {str(e)}"}

# Example usage
def main():
    # Create a neuromorphic system (assuming hardware interface is set up)
    neuromorphic_system = NeuromorphicSystem()

    # Create validator
    validator = HardwareCompatibilityValidator(neuromorphic_system)

    # Define model requirements
    model_requirements = {
        "neuron_count": 5000,
        "synapse_count": 20000,
        "requires_learning": True
    }

    # Validate compatibility
    is_compatible, details = validator.validate_model_compatibility(model_requirements)
    print(f"Compatibility: {is_compatible}, Details: {details}")

if __name__ == "__main__":
    main()