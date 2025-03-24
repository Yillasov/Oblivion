"""
Hardware Compatibility Validator

Validates hardware configurations for compatibility across platforms
and ensures configurations meet hardware-specific requirements.
"""

from typing import Dict, Any, List, Tuple, Optional
from src.core.hardware.config_validation import ConfigValidator
from src.core.utils.logging_framework import get_logger
from src.core.utils.exceptions import HardwareConfigurationError

logger = get_logger("compatibility_validator")

class HardwareCompatibilityValidator:
    """Validates hardware configurations for compatibility."""
    
    @staticmethod
    def validate_compatibility(hardware_type: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate hardware configuration for compatibility.
        
        Args:
            hardware_type: Target hardware type
            config: Hardware configuration
            
        Returns:
            Tuple[bool, List[str]]: (is_compatible, incompatibility_reasons)
        """
        # First run basic validation
        is_valid, errors = ConfigValidator.validate(hardware_type, config)
        if not is_valid:
            return False, errors
        
        # Check hardware-specific compatibility
        compatibility_errors = []
        
        # Check hardware type match
        if "hardware_type" in config and config["hardware_type"] != hardware_type:
            compatibility_errors.append(
                f"Configuration hardware_type '{config['hardware_type']}' doesn't match target '{hardware_type}'"
            )
        
        # Hardware-specific compatibility checks
        if hardware_type == "loihi":
            compatibility_errors.extend(HardwareCompatibilityValidator._check_loihi_compatibility(config))
        elif hardware_type == "spinnaker":
            compatibility_errors.extend(HardwareCompatibilityValidator._check_spinnaker_compatibility(config))
        elif hardware_type == "truenorth":
            compatibility_errors.extend(HardwareCompatibilityValidator._check_truenorth_compatibility(config))
        
        return len(compatibility_errors) == 0, compatibility_errors
    
    @staticmethod
    def _check_loihi_compatibility(config: Dict[str, Any]) -> List[str]:
        """Check Loihi-specific compatibility."""
        errors = []
        
        # Check neuron model compatibility
        if "neuron_params" in config:
            neuron_params = config["neuron_params"]
            
            # Check neuron type
            if "type" in neuron_params:
                neuron_type = neuron_params["type"]
                if neuron_type not in ["LIF", "ALIF", "Compartment"]:
                    errors.append(f"Neuron type '{neuron_type}' not supported on Loihi")
            
            # Check weight precision
            if "weight_precision" in neuron_params and neuron_params["weight_precision"] > 8:
                errors.append(f"Weight precision {neuron_params['weight_precision']} exceeds Loihi maximum (8)")
        
        # Check resource limits
        if "neurons_per_core" in config and config["neurons_per_core"] > 1024:
            errors.append(f"neurons_per_core ({config['neurons_per_core']}) exceeds Loihi maximum (1024)")
        
        if "cores_per_chip" in config and config["cores_per_chip"] > 128:
            errors.append(f"cores_per_chip ({config['cores_per_chip']}) exceeds Loihi maximum (128)")
        
        return errors
    
    @staticmethod
    def _check_spinnaker_compatibility(config: Dict[str, Any]) -> List[str]:
        """Check SpiNNaker-specific compatibility."""
        errors = []
        
        # Check neuron model compatibility
        if "neuron_params" in config:
            neuron_params = config["neuron_params"]
            
            # Check neuron type
            if "type" in neuron_params:
                neuron_type = neuron_params["type"]
                if neuron_type not in ["IF", "IZH", "LIF", "Poisson"]:
                    errors.append(f"Neuron type '{neuron_type}' not supported on SpiNNaker")
        
        # Check network topology
        if "network" in config and "topology" in config["network"]:
            topology = config["network"]["topology"]
            if topology == "fully_connected" and config.get("network", {}).get("size", 0) > 1000:
                errors.append("Fully connected topology with >1000 neurons not efficient on SpiNNaker")
        
        return errors
    
    @staticmethod
    def _check_truenorth_compatibility(config: Dict[str, Any]) -> List[str]:
        """Check TrueNorth-specific compatibility."""
        errors = []
        
        # Check neuron model compatibility
        if "neuron_params" in config:
            neuron_params = config["neuron_params"]
            
            # Check neuron type - TrueNorth only supports specific neuron models
            if "type" in neuron_params:
                neuron_type = neuron_params["type"]
                if neuron_type not in ["TrueNorthLIF"]:
                    errors.append(f"Neuron type '{neuron_type}' not supported on TrueNorth")
            
            # Check weight precision - TrueNorth uses binary weights
            if "weight_precision" in neuron_params and neuron_params["weight_precision"] != 1:
                errors.append(f"TrueNorth only supports binary weights (precision=1), got {neuron_params['weight_precision']}")
        
        return errors
    
    @staticmethod
    def check_migration_compatibility(source_hw: str, target_hw: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if configuration can be migrated between hardware types.
        
        Args:
            source_hw: Source hardware type
            target_hw: Target hardware type
            config: Configuration to migrate
            
        Returns:
            Tuple[bool, List[str]]: (can_migrate, migration_issues)
        """
        migration_issues = []
        
        # Check if source configuration is valid
        is_valid, errors = ConfigValidator.validate(source_hw, config)
        if not is_valid:
            return False, [f"Source configuration invalid: {err}" for err in errors]
        
        # Check specific migration paths
        if source_hw == "loihi" and target_hw == "spinnaker":
            migration_issues.extend(HardwareCompatibilityValidator._check_loihi_to_spinnaker(config))
        elif source_hw == "loihi" and target_hw == "truenorth":
            migration_issues.extend(HardwareCompatibilityValidator._check_loihi_to_truenorth(config))
        elif source_hw == "spinnaker" and target_hw == "loihi":
            migration_issues.extend(HardwareCompatibilityValidator._check_spinnaker_to_loihi(config))
        elif source_hw == "spinnaker" and target_hw == "truenorth":
            migration_issues.extend(HardwareCompatibilityValidator._check_spinnaker_to_truenorth(config))
        elif source_hw == "truenorth" and target_hw == "loihi":
            migration_issues.extend(HardwareCompatibilityValidator._check_truenorth_to_loihi(config))
        elif source_hw == "truenorth" and target_hw == "spinnaker":
            migration_issues.extend(HardwareCompatibilityValidator._check_truenorth_to_spinnaker(config))
        
        return len(migration_issues) == 0, migration_issues
    
    @staticmethod
    def _check_loihi_to_spinnaker(config: Dict[str, Any]) -> List[str]:
        """Check Loihi to SpiNNaker migration compatibility."""
        issues = []
        
        # Check neuron model compatibility
        if "neuron_params" in config and "type" in config["neuron_params"]:
            neuron_type = config["neuron_params"]["type"]
            if neuron_type == "Compartment":
                issues.append("Compartment neurons not directly supported on SpiNNaker")
        
        return issues
    
    @staticmethod
    def _check_loihi_to_truenorth(config: Dict[str, Any]) -> List[str]:
        """Check Loihi to TrueNorth migration compatibility."""
        issues = []
        
        # TrueNorth has binary weights
        if "neuron_params" in config and "weight_precision" in config["neuron_params"]:
            if config["neuron_params"]["weight_precision"] > 1:
                issues.append("TrueNorth only supports binary weights")
        
        return issues
    
    @staticmethod
    def _check_spinnaker_to_loihi(config: Dict[str, Any]) -> List[str]:
        """Check SpiNNaker to Loihi migration compatibility."""
        issues = []
        
        # Check neuron model compatibility
        if "neuron_params" in config and "type" in config["neuron_params"]:
            neuron_type = config["neuron_params"]["type"]
            if neuron_type == "IZH":
                issues.append("Izhikevich neurons require adaptation for Loihi")
        
        return issues
    
    @staticmethod
    def _check_spinnaker_to_truenorth(config: Dict[str, Any]) -> List[str]:
        """Check SpiNNaker to TrueNorth migration compatibility."""
        issues = []
        
        # TrueNorth has binary weights
        if "neuron_params" in config and "weight_precision" in config["neuron_params"]:
            if config["neuron_params"]["weight_precision"] > 1:
                issues.append("TrueNorth only supports binary weights")
        
        return issues
    
    @staticmethod
    def _check_truenorth_to_loihi(config: Dict[str, Any]) -> List[str]:
        """Check TrueNorth to Loihi migration compatibility."""
        # TrueNorth to Loihi is generally compatible
        return []
    
    @staticmethod
    def _check_truenorth_to_spinnaker(config: Dict[str, Any]) -> List[str]:
        """Check TrueNorth to SpiNNaker migration compatibility."""
        # TrueNorth to SpiNNaker is generally compatible
        return []