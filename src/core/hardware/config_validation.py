#!/usr/bin/env python3
"""
Hardware Configuration Validation

Provides validation for hardware-specific configurations.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, Tuple, List, Optional
import re
import ipaddress

from src.core.utils.logging_framework import get_logger

logger = get_logger("config_validation")


class ConfigValidator:
    """Base class for hardware configuration validators."""
    
    @staticmethod
    def validate(hardware_type: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration for specific hardware type.
        
        Args:
            hardware_type: Hardware type
            config: Configuration to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        # Get appropriate validator
        validator = VALIDATORS.get(hardware_type.lower(), GenericValidator())
        return validator.validate_config(config)


class GenericValidator:
    """Generic configuration validator."""
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate generic configuration."""
        errors = []
        
        # Check for basic required fields
        if "hardware_type" not in config:
            errors.append("Missing 'hardware_type' field")
        
        # Validate monitoring configuration if present
        if "monitoring" in config:
            monitoring = config["monitoring"]
            if not isinstance(monitoring, dict):
                errors.append("'monitoring' must be a dictionary")
            else:
                if "enabled" in monitoring and not isinstance(monitoring["enabled"], bool):
                    errors.append("'monitoring.enabled' must be a boolean")
                if "interval_ms" in monitoring:
                    if not isinstance(monitoring["interval_ms"], (int, float)):
                        errors.append("'monitoring.interval_ms' must be a number")
                    elif monitoring["interval_ms"] <= 0:
                        errors.append("'monitoring.interval_ms' must be positive")
        
        return len(errors) == 0, errors


class LoihiValidator(GenericValidator):
    """Loihi configuration validator."""
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate Loihi configuration."""
        is_valid, errors = super().validate_config(config)
        
        # Check Loihi-specific fields
        if "board_id" not in config and "chip_id" not in config:
            errors.append("Loihi configuration must include 'board_id' or 'chip_id'")
        
        if "board_id" in config and not isinstance(config["board_id"], int):
            errors.append("'board_id' must be an integer")
        
        if "chip_id" in config and not isinstance(config["chip_id"], int):
            errors.append("'chip_id' must be an integer")
        
        if "neurons_per_core" in config:
            if not isinstance(config["neurons_per_core"], int):
                errors.append("'neurons_per_core' must be an integer")
            elif config["neurons_per_core"] > 1024:
                errors.append("'neurons_per_core' cannot exceed 1024 for Loihi")
        
        if "cores_per_chip" in config:
            if not isinstance(config["cores_per_chip"], int):
                errors.append("'cores_per_chip' must be an integer")
            elif config["cores_per_chip"] > 128:
                errors.append("'cores_per_chip' cannot exceed 128 for Loihi")
        
        return len(errors) == 0, errors


class SpiNNakerValidator(GenericValidator):
    """SpiNNaker configuration validator."""
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate SpiNNaker configuration."""
        is_valid, errors = super().validate_config(config)
        
        # Check SpiNNaker-specific fields
        if "board_address" not in config and "ip_address" not in config:
            errors.append("SpiNNaker configuration must include 'board_address' or 'ip_address'")
        
        # Validate IP address format
        ip_address = config.get("board_address", config.get("ip_address", ""))
        if ip_address:
            try:
                ipaddress.ip_address(ip_address)
            except ValueError:
                errors.append(f"Invalid IP address format: {ip_address}")
        
        if "neurons_per_core" in config:
            if not isinstance(config["neurons_per_core"], int):
                errors.append("'neurons_per_core' must be an integer")
            elif config["neurons_per_core"] > 255:
                errors.append("'neurons_per_core' cannot exceed 255 for SpiNNaker")
        
        if "cores_per_chip" in config:
            if not isinstance(config["cores_per_chip"], int):
                errors.append("'cores_per_chip' must be an integer")
            elif config["cores_per_chip"] > 16:
                errors.append("'cores_per_chip' cannot exceed 16 for SpiNNaker")
        
        return len(errors) == 0, errors


class TrueNorthValidator(GenericValidator):
    """TrueNorth configuration validator."""
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate TrueNorth configuration."""
        is_valid, errors = super().validate_config(config)
        
        # Check TrueNorth-specific fields
        if "board_id" not in config and "chip_id" not in config:
            errors.append("TrueNorth configuration must include 'board_id' or 'chip_id'")
        
        if "board_id" in config and not isinstance(config["board_id"], int):
            errors.append("'board_id' must be an integer")
        
        if "chip_id" in config and not isinstance(config["chip_id"], int):
            errors.append("'chip_id' must be an integer")
        
        if "neurons_per_core" in config:
            if not isinstance(config["neurons_per_core"], int):
                errors.append("'neurons_per_core' must be an integer")
            elif config["neurons_per_core"] != 256:
                errors.append("'neurons_per_core' must be exactly 256 for TrueNorth")
        
        if "cores_per_chip" in config:
            if not isinstance(config["cores_per_chip"], int):
                errors.append("'cores_per_chip' must be an integer")
            elif config["cores_per_chip"] != 4096:
                errors.append("'cores_per_chip' must be exactly 4096 for TrueNorth")
        
        return len(errors) == 0, errors


# Register validators for each hardware type
VALIDATORS = {
    "loihi": LoihiValidator(),
    "spinnaker": SpiNNakerValidator(),
    "truenorth": TrueNorthValidator(),
}