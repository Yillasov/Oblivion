#!/usr/bin/env python3
"""
Unified Hardware Configuration Management System

Provides a centralized interface for managing hardware configurations
across different neuromorphic platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime
from enum import Enum
import shutil

from src.core.utils.logging_framework import get_logger

logger = get_logger("unified_config")


class HardwareType(Enum):
    """Supported hardware types."""
    LOIHI = "loihi"
    TRUENORTH = "truenorth"
    SPINNAKER = "spinnaker"
    SIMULATED = "simulated"
    
    @classmethod
    def list(cls) -> List[str]:
        """Get list of all hardware types."""
        return [hw.value for hw in cls]


class ConfigFormat(Enum):
    """Supported configuration formats."""
    JSON = "json"
    YAML = "yaml"


class ConfigCategory(Enum):
    """Configuration categories."""
    HARDWARE = "hardware"
    SIMULATION = "simulation"
    DEPLOYMENT = "deployment"
    TRAINING = "training"
    RUNTIME = "runtime"


class UnifiedConfigManager:
    """
    Unified manager for hardware configurations across platforms.
    
    Provides a centralized interface for:
    - Creating, loading, and saving configurations
    - Managing configuration templates
    - Validating configurations
    - Converting between different hardware platforms
    - Managing active configurations
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'UnifiedConfigManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self, config_root: str = "/Users/yessine/Oblivion/configs"):
        """
        Initialize the unified configuration manager.
        
        Args:
            config_root: Root directory for configurations
        """
        self.config_root = config_root
        self.active_configs: Dict[str, Dict[str, Any]] = {}
        
        # Create directory structure
        os.makedirs(config_root, exist_ok=True)
        
        # Create category directories
        for category in ConfigCategory:
            category_dir = os.path.join(config_root, category.value)
            os.makedirs(category_dir, exist_ok=True)
            
            # Create hardware-specific directories within each category
            for hw_type in HardwareType:
                hw_dir = os.path.join(category_dir, hw_type.value)
                os.makedirs(hw_dir, exist_ok=True)
        
        # Create templates directory
        self.templates_dir = os.path.join(config_root, "templates")
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Initialize default templates if they don't exist
        self._initialize_default_templates()
        
        logger.info(f"Unified configuration manager initialized at {config_root}")
    
    def _initialize_default_templates(self) -> None:
        """Initialize default configuration templates for each hardware type."""
        # Check if templates already exist
        if os.path.exists(os.path.join(self.templates_dir, "loihi_default.json")):
            return
            
        # Create default templates for each hardware type
        self._create_default_loihi_template()
        self._create_default_truenorth_template()
        self._create_default_spinnaker_template()
        self._create_default_simulated_template()
        
        logger.info("Default templates initialized")
    
    def _create_default_loihi_template(self) -> None:
        """Create default Loihi template."""
        template = {
            "board_id": 0,
            "chip_id": 0,
            "connection_type": "local",
            "neuron_params": {
                "threshold": 1.0,
                "decay": 0.5,
                "compartment_type": "LIF"
            },
            "simulation": {
                "timestep_ms": 1.0,
                "max_steps": 1000
            },
            "resources": {
                "cores_per_chip": 128,
                "neurons_per_core": 1024,
                "synapses_per_core": 65536
            }
        }
        
        self.save_template("loihi", "default", template)
    
    def _create_default_truenorth_template(self) -> None:
        """Create default TrueNorth template."""
        template = {
            "board_id": 0,
            "chip_id": 0,
            "connection_type": "local",
            "neuron_params": {
                "threshold": 1.0,
                "leak": 0,
                "reset_mode": "zero"
            },
            "simulation": {
                "timestep_ms": 1.0,
                "max_steps": 1000
            },
            "resources": {
                "cores_per_chip": 4096,
                "neurons_per_core": 256,
                "synapses_per_neuron": 256
            }
        }
        
        self.save_template("truenorth", "default", template)
    
    def _create_default_spinnaker_template(self) -> None:
        """Create default SpiNNaker template."""
        template = {
            "board_address": "192.168.1.1",
            "connection_type": "ethernet",
            "neuron_params": {
                "model": "IF_curr_exp",
                "threshold": 1.0,
                "tau_m": 20.0
            },
            "simulation": {
                "timestep_ms": 1.0,
                "max_steps": 1000
            },
            "resources": {
                "cores_per_chip": 18,
                "chips_per_board": 48,
                "neurons_per_core": 256
            }
        }
        
        self.save_template("spinnaker", "default", template)
    
    def _create_default_simulated_template(self) -> None:
        """Create default simulated template."""
        template = {
            "simulator": "nest",
            "neuron_params": {
                "model": "iaf_psc_alpha",
                "threshold": -55.0,
                "reset": -70.0,
                "tau_m": 10.0
            },
            "simulation": {
                "timestep_ms": 0.1,
                "max_steps": 10000
            },
            "resources": {
                "max_neurons": 100000,
                "max_synapses": 10000000
            }
        }
        
        self.save_template("simulated", "default", template)
    
    def save_template(self, hardware_type: str, template_name: str, template: Dict[str, Any]) -> bool:
        """
        Save a configuration template.
        
        Args:
            hardware_type: Hardware type
            template_name: Template name
            template: Template data
            
        Returns:
            bool: Success status
        """
        try:
            # Add metadata
            template_with_meta = template.copy()
            template_with_meta["_metadata"] = {
                "hardware_type": hardware_type,
                "template_name": template_name,
                "created": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            # Save to file
            path = os.path.join(self.templates_dir, f"{hardware_type}_{template_name}.json")
            with open(path, "w") as f:
                json.dump(template_with_meta, f, indent=2)
                
            logger.info(f"Saved template '{template_name}' for {hardware_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to save template: {str(e)}")
            return False
    
    def get_template(self, hardware_type: str, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a configuration template.
        
        Args:
            hardware_type: Hardware type
            template_name: Template name
            
        Returns:
            Optional[Dict[str, Any]]: Template data or None if not found
        """
        path = os.path.join(self.templates_dir, f"{hardware_type}_{template_name}.json")
        
        if not os.path.exists(path):
            logger.warning(f"Template not found: {hardware_type}/{template_name}")
            return None
            
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load template: {str(e)}")
            return None
    
    def list_templates(self, hardware_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available templates.
        
        Args:
            hardware_type: Optional hardware type filter
            
        Returns:
            Dict[str, List[str]]: Hardware types and their templates
        """
        result = {}
        
        # Get all template files
        template_files = [f for f in os.listdir(self.templates_dir) if f.endswith(".json")]
        
        for file in template_files:
            # Extract hardware type and template name
            parts = file[:-5].split("_", 1)
            if len(parts) != 2:
                continue
                
            hw_type, template_name = parts
            
            # Filter by hardware type if specified
            if hardware_type and hw_type != hardware_type:
                continue
                
            if hw_type not in result:
                result[hw_type] = []
                
            result[hw_type].append(template_name)
        
        return result
    
    def create_config_from_template(self, 
                                   hardware_type: str, 
                                   template_name: str, 
                                   config_name: str,
                                   category: Union[str, ConfigCategory] = ConfigCategory.HARDWARE,
                                   overrides: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Create a new configuration from a template.
        
        Args:
            hardware_type: Hardware type
            template_name: Template name
            config_name: Name for the new configuration
            category: Configuration category
            overrides: Optional parameter overrides
            
        Returns:
            Optional[Dict[str, Any]]: New configuration or None if failed
        """
        # Get template
        template = self.get_template(hardware_type, template_name)
        if not template:
            return None
        
        # Create a copy of the template
        config = template.copy()
        
        # Remove metadata if present
        if "_metadata" in config:
            del config["_metadata"]
        
        # Apply overrides
        if overrides:
            self._apply_overrides(config, overrides)
        
        # Save configuration
        if isinstance(category, ConfigCategory):
            category = category.value
            
        success = self.save_config(hardware_type, config_name, config, category)
        
        if success:
            return config
        return None
    
    def _apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        """
        Apply overrides to a configuration.
        
        Args:
            config: Configuration to modify
            overrides: Overrides to apply
        """
        for key, value in overrides.items():
            if "." in key:
                # Handle nested keys (e.g., "neuron_params.threshold")
                parts = key.split(".")
                current = config
                
                # Navigate to the nested dictionary
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value
                current[parts[-1]] = value
            else:
                # Simple key
                config[key] = value
    
    def save_config(self, 
                   hardware_type: str, 
                   config_name: str, 
                   config: Dict[str, Any],
                   category: Union[str, ConfigCategory] = ConfigCategory.HARDWARE) -> bool:
        """
        Save a hardware configuration.
        
        Args:
            hardware_type: Hardware type
            config_name: Configuration name
            config: Configuration data
            category: Configuration category
            
        Returns:
            bool: Success status
        """
        try:
            # Validate configuration
            is_valid, errors = self.validate_config(hardware_type, config)
            if not is_valid:
                logger.error(f"Invalid configuration for {hardware_type}: {', '.join(errors)}")
                return False
            
            # Add metadata
            config_with_meta = config.copy()
            config_with_meta["_metadata"] = {
                "hardware_type": hardware_type,
                "name": config_name,
                "category": category.value if isinstance(category, ConfigCategory) else category,
                "created": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            # Determine path
            category_str = category.value if isinstance(category, ConfigCategory) else category
            path = os.path.join(self.config_root, category_str, hardware_type, f"{config_name}.json")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save to file
            with open(path, "w") as f:
                json.dump(config_with_meta, f, indent=2)
                
            logger.info(f"Saved {hardware_type} configuration '{config_name}' in {category_str}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False
    
    def load_config(self, 
                   hardware_type: str, 
                   config_name: str,
                   category: Union[str, ConfigCategory] = ConfigCategory.HARDWARE) -> Optional[Dict[str, Any]]:
        """
        Load a hardware configuration.
        
        Args:
            hardware_type: Hardware type
            config_name: Configuration name
            category: Configuration category
            
        Returns:
            Optional[Dict[str, Any]]: Configuration data or None if not found
        """
        # Determine path
        category_str = category.value if isinstance(category, ConfigCategory) else category
        path = os.path.join(self.config_root, category_str, hardware_type, f"{config_name}.json")
        
        if not os.path.exists(path):
            logger.warning(f"Configuration not found: {category_str}/{hardware_type}/{config_name}")
            return None
            
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return None
    
    def list_configs(self, 
                    hardware_type: Optional[str] = None,
                    category: Union[str, ConfigCategory] = ConfigCategory.HARDWARE) -> Dict[str, List[str]]:
        """
        List available configurations.
        
        Args:
            hardware_type: Optional hardware type filter
            category: Configuration category
            
        Returns:
            Dict[str, List[str]]: Hardware types and their configurations
        """
        result = {}
        
        # Determine category directory
        category_str = category.value if isinstance(category, ConfigCategory) else category
        category_dir = os.path.join(self.config_root, category_str)
        
        if not os.path.exists(category_dir):
            return result
        
        if hardware_type:
            # List configs for specific hardware
            hw_dir = os.path.join(category_dir, hardware_type)
            if os.path.exists(hw_dir):
                configs = [f[:-5] for f in os.listdir(hw_dir) if f.endswith(".json")]
                result[hardware_type] = configs
        else:
            # List all configs
            for hw_type in os.listdir(category_dir):
                hw_dir = os.path.join(category_dir, hw_type)
                if os.path.isdir(hw_dir):
                    configs = [f[:-5] for f in os.listdir(hw_dir) if f.endswith(".json")]
                    result[hw_type] = configs
        
        return result
    
    def delete_config(self, 
                     hardware_type: str, 
                     config_name: str,
                     category: Union[str, ConfigCategory] = ConfigCategory.HARDWARE) -> bool:
        """
        Delete a configuration.
        
        Args:
            hardware_type: Hardware type
            config_name: Configuration name
            category: Configuration category
            
        Returns:
            bool: Success status
        """
        # Determine path
        category_str = category.value if isinstance(category, ConfigCategory) else category
        path = os.path.join(self.config_root, category_str, hardware_type, f"{config_name}.json")
        
        if not os.path.exists(path):
            logger.warning(f"Configuration not found: {category_str}/{hardware_type}/{config_name}")
            return False
            
        try:
            os.remove(path)
            logger.info(f"Deleted configuration: {category_str}/{hardware_type}/{config_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete configuration: {str(e)}")
            return False
    
    def set_active_config(self, 
                         hardware_type: str, 
                         config_name: str,
                         category: Union[str, ConfigCategory] = ConfigCategory.HARDWARE) -> bool:
        """
        Set active configuration for a hardware type.
        
        Args:
            hardware_type: Hardware type
            config_name: Configuration name
            category: Configuration category
            
        Returns:
            bool: Success status
        """
        # Load configuration
        config = self.load_config(hardware_type, config_name, category)
        if not config:
            return False
        
        # Set as active
        self.active_configs[hardware_type] = {
            "name": config_name,
            "category": category.value if isinstance(category, ConfigCategory) else category,
            "config": config
        }
        
        logger.info(f"Set active configuration for {hardware_type}: {config_name}")
        return True
    
    def get_active_config(self, hardware_type: str) -> Optional[Dict[str, Any]]:
        """
        Get active configuration for a hardware type.
        
        Args:
            hardware_type: Hardware type
            
        Returns:
            Optional[Dict[str, Any]]: Active configuration or None if not set
        """
        if hardware_type not in self.active_configs:
            return None
        
        return self.active_configs[hardware_type]["config"]
    
    # Add this to the validate_config method in UnifiedConfigManager class
    
    def validate_config(self, hardware_type: str, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a hardware configuration.
        
        Args:
            hardware_type: Hardware type
            config: Configuration to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        # Basic validation
        if not isinstance(config, dict):
            return False, ["Configuration must be a dictionary"]
        
        # Use the compatibility validator for thorough validation
        from src.core.hardware.compatibility_validator import HardwareCompatibilityValidator
        return HardwareCompatibilityValidator.validate_compatibility(hardware_type, config)
        errors = []
        
        # Hardware-specific validation
        if hardware_type == "loihi":
            if "board_id" not in config:
                errors.append("Missing required field: board_id")
            if "chip_id" not in config:
                errors.append("Missing required field: chip_id")
            if "neuron_params" not in config:
                errors.append("Missing required section: neuron_params")
                
        elif hardware_type == "truenorth":
            if "board_id" not in config:
                errors.append("Missing required field: board_id")
            if "chip_id" not in config:
                errors.append("Missing required field: chip_id")
            if "neuron_params" not in config:
                errors.append("Missing required section: neuron_params")
                
        elif hardware_type == "spinnaker":
            if "board_address" not in config:
                errors.append("Missing required field: board_address")
            if "neuron_params" not in config:
                errors.append("Missing required section: neuron_params")
                
        elif hardware_type == "simulated":
            if "simulator" not in config:
                errors.append("Missing required field: simulator")
            if "neuron_params" not in config:
                errors.append("Missing required section: neuron_params")
        
        return len(errors) == 0, errors
    
    def convert_config(self, 
                      source_hardware: str, 
                      target_hardware: str,
                      source_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert configuration between hardware platforms.
        
        Args:
            source_hardware: Source hardware type
            target_hardware: Target hardware type
            source_config: Source configuration
            
        Returns:
            Dict[str, Any]: Converted configuration
        """
        # Start with a default template for the target hardware
        template = self.get_template(target_hardware, "default")
        if not template:
            # Create an empty configuration if no template exists
            target_config = {"_converted_from": source_hardware}
        else:
            # Remove metadata from template
            target_config = template.copy()
            if "_metadata" in target_config:
                del target_config["_metadata"]
            
            # Add conversion metadata
            target_config["_converted_from"] = source_hardware
        
        # Copy common parameters
        if "simulation" in source_config:
            target_config["simulation"] = source_config["simulation"].copy()
        
        # Hardware-specific conversions
        if source_hardware == "simulated" and target_hardware in ["loihi", "truenorth", "spinnaker"]:
            self._convert_simulated_to_hardware(source_config, target_config, target_hardware)
        elif target_hardware == "simulated":
            self._convert_hardware_to_simulated(source_config, target_config, source_hardware)
        else:
            self._convert_between_hardware(source_config, target_config, source_hardware, target_hardware)
        
        return target_config
    
    def _convert_simulated_to_hardware(self, 
                                      source_config: Dict[str, Any], 
                                      target_config: Dict[str, Any],
                                      target_hardware: str) -> None:
        """
        Convert simulated configuration to hardware configuration.
        
        Args:
            source_config: Source configuration
            target_config: Target configuration to modify
            target_hardware: Target hardware type
        """
        # Copy neuron parameters with appropriate transformations
        if "neuron_params" in source_config:
            source_params = source_config["neuron_params"]
            target_params = target_config.get("neuron_params", {})
            
            # Map common parameters
            if "threshold" in source_params:
                target_params["threshold"] = source_params["threshold"]
            
            if target_hardware == "loihi":
                if "reset" in source_params:
                    target_params["reset_voltage"] = source_params["reset"]
                if "tau_m" in source_params:
                    # Convert time constant to decay
                    target_params["decay"] = 1.0 - (1.0 / source_params["tau_m"])
                    
            elif target_hardware == "truenorth":
                if "reset" in source_params:
                    target_params["reset"] = source_params["reset"]
                    
            elif target_hardware == "spinnaker":
                if "tau_m" in source_params:
                    target_params["tau_m"] = source_params["tau_m"]
                if "reset" in source_params:
                    target_params["v_reset"] = source_params["reset"]
            
            target_config["neuron_params"] = target_params
    
    def _convert_hardware_to_simulated(self, 
                                      source_config: Dict[str, Any], 
                                      target_config: Dict[str, Any],
                                      source_hardware: str) -> None:
        """
        Convert hardware configuration to simulated configuration.
        
        Args:
            source_config: Source configuration
            target_config: Target configuration to modify
            source_hardware: Source hardware type
        """
        # Set simulator based on source hardware
        if source_hardware == "loihi":
            target_config["simulator"] = "loihi_sim"
        elif source_hardware == "truenorth":
            target_config["simulator"] = "truenorth_sim"
        elif source_hardware == "spinnaker":
            target_config["simulator"] = "spinnaker_sim"
        
        # Copy neuron parameters with appropriate transformations
        if "neuron_params" in source_config:
            source_params = source_config["neuron_params"]
            target_params = target_config.get("neuron_params", {})
            
            # Map common parameters
            if "threshold" in source_params:
                target_params["threshold"] = source_params["threshold"]
            
            if source_hardware == "loihi":
                if "reset_voltage" in source_params:
                    target_params["reset"] = source_params["reset_voltage"]
                if "decay" in source_params:
                    # Convert decay to time constant
                    decay = source_params["decay"]
                    if decay < 1.0:
                        target_params["tau_m"] = 1.0 / (1.0 - decay)
                    
            elif source_hardware == "truenorth":
                if "reset" in source_params:
                    target_params["reset"] = source_params["reset"]
                    
            elif source_hardware == "spinnaker":
                if "tau_m" in source_params:
                    target_params["tau_m"] = source_params["tau_m"]
                if "v_reset" in source_params:
                    target_params["reset"] = source_params["v_reset"]
            
            target_config["neuron_params"] = target_params
    
    def _convert_between_hardware(self, 
                                 source_config: Dict[str, Any], 
                                 target_config: Dict[str, Any],
                                 source_hardware: str,
                                 target_hardware: str) -> None:
        """
        Convert between hardware configurations.
        
        Args:
            source_config: Source configuration
            target_config: Target configuration to modify
            source_hardware: Source hardware type
            target_hardware: Target hardware type
        """
        # Copy neuron parameters with appropriate transformations
        if "neuron_params" in source_config:
            source_params = source_config["neuron_params"]
            target_params = target_config.get("neuron_params", {})
            
            # Map common parameters
            if "threshold" in source_params:
                target_params["threshold"] = source_params["threshold"]
            
            # Specific conversions between hardware types
            if source_hardware == "loihi" and target_hardware == "truenorth":
                if "decay" in source_params:
                    # Convert decay to leak
                    target_params["leak"] = int(source_params["decay"] * 255)
                    
            elif source_hardware == "truenorth" and target_hardware == "loihi":
                if "leak" in source_params:
                    # Convert leak to decay
                    target_params["decay"] = float(source_params["leak"]) / 255.0
                    
            elif source_hardware == "loihi" and target_hardware == "spinnaker":
                if "decay" in source_params:
                    # Convert decay to time constant
                    decay = source_params["decay"]
                    if decay < 1.0:
                        target_params["tau_m"] = 1.0 / (1.0 - decay)
                        
            elif source_hardware == "spinnaker" and target_hardware == "loihi":
                if "tau_m" in source_params:
                    # Convert time constant to decay
                    target_params["decay"] = 1.0 - (1.0 / source_params["tau_m"])
            
            target_config["neuron_params"] = target_params