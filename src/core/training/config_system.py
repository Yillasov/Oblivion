#!/usr/bin/env python3
"""
Training Configuration System

Provides a comprehensive yet simple system for managing training configurations
for neuromorphic hardware.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import json
import copy
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict, field

from src.core.utils.logging_framework import get_logger
from src.core.training.trainer_base import TrainingConfig, TrainingMode

logger = get_logger("training_config")

class TrainingConfigSystem:
    """
    System for managing training configurations.
    
    This class provides utilities for creating, loading, saving, and managing
    training configurations for different hardware types and use cases.
    """
    
    def __init__(self, config_dir: str = "/Users/yessine/Oblivion/configs/training"):
        """
        Initialize the training configuration system.
        
        Args:
            config_dir: Directory to store training configurations
        """
        self.config_dir = config_dir
        self.templates = {}
        self.active_configs = {}
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Create hardware-specific directories
        for hw_type in ["loihi", "truenorth", "spinnaker", "simulated"]:
            os.makedirs(os.path.join(config_dir, hw_type), exist_ok=True)
        
        # Load default templates
        self._initialize_templates()
        
        logger.info(f"Initialized training configuration system at {config_dir}")
    
    def _initialize_templates(self) -> None:
        """Initialize default configuration templates for each hardware type."""
        # Base template with common settings
        base_template = TrainingConfig(
            learning_rate=0.01,
            batch_size=32,
            epochs=10,
            mode=TrainingMode.ONLINE,
            checkpoint_interval=5,
            early_stopping=True,
            patience=5,
            validation_split=0.2,
            shuffle=True,
            optimizer="adam",
            optimizer_params={"beta1": 0.9, "beta2": 0.999}
        )
        
        # Hardware-specific templates
        self.templates["loihi"] = TrainingConfig(
            **asdict(base_template),
            hardware_type="loihi",
            batch_size=16,  # Smaller batch size for Loihi
            custom_params={"weight_precision": 8, "delay_encoding": True}
        )
        
        self.templates["truenorth"] = TrainingConfig(
            **asdict(base_template),
            hardware_type="truenorth",
            learning_rate=0.005,  # Lower learning rate for TrueNorth
            custom_params={"binary_activations": True}
        )
        
        self.templates["spinnaker"] = TrainingConfig(
            **asdict(base_template),
            hardware_type="spinnaker",
            mode=TrainingMode.HYBRID,
            custom_params={"event_driven": True}
        )
        
        self.templates["simulated"] = TrainingConfig(
            **asdict(base_template),
            hardware_type="simulated",
            batch_size=64,  # Larger batch size for simulation
            checkpoint_interval=10
        )
        
        # Save templates to disk
        for hw_type, config in self.templates.items():
            template_path = os.path.join(self.config_dir, hw_type, "template.json")
            self.save_config(config, template_path)
    
    def create_config(self, hardware_type: str, name: str, 
                     overrides: Optional[Dict[str, Any]] = None) -> TrainingConfig:
        """
        Create a new training configuration based on a template.
        
        Args:
            hardware_type: Hardware type ('loihi', 'truenorth', 'spinnaker', 'simulated')
            name: Configuration name
            overrides: Optional parameter overrides
            
        Returns:
            TrainingConfig: New training configuration
        """
        # Get template for hardware type
        if hardware_type not in self.templates:
            logger.warning(f"Unknown hardware type: {hardware_type}, using 'simulated'")
            hardware_type = "simulated"
        
        # Create a copy of the template
        config_dict = asdict(self.templates[hardware_type])
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if key in config_dict:
                    config_dict[key] = value
                elif key in config_dict.get("custom_params", {}):
                    config_dict["custom_params"][key] = value
                else:
                    config_dict["custom_params"][key] = value
        
        # Create new config
        config = TrainingConfig(**config_dict)
        
        # Save config
        config_path = os.path.join(self.config_dir, hardware_type, f"{name}.json")
        self.save_config(config, config_path)
        
        # Add to active configs
        self.active_configs[name] = config
        
        logger.info(f"Created training configuration: {name} for {hardware_type}")
        return config
    
    def load_config(self, name: str, hardware_type: Optional[str] = None) -> Optional[TrainingConfig]:
        """
        Load a training configuration.
        
        Args:
            name: Configuration name
            hardware_type: Optional hardware type (searches all types if None)
            
        Returns:
            Optional[TrainingConfig]: Loaded configuration or None if not found
        """
        # Check if already loaded
        if name in self.active_configs:
            return self.active_configs[name]
        
        # Determine path
        if hardware_type:
            config_path = os.path.join(self.config_dir, hardware_type, f"{name}.json")
            if os.path.exists(config_path):
                return self._load_config_from_file(config_path)
        else:
            # Search in all hardware types
            for hw_type in ["loihi", "truenorth", "spinnaker", "simulated"]:
                config_path = os.path.join(self.config_dir, hw_type, f"{name}.json")
                if os.path.exists(config_path):
                    return self._load_config_from_file(config_path)
        
        logger.warning(f"Configuration not found: {name}")
        return None
    
    def _load_config_from_file(self, path: str) -> Optional[TrainingConfig]:
        """Load configuration from file."""
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            
            # Create config object
            config = TrainingConfig(**config_dict)
            
            # Add to active configs
            self.active_configs[os.path.basename(path).split('.')[0]] = config
            
            logger.info(f"Loaded training configuration from {path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration from {path}: {str(e)}")
            return None
    
    def save_config(self, config: TrainingConfig, path: Optional[str] = None) -> bool:
        """
        Save a training configuration.
        
        Args:
            config: Training configuration
            path: Optional path (uses default if None)
            
        Returns:
            bool: Success status
        """
        try:
            # Convert to dictionary
            config_dict = asdict(config)
            
            # Determine path if not provided
            if not path:
                name = next((name for name, cfg in self.active_configs.items() 
                           if cfg == config), "config")
                path = os.path.join(self.config_dir, config.hardware_type, f"{name}.json")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save to file
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Saved training configuration to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def list_configs(self, hardware_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available configurations.
        
        Args:
            hardware_type: Optional hardware type filter
            
        Returns:
            Dict[str, List[str]]: Dictionary of hardware types and their configurations
        """
        result = {}
        
        # Determine hardware types to search
        if hardware_type:
            hw_types = [hardware_type]
        else:
            hw_types = ["loihi", "truenorth", "spinnaker", "simulated"]
        
        # Search for configurations
        for hw_type in hw_types:
            hw_dir = os.path.join(self.config_dir, hw_type)
            if os.path.exists(hw_dir):
                configs = [f.split('.')[0] for f in os.listdir(hw_dir) 
                          if f.endswith('.json') and f != "template.json"]
                result[hw_type] = configs
        
        return result
    
    def get_template(self, hardware_type: str) -> TrainingConfig:
        """
        Get a template configuration for a hardware type.
        
        Args:
            hardware_type: Hardware type
            
        Returns:
            TrainingConfig: Template configuration
        """
        if hardware_type in self.templates:
            return copy.deepcopy(self.templates[hardware_type])
        else:
            logger.warning(f"Unknown hardware type: {hardware_type}, using 'simulated'")
            return copy.deepcopy(self.templates["simulated"])
    
    def update_config(self, name: str, updates: Dict[str, Any], 
                     hardware_type: Optional[str] = None) -> Optional[TrainingConfig]:
        """
        Update an existing configuration.
        
        Args:
            name: Configuration name
            updates: Parameter updates
            hardware_type: Optional hardware type
            
        Returns:
            Optional[TrainingConfig]: Updated configuration or None if not found
        """
        # Load existing config
        config = self.load_config(name, hardware_type)
        if not config:
            return None
        
        # Apply updates
        config_dict = asdict(config)
        for key, value in updates.items():
            if key in config_dict:
                config_dict[key] = value
            elif key in config_dict.get("custom_params", {}):
                config_dict["custom_params"][key] = value
            else:
                config_dict["custom_params"][key] = value
        
        # Create updated config
        updated_config = TrainingConfig(**config_dict)
        
        # Save updated config
        if hardware_type:
            config_path = os.path.join(self.config_dir, hardware_type, f"{name}.json")
        else:
            config_path = os.path.join(self.config_dir, updated_config.hardware_type, f"{name}.json")
        
        self.save_config(updated_config, config_path)
        
        # Update active configs
        self.active_configs[name] = updated_config
        
        logger.info(f"Updated training configuration: {name}")
        return updated_config


# Helper function to create a configuration system
def create_config_system(config_dir: Optional[str] = None) -> TrainingConfigSystem:
    """
    Create a training configuration system.
    
    Args:
        config_dir: Optional configuration directory
        
    Returns:
        TrainingConfigSystem: Configuration system instance
    """
    if config_dir:
        return TrainingConfigSystem(config_dir)
    else:
        return TrainingConfigSystem()