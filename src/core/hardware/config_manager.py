#!/usr/bin/env python3
"""
Hardware Configuration Management System

This module provides tools for managing hardware configurations for neuromorphic processors.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class HardwareConfigManager:
    """
    Manages hardware configurations for neuromorphic processors.
    """
    
    def __init__(self, config_dir: str = "/Users/yessine/Oblivion/configs"):
        """
        Initialize the hardware configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = config_dir
        self.active_configs = {}
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Create subdirectories for different hardware types
        os.makedirs(os.path.join(config_dir, "loihi"), exist_ok=True)
        os.makedirs(os.path.join(config_dir, "truenorth"), exist_ok=True)
        os.makedirs(os.path.join(config_dir, "spinnaker"), exist_ok=True)
        
        logger.info(f"Hardware configuration manager initialized with config directory: {config_dir}")
    
    def save_config(self, hardware_type: str, config_name: str, config: Dict[str, Any]) -> str:
        """
        Save a hardware configuration.
        
        Args:
            hardware_type: Type of hardware (e.g., 'loihi', 'truenorth')
            config_name: Name of the configuration
            config: Configuration dictionary
            
        Returns:
            str: Path to the saved configuration file
        """
        # Create hardware type directory if it doesn't exist
        hardware_dir = os.path.join(self.config_dir, hardware_type.lower())
        os.makedirs(hardware_dir, exist_ok=True)
        
        # Add metadata to the configuration
        config_with_metadata = config.copy()
        config_with_metadata['_metadata'] = {
            'hardware_type': hardware_type,
            'name': config_name,
            'created': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Save as JSON
        config_path = os.path.join(hardware_dir, f"{config_name}.json")
        with open(config_path, 'w') as f:
            json.dump(config_with_metadata, f, indent=2)
        
        logger.info(f"Saved configuration '{config_name}' for {hardware_type} hardware to {config_path}")
        return config_path
    
    def load_config(self, hardware_type: str, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a hardware configuration.
        
        Args:
            hardware_type: Type of hardware (e.g., 'loihi', 'truenorth')
            config_name: Name of the configuration
            
        Returns:
            Optional[Dict[str, Any]]: Configuration dictionary, or None if not found
        """
        hardware_dir = os.path.join(self.config_dir, hardware_type.lower())
        config_path = os.path.join(hardware_dir, f"{config_name}.json")
        
        if not os.path.exists(config_path):
            logger.warning(f"Configuration '{config_name}' for {hardware_type} not found")
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded configuration '{config_name}' for {hardware_type} hardware")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration '{config_name}': {str(e)}")
            return None
    
    def list_configs(self, hardware_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available configurations.
        
        Args:
            hardware_type: Optional type of hardware to list configs for
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping hardware types to lists of config names
        """
        result = {}
        
        if hardware_type:
            # List configs for specific hardware type
            hardware_dir = os.path.join(self.config_dir, hardware_type.lower())
            if os.path.exists(hardware_dir):
                config_files = [f[:-5] for f in os.listdir(hardware_dir) if f.endswith('.json')]
                result[hardware_type] = config_files
        else:
            # List configs for all hardware types
            for hw_type in os.listdir(self.config_dir):
                hardware_dir = os.path.join(self.config_dir, hw_type)
                if os.path.isdir(hardware_dir):
                    config_files = [f[:-5] for f in os.listdir(hardware_dir) if f.endswith('.json')]
                    result[hw_type] = config_files
        
        return result
    
    def delete_config(self, hardware_type: str, config_name: str) -> bool:
        """
        Delete a hardware configuration.
        
        Args:
            hardware_type: Type of hardware
            config_name: Name of the configuration
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        hardware_dir = os.path.join(self.config_dir, hardware_type.lower())
        config_path = os.path.join(hardware_dir, f"{config_name}.json")
        
        if not os.path.exists(config_path):
            logger.warning(f"Configuration '{config_name}' for {hardware_type} not found")
            return False
        
        try:
            os.remove(config_path)
            logger.info(f"Deleted configuration '{config_name}' for {hardware_type} hardware")
            return True
        except Exception as e:
            logger.error(f"Failed to delete configuration '{config_name}': {str(e)}")
            return False
    
    def set_active_config(self, hardware_type: str, config_name: str) -> bool:
        """
        Set a configuration as active for a hardware type.
        
        Args:
            hardware_type: Type of hardware
            config_name: Name of the configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        config = self.load_config(hardware_type, config_name)
        if not config:
            return False
        
        self.active_configs[hardware_type] = {
            'name': config_name,
            'config': config
        }
        
        logger.info(f"Set '{config_name}' as active configuration for {hardware_type} hardware")
        return True
    
    def get_active_config(self, hardware_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the active configuration for a hardware type.
        
        Args:
            hardware_type: Type of hardware
            
        Returns:
            Optional[Dict[str, Any]]: Active configuration, or None if not set
        """
        if hardware_type not in self.active_configs:
            logger.warning(f"No active configuration set for {hardware_type} hardware")
            return None
        
        return self.active_configs[hardware_type]['config']
    
    def create_default_configs(self):
        """Create default configurations for supported hardware types."""
        # Default Loihi configuration
        loihi_default = {
            'board_id': 0,
            'chip_id': 0,
            'connection_type': 'local',
            'neuron_params': {
                'threshold': 1.0,
                'decay': 0.5,
                'compartment_type': 'LIF'
            },
            'simulation': {
                'timestep_ms': 1.0,
                'max_steps': 1000
            }
        }
        self.save_config('loihi', 'default', loihi_default)
        
        # Default TrueNorth configuration
        truenorth_default = {
            'board_id': 0,
            'chip_id': 0,
            'connection_type': 'local',
            'neuron_params': {
                'threshold': 1.0,
                'leak': 0,
                'reset_mode': 'zero'
            },
            'simulation': {
                'timestep_ms': 1.0,
                'max_steps': 1000
            }
        }
        self.save_config('truenorth', 'default', truenorth_default)
        
        # Default SpiNNaker configuration
        spinnaker_default = {
            'board_address': '192.168.1.1',
            'connection_type': 'ethernet',
            'neuron_params': {
                'model': 'IF_curr_exp',
                'threshold': 1.0,
                'tau_m': 20.0
            },
            'simulation': {
                'timestep_ms': 1.0,
                'max_steps': 1000
            }
        }
        self.save_config('spinnaker', 'default', spinnaker_default)
    
    def clone_config(self, hardware_type: str, source_name: str, target_name: str) -> bool:
        """
        Clone a configuration.
        
        Args:
            hardware_type: Type of hardware
            source_name: Name of the source configuration
            target_name: Name of the target configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        config = self.load_config(hardware_type, source_name)
        if not config:
            return False
        
        # Update metadata
        if '_metadata' in config:
            config['_metadata']['name'] = target_name
            config['_metadata']['created'] = datetime.now().isoformat()
        
        # Save as new configuration
        self.save_config(hardware_type, target_name, config)
        return True
    
    def export_config(self, hardware_type: str, config_name: str, export_path: str) -> bool:
        """
        Export a configuration to a file.
        
        Args:
            hardware_type: Type of hardware
            config_name: Name of the configuration
            export_path: Path to export the configuration to
            
        Returns:
            bool: True if successful, False otherwise
        """
        config = self.load_config(hardware_type, config_name)
        if not config:
            return False
        
        try:
            # Determine format based on file extension
            _, ext = os.path.splitext(export_path)
            
            if ext.lower() == '.json':
                with open(export_path, 'w') as f:
                    json.dump(config, f, indent=2)
            elif ext.lower() in ['.yaml', '.yml']:
                with open(export_path, 'w') as f:
                    yaml.dump(config, f)
            else:
                logger.error(f"Unsupported export format: {ext}")
                return False
            
            logger.info(f"Exported configuration '{config_name}' to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export configuration: {str(e)}")
            return False
    
    def import_config(self, hardware_type: str, config_name: str, import_path: str) -> bool:
        """
        Import a configuration from a file.
        
        Args:
            hardware_type: Type of hardware
            config_name: Name to give the imported configuration
            import_path: Path to import the configuration from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Determine format based on file extension
            _, ext = os.path.splitext(import_path)
            
            if ext.lower() == '.json':
                with open(import_path, 'r') as f:
                    config = json.load(f)
            elif ext.lower() in ['.yaml', '.yml']:
                with open(import_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported import format: {ext}")
                return False
            
            # Save the imported configuration
            self.save_config(hardware_type, config_name, config)
            
            logger.info(f"Imported configuration from {import_path} as '{config_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to import configuration: {str(e)}")
            return False