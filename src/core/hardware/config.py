"""
Configuration utilities for neuromorphic hardware.
"""

from typing import Dict, Any, Optional
import json
import os
import yaml


class HardwareConfig:
    """
    Utility class for managing neuromorphic hardware configurations.
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load hardware configuration from a file.
        
        Args:
            config_path: Path to the configuration file (JSON or YAML)
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        _, ext = os.path.splitext(config_path)
        
        if ext.lower() in ['.json']:
            with open(config_path, 'r') as f:
                return json.load(f)
        elif ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """
        Save hardware configuration to a file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save the configuration file
        """
        _, ext = os.path.splitext(config_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        if ext.lower() in ['.json']:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries, with override_config taking precedence.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary
            
        Returns:
            Dict[str, Any]: Merged configuration dictionary
        """
        result = base_config.copy()
        
        for key, value in override_config.items():
            if (
                key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)
            ):
                result[key] = HardwareConfig.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result