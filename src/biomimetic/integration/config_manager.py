"""
Configuration management for biomimetic parameters.

Provides loading, saving, and validation of biomimetic configurations.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Union

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger

logger = get_logger("biomimetic_config")

class BiomimeticConfigManager:
    """Configuration manager for biomimetic parameters."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the biomimetic configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or os.path.join(
            project_root, "configs", "biomimetic", "default_config.json"
        )
        self.config_history: List[Dict[str, Any]] = []
        self.current_config: Dict[str, Any] = {}
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        logger.info(f"Biomimetic config manager initialized with path: {self.config_path}")
    
    def load_configuration(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.current_config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                self.current_config = self._create_default_config()
                logger.info("Created default configuration")
            
            return self.current_config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.current_config = self._create_default_config()
            return self.current_config
    
    def save_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Success status
        """
        try:
            # Add timestamp
            config["timestamp"] = time.time()
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Add to history
            self.config_history.append(self.current_config)
            self.current_config = config
            
            logger.info(f"Saved configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Default configuration dictionary
        """
        default_config = {
            "timestamp": time.time(),
            "version": "1.0",
            "power": {
                "state": "balanced",
                "energy_harvesting": False,
                "adaptive_distribution": True
            },
            "hardware": {
                "wing_flapping": {
                    "frequency": 2.0,
                    "amplitude": 0.5
                }
            },
            "sensors": {
                "proprioceptive": True,
                "exteroceptive": True
            },
            "biomimetic_parameters": {
                "organic_form_factor": 0.5,
                "surface_complexity": 0.3,
                "asymmetry_factor": 0.1,
                "morphing_capability": 0.2
            },
            "active_principles": [
                "FORM_FOLLOWS_FUNCTION",
                "ADAPTIVE_MORPHOLOGY",
                "SENSORY_INTEGRATION"
            ]
        }
        
        return default_config
    
    def get_parameter(self, param_path: str, default_value: Any = None) -> Any:
        """
        Get a parameter from the configuration.
        
        Args:
            param_path: Parameter path (dot notation)
            default_value: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        if not self.current_config:
            self.load_configuration()
        
        # Split path into components
        components = param_path.split('.')
        
        # Navigate through config
        current = self.current_config
        for component in components:
            if component in current:
                current = current[component]
            else:
                return default_value
        
        return current
    
    def set_parameter(self, param_path: str, value: Any) -> bool:
        """
        Set a parameter in the configuration.
        
        Args:
            param_path: Parameter path (dot notation)
            value: Parameter value
            
        Returns:
            Success status
        """
        if not self.current_config:
            self.load_configuration()
        
        # Split path into components
        components = param_path.split('.')
        
        # Navigate through config
        current = self.current_config
        for i, component in enumerate(components[:-1]):
            if component not in current:
                current[component] = {}
            current = current[component]
        
        # Set value
        current[components[-1]] = value
        
        return True
    
    def revert_to_previous(self) -> Dict[str, Any]:
        """
        Revert to previous configuration.
        
        Returns:
            Previous configuration
        """
        if not self.config_history:
            logger.warning("No configuration history available")
            return self.current_config
        
        previous_config = self.config_history.pop()
        self.save_configuration(previous_config)
        
        return previous_config