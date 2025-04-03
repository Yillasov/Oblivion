#!/usr/bin/env python3
"""
Hardware Abstraction Registry

Provides a central registry for hardware abstractions, enabling dynamic
hardware discovery and management.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Type, Callable
import threading
import importlib

from src.core.utils.logging_framework import get_logger
from src.core.hardware.hardware_abstraction import NeuromorphicHardware
from src.core.hardware.hardware_detection import HardwareDetector
from src.core.hardware.exceptions import NeuromorphicHardwareError

logger = get_logger("hardware_registry")


class HardwareRegistry:
    """
    Central registry for neuromorphic hardware abstractions.
    
    Provides dynamic discovery and management of hardware implementations.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(HardwareRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize hardware registry."""
        if self._initialized:
            return
            
        self._hardware_types = {}
        self._hardware_instances = {}
        self._detector = HardwareDetector()
        self._initialized = True
        
        # Register built-in hardware types
        self.register_hardware_type("loihi", "src.core.hardware.hardware_abstraction", "LoihiHardware")
        self.register_hardware_type("spinnaker", "src.core.hardware.hardware_abstraction", "SpiNNakerHardware")
        self.register_hardware_type("truenorth", "src.core.hardware.truenorth_hardware", "TrueNorthHardware")
        self.register_hardware_type("simulated", "src.core.hardware.simulated_hardware", "SimulatedHardware")
    
    def __getitem__(self, hardware_type: str) -> Optional[NeuromorphicHardware]:
        """
        Get hardware instance using dictionary-style access.
        
        Args:
            hardware_type: Hardware type identifier
            
        Returns:
            Optional[NeuromorphicHardware]: Hardware instance
            
        Raises:
            KeyError: If hardware type is not found
        """
        hardware = self.get_hardware(hardware_type)
        if hardware is None:
            raise KeyError(f"Hardware type not found: {hardware_type}")
        return hardware
    
    def register_hardware_type(self, hardware_type: str, module_path: str, class_name: str) -> bool:
        """
        Register a hardware type with the registry.
        
        Args:
            hardware_type: Hardware type identifier
            module_path: Python module path containing the hardware class
            class_name: Name of the hardware class
            
        Returns:
            bool: Success status
        """
        try:
            self._hardware_types[hardware_type] = (module_path, class_name)
            logger.info(f"Registered hardware type: {hardware_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to register hardware type {hardware_type}: {str(e)}")
            return False
    
    def get_hardware(self, hardware_type: str) -> Optional[NeuromorphicHardware]:
        """
        Get hardware instance by type.
        
        Args:
            hardware_type: Hardware type identifier
            
        Returns:
            Optional[NeuromorphicHardware]: Hardware instance or None if not found
        """
        # Return cached instance if available
        if hardware_type in self._hardware_instances:
            return self._hardware_instances[hardware_type]
        
        # Create new instance
        if hardware_type in self._hardware_types:
            try:
                module_path, class_name = self._hardware_types[hardware_type]
                module = importlib.import_module(module_path)
                hardware_class = getattr(module, class_name)
                
                hardware = hardware_class()
                self._hardware_instances[hardware_type] = hardware
                
                logger.info(f"Created hardware instance: {hardware_type}")
                return hardware
            except Exception as e:
                logger.error(f"Failed to create hardware instance {hardware_type}: {str(e)}")
                return None
        
        logger.warning(f"Unknown hardware type: {hardware_type}")
        return None
    
    def discover_hardware(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover available hardware.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of detected hardware
        """
        return self._detector.detect_hardware()
    
    def get_available_hardware_types(self) -> List[str]:
        """
        Get list of available hardware types.
        
        Returns:
            List[str]: List of hardware type identifiers
        """
        return list(self._hardware_types.keys())
    
    def get_best_hardware(self) -> Optional[NeuromorphicHardware]:
        """
        Get best available hardware.
        
        Returns:
            Optional[NeuromorphicHardware]: Best hardware instance or None if none available
        """
        detected = self._detector.detect_hardware()
        if detected:
            # Get the first available hardware type
            for hw_type in self.get_available_hardware_types():
                if hw_type in detected:
                    return self.get_hardware(hw_type)
        return None


# Global registry instance
hardware_registry = HardwareRegistry()


def get_hardware(hardware_type: str) -> Optional[NeuromorphicHardware]:
    """
    Get hardware instance by type.
    
    Args:
        hardware_type: Hardware type identifier
        
    Returns:
        Optional[NeuromorphicHardware]: Hardware instance or None if not found
    """
    return hardware_registry.get_hardware(hardware_type)


def discover_hardware() -> Dict[str, Dict[str, Any]]:
    """
    Discover available hardware.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of detected hardware
    """
    return hardware_registry.discover_hardware()


def get_best_hardware() -> Optional[NeuromorphicHardware]:
    """
    Get best available hardware.
    
    Returns:
        Optional[NeuromorphicHardware]: Best hardware instance or None if none available
    """
    return hardware_registry.get_best_hardware()