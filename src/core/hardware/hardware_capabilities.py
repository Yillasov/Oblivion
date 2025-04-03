#!/usr/bin/env python3
"""
Hardware Capabilities Discovery System

Provides functionality to discover and report capabilities of neuromorphic hardware.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import socket
import json
import subprocess
from typing import Dict, Any, List, Optional

from src.core.utils.logging_framework import get_logger

logger = get_logger("hardware_capabilities")


class HardwareCapabilitiesDiscovery:
    """Discovers capabilities of neuromorphic hardware devices."""
    
    # Default capabilities for known hardware types
    DEFAULT_CAPABILITIES = {
        "loihi": {
            "neurons_per_core": 1024,
            "cores_per_chip": 128,
            "on_chip_learning": True,
            "spike_based": True,
            "supports_stdp": True,
            "max_fan_in": 4096,
            "max_fan_out": 4096,
            "supports_recurrence": True,
            "neuron_models": ["LIF", "AdEx", "Custom"],
            "precision": "8-bit",
            "clock_speed_mhz": 100
        },
        "truenorth": {
            "neurons_per_core": 256,
            "cores_per_chip": 4096,
            "on_chip_learning": False,
            "spike_based": True,
            "supports_stdp": False,
            "max_fan_in": 256,
            "max_fan_out": 256,
            "supports_recurrence": True,
            "neuron_models": ["LIF"],
            "precision": "1-bit",
            "clock_speed_mhz": 1000
        },
        "spinnaker": {
            "neurons_per_core": 1000,
            "cores_per_chip": 18,
            "on_chip_learning": True,
            "spike_based": True,
            "supports_stdp": True,
            "max_fan_in": 10000,
            "max_fan_out": 10000,
            "supports_recurrence": True,
            "neuron_models": ["LIF", "Izhikevich", "Custom"],
            "precision": "16-bit",
            "clock_speed_mhz": 200
        }
    }
    
    def __init__(self):
        """Initialize the hardware capabilities discovery system."""
        self.hardware_capabilities = {}
    
    def discover_capabilities(self, hardware_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discover capabilities of a hardware device.
        
        Args:
            hardware_info: Information about the hardware device
            
        Returns:
            Dict[str, Any]: Hardware capabilities
        """
        hw_type = hardware_info.get("type", "unknown")
        hw_id = hardware_info.get("device_id", 0)
        
        # Check if we already discovered capabilities for this hardware
        cache_key = f"{hw_type}_{hw_id}"
        if cache_key in self.hardware_capabilities:
            return self.hardware_capabilities[cache_key]
        
        # Start with default capabilities for this hardware type
        capabilities = self.DEFAULT_CAPABILITIES.get(hw_type, {}).copy()
        
        # Add basic hardware info
        capabilities.update({
            "hardware_type": hw_type,
            "device_id": hw_id,
            "connection_type": hardware_info.get("connection_type", "unknown"),
            "platform": hardware_info.get("platform", "unknown")
        })
        
        # Try to discover actual capabilities from the hardware
        if hardware_info.get("connection_type") == "network":
            self._discover_network_capabilities(hardware_info, capabilities)
        elif hardware_info.get("connection_type") == "usb":
            self._discover_usb_capabilities(hardware_info, capabilities)
        
        # Cache the discovered capabilities
        self.hardware_capabilities[cache_key] = capabilities
        
        return capabilities
    
    def _discover_network_capabilities(self, hardware_info: Dict[str, Any], capabilities: Dict[str, Any]):
        """
        Discover capabilities of network-connected hardware.
        
        Args:
            hardware_info: Hardware information
            capabilities: Capabilities dictionary to update
        """
        try:
            address = hardware_info.get("address", "127.0.0.1")
            port = hardware_info.get("port", 22222)
            
            # Try to connect and query capabilities
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            result = s.connect_ex((address, port))
            
            if result == 0:
                # Send capability query command
                s.sendall(b'{"command": "get_capabilities"}\n')
                
                # Receive response
                data = s.recv(4096)
                if data:
                    try:
                        response = json.loads(data.decode('utf-8'))
                        if "capabilities" in response:
                            capabilities.update(response["capabilities"])
                            logger.info(f"Discovered capabilities from network device: {address}:{port}")
                    except json.JSONDecodeError:
                        pass
            
            s.close()
        except Exception as e:
            logger.debug(f"Network capability discovery error: {str(e)}")
    
    def _discover_usb_capabilities(self, hardware_info: Dict[str, Any], capabilities: Dict[str, Any]):
        """
        Discover capabilities of USB-connected hardware.
        
        Args:
            hardware_info: Hardware information
            capabilities: Capabilities dictionary to update
        """
        try:
            hw_type = hardware_info.get("type", "unknown")
            
            if hw_type == "loihi":
                # For Loihi, try to use nxsdk to query capabilities
                try:
                    # This would normally use the nxsdk Python API
                    # Here we're just simulating the discovery
                    capabilities.update({
                        "board_version": "Kapoho Bay" if "kapoho" in str(hardware_info).lower() else "Nahuku",
                        "chips_available": 8,
                        "total_cores": capabilities.get("cores_per_chip", 128) * 8,
                        "total_neurons": capabilities.get("neurons_per_core", 1024) * capabilities.get("cores_per_chip", 128) * 8
                    })
                except ImportError:
                    pass
            
            elif hw_type == "truenorth":
                # For TrueNorth, update with board-specific info
                capabilities.update({
                    "board_version": "NS1e",
                    "chips_available": 1,
                    "total_cores": capabilities.get("cores_per_chip", 4096),
                    "total_neurons": capabilities.get("neurons_per_core", 256) * capabilities.get("cores_per_chip", 4096)
                })
            
            elif hw_type == "spinnaker":
                # For SpiNNaker, update with board-specific info
                capabilities.update({
                    "board_version": "SpiNN-5",
                    "chips_available": 48,
                    "total_cores": capabilities.get("cores_per_chip", 18) * 48,
                    "total_neurons": capabilities.get("neurons_per_core", 1000) * capabilities.get("cores_per_chip", 18) * 48
                })
        
        except Exception as e:
            logger.debug(f"USB capability discovery error: {str(e)}")
    
    def get_all_capabilities(self, hardware_devices: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Get capabilities for all hardware devices.
        
        Args:
            hardware_devices: Dictionary of hardware devices
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of hardware capabilities
        """
        result = {}
        
        for hw_id, hw_info in hardware_devices.items():
            result[hw_id] = self.discover_capabilities(hw_info)
        
        return result
    
    def check_capability_support(self, capabilities: Dict[str, Any], feature: str) -> bool:
        """
        Check if a specific capability/feature is supported.
        
        Args:
            capabilities: Hardware capabilities
            feature: Feature to check
            
        Returns:
            bool: True if feature is supported
        """
        if feature in capabilities:
            return bool(capabilities[feature])
        
        # Check for specific features
        if feature == "learning":
            return capabilities.get("on_chip_learning", False)
        elif feature == "recurrent_connections":
            return capabilities.get("supports_recurrence", False)
        elif feature == "custom_neurons":
            return "Custom" in capabilities.get("neuron_models", [])
        
        return False
    
    def get_capability_value(self, capabilities: Dict[str, Any], capability: str, default=None) -> Any:
        """
        Get the value of a specific capability.
        
        Args:
            capabilities: Hardware capabilities
            capability: Capability to retrieve
            default: Default value if capability not found
            
        Returns:
            Any: Capability value
        """
        return capabilities.get(capability, default)


# Create a singleton instance
capabilities_discovery = HardwareCapabilitiesDiscovery()