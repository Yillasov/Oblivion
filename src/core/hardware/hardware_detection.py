"""
Hardware Detection Module

Provides functionality to automatically detect and configure neuromorphic hardware.
"""

import os
import platform
import socket
import json
import subprocess
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.core.utils.logging_framework import get_logger

logger = get_logger("hardware_detection")


class HardwareDetector:
    """Detects available neuromorphic hardware with enhanced reliability."""
    
    # Known hardware identifiers
    LOIHI_IDENTIFIERS = ["loihi", "nxsdk", "kapoho", "nahuku"]
    TRUENORTH_IDENTIFIERS = ["truenorth", "tn", "ibm-tn", "tn-board"]
    SPINNAKER_IDENTIFIERS = ["spinnaker", "spin", "spinn", "spinn-board"]
    
    # Default ports for hardware communication
    DEFAULT_PORTS = {
        "loihi": 22222,
        "truenorth": 54321,
        "spinnaker": 17893
    }
    
    # Common hardware addresses
    COMMON_ADDRESSES = {
        "loihi": ["127.0.0.1", "192.168.1.10", "10.0.0.10"],
        "truenorth": ["127.0.0.1", "192.168.1.20", "10.0.0.20"],
        "spinnaker": ["192.168.1.1", "192.168.240.1", "10.0.0.30"]
    }
    
    def __init__(self):
        """Initialize the hardware detector."""
        self.detected_hardware = {}
        self.detection_attempts = 0
        self.config_path = Path(os.path.expanduser("~/.oblivion/hardware_config.json"))
        self.current_platform = platform.system().lower()
    
    def detect_hardware(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect available neuromorphic hardware with enhanced reliability.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of detected hardware
        """
        logger.info("Detecting available neuromorphic hardware")
        
        # Clear previous detections
        self.detected_hardware = {}
        self.detection_attempts = 0
        
        # Try to load cached hardware configuration
        self._try_load_cached_config()
        
        # Primary detection methods
        self._detect_via_environment()
        self._detect_via_network()
        
        # Platform-specific detection methods
        self._detect_platform_specific()
        
        # If no hardware detected, try fallback methods
        if not self.detected_hardware and self.detection_attempts < 3:
            logger.info("Primary detection failed, trying fallback methods...")
            self._detect_via_fallback()
        
        if not self.detected_hardware:
            logger.warning("No neuromorphic hardware detected")
        else:
            logger.info(f"Detected {len(self.detected_hardware)} neuromorphic hardware devices")
            # Cache successful detection
            self._cache_detection_results()
            
        return self.detected_hardware
    
    def _detect_via_environment(self):
        """Detect hardware through environment variables."""
        # Check for multiple hardware instances in environment variables
        hardware_counts = {"loihi": 0, "truenorth": 0, "spinnaker": 0}
        
        # Check for Loihi
        for env_var in os.environ:
            # Check for Loihi hardware
            if any(id_str in env_var.lower() for id_str in self.LOIHI_IDENTIFIERS):
                hw_id = f"loihi_{hardware_counts['loihi']}"
                logger.info(f"Detected Loihi hardware via environment variables: {hw_id}")
                self.detected_hardware[hw_id] = {
                    "type": "loihi",
                    "connection_type": "usb" if "USB" in env_var else "network",
                    "address": os.environ.get(env_var, "127.0.0.1"),
                    "device_id": hardware_counts["loihi"]
                }
                hardware_counts["loihi"] += 1
                
            # Check for TrueNorth
            elif any(id_str in env_var.lower() for id_str in self.TRUENORTH_IDENTIFIERS):
                hw_id = f"truenorth_{hardware_counts['truenorth']}"
                logger.info(f"Detected TrueNorth hardware via environment variables: {hw_id}")
                self.detected_hardware[hw_id] = {
                    "type": "truenorth",
                    "connection_type": "network",
                    "address": os.environ.get(env_var, "127.0.0.1"),
                    "device_id": hardware_counts["truenorth"]
                }
                hardware_counts["truenorth"] += 1
                
            # Check for SpiNNaker
            elif any(id_str in env_var.lower() for id_str in self.SPINNAKER_IDENTIFIERS):
                hw_id = f"spinnaker_{hardware_counts['spinnaker']}"
                logger.info(f"Detected SpiNNaker hardware via environment variables: {hw_id}")
                self.detected_hardware[hw_id] = {
                    "type": "spinnaker",
                    "connection_type": "ethernet",
                    "address": os.environ.get(env_var, "192.168.1.1"),
                    "device_id": hardware_counts["spinnaker"]
                }
                hardware_counts["spinnaker"] += 1
        
        self.detection_attempts += 1
    
    def _detect_via_network(self):
        """Detect hardware through network connections."""
        # Try to connect to common hardware addresses
        hardware_counts = {"loihi": 0, "truenorth": 0, "spinnaker": 0}
        
        for hw_type, addresses in self.COMMON_ADDRESSES.items():
            port = self.DEFAULT_PORTS.get(hw_type, 0)
            
            for address in addresses:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(0.5)
                    result = s.connect_ex((address, port))
                    s.close()
                    
                    if result == 0:
                        hw_id = f"{hw_type}_{hardware_counts[hw_type]}"
                        logger.info(f"Detected {hw_type} hardware at {address}:{port} - {hw_id}")
                        self.detected_hardware[hw_id] = {
                            "type": hw_type,
                            "connection_type": "network",
                            "address": address,
                            "port": port,
                            "device_id": hardware_counts[hw_type]
                        }
                        hardware_counts[hw_type] += 1
                except:
                    pass
        
        self.detection_attempts += 1
    
    def _detect_platform_specific(self):
        """Run platform-specific hardware detection."""
        if self.current_platform == "darwin":  # macOS
            self._detect_macos()
        elif self.current_platform == "linux":
            self._detect_linux()
        elif self.current_platform == "windows":
            self._detect_windows()
        
        self.detection_attempts += 1
    
    def _detect_macos(self):
        """macOS-specific hardware detection."""
        hardware_counts = {"loihi": 0, "truenorth": 0, "spinnaker": 0}
        
        # Check for Loihi via USB on macOS
        try:
            result = subprocess.run(
                ["system_profiler", "SPUSBDataType"], 
                capture_output=True, 
                text=True,
                timeout=2
            )
            
            output = result.stdout.lower()
            
            # Count occurrences of Loihi identifiers to detect multiple devices
            for line in output.split('\n'):
                if any(id_str in line for id_str in self.LOIHI_IDENTIFIERS):
                    hw_id = f"loihi_{hardware_counts['loihi']}"
                    logger.info(f"Detected Loihi hardware via macOS USB: {hw_id}")
                    self.detected_hardware[hw_id] = {
                        "type": "loihi",
                        "connection_type": "usb",
                        "board_id": hardware_counts["loihi"],
                        "platform": "macos",
                        "device_id": hardware_counts["loihi"]
                    }
                    hardware_counts["loihi"] += 1
            
            # Similar modifications for other hardware types...
            
        except Exception as e:
            logger.debug(f"macOS detection error: {str(e)}")
    
    def _detect_linux(self):
        """Linux-specific hardware detection."""
        # Check for Loihi via lsusb
        try:
            result = subprocess.run(
                ["lsusb"], 
                capture_output=True, 
                text=True,
                timeout=2
            )
            
            output = result.stdout.lower()
            
            # Intel Loihi USB detection
            if "intel" in output and any(id_str in output for id_str in self.LOIHI_IDENTIFIERS):
                logger.info("Detected Loihi hardware via Linux USB")
                self.detected_hardware["loihi"] = {
                    "type": "loihi",
                    "connection_type": "usb",
                    "board_id": 0,
                    "platform": "linux"
                }
            
            # Check for TrueNorth via PCIe
            result = subprocess.run(
                ["lspci"], 
                capture_output=True, 
                text=True,
                timeout=2
            )
            
            output = result.stdout.lower()
            if "ibm" in output and any(id_str in output for id_str in self.TRUENORTH_IDENTIFIERS):
                logger.info("Detected TrueNorth hardware via Linux PCIe")
                self.detected_hardware["truenorth"] = {
                    "type": "truenorth",
                    "connection_type": "pcie",
                    "board_id": 0,
                    "platform": "linux"
                }
            
            # Check for SpiNNaker via network interfaces
            result = subprocess.run(
                ["ip", "addr"], 
                capture_output=True, 
                text=True,
                timeout=2
            )
            
            output = result.stdout.lower()
            if "spinn" in output or any(id_str in output for id_str in self.SPINNAKER_IDENTIFIERS):
                logger.info("Detected SpiNNaker hardware via Linux network interface")
                self.detected_hardware["spinnaker"] = {
                    "type": "spinnaker",
                    "connection_type": "ethernet",
                    "address": "192.168.240.1",
                    "platform": "linux"
                }
                
            # Check for neuromorphic hardware in dmesg logs
            result = subprocess.run(
                ["dmesg"], 
                capture_output=True, 
                text=True,
                timeout=2
            )
            
            output = result.stdout.lower()
            
            for hw_type, identifiers in [
                ("loihi", self.LOIHI_IDENTIFIERS),
                ("truenorth", self.TRUENORTH_IDENTIFIERS),
                ("spinnaker", self.SPINNAKER_IDENTIFIERS)
            ]:
                if any(id_str in output for id_str in identifiers) and hw_type not in self.detected_hardware:
                    logger.info(f"Detected {hw_type} hardware via Linux dmesg logs")
                    self.detected_hardware[hw_type] = {
                        "type": hw_type,
                        "connection_type": "unknown",
                        "platform": "linux"
                    }
                
        except Exception as e:
            logger.debug(f"Linux detection error: {str(e)}")
    
    def _detect_windows(self):
        """Windows-specific hardware detection."""
        try:
            # Check for hardware via Windows Management Instrumentation (WMI)
            result = subprocess.run(
                ["wmic", "path", "Win32_USBControllerDevice", "get", "/value"], 
                capture_output=True, 
                text=True,
                timeout=3
            )
            
            output = result.stdout.lower()
            
            # Check for known hardware in USB devices
            for hw_type, identifiers in [
                ("loihi", self.LOIHI_IDENTIFIERS),
                ("truenorth", self.TRUENORTH_IDENTIFIERS),
                ("spinnaker", self.SPINNAKER_IDENTIFIERS)
            ]:
                if any(id_str in output for id_str in identifiers):
                    logger.info(f"Detected {hw_type} hardware via Windows USB")
                    self.detected_hardware[hw_type] = {
                        "type": hw_type,
                        "connection_type": "usb",
                        "board_id": 0,
                        "platform": "windows"
                    }
            
            # Check for PCI devices
            result = subprocess.run(
                ["wmic", "path", "Win32_PnPEntity", "get", "Caption,DeviceID", "/format:list"], 
                capture_output=True, 
                text=True,
                timeout=3
            )
            
            output = result.stdout.lower()
            
            # Check for TrueNorth via PCIe
            if "ibm" in output and any(id_str in output for id_str in self.TRUENORTH_IDENTIFIERS):
                logger.info("Detected TrueNorth hardware via Windows PCIe")
                self.detected_hardware["truenorth"] = {
                    "type": "truenorth",
                    "connection_type": "pcie",
                    "board_id": 0,
                    "platform": "windows"
                }
                
            # Check for network adapters
            result = subprocess.run(
                ["ipconfig", "/all"], 
                capture_output=True, 
                text=True,
                timeout=2
            )
            
            output = result.stdout.lower()
            
            # SpiNNaker often uses a dedicated network interface
            if "spinn" in output or any(id_str in output for id_str in self.SPINNAKER_IDENTIFIERS):
                logger.info("Detected SpiNNaker hardware via Windows network interface")
                self.detected_hardware["spinnaker"] = {
                    "type": "spinnaker",
                    "connection_type": "ethernet",
                    "address": "192.168.240.1",
                    "platform": "windows"
                }
                
        except Exception as e:
            logger.debug(f"Windows detection error: {str(e)}")

    def _detect_via_fallback(self):
        """Use fallback detection methods."""
        # Try ping-based detection for network devices
        for hw_type, addresses in self.COMMON_ADDRESSES.items():
            for address in addresses:
                try:
                    # Use ping with short timeout
                    ping_cmd = ["ping", "-c", "1", "-W", "1", address]
                    result = subprocess.run(ping_cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"Detected possible {hw_type} hardware at {address} via ping")
                        self.detected_hardware[hw_type] = {
                            "type": hw_type,
                            "connection_type": "network",
                            "address": address,
                            "detection_method": "fallback_ping"
                        }
                        return
                except:
                    pass
        
        # Check for hardware-specific files or directories
        paths_to_check = {
            "loihi": ["/opt/intel/nxsdk", "/Applications/Intel/nxsdk"],
            "truenorth": ["/opt/ibm/truenorth", "/Applications/IBM/TrueNorth"],
            "spinnaker": ["/opt/spinnaker", "/Applications/SpiNNaker"]
        }
        
        for hw_type, paths in paths_to_check.items():
            for path in paths:
                if os.path.exists(path):
                    logger.info(f"Detected {hw_type} software installation at {path}")
                    self.detected_hardware[hw_type] = {
                        "type": hw_type,
                        "connection_type": "unknown",
                        "software_path": path,
                        "detection_method": "fallback_path"
                    }
                    return
    
    def _try_load_cached_config(self):
        """Try to load cached hardware configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    cached_config = json.load(f)
                
                # Check if cache is recent (less than 1 day old)
                if cached_config.get("timestamp", 0) > time.time() - 86400:
                    logger.info("Using cached hardware configuration")
                    self.detected_hardware = cached_config.get("hardware", {})
                    return True
            except Exception as e:
                logger.debug(f"Failed to load cached config: {str(e)}")
        
        return False
    
    def _cache_detection_results(self):
        """Cache successful detection results."""
        if self.detected_hardware:
            try:
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump({
                        "hardware": self.detected_hardware,
                        "timestamp": time.time()
                    }, f)
            except Exception as e:
                logger.debug(f"Failed to cache detection results: {str(e)}")
    
    def get_hardware_by_type(self, hardware_type: str) -> List[Dict[str, Any]]:
        """
        Get all detected hardware of a specific type.
        
        Args:
            hardware_type: Type of hardware to retrieve ('loihi', 'spinnaker', 'truenorth')
            
        Returns:
            List[Dict[str, Any]]: List of hardware info dictionaries
        """
        if not self.detected_hardware:
            self.detect_hardware()
            
        result = []
        for hw_id, hw_info in self.detected_hardware.items():
            if hw_info.get("type") == hardware_type:
                result.append(hw_info)
                
        return result
    
    def get_best_hardware(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get the best available hardware.
        
        Returns:
            Optional[Tuple[str, Dict[str, Any]]]: Tuple of (hardware_id, hardware_info)
        """
        if not self.detected_hardware:
            self.detect_hardware()
            
        if not self.detected_hardware:
            return None
            
        # Priority order: Loihi > TrueNorth > SpiNNaker
        for hw_type in ["loihi", "truenorth", "spinnaker"]:
            for hw_id, hw_info in self.detected_hardware.items():
                if hw_info.get("type") == hw_type:
                    return hw_id, hw_info
                
        # If we get here, just return the first one
        hw_id = next(iter(self.detected_hardware))
        return hw_id, self.detected_hardware[hw_id]
    
    def get_all_hardware(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all detected hardware devices.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of all detected hardware
        """
        if not self.detected_hardware:
            self.detect_hardware()
            
        return self.detected_hardware


# Add this import at the top of the file
from src.core.hardware.hardware_capabilities import capabilities_discovery

# Add these methods to the HardwareDetector class
def get_hardware_capabilities(self, hardware_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get capabilities of a specific hardware device or the best available hardware.
    
    Args:
        hardware_id: ID of the hardware device (optional)
        
    Returns:
        Dict[str, Any]: Hardware capabilities
    """
    if not self.detected_hardware:
        self.detect_hardware()
        
    if not self.detected_hardware:
        return {}
        
    if hardware_id is None:
        # Get best hardware
        hw_id, hw_info = self.get_best_hardware()
    elif hardware_id in self.detected_hardware:
        hw_id = hardware_id
        hw_info = self.detected_hardware[hw_id]
    else:
        return {}
    
    # Discover capabilities
    return capabilities_discovery.discover_capabilities(hw_info)

def get_all_hardware_capabilities(self) -> Dict[str, Dict[str, Any]]:
    """
    Get capabilities of all detected hardware devices.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of hardware capabilities
    """
    if not self.detected_hardware:
        self.detect_hardware()
        
    return capabilities_discovery.get_all_capabilities(self.detected_hardware)

def check_feature_support(self, feature: str, hardware_id: Optional[str] = None) -> bool:
    """
    Check if a specific feature is supported by hardware.
    
    Args:
        feature: Feature to check
        hardware_id: ID of the hardware device (optional)
        
    Returns:
        bool: True if feature is supported
    """
    capabilities = self.get_hardware_capabilities(hardware_id)
    return capabilities_discovery.check_capability_support(capabilities, feature)


# Create a singleton instance
hardware_detector = HardwareDetector()
