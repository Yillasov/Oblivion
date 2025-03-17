"""
Hardware Detection Module

Provides functionality to automatically detect and configure neuromorphic hardware.
"""

import os
import platform
import socket
import json
from typing import Dict, List, Any, Optional, Tuple

from src.core.utils.logging_framework import get_logger

logger = get_logger("hardware_detection")


class HardwareDetector:
    """Detects available neuromorphic hardware."""
    
    # Known hardware identifiers
    LOIHI_IDENTIFIERS = ["loihi", "nxsdk", "kapoho"]
    TRUENORTH_IDENTIFIERS = ["truenorth", "tn", "ibm-tn"]
    SPINNAKER_IDENTIFIERS = ["spinnaker", "spin", "spinn"]
    
    def __init__(self):
        """Initialize the hardware detector."""
        self.detected_hardware = {}
    
    def detect_hardware(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect available neuromorphic hardware.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of detected hardware
        """
        logger.info("Detecting available neuromorphic hardware")
        
        # Clear previous detections
        self.detected_hardware = {}
        
        # Detect Loihi hardware
        loihi_info = self._detect_loihi()
        if loihi_info:
            self.detected_hardware["loihi"] = loihi_info
            
        # Detect TrueNorth hardware
        truenorth_info = self._detect_truenorth()
        if truenorth_info:
            self.detected_hardware["truenorth"] = truenorth_info
            
        # Detect SpiNNaker hardware
        spinnaker_info = self._detect_spinnaker()
        if spinnaker_info:
            self.detected_hardware["spinnaker"] = spinnaker_info
        
        if not self.detected_hardware:
            logger.warning("No neuromorphic hardware detected")
        else:
            logger.info(f"Detected {len(self.detected_hardware)} neuromorphic hardware devices")
            
        return self.detected_hardware
    
    def _detect_loihi(self) -> Optional[Dict[str, Any]]:
        """Detect Intel Loihi hardware."""
        # Check for Loihi USB devices or network devices
        # This is a simplified implementation - in a real system, this would
        # use hardware-specific APIs to detect Loihi devices
        
        # Check for environment variables that might indicate Loihi presence
        for env_var in os.environ:
            if any(id_str in env_var.lower() for id_str in self.LOIHI_IDENTIFIERS):
                logger.info("Detected Loihi hardware via environment variables")
                return {
                    "type": "loihi",
                    "connection_type": "usb",
                    "board_id": 0,
                    "chip_id": 0
                }
        
        # Check for Loihi network devices
        try:
            # Try to connect to default Loihi port
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            result = s.connect_ex(('127.0.0.1', 22222))
            s.close()
            
            if result == 0:
                logger.info("Detected Loihi hardware via network connection")
                return {
                    "type": "loihi",
                    "connection_type": "network",
                    "address": "127.0.0.1",
                    "port": 22222
                }
        except:
            pass
            
        return None
    
    def _detect_truenorth(self) -> Optional[Dict[str, Any]]:
        """Detect IBM TrueNorth hardware."""
        # Check for TrueNorth hardware
        # This is a simplified implementation
        
        # Check for environment variables
        for env_var in os.environ:
            if any(id_str in env_var.lower() for id_str in self.TRUENORTH_IDENTIFIERS):
                logger.info("Detected TrueNorth hardware via environment variables")
                return {
                    "type": "truenorth",
                    "connection_type": "network",
                    "address": os.environ.get(env_var, "127.0.0.1")
                }
                
        return None
    
    def _detect_spinnaker(self) -> Optional[Dict[str, Any]]:
        """Detect SpiNNaker hardware."""
        # Check for SpiNNaker hardware
        # This is a simplified implementation
        
        # Check for environment variables
        for env_var in os.environ:
            if any(id_str in env_var.lower() for id_str in self.SPINNAKER_IDENTIFIERS):
                logger.info("Detected SpiNNaker hardware via environment variables")
                return {
                    "type": "spinnaker",
                    "connection_type": "ethernet",
                    "address": os.environ.get(env_var, "192.168.1.1")
                }
                
        # Try to connect to default SpiNNaker port
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            result = s.connect_ex(('192.168.1.1', 17893))
            s.close()
            
            if result == 0:
                logger.info("Detected SpiNNaker hardware via network connection")
                return {
                    "type": "spinnaker",
                    "connection_type": "ethernet",
                    "address": "192.168.1.1",
                    "port": 17893
                }
        except:
            pass
                
        return None
    
    def get_best_hardware(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get the best available hardware.
        
        Returns:
            Optional[Tuple[str, Dict[str, Any]]]: Tuple of (hardware_type, hardware_info)
        """
        if not self.detected_hardware:
            self.detect_hardware()
            
        if not self.detected_hardware:
            return None
            
        # Priority order: Loihi > TrueNorth > SpiNNaker
        for hw_type in ["loihi", "truenorth", "spinnaker"]:
            if hw_type in self.detected_hardware:
                return hw_type, self.detected_hardware[hw_type]
                
        # If we get here, just return the first one
        hw_type = next(iter(self.detected_hardware))
        return hw_type, self.detected_hardware[hw_type]


# Create a singleton instance
hardware_detector = HardwareDetector()