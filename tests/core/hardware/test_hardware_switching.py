#!/usr/bin/env python3
"""
Test hardware switching and compatibility.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
import os
import sys
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.core.hardware.compatibility_layer import HardwareCompatibilityLayer, create_hardware_interface
from src.core.hardware.exceptions import UnsupportedFeatureError

class TestHardwareSwitching(unittest.TestCase):
    
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock hardware interfaces for each platform
        self.mock_loihi_info = {
            "type": "loihi",
            "version": "2.0",
            "capabilities": {
                "on_chip_learning": True,
                "phase_encoding": True,
                "binary_weights": False,
                "max_neurons": 128000,
                "max_synapses": 128000000
            }
        }
        
        self.mock_spinnaker_info = {
            "type": "spinnaker",
            "version": "5.0",
            "capabilities": {
                "on_chip_learning": True,
                "phase_encoding": False,
                "binary_weights": False,
                "max_neurons": 1000000,
                "max_synapses": 1000000000
            }
        }
        
        self.mock_truenorth_info = {
            "type": "truenorth",
            "version": "1.0",
            "capabilities": {
                "on_chip_learning": False,
                "phase_encoding": False,
                "binary_weights": True,
                "max_neurons": 1000000,
                "max_synapses": 256000000
            }
        }
    
    @patch('src.core.hardware.unified_interface.NeuromorphicHardwareInterface')
    def test_hardware_type_detection(self, mock_interface_class):
        """Test hardware type detection and initialization."""
        # Configure mock
        mock_interface = MagicMock()
        mock_interface.hardware_type = "loihi"
        mock_interface.get_hardware_info.return_value = self.mock_loihi_info
        mock_interface_class.return_value = mock_interface
        
        # Create hardware layer
        hardware = HardwareCompatibilityLayer()
        
        # Verify hardware type was detected
        self.assertEqual(hardware.hardware_type, "loihi")
        self.assertEqual(hardware.current_optimizations["weight_precision"], 8)
    
    @patch('src.core.hardware.unified_interface.NeuromorphicHardwareInterface')
    def test_loihi_specific_optimizations(self, mock_interface_class):
        """Test Loihi-specific optimizations."""
        # Configure mock
        mock_interface = MagicMock()
        mock_interface.hardware_type = "loihi"
        mock_interface.get_hardware_info.return_value = self.mock_loihi_info
        mock_interface_class.return_value = mock_interface
        
        # Create hardware layer
        hardware = HardwareCompatibilityLayer(hardware_type="loihi")
        
        # Apply optimizations
        hardware._optimize_for_loihi()
        
        # Verify Loihi-specific optimizations
        self.assertTrue(hardware.config.get("phase_encoding", False))
        if "learning" in hardware.config:
            self.assertEqual(hardware.config["learning"]["weight_precision"], 8)
            self.assertTrue(hardware.config["learning"]["use_compartments"])
    
    @patch('src.core.hardware.unified_interface.NeuromorphicHardwareInterface')
    def test_spinnaker_specific_optimizations(self, mock_interface_class):
        """Test SpiNNaker-specific optimizations."""
        # Configure mock
        mock_interface = MagicMock()
        mock_interface.hardware_type = "spinnaker"
        mock_interface.get_hardware_info.return_value = self.mock_spinnaker_info
        mock_interface_class.return_value = mock_interface
        
        # Create hardware layer
        hardware = HardwareCompatibilityLayer(hardware_type="spinnaker")
        
        # Apply optimizations
        hardware._optimize_for_spinnaker()
        
        # Verify SpiNNaker-specific optimizations
        self.assertEqual(hardware.config.get("packet_routing"), "multicast")
        if "learning" in hardware.config:
            self.assertEqual(hardware.config["learning"]["weight_precision"], 16)
            self.assertTrue(hardware.config["learning"]["use_sdram_for_weights"])
    
    @patch('src.core.hardware.unified_interface.NeuromorphicHardwareInterface')
    def test_truenorth_specific_optimizations(self, mock_interface_class):
        """Test TrueNorth-specific optimizations."""
        # Configure mock
        mock_interface = MagicMock()
        mock_interface.hardware_type = "truenorth"
        mock_interface.get_hardware_info.return_value = self.mock_truenorth_info
        mock_interface_class.return_value = mock_interface
        
        # Create hardware layer
        hardware = HardwareCompatibilityLayer(hardware_type="truenorth")
        
        # Apply optimizations
        hardware._optimize_for_truenorth()
        
        # Verify TrueNorth-specific optimizations
        self.assertTrue(hardware.config.get("binary_weights", False))
        self.assertEqual(hardware.config.get("cores_per_chip"), 4096)
        if "learning" in hardware.config:
            self.assertEqual(hardware.config["learning"]["weight_precision"], 1)
            self.assertTrue(hardware.config["learning"]["offline_only"])
    
    @patch('src.core.hardware.unified_interface.NeuromorphicHardwareInterface')
    def test_feature_support_detection(self, mock_interface_class):
        """Test feature support detection across hardware platforms."""
        # Test for each hardware type
        hardware_types = ["loihi", "spinnaker", "truenorth"]
        mock_info = {
            "loihi": self.mock_loihi_info,
            "spinnaker": self.mock_spinnaker_info,
            "truenorth": self.mock_truenorth_info
        }
        
        for hw_type in hardware_types:
            # Configure mock
            mock_interface = MagicMock()
            mock_interface.hardware_type = hw_type
            mock_interface.get_hardware_info.return_value = mock_info[hw_type]
            mock_interface_class.return_value = mock_interface
            
            # Create hardware layer
            hardware = HardwareCompatibilityLayer(hardware_type=hw_type)
            
            # Test feature support
            self.assertEqual(hardware.supports_feature("on_chip_learning"), 
                            mock_info[hw_type]["capabilities"]["on_chip_learning"])
            self.assertEqual(hardware.supports_feature("binary_weights"), 
                            mock_info[hw_type]["capabilities"]["binary_weights"])
    
    @patch('src.core.hardware.unified_interface.NeuromorphicHardwareInterface')
    def test_hardware_switching(self, mock_interface_class):
        """Test switching between hardware platforms."""
        # Configure mocks for different hardware types
        mock_interfaces = {}
        for hw_type in ["loihi", "spinnaker", "truenorth"]:
            mock_interface = MagicMock()
            mock_interface.hardware_type = hw_type
            mock_interface.get_hardware_info.return_value = getattr(self, f"mock_{hw_type}_info")
            mock_interfaces[hw_type] = mock_interface
        
        # Test switching between platforms
        for hw_type in ["loihi", "spinnaker", "truenorth"]:
            mock_interface_class.return_value = mock_interfaces[hw_type]
            
            # Create hardware layer
            hardware = HardwareCompatibilityLayer(hardware_type=hw_type)
            
            # Verify correct hardware type
            self.assertEqual(hardware.hardware_type, hw_type)
            
            # Verify hardware-specific optimizations
            if hw_type == "loihi":
                self.assertEqual(hardware.current_optimizations["weight_precision"], 8)
            elif hw_type == "spinnaker":
                self.assertEqual(hardware.current_optimizations["weight_precision"], 16)
            elif hw_type == "truenorth":
                self.assertEqual(hardware.current_optimizations["weight_precision"], 1)

if __name__ == "__main__":
    unittest.main()