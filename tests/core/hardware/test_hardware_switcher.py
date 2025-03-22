"""
Test hardware switching functionality between neuromorphic platforms.
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional

from src.core.hardware.hardware_switcher import HardwareSwitcher
from src.core.hardware.compatibility_layer import HardwareCompatibilityLayer
from src.core.hardware.exceptions import HardwareSwitchingError

class TestHardwareSwitcher(unittest.TestCase):
    """Test hardware switching between different neuromorphic platforms."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock hardware info for each platform
        self.hardware_info = {
            "loihi": {"type": "loihi", "capabilities": {"on_chip_learning": True}},
            "spinnaker": {"type": "spinnaker", "capabilities": {"multicast_routing": True}},
            "truenorth": {"type": "truenorth", "capabilities": {"binary_weights": True}}
        }
        
        # Create patch for HardwareCompatibilityLayer
        self.patcher = patch('src.core.hardware.hardware_switcher.HardwareCompatibilityLayer')
        self.mock_layer_class = self.patcher.start()
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.patcher.stop()
    
    def _create_mock_hardware(self, hardware_type):
        """Create a mock hardware layer for the specified type."""
        mock = MagicMock(spec=HardwareCompatibilityLayer)
        mock.hardware_type = hardware_type
        mock.get_hardware_info.return_value = self.hardware_info[hardware_type]
        mock.initialize.return_value = True
        mock.shutdown.return_value = True
        # Add config attribute to fix the AttributeError
        mock.config = {"hardware_type": hardware_type}
        return mock
    
    def test_switch_between_platforms(self):
        """Test switching between all three hardware platforms."""
        # Configure mocks for different hardware types
        loihi_mock = self._create_mock_hardware("loihi")
        spinnaker_mock = self._create_mock_hardware("spinnaker")
        truenorth_mock = self._create_mock_hardware("truenorth")
        
        # Set up mock to return different hardware based on hardware_type
        def create_hardware_mock(hardware_type, config):
            if hardware_type == "loihi": return loihi_mock
            elif hardware_type == "spinnaker": return spinnaker_mock
            elif hardware_type == "truenorth": return truenorth_mock
            return MagicMock()
            
        self.mock_layer_class.side_effect = create_hardware_mock
        
        # Create hardware switcher with initial Loihi hardware
        switcher = HardwareSwitcher(initial_hardware_type="loihi")
        
        # Mock the state capture and migration methods
        switcher._capture_network_state = MagicMock(return_value={"neurons": {}, "synapses": {}})
        switcher._migrate_network_state = MagicMock(return_value=True)
        
        # Test switching to SpiNNaker
        result = switcher.switch_hardware("spinnaker")
        self.assertTrue(result)
        self.assertEqual(switcher.hardware_type, "spinnaker")
        loihi_mock.shutdown.assert_called_once()
        
        # Test switching to TrueNorth
        result = switcher.switch_hardware("truenorth")
        self.assertTrue(result)
        self.assertEqual(switcher.hardware_type, "truenorth")
        spinnaker_mock.shutdown.assert_called_once()
        
        # Test switching back to Loihi
        result = switcher.switch_hardware("loihi")
        self.assertTrue(result)
        self.assertEqual(switcher.hardware_type, "loihi")
        truenorth_mock.shutdown.assert_called_once()
    
    def test_switch_failure_handling(self):
        """Test handling of hardware switching failures."""
        # Configure mocks
        initial_mock = self._create_mock_hardware("loihi")
        target_mock = self._create_mock_hardware("spinnaker")
        target_mock.initialize.return_value = False  # Simulate initialization failure
        
        # Set up mock to return different hardware based on hardware_type
        def create_hardware_mock(hardware_type, config):
            if hardware_type == "loihi": return initial_mock
            elif hardware_type == "spinnaker": return target_mock
            return MagicMock()
            
        self.mock_layer_class.side_effect = create_hardware_mock
        
        # Create hardware switcher
        switcher = HardwareSwitcher(initial_hardware_type="loihi")
        
        # Test switching with initialization failure
        result = switcher.switch_hardware("spinnaker")
        self.assertFalse(result)
        self.assertEqual(switcher.hardware_type, "loihi")  # Should remain on original hardware
        initial_mock.shutdown.assert_not_called()  # Original hardware should not be shut down

if __name__ == "__main__":
    unittest.main()