#!/usr/bin/env python3
"""
Test the manufacturing simulation functionality.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
import sys
import os
from unittest.mock import MagicMock, patch
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.manufacturing.simulation.process_simulator import ManufacturingSimulator
from src.simulation.aerodynamics.ucav_model import UCAVGeometry


class TestManufacturingSimulator(unittest.TestCase):
    """Test the manufacturing simulation process."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects for dependencies
        self.mock_system = MagicMock()
        self.mock_material_selector = MagicMock()
        
        # Create sample geometry
        self.geometry = UCAVGeometry(
            length=12.0,
            wingspan=15.0,
            wing_area=45.0,  # Reasonable wing area based on length and wingspan
            aspect_ratio=5.0,  # Typical aspect ratio for UCAV
            taper_ratio=0.3,  # Common taper ratio for swept wings
            sweep_angle=45.0
        )
        self.geometry.wingspan = 15.0
        self.geometry.length = 12.0
        self.geometry.sweep_angle = 45.0
        
        # Create sample material config
        self.material_config = {
            'composite': {
                'type': 'carbon_fiber',
                'thickness': 0.05,
                'coverage': 0.8
            },
            'coating': {
                'type': 'radar_absorbing',
                'thickness': 0.01,
                'coverage': 1.0
            }
        }
        
        # Create patches for dependencies
        self.system_patcher = patch('src.core.integration.neuromorphic_system.NeuromorphicSystem')
        self.material_selector_patcher = patch('src.manufacturing.materials.material_selector.NeuromorphicMaterialSelector')
        
        # Start patches
        self.mock_system_class = self.system_patcher.start()
        self.mock_material_selector_class = self.material_selector_patcher.start()
        
        # Configure mocks
        self.mock_system_class.return_value = self.mock_system
        self.mock_material_selector_class.return_value = self.mock_material_selector
        
        # Configure stage simulation results
        def mock_simulate_stage(stage, current_state, params):
            return {
                'time': params['time'],
                'failure_rate': params['failure_rate'],
                'state_updates': {
                    f'{stage}_completed': True,
                    f'{stage}_quality': 0.9
                }
            }
        
        self.mock_system.process_data.side_effect = self._mock_process_data
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.system_patcher.stop()
        self.material_selector_patcher.stop()
    
    def _mock_process_data(self, data):
        """Mock the process_data method to return appropriate results based on computation type."""
        if data.get('computation') == 'process_simulation':
            stage = data.get('stage')
            params = data.get('parameters')
            return {
                'time': params['time'],
                'failure_rate': params['failure_rate'],
                'state_updates': {
                    f'{stage}_completed': True,
                    f'{stage}_quality': 0.9
                }
            }
        elif data.get('computation') == 'quality_assessment':
            return {
                'overall_quality': 0.88,
                'dimensional_accuracy': 0.92,
                'surface_finish': 0.85,
                'structural_integrity': 0.90
            }
        return {}
    
    def test_manufacturing_simulation(self):
        """Test the manufacturing simulation process."""
        # Create simulator
        simulator = ManufacturingSimulator()
        
        # FIXED: Replace the mock_system with our mock in the simulator
        # The simulator might be creating its own system instance
        simulator.system = self.mock_system
        
        # Run simulation
        results = simulator.simulate_manufacturing(self.geometry, self.material_config)
        
        # Verify system was initialized and cleaned up
        self.mock_system.initialize.assert_called_once()
        self.mock_system.cleanup.assert_called_once()
        
        # Verify all stages were simulated
        expected_stages = ['composite_layup', 'metal_forming', 'assembly', 'curing', 'quality_check']
        for stage in expected_stages:
            self.assertIn(stage, results['stage_results'])
        
        # Verify completion time calculation
        expected_time = sum([24, 12, 36, 48, 8])  # Sum of all stage times
        self.assertEqual(results['completion_time'], expected_time)
        
        # Verify success probability calculation
        expected_probs = [0.95, 0.97, 0.98, 0.96, 0.99]  # 1 - failure_rate for each stage
        expected_success_prob = float(np.prod(expected_probs))
        self.assertAlmostEqual(results['success_probability'], expected_success_prob, places=6)
        
        # Verify quality score
        self.assertEqual(results['quality_score'], 0.88)


if __name__ == "__main__":
    unittest.main()