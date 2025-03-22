"""
Test the quality inspection functionality.
"""

import unittest
import sys
import os
from unittest.mock import MagicMock, patch
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.manufacturing.quality.quality_inspector import QualityInspector
from src.simulation.aerodynamics.ucav_model import UCAVGeometry


class TestQualityInspector(unittest.TestCase):
    """Test the quality inspection process."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects for dependencies
        self.mock_system = MagicMock()
        self.mock_simulator = MagicMock()
        
        # Create sample geometry
        self.geometry = UCAVGeometry(
            length=12.0,
            wingspan=15.0,
            wing_area=45.0,
            aspect_ratio=5.0,
            taper_ratio=0.3,
            sweep_angle=45.0
        )
        
        # Create sample manufacturing data
        self.manufacturing_data = {
            'composite_layup_completed': True,
            'composite_layup_quality': 0.92,
            'metal_forming_completed': True,
            'metal_forming_quality': 0.88,
            'assembly_completed': True,
            'assembly_quality': 0.95,
            'curing_completed': True,
            'curing_quality': 0.90,
            'quality_check_completed': True,
            'quality_check_quality': 0.93,
            'material_properties': {
                'composite': {
                    'type': 'carbon_fiber',
                    'thickness': 0.05,
                    'coverage': 0.98
                },
                'coating': {
                    'type': 'radar_absorbing',
                    'thickness': 0.01,
                    'coverage': 0.99
                }
            }
        }
        
        # Create patches for dependencies
        self.system_patcher = patch('src.core.integration.neuromorphic_system.NeuromorphicSystem')
        self.simulator_patcher = patch('src.manufacturing.simulation.process_simulator.ManufacturingSimulator')
        
        # Start patches
        self.mock_system_class = self.system_patcher.start()
        self.mock_simulator_class = self.simulator_patcher.start()
        
        # Configure mocks
        self.mock_system_class.return_value = self.mock_system
        self.mock_simulator_class.return_value = self.mock_simulator
        
        # Configure inspection results
        self.mock_system.process_data.side_effect = self._mock_process_data
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.system_patcher.stop()
        self.simulator_patcher.stop()
    
    def _mock_process_data(self, data):
        """Mock the process_data method to return appropriate results based on computation type."""
        if data.get('computation') == 'inspection':
            aspect = data.get('aspect')
            return {
                'passed': True,
                'score': 0.9,
                'details': {
                    'points_checked': 12,
                    'points_passed': 11,
                    'tolerance': data.get('parameters', {}).get('tolerance', 0.05)
                }
            }
        elif data.get('computation') == 'quality_assessment':
            # Return overall quality assessment
            return {
                'overall_quality': 0.91,
                'dimensional_accuracy': 0.93,
                'surface_finish': 0.89,
                'structural_integrity': 0.92,
                'detected_defects': [
                    {'type': 'minor_surface_imperfection', 'location': 'wing_trailing_edge', 'severity': 0.2}
                ],
                'improvement_suggestions': [
                    'Increase curing time by 5% for better surface finish',
                    'Verify alignment of wing-fuselage junction'
                ]
            }
        return {}
    
    def test_quality_inspection(self):
        """Test the quality inspection process."""
        # Create inspector
        inspector = QualityInspector()
        
        # Replace the system and simulator with our mocks
        inspector.system = self.mock_system
        inspector.simulator = self.mock_simulator
        
        # Run inspection
        results = inspector.inspect_ucav(self.geometry, self.manufacturing_data)
        
        # Verify system was initialized and cleaned up
        self.mock_system.initialize.assert_called_once()
        self.mock_system.cleanup.assert_called_once()
        
        # Verify all aspects were inspected
        expected_aspects = ['structural', 'aerodynamic', 'material']
        for aspect in expected_aspects:
            self.assertIn(aspect, results['detailed_results'])
        
        # Verify quality score
        self.assertEqual(results['quality_score'], 0.91)
        
        # Verify pass/fail status
        self.assertTrue(results['passed'])
        
        # Verify defects were detected
        self.assertEqual(len(results['defects']), 1)
        self.assertEqual(results['defects'][0]['type'], 'minor_surface_imperfection')
        
        # Verify recommendations were provided
        self.assertEqual(len(results['recommendations']), 2)


if __name__ == "__main__":
    unittest.main()