"""
Test the prototype testing functionality.
"""

import unittest
import sys
import os
from unittest.mock import MagicMock, patch
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.manufacturing.testing.prototype_tester import PrototypeTester
from src.simulation.aerodynamics.ucav_model import UCAVGeometry


class TestPrototypeTester(unittest.TestCase):
    """Test the prototype testing process."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects for dependencies
        self.mock_system = MagicMock()
        self.mock_inspector = MagicMock()
        
        # Create sample geometry
        self.prototype = UCAVGeometry(
            length=12.0,
            wingspan=15.0,
            wing_area=45.0,
            aspect_ratio=5.0,
            taper_ratio=0.3,
            sweep_angle=45.0
        )
        
        # Create sample test configuration
        self.test_config = {
            'wind_tunnel_settings': {
                'max_speed': 1.2,  # Mach
                'test_points': 5
            },
            'structural_settings': {
                'max_load': 4.0,  # G forces
                'test_points': 24
            },
            'thermal_settings': {
                'min_temp': -40,  # Celsius
                'max_temp': 85,
                'cycles': 10
            }
        }
        
        # Create patches for dependencies
        self.system_patcher = patch('src.core.integration.neuromorphic_system.NeuromorphicSystem')
        self.inspector_patcher = patch('src.manufacturing.quality.quality_inspector.QualityInspector')
        
        # Start patches
        self.mock_system_class = self.system_patcher.start()
        self.mock_inspector_class = self.inspector_patcher.start()
        
        # Configure mocks
        self.mock_system_class.return_value = self.mock_system
        self.mock_inspector_class.return_value = self.mock_inspector
        
        # Configure test results
        self.mock_system.process_data.side_effect = self._mock_process_data
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.system_patcher.stop()
        self.inspector_patcher.stop()
    
    def _mock_process_data(self, data):
        """Mock the process_data method to return appropriate results based on computation type."""
        computation = data.get('computation', '')
        
        if computation == 'aero_testing':
            return {
                'lift_to_drag_ratio': 18.5,
                'stall_angle': 14.2,
                'max_speed': 1.8,
                'test_points': 15,
                'passed': True
            }
        elif computation == 'structural_testing':
            return {
                'max_stress': 420.5,  # MPa
                'safety_factor': 1.8,
                'critical_points': [
                    {'location': 'wing_root', 'stress': 380.2},
                    {'location': 'fuselage_joint', 'stress': 420.5}
                ],
                'passed': True
            }
        elif computation == 'thermal_testing':
            return {
                'max_temp_deformation': 0.8,  # mm
                'thermal_cycles_completed': 10,
                'critical_points': [
                    {'location': 'engine_bay', 'max_temp': 82.3},
                    {'location': 'leading_edge', 'max_temp': 65.7}
                ],
                'passed': True
            }
        elif computation == 'loihi_testing':
            return {
                'neuron_tests_passed': 3,
                'synapse_tests_passed': 3,
                'performance_metrics': {
                    'latency': 0.8,  # ms
                    'power_consumption': 0.12  # W
                },
                'passed': True
            }
        elif computation == 'test_analysis':
            return {
                'overall_score': 0.92,
                'strengths': ['aerodynamic_efficiency', 'thermal_resistance'],
                'weaknesses': ['structural_weight'],
                'recommendations': [
                    'Reinforce wing root connection',
                    'Optimize fuselage joint design'
                ]
            }
        
        return {}
    
    def test_run_prototype_tests(self):
        """Test running prototype tests."""
        # Create tester
        tester = PrototypeTester()
        
        # Replace the system and inspector with our mocks
        tester.system = self.mock_system
        tester.inspector = self.mock_inspector
        
        # Run tests
        results = tester.run_prototype_tests(self.prototype, self.test_config)
        
        # Verify system was initialized and cleaned up
        self.mock_system.initialize.assert_called_once()
        self.mock_system.cleanup.assert_called_once()
        
        # Verify all test types were run
        self.assertIn('aerodynamic', results)
        self.assertIn('structural', results)
        self.assertIn('thermal', results)
        self.assertIn('hardware_specific', results)
        self.assertIn('analysis', results)
        
        # Verify test status
        self.assertEqual(results['status'], 'completed')
        
        # Verify aerodynamic test results
        self.assertEqual(results['aerodynamic']['lift_to_drag_ratio'], 18.5)
        self.assertTrue(results['aerodynamic']['passed'])
        
        # Verify structural test results
        self.assertEqual(results['structural']['safety_factor'], 1.8)
        self.assertTrue(results['structural']['passed'])
        
        # Verify thermal test results
        self.assertEqual(results['thermal']['thermal_cycles_completed'], 10)
        self.assertTrue(results['thermal']['passed'])
        
        # Verify analysis results
        self.assertEqual(results['analysis']['overall_score'], 0.92)
        self.assertEqual(len(results['analysis']['recommendations']), 2)
    
    def test_error_handling(self):
        """Test error handling during prototype testing."""
        # Create tester
        tester = PrototypeTester()
        
        # Replace the system with our mock
        tester.system = self.mock_system
        
        # Configure mock to raise an exception
        self.mock_system.process_data.side_effect = Exception("Simulated test failure")
        
        # Run tests
        results = tester.run_prototype_tests(self.prototype, self.test_config)
        
        # Verify system was initialized and cleaned up even with error
        self.mock_system.initialize.assert_called_once()
        self.mock_system.cleanup.assert_called_once()
        
        # Verify error status
        self.assertEqual(results['status'], 'failed')
        self.assertIn('error', results)
        self.assertEqual(results['error'], "Simulated test failure")


if __name__ == "__main__":
    unittest.main()