import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from typing import Dict, Any

from src.core.integration.neuromorphic_system import NeuromorphicSystem

class TestNeuromorphicSystem(unittest.TestCase):
    """Test cases for the NeuromorphicSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = NeuromorphicSystem()
        
        # Create mock components
        self.sensor_component = MagicMock()
        self.processor_component = MagicMock()
        self.output_component = MagicMock()
        
        # Configure mock components
        self.sensor_component.process.return_value = {"sensor_data": np.random.random(10)}
        self.processor_component.process.return_value = {"neuron_states": {i: 0.5 for i in range(10)}}
        self.output_component.process.return_value = {"output": 0.75}
        
        # Mock learning algorithm
        self.learning_algorithm = MagicMock()
        self.learning_algorithm.update_weights.return_value = {"connection1": 0.5}
    
    def test_initialization(self):
        """Test system initialization."""
        # Test default initialization
        self.assertIsNone(self.system.hardware)
        self.assertEqual(self.system.components, {})
        self.assertEqual(self.system.connections, {})
        self.assertEqual(self.system.learning_algorithms, {})
        self.assertEqual(self.system.data_buffers, {})
        self.assertFalse(self.system.running)
        self.assertEqual(self.system.timestep, 0)
        
        # Test with hardware interface
        mock_hardware = MagicMock()
        system_with_hardware = NeuromorphicSystem(mock_hardware)
        self.assertEqual(system_with_hardware.hardware, mock_hardware)
    
    def test_add_component(self):
        """Test adding components to the system."""
        # Add a component
        result = self.system.add_component("sensor", self.sensor_component)
        self.assertTrue(result)
        self.assertIn("sensor", self.system.components)
        self.assertIn("sensor", self.system.data_buffers)
        
        # Try adding a component with the same name
        result = self.system.add_component("sensor", self.sensor_component)
        self.assertFalse(result)
        
        # Add more components
        self.system.add_component("processor", self.processor_component)
        self.system.add_component("output", self.output_component)
        self.assertEqual(len(self.system.components), 3)
    
    def test_add_learning_algorithm(self):
        """Test adding learning algorithms to the system."""
        # Add a learning algorithm
        result = self.system.add_learning_algorithm("test_algo", self.learning_algorithm)
        self.assertTrue(result)
        self.assertIn("test_algo", self.system.learning_algorithms)
        
        # Try adding an algorithm with the same name
        result = self.system.add_learning_algorithm("test_algo", self.learning_algorithm)
        self.assertFalse(result)
    
    def test_connect(self):
        """Test connecting components."""
        # Add components first
        self.system.add_component("sensor", self.sensor_component)
        self.system.add_component("processor", self.processor_component)
        
        # Connect components
        result = self.system.connect("sensor", "processor")
        self.assertTrue(result)
        
        # Check connection was created
        connection_id = "sensor->processor"
        self.assertIn(connection_id, self.system.connections)
        self.assertEqual(self.system.connections[connection_id]["source"], "sensor")
        self.assertEqual(self.system.connections[connection_id]["target"], "processor")
        
        # Test connecting non-existent components
        result = self.system.connect("sensor", "nonexistent")
        self.assertFalse(result)
    
    def test_initialize(self):
        """Test system initialization."""
        # Add components
        self.system.add_component("sensor", self.sensor_component)
        self.system.add_component("processor", self.processor_component)
        
        # Add learning algorithm
        self.system.add_learning_algorithm("test_algo", self.learning_algorithm)
        
        # Initialize system
        result = self.system.initialize()
        self.assertTrue(result)
        
        # Check if component initialize methods were called
        self.sensor_component.initialize.assert_called_once()
        self.processor_component.initialize.assert_called_once()
        
        # Check if learning algorithm reset was called (if it has the method)
        if hasattr(self.learning_algorithm, 'reset'):
            self.learning_algorithm.reset.assert_called_once()
    
    def test_process_data(self):
        """Test processing data through the system."""
        # Set up a simple system
        self.system.add_component("sensor", self.sensor_component)
        self.system.add_component("processor", self.processor_component)
        self.system.add_component("output", self.output_component)
        
        # Connect components
        self.system.connect("sensor", "processor")
        self.system.connect("processor", "output")
        
        # Initialize
        self.system.initialize()
        
        # Process data
        input_data = {"sensor": {"raw_data": [0.1, 0.2, 0.3]}}
        output = self.system.process_data(input_data)
        
        # Check if components were processed
        self.sensor_component.process.assert_called_once()
        self.processor_component.process.assert_called_once()
        self.output_component.process.assert_called_once()
        
        # Check output
        self.assertIn("sensor", output)
        self.assertIn("processor", output)
        self.assertIn("output", output)
    
    def test_run(self):
        """Test running the system for multiple timesteps."""
        # Set up a simple system
        self.system.add_component("sensor", self.sensor_component)
        self.system.add_component("processor", self.processor_component)
        
        # Initialize
        self.system.initialize()
        
        # Define input provider
        def input_provider(step):
            return {"sensor": {"step": step, "data": [0.1, 0.2, 0.3]}}
        
        # Run for 5 steps
        results = self.system.run(5, input_provider)
        
        # Check results
        self.assertEqual(len(results["sensor"]), 5)
        self.assertEqual(len(results["processor"]), 5)
        
        # Check if process was called 5 times
        self.assertEqual(self.sensor_component.process.call_count, 5)
        self.assertEqual(self.processor_component.process.call_count, 5)
    
    def test_gather_inputs(self):
        """Test gathering inputs for a component."""
        # Set up a simple system
        self.system.add_component("sensor", self.sensor_component)
        self.system.add_component("processor", self.processor_component)
        
        # Connect components
        self.system.connect("sensor", "processor")
        
        # Add data to buffers
        self.system.data_buffers["sensor"]["output"] = {"sensor_data": [0.1, 0.2, 0.3]}
        
        # Gather inputs for processor
        processed = {"sensor"}
        inputs = self.system._gather_inputs("processor", processed)
        
        # Check inputs
        self.assertIn("sensor", inputs)
        self.assertEqual(inputs["sensor"], {"sensor_data": [0.1, 0.2, 0.3]})

if __name__ == "__main__":
    unittest.main()