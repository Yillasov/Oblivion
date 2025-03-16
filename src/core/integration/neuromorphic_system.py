"""
Simple System Integration Framework

Provides a unified interface for integrating various neuromorphic components
and learning algorithms into a cohesive system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import time

from src.core.utils.logging_framework import get_logger
# Import NeuromorphicInterface from a type-safe interface
from typing import Protocol, runtime_checkable

@runtime_checkable
class NeuromorphicInterface(Protocol):
    """Protocol defining the interface for neuromorphic hardware."""
    def initialize(self) -> None: ...
    def cleanup(self) -> None: ...

logger = get_logger("neuromorphic_system")


class NeuromorphicSystem:
    """
    Simple integration framework for neuromorphic components.
    
    Provides a unified interface for managing neuromorphic hardware,
    learning algorithms, and data flow between components.
    """
    
    def __init__(self, hardware_interface: Optional[NeuromorphicInterface] = None):
        """
        Initialize the neuromorphic system.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware (optional)
        """
        self.hardware = hardware_interface
        self.components = {}
        self.connections = {}
        self.learning_algorithms = {}
        self.data_buffers = {}
        self.running = False
        self.timestep = 0
        
        logger.info("Initialized neuromorphic system integration framework")
    
    def add_component(self, name: str, component: Any) -> bool:
        """
        Add a component to the system.
        
        Args:
            name: Unique identifier for the component
            component: The component object
            
        Returns:
            bool: True if successful, False if name already exists
        """
        if name in self.components:
            logger.warning(f"Component '{name}' already exists")
            return False
        
        self.components[name] = component
        self.data_buffers[name] = {}
        logger.info(f"Added component '{name}' to system")
        return True
    
    def add_learning_algorithm(self, name: str, algorithm: Any) -> bool:
        """
        Add a learning algorithm to the system.
        
        Args:
            name: Unique identifier for the algorithm
            algorithm: The learning algorithm object
            
        Returns:
            bool: True if successful, False if name already exists
        """
        if name in self.learning_algorithms:
            logger.warning(f"Learning algorithm '{name}' already exists")
            return False
        
        self.learning_algorithms[name] = algorithm
        logger.info(f"Added learning algorithm '{name}' to system")
        return True
    
    def connect(self, source: str, target: str, 
               connection_type: str = "data", 
               transform: Optional[Callable] = None) -> bool:
        """
        Connect two components in the system.
        
        Args:
            source: Name of source component
            target: Name of target component
            connection_type: Type of connection (e.g., "data", "control")
            transform: Optional function to transform data during transfer
            
        Returns:
            bool: True if successful, False otherwise
        """
        if source not in self.components:
            logger.error(f"Source component '{source}' not found")
            return False
        
        if target not in self.components:
            logger.error(f"Target component '{target}' not found")
            return False
        
        connection_id = f"{source}->{target}"
        self.connections[connection_id] = {
            "source": source,
            "target": target,
            "type": connection_type,
            "transform": transform
        }
        
        logger.info(f"Connected {source} to {target} ({connection_type})")
        return True
    
    def initialize(self) -> bool:
        """
        Initialize the system and all components.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Initialize hardware if available
            if self.hardware is not None:
                self.hardware.initialize()
            
            # Initialize all components
            for name, component in self.components.items():
                if hasattr(component, 'initialize'):
                    component.initialize()
            
            # Reset learning algorithms
            for name, algorithm in self.learning_algorithms.items():
                if hasattr(algorithm, 'reset'):
                    algorithm.reset()
            
            self.timestep = 0
            logger.info("System initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}")
            return False
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through the system for one timestep.
        
        Args:
            input_data: Dictionary mapping component names to input data
            
        Returns:
            Dict[str, Any]: Output data from all components
        """
        # Store input data in buffers
        for name, data in input_data.items():
            if name in self.data_buffers:
                self.data_buffers[name]["input"] = data
            else:
                logger.warning(f"Input provided for unknown component: {name}")
        
        # Process components in dependency order
        processed_components = set()
        output_data = {}
        
        while len(processed_components) < len(self.components):
            progress_made = False
            
            for name, component in self.components.items():
                if name in processed_components:
                    continue
                
                # Check if all dependencies are processed
                dependencies_met = True
                for conn_id, conn in self.connections.items():
                    if conn["target"] == name and conn["source"] not in processed_components:
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    # Gather inputs from connected components
                    component_inputs = {}
                    for conn_id, conn in self.connections.items():
                        if conn["target"] == name:
                            source_name = conn["source"]
                            if source_name in self.data_buffers and "output" in self.data_buffers[source_name]:
                                data = self.data_buffers[source_name]["output"]
                                
                                # Apply transform if specified
                                if conn["transform"] is not None:
                                    data = conn["transform"](data)
                                
                                component_inputs[source_name] = data
                    
                    # Add direct inputs
                    if name in self.data_buffers and "input" in self.data_buffers[name]:
                        component_inputs["direct"] = self.data_buffers[name]["input"]
                    
                    # Process component
                    if hasattr(component, 'process'):
                        try:
                            result = component.process(component_inputs)
                            self.data_buffers[name]["output"] = result
                            output_data[name] = result
                        except Exception as e:
                            logger.error(f"Error processing component '{name}': {str(e)}")
                            # Provide empty result to avoid breaking the pipeline
                            self.data_buffers[name]["output"] = {}
                            output_data[name] = {}
                    
                    processed_components.add(name)
                    progress_made = True
            
            # If no progress was made in this iteration, we have a circular dependency
            if not progress_made and len(processed_components) < len(self.components):
                unprocessed = set(self.components.keys()) - processed_components
                logger.error(f"Circular dependency detected. Unprocessed components: {unprocessed}")
                break
        
        # Update learning algorithms
        for name, algorithm in self.learning_algorithms.items():
            try:
                if hasattr(algorithm, 'update_weights'):
                    # Get relevant component outputs for this algorithm
                    algorithm_data = {comp_name: self.data_buffers[comp_name].get("output", None) 
                                    for comp_name in self.components 
                                    if comp_name in self.data_buffers}
                    
                    # Update algorithm state
                    if hasattr(algorithm, 'record_activity'):
                        for comp_name, data in algorithm_data.items():
                            if data is not None and hasattr(data, 'items'):
                                for neuron_id, activity in data.items():
                                    algorithm.record_activity(neuron_id, activity)
                    
                    # If the algorithm has connections to update, call update_weights
                    # This assumes connections are stored elsewhere and passed to the algorithm
                    if hasattr(self, 'neural_connections'):
                        updated_connections = algorithm.update_weights(self.neural_connections, self.timestep)
                        self.neural_connections = updated_connections
            except Exception as e:
                logger.error(f"Error updating learning algorithm '{name}': {str(e)}")
        
        self.timestep += 1
        return output_data
    
    def run(self, steps: int, input_provider: Callable[[int], Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Run the system for multiple timesteps.
        
        Args:
            steps: Number of timesteps to run
            input_provider: Function that returns input data for each timestep
            
        Returns:
            Dict[str, List[Any]]: Collected outputs from all components
        """
        self.running = True
        collected_outputs = {name: [] for name in self.components}
        
        try:
            for step in range(steps):
                if not self.running:
                    logger.info("System execution stopped")
                    break
                
                # Get input for this timestep
                input_data = input_provider(step)
                
                # Process data
                outputs = self.process_data(input_data)
                
                # Collect outputs
                for name, output in outputs.items():
                    collected_outputs[name].append(output)
                
                logger.debug(f"Completed timestep {step+1}/{steps}")
            
            logger.info(f"System execution completed ({steps} steps)")
            return collected_outputs
            
        except Exception as e:
            logger.error(f"Error during system execution: {str(e)}")
            self.running = False
            return collected_outputs
    
    def stop(self):
        """Stop system execution."""
        self.running = False
        logger.info("System execution stop requested")
    
    def cleanup(self):
        """Clean up resources used by the system."""
        try:
            # Clean up hardware resources
            if self.hardware is not None:
                self.hardware.cleanup()
            
            # Clean up components
            for name, component in self.components.items():
                if hasattr(component, 'cleanup'):
                    component.cleanup()
            
            logger.info("System resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during system cleanup: {str(e)}")


# Example usage
def create_example_system():
    """Create a simple example neuromorphic system."""
    from src.core.neuromorphic.simple_stdp import SimpleSTDP
    from src.core.neuromorphic.simple_hebbian import SimpleHebbian
    
    # Create system
    system = NeuromorphicSystem()
    
    # Add components (these would be your actual components)
    class SensorComponent:
        def initialize(self):
            pass
        
        def process(self, inputs):
            # Process sensor data
            return {"sensor_data": np.random.random(10)}
    
    class ProcessorComponent:
        def initialize(self):
            self.neurons = {i: 0.0 for i in range(10)}
            
        def process(self, inputs):
            # Process input data and update neuron states
            if "sensor" in inputs:
                sensor_data = inputs["sensor"]
                for i, value in enumerate(sensor_data.get("sensor_data", [])):
                    if i in self.neurons:
                        self.neurons[i] = value
            return self.neurons
    
    class OutputComponent:
        def initialize(self):
            pass
            
        def process(self, inputs):
            # Generate output based on processor state
            if "processor" in inputs:
                neuron_states = inputs["processor"]
                return {"output": sum(neuron_states.values()) / len(neuron_states)}
            return {"output": 0.0}
    
    # Add components to system
    system.add_component("sensor", SensorComponent())
    system.add_component("processor", ProcessorComponent())
    system.add_component("output", OutputComponent())
    
    # Add learning algorithms
    system.add_learning_algorithm("stdp", SimpleSTDP())
    system.add_learning_algorithm("hebbian", SimpleHebbian())
    
    # Connect components
    system.connect("sensor", "processor")
    system.connect("processor", "output")
    
    # Initialize system
    system.initialize()
    
    return system


    def set_neural_connections(self, connections: Dict[Tuple[int, int], float]):
        """
        Set neural connections for learning algorithms to update.
        
        Args:
            connections: Dictionary mapping (pre_id, post_id) to weight
        """
        self.neural_connections = connections
        logger.info(f"Set {len(connections)} neural connections for learning")