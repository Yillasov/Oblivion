"""
Neuromorphic System Integration Framework

Provides a unified interface for integrating neuromorphic components
and learning algorithms into a cohesive system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import time

from src.core.utils.logging_framework import get_logger
from typing import Protocol, runtime_checkable

from src.core.hardware.fault_tolerance import FaultToleranceManager, FaultType, FaultSeverity
from src.core.hardware.hardware_monitor import NeuromorphicHardwareMonitor, HardwareMetricType

@runtime_checkable
class NeuromorphicInterface(Protocol):
    """Protocol defining the interface for neuromorphic hardware."""
    def initialize(self) -> None: ...
    def cleanup(self) -> None: ...
    def get_info(self) -> Dict[str, Any]: ...
    
logger = get_logger("neuromorphic_system")


class NeuromorphicSystem:
    """
    Integration framework for neuromorphic components.
    
    Provides a unified interface for managing neuromorphic hardware,
    learning algorithms, and data flow between components.
    """
    
    # Add these imports at the top of the file
    from src.core.hardware.fault_tolerance import FaultToleranceManager, FaultType, FaultSeverity
    from src.core.hardware.hardware_monitor import NeuromorphicHardwareMonitor, HardwareMetricType
    
    # Add these attributes to the __init__ method
    def __init__(self, hardware_interface: Optional[NeuromorphicInterface] = None):
            """Initialize the neuromorphic system."""
            self.hardware = hardware_interface
            self.components = {}
            self.connections = {}
            self.learning_algorithms = {}
            self.data_buffers = {}
            self.running = False
            self.timestep = 0
            self.neural_connections = {}
            self.hardware_interfaces = {}  # Add this line to store hardware interfaces
            
            # Fault tolerance components
            self.fault_manager = FaultToleranceManager()
            self.hardware_monitor = NeuromorphicHardwareMonitor(self.fault_manager)
            self.fault_tolerance_enabled = True
            
            logger.info("Initialized neuromorphic system integration framework")
    
    # Add these methods to the NeuromorphicSystem class
    def enable_fault_tolerance(self, enabled: bool = True):
        """
        Enable or disable fault tolerance mechanisms.
        
        Args:
            enabled: Whether fault tolerance should be enabled
        """
        self.fault_tolerance_enabled = enabled
        
        if enabled:
            # Start monitoring if system is running
            if self.running:
                self.hardware_monitor.start_monitoring()
            logger.info("Fault tolerance mechanisms enabled")
        else:
            # Stop monitoring
            self.hardware_monitor.stop_monitoring()
            logger.info("Fault tolerance mechanisms disabled")
    
    def register_hardware_component(self, 
                                hardware_id: str, 
                                hardware_component: Any,
                                hardware_type: str,
                                is_critical: bool = False,
                                redundancy_group: Optional[str] = None) -> bool:
        """
        Register hardware component for fault tolerance monitoring.
        
        Args:
            hardware_id: Unique identifier for the hardware
            hardware_component: The hardware component object
            hardware_type: Type of hardware (e.g., "loihi", "spinnaker")
            is_critical: Whether this is a critical component
            redundancy_group: Optional group name for redundant components
            
        Returns:
            bool: Success status
        """
        if not self.fault_tolerance_enabled:
            logger.warning("Cannot register hardware: Fault tolerance is disabled")
            return False
        
        # Register with hardware monitor
        success = self.hardware_monitor.register_hardware(
            hardware_id,
            hardware_component,
            hardware_type,
            is_critical,
            redundancy_group
        )
        
        if success:
            # Store in hardware interfaces
            self.hardware_interfaces[hardware_id] = {
                "component": hardware_component,
                "type": hardware_type,
                "is_critical": is_critical,
                "redundancy_group": redundancy_group
            }
        
        return success
    
    def report_hardware_metrics(self, hardware_id: str, metrics: Dict[str, float]) -> bool:
        """
        Report hardware metrics for monitoring.
        
        Args:
            hardware_id: Hardware identifier
            metrics: Dictionary of metric values (keys should match HardwareMetricType values)
            
        Returns:
            bool: Success status
        """
        if not self.fault_tolerance_enabled:
            return False
        
        # Convert string metric types to enum
        enum_metrics = {}
        for metric_name, value in metrics.items():
            try:
                metric_type = HardwareMetricType(metric_name)
                enum_metrics[metric_type] = value
            except ValueError:
                logger.warning(f"Unknown metric type: {metric_name}")
        
        # Update metrics in hardware monitor
        return self.hardware_monitor.update_metrics(hardware_id, enum_metrics)
    
    def report_hardware_fault(self, 
                            hardware_id: str, 
                            fault_type_str: str, 
                            severity_str: str,
                            details: Dict[str, Any] = {}) -> Optional[str]:
        """
        Report a hardware fault.
        
        Args:
            hardware_id: Hardware identifier
            fault_type_str: Type of fault (should match FaultType values)
            severity_str: Severity level (should match FaultSeverity values)
            details: Additional details about the fault
            
        Returns:
            Optional[str]: Fault ID if successfully reported
        """
        if not self.fault_tolerance_enabled:
            return None
        
        try:
            fault_type = FaultType(fault_type_str)
            severity = FaultSeverity(severity_str)
        except ValueError:
            logger.warning(f"Invalid fault type or severity: {fault_type_str}, {severity_str}")
            return None
        
        return self.fault_manager.report_fault(
            hardware_id,
            fault_type,
            severity,
            details
        )
    
    def resolve_hardware_fault(self, fault_id: str, resolution_strategy: str) -> bool:
        """
        Resolve a hardware fault.
        
        Args:
            fault_id: Fault identifier
            resolution_strategy: Description of how the fault was resolved
            
        Returns:
            bool: Success status
        """
        if not self.fault_tolerance_enabled:
            return False
        
        return self.fault_manager.resolve_fault(fault_id, resolution_strategy)
    
    def get_hardware_status(self, hardware_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get hardware status information.
        
        Args:
            hardware_id: Optional hardware identifier (if None, returns all hardware status)
            
        Returns:
            Dict[str, Any]: Status information
        """
        if not self.fault_tolerance_enabled:
            return {"error": "Fault tolerance is disabled"}
        
        if hardware_id:
            return self.hardware_monitor.get_hardware_status(hardware_id)
        else:
            return {
                "system_health": self.fault_manager.get_system_health(),
                "hardware_status": self.hardware_monitor.get_all_hardware_status()
            }
    
    # Modify the start and stop methods to include fault tolerance
    def start(self):
        """Start system execution."""
        if not self.running:
            self.running = True
            
            # Start fault tolerance monitoring if enabled
            if self.fault_tolerance_enabled:
                self.hardware_monitor.start_monitoring()
            
            logger.info("System execution started")
            return True
        return False
    
    def halt_execution(self):
        """Stop system execution."""
        self.running = False
        
        # Stop fault tolerance monitoring if enabled
        if self.fault_tolerance_enabled:
            self.hardware_monitor.stop_monitoring()
        
        logger.info("System execution stop requested")
    
    def add_hardware_interface(self, name: str, interface: Any) -> bool:
        """
        Add a hardware interface to the system.
        
        Args:
            name: Name of the hardware interface
            interface: Hardware interface object
            
        Returns:
            bool: True if the interface was added successfully
        """
        if name in self.hardware_interfaces:
            logger.warning(f"Hardware interface '{name}' already exists")
            return False
        
        self.hardware_interfaces[name] = interface
        logger.info(f"Added hardware interface '{name}' to system")
        return True
    
    def add_component(self, name: str, component: Any) -> bool:
        """Add a component to the system."""
        if name in self.components:
            logger.warning(f"Component '{name}' already exists")
            return False
        
        self.components[name] = component
        self.data_buffers[name] = {}
        logger.info(f"Added component '{name}' to system")
        return True
    
    def add_learning_algorithm(self, name: str, algorithm: Any) -> bool:
        """Add a learning algorithm to the system."""
        if name in self.learning_algorithms:
            logger.warning(f"Learning algorithm '{name}' already exists")
            return False
        
        self.learning_algorithms[name] = algorithm
        logger.info(f"Added learning algorithm '{name}' to system")
        return True
    
    def connect(self, source: str, target: str, 
               connection_type: str = "data", 
               transform: Optional[Callable] = None) -> bool:
        """Connect two components in the system."""
        if source not in self.components or target not in self.components:
            logger.error(f"Connection failed: component not found")
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
        """Initialize the system and all components."""
        try:
            # Initialize hardware if available
            if self.hardware is not None:
                self.hardware.initialize()
            
            # Initialize components and reset learning algorithms
            for name, component in self.components.items():
                if hasattr(component, 'initialize'):
                    component.initialize()
            
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
        """Process data through the system for one timestep."""
        # Store input data in buffers
        for name, data in input_data.items():
            if name in self.data_buffers:
                self.data_buffers[name]["input"] = data
        
        # Process components in dependency order
        processed = set()
        output_data = {}
        
        # Simple topological sort and processing
        while len(processed) < len(self.components):
            progress_made = False
            
            for name, component in self.components.items():
                if name in processed:
                    continue
                
                # Check if dependencies are met
                deps_met = all(
                    conn["source"] in processed 
                    for conn_id, conn in self.connections.items() 
                    if conn["target"] == name
                )
                
                if deps_met:
                    # Gather inputs and process component
                    try:
                        component_inputs = self._gather_inputs(name, processed)
                        if hasattr(component, 'process'):
                            result = component.process(component_inputs)
                            self.data_buffers[name]["output"] = result
                            output_data[name] = result
                    except Exception as e:
                        logger.error(f"Error processing component '{name}': {str(e)}")
                        self.data_buffers[name]["output"] = {}
                        output_data[name] = {}
                    
                    processed.add(name)
                    progress_made = True
            
            # Check for circular dependencies
            if not progress_made and len(processed) < len(self.components):
                unprocessed = set(self.components.keys()) - processed
                logger.error(f"Circular dependency detected: {unprocessed}")
                break
        
        # Update learning algorithms
        self._update_learning_algorithms()
        
        self.timestep += 1
        return output_data
    
    def _gather_inputs(self, target_name: str, processed_components: set) -> Dict[str, Any]:
        """Gather inputs for a component from its connections."""
        inputs = {}
        
        # Get inputs from connections
        for conn_id, conn in self.connections.items():
            if conn["target"] == target_name and conn["source"] in processed_components:
                source_name = conn["source"]
                if "output" in self.data_buffers.get(source_name, {}):
                    data = self.data_buffers[source_name]["output"]
                    
                    # Apply transform if specified
                    if conn["transform"] is not None:
                        data = conn["transform"](data)
                    
                    inputs[source_name] = data
        
        # Add direct inputs
        if "input" in self.data_buffers.get(target_name, {}):
            inputs["direct"] = self.data_buffers[target_name]["input"]
        
        return inputs
    
    def _update_learning_algorithms(self):
        """Update all learning algorithms with current system state."""
        for name, algorithm in self.learning_algorithms.items():
            try:
                if hasattr(algorithm, 'update_weights'):
                    # Get component outputs for this algorithm
                    algorithm_data = {
                        comp_name: self.data_buffers.get(comp_name, {}).get("output", {})
                        for comp_name in self.components
                    }
                    
                    # Record activity if supported
                    if hasattr(algorithm, 'record_activity'):
                        for comp_name, data in algorithm_data.items():
                            if isinstance(data, dict):  # Ensure data is a dictionary before iterating
                                for neuron_id, activity in data.items():
                                    algorithm.record_activity(neuron_id, activity)
                    
                    # Update weights if connections exist
                    if self.neural_connections:
                        updated_connections = algorithm.update_weights(
                            self.neural_connections, self.timestep
                        )
                        # Verify the returned connections are valid before assigning
                        if updated_connections is not None:
                            self.neural_connections = updated_connections
            except Exception as e:
                logger.error(f"Error updating algorithm '{name}': {str(e)}")
    
    def run(self, steps: int, input_provider: Callable[[int], Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Run the system for multiple timesteps."""
        self.running = True
        collected_outputs = {name: [] for name in self.components}
        
        try:
            for step in range(steps):
                if not self.running:
                    break
                
                outputs = self.process_data(input_provider(step))
                
                # Collect outputs
                for name, output in outputs.items():
                    collected_outputs[name].append(output)
            
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
            if self.hardware is not None:
                self.hardware.cleanup()
            
            for name, component in self.components.items():
                if hasattr(component, 'cleanup'):
                    component.cleanup()
            
            logger.info("System resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during system cleanup: {str(e)}")
    
    def set_neural_connections(self, connections: Dict[Tuple[int, int], float]):
        """Set neural connections for learning algorithms to update."""
        self.neural_connections = connections
        logger.info(f"Set {len(connections)} neural connections for learning")
    
    def switch_hardware(self, new_hardware_interface: NeuromorphicInterface) -> bool:
        """Switch to a new hardware interface."""
        try:
            logger.info("Switching hardware interface...")
            
            # Shutdown current hardware
            if self.hardware:
                self.hardware.cleanup()
            
            # Initialize new hardware
            self.hardware = new_hardware_interface
            self.hardware.initialize()
            
            # Reinitialize components and learning algorithms
            for name, component in self.components.items():
                if hasattr(component, 'initialize'):
                    component.initialize()
            
            for name, algorithm in self.learning_algorithms.items():
                if hasattr(algorithm, 'reset'):
                    algorithm.reset()
            
            logger.info("Hardware interface switched successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to switch hardware interface: {str(e)}")
            return False
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get information about the current hardware interface."""
        if self.hardware:
            return self.hardware.get_info()
        else:
            logger.warning("No hardware interface available")
            return {}


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


# Add this method to the NeuromorphicSystem class

def train(self, training_data: Dict[str, Any]) -> bool:
    """
    Train the neuromorphic system with provided data.
    
    Args:
        training_data: Dictionary containing training data and parameters
            - 'algorithm': Learning algorithm to use
            - 'algorithm_name': Name for the algorithm
            - 'inputs': Input data for training
            - 'targets': Target outputs for supervised learning
            - 'component_data': Component-specific training data
            
    Returns:
        bool: Success status
    """
    try:
        # Add a learning algorithm if provided
        if "algorithm" in training_data and "algorithm_name" in training_data:
            self.add_learning_algorithm(
                training_data["algorithm_name"], 
                training_data["algorithm"]
            )
        
        # Process training data through the system
        if "inputs" in training_data:
            self.process_data(training_data["inputs"])
            
        # If there are specific components to train
        if "component_data" in training_data:
            for component_name, data in training_data["component_data"].items():
                if component_name in self.components:
                    component = self.components[component_name]
                    if hasattr(component, 'train'):
                        component.train(data)
        
        # Update learning algorithms with target data if available
        if "targets" in training_data and self.learning_algorithms:
            for name, algorithm in self.learning_algorithms.items():
                if hasattr(algorithm, 'train'):
                    algorithm.train(training_data["inputs"], training_data["targets"])
        
        logger.info("Neuromorphic system training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error training neuromorphic system: {str(e)}")
        return False
