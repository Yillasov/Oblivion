"""
Hardware Compatibility Layer

Provides a consistent API for interacting with different neuromorphic hardware platforms,
abstracting away hardware-specific details and ensuring code portability.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from functools import wraps

from src.core.utils.logging_framework import get_logger
from src.core.hardware.exceptions import (
    NeuromorphicHardwareError,
    HardwareInitializationError,
    HardwareAllocationError,
    HardwareSimulationError,
    HardwareCommunicationError,
    UnsupportedFeatureError
)
from src.core.hardware.unified_interface import NeuromorphicHardwareInterface
from src.core.hardware.error_codes import HardwareErrorCode, HardwareErrorInfo
from src.core.hardware.error_handler import handle_hardware_error
from src.core.hardware.recovery_strategies import attempt_recovery
from src.core.hardware.capability_negotiator import create_capability_negotiator

logger = get_logger("hardware_compatibility")


def hardware_operation(operation_name: str):
    """
    Decorator for hardware operations to provide consistent error handling.
    
    Args:
        operation_name: Name of the operation for logging
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                logger.debug(f"Executing {operation_name} on {self.hardware_type}")
                return func(self, *args, **kwargs)
            except HardwareInitializationError as e:
                logger.error(f"{operation_name} failed: hardware not initialized - {str(e)}")
                raise
            except HardwareAllocationError as e:
                logger.error(f"{operation_name} failed: resource allocation error - {str(e)}")
                raise
            except HardwareSimulationError as e:
                logger.error(f"{operation_name} failed: simulation error - {str(e)}")
                raise
            except HardwareCommunicationError as e:
                logger.error(f"{operation_name} failed: communication error - {str(e)}")
                raise
            except UnsupportedFeatureError as e:
                logger.error(f"{operation_name} failed: unsupported feature - {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during {operation_name}: {str(e)}")
                raise NeuromorphicHardwareError(f"Unexpected error during {operation_name}: {str(e)}")
        return wrapper
    return decorator


class HardwareCompatibilityLayer:
    """
    Provides a consistent API for neuromorphic hardware platforms.
    
    This layer ensures that code written for one hardware platform can run on
    another with minimal or no changes, abstracting away hardware-specific details.
    """
    
    def __init__(self, hardware_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hardware compatibility layer.
        
        Args:
            hardware_type: Type of hardware ('loihi', 'spinnaker', 'truenorth'), or None for auto-detection
            config: Optional configuration parameters
        """
        self.interface = NeuromorphicHardwareInterface(hardware_type, config)
        self.hardware_type = self.interface.hardware_type
        self.config = self.interface.config
        self.initialized = False
        
        # Hardware capabilities cache
        self._capabilities = None
        
        # Initialize hardware-specific optimizations
        self._init_hardware_optimizations()
        
        # Initialize capability negotiator
        hardware_info = self.interface.get_hardware_info()
        self.capability_negotiator = create_capability_negotiator(self.hardware_type, hardware_info)
    
    def _init_hardware_optimizations(self):
        """Initialize hardware-specific optimizations based on hardware type."""
        self.hw_optimizations = {
            # Loihi-specific optimizations
            "loihi": {
                "neuron_grouping": True,       # Group similar neurons for better core utilization
                "weight_precision": 8,          # 8-bit weight precision
                "max_fan_in": 4096,            # Maximum fan-in per neuron
                "compartment_types": [0, 1, 2], # Supported compartment types
                "learning_rules": ["STDP", "Hebbian", "Homeostasis"],
                "power_modes": ["high_performance", "balanced", "power_saving"],
                "data_flow_optimization": "phase_encoding"  # Optimize data flow using phase encoding
            },
            # SpiNNaker-specific optimizations
            "spinnaker": {
                "neuron_grouping": False,      # Less benefit from grouping
                "weight_precision": 16,         # 16-bit weight precision
                "max_fan_in": 16384,           # Higher fan-in capacity
                "packet_routing": "multicast", # Efficient multicast routing
                "learning_rules": ["STDP", "BCM", "Reinforcement"],
                "power_modes": ["standard", "power_saving"],
                "data_flow_optimization": "multicast_routing"  # Optimize data flow using multicast routing
            },
            # TrueNorth-specific optimizations
            "truenorth": {
                "neuron_grouping": True,       # Core-based architecture benefits from grouping
                "weight_precision": 1,          # Binary weights
                "max_fan_in": 256,             # Limited fan-in
                "deterministic": True,         # Fully deterministic operation
                "learning_rules": ["Offline"],  # Primarily offline learning
                "power_modes": ["ultra_low_power"],
                "data_flow_optimization": "binary_encoding"  # Optimize data flow using binary encoding
            }
        }
        
        # Set current optimizations based on hardware type
        self.current_optimizations = self.hw_optimizations.get(
            self.hardware_type, 
            self.hw_optimizations.get("loihi", {})  # Default to Loihi if unknown
        )
    
    @hardware_operation("optimize_data_flow")
    def optimize_data_flow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize data flow between components based on hardware-specific strategies.
        
        Args:
            data: Data to be optimized
            
        Returns:
            Dict[str, Any]: Optimized data
        """
        optimization_strategy = self.current_optimizations.get("data_flow_optimization")
        
        if optimization_strategy == "phase_encoding":
            return self._optimize_phase_encoding(data)
        elif optimization_strategy == "multicast_routing":
            return self._optimize_multicast_routing(data)
        elif optimization_strategy == "binary_encoding":
            return self._optimize_binary_encoding(data)
        else:
            logger.warning(f"No specific data flow optimization strategy for {self.hardware_type}")
            return data
    
    def _optimize_phase_encoding(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize data flow using phase encoding."""
        logger.info("Applying phase encoding optimization")
        # Implement phase encoding optimization logic here
        # Example: Adjust spike timings for efficient transmission
        return data
    
    def _optimize_multicast_routing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize data flow using multicast routing."""
        logger.info("Applying multicast routing optimization")
        # Implement multicast routing optimization logic here
        # Example: Efficiently route packets to multiple destinations
        return data
    
    def _optimize_binary_encoding(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize data flow using binary encoding."""
        logger.info("Applying binary encoding optimization")
        # Implement binary encoding optimization logic here
        # Example: Use binary representation for efficient data transfer
        return data
    
    @hardware_operation("initialize")
    @handle_hardware_error
    def _apply_hardware_optimizations(self):
        """Apply hardware-specific optimizations with error handling."""
        try:
            if self.hardware_type == "loihi":
                self._optimize_for_loihi()
            elif self.hardware_type == "spinnaker":
                self._optimize_for_spinnaker()
            elif self.hardware_type == "truenorth":
                self._optimize_for_truenorth()
        except Exception as e:
            error_info = HardwareErrorInfo(
                code=HardwareErrorCode.INITIALIZATION_FAILED,
                message=f"Hardware optimization failed: {str(e)}",
                hardware_type=self.hardware_type
            )
            if attempt_recovery(self.hardware_type, error_info.code, error_info.details):
                # Retry optimization after recovery
                self._apply_hardware_optimizations()
            else:
                raise HardwareInitializationError(f"Failed to optimize for {self.hardware_type}: {str(e)}")

    @handle_hardware_error
    def initialize(self) -> bool:
        """Initialize hardware with enhanced error handling."""
        try:
            self._apply_hardware_optimizations()
            result = self.interface.initialize()
            self.initialized = result
            return result
        except Exception as e:
            error_info = HardwareErrorInfo(
                code=HardwareErrorCode.INITIALIZATION_FAILED,
                message=f"Hardware initialization failed: {str(e)}",
                hardware_type=self.hardware_type
            )
            if attempt_recovery(self.hardware_type, error_info.code, error_info.details):
                return self.initialize()  # Retry initialization after recovery
            return False
    
    def _optimize_for_loihi(self):
        """Apply Loihi-specific optimizations."""
        # Ensure neuron counts are multiples of 4 for better core utilization
        if "neuron_count" in self.config:
            self.config["neuron_count"] = ((self.config["neuron_count"] + 3) // 4) * 4
        
        # Set optimal core voltage for power/performance balance
        if "core_voltage" not in self.config:
            self.config["core_voltage"] = 1.2  # V
        
        # Enable phase encoding for efficient spike representation
        self.config["phase_encoding"] = self.config.get("phase_encoding", True)
        
        # Set optimal learning parameters
        if "learning" in self.config:
            self.config["learning"]["weight_precision"] = 8
            self.config["learning"]["use_compartments"] = True
    
    def _optimize_for_spinnaker(self):
        """Apply SpiNNaker-specific optimizations."""
        # Optimize packet routing
        self.config["packet_routing"] = self.config.get("packet_routing", "multicast")
        
        # Set appropriate timing parameters
        self.config["time_scale_factor"] = self.config.get("time_scale_factor", 1.0)
        
        # Configure memory allocation for efficient operation
        self.config["sdram_allocation_factor"] = self.config.get("sdram_allocation_factor", 1.2)
        
        # Set optimal learning parameters
        if "learning" in self.config:
            self.config["learning"]["weight_precision"] = 16
            self.config["learning"]["use_sdram_for_weights"] = True
    
    def _optimize_for_truenorth(self):
        """Apply TrueNorth-specific optimizations."""
        # Enforce binary weights
        self.config["binary_weights"] = True
        
        # Optimize core allocation
        self.config["cores_per_chip"] = 4096  # TrueNorth has 4096 cores per chip
        
        # Set ultra-low power mode
        self.config["power_mode"] = "ultra_low_power"
        
        # Configure for deterministic operation
        self.config["deterministic"] = True
        
        # Set optimal learning parameters
        if "learning" in self.config:
            self.config["learning"]["weight_precision"] = 1
            self.config["learning"]["offline_only"] = True

    @hardware_operation("shutdown")
    def shutdown(self) -> bool:
        """
        Safely shutdown the hardware.
        
        Returns:
            bool: True if shutdown was successful
        """
        result = self.interface.shutdown()
        self.initialized = False
        return result
    
    @hardware_operation("create_network")
    def create_network(self, network_config: Dict[str, Any]) -> str:
        """
        Create a neural network on the hardware.
        
        Args:
            network_config: Neural network configuration
            
        Returns:
            str: Network ID
        """
        # Optimize network for target hardware
        optimized_config = self.interface.optimize_network(network_config)
        
        # Extract neurons and connections from network config
        neurons = optimized_config.get("neurons", [])
        connections = optimized_config.get("connections", [])
        
        # Allocate neurons
        neuron_ids = self.interface.allocate_neurons(
            len(neurons), 
            {"neuron_params": neurons}
        )
        
        # Create connections
        connection_tuples = []
        for conn in connections:
            src_idx = conn.get("source")
            tgt_idx = conn.get("target")
            weight = conn.get("weight", 1.0)
            if src_idx < len(neuron_ids) and tgt_idx < len(neuron_ids):
                connection_tuples.append((neuron_ids[src_idx], neuron_ids[tgt_idx], weight))
        
        synapse_ids = self.interface.create_connections(connection_tuples)
        
        # Return network ID (implementation-specific)
        return f"network_{len(neuron_ids)}_{len(synapse_ids)}"
    
    @hardware_operation("run_simulation")
    def run_simulation(self, 
                      network_id: str, 
                      inputs: Dict[str, List[float]], 
                      duration_ms: float, 
                      output_neurons: Optional[List[str]] = None) -> Dict[str, List[float]]:
        """
        Run a simulation on the hardware.
        
        Args:
            network_id: Network identifier
            inputs: Dictionary mapping input names to spike times
            duration_ms: Duration of simulation in milliseconds
            output_neurons: Optional list of output neuron names to record from
            
        Returns:
            Dict[str, List[float]]: Dictionary mapping output names to spike times
        """
        # Convert named inputs to neuron IDs (implementation-specific)
        input_map = self._convert_named_inputs_to_ids(network_id, inputs)
        
        # Convert named outputs to neuron IDs (implementation-specific)
        output_ids = self._convert_named_outputs_to_ids(network_id, output_neurons)
        
        # Run simulation
        results = self.interface.run_simulation(input_map, duration_ms, output_ids)
        
        # Convert results back to named outputs
        return self._convert_ids_to_named_outputs(network_id, results)
    
    @hardware_operation("get_hardware_info")
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get information about the hardware.
        
        Returns:
            Dict[str, Any]: Hardware information
        """
        return self.interface.get_hardware_info()
    
    @hardware_operation("get_capabilities")
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get hardware capabilities.
        
        Returns:
            Dict[str, Any]: Hardware capabilities
        """
        if self._capabilities is None:
            hardware_info = self.get_hardware_info()
            self._capabilities = hardware_info.get("capabilities", {})
        return self._capabilities
    
    @hardware_operation("supports_feature")
    def supports_feature(self, feature_name: str) -> bool:
        """
        Check if hardware supports a specific feature.
        
        Args:
            feature_name: Feature name to check
            
        Returns:
            bool: True if feature is supported
        """
        capabilities = self.get_capabilities()
        return feature_name in capabilities and capabilities[feature_name]
    
    @hardware_operation("get_optimization_recommendations")
    def get_optimization_recommendations(self) -> List[str]:
        """
        Get hardware-specific optimization recommendations.
        
        Returns:
            List[str]: List of optimization recommendations
        """
        return self.interface.get_optimization_recommendations()
    
    # Helper methods for name-to-id conversion (implementation-specific)
    def _convert_named_inputs_to_ids(self, network_id: str, 
                                   inputs: Dict[str, List[float]]) -> Dict[int, List[float]]:
        """Convert named inputs to neuron IDs."""
        # This would be implemented based on how networks are stored
        # For now, we'll use a simple mapping
        result = {}
        for i, (name, spikes) in enumerate(inputs.items()):
            result[i] = spikes
        return result
    
    def _convert_named_outputs_to_ids(self, network_id: str, 
                                    output_neurons: Optional[List[str]]) -> Optional[List[int]]:
        """Convert named outputs to neuron IDs."""
        if not output_neurons:
            return None
        # Simple mapping for now
        return list(range(len(output_neurons)))
    
    def _convert_ids_to_named_outputs(self, network_id: str, 
                                    results: Dict[int, List[float]]) -> Dict[str, List[float]]:
        """Convert neuron IDs back to named outputs."""
        # Simple mapping for now
        return {f"neuron_{id}": spikes for id, spikes in results.items()}


def create_hardware_interface(hardware_type: Optional[str] = None, 
                             config: Optional[Dict[str, Any]] = None) -> HardwareCompatibilityLayer:
    """
    Create a hardware compatibility layer for the specified hardware type.
    
    Args:
        hardware_type: Type of hardware ('loihi', 'spinnaker', 'truenorth'), or None for auto-detection
        config: Optional configuration parameters
        
    Returns:
        HardwareCompatibilityLayer: Hardware compatibility layer
    """
    return HardwareCompatibilityLayer(hardware_type, config)

# The following methods should be moved inside the HardwareCompatibilityLayer class
# before the create_hardware_interface function
    @hardware_operation("negotiate_features")
    def negotiate_features(self, requested_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Negotiate features based on hardware capabilities.
        
        Args:
            requested_features: Dictionary of requested features and parameters
            
        Returns:
            Dict[str, Any]: Dictionary of negotiated features and parameters
        """
        return self.capability_negotiator.negotiate_features(requested_features)
    
    @hardware_operation("check_network_compatibility")
    def check_network_compatibility(self, network_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if a network configuration is compatible with the hardware.
        
        Args:
            network_config: Neural network configuration
            
        Returns:
            Tuple[bool, List[str]]: (is_compatible, incompatibility_reasons)
        """
        return self.capability_negotiator.check_feature_compatibility(network_config)
    
    @hardware_operation("get_supported_features")
    def get_supported_features(self) -> List[str]:
        """
        Get list of supported features on this hardware.
        
        Returns:
            List[str]: List of supported features
        """
        return self.capability_negotiator.get_supported_features()
