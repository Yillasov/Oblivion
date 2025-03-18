"""
Cross-Platform Resource Mapping Algorithms

Provides algorithms for mapping resources between different neuromorphic hardware platforms.
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import time
from datetime import datetime

from src.core.utils.logging_framework import get_logger

logger = get_logger("resource_mapping")


class ResourceMapper:
    """Base class for cross-platform resource mapping."""
    
    @staticmethod
    def map_neurons(source_hw: str, target_hw: str, neuron_ids: List[int], 
                   neuron_params: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
        """
        Map neurons from source hardware to target hardware.
        
        Args:
            source_hw: Source hardware type
            target_hw: Target hardware type
            neuron_ids: List of neuron IDs on source hardware
            neuron_params: Neuron parameters on source hardware
            
        Returns:
            Tuple[List[int], Dict[str, Any]]: Mapped neuron IDs and parameters
        """
        # Simple direct mapping for compatible hardware
        if source_hw == target_hw:
            return neuron_ids, neuron_params
            
        # Get appropriate mapping strategy
        mapper = PlatformMapperFactory.get_mapper(source_hw, target_hw)
        return mapper.map_neurons(neuron_ids, neuron_params)
    
    @staticmethod
    def map_connections(source_hw: str, target_hw: str, 
                       connections: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """
        Map connections from source hardware to target hardware.
        
        Args:
            source_hw: Source hardware type
            target_hw: Target hardware type
            connections: List of (pre_id, post_id, weight) tuples
            
        Returns:
            List[Tuple[int, int, float]]: Mapped connections
        """
        # Simple direct mapping for compatible hardware
        if source_hw == target_hw:
            return connections
            
        # Get appropriate mapping strategy
        mapper = PlatformMapperFactory.get_mapper(source_hw, target_hw)
        return mapper.map_connections(connections)


class PlatformMapper:
    """Base class for platform-specific mapping strategies."""
    
    def map_neurons(self, neuron_ids: List[int], 
                   neuron_params: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
        """Map neurons between platforms."""
        raise NotImplementedError("Subclasses must implement map_neurons")
    
    def map_connections(self, connections: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Map connections between platforms."""
        raise NotImplementedError("Subclasses must implement map_connections")


class LoihiToSpiNNakerMapper(PlatformMapper):
    """Maps resources from Loihi to SpiNNaker."""
    
    def map_neurons(self, neuron_ids: List[int], 
                   neuron_params: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
        """Map neurons from Loihi to SpiNNaker."""
        # Simple ID mapping (1:1)
        mapped_ids = neuron_ids
        
        # Parameter mapping
        mapped_params = neuron_params.copy()
        
        # Convert Loihi compartment type to SpiNNaker neuron type
        if "compartment_type" in mapped_params:
            comp_type = mapped_params.pop("compartment_type")
            if comp_type == 1:
                mapped_params["neuron_type"] = "LIF"
            elif comp_type == 2:
                mapped_params["neuron_type"] = "AdEx"
            else:
                mapped_params["neuron_type"] = "IF"
        
        logger.info(f"Mapped {len(neuron_ids)} neurons from Loihi to SpiNNaker")
        return mapped_ids, mapped_params
    
    def map_connections(self, connections: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Map connections from Loihi to SpiNNaker."""
        # SpiNNaker uses 16-bit weights, Loihi uses 8-bit
        # Scale weights appropriately
        mapped_connections = []
        for pre_id, post_id, weight in connections:
            # Simple weight scaling (Loihi [-128, 127] to SpiNNaker [-32768, 32767])
            scaled_weight = weight * 256.0
            mapped_connections.append((pre_id, post_id, scaled_weight))
        
        logger.info(f"Mapped {len(connections)} connections from Loihi to SpiNNaker")
        return mapped_connections


class SpiNNakerToLoihiMapper(PlatformMapper):
    """Maps resources from SpiNNaker to Loihi."""
    
    def map_neurons(self, neuron_ids: List[int], 
                   neuron_params: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
        """Map neurons from SpiNNaker to Loihi."""
        # Simple ID mapping (1:1)
        mapped_ids = neuron_ids
        
        # Parameter mapping
        mapped_params = neuron_params.copy()
        
        # Convert SpiNNaker neuron type to Loihi compartment type
        if "neuron_type" in mapped_params:
            neuron_type = mapped_params.get("neuron_type")
            if neuron_type == "LIF":
                mapped_params["compartment_type"] = 1
            elif neuron_type == "AdEx":
                mapped_params["compartment_type"] = 2
            else:
                mapped_params["compartment_type"] = 0
        
        logger.info(f"Mapped {len(neuron_ids)} neurons from SpiNNaker to Loihi")
        return mapped_ids, mapped_params
    
    def map_connections(self, connections: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Map connections from SpiNNaker to Loihi."""
        # SpiNNaker uses 16-bit weights, Loihi uses 8-bit
        # Scale weights appropriately
        mapped_connections = []
        for pre_id, post_id, weight in connections:
            # Simple weight scaling (SpiNNaker [-32768, 32767] to Loihi [-128, 127])
            scaled_weight = weight / 256.0
            # Clamp to Loihi range
            scaled_weight = max(-128, min(127, scaled_weight))
            mapped_connections.append((pre_id, post_id, scaled_weight))
        
        logger.info(f"Mapped {len(connections)} connections from SpiNNaker to Loihi")
        return mapped_connections


class LoihiToTrueNorthMapper(PlatformMapper):
    """Maps resources from Loihi to TrueNorth."""
    
    def map_neurons(self, neuron_ids: List[int], 
                   neuron_params: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
        """Map neurons from Loihi to TrueNorth."""
        # Simple ID mapping (1:1)
        mapped_ids = neuron_ids
        
        # TrueNorth only supports LIF neurons with limited parameters
        mapped_params = {
            "neuron_type": "LIF",
            "threshold": neuron_params.get("threshold", 1),
            "leak": neuron_params.get("leak", 0),
            "binary_weights": True
        }
        
        logger.info(f"Mapped {len(neuron_ids)} neurons from Loihi to TrueNorth (with parameter simplification)")
        return mapped_ids, mapped_params
    
    def map_connections(self, connections: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Map connections from Loihi to TrueNorth."""
        # TrueNorth only supports binary weights (0 or 1)
        mapped_connections = []
        for pre_id, post_id, weight in connections:
            # Convert to binary weight
            binary_weight = 1 if weight > 0 else 0
            mapped_connections.append((pre_id, post_id, binary_weight))
        
        logger.info(f"Mapped {len(connections)} connections from Loihi to TrueNorth (with binary weights)")
        return mapped_connections


class PlatformMapperFactory:
    """Factory for creating platform-specific mappers."""
    
    _mappers = {
        ("loihi", "spinnaker"): LoihiToSpiNNakerMapper(),
        ("spinnaker", "loihi"): SpiNNakerToLoihiMapper(),
        ("loihi", "truenorth"): LoihiToTrueNorthMapper(),
    }
    
    @classmethod
    def get_mapper(cls, source_hw: str, target_hw: str) -> PlatformMapper:
        """
        Get appropriate mapper for the given hardware platforms.
        
        Args:
            source_hw: Source hardware type
            target_hw: Target hardware type
            
        Returns:
            PlatformMapper: Platform-specific mapper
        """
        source_hw = source_hw.lower()
        target_hw = target_hw.lower()
        
        key = (source_hw, target_hw)
        if key in cls._mappers:
            return cls._mappers[key]
        
        # If no specific mapper exists, create a default mapper
        logger.warning(f"No specific mapper for {source_hw} to {target_hw}, using default mapping")
        return DefaultMapper()


class DefaultMapper(PlatformMapper):
    """Default mapper for unsupported platform combinations."""
    
    def map_neurons(self, neuron_ids: List[int], 
                   neuron_params: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
        """Default neuron mapping (pass-through)."""
        return neuron_ids, neuron_params
    
    def map_connections(self, connections: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Default connection mapping (pass-through)."""
        return connections


# Simple usage example
def map_network_resources(source_hw: str, target_hw: str, 
                         neuron_ids: List[int], 
                         neuron_params: Dict[str, Any],
                         connections: List[Tuple[int, int, float]]) -> Dict[str, Any]:
    """
    Map network resources from source hardware to target hardware.
    
    Args:
        source_hw: Source hardware type
        target_hw: Target hardware type
        neuron_ids: List of neuron IDs on source hardware
        neuron_params: Neuron parameters on source hardware
        connections: List of (pre_id, post_id, weight) tuples
        
    Returns:
        Dict[str, Any]: Mapped resources
    """
    mapped_ids, mapped_params = ResourceMapper.map_neurons(
        source_hw, target_hw, neuron_ids, neuron_params)
    
    mapped_connections = ResourceMapper.map_connections(
        source_hw, target_hw, connections)
    
    return {
        "neuron_ids": mapped_ids,
        "neuron_params": mapped_params,
        "connections": mapped_connections
    }


# Add this class at the end of the file
class ResourceUsageTracker:
    """Simple resource usage tracking and reporting."""
    
    def __init__(self, max_history: int = 50):
        """
        Initialize resource usage tracker.
        
        Args:
            max_history: Maximum number of history entries to keep
        """
        self.max_history = max_history
        self.usage_history = []
        self.start_time = datetime.now()
    
    def record_usage(self, usage_data: Dict[str, Any]) -> None:
        """
        Record resource usage data.
        
        Args:
            usage_data: Resource usage data dictionary
        """
        # Add timestamp if not present
        if "timestamp" not in usage_data:
            usage_data["timestamp"] = datetime.now().isoformat()
        
        # Add to history
        self.usage_history.append(usage_data)
        
        # Trim history if needed
        if len(self.usage_history) > self.max_history:
            self.usage_history = self.usage_history[-self.max_history:]
    
    def get_current_usage(self) -> Dict[str, Any]:
        """
        Get most recent usage data.
        
        Returns:
            Dict[str, Any]: Most recent usage data or empty dict
        """
        return self.usage_history[-1] if self.usage_history else {}
    
    def get_usage_report(self) -> Dict[str, Any]:
        """
        Generate usage report with statistics.
        
        Returns:
            Dict[str, Any]: Usage report
        """
        if not self.usage_history:
            return {"error": "No usage data available"}
        
        # Get resource types from first entry
        resource_types = [k for k in self.usage_history[0].keys() 
                         if k != "timestamp" and not isinstance(self.usage_history[0][k], dict)]
        
        # Calculate statistics
        stats = {}
        for res_type in resource_types:
            values = [entry.get(res_type, 0) for entry in self.usage_history if res_type in entry]
            if values:
                stats[res_type] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "current": values[-1]
                }
        
        # Create report
        report = {
            "current": self.get_current_usage(),
            "stats": stats,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "samples": len(self.usage_history),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def reset(self) -> None:
        """Reset usage history."""
        self.usage_history = []
        self.start_time = datetime.now()


# Create a global tracker instance
global_usage_tracker = ResourceUsageTracker()


def track_resource_usage(resource_manager) -> Dict[str, Any]:
    """
    Track resource usage from a resource manager.
    
    Args:
        resource_manager: Resource manager instance
        
    Returns:
        Dict[str, Any]: Current usage data
    """
    usage = resource_manager.get_resource_usage()
    global_usage_tracker.record_usage(usage)
    return usage


def get_resource_report() -> Dict[str, Any]:
    """
    Get resource usage report.
    
    Returns:
        Dict[str, Any]: Resource usage report
    """
    return global_usage_tracker.get_usage_report()