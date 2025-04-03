#!/usr/bin/env python3
"""
Data Format Converters

Provides utilities for converting data formats between different neuromorphic hardware platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from src.core.utils.logging_framework import get_logger

logger = get_logger("data_converters")


class DataFormatConverter:
    """Base class for data format conversion between hardware platforms."""
    
    @staticmethod
    def convert(data: Any, source_format: str, target_format: str) -> Any:
        """
        Convert data from source format to target format.
        
        Args:
            data: Data to convert
            source_format: Source data format
            target_format: Target data format
            
        Returns:
            Any: Converted data
        """
        converter = FormatConverterFactory.get_converter(source_format, target_format)
        return converter.convert_data(data)


class SpikeDataConverter:
    """Converts spike data between different formats."""
    
    @staticmethod
    def loihi_to_spinnaker(spike_data: Dict[int, List[float]]) -> Dict[str, Any]:
        """Convert Loihi spike format to SpiNNaker format."""
        result = {"spike_times": {}, "neuron_ids": []}
        
        for neuron_id, times in spike_data.items():
            result["neuron_ids"].append(neuron_id)
            result["spike_times"][str(neuron_id)] = times
            
        return result
    
    @staticmethod
    def spinnaker_to_loihi(spike_data: Dict[str, Any]) -> Dict[int, List[float]]:
        """Convert SpiNNaker spike format to Loihi format."""
        result = {}
        
        for neuron_id_str, times in spike_data.get("spike_times", {}).items():
            try:
                neuron_id = int(neuron_id_str)
                result[neuron_id] = times
            except ValueError:
                logger.warning(f"Invalid neuron ID: {neuron_id_str}")
                
        return result
    
    @staticmethod
    def loihi_to_truenorth(spike_data: Dict[int, List[float]]) -> Dict[str, Any]:
        """Convert Loihi spike format to TrueNorth format."""
        events = []
        
        for neuron_id, times in spike_data.items():
            for t in times:
                events.append({"neuron": neuron_id, "time": t})
                
        return {"events": sorted(events, key=lambda x: x["time"])}
    
    @staticmethod
    def truenorth_to_loihi(spike_data: Dict[str, Any]) -> Dict[int, List[float]]:
        """Convert TrueNorth spike format to Loihi format."""
        result = {}
        
        for event in spike_data.get("events", []):
            neuron_id = event.get("neuron")
            time = event.get("time")
            
            if neuron_id is not None and time is not None:
                if neuron_id not in result:
                    result[neuron_id] = []
                result[neuron_id].append(time)
                
        return result


class WeightDataConverter:
    """Converts weight data between different formats."""
    
    @staticmethod
    def loihi_to_spinnaker(weights: Union[List[Tuple[int, int, float]], np.ndarray]) -> Dict[str, Any]:
        """Convert Loihi weight format to SpiNNaker format."""
        if isinstance(weights, list):
            # Convert list of tuples to matrix format
            pre_ids = set()
            post_ids = set()
            
            for pre, post, _ in weights:
                pre_ids.add(pre)
                post_ids.add(post)
                
            pre_ids = sorted(list(pre_ids))
            post_ids = sorted(list(post_ids))
            
            matrix = np.zeros((len(pre_ids), len(post_ids)))
            pre_map = {id: idx for idx, id in enumerate(pre_ids)}
            post_map = {id: idx for idx, id in enumerate(post_ids)}
            
            for pre, post, weight in weights:
                matrix[pre_map[pre], post_map[post]] = weight
                
            return {
                "weight_matrix": matrix.tolist(),
                "pre_ids": pre_ids,
                "post_ids": post_ids
            }
        else:
            # Already in matrix format
            return {
                "weight_matrix": weights.tolist() if isinstance(weights, np.ndarray) else weights,
                "pre_ids": list(range(weights.shape[0])),
                "post_ids": list(range(weights.shape[1]))
            }
    
    @staticmethod
    def spinnaker_to_loihi(weights: Dict[str, Any]) -> List[Tuple[int, int, float]]:
        """Convert SpiNNaker weight format to Loihi format."""
        result = []
        
        matrix = weights.get("weight_matrix", [])
        pre_ids = weights.get("pre_ids", list(range(len(matrix))))
        post_ids = weights.get("post_ids", list(range(len(matrix[0]))) if matrix else [])
        
        for i, pre in enumerate(pre_ids):
            for j, post in enumerate(post_ids):
                weight = matrix[i][j]
                if weight != 0:
                    result.append((pre, post, weight))
                    
        return result


class FormatConverterFactory:
    """Factory for creating data format converters."""
    
    _converters = {
        ("loihi_spike", "spinnaker_spike"): SpikeDataConverter.loihi_to_spinnaker,
        ("spinnaker_spike", "loihi_spike"): SpikeDataConverter.spinnaker_to_loihi,
        ("loihi_spike", "truenorth_spike"): SpikeDataConverter.loihi_to_truenorth,
        ("truenorth_spike", "loihi_spike"): SpikeDataConverter.truenorth_to_loihi,
        ("loihi_weight", "spinnaker_weight"): WeightDataConverter.loihi_to_spinnaker,
        ("spinnaker_weight", "loihi_weight"): WeightDataConverter.spinnaker_to_loihi,
    }
    
    @classmethod
    def get_converter(cls, source_format: str, target_format: str) -> Any:
        """Get appropriate converter for the given formats."""
        key = (source_format.lower(), target_format.lower())
        
        if key in cls._converters:
            return cls._converters[key]
        
        # If direct conversion not available, try to find a path
        for k in cls._converters:
            if k[0] == key[0]:
                intermediate_format = k[1]
                if (intermediate_format, key[1]) in cls._converters:
                    # Found a two-step conversion path
                    first_step = cls._converters[k]
                    second_step = cls._converters[(intermediate_format, key[1])]
                    
                    # Return a function that performs both conversions
                    return lambda data: second_step(first_step(data))
        
        # Default to identity conversion if no converter found
        logger.warning(f"No converter found for {source_format} to {target_format}, using identity")
        return lambda x: x