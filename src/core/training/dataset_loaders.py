#!/usr/bin/env python3
"""
Dataset Loaders for Neuromorphic Training

Provides utilities to load and preprocess datasets for neuromorphic training.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import os
import pickle
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

# Add this import at the top of the file
from src.core.training.data_preprocessing import NeuromorphicPreprocessor, EncodingType

logger = get_logger("dataset_loaders")


@dataclass
class NeuromorphicDataset:
    """Container for neuromorphic training datasets."""
    
    inputs: np.ndarray
    targets: np.ndarray
    test_inputs: Optional[np.ndarray] = None
    test_targets: Optional[np.ndarray] = None
    validation_inputs: Optional[np.ndarray] = None
    validation_targets: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, data_dir: str = "/Users/yessine/Oblivion/data"):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory containing datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_dataset(self, name: str, **kwargs) -> NeuromorphicDataset:
        """
        Load a dataset by name.
        
        Args:
            name: Name of the dataset
            **kwargs: Additional dataset-specific parameters
            
        Returns:
            NeuromorphicDataset: Loaded dataset
        """
        if name == "mnist":
            return self._load_mnist(**kwargs)
        elif name == "nmnist":
            return self._load_nmnist(**kwargs)
        elif name == "custom":
            return self._load_custom(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {name}")
    
    def _load_mnist(self, train_size: int = 60000, test_size: int = 10000, 
                   val_split: float = 0.1) -> NeuromorphicDataset:
        """
        Load MNIST dataset and convert to spike-compatible format.
        
        Args:
            train_size: Number of training samples
            test_size: Number of test samples
            val_split: Validation split ratio
            
        Returns:
            NeuromorphicDataset: MNIST dataset
        """
        try:
            # This is a placeholder - in a real implementation, you would:
            # 1. Download MNIST if not available
            # 2. Load the data
            # 3. Normalize and preprocess
            
            # Simulate loading with random data for demonstration
            inputs = np.random.random((train_size, 28*28))
            targets = np.zeros((train_size, 10))
            for i in range(train_size):
                targets[i, np.random.randint(0, 10)] = 1.0
                
            test_inputs = np.random.random((test_size, 28*28))
            test_targets = np.zeros((test_size, 10))
            for i in range(test_size):
                test_targets[i, np.random.randint(0, 10)] = 1.0
            
            # Create validation split
            if val_split > 0:
                val_size = int(train_size * val_split)
                val_inputs = inputs[-val_size:]
                val_targets = targets[-val_size:]
                inputs = inputs[:-val_size]
                targets = targets[:-val_size]
            else:
                val_inputs = None
                val_targets = None
            
            logger.info(f"Loaded MNIST dataset: {len(inputs)} training, {len(test_inputs)} test samples")
            
            return NeuromorphicDataset(
                inputs=inputs,
                targets=targets,
                test_inputs=test_inputs,
                test_targets=test_targets,
                validation_inputs=val_inputs,
                validation_targets=val_targets,
                metadata={"name": "mnist", "input_shape": (28, 28), "classes": 10}
            )
            
        except Exception as e:
            logger.error(f"Error loading MNIST dataset: {str(e)}")
            raise
    
    def _load_nmnist(self, **kwargs) -> NeuromorphicDataset:
        """
        Load Neuromorphic-MNIST (N-MNIST) dataset.
        
        Returns:
            NeuromorphicDataset: N-MNIST dataset
        """
        # N-MNIST is a spiking version of MNIST recorded with DVS cameras
        logger.info("Loading N-MNIST dataset")
        
        # Placeholder implementation - would actually load from files
        # Simulate spike data with random timestamps
        train_size = kwargs.get("train_size", 10000)
        test_size = kwargs.get("test_size", 2000)
        
        # Create spike data format - list of (x, y, t) tuples
        inputs = np.random.random((train_size, 100, 3))  # 100 spikes per sample
        inputs[:, :, 2] = inputs[:, :, 2] * 100  # Time in ms
        
        targets = np.zeros((train_size, 10))
        for i in range(train_size):
            targets[i, np.random.randint(0, 10)] = 1.0
            
        test_inputs = np.random.random((test_size, 100, 3))
        test_inputs[:, :, 2] = test_inputs[:, :, 2] * 100
        
        test_targets = np.zeros((test_size, 10))
        for i in range(test_size):
            test_targets[i, np.random.randint(0, 10)] = 1.0
        
        return NeuromorphicDataset(
            inputs=inputs,
            targets=targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            metadata={"name": "nmnist", "input_shape": (34, 34), "classes": 10, "is_spike_data": True}
        )
    
    def _load_custom(self, file_path: str, **kwargs) -> NeuromorphicDataset:
        """
        Load a custom dataset from file.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            NeuromorphicDataset: Custom dataset
        """
        logger.info(f"Loading custom dataset from {file_path}")
        
        try:
            # Support different file formats
            if file_path.endswith('.npz'):
                data = np.load(file_path)
                
                # Create a proper dictionary for metadata if it exists in the file
                metadata = None
                if 'metadata' in data:
                    # Convert ndarray to dict if needed
                    metadata_array = data['metadata']
                    if isinstance(metadata_array, np.ndarray):
                        # If it's a structured array, convert to dict
                        if metadata_array.dtype.names is not None:
                            metadata = {name: metadata_array[name].item() for name in metadata_array.dtype.names}
                        else:
                            # If it's a regular array, use a default key
                            metadata = {"data": metadata_array.tolist()}
                        
                    else:
                        metadata = metadata_array
                
                # Handle optional fields properly
                test_inputs = None
                if 'test_inputs' in data:
                    test_inputs = data['test_inputs']
                    
                test_targets = None
                if 'test_targets' in data:
                    test_targets = data['test_targets']
                    
                return NeuromorphicDataset(
                    inputs=data['inputs'],
                    targets=data['targets'],
                    test_inputs=test_inputs,
                    test_targets=test_targets,
                    metadata=metadata
                )
            elif file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                return NeuromorphicDataset(**data)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Error loading custom dataset: {str(e)}")
            raise


def convert_to_spike_format(data: np.ndarray, method: str = "rate", 
                           duration: float = 100.0) -> Dict[int, List[float]]:
    """
    Convert standard data to spike format.
    
    Args:
        data: Input data array
        method: Conversion method ("rate", "temporal", "ttfs")
        duration: Simulation duration in ms
        
    Returns:
        Dict[int, List[float]]: Dictionary mapping neuron IDs to spike times
    """
    spikes = {}
    
    if method == "rate":
        # Rate coding: higher values = more spikes
        for i, value in enumerate(data):
            if value > 0:
                # Number of spikes proportional to value
                spike_count = int(value * 10)  # Scale factor
                if spike_count > 0:
                    # Distribute spikes evenly across duration
                    spikes[i] = [j * (duration / spike_count) for j in range(spike_count)]
    
    elif method == "ttfs":
        # Time-to-first-spike: higher values = earlier spikes
        for i, value in enumerate(data):
            if value > 0:
                # Normalize to get spike time
                spike_time = (1.0 - value) * duration
                spikes[i] = [spike_time]
    
    elif method == "temporal":
        # Temporal coding: distribute across time window
        for i, value in enumerate(data):
            if value > 0:
                # Create multiple spikes with temporal pattern
                base_time = (1.0 - value) * (duration * 0.5)
                spikes[i] = [base_time, base_time + 10, base_time + 25]
    
    return spikes


def create_dataset_from_dict(data_dict: Dict[str, np.ndarray]) -> NeuromorphicDataset:
    """
    Create a NeuromorphicDataset from a dictionary.
    
    Args:
        data_dict: Dictionary containing dataset components
        
    Returns:
        NeuromorphicDataset: Created dataset
    """
    required_keys = ["inputs", "targets"]
    for key in required_keys:
        if key not in data_dict:
            raise ValueError(f"Missing required key in dataset dictionary: {key}")
    
    # Handle metadata properly
    metadata = None
    if "metadata" in data_dict:
        metadata_value = data_dict["metadata"]
        # Convert ndarray to dict if needed
        if isinstance(metadata_value, np.ndarray):
            # If it's a structured array, convert to dict
            if metadata_value.dtype.names is not None:
                metadata = {name: metadata_value[name].item() for name in metadata_value.dtype.names}
            else:
                # If it's a regular array, use a default key
                metadata = {"data": metadata_value.tolist()}
        else:
            metadata = metadata_value
    
    return NeuromorphicDataset(
        inputs=data_dict["inputs"],
        targets=data_dict["targets"],
        test_inputs=data_dict.get("test_inputs"),
        test_targets=data_dict.get("test_targets"),
        validation_inputs=data_dict.get("validation_inputs"),
        validation_targets=data_dict.get("validation_targets"),
        metadata=metadata
    )


# Add this method to the DatasetLoader class
def preprocess_for_neuromorphic(self, dataset: NeuromorphicDataset, 
                               encoding_type: str = "rate",
                               time_steps: int = 100,
                               threshold: float = 0.5) -> NeuromorphicDataset:
    """
    Preprocess a dataset for neuromorphic hardware.
    
    Args:
        dataset: Dataset to preprocess
        encoding_type: Type of spike encoding to use
        time_steps: Number of time steps for temporal encoding
        threshold: Threshold for spike generation
        
    Returns:
        NeuromorphicDataset: Preprocessed dataset
    """
    # Create preprocessor with configuration
    config = {
        "default_encoding": EncodingType(encoding_type),
        "time_steps": time_steps,
        "threshold": threshold
    }
    
    preprocessor = NeuromorphicPreprocessor(config)
    
    # Preprocess dataset
    return preprocessor.preprocess_dataset(dataset)