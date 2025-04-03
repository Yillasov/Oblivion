#!/usr/bin/env python3
"""
Neuromorphic Data Preprocessing

Provides utilities for preprocessing data for neuromorphic hardware,
including spike encoding and temporal transformations.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from enum import Enum

from src.core.utils.logging_framework import get_logger
from src.core.training.dataset_loaders import NeuromorphicDataset

logger = get_logger("data_preprocessing")


class EncodingType(Enum):
    """Types of spike encoding methods."""
    RATE = "rate"
    TEMPORAL = "temporal"
    PHASE = "phase"
    BURST = "burst"
    DIRECT = "direct"  # For already encoded spike data


class NeuromorphicPreprocessor:
    """
    Preprocessor for neuromorphic data.
    
    Handles conversion of conventional data to spike-based formats
    and other preprocessing needed for neuromorphic hardware.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.default_encoding = self.config.get("default_encoding", EncodingType.RATE)
        self.time_steps = self.config.get("time_steps", 100)
        self.threshold = self.config.get("threshold", 0.5)
        
    def preprocess_dataset(self, dataset: NeuromorphicDataset, 
                          encoding_type: Optional[EncodingType] = None) -> NeuromorphicDataset:
        """
        Preprocess an entire dataset for neuromorphic hardware.
        
        Args:
            dataset: Dataset to preprocess
            encoding_type: Type of encoding to use
            
        Returns:
            NeuromorphicDataset: Preprocessed dataset
        """
        encoding = encoding_type or self.default_encoding
        
        # Check if data is already in spike format
        if dataset.metadata and dataset.metadata.get("is_spike_data", False):
            logger.info("Dataset already contains spike data, skipping encoding")
            return dataset
        
        # Process training data
        processed_inputs = self.encode_data(dataset.inputs, encoding)
        
        # Process test data if available
        processed_test_inputs = None
        if dataset.test_inputs is not None:
            processed_test_inputs = self.encode_data(dataset.test_inputs, encoding)
            
        # Process validation data if available
        processed_val_inputs = None
        if dataset.validation_inputs is not None:
            processed_val_inputs = self.encode_data(dataset.validation_inputs, encoding)
        
        # Create new metadata
        metadata = dataset.metadata.copy() if dataset.metadata else {}
        metadata.update({
            "is_spike_data": True,
            "encoding_type": encoding.value,
            "time_steps": self.time_steps,
            "threshold": self.threshold
        })
        
        # Return new dataset with processed data
        return NeuromorphicDataset(
            inputs=processed_inputs,
            targets=dataset.targets,
            test_inputs=processed_test_inputs,
            test_targets=dataset.test_targets,
            validation_inputs=processed_val_inputs,
            validation_targets=dataset.validation_targets,
            metadata=metadata
        )
    
    def encode_data(self, data: np.ndarray, encoding_type: EncodingType) -> np.ndarray:
        """
        Encode data into spike format.
        
        Args:
            data: Input data to encode
            encoding_type: Type of encoding to use
            
        Returns:
            np.ndarray: Spike-encoded data
        """
        if encoding_type == EncodingType.RATE:
            return self._rate_encoding(data)
        elif encoding_type == EncodingType.TEMPORAL:
            return self._temporal_encoding(data)
        elif encoding_type == EncodingType.PHASE:
            return self._phase_encoding(data)
        elif encoding_type == EncodingType.BURST:
            return self._burst_encoding(data)
        elif encoding_type == EncodingType.DIRECT:
            return data
        else:
            logger.warning(f"Unknown encoding type: {encoding_type}, using rate encoding")
            return self._rate_encoding(data)
    
    def _rate_encoding(self, data: np.ndarray) -> np.ndarray:
        """
        Convert data to rate-coded spikes.
        
        Higher values produce more spikes in the time window.
        
        Args:
            data: Input data (samples, features)
            
        Returns:
            np.ndarray: Spike data (samples, features, time_steps)
        """
        samples, features = data.shape
        spikes = np.zeros((samples, features, self.time_steps), dtype=np.float32)
        
        # Normalize data to [0, 1] if not already
        if np.max(data) > 1.0 or np.min(data) < 0.0:
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            normalized_data = data
        
        # Generate spikes based on probability
        for t in range(self.time_steps):
            random_values = np.random.rand(samples, features)
            spikes[:, :, t] = (random_values < normalized_data).astype(np.float32)
        
        return spikes
    
    def _temporal_encoding(self, data: np.ndarray) -> np.ndarray:
        """
        Convert data to temporal-coded spikes.
        
        Higher values fire earlier in the time window.
        
        Args:
            data: Input data (samples, features)
            
        Returns:
            np.ndarray: Spike data (samples, features, time_steps)
        """
        samples, features = data.shape
        spikes = np.zeros((samples, features, self.time_steps), dtype=np.float32)
        
        # Normalize data to [0, 1] if not already
        if np.max(data) > 1.0 or np.min(data) < 0.0:
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            normalized_data = data
        
        # Calculate firing time
        firing_time = ((1.0 - normalized_data) * self.time_steps).astype(int)
        
        # Set spikes at appropriate times
        for i in range(samples):
            for j in range(features):
                if normalized_data[i, j] > self.threshold:
                    t = min(firing_time[i, j], self.time_steps - 1)
                    spikes[i, j, t] = 1.0
        
        return spikes
    
    def _phase_encoding(self, data: np.ndarray) -> np.ndarray:
        """
        Convert data to phase-coded spikes.
        
        Values are encoded in the phase of spike patterns.
        
        Args:
            data: Input data (samples, features)
            
        Returns:
            np.ndarray: Spike data (samples, features, time_steps)
        """
        # Simplified implementation - in a real system this would be more sophisticated
        samples, features = data.shape
        spikes = np.zeros((samples, features, self.time_steps), dtype=np.float32)
        
        # Normalize data to [0, 1] if not already
        if np.max(data) > 1.0 or np.min(data) < 0.0:
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            normalized_data = data
        
        # Create phase patterns
        for i in range(samples):
            for j in range(features):
                if normalized_data[i, j] > self.threshold:
                    phase_shift = int(normalized_data[i, j] * (self.time_steps // 2))
                    for t in range(0, self.time_steps, self.time_steps // 4):
                        spike_time = (t + phase_shift) % self.time_steps
                        spikes[i, j, spike_time] = 1.0
        
        return spikes
    
    def _burst_encoding(self, data: np.ndarray) -> np.ndarray:
        """
        Convert data to burst-coded spikes.
        
        Higher values produce more consecutive spikes (bursts).
        
        Args:
            data: Input data (samples, features)
            
        Returns:
            np.ndarray: Spike data (samples, features, time_steps)
        """
        samples, features = data.shape
        spikes = np.zeros((samples, features, self.time_steps), dtype=np.float32)
        
        # Normalize data to [0, 1] if not already
        if np.max(data) > 1.0 or np.min(data) < 0.0:
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            normalized_data = data
        
        # Calculate burst length
        max_burst = self.time_steps // 4
        burst_length = (normalized_data * max_burst).astype(int)
        
        # Set spikes in bursts
        for i in range(samples):
            for j in range(features):
                if normalized_data[i, j] > self.threshold:
                    start_time = np.random.randint(0, self.time_steps - max_burst)
                    for t in range(burst_length[i, j]):
                        if start_time + t < self.time_steps:
                            spikes[i, j, start_time + t] = 1.0
        
        return spikes