#!/usr/bin/env python3
"""
Simulation-Based Training for Hardware Testing

Provides a lightweight framework for training and testing neuromorphic networks
in a simulated hardware environment before deployment to physical hardware.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, Optional, List
import numpy as np
import time

from src.core.utils.logging_framework import get_logger
from src.core.hardware.simulated_hardware import SimulatedHardware
from src.core.training.trainer_base import NeuromorphicTrainer, TrainingConfig
from src.core.hardware.hardware_registry import hardware_registry
from src.core.training.stdp_learning import create_stdp_component

# Add this import at the top with the other imports
from src.core.training.hebbian_learning import create_hebbian_component
from src.core.training.reinforcement_learning import create_reinforcement_component
from src.core.training.bptt_learning import create_bptt_component
from src.core.training.dataset_loaders import DatasetLoader, convert_to_spike_format

logger = get_logger("sim_training")


class SimulationTrainer:
    """
    Simulation-based trainer for neuromorphic networks.
    
    Allows training and testing on simulated hardware before
    deploying to physical neuromorphic hardware.
    """
    
    def __init__(self, hardware_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize simulation trainer.
        
        Args:
            hardware_type: Type of hardware to simulate ("loihi", "spinnaker", "truenorth")
            config: Configuration parameters
        """
        self.config = config or {}
        self.hardware_type = hardware_type
        self.sim_hardware = SimulatedHardware(hardware_type)
        self.sim_hardware.initialize()
        self.metrics = {
            "iterations": 0,
            "training_time": 0,
            "accuracy": 0.0,
            "loss": 0.0
        }
        
        logger.info(f"Initialized simulation trainer for {hardware_type}")
    
    def train(self, dataset: Dict[str, np.ndarray], epochs: int = 10) -> Dict[str, Any]:
        """
        Train network on simulated hardware.
        
        Args:
            dataset: Training dataset with inputs and targets
            epochs: Number of training epochs
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        start_time = time.time()
        
        # Allocate neurons for the network
        neuron_count = self.config.get("neuron_count", 100)
        resource_request = {
            "neuron_count": neuron_count,
            "learning_enabled": True
        }
        allocation_result = self.sim_hardware.allocate_resources(resource_request)
        if not allocation_result:
            logger.error("Failed to allocate resources on simulated hardware")
            return self.metrics
        
        # Create learning component based on config
        learning_type = self.config.get("learning_type", "stdp")
        if learning_type == "hebbian":
            hebbian_config = self.config.get("hebbian", {})
            learning_component = create_hebbian_component(hebbian_config)
            logger.info("Using Hebbian learning")
        elif learning_type == "reinforcement":
            rl_config = self.config.get("reinforcement", {})
            learning_component = create_reinforcement_component(rl_config)
            logger.info("Using Reinforcement learning")
        elif learning_type == "bptt":
            bptt_config = self.config.get("bptt", {})
            learning_component = create_bptt_component(bptt_config)
            logger.info("Using Backpropagation Through Time learning")
        else:
            # Default to STDP
            stdp_config = self.config.get("stdp", {})
            learning_component = create_stdp_component(stdp_config)
            logger.info("Using STDP learning")
        
        # Create initial random connections
        connections = {}
        for i in range(neuron_count):
            for j in range(neuron_count):
                if i != j and np.random.random() < 0.3:  # 30% connectivity
                    connections[(i, j)] = np.random.random() * 0.5
        
        # Initialize learning component with connections
        learning_component.initialize(connections)
        
        # Simple training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Process each sample in dataset
            for i in range(len(dataset["inputs"])):
                input_data = dataset["inputs"][i]
                target = dataset["targets"][i]
                
                # Convert input to spike format
                input_spikes = {0: self._convert_to_spikes(input_data)}
                
                # Run simulation
                duration_ms = 100.0
                result = self.sim_hardware.run_computation({"spikes": input_spikes}, duration_ms)
                
                # Process spikes with learning component
                learning_component.process_spikes(result["spikes"])
                
                # Calculate simple loss (placeholder)
                output = self._process_output_spikes(result["spikes"])
                loss = np.mean((output - target) ** 2)
                epoch_loss += loss
                
                # For BPTT, set target
                if learning_type == "bptt":
                    learning_component.set_target(target)
                
                # For reinforcement learning, set reward based on negative loss
                if learning_type == "reinforcement":
                    # Convert loss to reward (-1 to 1 range)
                    reward = 1.0 - min(1.0, loss * 10.0)  # Scale loss and invert
                    learning_component.set_reward(reward)
                
                self.metrics["iterations"] += 1
            
            # Apply updated weights to hardware
            learning_component.apply_to_hardware(self.sim_hardware)
            
            avg_epoch_loss = epoch_loss / len(dataset["inputs"])
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            self.metrics["loss"] = avg_epoch_loss
        
        # Calculate training time
        self.metrics["training_time"] = time.time() - start_time
        
        # Evaluate on test data if provided
        if "test_inputs" in dataset and "test_targets" in dataset:
            self.metrics["accuracy"] = self._evaluate(dataset["test_inputs"], dataset["test_targets"])
        
        return self.metrics
    
    def _convert_to_spikes(self, data: np.ndarray) -> List[float]:
        """Convert input data to spike timing."""
        # Simple conversion: higher values = earlier spikes
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-9)
        spike_times = 100 * (1 - normalized)  # Invert so higher values spike earlier
        return spike_times.tolist()
    
    def _process_output_spikes(self, spikes: Dict[int, List[float]]) -> np.ndarray:
        """Process output spikes into a prediction."""
        # Simple processing: count spikes per neuron
        # Get hardware info to determine neuron count
        hw_info = self.sim_hardware.get_hardware_info()
        neuron_count = hw_info.get("allocated_neurons", 100)
        
        output = np.zeros(neuron_count)
        
        for neuron_id, spike_times in spikes.items():
            # Just use the neuron_id directly as an index
            if 0 <= neuron_id < neuron_count:
                output[neuron_id] = len(spike_times)
        
        # Normalize
        if np.sum(output) > 0:
            output = output / np.sum(output)
        
        return output
    
    def _update_weights(self, loss: float) -> None:
        """Update network weights based on loss (simplified)."""
        # This is a placeholder for actual weight updates
        # In a real implementation, this would use backpropagation or STDP
        pass
    
    def _evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Evaluate model on test data."""
        correct = 0
        
        for i in range(len(inputs)):
            input_data = inputs[i]
            target = targets[i]
            
            # Convert input to spike format - use first neuron ID (0)
            input_spikes = {0: self._convert_to_spikes(input_data)}
            
            # Run simulation
            result = self.sim_hardware.run_computation({"spikes": input_spikes}, 100.0)
            
            # Get prediction
            output = self._process_output_spikes(result["spikes"])
            prediction = np.argmax(output)
            true_class = np.argmax(target)
            
            if prediction == true_class:
                correct += 1
        
        return correct / len(inputs)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.sim_hardware:
            # Use shutdown instead of a method with the same name
            self.sim_hardware.shutdown()


def run_simulation_training(hardware_type: str, dataset: Dict[str, np.ndarray], 
                           config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run simulation-based training for a specific hardware type.
    
    Args:
        hardware_type: Type of hardware to simulate
        dataset: Training dataset
        config: Configuration parameters
        
    Returns:
        Dict[str, Any]: Training metrics
    """
    trainer = SimulationTrainer(hardware_type, config)
    try:
        metrics = trainer.train(dataset)
        return metrics
    finally:
        trainer.cleanup()


def run_simulation_with_dataset(hardware_type: str, dataset_name: str,
                               config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run simulation training with a named dataset.
    
    Args:
        hardware_type: Type of hardware to simulate
        dataset_name: Name of the dataset to use
        config: Configuration parameters
        
    Returns:
        Dict[str, Any]: Training metrics
    """
    # Load dataset
    loader = DatasetLoader()
    dataset = loader.load_dataset(dataset_name)
    
    # Convert to dictionary format expected by trainer
    data_dict = {
        "inputs": dataset.inputs,
        "targets": dataset.targets,
    }
    
    if dataset.test_inputs is not None and dataset.test_targets is not None:
        data_dict["test_inputs"] = dataset.test_inputs
        data_dict["test_targets"] = dataset.test_targets
    
    # Run training
    return run_simulation_training(hardware_type, data_dict, config)
