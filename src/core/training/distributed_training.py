"""
Distributed Training Coordinator

Provides functionality to coordinate training across multiple neuromorphic hardware devices.
"""

from typing import Dict, List, Any, Optional, Tuple
import threading
import time
import uuid
import numpy as np
import copy
from dataclasses import asdict  # Add this import for asdict

from src.core.utils.logging_framework import get_logger
from src.core.training.hardware_adapters import create_trainer
from src.core.training.trainer_base import TrainingConfig, NeuromorphicTrainer
from src.core.hardware.hardware_registry import get_hardware

logger = get_logger("distributed_training")


class DistributedTrainingCoordinator:
    """Coordinates training across multiple neuromorphic hardware devices."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize distributed training coordinator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.hardware_devices = []
        self.trainers = {}
        self.results = {}
        self.is_training = False
        self.coordinator_id = str(uuid.uuid4())[:8]
        self.lock = threading.Lock()
        self.sync_interval = self.config.get("sync_interval", 5)  # Sync every 5 batches by default
        self.global_parameters = None
        self.sync_event = threading.Event()
        
        # Add hardware performance tracking
        self.hardware_performance = {}
        self.load_balance_enabled = self.config.get("load_balance", True)
        
        logger.info(f"Initialized distributed training coordinator {self.coordinator_id}")
    
    def add_hardware(self, hardware_type: str, hardware_id: Optional[str] = None) -> str:
        """
        Add a hardware device to the distributed training pool.
        
        Args:
            hardware_type: Type of hardware ('loihi', 'spinnaker', 'truenorth')
            hardware_id: Optional identifier for the hardware
            
        Returns:
            str: Hardware ID
        """
        if hardware_id is None:
            hardware_id = f"{hardware_type}_{len(self.hardware_devices)}"
            
        # Create hardware instance using the registry instead of factory
        hardware = get_hardware(hardware_type)
        if not hardware:
            logger.error(f"Failed to create hardware of type {hardware_type}")
            return ""
            
        # Initialize hardware
        if not hardware.initialize():
            logger.error(f"Failed to initialize hardware {hardware_id}")
            return ""
            
        # Add to pool
        self.hardware_devices.append((hardware_id, hardware, hardware_type))
        logger.info(f"Added {hardware_type} hardware with ID {hardware_id} to training pool")
        
        return hardware_id
    
    def _split_dataset_with_load_balancing(self, dataset: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """Split dataset across devices with load balancing."""
        if not self.load_balance_enabled or not self.hardware_performance:
            # Fall back to equal split if no performance data or load balancing disabled
            return self._split_dataset(dataset, len(self.hardware_devices))
        
        # Calculate relative performance weights
        total_perf = sum(self.hardware_performance.values())
        weights = [self.hardware_performance.get(hw_id, 1.0) / total_perf 
                  for hw_id, _, _ in self.hardware_devices]
        
        # Get training data
        inputs = dataset.get("inputs", None)
        targets = dataset.get("targets", None)
        
        if inputs is None or targets is None:
            logger.error("Dataset missing required 'inputs' or 'targets' keys")
            return self._split_dataset(dataset, len(self.hardware_devices))
        
        # Calculate split sizes based on weights
        total_samples = len(inputs)
        split_sizes = [int(total_samples * w) for w in weights]
        
        # Adjust for rounding errors
        remainder = total_samples - sum(split_sizes)
        split_sizes[0] += remainder
        
        # Create split datasets
        split_datasets = []
        start_idx = 0
        
        for size in split_sizes:
            end_idx = start_idx + size
            
            # Create split dataset
            split_dataset = {
                "inputs": inputs[start_idx:end_idx],
                "targets": targets[start_idx:end_idx]
            }
            
            # Copy other keys
            for key, value in dataset.items():
                if key not in ["inputs", "targets"] and isinstance(value, np.ndarray):
                    if len(value) == len(inputs):
                        split_dataset[key] = value[start_idx:end_idx]
                    else:
                        split_dataset[key] = value
            
            split_datasets.append(split_dataset)
            start_idx = end_idx
        
        return split_datasets
    
    def prepare_training(self, model: Any, dataset: Dict[str, np.ndarray]) -> bool:
        """Prepare for distributed training with load balancing."""
        if not self.hardware_devices:
            logger.error("No hardware devices in the pool")
            return False
        
        # Measure hardware performance if not already done
        if not self.hardware_performance and self.load_balance_enabled:
            self._measure_hardware_performance()
        
        # Split dataset with load balancing
        split_datasets = self._split_dataset_with_load_balancing(dataset)
        
        # Create and initialize trainers for each hardware
        for i, (hw_id, hardware, hw_type) in enumerate(self.hardware_devices):
            # Create training config
            training_config = TrainingConfig(
                hardware_type=hw_type,
                batch_size=self.config.get("batch_size", 32),
                learning_rate=self.config.get("learning_rate", 0.01),
                epochs=self.config.get("epochs", 10)
            )
            
            # Create trainer
            trainer = create_trainer(hw_type, training_config)
            
            # Initialize trainer with model and hardware
            if not trainer.initialize(model, hardware):
                logger.error(f"Failed to initialize trainer for hardware {hw_id}")
                continue
            
            # Store trainer and its dataset
            self.trainers[hw_id] = (trainer, split_datasets[i])
        
        logger.info(f"Prepared distributed training across {len(self.trainers)} devices with load balancing")
        return len(self.trainers) > 0
    
    def _measure_hardware_performance(self) -> None:
        """Measure relative performance of hardware devices."""
        logger.info("Measuring hardware performance for load balancing")
        
        # Create a small test model and dataset
        test_size = 1000
        test_dim = 10
        test_model = {"type": "test"}  # Simplified test model
        test_dataset = {
            "inputs": np.random.random((test_size, test_dim)),
            "targets": np.random.random((test_size, 1))
        }
        
        # Measure performance for each hardware
        for hw_id, hardware, hw_type in self.hardware_devices:
            try:
                # Create test trainer
                training_config = TrainingConfig(
                    hardware_type=hw_type,
                    batch_size=32,
                    learning_rate=0.01,
                    epochs=1
                )
                trainer = create_trainer(hw_type, training_config)
                
                # Initialize trainer
                if not trainer.initialize(test_model, hardware):
                    logger.warning(f"Could not initialize test trainer for {hw_id}")
                    self.hardware_performance[hw_id] = 1.0
                    continue
                
                # Measure time for a small training run
                start_time = time.time()
                trainer.train_batch_range(test_dataset, 0, 5)
                elapsed = time.time() - start_time
                
                # Store inverse of time (higher is better)
                perf_score = 1.0 / max(elapsed, 0.001)
                self.hardware_performance[hw_id] = perf_score
                
                logger.info(f"Hardware {hw_id} performance score: {perf_score:.2f}")
                
            except Exception as e:
                logger.warning(f"Error measuring performance for {hw_id}: {str(e)}")
                self.hardware_performance[hw_id] = 1.0
    
    def start_training(self) -> bool:
        """
        Start distributed training on all hardware devices.
        
        Returns:
            bool: Success status
        """
        if self.is_training:
            logger.warning("Training already in progress")
            return False
            
        if not self.trainers:
            logger.error("No trainers prepared")
            return False
        
        # Initialize global parameters from first trainer
        first_hw_id = next(iter(self.trainers))
        first_trainer = self.trainers[first_hw_id][0]
        self.global_parameters = first_trainer.get_parameters()
        
        # Start training threads
        self.is_training = True
        self.results = {}
        threads = []
        
        # Start parameter synchronization thread
        sync_thread = threading.Thread(target=self._parameter_sync_worker)
        sync_thread.daemon = True
        sync_thread.start()
        
        for hw_id, (trainer, dataset) in self.trainers.items():
            thread = threading.Thread(
                target=self._train_on_device,
                args=(hw_id, trainer, dataset)
            )
            thread.start()
            threads.append(thread)
            
        # Wait for all training to complete
        for thread in threads:
            thread.join()
            
        self.is_training = False
        self.sync_event.set()  # Signal sync thread to exit
        
        # Aggregate results
        aggregated_results = self._aggregate_results()
        logger.info(f"Completed distributed training with aggregated results: {aggregated_results}")
        
        return True
    
    def _train_on_device(self, hw_id: str, trainer: NeuromorphicTrainer, dataset: Dict[str, np.ndarray]) -> None:
        """Train on a single device and store results."""
        try:
            logger.info(f"Starting training on hardware {hw_id}")
            
            # Get batch size for sync interval calculation
            batch_size = self.config.get("batch_size", 32)
            total_samples = len(dataset.get("inputs", []))
            total_batches = total_samples // batch_size
            
            # Train with parameter synchronization
            for epoch in range(self.config.get("epochs", 10)):
                for batch_idx in range(0, total_batches, self.sync_interval):
                    # Synchronize parameters before processing batch group
                    with self.lock:
                        if self.global_parameters:
                            trainer.set_parameters(copy.deepcopy(self.global_parameters))
                    
                    # Process batch group
                    end_idx = min(batch_idx + self.sync_interval, total_batches)
                    batch_metrics = trainer.train_batch_range(dataset, batch_idx, end_idx)
                    
                    # Contribute to global parameters
                    self._contribute_parameters(trainer)
            
            # Final metrics - use the metrics attribute instead of a get_metrics method
            metrics = asdict(trainer.metrics) if hasattr(trainer, 'metrics') else {}
            
            with self.lock:
                self.results[hw_id] = metrics
                
            logger.info(f"Completed training on hardware {hw_id} with metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error training on hardware {hw_id}: {str(e)}")
            with self.lock:
                self.results[hw_id] = {"error": str(e)}
    
    def _contribute_parameters(self, trainer: NeuromorphicTrainer) -> None:
        """Contribute device parameters to global parameters."""
        with self.lock:
            device_params = trainer.get_parameters()
            if device_params and self.global_parameters:
                # Simple averaging of parameters
                for key in self.global_parameters:
                    if key in device_params:
                        # Average with existing parameters
                        self.global_parameters[key] = 0.5 * self.global_parameters[key] + 0.5 * device_params[key]
    
    def _parameter_sync_worker(self) -> None:
        """Worker thread for parameter synchronization."""
        logger.info("Started parameter synchronization worker")
        
        while self.is_training and not self.sync_event.is_set():
            # Periodically check if parameters need to be synchronized
            time.sleep(0.1)  # Check frequently but don't consume too much CPU
            
            # The actual synchronization happens in _train_on_device
            # This thread just keeps running to handle any future sync needs
        
        logger.info("Parameter synchronization worker stopped")
    
    def _split_dataset(self, dataset: Dict[str, np.ndarray], num_splits: int) -> List[Dict[str, np.ndarray]]:
        """Split dataset across devices."""
        split_datasets = []
        
        # Get training data
        inputs = dataset.get("inputs", None)
        targets = dataset.get("targets", None)
        
        if inputs is None or targets is None:
            logger.error("Dataset missing required 'inputs' or 'targets' keys")
            return [dataset] * num_splits  # Just duplicate as fallback
            
        # Calculate split sizes
        split_size = len(inputs) // num_splits
        
        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < num_splits - 1 else len(inputs)
            
            # Create split dataset
            split_dataset = {
                "inputs": inputs[start_idx:end_idx],
                "targets": targets[start_idx:end_idx]
            }
            
            # Copy other keys
            for key, value in dataset.items():
                if key not in ["inputs", "targets"] and isinstance(value, np.ndarray):
                    if len(value) == len(inputs):
                        split_dataset[key] = value[start_idx:end_idx]
                    else:
                        split_dataset[key] = value
                        
            split_datasets.append(split_dataset)
            
        return split_datasets
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results from all devices."""
        if not self.results:
            return {}
            
        # Initialize aggregated results
        aggregated = {}
        
        # Collect metrics from all devices
        for hw_id, metrics in self.results.items():
            if "error" in metrics:
                continue
                
            for key, value in metrics.items():
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].append(value)
        
        # Average metrics
        for key in aggregated:
            if aggregated[key] and isinstance(aggregated[key][0], (int, float)):
                aggregated[key] = sum(aggregated[key]) / len(aggregated[key])
                
        return aggregated
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Get training results from all devices."""
        return self.results
    
    def get_aggregated_results(self) -> Dict[str, Any]:
        """Get aggregated training results."""
        return self._aggregate_results()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        for hw_id, (trainer, _) in self.trainers.items():
            try:
                trainer.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up trainer for hardware {hw_id}: {str(e)}")
                
        self.trainers = {}
        
        # Release hardware
        for hw_id, hardware, _ in self.hardware_devices:
            try:
                hardware.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down hardware {hw_id}: {str(e)}")
                
        self.hardware_devices = []
        logger.info("Cleaned up distributed training resources")


# Simple usage example
def run_distributed_training(model: Any, dataset: Dict[str, np.ndarray], 
                            hardware_types: List[str], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run distributed training across multiple hardware devices.
    
    Args:
        model: Neural network model to train
        dataset: Training dataset
        hardware_types: List of hardware types to use
        config: Training configuration
        
    Returns:
        Dict[str, Any]: Aggregated training results
    """
    coordinator = DistributedTrainingCoordinator(config)
    
    try:
        # Add hardware devices
        for hw_type in hardware_types:
            coordinator.add_hardware(hw_type)
            
        # Prepare and start training
        if coordinator.prepare_training(model, dataset):
            coordinator.start_training()
            return coordinator.get_aggregated_results()
        else:
            logger.error("Failed to prepare distributed training")
            return {"error": "Failed to prepare training"}
    finally:
        coordinator.cleanup()