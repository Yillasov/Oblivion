#!/usr/bin/env python3
"""
Neuromorphic Hardware Optimization API

Provides a simple API for other systems to interact with the
neuromorphic hardware optimizer.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional
import threading
import time
import json
import os
from pathlib import Path

from src.core.hardware.neuromorphic_optimizer import NeuromorphicHardwareOptimizer
from src.core.optimization.adaptive_realtime_optimizer import (
    AdaptiveOptimizationConfig,
    OptimizationTarget
)
from src.core.utils.logging_framework import get_logger

logger = get_logger("optimization_api")

class OptimizationAPI:
    """API for interacting with neuromorphic hardware optimizers."""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = OptimizationAPI()
            return cls._instance
    
    def __init__(self):
        """Initialize optimization API."""
        self.optimizers: Dict[str, NeuromorphicHardwareOptimizer] = {}
        self.hardware_registry: Dict[str, Dict[str, Any]] = {}
        self.optimization_threads: Dict[str, threading.Thread] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        
        # Create config directory if it doesn't exist
        self.config_dir = Path(os.path.expanduser("~")) / "Oblivion" / "configs" / "optimization"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing optimizers
        self._load_optimizers()
    
    def _load_optimizers(self):
        """Load existing optimizers from config files."""
        try:
            # Load optimizer configs
            config_files = list(self.config_dir.glob("*.json"))
            
            for config_file in config_files:
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    hardware_type = config_data.get("hardware_type")
                    if hardware_type:
                        # Create optimizer config
                        optimizer_config = AdaptiveOptimizationConfig(
                            learning_rate=config_data.get("learning_rate", 0.05),
                            exploration_rate=config_data.get("exploration_rate", 0.1),
                            memory_size=config_data.get("memory_size", 20),
                            adaptation_threshold=config_data.get("adaptation_threshold", 0.02),
                            update_interval=config_data.get("update_interval", 5.0),
                            target=OptimizationTarget(config_data.get("target", "balanced"))
                        )
                        
                        # Set metrics weights
                        if "metrics_weights" in config_data:
                            optimizer_config.metrics_weights = config_data["metrics_weights"]
                        
                        # Create optimizer
                        self.optimizers[hardware_type] = NeuromorphicHardwareOptimizer(
                            hardware_type, optimizer_config
                        )
                        
                        logger.info(f"Loaded optimizer for {hardware_type}")
                except Exception as e:
                    logger.error(f"Failed to load optimizer config from {config_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load optimizers: {str(e)}")
    
    def _save_optimizer_config(self, hardware_type: str):
        """Save optimizer configuration to file."""
        try:
            optimizer = self.optimizers.get(hardware_type)
            if optimizer:
                config_data = {
                    "hardware_type": hardware_type,
                    "learning_rate": optimizer.config.learning_rate,
                    "exploration_rate": optimizer.config.exploration_rate,
                    "memory_size": optimizer.config.memory_size,
                    "adaptation_threshold": optimizer.config.adaptation_threshold,
                    "update_interval": optimizer.config.update_interval,
                    "target": optimizer.config.target.value,
                    "metrics_weights": optimizer.config.metrics_weights
                }
                
                config_file = self.config_dir / f"{hardware_type}_optimizer.json"
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                logger.info(f"Saved optimizer config for {hardware_type}")
        except Exception as e:
            logger.error(f"Failed to save optimizer config for {hardware_type}: {str(e)}")
    
    def get_optimizer(self, hardware_type: str) -> Optional[NeuromorphicHardwareOptimizer]:
        """
        Get optimizer for hardware type.
        
        Args:
            hardware_type: Hardware type
            
        Returns:
            Optional[NeuromorphicHardwareOptimizer]: Optimizer or None if not found
        """
        # Create optimizer if it doesn't exist
        if hardware_type not in self.optimizers:
            self.optimizers[hardware_type] = NeuromorphicHardwareOptimizer(hardware_type)
            self._save_optimizer_config(hardware_type)
        
        return self.optimizers.get(hardware_type)
    
    def register_hardware(self, 
                         hardware_type: str, 
                         hardware_id: str, 
                         hardware_instance: Any,
                         hardware_monitor: Any) -> bool:
        """
        Register hardware instance for optimization.
        
        Args:
            hardware_type: Hardware type
            hardware_id: Hardware instance identifier
            hardware_instance: Hardware instance
            hardware_monitor: Hardware monitor
            
        Returns:
            bool: Success status
        """
        optimizer = self.get_optimizer(hardware_type)
        if not optimizer:
            return False
        
        # Register hardware with optimizer
        success = optimizer.register_hardware(hardware_id, hardware_instance, hardware_monitor)
        
        if success:
            # Add to registry
            if hardware_type not in self.hardware_registry:
                self.hardware_registry[hardware_type] = {}
            
            self.hardware_registry[hardware_type][hardware_id] = {
                "instance": hardware_instance,
                "monitor": hardware_monitor,
                "registered_at": time.time()
            }
            
            logger.info(f"Registered hardware {hardware_id} of type {hardware_type}")
        
        return success
    
    def start_optimization(self, hardware_type: str, hardware_id: str, interval: float = 5.0) -> bool:
        """
        Start optimization thread for hardware instance.
        
        Args:
            hardware_type: Hardware type
            hardware_id: Hardware instance identifier
            interval: Update interval in seconds
            
        Returns:
            bool: Success status
        """
        optimizer = self.get_optimizer(hardware_type)
        if not optimizer:
            return False
        
        # Check if hardware is registered
        if (hardware_type not in self.hardware_registry or 
            hardware_id not in self.hardware_registry[hardware_type]):
            return False
        
        # Check if optimization is already running
        thread_key = f"{hardware_type}_{hardware_id}"
        if thread_key in self.optimization_threads and self.optimization_threads[thread_key].is_alive():
            return True
        
        # Create stop event
        stop_event = threading.Event()
        self.stop_events[thread_key] = stop_event
        
        # Create and start optimization thread
        thread = threading.Thread(
            target=self._optimization_thread,
            args=(hardware_type, hardware_id, interval, stop_event)
        )
        thread.daemon = True
        thread.start()
        
        self.optimization_threads[thread_key] = thread
        
        logger.info(f"Started optimization for {hardware_id} of type {hardware_type}")
        
        return True
    
    def stop_optimization(self, hardware_type: str, hardware_id: str) -> bool:
        """
        Stop optimization thread for hardware instance.
        
        Args:
            hardware_type: Hardware type
            hardware_id: Hardware instance identifier
            
        Returns:
            bool: Success status
        """
        thread_key = f"{hardware_type}_{hardware_id}"
        
        if thread_key in self.stop_events:
            # Set stop event
            self.stop_events[thread_key].set()
            
            # Wait for thread to stop
            if thread_key in self.optimization_threads:
                self.optimization_threads[thread_key].join(timeout=2.0)
                
                # Remove thread
                del self.optimization_threads[thread_key]
            
            # Remove stop event
            del self.stop_events[thread_key]
            
            logger.info(f"Stopped optimization for {hardware_id} of type {hardware_type}")
            
            return True
        
        return False
    
    def _optimization_thread(self, 
                            hardware_type: str, 
                            hardware_id: str, 
                            interval: float,
                            stop_event: threading.Event):
        """Optimization thread function."""
        optimizer = self.get_optimizer(hardware_type)
        if not optimizer:
            return
        
        while not stop_event.is_set():
            try:
                # Update optimization
                optimizer.update(hardware_id)
                
                # Sleep until next update
                stop_event.wait(interval)
            except Exception as e:
                logger.error(f"Error in optimization thread for {hardware_id}: {str(e)}")
                # Sleep before retrying
                stop_event.wait(1.0)
    
    def get_optimization_stats(self, hardware_type: str, hardware_id: str) -> Dict[str, Any]:
        """
        Get optimization statistics for hardware instance.
        
        Args:
            hardware_type: Hardware type
            hardware_id: Hardware instance identifier
            
        Returns:
            Dict[str, Any]: Optimization statistics
        """
        optimizer = self.get_optimizer(hardware_type)
        if not optimizer:
            return {"success": False, "error": "Optimizer not found"}
        
        return optimizer.get_optimization_stats(hardware_id)
    
    def reset_optimization(self, hardware_type: str, hardware_id: str, keep_learning: bool = False) -> bool:
        """
        Reset optimization for hardware instance.
        
        Args:
            hardware_type: Hardware type
            hardware_id: Hardware instance identifier
            keep_learning: Whether to keep learned parameters
            
        Returns:
            bool: Success status
        """
        optimizer = self.get_optimizer(hardware_type)
        if not optimizer:
            return False
        
        return optimizer.reset(hardware_id, keep_learning)
    
    def update_optimizer_config(self, 
                               hardware_type: str, 
                               config_updates: Dict[str, Any]) -> bool:
        """
        Update optimizer configuration.
        
        Args:
            hardware_type: Hardware type
            config_updates: Configuration updates
            
        Returns:
            bool: Success status
        """
        optimizer = self.get_optimizer(hardware_type)
        if not optimizer:
            return False
        
        # Update configuration
        for key, value in config_updates.items():
            if key == "learning_rate":
                optimizer.config.learning_rate = float(value)
            elif key == "exploration_rate":
                optimizer.config.exploration_rate = float(value)
            elif key == "memory_size":
                optimizer.config.memory_size = int(value)
            elif key == "adaptation_threshold":
                optimizer.config.adaptation_threshold = float(value)
            elif key == "update_interval":
                optimizer.config.update_interval = float(value)
            elif key == "target":
                try:
                    optimizer.config.target = OptimizationTarget(value)
                except ValueError:
                    pass
            elif key == "metrics_weights":
                if isinstance(value, dict):
                    optimizer.config.metrics_weights.update(value)
        
        # Save updated configuration
        self._save_optimizer_config(hardware_type)
        
        return True
    
    def get_registered_hardware(self) -> Dict[str, List[str]]:
        """
        Get registered hardware instances.
        
        Returns:
            Dict[str, List[str]]: Hardware instances by type
        """
        result = {}
        
        for hardware_type, instances in self.hardware_registry.items():
            result[hardware_type] = list(instances.keys())
        
        return result
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get optimization status for all hardware instances.
        
        Returns:
            Dict[str, Any]: Optimization status
        """
        status = {
            "hardware_types": list(self.optimizers.keys()),
            "registered_hardware": self.get_registered_hardware(),
            "active_optimizations": []
        }
        
        # Get active optimizations
        for thread_key, thread in self.optimization_threads.items():
            if thread.is_alive():
                status["active_optimizations"].append(thread_key)
        
        return status