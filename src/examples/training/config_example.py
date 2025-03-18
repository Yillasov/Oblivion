"""
Training Configuration Example

Demonstrates how to use the training configuration system.
"""

import os
import sys
import numpy as np
from typing import Dict, Any, Optional

from src.core.training.config_system import create_config_system
from src.core.training.trainer_base import TrainingConfig, TrainingMode
from src.core.training.hardware_adapters.loihi_adapter import LoihiTrainer
from src.core.training.hardware_adapters.default_adapter import DefaultTrainer
from src.core.utils.logging_framework import get_logger

logger = get_logger("config_example")


def create_sample_dataset() -> Dict[str, np.ndarray]:
    """Create a sample dataset for demonstration."""
    # Create random data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=(100, 1))
    
    return {"inputs": X, "targets": y}


def train_with_config(config_name: str, hardware_type: Optional[str] = None) -> None:
    """
    Train a model using a configuration from the configuration system.
    
    Args:
        config_name: Configuration name
        hardware_type: Optional hardware type
    """
    # Create configuration system
    config_system = create_config_system()
    
    # Load configuration
    config = config_system.load_config(config_name, hardware_type)
    
    if not config:
        logger.error(f"Configuration not found: {config_name}")
        return
    
    logger.info(f"Loaded configuration: {config_name}")
    logger.info(f"Hardware type: {config.hardware_type}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Epochs: {config.epochs}")
    
    # Create trainer based on hardware type
    if config.hardware_type == "loihi":
        trainer = LoihiTrainer(config)
    else:
        trainer = DefaultTrainer(config)
    
    # Create sample dataset
    dataset = create_sample_dataset()
    
    # Initialize model (simplified for example)
    model = {"layers": [10, 5, 1], "activation": "sigmoid"}
    
    # Initialize trainer
    if trainer.initialize(model):
        # Train model
        metrics = trainer.train(dataset, session_name=f"{config_name}_session")
        
        # Print results
        logger.info(f"Training completed with accuracy: {metrics.best_accuracy:.4f}")
        logger.info(f"Training time: {metrics.training_time:.2f} seconds")
    else:
        logger.error("Failed to initialize trainer")


def create_custom_config() -> None:
    """Create a custom configuration and train with it."""
    # Create configuration system
    config_system = create_config_system()
    
    # Create custom configuration
    custom_config = config_system.create_config(
        "simulated", 
        "custom_snn_config",
        {
            "learning_rate": 0.005,
            "batch_size": 32,
            "epochs": 20,
            "mode": TrainingMode.HYBRID,
            "optimizer": "adam",
            "optimizer_params": {"beta1": 0.9, "beta2": 0.999},
            "custom_params": {
                "neuron_model": "LIF",
                "threshold": 1.0,
                "refractory_period": 2,
                "time_steps": 100
            }
        }
    )
    
    logger.info(f"Created custom configuration: custom_snn_config")
    
    # Train with custom configuration
    train_with_config("custom_snn_config")


def main():
    """Main entry point."""
    # Create and train with custom configuration
    create_custom_config()
    
    # Train with existing configurations
    train_with_config("default_snn", "simulated")
    
    # Update and train with modified configuration
    config_system = create_config_system()
    config_system.update_config(
        "default_snn",
        {
            "learning_rate": 0.001,
            "epochs": 30,
            "early_stopping": True
        }
    )
    
    train_with_config("default_snn", "simulated")


if __name__ == "__main__":
    main()