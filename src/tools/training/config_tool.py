#!/usr/bin/env python3
"""
Training Configuration Tool

Command-line utility for managing training configurations.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import sys
import argparse
import json
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.training.config_system import create_config_system, TrainingConfigSystem
from src.core.training.trainer_base import TrainingMode
from src.core.utils.logging_framework import get_logger

logger = get_logger("config_tool")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Configuration Tool")
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new configuration")
    create_parser.add_argument("name", help="Configuration name")
    create_parser.add_argument("--hardware", "-hw", default="simulated", 
                              help="Hardware type (loihi, truenorth, spinnaker, simulated)")
    create_parser.add_argument("--learning-rate", "-lr", type=float, 
                              help="Learning rate")
    create_parser.add_argument("--batch-size", "-bs", type=int, 
                              help="Batch size")
    create_parser.add_argument("--epochs", "-e", type=int, 
                              help="Number of epochs")
    create_parser.add_argument("--mode", "-m", choices=["online", "offline", "hybrid", "transfer"],
                              help="Training mode")
    create_parser.add_argument("--optimizer", "-opt", 
                              help="Optimizer (sgd, adam, rmsprop)")
    create_parser.add_argument("--custom", "-c", type=json.loads,
                              help="Custom parameters as JSON string")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List configurations")
    list_parser.add_argument("--hardware", "-hw", 
                            help="Filter by hardware type")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show configuration details")
    show_parser.add_argument("name", help="Configuration name")
    show_parser.add_argument("--hardware", "-hw", 
                            help="Hardware type")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update configuration")
    update_parser.add_argument("name", help="Configuration name")
    update_parser.add_argument("--hardware", "-hw", 
                              help="Hardware type")
    update_parser.add_argument("--learning-rate", "-lr", type=float, 
                              help="Learning rate")
    update_parser.add_argument("--batch-size", "-bs", type=int, 
                              help="Batch size")
    update_parser.add_argument("--epochs", "-e", type=int, 
                              help="Number of epochs")
    update_parser.add_argument("--mode", "-m", choices=["online", "offline", "hybrid", "transfer"],
                              help="Training mode")
    update_parser.add_argument("--optimizer", "-opt", 
                              help="Optimizer (sgd, adam, rmsprop)")
    update_parser.add_argument("--custom", "-c", type=json.loads,
                              help="Custom parameters as JSON string")
    
    return parser.parse_args()


def create_config(config_system: TrainingConfigSystem, args) -> None:
    """Create a new configuration."""
    # Build overrides dictionary
    overrides = {}
    if args.learning_rate:
        overrides["learning_rate"] = args.learning_rate
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.mode:
        overrides["mode"] = TrainingMode(args.mode)
    if args.optimizer:
        overrides["optimizer"] = args.optimizer
    if args.custom:
        for key, value in args.custom.items():
            overrides[key] = value
    
    # Create configuration
    config = config_system.create_config(args.hardware, args.name, overrides)
    
    # Print configuration
    print(f"Created configuration: {args.name}")
    print(f"Hardware type: {config.hardware_type}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Mode: {config.mode.value}")
    print(f"Optimizer: {config.optimizer}")


def list_configs(config_system: TrainingConfigSystem, args) -> None:
    """List configurations."""
    configs = config_system.list_configs(args.hardware)
    
    if not configs:
        print("No configurations found.")
        return
    
    print("Available configurations:")
    for hw_type, config_names in configs.items():
        print(f"\n{hw_type.upper()}:")
        for name in config_names:
            print(f"  - {name}")


def show_config(config_system: TrainingConfigSystem, args) -> None:
    """Show configuration details."""
    config = config_system.load_config(args.name, args.hardware)
    
    if not config:
        print(f"Configuration not found: {args.name}")
        return
    
    print(f"Configuration: {args.name}")
    print(f"Hardware type: {config.hardware_type}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Mode: {config.mode.value}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Optimizer params: {config.optimizer_params}")
    print(f"Early stopping: {config.early_stopping}")
    print(f"Patience: {config.patience}")
    print(f"Validation split: {config.validation_split}")
    print(f"Shuffle: {config.shuffle}")
    print(f"Seed: {config.seed}")
    print(f"Custom params: {config.custom_params}")


def update_config(config_system: TrainingConfigSystem, args) -> None:
    """Update configuration."""
    # Build updates dictionary
    updates = {}
    if args.learning_rate:
        updates["learning_rate"] = args.learning_rate
    if args.batch_size:
        updates["batch_size"] = args.batch_size
    if args.epochs:
        updates["epochs"] = args.epochs
    if args.mode:
        updates["mode"] = TrainingMode(args.mode)
    if args.optimizer:
        updates["optimizer"] = args.optimizer
    if args.custom:
        for key, value in args.custom.items():
            updates[key] = value
    
    # Update configuration
    config = config_system.update_config(args.name, updates, args.hardware)
    
    if not config:
        print(f"Configuration not found: {args.name}")
        return
    
    print(f"Updated configuration: {args.name}")
    print(f"Hardware type: {config.hardware_type}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Mode: {config.mode.value}")
    print(f"Optimizer: {config.optimizer}")


def main():
    """Main entry point."""
    args = parse_args()
    
    if not args.command:
        print("No command specified. Use --help for usage information.")
        return
    
    # Create configuration system
    config_system = create_config_system()
    
    # Execute command
    if args.command == "create":
        create_config(config_system, args)
    elif args.command == "list":
        list_configs(config_system, args)
    elif args.command == "show":
        show_config(config_system, args)
    elif args.command == "update":
        update_config(config_system, args)
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()