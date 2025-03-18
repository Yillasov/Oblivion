#!/usr/bin/env python3
"""
Configuration Migration Tool

Command-line tool for migrating hardware configurations between different hardware types.
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.append("/Users/yessine/Oblivion")

from src.core.utils.logging_framework import get_logger
from src.core.hardware.config_migration import ConfigMigration
from src.core.hardware.hardware_config import config_store

logger = get_logger("migrate_config")


def list_hardware_types():
    """List available hardware types."""
    hardware_types = ["loihi", "spinnaker", "truenorth", "simulated"]
    print("\nAvailable hardware types:")
    for hw_type in hardware_types:
        print(f"  - {hw_type}")


def list_configs(hardware_type: str):
    """List configurations for a hardware type."""
    configs = config_store.list_configs(hardware_type)
    
    if hardware_type in configs and configs[hardware_type]:
        print(f"\nConfigurations for {hardware_type}:")
        for config in configs[hardware_type]:
            print(f"  - {config}")
    else:
        print(f"No configurations found for {hardware_type}")


def migrate_config(source_type: str, source_name: str, target_type: str, target_name: str):
    """Migrate configuration between hardware types."""
    print(f"Migrating {source_type}/{source_name} to {target_type}/{target_name}...")
    
    result = ConfigMigration.migrate_config(source_type, source_name, target_type, target_name)
    
    if result:
        print(f"Successfully migrated configuration to {target_type}/{target_name}")
    else:
        print(f"Failed to migrate configuration")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hardware Configuration Migration Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List hardware types
    list_hw_parser = subparsers.add_parser("list-hardware", help="List available hardware types")
    
    # List configurations
    list_config_parser = subparsers.add_parser("list-configs", help="List configurations for hardware type")
    list_config_parser.add_argument("hardware_type", help="Hardware type")
    
    # Migrate configuration
    migrate_parser = subparsers.add_parser("migrate", help="Migrate configuration")
    migrate_parser.add_argument("source_type", help="Source hardware type")
    migrate_parser.add_argument("source_name", help="Source configuration name")
    migrate_parser.add_argument("target_type", help="Target hardware type")
    migrate_parser.add_argument("target_name", help="Target configuration name")
    
    args = parser.parse_args()
    
    if args.command == "list-hardware":
        list_hardware_types()
    elif args.command == "list-configs":
        list_configs(args.hardware_type)
    elif args.command == "migrate":
        migrate_config(args.source_type, args.source_name, args.target_type, args.target_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()