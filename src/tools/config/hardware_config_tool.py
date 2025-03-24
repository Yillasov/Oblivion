#!/usr/bin/env python3
"""
Hardware Configuration Tool

Command-line tool for managing hardware configurations.
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append("/Users/yessine/Oblivion")

from src.core.hardware.unified_config_manager import (
    UnifiedConfigManager, ConfigCategory, HardwareType
)
from src.core.utils.logging_framework import get_logger

logger = get_logger("config_tool")


def list_configs(args):
    """List configurations."""
    config_manager = UnifiedConfigManager.get_instance()
    
    if args.templates:
        # List templates
        templates = config_manager.list_templates(args.hardware)
        
        print("\nAvailable Templates:")
        for hw_type, hw_templates in templates.items():
            print(f"\n{hw_type.upper()}:")
            for template in hw_templates:
                print(f"  - {template}")
    else:
        # List configurations
        category = ConfigCategory(args.category) if args.category else ConfigCategory.HARDWARE
        configs = config_manager.list_configs(args.hardware, category)
        
        print(f"\nAvailable Configurations ({category.value}):")
        for hw_type, hw_configs in configs.items():
            print(f"\n{hw_type.upper()}:")
            for config in hw_configs:
                print(f"  - {config}")


def show_config(args):
    """Show configuration details."""
    config_manager = UnifiedConfigManager.get_instance()
    
    if args.template:
        # Show template
        template = config_manager.get_template(args.hardware, args.name)
        if template:
            print(f"\nTemplate: {args.name} ({args.hardware})")
            print(json.dumps(template, indent=2))
        else:
            print(f"Template '{args.name}' not found for {args.hardware}")
    else:
        # Show configuration
        category = ConfigCategory(args.category) if args.category else ConfigCategory.HARDWARE
        config = config_manager.load_config(args.hardware, args.name, category)
        
        if config:
            print(f"\nConfiguration: {args.name} ({args.hardware}, {category.value})")
            print(json.dumps(config, indent=2))
        else:
            print(f"Configuration '{args.name}' not found for {args.hardware} in {category.value}")


def create_config(args):
    """Create a new configuration."""
    config_manager = UnifiedConfigManager.get_instance()
    
    # Parse overrides
    overrides = {}
    if args.override:
        for override in args.override:
            if "=" in override:
                key, value = override.split("=", 1)
                
                # Try to convert value to appropriate type
                try:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif "." in value and all(p.isdigit() for p in value.split(".")):
                        value = float(value)
                except ValueError:
                    pass  # Keep as string if conversion fails
                
                overrides[key] = value
    
    # Create configuration
    category = ConfigCategory(args.category) if args.category else ConfigCategory.HARDWARE
    
    if args.template:
        # Create from template
        config = config_manager.create_config_from_template(
            args.hardware, args.template, args.name, category, overrides
        )
        
        if config:
            print(f"Created configuration '{args.name}' for {args.hardware} from template '{args.template}'")
        else:
            print(f"Failed to create configuration from template")
    else:
        # Create empty configuration with overrides
        if not overrides:
            print("Error: Must provide either a template or overrides")
            return
            
        # Start with default template if available
        config = config_manager.get_template(args.hardware, "default")
        if config:
            # Remove metadata
            if "_metadata" in config:
                del config["_metadata"]
                
            # Apply overrides
            for key, value in overrides.items():
                config[key] = value
        else:
            # Create from scratch
            config = overrides
        
        # Save configuration
        success = config_manager.save_config(args.hardware, args.name, config, category)
        
        if success:
            print(f"Created configuration '{args.name}' for {args.hardware}")
        else:
            print(f"Failed to create configuration")


def delete_config(args):
    """Delete a configuration."""
    config_manager = UnifiedConfigManager.get_instance()
    
    if args.template:
        # Delete template
        # Not implemented - templates are read-only for now
        print("Deleting templates is not supported")
    else:
        # Delete configuration
        category = ConfigCategory(args.category) if args.category else ConfigCategory.HARDWARE
        success = config_manager.delete_config(args.hardware, args.name, category)
        
        if success:
            print(f"Deleted configuration '{args.name}' for {args.hardware}")
        else:
            print(f"Failed to delete configuration")


def convert_config(args):
    """Convert configuration between hardware types."""
    config_manager = UnifiedConfigManager.get_instance()
    
    # Load source configuration
    source_category = ConfigCategory(args.source_category) if args.source_category else ConfigCategory.HARDWARE
    source_config = config_manager.load_config(args.source_hardware, args.source_name, source_category)
    
    if not source_config:
        print(f"Source configuration '{args.source_name}' not found")
        return
    
    # Convert configuration
    target_config = config_manager.convert_config(
        args.source_hardware, args.target_hardware, source_config
    )
    
    # Save target configuration
    target_category = ConfigCategory(args.target_category) if args.target_category else source_category
    success = config_manager.save_config(
        args.target_hardware, args.target_name, target_config, target_category
    )
    
    if success:
        print(f"Converted configuration from {args.source_hardware}/{args.source_name} to {args.target_hardware}/{args.target_name}")
    else:
        print(f"Failed to convert configuration")


def set_active(args):
    """Set active configuration."""
    config_manager = UnifiedConfigManager.get_instance()
    
    category = ConfigCategory(args.category) if args.category else ConfigCategory.HARDWARE
    success = config_manager.set_active_config(args.hardware, args.name, category)
    
    if success:
        print(f"Set active configuration for {args.hardware} to '{args.name}'")
    else:
        print(f"Failed to set active configuration")


def show_active(args):
    """Show active configuration."""
    config_manager = UnifiedConfigManager.get_instance()
    
    if args.hardware:
        # Show active configuration for specific hardware
        config = config_manager.get_active_config(args.hardware)
        
        if config:
            print(f"\nActive Configuration for {args.hardware}:")
            print(json.dumps(config, indent=2))
        else:
            print(f"No active configuration set for {args.hardware}")
    else:
        # Show all active configurations
        active_configs = {}
        for hw_type in HardwareType.list():
            config = config_manager.get_active_config(hw_type)
            if config:
                active_configs[hw_type] = config
        
        if active_configs:
            print("\nActive Configurations:")
            for hw_type, config in active_configs.items():
                print(f"\n{hw_type.upper()}:")
                if "_metadata" in config:
                    print(f"  Name: {config['_metadata']['name']}")
                    print(f"  Category: {config['_metadata']['category']}")
                else:
                    print("  (No metadata available)")
        else:
            print("No active configurations set")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hardware Configuration Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List configurations")
    list_parser.add_argument("--hardware", "-hw", help="Hardware type filter")
    list_parser.add_argument("--category", "-c", help="Configuration category")
    list_parser.add_argument("--templates", "-t", action="store_true", help="List templates instead of configurations")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show configuration details")
    show_parser.add_argument("hardware", help="Hardware type")
    show_parser.add_argument("name", help="Configuration name")
    show_parser.add_argument("--category", "-c", help="Configuration category")
    show_parser.add_argument("--template", "-t", action="store_true", help="Show template instead of configuration")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new configuration")
    create_parser.add_argument("hardware", help="Hardware type")
    create_parser.add_argument("name", help="Configuration name")
    create_parser.add_argument("--template", "-t", help="Template to use")
    create_parser.add_argument("--category", "-c", help="Configuration category")
    create_parser.add_argument("--override", "-o", action="append", help="Override parameter (key=value)")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a configuration")
    delete_parser.add_argument("hardware", help="Hardware type")
    delete_parser.add_argument("name", help="Configuration name")
    delete_parser.add_argument("--category", "-c", help="Configuration category")
    delete_parser.add_argument("--template", "-t", action="store_true", help="Delete template instead of configuration")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert configuration between hardware types")
    convert_parser.add_argument("source_hardware", help="Source hardware type")
    convert_parser.add_argument("source_name", help="Source configuration name")
    convert_parser.add_argument("target_hardware", help="Target hardware type")
    convert_parser.add_argument("target_name", help="Target configuration name")
    convert_parser.add_argument("--source-category", "-sc", help="Source configuration category")
    convert_parser.add_argument("--target-category", "-tc", help="Target configuration category")
    
    # Set active command
    active_parser = subparsers.add_parser("set-active", help="Set active configuration")
    active_parser.add_argument("hardware", help="Hardware type")
    active_parser.add_argument("name", help="Configuration name")
    active_parser.add_argument("--category", "-c", help="Configuration category")
    
    # Show active command
    show_active_parser = subparsers.add_parser("show-active", help="Show active configuration")
    show_active_parser.add_argument("--hardware", "-hw", help="Hardware type")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_configs(args)
    elif args.command == "show":
        show_config(args)
    elif args.command == "create":
        create_config(args)
    elif args.command == "delete":
        delete_config(args)
    elif args.command == "convert":
        convert_config(args)
    elif args.command == "set-active":
        set_active(args)
    elif args.command == "show-active":
        show_active(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()