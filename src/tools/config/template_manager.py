#!/usr/bin/env python3
"""
Configuration Template Manager

Command-line tool for managing hardware configuration templates.
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.append("/Users/yessine/Oblivion")

from src.core.utils.logging_framework import get_logger
from src.core.hardware.config_templates import ConfigTemplates

logger = get_logger("template_manager")


def list_templates(hardware_type: Optional[str] = None) -> None:
    """
    List available templates.
    
    Args:
        hardware_type: Optional hardware type filter
    """
    templates = ConfigTemplates.get_template_list()
    
    if hardware_type:
        if hardware_type in templates:
            print(f"\nTemplates for {hardware_type}:")
            for template in templates[hardware_type]:
                print(f"  - {template}")
        else:
            print(f"No templates available for {hardware_type}")
    else:
        print("\nAvailable templates:")
        for hw_type, hw_templates in templates.items():
            print(f"\n{hw_type.upper()}:")
            for template in hw_templates:
                print(f"  - {template}")


def apply_template(hardware_type: str, template_name: str, config_name: str) -> None:
    """
    Apply template and save as configuration.
    
    Args:
        hardware_type: Hardware type
        template_name: Template name
        config_name: Name for the new configuration
    """
    result = ConfigTemplates.apply_template(hardware_type, template_name, config_name)
    
    if result:
        print(f"Successfully applied template '{template_name}' for {hardware_type} as '{config_name}'")
    else:
        print(f"Failed to apply template '{template_name}' for {hardware_type}")


def show_template(hardware_type: str, template_name: str) -> None:
    """
    Show template details.
    
    Args:
        hardware_type: Hardware type
        template_name: Template name
    """
    import json
    
    template = ConfigTemplates.get_template(hardware_type, template_name)
    
    if template:
        print(f"\nTemplate: {template_name} ({hardware_type})")
        print(json.dumps(template, indent=2))
    else:
        print(f"Template '{template_name}' not found for {hardware_type}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hardware Configuration Template Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List templates
    list_parser = subparsers.add_parser("list", help="List available templates")
    list_parser.add_argument("--hardware", help="Hardware type filter")
    
    # Show template
    show_parser = subparsers.add_parser("show", help="Show template details")
    show_parser.add_argument("hardware", help="Hardware type")
    show_parser.add_argument("template", help="Template name")
    
    # Apply template
    apply_parser = subparsers.add_parser("apply", help="Apply template")
    apply_parser.add_argument("hardware", help="Hardware type")
    apply_parser.add_argument("template", help="Template name")
    apply_parser.add_argument("config_name", help="Name for the new configuration")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_templates(args.hardware)
    elif args.command == "show":
        show_template(args.hardware, args.template)
    elif args.command == "apply":
        apply_template(args.hardware, args.template, args.config_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()