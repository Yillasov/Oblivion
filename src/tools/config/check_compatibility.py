#!/usr/bin/env python3
"""
Hardware Compatibility Checker

Command-line tool to check hardware configuration compatibility.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#!/usr/bin/env python3


import argparse
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.hardware.compatibility_validator import HardwareCompatibilityValidator
from src.core.hardware.unified_config_manager import UnifiedConfigManager

def check_config(args):
    """Check configuration compatibility."""
    # Load configuration
    config_manager = UnifiedConfigManager.get_instance()
    
    if args.file:
        # Load from file
        try:
            with open(args.file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading configuration file: {str(e)}")
            return False
    else:
        # Load from config manager
        config = config_manager.load_config(args.hardware, args.name)
        if not config:
            print(f"Configuration '{args.name}' not found for {args.hardware}")
            return False
    
    # Validate configuration
    is_compatible, issues = HardwareCompatibilityValidator.validate_compatibility(args.hardware, config)
    
    if is_compatible:
        print(f"Configuration is compatible with {args.hardware}")
        return True
    else:
        print(f"Configuration is NOT compatible with {args.hardware}:")
        for issue in issues:
            print(f"  - {issue}")
        return False

def check_migration(args):
    """Check migration compatibility."""
    # Load configuration
    config_manager = UnifiedConfigManager.get_instance()
    
    if args.file:
        # Load from file
        try:
            with open(args.file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading configuration file: {str(e)}")
            return False
    else:
        # Load from config manager
        config = config_manager.load_config(args.source, args.name)
        if not config:
            print(f"Configuration '{args.name}' not found for {args.source}")
            return False
    
    # Check migration compatibility
    can_migrate, issues = HardwareCompatibilityValidator.check_migration_compatibility(
        args.source, args.target, config
    )
    
    if can_migrate:
        print(f"Configuration can be migrated from {args.source} to {args.target}")
        return True
    else:
        print(f"Configuration CANNOT be migrated from {args.source} to {args.target}:")
        for issue in issues:
            print(f"  - {issue}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hardware Compatibility Checker")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check configuration compatibility")
    check_parser.add_argument("hardware", help="Hardware type")
    check_parser.add_argument("--name", "-n", help="Configuration name")
    check_parser.add_argument("--file", "-f", help="Configuration file path")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Check migration compatibility")
    migrate_parser.add_argument("source", help="Source hardware type")
    migrate_parser.add_argument("target", help="Target hardware type")
    migrate_parser.add_argument("--name", "-n", help="Configuration name")
    migrate_parser.add_argument("--file", "-f", help="Configuration file path")
    
    args = parser.parse_args()
    
    if args.command == "check":
        if not args.name and not args.file:
            check_parser.error("Either --name or --file is required")
        check_config(args)
    elif args.command == "migrate":
        if not args.name and not args.file:
            migrate_parser.error("Either --name or --file is required")
        check_migration(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()