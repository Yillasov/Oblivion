#!/usr/bin/env python3
"""
UCAV Hardware Profile Tool

Command-line tool for working with UCAV hardware profiles.
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

from src.core.hardware.ucav_profile_integration import (
    register_ucav_profiles, 
    get_ucav_profile, 
    create_ucav_config, 
    list_ucav_profiles
)
from src.core.hardware.unified_config_manager import UnifiedConfigManager

def list_profiles(args):
    """List available UCAV profiles."""
    profiles = list_ucav_profiles()
    
    print("\nAvailable UCAV Hardware Profiles:")
    for hw_type, profile_names in profiles.items():
        if args.hardware and hw_type != args.hardware:
            continue
        print(f"\n{hw_type.upper()}:")
        for name in profile_names:
            print(f"  - {name}")

def show_profile(args):
    """Show UCAV profile details."""
    profile = get_ucav_profile(args.hardware, args.profile)
    
    if profile:
        print(f"\nUCAV Profile: {args.hardware}/{args.profile}")
        print(json.dumps(profile, indent=2))
    else:
        print(f"Profile not found: {args.hardware}/{args.profile}")

def create_config(args):
    """Create configuration from UCAV profile."""
    # Parse overrides
    overrides = {}
    if args.override:
        for override in args.override:
            if "=" in override:
                key, value = override.split("=", 1)
                
                # Try to convert to appropriate type
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
    success = create_ucav_config(args.hardware, args.profile, args.name, overrides)
    
    if success:
        print(f"Created configuration '{args.name}' for {args.hardware} from UCAV profile '{args.profile}'")
    else:
        print(f"Failed to create configuration from UCAV profile")

def main():
    """Main entry point."""
    # Register profiles
    register_ucav_profiles()
    
    parser = argparse.ArgumentParser(description="UCAV Hardware Profile Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List UCAV profiles")
    list_parser.add_argument("--hardware", "-hw", help="Hardware type filter")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show UCAV profile details")
    show_parser.add_argument("hardware", help="Hardware type")
    show_parser.add_argument("profile", help="Profile name")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create configuration from UCAV profile")
    create_parser.add_argument("hardware", help="Hardware type")
    create_parser.add_argument("profile", help="Profile name")
    create_parser.add_argument("name", help="Configuration name")
    create_parser.add_argument("--override", "-o", action="append", help="Override parameter (key=value)")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_profiles(args)
    elif args.command == "show":
        show_profile(args)
    elif args.command == "create":
        create_config(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()