"""
Simple Configuration Management System

Provides utilities for managing configurations across the system.
"""

import os
import sys
import json
import yaml
import shutil
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Add project root to path
sys.path.append("/Users/yessine/Oblivion")

from src.core.utils.logging_framework import get_logger

logger = get_logger("config_manager")


class ConfigManager:
    """Simple configuration management system."""
    
    def __init__(self, 
                 config_dir: str = "/Users/yessine/Oblivion/configs",
                 default_format: str = "json"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configurations
            default_format: Default configuration format (json or yaml)
        """
        self.config_dir = config_dir
        self.default_format = default_format
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Create subdirectories for different config types
        self.categories = ["hardware", "monitoring", "deployment", "simulation"]
        for category in self.categories:
            os.makedirs(os.path.join(self.config_dir, category), exist_ok=True)
        
        logger.info(f"Initialized configuration manager with root at {self.config_dir}")
    
    def list_configs(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available configurations.
        
        Args:
            category: Configuration category (optional)
            
        Returns:
            Dict[str, List[str]]: Dictionary of categories and their configurations
        """
        result = {}
        
        if category:
            if category not in self.categories:
                logger.warning(f"Unknown category: {category}")
                return {}
            
            categories = [category]
        else:
            categories = self.categories
        
        for cat in categories:
            cat_dir = os.path.join(self.config_dir, cat)
            if os.path.exists(cat_dir):
                configs = []
                for file in os.listdir(cat_dir):
                    if file.endswith((".json", ".yaml", ".yml")):
                        configs.append(os.path.splitext(file)[0])
                result[cat] = configs
        
        return result
    
    def load_config(self, category: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Load a configuration.
        
        Args:
            category: Configuration category
            name: Configuration name
            
        Returns:
            Optional[Dict[str, Any]]: Configuration data or None if not found
        """
        if category not in self.categories:
            logger.warning(f"Unknown category: {category}")
            return None
        
        # Check for JSON file
        json_path = os.path.join(self.config_dir, category, f"{name}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading JSON config {name}: {str(e)}")
                return None
        
        # Check for YAML file
        yaml_path = os.path.join(self.config_dir, category, f"{name}.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading YAML config {name}: {str(e)}")
                return None
        
        # Check for YML file
        yml_path = os.path.join(self.config_dir, category, f"{name}.yml")
        if os.path.exists(yml_path):
            try:
                with open(yml_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading YAML config {name}: {str(e)}")
                return None
        
        logger.warning(f"Configuration not found: {category}/{name}")
        return None
    
    def save_config(self, 
                   category: str, 
                   name: str, 
                   config: Dict[str, Any],
                   format: Optional[str] = None) -> bool:
        """
        Save a configuration.
        
        Args:
            category: Configuration category
            name: Configuration name
            config: Configuration data
            format: File format (json or yaml, defaults to self.default_format)
            
        Returns:
            bool: True if saved successfully
        """
        if category not in self.categories:
            logger.warning(f"Unknown category: {category}")
            return False
        
        # Use default format if not specified
        if not format:
            format = self.default_format
        
        # Add metadata
        if "_metadata" not in config:
            config["_metadata"] = {}
        
        config["_metadata"]["name"] = name
        config["_metadata"]["category"] = category
        config["_metadata"]["last_updated"] = datetime.now().isoformat()
        
        # Save in specified format
        cat_dir = os.path.join(self.config_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        try:
            if format.lower() == "json":
                file_path = os.path.join(cat_dir, f"{name}.json")
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
            elif format.lower() in ["yaml", "yml"]:
                file_path = os.path.join(cat_dir, f"{name}.yaml")
                with open(file_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Saved configuration: {category}/{name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration {category}/{name}: {str(e)}")
            return False
    
    def delete_config(self, category: str, name: str) -> bool:
        """
        Delete a configuration.
        
        Args:
            category: Configuration category
            name: Configuration name
            
        Returns:
            bool: True if deleted successfully
        """
        if category not in self.categories:
            logger.warning(f"Unknown category: {category}")
            return False
        
        # Check for all possible file extensions
        for ext in [".json", ".yaml", ".yml"]:
            file_path = os.path.join(self.config_dir, category, f"{name}{ext}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted configuration: {category}/{name}")
                    return True
                except Exception as e:
                    logger.error(f"Error deleting configuration {category}/{name}: {str(e)}")
                    return False
        
        logger.warning(f"Configuration not found: {category}/{name}")
        return False
    
    def create_backup(self, output_dir: Optional[str] = None) -> str:
        """
        Create a backup of all configurations.
        
        Args:
            output_dir: Directory to store backup (optional)
            
        Returns:
            str: Path to backup file
        """
        if not output_dir:
            output_dir = os.path.join(self.config_dir, "backups")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(output_dir, f"config_backup_{timestamp}.zip")
        
        try:
            # Create zip archive
            shutil.make_archive(
                os.path.splitext(backup_file)[0],
                'zip',
                self.config_dir
            )
            
            logger.info(f"Created configuration backup: {backup_file}")
            return backup_file
            
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return ""
    
    def restore_backup(self, backup_file: str) -> bool:
        """
        Restore configurations from backup.
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            bool: True if restored successfully
        """
        if not os.path.exists(backup_file):
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        try:
            # Create temporary directory
            temp_dir = os.path.join(self.config_dir, "temp_restore")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract backup
            shutil.unpack_archive(backup_file, temp_dir)
            
            # Copy configurations
            for category in self.categories:
                src_dir = os.path.join(temp_dir, category)
                dst_dir = os.path.join(self.config_dir, category)
                
                if os.path.exists(src_dir):
                    # Create category directory if it doesn't exist
                    os.makedirs(dst_dir, exist_ok=True)
                    
                    # Copy files
                    for file in os.listdir(src_dir):
                        if file.endswith((".json", ".yaml", ".yml")):
                            shutil.copy2(
                                os.path.join(src_dir, file),
                                os.path.join(dst_dir, file)
                            )
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            logger.info(f"Restored configurations from backup: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup: {str(e)}")
            return False


def main():
    """Main entry point for configuration manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Management System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List configurations
    list_parser = subparsers.add_parser("list", help="List configurations")
    list_parser.add_argument("--category", help="Configuration category")
    
    # Get configuration
    get_parser = subparsers.add_parser("get", help="Get configuration")
    get_parser.add_argument("category", help="Configuration category")
    get_parser.add_argument("name", help="Configuration name")
    
    # Create/update configuration
    set_parser = subparsers.add_parser("set", help="Set configuration")
    set_parser.add_argument("category", help="Configuration category")
    set_parser.add_argument("name", help="Configuration name")
    set_parser.add_argument("--file", help="JSON/YAML file to load")
    set_parser.add_argument("--format", choices=["json", "yaml"], help="Output format")
    
    # Delete configuration
    delete_parser = subparsers.add_parser("delete", help="Delete configuration")
    delete_parser.add_argument("category", help="Configuration category")
    delete_parser.add_argument("name", help="Configuration name")
    
    # Backup configurations
    backup_parser = subparsers.add_parser("backup", help="Backup configurations")
    backup_parser.add_argument("--output-dir", help="Output directory")
    
    # Restore configurations
    restore_parser = subparsers.add_parser("restore", help="Restore configurations")
    restore_parser.add_argument("backup_file", help="Backup file")
    
    args = parser.parse_args()
    
    # Create configuration manager
    config_manager = ConfigManager()
    
    if args.command == "list":
        configs = config_manager.list_configs(args.category)
        
        for category, names in configs.items():
            print(f"{category}:")
            for name in names:
                print(f"  - {name}")
    
    elif args.command == "get":
        config = config_manager.load_config(args.category, args.name)
        
        if config:
            print(json.dumps(config, indent=2))
        else:
            print(f"Configuration not found: {args.category}/{args.name}")
            sys.exit(1)
    
    elif args.command == "set":
        if args.file:
            # Load from file
            try:
                if args.file.endswith(".json"):
                    with open(args.file, 'r') as f:
                        config = json.load(f)
                elif args.file.endswith((".yaml", ".yml")):
                    with open(args.file, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    print(f"Unsupported file format: {args.file}")
                    sys.exit(1)
            except Exception as e:
                print(f"Error loading file: {str(e)}")
                sys.exit(1)
        else:
            # Interactive mode
            print("Enter configuration data (JSON format):")
            try:
                config = json.loads(input())
            except Exception as e:
                print(f"Error parsing input: {str(e)}")
                sys.exit(1)
        
        # Save configuration
        success = config_manager.save_config(
            args.category,
            args.name,
            config,
            args.format
        )
        
        if success:
            print(f"Saved configuration: {args.category}/{args.name}")
        else:
            print(f"Failed to save configuration: {args.category}/{args.name}")
            sys.exit(1)
    
    elif args.command == "delete":
        success = config_manager.delete_config(args.category, args.name)
        
        if success:
            print(f"Deleted configuration: {args.category}/{args.name}")
        else:
            print(f"Failed to delete configuration: {args.category}/{args.name}")
            sys.exit(1)
    
    elif args.command == "backup":
        backup_file = config_manager.create_backup(args.output_dir)
        
        if backup_file:
            print(f"Created backup: {backup_file}")
        else:
            print("Failed to create backup")
            sys.exit(1)
    
    elif args.command == "restore":
        success = config_manager.restore_backup(args.backup_file)
        
        if success:
            print(f"Restored configurations from: {args.backup_file}")
        else:
            print(f"Failed to restore configurations from: {args.backup_file}")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()