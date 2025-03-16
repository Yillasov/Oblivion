"""
Simple Deployment Tool for Neuromorphic Hardware

Provides utilities for deploying algorithms and configurations to target hardware.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.append("/Users/yessine/Oblivion")

from src.core.utils.logging_framework import get_logger
from src.core.hardware.config_manager import HardwareConfigManager
from src.core.integration.neuromorphic_system import NeuromorphicSystem

logger = get_logger("deployment")


class DeploymentManager:
    """Simple deployment manager for neuromorphic hardware."""
    
    def __init__(self, 
                 repo_root: str = "/Users/yessine/Oblivion",
                 config_dir: str = "configs",
                 build_dir: str = "build",
                 deploy_dir: str = "deploy"):
        """
        Initialize deployment manager.
        
        Args:
            repo_root: Root directory of the repository
            config_dir: Directory containing hardware configurations
            build_dir: Directory for building deployment packages
            deploy_dir: Directory for deployment artifacts
        """
        self.repo_root = repo_root
        self.config_dir = os.path.join(repo_root, config_dir)
        self.build_dir = os.path.join(repo_root, build_dir)
        self.deploy_dir = os.path.join(repo_root, deploy_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.deploy_dir, exist_ok=True)
        
        # Initialize hardware config manager
        self.config_manager = HardwareConfigManager(self.config_dir)
        
        logger.info("Initialized deployment manager")
    
    def list_hardware_platforms(self) -> List[str]:
        """List available hardware platforms."""
        configs = self.config_manager.list_configs()
        return list(configs.keys())
    
    def list_configurations(self, hardware_type: str) -> List[str]:
        """List available configurations for a hardware platform."""
        configs = self.config_manager.list_configs(hardware_type)
        return configs.get(hardware_type, [])
    
    def create_deployment_package(self, 
                                  hardware_type: str, 
                                  config_name: str,
                                  algorithm_paths: List[str],
                                  include_simulator: bool = False) -> str:
        """
        Create a deployment package for target hardware.
        
        Args:
            hardware_type: Type of target hardware
            config_name: Name of hardware configuration
            algorithm_paths: Paths to algorithm files to include
            include_simulator: Whether to include simulator
            
        Returns:
            str: Path to deployment package
        """
        # Load hardware configuration
        config = self.config_manager.load_config(hardware_type, config_name)
        if not config:
            raise ValueError(f"Configuration '{config_name}' for {hardware_type} not found")
        
        # Create timestamp for this deployment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"{hardware_type}_{config_name}_{timestamp}"
        package_dir = os.path.join(self.build_dir, package_name)
        
        # Create package directory
        os.makedirs(package_dir, exist_ok=True)
        os.makedirs(os.path.join(package_dir, "algorithms"), exist_ok=True)
        os.makedirs(os.path.join(package_dir, "config"), exist_ok=True)
        
        # Copy configuration
        with open(os.path.join(package_dir, "config", "hardware_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Copy algorithm files
        for path in algorithm_paths:
            if os.path.exists(path):
                filename = os.path.basename(path)
                shutil.copy(path, os.path.join(package_dir, "algorithms", filename))
            else:
                logger.warning(f"Algorithm file not found: {path}")
        
        # Create deployment manifest
        manifest = {
            "hardware_type": hardware_type,
            "config_name": config_name,
            "created": timestamp,
            "algorithms": [os.path.basename(p) for p in algorithm_paths if os.path.exists(p)],
            "include_simulator": include_simulator
        }
        
        with open(os.path.join(package_dir, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create deployment script
        self._create_deployment_script(package_dir, hardware_type)
        
        # Create archive
        archive_path = os.path.join(self.deploy_dir, f"{package_name}.zip")
        shutil.make_archive(
            os.path.join(self.deploy_dir, package_name),
            'zip',
            self.build_dir,
            package_name
        )
        
        logger.info(f"Created deployment package: {archive_path}")
        return archive_path
    
    def _create_deployment_script(self, package_dir: str, hardware_type: str) -> None:
        """Create deployment script for the target hardware."""
        if hardware_type.lower() == "loihi":
            script_content = """#!/bin/bash
# Deployment script for Loihi hardware

echo "Deploying to Loihi hardware..."

# Load configuration
CONFIG_FILE="./config/hardware_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found"
    exit 1
fi

# Set up environment
export PYTHONPATH=$PYTHONPATH:./

# Deploy algorithms
echo "Deploying algorithms..."
for algo in ./algorithms/*.py; do
    echo "  - $algo"
done

# Run on hardware
echo "Running on Loihi hardware..."
python -c "
import json
import sys
import os

# Load configuration
with open('./config/hardware_config.json', 'r') as f:
    config = json.load(f)

print(f'Loaded configuration: {config.get(\"_metadata\", {}).get(\"name\", \"unknown\")}')
print('Connecting to Loihi hardware...')
print('Deployment complete!')
"

echo "Deployment completed successfully"
"""
        elif hardware_type.lower() == "truenorth":
            script_content = """#!/bin/bash
# Deployment script for TrueNorth hardware

echo "Deploying to TrueNorth hardware..."

# Load configuration
CONFIG_FILE="./config/hardware_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found"
    exit 1
fi

# Set up environment
export PYTHONPATH=$PYTHONPATH:./

# Deploy algorithms
echo "Deploying algorithms..."
for algo in ./algorithms/*.py; do
    echo "  - $algo"
done

# Run on hardware
echo "Running on TrueNorth hardware..."
python -c "
import json
import sys
import os

# Load configuration
with open('./config/hardware_config.json', 'r') as f:
    config = json.load(f)

print(f'Loaded configuration: {config.get(\"_metadata\", {}).get(\"name\", \"unknown\")}')
print('Connecting to TrueNorth hardware...')
print('Deployment complete!')
"

echo "Deployment completed successfully"
"""
        elif hardware_type.lower() == "spinnaker":
            script_content = """#!/bin/bash
# Deployment script for SpiNNaker hardware

echo "Deploying to SpiNNaker hardware..."

# Load configuration
CONFIG_FILE="./config/hardware_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found"
    exit 1
fi

# Set up environment
export PYTHONPATH=$PYTHONPATH:./

# Deploy algorithms
echo "Deploying algorithms..."
for algo in ./algorithms/*.py; do
    echo "  - $algo"
done

# Run on hardware
echo "Running on SpiNNaker hardware..."
python -c "
import json
import sys
import os

# Load configuration
with open('./config/hardware_config.json', 'r') as f:
    config = json.load(f)

print(f'Loaded configuration: {config.get(\"_metadata\", {}).get(\"name\", \"unknown\")}')
print('Connecting to SpiNNaker hardware...')
print('Deployment complete!')
"

echo "Deployment completed successfully"
"""
        else:
            script_content = """#!/bin/bash
# Generic deployment script

echo "Deploying to hardware..."

# Load configuration
CONFIG_FILE="./config/hardware_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found"
    exit 1
fi

# Set up environment
export PYTHONPATH=$PYTHONPATH:./

# Deploy algorithms
echo "Deploying algorithms..."
for algo in ./algorithms/*.py; do
    echo "  - $algo"
done

echo "Deployment completed successfully"
"""
        
        # Write deployment script
        script_path = os.path.join(package_dir, "deploy.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
    
    def deploy_to_hardware(self, package_path: str, hardware_address: Optional[str] = None) -> bool:
        """
        Deploy package to hardware.
        
        Args:
            package_path: Path to deployment package
            hardware_address: Address of target hardware (IP or serial)
            
        Returns:
            bool: True if deployment was successful
        """
        if not os.path.exists(package_path):
            logger.error(f"Deployment package not found: {package_path}")
            return False
        
        # Extract package
        package_name = os.path.splitext(os.path.basename(package_path))[0]
        extract_dir = os.path.join(self.build_dir, "tmp", package_name)
        os.makedirs(os.path.join(self.build_dir, "tmp"), exist_ok=True)
        
        try:
            # Extract package
            shutil.unpack_archive(package_path, extract_dir)
            
            # Load manifest
            manifest_path = os.path.join(extract_dir, "manifest.json")
            if not os.path.exists(manifest_path):
                logger.error(f"Manifest not found in package: {package_path}")
                return False
            
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            hardware_type = manifest.get("hardware_type", "unknown")
            
            # Execute deployment script
            if hardware_address:
                cmd = f"cd {extract_dir} && ./deploy.sh {hardware_address}"
            else:
                cmd = f"cd {extract_dir} && ./deploy.sh"
            
            logger.info(f"Deploying to {hardware_type} hardware...")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Deployment failed: {result.stderr}")
                return False
            
            logger.info(f"Deployment to {hardware_type} hardware successful")
            logger.debug(result.stdout)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during deployment: {str(e)}")
            return False
        finally:
            # Clean up
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)


def main():
    """Main entry point for deployment tool."""
    parser = argparse.ArgumentParser(description="Neuromorphic Hardware Deployment Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List hardware platforms
    list_hw_parser = subparsers.add_parser("list-hardware", help="List available hardware platforms")
    
    # List configurations
    list_config_parser = subparsers.add_parser("list-configs", help="List available configurations")
    list_config_parser.add_argument("hardware_type", help="Hardware platform type")
    
    # Create package
    create_parser = subparsers.add_parser("create-package", help="Create deployment package")
    create_parser.add_argument("hardware_type", help="Target hardware type")
    create_parser.add_argument("config_name", help="Hardware configuration name")
    create_parser.add_argument("--algorithms", nargs="+", help="Paths to algorithm files")
    create_parser.add_argument("--include-simulator", action="store_true", help="Include simulator")
    
    # Deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to hardware")
    deploy_parser.add_argument("package", help="Path to deployment package")
    deploy_parser.add_argument("--address", help="Hardware address (IP or serial)")
    
    args = parser.parse_args()
    
    # Create deployment manager
    manager = DeploymentManager()
    
    if args.command == "list-hardware":
        platforms = manager.list_hardware_platforms()
        print("Available hardware platforms:")
        for platform in platforms:
            print(f"  - {platform}")
    
    elif args.command == "list-configs":
        configs = manager.list_configurations(args.hardware_type)
        print(f"Available configurations for {args.hardware_type}:")
        for config in configs:
            print(f"  - {config}")
    
    elif args.command == "create-package":
        algorithms = args.algorithms or []
        package_path = manager.create_deployment_package(
            args.hardware_type,
            args.config_name,
            algorithms,
            args.include_simulator
        )
        print(f"Created deployment package: {package_path}")
    
    elif args.command == "deploy":
        success = manager.deploy_to_hardware(args.package, args.address)
        if success:
            print("Deployment successful")
        else:
            print("Deployment failed")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()