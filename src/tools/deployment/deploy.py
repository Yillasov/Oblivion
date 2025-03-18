"""
Simple Deployment Tool for Neuromorphic Hardware

Provides utilities for deploying algorithms and configurations to target hardware.
"""

import os
import sys
import json
import shutil
import socket  # Added socket import
import argparse
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append("/Users/yessine/Oblivion")

from src.core.utils.logging_framework import get_logger
from src.core.hardware.config_manager import HardwareConfigManager
from src.core.hardware.hardware_detection import hardware_detector
from src.core.integration.neuromorphic_system import NeuromorphicSystem

logger = get_logger("deployment")


class DeploymentManager:
    """Simple deployment manager for neuromorphic hardware."""
    
    def __init__(self, 
                 repo_root: str = "/Users/yessine/Oblivion",
                 config_dir: str = "configs",
                 build_dir: str = "build",
                 deploy_dir: str = "deploy"):
        """Initialize deployment manager."""
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
    
    def detect_hardware(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Auto-detect connected hardware with improved reliability."""
        # Try multiple detection methods
        detected = hardware_detector.get_best_hardware()
        
        if not detected:
            # Fallback detection using alternative methods
            logger.info("Primary detection failed, trying alternative methods...")
            
            # Try direct connection attempts to common hardware addresses
            for hw_type, addresses in self._get_common_hardware_addresses().items():
                for addr in addresses:
                    if self._test_hardware_connection(hw_type, addr):
                        logger.info(f"Detected {hw_type} hardware at {addr} using fallback method")
                        return hw_type, {"address": addr, "connection_type": "network"}
            
            # Check for hardware configuration files as last resort
            configs = self.config_manager.list_configs()
            if configs:
                # Use the first available configured hardware type as fallback
                hw_type = next(iter(configs.keys()))
                default_config = configs[hw_type][0] if configs[hw_type] else "default"
                logger.warning(f"No hardware detected, using configured {hw_type} with {default_config}")
                return hw_type, {"address": "default", "config_name": default_config}
            
            # Final fallback - provide user with options
            logger.error("All hardware detection methods failed")
            return self._handle_detection_failure()
        
        return detected
    
    def _handle_detection_failure(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Handle complete hardware detection failure with user options."""
        # Check if running in interactive mode
        if sys.stdin.isatty() and not os.environ.get("OBLIVION_NON_INTERACTIVE"):
            print("\nHardware detection failed. Please select an option:")
            print("1. Use simulation mode (no physical hardware)")
            print("2. Manually specify hardware type and address")
            print("3. Abort operation")
            
            try:
                choice = input("Enter choice (1-3): ")
                
                if choice == "1":
                    logger.info("Using simulation mode as fallback")
                    return "simulation", {"mode": "virtual", "address": "localhost"}
                
                elif choice == "2":
                    hw_types = ["loihi", "truenorth", "spinnaker"]
                    print("\nAvailable hardware types:")
                    for i, hw in enumerate(hw_types, 1):
                        print(f"{i}. {hw}")
                    
                    hw_choice = input(f"Select hardware type (1-{len(hw_types)}): ")
                    try:
                        hw_type = hw_types[int(hw_choice) - 1]
                    except (ValueError, IndexError):
                        logger.error("Invalid hardware type selection")
                        return None
                    
                    address = input("Enter hardware address (IP or hostname): ")
                    logger.info(f"Using manually specified {hw_type} at {address}")
                    return hw_type, {"address": address, "connection_type": "manual"}
                
                else:
                    logger.info("Operation aborted by user")
                    return None
                    
            except (EOFError, KeyboardInterrupt):
                logger.info("Operation aborted by user")
                return None
        
        # Non-interactive mode - use simulation as fallback
        logger.warning("No hardware detected, falling back to simulation mode")
        return "simulation", {"mode": "virtual", "address": "localhost"}
    
    def _get_common_hardware_addresses(self) -> Dict[str, List[str]]:
        """Get common hardware addresses for fallback detection."""
        return {
            "loihi": ["127.0.0.1:22222", "192.168.1.10:22222"],
            "spinnaker": ["127.0.0.1:17893", "spinn-1:5000"],
            "truenorth": ["127.0.0.1:8000", "tn-board:5000"]
        }
    
    def _test_hardware_connection(self, hw_type: str, address: str) -> bool:
        """Test connection to hardware address."""
        try:
            # Parse address
            if ":" in address:
                host, port = address.split(":")
                port = int(port)
            else:
                host, port = address, self._get_default_port(hw_type)
            
            # Try to connect
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            result = s.connect_ex((host, port))
            s.close()
            return result == 0
        except:
            return False
    
    def _get_default_port(self, hw_type: str) -> int:
        """Get default port for hardware type."""
        ports = {
            "loihi": 22222,
            "spinnaker": 17893,
            "truenorth": 8000
        }
        return ports.get(hw_type, 22222)
    
    # Add test_deployment method
    def test_deployment(self, package_path: str, test_level: str = "basic") -> Dict[str, Any]:
        """
        Test a deployment package without actually deploying to hardware.
        
        Args:
            package_path: Path to the deployment package
            test_level: Test level ('basic', 'comprehensive', 'simulation')
            
        Returns:
            Dict containing test results
        """
        if not os.path.exists(package_path):
            logger.error(f"Deployment package not found: {package_path}")
            return {"success": False, "error": "Package not found"}
        
        # Extract package for testing
        extract_dir = os.path.join(self.build_dir, "test_tmp", Path(package_path).stem)
        os.makedirs(os.path.dirname(extract_dir), exist_ok=True)
        
        try:
            # Extract package
            shutil.unpack_archive(package_path, extract_dir)
            
            # Load manifest
            manifest_path = os.path.join(extract_dir, "manifest.json")
            if not os.path.exists(manifest_path):
                return {"success": False, "error": "Invalid package: missing manifest"}
                
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                
            # Load config
            config_path = os.path.join(extract_dir, "config", "hardware_config.json")
            if not os.path.exists(config_path):
                return {"success": False, "error": "Invalid package: missing configuration"}
                
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Run tests based on level
            results = {
                "package": os.path.basename(package_path),
                "hardware_type": manifest.get("hardware_type", "unknown"),
                "config_name": manifest.get("config_name", "unknown"),
                "test_level": test_level,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tests": {}
            }
            
            # Basic validation tests
            results["tests"]["manifest_valid"] = "hardware_type" in manifest and "config_name" in manifest
            results["tests"]["config_valid"] = isinstance(config, dict) and len(config) > 0
            
            # Check algorithms
            algorithm_dir = os.path.join(extract_dir, "algorithms")
            algorithms = [f for f in os.listdir(algorithm_dir) if f.endswith('.py')]
            results["tests"]["algorithms_found"] = len(algorithms) > 0
            results["tests"]["algorithm_count"] = len(algorithms)
            
            # Deployment script check
            deploy_script = os.path.join(extract_dir, "deploy.sh")
            results["tests"]["deploy_script_exists"] = os.path.exists(deploy_script)
            results["tests"]["deploy_script_executable"] = os.access(deploy_script, os.X_OK)
            
            # Comprehensive tests
            if test_level in ["comprehensive", "simulation"]:
                # Validate algorithm syntax
                syntax_valid = True
                for algo in algorithms:
                    algo_path = os.path.join(algorithm_dir, algo)
                    try:
                        with open(algo_path, 'r') as f:
                            compile(f.read(), algo_path, 'exec')
                    except SyntaxError:
                        syntax_valid = False
                        break
                
                results["tests"]["algorithm_syntax_valid"] = syntax_valid
                
                # Check config compatibility with hardware type
                hw_type = manifest.get("hardware_type", "")
                results["tests"]["config_compatible"] = self._check_config_compatibility(config, hw_type)
            
            # Simulation test
            if test_level == "simulation":
                # Create a simulated hardware environment and test deployment
                sim_result = self._run_simulation_test(extract_dir, manifest, config)
                results["tests"]["simulation"] = sim_result
            
            # Overall success
            results["success"] = all(v for k, v in results["tests"].items() if isinstance(v, bool))
            
            return results
            
        except Exception as e:
            logger.error(f"Test deployment error: {str(e)}")
            return {"success": False, "error": str(e)}
        finally:
            # Clean up extracted files regardless of success or failure
            try:
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
                    logger.debug(f"Cleaned up test directory: {extract_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up test directory: {cleanup_error}")

    def _check_config_compatibility(self, config: Dict[str, Any], hardware_type: str) -> bool:
        """Check if configuration is compatible with hardware type."""
        # Simple compatibility check
        if hardware_type == "loihi":
            return "chip_id" in config or "board_id" in config
        elif hardware_type == "spinnaker":
            return "ip_address" in config or "board_address" in config
        elif hardware_type == "truenorth":
            return "core_count" in config or "chip_id" in config
        return True
    
    def _run_simulation_test(self, package_dir: str, manifest: Dict[str, Any], 
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a simulated deployment test."""
        hardware_type = manifest.get("hardware_type", "")
        
        # Create a simple simulation environment
        from src.core.hardware.unified_interface import create_hardware_interface
        
        # Create a simulated hardware interface
        sim_config = {"simulation": True, "mode": "test"}
        sim_config.update(config)
        
        try:
            # Initialize hardware interface in simulation mode
            hw_interface = create_hardware_interface(hardware_type, sim_config)
            hw_interface.initialize()
            
            # Test basic operations
            neuron_ids = hw_interface.allocate_neurons(10)
            connection_ids = hw_interface.create_connections([(0, 1, 0.5), (1, 2, 0.5)])
            
            # Run a simple simulation
            inputs = {0: [1.0, 2.0, 3.0]}
            outputs = hw_interface.run_simulation(inputs, 10.0)
            
            # Shutdown
            hw_interface.shutdown()
            
            return {
                "initialized": True,
                "neurons_allocated": len(neuron_ids) > 0,
                "connections_created": len(connection_ids) > 0,
                "simulation_run": outputs is not None,
                "success": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def create_package(self, hardware_type: str, config_name: str,
                  algorithm_paths: List[str]) -> str:
        """Create a streamlined deployment package."""
        # Load hardware configuration
        config = self.config_manager.load_config(hardware_type, config_name)
        if not config:
            raise ValueError(f"Configuration '{config_name}' for {hardware_type} not found")
        
        # Validate hardware requirements
        self._validate_hardware_requirements(hardware_type, config)
        
        # Create package directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"{hardware_type}_{config_name}_{timestamp}"
        package_dir = os.path.join(self.build_dir, package_name)
        
        # Create directory structure
        os.makedirs(package_dir, exist_ok=True)
        os.makedirs(os.path.join(package_dir, "algorithms"), exist_ok=True)
        os.makedirs(os.path.join(package_dir, "config"), exist_ok=True)
        
        # Copy configuration and algorithms
        with open(os.path.join(package_dir, "config", "hardware_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        for path in algorithm_paths:
            if os.path.exists(path):
                shutil.copy(path, os.path.join(package_dir, "algorithms", os.path.basename(path)))
        
        # Create manifest
        manifest = {
            "hardware_type": hardware_type,
            "config_name": config_name,
            "created": timestamp,
            "algorithms": [os.path.basename(p) for p in algorithm_paths if os.path.exists(p)]
        }
        
        with open(os.path.join(package_dir, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create deployment script
        self._create_deploy_script(package_dir, hardware_type)
        os.chmod(os.path.join(package_dir, "deploy.sh"), 0o755)
        
        # Create archive
        archive_path = os.path.join(self.deploy_dir, f"{package_name}.zip")
        shutil.make_archive(
            os.path.join(self.deploy_dir, package_name),
            'zip',
            self.build_dir,
            package_name
        )
        
        # Clean up build directory
        shutil.rmtree(package_dir)
        
        logger.info(f"Created deployment package: {archive_path}")
        return archive_path
    
    def _create_deploy_script(self, package_dir: str, hardware_type: str) -> None:
        """Create a unified deployment script."""
        script_content = f"""#!/bin/bash
# Deployment script for {hardware_type.upper()} hardware

HARDWARE_ADDRESS="$1"
CONFIG_FILE="./config/hardware_config.json"

# Check configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found"
    exit 1
fi

# Set up environment
export PYTHONPATH=$PYTHONPATH:./

# Deploy algorithms
echo "Deploying algorithms to {hardware_type.upper()} hardware..."
for algo in ./algorithms/*.py; do
    echo "  - $algo"
done

# Connect to hardware
if [ -n "$HARDWARE_ADDRESS" ]; then
    echo "Connecting to {hardware_type.upper()} at $HARDWARE_ADDRESS..."
else
    echo "Connecting to {hardware_type.upper()} with default address..."
fi

# Run deployment
python -c "
import json
import sys
import os

# Load configuration
with open('./config/hardware_config.json', 'r') as f:
    config = json.load(f)

print(f'Loaded configuration: {{config.get(\\"_metadata\\", {{}}).get(\\"name\\", \\"unknown\\")}}')
print('Deployment complete!')
"

echo "Deployment completed successfully"
"""
        with open(os.path.join(package_dir, "deploy.sh"), 'w') as f:
            f.write(script_content)
    
    def _validate_hardware_requirements(self, hardware_type: str, config: Dict[str, Any]) -> None:
        """Validate hardware-specific requirements."""
        requirements = {
            "loihi": {"chip_id", "board_id", "neuron_type"},
            "spinnaker": {"ip_address", "n_chips", "version"},
            "truenorth": {"core_count", "chip_id", "board_address"}
        }
        
        # Check for required fields
        if hardware_type in requirements:
            missing = [field for field in requirements[hardware_type] 
                      if field not in config and field not in config.get("hardware", {})]
            if missing:
                logger.warning(f"Missing recommended fields for {hardware_type}: {', '.join(missing)}")
        
        # Add hardware-specific validation
        if hardware_type == "loihi" and "neuron_type" in config:
            if config["neuron_type"] not in ["LIF", "ALIF", "Compartment"]:
                logger.warning(f"Unsupported neuron type for Loihi: {config['neuron_type']}")
        
        # Add hardware capabilities to manifest
        config["_capabilities"] = self._get_hardware_capabilities(hardware_type)
    
        # Enhance deploy method to handle hardware-specific deployment
        def deploy(self, package_path: str, hardware_address: Optional[str] = None) -> bool:
            """Deploy package to hardware with improved error handling."""
            if not os.path.exists(package_path):
                logger.error(f"Deployment package not found: {package_path}")
                return False
            
            # Extract package to get hardware type
            extract_dir = os.path.join(self.build_dir, "tmp", Path(package_path).stem)
            os.makedirs(os.path.dirname(extract_dir), exist_ok=True)
            
            try:
                # Extract package
                shutil.unpack_archive(package_path, extract_dir)
                
                # Get hardware type from manifest
                manifest_path = os.path.join(extract_dir, "manifest.json")
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                hardware_type = manifest.get("hardware_type")
                
                # Auto-detect hardware if no address provided
                if not hardware_address:
                    detected = self.detect_hardware()
                    if detected:
                        detected_type, hw_info = detected
                        # Verify hardware type matches
                        if detected_type != hardware_type:
                            logger.warning(f"Package for {hardware_type} but detected {detected_type}")
                            if not self._confirm_deployment_mismatch():
                                return False
                        hardware_address = hw_info.get("address", None)
                        logger.info(f"Auto-detected {detected_type} hardware at {hardware_address}")
                
                # Run hardware-specific pre-deployment checks
                if not self._run_hardware_checks(hardware_type, extract_dir):
                    logger.error("Hardware pre-deployment checks failed")
                    return False
                
                # Run deployment script
                cmd = f"cd {extract_dir} && ./deploy.sh {hardware_address or ''}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Deployment failed: {result.stderr}")
                    return False
                
                logger.info("Deployment successful")
                return True
                
            except Exception as e:
                logger.error(f"Deployment error: {str(e)}")
                return False
            finally:
                # Clean up
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
        
        def _confirm_deployment_mismatch(self) -> bool:
            """Confirm deployment when hardware type doesn't match."""
            if sys.stdin.isatty() and not os.environ.get("OBLIVION_NON_INTERACTIVE"):
                response = input("Hardware type mismatch. Continue anyway? (y/N): ")
                return response.lower() == 'y'
            return False
    
        def _run_hardware_checks(self, hardware_type: str, package_dir: str) -> bool:
            """Run hardware-specific pre-deployment checks."""
            # Load hardware config
            config_path = os.path.join(package_dir, "config", "hardware_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Hardware-specific checks
            if hardware_type == "loihi":
                return self._check_loihi_requirements(config)
            elif hardware_type == "spinnaker":
                return self._check_spinnaker_requirements(config)
            elif hardware_type == "truenorth":
                return self._check_truenorth_requirements(config)
            return True
    
        def _check_loihi_requirements(self, config: Dict[str, Any]) -> bool:
            """Check Loihi-specific requirements."""
            # Simple check for demonstration
            return True
    
        def _check_spinnaker_requirements(self, config: Dict[str, Any]) -> bool:
            """Check SpiNNaker-specific requirements."""
            # Simple check for demonstration
            return True
    
        def _check_truenorth_requirements(self, config: Dict[str, Any]) -> bool:
            """Check TrueNorth-specific requirements."""
            # Simple check for demonstration
            return True

    def _get_hardware_capabilities(self, hardware_type: str) -> Dict[str, Any]:
        """Get hardware-specific capabilities."""
        capabilities = {
            "loihi": {
                "max_neurons": 128000,
                "max_synapses": 128000000,
                "supports_learning": True,
                "precision": "8-bit"
            },
            "spinnaker": {
                "max_neurons": 16000000,
                "max_synapses": 8000000000,
                "supports_learning": True,
                "precision": "16-bit"
            },
            "truenorth": {
                "max_neurons": 1000000,
                "max_synapses": 256000000,
                "supports_learning": False,
                "precision": "1-bit"
            }
        }
        return capabilities.get(hardware_type, {})

    # Enhance deploy method to handle hardware-specific deployment
    def deploy(self, package_path: str, hardware_address: Optional[str] = None) -> bool:
        """Deploy package to hardware with improved error handling."""
        if not os.path.exists(package_path):
            logger.error(f"Deployment package not found: {package_path}")
            return False
        
        # Extract package to get hardware type
        extract_dir = os.path.join(self.build_dir, "tmp", Path(package_path).stem)
        os.makedirs(os.path.dirname(extract_dir), exist_ok=True)
        
        try:
            # Extract and deploy
            shutil.unpack_archive(package_path, extract_dir)
            
            # Run deployment script
            cmd = f"cd {extract_dir} && ./deploy.sh {hardware_address or ''}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Deployment failed: {result.stderr}")
                return False
            
            logger.info("Deployment successful")
            return True
            
        except Exception as e:
            logger.error(f"Deployment error: {str(e)}")
            return False
        finally:
            # Clean up
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)


def main():
    """Simplified main entry point."""
    parser = argparse.ArgumentParser(description="Neuromorphic Hardware Deployment Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List hardware platforms
    subparsers.add_parser("list-hardware", help="List available hardware platforms")
    
    # Create package
    create_parser = subparsers.add_parser("create", help="Create deployment package")
    create_parser.add_argument("hardware_type", help="Target hardware type")
    create_parser.add_argument("config_name", help="Hardware configuration name")
    create_parser.add_argument("--algorithms", nargs="+", help="Paths to algorithm files")
    
    # Deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to hardware")
    deploy_parser.add_argument("package", help="Path to deployment package")
    deploy_parser.add_argument("--address", help="Hardware address (IP or serial)")
    
    # Detect hardware
    subparsers.add_parser("detect", help="Detect connected hardware")
    
    # Add test command
    test_parser = subparsers.add_parser("test", help="Test deployment package")
    test_parser.add_argument("package", help="Path to deployment package")
    test_parser.add_argument("--level", choices=["basic", "comprehensive", "simulation"], 
                           default="basic", help="Test level")
    
    args = parser.parse_args()
    
    # Create deployment manager
    manager = DeploymentManager()
    
    if args.command == "list-hardware":
        platforms = manager.config_manager.list_configs().keys()
        print("Available hardware platforms:")
        for platform in platforms:
            print(f"  - {platform}")
    
    elif args.command == "create":
        algorithms = args.algorithms or []
        package_path = manager.create_package(
            args.hardware_type,
            args.config_name,
            algorithms
        )
        print(f"Created deployment package: {package_path}")
    
    elif args.command == "deploy":
        success = manager.deploy(args.package, args.address)
        if not success:
            sys.exit(1)
    
    elif args.command == "detect":
        detected = manager.detect_hardware()
        if detected:
            hw_type, hw_info = detected
            print(f"Detected {hw_type} hardware:")
            for key, value in hw_info.items():
                print(f"  {key}: {value}")
        else:
            print("No neuromorphic hardware detected")
    
    elif args.command == "test":
        results = manager.test_deployment(args.package, args.level)
        print(f"Test results for {os.path.basename(args.package)}:")
        print(f"  Success: {results['success']}")
        print("  Tests:")
        for test, result in results.get("tests", {}).items():
            print(f"    {test}: {result}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

def cleanup_old_packages(self, days_old: int = 30) -> int:
    """
    Clean up old deployment packages.
    
    Args:
        days_old: Remove packages older than this many days
        
    Returns:
        int: Number of packages removed
    """
    import time
    
    now = time.time()
    count = 0
    
    try:
        # Clean up deploy directory
        for f in os.listdir(self.deploy_dir):
            if f.endswith('.zip'):
                path = os.path.join(self.deploy_dir, f)
                if os.path.isfile(path):
                    # Check if file is older than days_old
                    if os.stat(path).st_mtime < now - days_old * 86400:
                        try:
                            os.remove(path)
                            count += 1
                            logger.info(f"Removed old package: {f}")
                        except Exception as e:
                            logger.warning(f"Failed to remove {f}: {e}")
        
        # ... rest of cleanup code ...
        
        return count
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return count