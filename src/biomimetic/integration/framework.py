"""
Core integration framework for biomimetic systems.

Connects all biomimetic modules and provides unified interfaces.
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Callable

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.biomimetic.design.principles import BiomimeticDesignFramework, BiomimeticPrinciple
# Fix the import to use the correct class name
from src.core.integration.neuromorphic_system import NeuromorphicInterface
from src.biomimetic.hardware.integration import BiomimeticHardwareIntegration
from src.biomimetic.sensors.integration_framework import BiomimeticSensorInterface
from src.biomimetic.control.cpg_models import BiomimeticCPGController
from src.power.biomimetic_integration import BiomimeticPowerIntegrator
from src.manufacturing.materials.biomimetic_materials import BiomimeticMaterialSelector
from src.biomimetic.integration.validator import BiomimeticValidator
from src.biomimetic.integration.config_manager import BiomimeticConfigManager

logger = get_logger("biomimetic_integration")

class BiomimeticIntegrationFramework:
    """
    Comprehensive integration framework for biomimetic systems.
    
    Connects all biomimetic modules and provides cross-module validation
    and configuration management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the biomimetic integration framework.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Initialize subsystems
        self.design_framework = BiomimeticDesignFramework()
        self.hardware_integration = None
        self.sensor_interface = BiomimeticSensorInterface()
        self.power_integrator = None
        self.material_selector = BiomimeticMaterialSelector()
        
        # Initialize validator and config manager
        self.config_manager = BiomimeticConfigManager(config_path)
        self.validator = BiomimeticValidator(self.design_framework)
        
        # Integration state
        self.initialized = False
        self.active_modules: Set[str] = set()
        self.module_dependencies: Dict[str, List[str]] = {
            "hardware": ["power"],
            "sensors": ["hardware"],
            "control": ["sensors", "hardware"],
            "manufacturing": ["materials"]
        }
        
        # Event callbacks
        self.callbacks: Dict[str, List[Callable]] = {}
        
        logger.info("Biomimetic integration framework created")
    
    def initialize(self, modules: Optional[List[str]] = None) -> bool:
        """
        Initialize the integration framework with specified modules.
        
        Args:
            modules: List of modules to initialize (None for all)
            
        Returns:
            Success status
        """
        try:
            # Load configuration
            config = self.config_manager.load_configuration()
            
            # Determine modules to initialize
            if modules is None:
                modules = ["design", "hardware", "sensors", "power", "materials", "manufacturing"]
            
            # Initialize modules in dependency order
            self._initialize_modules(modules, config)
            
            # Validate cross-module compatibility
            validation_result = self.validator.validate_integration(self)
            if not validation_result["success"]:
                logger.error(f"Integration validation failed: {validation_result['errors']}")
                return False
            
            self.initialized = True
            logger.info(f"Biomimetic integration framework initialized with modules: {', '.join(self.active_modules)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize biomimetic integration framework: {e}")
            return False
    
    def _initialize_modules(self, modules: List[str], config: Dict[str, Any]) -> None:
        """
        Initialize modules in dependency order.
        
        Args:
            modules: List of modules to initialize
            config: Configuration dictionary
        """
        # Initialize hardware if needed
        if "hardware" in modules:
            hardware_config = config.get("hardware", {})
            self.hardware_integration = BiomimeticHardwareIntegration()
            if self.hardware_integration.initialize():
                self.active_modules.add("hardware")
        
        # Initialize power if needed
        if "power" in modules:
            power_config = config.get("power", {})
            biological_reference = power_config.get("biological_reference", "peregrine_falcon")
            self.power_integrator = BiomimeticPowerIntegrator(
                hardware_interface=self.hardware_integration.hardware_interface if self.hardware_integration else None,
                config=power_config,
                biological_reference=biological_reference
            )
            self.active_modules.add("power")
        
        # Initialize sensors if needed
        if "sensors" in modules and "hardware" in self.active_modules:
            # Connect sensors to hardware
            if self.hardware_integration and hasattr(self.hardware_integration, "actuator_controller"):
                # Register actuator groups with sensor interface
                for group_id in self.hardware_integration.actuator_controller.actuator_groups:
                    self._register_sensor_for_actuator_group(group_id)
            
            self.active_modules.add("sensors")
    
    def _register_sensor_for_actuator_group(self, group_id: str) -> None:
        """Register proprioceptive sensors for an actuator group."""
        from src.biomimetic.sensors.integration_framework import ProprioceptiveSensor
        
        # Create proprioceptive sensor for the actuator group
        proprioceptive_sensor = ProprioceptiveSensor(group_id, self.sensor_interface)
        logger.info(f"Registered proprioceptive sensor for actuator group: {group_id}")
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback for a specific event type.
        
        Args:
            event_type: Type of event to register for
            callback: Callback function
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        self.callbacks[event_type].append(callback)
        logger.info(f"Registered callback for event type: {event_type}")
    
    def update(self, dt: float = 0.01) -> Dict[str, Any]:
        """
        Update the integrated system.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            System state
        """
        if not self.initialized:
            logger.error("Cannot update uninitialized system")
            return {}
        
        state = {}
        
        # Update hardware integration
        if self.hardware_integration:
            hardware_state = self.hardware_integration.update(dt)
            state["hardware"] = hardware_state
        
        # Update power integration
        if self.power_integrator:
            self.power_integrator.update(dt)
            state["power"] = {"state": self.power_integrator.current_power_state}
        
        # Update sensor interface
        platform_state = {"time": time.time(), "hardware_state": state.get("hardware", {})}
        environment = {}  # Would be populated from simulation or real sensors
        sensor_data = self.sensor_interface.update(platform_state, environment)
        state["sensors"] = sensor_data
        
        # Trigger callbacks
        self._trigger_callbacks("update", state)
        
        return state
    
    def _trigger_callbacks(self, event_type: str, data: Any) -> None:
        """
        Trigger callbacks for a specific event type.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for event {event_type}: {e}")
    
    def apply_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Apply a new configuration to the system.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Success status
        """
        # Validate configuration
        validation_result = self.validator.validate_configuration(config)
        if not validation_result["success"]:
            logger.error(f"Configuration validation failed: {validation_result['errors']}")
            return False
        
        # Apply configuration to each module
        if "power" in config and self.power_integrator:
            power_config = config["power"]
            if "state" in power_config:
                self.power_integrator.set_power_state(power_config["state"])
            
            if "energy_harvesting" in power_config:
                self.power_integrator.enable_energy_harvesting(power_config["energy_harvesting"])
        
        if "hardware" in config and self.hardware_integration:
            hardware_config = config["hardware"]
            if "wing_flapping" in hardware_config:
                wing_config = hardware_config["wing_flapping"]
                self.hardware_integration.configure_wing_flapping(
                    wing_config.get("frequency", 2.0),
                    wing_config.get("amplitude", 0.5)
                )
        
        # Save configuration
        self.config_manager.save_configuration(config)
        
        logger.info("Applied new configuration")
        return True
    
    def get_active_principles(self) -> List[BiomimeticPrinciple]:
        """
        Get the active biomimetic principles in the current design.
        
        Returns:
            List of active principles
        """
        return self.validator.get_active_principles()