"""
Base integrator class for standardizing subsystem integration.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import logging

from src.core.integration.neuromorphic_system import NeuromorphicSystem, NeuromorphicInterface
from src.core.integration.system_factory import NeuromorphicSystemFactory

logger = logging.getLogger(__name__)

class BaseIntegrator(ABC):
    """Base class for all subsystem integrators."""
    
    def __init__(self, 
                 subsystem_type: str,
                 hardware_interface: Optional[NeuromorphicInterface] = None,
                 config: Optional[Dict[str, Any]] = None,
                 neuromorphic_system: Optional[NeuromorphicSystem] = None):
        """
        Initialize the base integrator.
        
        Args:
            subsystem_type: Type of subsystem ('propulsion', 'communication', etc.)
            hardware_interface: Optional hardware interface
            config: Optional configuration parameters
            neuromorphic_system: Optional pre-configured neuromorphic system
        """
        self.subsystem_type = subsystem_type
        self.config = config or {}
        
        # Use provided system or create a new one
        if neuromorphic_system:
            self.system = neuromorphic_system
        else:
            self.system = NeuromorphicSystemFactory.create_system(hardware_interface, config)
            NeuromorphicSystemFactory.apply_standard_configuration(self.system, subsystem_type)
        
        self.components: Dict[str, Any] = {}
        self.system_states: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.initialized = False
        
        logger.info(f"Initialized {subsystem_type} integrator")
    
    def initialize(self) -> bool:
        """
        Initialize the integrator and all registered components.
        
        Returns:
            bool: Success status
        """
        try:
            # Initialize the neuromorphic system if not already initialized
            if not self.system.running:
                success = self.system.initialize()
                if not success:
                    logger.error(f"Failed to initialize neuromorphic system for {self.subsystem_type}")
                    return False
            
            # Initialize all components
            for component_id, component in self.components.items():
                if hasattr(component, 'initialize'):
                    success = component.initialize()
                    self.system_states[component_id]["initialized"] = success
                    if not success:
                        logger.warning(f"Failed to initialize component {component_id}")
            
            # Perform subsystem-specific initialization
            success = self._initialize_subsystem()
            
            self.initialized = success
            return success
        
        except Exception as e:
            logger.error(f"Error during {self.subsystem_type} initialization: {str(e)}")
            return False
    
    @abstractmethod
    def _initialize_subsystem(self) -> bool:
        """
        Perform subsystem-specific initialization.
        
        This method should be implemented by each subsystem integrator.
        
        Returns:
            bool: Success status
        """
        pass
    
    def register_component(self, component_id: str, component: Any) -> bool:
        """
        Register a component with the integrator.
        
        Args:
            component_id: Unique identifier for the component
            component: The component to register
            
        Returns:
            bool: Success status
        """
        if component_id in self.components:
            logger.warning(f"Component {component_id} already registered")
            return False
            
        self.components[component_id] = component
        self.system_states[component_id] = {
            "initialized": False,
            "active": False,
            "health": 1.0
        }
        self.performance_history[component_id] = []
        
        logger.info(f"Registered component {component_id} with {self.subsystem_type} integrator")
        return True
    
    def get_system_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current state of all components.
        
        Returns:
            Dict[str, Dict[str, Any]]: Current states
        """
        return self.system_states