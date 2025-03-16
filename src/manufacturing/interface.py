from typing import Dict, Any, Optional, List, Tuple, Protocol, Callable, Type, ClassVar
import os
import logging
from src.airframe.base import AirframeBase
from src.core.utils.logging_framework import get_logger, handle_error, error_context

# Set up logger using the centralized logging framework
logger = get_logger("manufacturing")

# Define protocols for dependency injection
class ManufacturingPipeline(Protocol):
    """
    Protocol defining the interface for manufacturing pipelines.
    
    This protocol specifies the required methods that any manufacturing
    pipeline implementation must provide to be compatible with the
    manufacturing interface.
    """
    
    def process_design(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a design through the manufacturing pipeline.
        
        Args:
            design_data: Dictionary containing design specifications
            
        Returns:
            Dictionary containing manufacturing instructions and metadata
        """
        ...

class PipelineFactory(Protocol):
    """
    Protocol defining the interface for pipeline factories.
    
    This protocol specifies the required methods that any pipeline factory
    implementation must provide to create manufacturing pipelines.
    """
    
    @staticmethod
    def create_pipeline(airframe_type: str, config: Dict[str, Any]) -> ManufacturingPipeline:
        """
        Create a manufacturing pipeline for the given airframe type.
        
        Args:
            airframe_type: Type of airframe to create a pipeline for
            config: Configuration for the pipeline
            
        Returns:
            A manufacturing pipeline instance
            
        Raises:
            PipelineCreationError: If the pipeline cannot be created
        """
        ...

class DesignToManufacturingInterface:
    """
    Interface for converting airframe designs to manufacturing instructions.
    
    This class provides methods to transform airframe designs into detailed
    manufacturing instructions, validate designs against manufacturing
    constraints, and manage the manufacturing pipeline process.
    
    Attributes:
        config: Configuration dictionary for the interface
        output_dir: Directory where manufacturing output will be stored
        pipeline_factory: Factory for creating manufacturing pipelines
    """
    
    # Class variables with type annotations
    DEFAULT_DIMENSIONS: ClassVar[Dict[str, float]] = {
        "length": 10.0,
        "width": 15.0,
        "height": 3.0
    }
    
    DEFAULT_CAPABILITIES: ClassVar[List[str]] = [
        "3d_printing", "cnc_machining", "composite_layup"
    ]
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 pipeline_factory: Optional[PipelineFactory] = None) -> None:
        """
        Initialize the manufacturing interface.
        
        Args:
            config: Configuration dictionary containing settings for the interface
            pipeline_factory: Factory for creating manufacturing pipelines (optional)
        """
        self.config: Dict[str, Any] = config
        self.output_dir: str = config.get("output_dir", os.path.join(os.getcwd(), "manufacturing_output"))
        
        # Use provided factory or import the default one
        if pipeline_factory is None:
            from .factory import ManufacturingPipelineFactory
            self.pipeline_factory: PipelineFactory = ManufacturingPipelineFactory
        else:
            self.pipeline_factory: PipelineFactory = pipeline_factory
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Created manufacturing output directory: {self.output_dir}")
    
    def convert_airframe_to_manufacturing(self, airframe: AirframeBase) -> Dict[str, Any]:
        """
        Convert an airframe design to manufacturing instructions.
        
        Args:
            airframe: The airframe object to convert
            
        Returns:
            Dictionary containing manufacturing instructions and metadata
            
        Raises:
            ManufacturingValidationError: If the airframe design fails validation
            PipelineCreationError: If the manufacturing pipeline cannot be created
        """
        # Use error context for standardized error handling
        with error_context({"operation": "convert_airframe", "airframe_type": airframe.__class__.__name__}):
            # Validate the airframe design first
            validation_result = self.validate_airframe_design(airframe)
            if not validation_result[0]:
                from .exceptions import ManufacturingValidationError
                raise ManufacturingValidationError(f"Validation failed: {validation_result[1]}")
            
            # Extract airframe type
            airframe_type = airframe.__class__.__name__.lower().replace("drone", "")
            
            # Convert airframe to design data
            design_data = self._airframe_to_design_data(airframe)
            
            # Create manufacturing pipeline
            pipeline_config = {
                "output_dir": self.output_dir,
                "airframe_type": airframe_type
            }
            
            pipeline = self.pipeline_factory.create_pipeline(airframe_type, pipeline_config)
            
            # Process design through pipeline
            logger.info(f"Processing {airframe_type} design through manufacturing pipeline")
            manufacturing_results = pipeline.process_design(design_data)
            
            # Validate manufacturing results
            self._validate_manufacturing_results(manufacturing_results)
            
            return manufacturing_results
    
    def validate_airframe_design(self, airframe: AirframeBase) -> Tuple[bool, Optional[str]]:
        """
        Validate an airframe design before manufacturing.
        
        Args:
            airframe: The airframe object to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for required dimensions
        if not all(key in airframe.config for key in ["length", "wingspan", "height"]):
            return False, "Missing required dimensions in airframe configuration"
        
        # Check for material requirements
        materials = airframe.get_material_requirements()
        if not materials:
            return False, "No material requirements specified"
        
        # Check for control surfaces
        if "control_surfaces" not in airframe.config or not airframe.config["control_surfaces"]:
            return False, "No control surfaces specified"
        
        # Validate dimensions against manufacturing constraints
        dimension_validation = self._validate_dimensions(airframe)
        if not dimension_validation[0]:
            return dimension_validation
        
        # Validate materials against available materials
        material_validation = self._validate_materials(airframe)
        if not material_validation[0]:
            return material_validation
        
        # Validate aerodynamic properties
        aero_validation = self._validate_aerodynamics(airframe)
        if not aero_validation[0]:
            return aero_validation
        
        # Validate structural integrity
        structural_validation = self._validate_structural_integrity(airframe)
        if not structural_validation[0]:
            return structural_validation
        
        return True, None
    
    def _validate_dimensions(self, airframe: AirframeBase) -> Tuple[bool, Optional[str]]:
        """
        Validate airframe dimensions against manufacturing constraints.
        
        Args:
            airframe: The airframe object to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        constraints = self.get_manufacturing_constraints()
        max_dimensions = constraints["max_dimensions"]
        
        # Check length
        if airframe.config.get("length", 0) > max_dimensions.get("length", float('inf')):
            return False, f"Airframe length exceeds manufacturing capability: {airframe.config.get('length')} > {max_dimensions.get('length')}"
        
        # Check wingspan
        if airframe.config.get("wingspan", 0) > max_dimensions.get("width", float('inf')):
            return False, f"Airframe wingspan exceeds manufacturing capability: {airframe.config.get('wingspan')} > {max_dimensions.get('width')}"
        
        # Check height
        if airframe.config.get("height", 0) > max_dimensions.get("height", float('inf')):
            return False, f"Airframe height exceeds manufacturing capability: {airframe.config.get('height')} > {max_dimensions.get('height')}"
        
        # Check minimum dimensions
        if airframe.config.get("length", 0) < 0.1:
            return False, "Airframe length is too small for manufacturing"
        
        if airframe.config.get("wingspan", 0) < 0.1:
            return False, "Airframe wingspan is too small for manufacturing"
        
        if airframe.config.get("height", 0) < 0.05:
            return False, "Airframe height is too small for manufacturing"
        
        return True, None
    
    def _validate_materials(self, airframe: AirframeBase) -> Tuple[bool, Optional[str]]:
        """
        Validate airframe materials against available materials.
        
        Args:
            airframe: The airframe object to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        constraints = self.get_manufacturing_constraints()
        available_materials = constraints["available_materials"]
        
        # If no materials are specified in constraints, assume all materials are available
        if not available_materials:
            return True, None
        
        required_materials = airframe.get_material_requirements()
        
        # Check if all required materials are available
        for material in required_materials:
            if material not in available_materials:
                return False, f"Required material '{material}' is not available for manufacturing"
        
        return True, None
    
    def _validate_aerodynamics(self, airframe: AirframeBase) -> Tuple[bool, Optional[str]]:
        """
        Validate airframe aerodynamic properties.
        
        Args:
            airframe: The airframe object to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for valid speed ranges
        max_speed = airframe.config.get("max_speed", 0)
        cruise_speed = airframe.config.get("cruise_speed", 0)
        
        if max_speed <= 0:
            return False, "Invalid maximum speed specified"
        
        if cruise_speed <= 0:
            return False, "Invalid cruise speed specified"
        
        if cruise_speed > max_speed:
            return False, f"Cruise speed ({cruise_speed}) exceeds maximum speed ({max_speed})"
        
        # Check wing profile
        wing_profile = airframe.config.get("wing_profile", "")
        if not wing_profile:
            return False, "No wing profile specified"
        
        # Additional aerodynamic validation could be added here
        
        return True, None
    
    def _validate_structural_integrity(self, airframe: AirframeBase) -> Tuple[bool, Optional[str]]:
        """
        Validate airframe structural integrity.
        
        Args:
            airframe: The airframe object to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for valid weight
        weight = airframe.config.get("weight", 0)
        if weight <= 0:
            return False, "Invalid airframe weight specified"
        
        # Check weight-to-wingspan ratio for structural feasibility
        wingspan = airframe.config.get("wingspan", 0)
        if wingspan > 0:
            weight_to_wingspan_ratio = weight / wingspan
            if weight_to_wingspan_ratio > 100:  # Example threshold
                return False, f"Weight-to-wingspan ratio too high: {weight_to_wingspan_ratio}"
        
        # Additional structural validation could be added here
        
        return True, None
    
    def _validate_manufacturing_results(self, results: Dict[str, Any]) -> None:
        """
        Validate manufacturing results to ensure they meet requirements.
        
        Args:
            results: Manufacturing results to validate
            
        Raises:
            ManufacturingValidationError: If results fail validation
        """
        required_keys = ["components", "assembly_instructions", "material_usage"]
        
        for key in required_keys:
            if key not in results:
                from .exceptions import ManufacturingValidationError
                raise ManufacturingValidationError(f"Manufacturing results missing required key: {key}")
        
        # Validate components
        if not self._validate_components(results["components"]):
            from .exceptions import ManufacturingValidationError
            raise ManufacturingValidationError("Invalid component specifications in manufacturing results")
        
        # Validate assembly instructions
        if not self._validate_assembly_instructions(results["assembly_instructions"]):
            from .exceptions import ManufacturingValidationError
            raise ManufacturingValidationError("Invalid assembly instructions in manufacturing results")
        
        # Validate material usage
        if not self._validate_material_usage(results["material_usage"]):
            from .exceptions import ManufacturingValidationError
            raise ManufacturingValidationError("Invalid material usage in manufacturing results")
    
    def _validate_components(self, components: Dict[str, Any]) -> bool:
        """
        Validate component specifications in manufacturing results.
        
        Args:
            components: Component specifications to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(components, dict):
            return False
        
        required_component_types = ["wings", "fuselage", "control_surfaces"]
        for component_type in required_component_types:
            if component_type not in components:
                return False
        
        # Additional component validation could be added here
        
        return True
    
    def _validate_assembly_instructions(self, instructions: List[Dict[str, Any]]) -> bool:
        """
        Validate assembly instructions in manufacturing results.
        
        Args:
            instructions: Assembly instructions to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(instructions, list) or not instructions:
            return False
        
        for step in instructions:
            if not isinstance(step, dict):
                return False
            if "step_number" not in step or "description" not in step:
                return False
        
        # Check that step numbers are sequential
        step_numbers = [step["step_number"] for step in instructions]
        if sorted(step_numbers) != list(range(1, len(instructions) + 1)):
            return False
        
        return True
    
    def _validate_material_usage(self, material_usage: Dict[str, Any]) -> bool:
        """
        Validate material usage in manufacturing results.
        
        Args:
            material_usage: Material usage to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(material_usage, dict):
            return False
        
        for material, amount in material_usage.items():
            if not isinstance(material, str) or not material:
                return False
            if not isinstance(amount, (int, float)) or amount <= 0:
                return False
        
        return True
    
    def _airframe_to_design_data(self, airframe: AirframeBase) -> Dict[str, Any]:
        """
        Convert airframe object to design data dictionary.
        
        Args:
            airframe: The airframe object to convert
            
        Returns:
            Dictionary containing design data
        """
        # Extract relevant design information from airframe
        design_data = {
            "airframe_type": airframe.__class__.__name__.lower().replace("drone", ""),
            "dimensions": {
                "length": airframe.config.get("length", 1.0),
                "wingspan": airframe.config.get("wingspan", 1.0),
                "height": airframe.config.get("height", 0.2)
            },
            "materials": airframe.get_material_requirements(),
            "aerodynamics": {
                "max_speed": airframe.config.get("max_speed", 100),
                "cruise_speed": airframe.config.get("cruise_speed", 50)
            },
            "components": {
                "wings": {
                    "count": 2,
                    "profile": airframe.config.get("wing_profile", "naca2412")
                },
                "fuselage": {
                    "shape": airframe.config.get("fuselage_shape", "cylindrical")
                },
                "control_surfaces": airframe.config.get("control_surfaces", ["elevator", "rudder", "ailerons"])
            }
        }
        
        return design_data
    
    def get_manufacturing_constraints(self) -> Dict[str, Dict[str, Any]]:
        """
        Get manufacturing constraints from the current configuration.
        
        Returns:
            Dictionary of manufacturing constraints
        """
        return {
            "max_dimensions": self.config.get("max_dimensions", self.DEFAULT_DIMENSIONS),
            "available_materials": self.config.get("available_materials", []),
            "manufacturing_capabilities": self.config.get("manufacturing_capabilities", 
                                                         self.DEFAULT_CAPABILITIES)
        }
