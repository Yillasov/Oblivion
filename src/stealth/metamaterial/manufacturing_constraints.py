"""
Manufacturing constraints for metamaterial production.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class MetamaterialManufacturingProcess(Enum):
    """Manufacturing processes for metamaterials."""
    LITHOGRAPHY = "lithography"
    NANO_IMPRINTING = "nano_imprinting"
    ADDITIVE_MANUFACTURING = "additive_manufacturing"
    SELF_ASSEMBLY = "self_assembly"
    ETCHING = "etching"


@dataclass
class MetamaterialManufacturingConstraints:
    """Manufacturing constraints for metamaterials."""
    min_feature_size_nm: float  # Minimum feature size in nanometers
    max_aspect_ratio: float  # Maximum height-to-width ratio
    min_layer_thickness_nm: float  # Minimum layer thickness
    max_layer_count: int  # Maximum number of layers
    temperature_range: Dict[str, float]  # Min/max temperature in Celsius
    compatible_substrates: List[str]  # Compatible substrate materials
    max_area_cm2: float  # Maximum continuous area in cm²
    precision_tolerance_nm: float  # Manufacturing precision tolerance
    process_type: MetamaterialManufacturingProcess  # Manufacturing process


class MetamaterialManufacturingValidator:
    """
    Validator for metamaterial manufacturing constraints.
    Ensures designs can be manufactured with available processes.
    """
    
    def __init__(self):
        """Initialize manufacturing validator with default constraints."""
        self.process_constraints = {
            MetamaterialManufacturingProcess.LITHOGRAPHY: MetamaterialManufacturingConstraints(
                min_feature_size_nm=100.0,
                max_aspect_ratio=10.0,
                min_layer_thickness_nm=50.0,
                max_layer_count=20,
                temperature_range={"min": 15.0, "max": 25.0},
                compatible_substrates=["silicon", "glass", "quartz"],
                max_area_cm2=100.0,
                precision_tolerance_nm=10.0,
                process_type=MetamaterialManufacturingProcess.LITHOGRAPHY
            ),
            MetamaterialManufacturingProcess.NANO_IMPRINTING: MetamaterialManufacturingConstraints(
                min_feature_size_nm=50.0,
                max_aspect_ratio=15.0,
                min_layer_thickness_nm=20.0,
                max_layer_count=10,
                temperature_range={"min": 20.0, "max": 180.0},
                compatible_substrates=["polymer", "silicon", "glass"],
                max_area_cm2=400.0,
                precision_tolerance_nm=5.0,
                process_type=MetamaterialManufacturingProcess.NANO_IMPRINTING
            ),
            MetamaterialManufacturingProcess.ADDITIVE_MANUFACTURING: MetamaterialManufacturingConstraints(
                min_feature_size_nm=1000.0,  # 1 micron
                max_aspect_ratio=20.0,
                min_layer_thickness_nm=500.0,
                max_layer_count=100,
                temperature_range={"min": 15.0, "max": 200.0},
                compatible_substrates=["polymer", "ceramic", "metal"],
                max_area_cm2=1000.0,
                precision_tolerance_nm=100.0,
                process_type=MetamaterialManufacturingProcess.ADDITIVE_MANUFACTURING
            )
        }
    
    def validate_design(self, 
                       design_specs: Dict[str, Any], 
                       process: MetamaterialManufacturingProcess) -> Dict[str, Any]:
        """
        Validate metamaterial design against manufacturing constraints.
        
        Args:
            design_specs: Metamaterial design specifications
            process: Manufacturing process to validate against
            
        Returns:
            Validation results
        """
        if process not in self.process_constraints:
            return {
                "valid": False,
                "errors": [f"Manufacturing process {process.value} not supported"]
            }
            
        constraints = self.process_constraints[process]
        errors = []
        warnings = []
        
        # Validate feature size
        feature_size = design_specs.get("feature_size_nm", 0)
        if feature_size < constraints.min_feature_size_nm:
            errors.append(
                f"Feature size {feature_size}nm is below minimum {constraints.min_feature_size_nm}nm"
            )
        
        # Validate aspect ratio
        aspect_ratio = design_specs.get("aspect_ratio", 0)
        if aspect_ratio > constraints.max_aspect_ratio:
            errors.append(
                f"Aspect ratio {aspect_ratio} exceeds maximum {constraints.max_aspect_ratio}"
            )
        
        # Validate layer thickness
        layer_thickness = design_specs.get("layer_thickness_nm", 0)
        if layer_thickness < constraints.min_layer_thickness_nm:
            errors.append(
                f"Layer thickness {layer_thickness}nm is below minimum {constraints.min_layer_thickness_nm}nm"
            )
        
        # Validate layer count
        layer_count = design_specs.get("layer_count", 0)
        if layer_count > constraints.max_layer_count:
            errors.append(
                f"Layer count {layer_count} exceeds maximum {constraints.max_layer_count}"
            )
        
        # Validate substrate compatibility
        substrate = design_specs.get("substrate", "")
        if substrate not in constraints.compatible_substrates:
            errors.append(
                f"Substrate {substrate} not compatible with {process.value}"
            )
        
        # Validate area
        area = design_specs.get("area_cm2", 0)
        if area > constraints.max_area_cm2:
            errors.append(
                f"Area {area}cm² exceeds maximum {constraints.max_area_cm2}cm²"
            )
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "process": process.value,
            "constraints": {
                "min_feature_size_nm": constraints.min_feature_size_nm,
                "max_aspect_ratio": constraints.max_aspect_ratio,
                "min_layer_thickness_nm": constraints.min_layer_thickness_nm,
                "max_layer_count": constraints.max_layer_count,
                "temperature_range": constraints.temperature_range,
                "compatible_substrates": constraints.compatible_substrates,
                "max_area_cm2": constraints.max_area_cm2,
                "precision_tolerance_nm": constraints.precision_tolerance_nm
            }
        }
    
    def get_manufacturing_requirements(self, 
                                     design_specs: Dict[str, Any], 
                                     process: MetamaterialManufacturingProcess) -> Dict[str, Any]:
        """
        Get manufacturing requirements for a metamaterial design.
        
        Args:
            design_specs: Metamaterial design specifications
            process: Manufacturing process
            
        Returns:
            Manufacturing requirements
        """
        # Validate design first
        validation = self.validate_design(design_specs, process)
        if not validation["valid"]:
            return {
                "error": "Design cannot be manufactured with selected process",
                "validation": validation
            }
        
        # Calculate manufacturing time (simplified model)
        area = design_specs.get("area_cm2", 0)
        layer_count = design_specs.get("layer_count", 0)
        feature_size = design_specs.get("feature_size_nm", 0)
        
        # Base time factors for different processes (hours per cm² per layer)
        time_factors = {
            MetamaterialManufacturingProcess.LITHOGRAPHY: 0.5,
            MetamaterialManufacturingProcess.NANO_IMPRINTING: 0.2,
            MetamaterialManufacturingProcess.ADDITIVE_MANUFACTURING: 0.1
        }
        
        # Feature size factor (smaller features take longer)
        feature_factor = (1000.0 / feature_size) if feature_size > 0 else 1.0
        
        # Calculate estimated production time
        production_time = area * layer_count * time_factors.get(process, 0.5) * feature_factor
        
        return {
            "estimated_production_time_hours": production_time,
            "required_equipment": [process.value],
            "material_requirements": {
                "substrate": design_specs.get("substrate", "silicon"),
                "quantity_cm2": area
            },
            "process_parameters": {
                "temperature_c": self.process_constraints[process].temperature_range["max"] * 0.8,
                "precision_nm": self.process_constraints[process].precision_tolerance_nm
            },
            "quality_checks": [
                "dimensional_verification",
                "electromagnetic_response_testing",
                "surface_uniformity_inspection"
            ]
        }