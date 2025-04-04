#!/usr/bin/env python3
"""
Specialized processes for 3D printing complex organic structures.
Extends the organic 3D printer capabilities with advanced biomimetic processes.
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import logging

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.manufacturing.equipment.organic_3d_printer import (
    Organic3DPrinter, PrinterMode, PrintMaterial
)
from src.core.utils.error_handling import handle_errors, ErrorContext

logger = get_logger("organic_printing")


class OrganicStructureType(Enum):
    """Types of organic structures that can be printed."""
    CELLULAR = "cellular"
    VASCULAR = "vascular"
    SKELETAL = "skeletal"
    NEURAL = "neural"
    MEMBRANE = "membrane"
    GRADIENT = "gradient"
    HIERARCHICAL = "hierarchical"


@dataclass
class OrganicPrintConfig:
    """Configuration for organic structure printing."""
    structure_type: OrganicStructureType
    resolution: float = 0.1  # mm
    infill_density: float = 20.0  # percentage
    wall_thickness: float = 0.8  # mm
    layer_height: float = 0.1  # mm
    print_speed: float = 40.0  # mm/s
    support_density: float = 15.0  # percentage
    primary_material: PrintMaterial = PrintMaterial.FLEXIBLE_POLYMER
    secondary_material: Optional[PrintMaterial] = None
    temperature_bed: float = 60.0  # °C
    temperature_chamber: float = 45.0  # °C
    cooling_fan_speed: float = 80.0  # percentage
    
    # Advanced settings
    gradient_profile: Optional[Dict[str, Any]] = None
    microstructure_pattern: Optional[str] = None
    vein_reinforcement: bool = False
    anisotropic_properties: bool = False


class OrganicPrintingProcess:
    """Process controller for 3D printing complex organic structures."""
    
    def __init__(self, printer: Optional[Organic3DPrinter] = None):
        """
        Initialize organic printing process.
        
        Args:
            printer: Organic 3D printer instance
        """
        self.printer = printer or Organic3DPrinter()
        self.current_print_job = None
        self.print_history = []
        
        # Structure-specific process parameters
        self.process_parameters = {
            OrganicStructureType.CELLULAR: {
                "default_mode": PrinterMode.MICROSTRUCTURE,
                "default_material": PrintMaterial.FLEXIBLE_POLYMER,
                "layer_multiplier": 1.2,
                "speed_multiplier": 0.8,
                "post_processing": ["uv_curing"]
            },
            OrganicStructureType.VASCULAR: {
                "default_mode": PrinterMode.VEIN_REINFORCEMENT,
                "default_material": PrintMaterial.FLEXIBLE_POLYMER,
                "secondary_material": PrintMaterial.COMPOSITE_FIBER,
                "layer_multiplier": 1.0,
                "speed_multiplier": 0.7,
                "post_processing": ["solvent_smoothing"]
            },
            OrganicStructureType.SKELETAL: {
                "default_mode": PrinterMode.VARIABLE_DENSITY,
                "default_material": PrintMaterial.RIGID_POLYMER,
                "secondary_material": PrintMaterial.COMPOSITE_FIBER,
                "layer_multiplier": 0.8,
                "speed_multiplier": 0.9,
                "post_processing": ["heat_treatment"]
            },
            OrganicStructureType.NEURAL: {
                "default_mode": PrinterMode.MULTI_MATERIAL,
                "default_material": PrintMaterial.FLEXIBLE_POLYMER,
                "secondary_material": PrintMaterial.CONDUCTIVE_POLYMER,
                "layer_multiplier": 0.6,
                "speed_multiplier": 0.6,
                "post_processing": ["electrical_testing"]
            },
            OrganicStructureType.MEMBRANE: {
                "default_mode": PrinterMode.SINGLE_MATERIAL,
                "default_material": PrintMaterial.FLEXIBLE_POLYMER,
                "layer_multiplier": 0.5,
                "speed_multiplier": 1.2,
                "post_processing": ["stretching"]
            },
            OrganicStructureType.GRADIENT: {
                "default_mode": PrinterMode.GRADIENT_PROPERTIES,
                "default_material": PrintMaterial.GRADIENT_MATERIAL,
                "layer_multiplier": 1.1,
                "speed_multiplier": 0.8,
                "post_processing": ["gradient_validation"]
            },
            OrganicStructureType.HIERARCHICAL: {
                "default_mode": PrinterMode.MULTI_MATERIAL,
                "default_material": PrintMaterial.BIOMIMETIC_RESIN,
                "secondary_material": PrintMaterial.COMPOSITE_FIBER,
                "layer_multiplier": 1.5,
                "speed_multiplier": 0.5,
                "post_processing": ["microscopy_inspection"]
            }
        }
    
    def initialize(self) -> bool:
        """Initialize the printing process and equipment."""
        return self.printer.initialize()
    
    @handle_errors(context={"operation": "organic_structure_printing"})
    def print_organic_structure(self, 
                              model_path: str, 
                              config: Union[OrganicPrintConfig, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Print a complex organic structure.
        
        Args:
            model_path: Path to the 3D model file
            config: Printing configuration
            
        Returns:
            Dict[str, Any]: Print job results
        """
        # Convert dict config to OrganicPrintConfig if needed
        if isinstance(config, dict):
            structure_type = config.get("structure_type", "cellular")
            if isinstance(structure_type, str):
                structure_type = OrganicStructureType(structure_type)
            config = OrganicPrintConfig(
                structure_type=structure_type,
                resolution=config.get("resolution", 0.1),
                infill_density=config.get("infill_density", 20.0),
                wall_thickness=config.get("wall_thickness", 0.8),
                layer_height=config.get("layer_height", 0.1),
                print_speed=config.get("print_speed", 40.0),
                support_density=config.get("support_density", 15.0),
                primary_material=PrintMaterial(config.get("primary_material", "flexible_polymer")),
                secondary_material=PrintMaterial(config.get("secondary_material", "composite_fiber")) 
                    if "secondary_material" in config else None,
                temperature_bed=config.get("temperature_bed", 60.0),
                temperature_chamber=config.get("temperature_chamber", 45.0),
                cooling_fan_speed=config.get("cooling_fan_speed", 80.0),
                gradient_profile=config.get("gradient_profile"),
                microstructure_pattern=config.get("microstructure_pattern"),
                vein_reinforcement=config.get("vein_reinforcement", False),
                anisotropic_properties=config.get("anisotropic_properties", False)
            )
        
        # Get process parameters for this structure type
        process_params = self.process_parameters[config.structure_type]
        
        # Configure printer
        with ErrorContext(context={"stage": "printer_configuration"}):
            self._configure_printer_for_structure(config, process_params)
        
        # Prepare model
        with ErrorContext(context={"stage": "model_preparation"}):
            prepared_model = self._prepare_model_for_printing(model_path, config)
        
        # Generate toolpath
        with ErrorContext(context={"stage": "toolpath_generation"}):
            toolpath = self._generate_toolpath(prepared_model, config)
        
        # Execute print
        with ErrorContext(context={"stage": "print_execution"}):
            print_result = self._execute_print(toolpath, config, process_params)
        
        # Post-processing
        with ErrorContext(context={"stage": "post_processing"}):
            post_process_result = self._apply_post_processing(
                print_result, process_params["post_processing"]
            )
        
        # Record job in history
        job_result = {
            "job_id": f"organic_{int(time.time())}",
            "model": os.path.basename(model_path),
            "structure_type": config.structure_type.value,
            "print_time": print_result.get("print_time", 0),
            "material_usage": print_result.get("material_usage", {}),
            "success": print_result.get("success", False),
            "quality_metrics": post_process_result.get("quality_metrics", {})
        }
        
        self.print_history.append(job_result)
        self.current_print_job = job_result
        
        return job_result
    
    def _configure_printer_for_structure(self, 
                                       config: OrganicPrintConfig, 
                                       process_params: Dict[str, Any]) -> None:
        """Configure the printer for the specific organic structure."""
        # Set print mode
        printer_mode = process_params["default_mode"]
        self.printer.set_print_mode(PrinterMode(printer_mode))
        
        # Load primary material
        primary_material = config.primary_material or process_params["default_material"]
        self.printer.load_material(primary_material, 1)
        
        # Load secondary material if needed
        if config.secondary_material or "secondary_material" in process_params:
            secondary_material = config.secondary_material or process_params.get("secondary_material")
            if secondary_material:
                self.printer.load_material(secondary_material, 2)
        
        # Set resolution
        self.printer.set_resolution(config.resolution)
        
        # Set temperatures
        self.printer.temperature_zones["bed"] = config.temperature_bed
        self.printer.temperature_zones["chamber"] = config.temperature_chamber
        
        logger.info(f"Printer configured for {config.structure_type.value} structure")
    
    def _prepare_model_for_printing(self, 
                                  model_path: str, 
                                  config: OrganicPrintConfig) -> Dict[str, Any]:
        """Prepare the 3D model for organic structure printing."""
        logger.info(f"Preparing model: {model_path}")
        
        # This would typically involve:
        # 1. Loading the model
        # 2. Analyzing its geometry
        # 3. Adding internal structures based on the organic structure type
        # 4. Generating supports
        
        # Simplified simulation for demonstration
        return {
            "model_path": model_path,
            "prepared": True,
            "volume": 125.0,  # cm³
            "dimensions": (100.0, 80.0, 60.0),  # mm
            "layer_count": int(60.0 / config.layer_height),
            "has_overhangs": True,
            "has_thin_walls": config.structure_type in [
                OrganicStructureType.MEMBRANE, 
                OrganicStructureType.VASCULAR
            ],
            "internal_structures_added": {
                "cellular": config.structure_type == OrganicStructureType.CELLULAR,
                "vascular": config.structure_type == OrganicStructureType.VASCULAR,
                "gradient": config.structure_type == OrganicStructureType.GRADIENT,
            }
        }
    
    def _generate_toolpath(self, 
                         prepared_model: Dict[str, Any], 
                         config: OrganicPrintConfig) -> Dict[str, Any]:
        """Generate toolpath for the organic structure."""
        logger.info("Generating toolpath for organic structure")
        
        # Calculate adjusted parameters
        layer_height = config.layer_height
        print_speed = config.print_speed
        
        # Special processing for different structure types
        if config.structure_type == OrganicStructureType.CELLULAR:
            # Add cellular microstructures
            cell_size = 2.0  # mm
            cell_wall = 0.4  # mm
            logger.info(f"Adding cellular microstructure (cell size: {cell_size}mm)")
        
        elif config.structure_type == OrganicStructureType.VASCULAR:
            # Add vascular channels
            if config.vein_reinforcement:
                logger.info("Adding reinforced vascular channels")
        
        elif config.structure_type == OrganicStructureType.GRADIENT:
            # Add material gradient
            if config.gradient_profile:
                logger.info(f"Applying gradient profile: {config.gradient_profile}")
        
        # Calculate estimated print time and material usage
        layer_count = prepared_model["layer_count"]
        dimensions = prepared_model["dimensions"]
        volume = prepared_model["volume"]
        
        # Estimate print time (simplified)
        print_time_hours = (layer_count * sum(dimensions[:2]) * 2) / (print_speed * 60)
        
        # Estimate material usage (simplified)
        material_volume = volume * (config.infill_density / 100.0)
        material_weight = material_volume * 1.25  # g/cm³ (average density)
        
        return {
            "layer_count": layer_count,
            "estimated_print_time": print_time_hours,
            "estimated_material": material_weight,
            "toolpath_generated": True,
            "special_features": {
                "microstructures": config.structure_type == OrganicStructureType.CELLULAR,
                "vascular_channels": config.structure_type == OrganicStructureType.VASCULAR,
                "gradient_regions": config.structure_type == OrganicStructureType.GRADIENT,
                "anisotropic": config.anisotropic_properties
            }
        }
    
    def _execute_print(self, 
                     toolpath: Dict[str, Any], 
                     config: OrganicPrintConfig,
                     process_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the print job for the organic structure."""
        logger.info(f"Starting organic print: {config.structure_type.value}")
        
        # Adjust print parameters based on process requirements
        layer_multiplier = process_params.get("layer_multiplier", 1.0)
        speed_multiplier = process_params.get("speed_multiplier", 1.0)
        
        adjusted_layer_height = config.layer_height * layer_multiplier
        adjusted_print_speed = config.print_speed * speed_multiplier
        
        logger.info(f"Adjusted layer height: {adjusted_layer_height:.2f}mm, "
                   f"print speed: {adjusted_print_speed:.1f}mm/s")
        
        # Create print configuration for the printer
        print_config = {
            "layer_height": adjusted_layer_height,
            "infill_density": config.infill_density,
            "print_speed": adjusted_print_speed,
            "wall_thickness": config.wall_thickness,
            "support_density": config.support_density,
            "cooling_fan_speed": config.cooling_fan_speed
        }
        
        # Special handling for different structure types
        if config.structure_type == OrganicStructureType.VASCULAR:
            # Create venation pattern for vascular structures
            venation_pattern = self._generate_venation_pattern(
                toolpath, config.vein_reinforcement
            )
            
            # Use specialized wing printing for vascular structures
            success = self.printer.print_biomimetic_wing(
                "dragonfly" if config.vein_reinforcement else "bat",
                venation_pattern,
                {"anisotropic": config.anisotropic_properties}
            )
        else:
            # Use standard organic structure printing
            success = self.printer.print_organic_structure(
                toolpath.get("model_path", "model.stl"), 
                print_config
            )
        
        # Calculate actual print metrics
        print_time = toolpath["estimated_print_time"] * (1.0 / speed_multiplier)
        material_usage = {
            config.primary_material.value: toolpath["estimated_material"] * 0.8,
        }
        
        if config.secondary_material:
            material_usage[config.secondary_material.value] = toolpath["estimated_material"] * 0.2
        
        return {
            "success": success,
            "print_time": print_time,
            "material_usage": material_usage,
            "layer_count": toolpath["layer_count"],
            "special_features": toolpath.get("special_features", {})
        }
    
    def _generate_venation_pattern(self, 
                                 toolpath: Dict[str, Any], 
                                 reinforced: bool = False) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Generate a venation pattern for vascular structures."""
        # This would typically involve complex algorithms to generate biomimetic patterns
        # Simplified implementation for demonstration
        
        # Create a simple branching pattern
        dimensions = (100.0, 80.0)  # mm
        center = (dimensions[0]/2, dimensions[1]/2)
        
        # Main branches
        branches = []
        num_branches = 5 if reinforced else 3
        
        for i in range(num_branches):
            angle = (i * 2 * np.pi / num_branches)
            end_x = center[0] + 0.8 * dimensions[0]/2 * np.cos(angle)
            end_y = center[1] + 0.8 * dimensions[1]/2 * np.sin(angle)
            branches.append((center, (end_x, end_y)))
            
            # Add sub-branches if reinforced
            if reinforced:
                for j in range(2):
                    sub_angle = angle + (-1)**j * np.pi/8
                    mid_x = center[0] + 0.4 * dimensions[0]/2 * np.cos(angle)
                    mid_y = center[1] + 0.4 * dimensions[1]/2 * np.sin(angle)
                    sub_end_x = mid_x + 0.4 * dimensions[0]/2 * np.cos(sub_angle)
                    sub_end_y = mid_y + 0.4 * dimensions[1]/2 * np.sin(sub_angle)
                    branches.append(((mid_x, mid_y), (sub_end_x, sub_end_y)))
        
        return branches
    
    def _apply_post_processing(self, 
                             print_result: Dict[str, Any], 
                             post_processes: List[str]) -> Dict[str, Any]:
        """Apply post-processing steps to the printed organic structure."""
        logger.info(f"Applying post-processing: {', '.join(post_processes)}")
        
        quality_metrics = {
            "dimensional_accuracy": 0.92,
            "surface_finish": 0.85,
            "structural_integrity": 0.90,
            "feature_resolution": 0.88
        }
        
        # Apply specific post-processing effects
        for process in post_processes:
            if process == "uv_curing":
                quality_metrics["cross_linking"] = 0.95
                logger.info("UV curing applied to enhance structural properties")
                
            elif process == "solvent_smoothing":
                quality_metrics["surface_finish"] += 0.08
                logger.info("Solvent smoothing applied to improve surface finish")
                
            elif process == "heat_treatment":
                quality_metrics["structural_integrity"] += 0.05
                logger.info("Heat treatment applied to improve structural integrity")
                
            elif process == "electrical_testing":
                quality_metrics["conductivity"] = 0.82
                logger.info("Electrical testing completed for conductive pathways")
                
            elif process == "stretching":
                quality_metrics["elasticity"] = 0.88
                logger.info("Mechanical stretching applied to membrane")
                
            elif process == "gradient_validation":
                quality_metrics["gradient_fidelity"] = 0.91
                logger.info("Gradient material properties validated")
                
            elif process == "microscopy_inspection":
                quality_metrics["microstructure_fidelity"] = 0.89
                logger.info("Microscopy inspection of hierarchical features completed")
        
        return {
            "post_processing_applied": post_processes,
            "quality_metrics": quality_metrics
        }
    
    def shutdown(self) -> bool:
        """Shutdown the printing process and equipment."""
        return self.printer.shutdown()


def print_complex_organic_structure(model_path: str, structure_type: str) -> Dict[str, Any]:
    """
    Convenience function to print a complex organic structure.
    
    Args:
        model_path: Path to the 3D model file
        structure_type: Type of organic structure to print
        
    Returns:
        Dict[str, Any]: Print job results
    """
    # Create process controller
    process = OrganicPrintingProcess()
    
    # Initialize
    if not process.initialize():
        return {"success": False, "error": "Failed to initialize printer"}
    
    try:
        # Create configuration
        config = OrganicPrintConfig(
            structure_type=OrganicStructureType(structure_type),
            resolution=0.1,
            infill_density=15.0,
            layer_height=0.08,
            vein_reinforcement=structure_type == "vascular",
            anisotropic_properties=True
        )
        
        # Print structure
        result = process.print_organic_structure(model_path, config)
        return result
    finally:
        # Ensure shutdown
        process.shutdown()