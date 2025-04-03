#!/usr/bin/env python3
"""
Parametric design system for stealth features.

This module provides tools for CAD integration and parametric design
of stealth features for various platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np
import math

from src.stealth.base.interfaces import StealthType, StealthSpecs
from src.stealth.base.config import (
    StealthSystemConfig, 
    StealthMaterialConfig,
    StealthSignatureConfig
)


class DesignOptimizationGoal(Enum):
    """Optimization goals for stealth design."""
    RADAR_MINIMIZATION = 0
    INFRARED_REDUCTION = 1
    ACOUSTIC_DAMPENING = 2
    ELECTROMAGNETIC_SHIELDING = 3
    MULTI_SPECTRUM = 4
    WEIGHT_REDUCTION = 5
    COST_EFFICIENCY = 6


class StealthFeatureType(Enum):
    """Types of stealth features that can be designed."""
    EDGE_TREATMENT = 0
    SURFACE_CONTOUR = 1
    MATERIAL_LAYER = 2
    INTAKE_DESIGN = 3
    EXHAUST_DESIGN = 4
    PANEL_ALIGNMENT = 5
    COATING_PATTERN = 6


class ParametricStealthDesigner:
    """Parametric design system for stealth features."""
    
    def __init__(self):
        """Initialize the parametric stealth designer."""
        self.design_library = self._initialize_design_library()
        self.material_properties = self._initialize_material_properties()
        self.manufacturing_constraints = self._initialize_manufacturing_constraints()
        self.current_design: Dict[str, Any] = {}
        
    def _initialize_manufacturing_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Initialize manufacturing constraints for stealth materials."""
        return {
            "material_constraints": {
                "carbon_composite": {
                    "min_thickness_mm": 1.0,
                    "max_thickness_mm": 15.0,
                    "min_curvature_radius_mm": 50.0,
                    "max_temperature_c": 180.0,
                    "curing_time_hours": 4.0
                },
                "ceramic_matrix": {
                    "min_thickness_mm": 2.0,
                    "max_thickness_mm": 20.0,
                    "min_curvature_radius_mm": 100.0,
                    "max_temperature_c": 1200.0,
                    "curing_time_hours": 8.0
                },
                "metamaterial": {
                    "min_thickness_mm": 0.5,
                    "max_thickness_mm": 10.0,
                    "min_curvature_radius_mm": 25.0,
                    "max_temperature_c": 120.0,
                    "curing_time_hours": 6.0
                }
            },
            "feature_constraints": {
                "edge_treatments": {
                    "min_feature_size_mm": 2.0,
                    "max_aspect_ratio": 5.0,
                    "min_spacing_mm": 5.0
                },
                "surface_contours": {
                    "min_feature_size_mm": 10.0,
                    "max_slope_angle_degrees": 60.0,
                    "min_radius_mm": 20.0
                },
                "coating_patterns": {
                    "min_cell_size_mm": 5.0,
                    "max_layer_count": 5,
                    "min_layer_thickness_mm": 0.2
                }
            },
            "process_constraints": {
                "max_production_rate_m2_per_day": 10.0,
                "max_continuous_area_m2": 5.0,
                "max_weight_per_piece_kg": 50.0,
                "available_equipment": ["autoclave", "spray_booth", "cnc_mill", "laser_cutter"]
            }
        }
        
    def _initialize_design_library(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the library of stealth design patterns."""
        return {
            "edge_treatments": {
                "sawtooth": {
                    "rcs_reduction": 0.35,
                    "weight_impact": 0.05,
                    "cost_factor": 1.2,
                    "parameters": {
                        "depth_mm": 5.0,
                        "spacing_mm": 10.0,
                        "angle_degrees": 60.0
                    }
                },
                "serpentine": {
                    "rcs_reduction": 0.42,
                    "weight_impact": 0.08,
                    "cost_factor": 1.5,
                    "parameters": {
                        "depth_mm": 8.0,
                        "spacing_mm": 15.0,
                        "curve_radius_mm": 3.0
                    }
                }
            },
            "surface_contours": {
                "faceted": {
                    "rcs_reduction": 0.30,
                    "weight_impact": 0.10,
                    "cost_factor": 1.3,
                    "parameters": {
                        "facet_size_mm": 200.0,
                        "angle_variance_degrees": 15.0
                    }
                },
                "curved": {
                    "rcs_reduction": 0.25,
                    "weight_impact": 0.05,
                    "cost_factor": 1.1,
                    "parameters": {
                        "curvature_radius_m": 2.0,
                        "smoothness_factor": 0.8
                    }
                }
            },
            "coating_patterns": {
                "checkerboard": {
                    "rcs_reduction": 0.20,
                    "ir_reduction": 0.15,
                    "weight_impact": 0.02,
                    "cost_factor": 1.1,
                    "parameters": {
                        "cell_size_mm": 50.0,
                        "thickness_mm": 2.0
                    }
                },
                "gradient": {
                    "rcs_reduction": 0.25,
                    "ir_reduction": 0.20,
                    "weight_impact": 0.03,
                    "cost_factor": 1.2,
                    "parameters": {
                        "layers": 3,
                        "thickness_mm": 3.0
                    }
                }
            }
        }
        
    def _initialize_material_properties(self) -> Dict[str, Dict[str, Any]]:
        """Initialize material properties for stealth design."""
        return {
            "carbon_composite": {
                "density": 1600.0,  # kg/m³
                "radar_absorption": 0.7,
                "ir_emission": 0.8,
                "thermal_conductivity": 5.0,
                "cost_per_kg": 80.0
            },
            "ceramic_matrix": {
                "density": 2200.0,  # kg/m³
                "radar_absorption": 0.5,
                "ir_emission": 0.6,
                "thermal_conductivity": 2.0,
                "cost_per_kg": 120.0
            },
            "metamaterial": {
                "density": 1200.0,  # kg/m³
                "radar_absorption": 0.9,
                "ir_emission": 0.7,
                "thermal_conductivity": 3.0,
                "cost_per_kg": 200.0
            }
        }
        
    def create_new_design(self, 
                        platform_type: str,
                        dimensions: Dict[str, float],
                        primary_goal: DesignOptimizationGoal) -> Dict[str, Any]:
        """
        Create a new stealth design for a platform.
        
        Args:
            platform_type: Type of platform (aircraft, ship, vehicle, etc.)
            dimensions: Key dimensions of the platform
            primary_goal: Primary optimization goal
            
        Returns:
            Initial design parameters
        """
        # Create basic design
        self.current_design = {
            "platform_type": platform_type,
            "dimensions": dimensions,
            "primary_goal": primary_goal,
            "features": {},
            "materials": {},
            "performance_estimates": {
                "rcs_reduction": 0.0,
                "ir_reduction": 0.0,
                "acoustic_reduction": 0.0,
                "em_reduction": 0.0
            },
            "weight_impact_kg": 0.0,
            "cost_estimate": 0.0
        }
        
        # Apply initial recommendations based on platform type and goal
        self._apply_initial_recommendations()
        
        return self.current_design
        
    def _apply_initial_recommendations(self) -> None:
        """Apply initial design recommendations based on platform and goal."""
        platform = self.current_design["platform_type"]
        goal = self.current_design["primary_goal"]
        
        # Select appropriate features based on platform and goal
        if platform == "aircraft":
            if goal == DesignOptimizationGoal.RADAR_MINIMIZATION:
                self.add_feature(StealthFeatureType.EDGE_TREATMENT, "sawtooth")
                self.add_feature(StealthFeatureType.SURFACE_CONTOUR, "faceted")
            elif goal == DesignOptimizationGoal.INFRARED_REDUCTION:
                self.add_feature(StealthFeatureType.COATING_PATTERN, "gradient")
                
        elif platform == "ship":
            if goal == DesignOptimizationGoal.RADAR_MINIMIZATION:
                self.add_feature(StealthFeatureType.SURFACE_CONTOUR, "faceted")
            elif goal == DesignOptimizationGoal.ACOUSTIC_DAMPENING:
                self.add_feature(StealthFeatureType.COATING_PATTERN, "checkerboard")
                
        # Add default material based on goal
        if goal == DesignOptimizationGoal.RADAR_MINIMIZATION:
            self.add_material("metamaterial", 0.6)  # 60% coverage
        elif goal == DesignOptimizationGoal.INFRARED_REDUCTION:
            self.add_material("ceramic_matrix", 0.4)  # 40% coverage
        else:
            self.add_material("carbon_composite", 0.5)  # 50% coverage
            
    def add_feature(self, 
                  feature_type: StealthFeatureType, 
                  feature_name: str,
                  custom_parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a stealth feature to the design.
        
        Args:
            feature_type: Type of stealth feature
            feature_name: Name of the feature from the library
            custom_parameters: Optional custom parameters
            
        Returns:
            Success status
        """
        if not self.current_design:
            return False
            
        # Get feature category
        category = None
        if feature_type == StealthFeatureType.EDGE_TREATMENT:
            category = "edge_treatments"
        elif feature_type == StealthFeatureType.SURFACE_CONTOUR:
            category = "surface_contours"
        elif feature_type == StealthFeatureType.COATING_PATTERN:
            category = "coating_patterns"
        else:
            return False
            
        # Check if feature exists in library
        if category not in self.design_library or feature_name not in self.design_library[category]:
            return False
            
        # Get feature from library
        feature = self.design_library[category][feature_name].copy()
        
        # Apply custom parameters if provided
        if custom_parameters and "parameters" in feature:
            for param, value in custom_parameters.items():
                if param in feature["parameters"]:
                    feature["parameters"][param] = value
                    
        # Add feature to design
        if "features" not in self.current_design:
            self.current_design["features"] = {}
            
        feature_id = f"{feature_type.name.lower()}_{feature_name}"
        self.current_design["features"][feature_id] = feature
        
        # Update performance estimates
        self._update_performance_estimates()
        
        return True
        
    def add_material(self, material_name: str, coverage_percentage: float) -> bool:
        """
        Add a material to the design.
        
        Args:
            material_name: Name of the material
            coverage_percentage: Percentage of coverage (0.0-1.0)
            
        Returns:
            Success status
        """
        if not self.current_design or material_name not in self.material_properties:
            return False
            
        # Add material to design
        if "materials" not in self.current_design:
            self.current_design["materials"] = {}
            
        self.current_design["materials"][material_name] = {
            "properties": self.material_properties[material_name],
            "coverage": coverage_percentage
        }
        
        # Update performance estimates
        self._update_performance_estimates()
        
        return True
        
    def _update_performance_estimates(self) -> None:
        """Update performance estimates based on current design."""
        design = self.current_design
        
        # Reset estimates
        performance = {
            "rcs_reduction": 0.0,
            "ir_reduction": 0.0,
            "acoustic_reduction": 0.0,
            "em_reduction": 0.0
        }
        
        weight_impact = 0.0
        cost_estimate = 0.0
        
        # Calculate feature contributions
        for feature_id, feature in design.get("features", {}).items():
            if "rcs_reduction" in feature:
                performance["rcs_reduction"] += feature["rcs_reduction"]
            if "ir_reduction" in feature:
                performance["ir_reduction"] += feature["ir_reduction"]
                
            # Add weight impact
            if "weight_impact" in feature:
                weight_impact += feature["weight_impact"]
                
            # Add cost factor
            if "cost_factor" in feature:
                cost_estimate += feature["cost_factor"]
                
        # Calculate material contributions
        for material_name, material_data in design.get("materials", {}).items():
            properties = material_data["properties"]
            coverage = material_data["coverage"]
            
            # Add radar absorption
            if "radar_absorption" in properties:
                performance["rcs_reduction"] += properties["radar_absorption"] * coverage * 0.3
                
            # Add IR emission reduction
            if "ir_emission" in properties:
                performance["ir_reduction"] += (1 - properties["ir_emission"]) * coverage * 0.3
                
            # Calculate weight impact
            if "density" in properties:
                # Simplified calculation based on platform dimensions
                surface_area = self._estimate_surface_area(design["dimensions"])
                thickness = 0.005  # Assume 5mm thickness
                volume = surface_area * thickness * coverage
                material_weight = volume * properties["density"]
                weight_impact += material_weight
                
            # Calculate cost
            if "cost_per_kg" in properties:
                material_cost = weight_impact * properties["cost_per_kg"]
                cost_estimate += material_cost
                
        # Apply diminishing returns and caps
        for key in performance:
            performance[key] = min(0.95, 1 - math.exp(-performance[key]))
            
        # Update design with new estimates
        design["performance_estimates"] = performance
        design["weight_impact_kg"] = weight_impact
        design["cost_estimate"] = cost_estimate
        
    def _estimate_surface_area(self, dimensions: Dict[str, float]) -> float:
        """Estimate surface area based on platform dimensions."""
        if "length" in dimensions and "width" in dimensions:
            if "height" in dimensions:
                # Simplified box model
                length = dimensions["length"]
                width = dimensions["width"]
                height = dimensions["height"]
                return 2 * (length * width + length * height + width * height)
            else:
                # Simplified flat model
                return dimensions["length"] * dimensions["width"]
        return 10.0  # Default value if dimensions are not provided
        
    def export_to_cad(self, format_type: str = "json") -> Dict[str, Any]:
        """
        Export the design to CAD format.
        
        Args:
            format_type: Export format type
            
        Returns:
            CAD export data
        """
        if not self.current_design:
            return {"error": "No active design"}
            
        # Create CAD export
        cad_export = {
            "metadata": {
                "platform_type": self.current_design["platform_type"],
                "primary_goal": self.current_design["primary_goal"].name,
                "export_format": format_type
            },
            "dimensions": self.current_design["dimensions"],
            "features": [],
            "materials": [],
            "performance": self.current_design["performance_estimates"]
        }
        
        # Add features
        for feature_id, feature in self.current_design.get("features", {}).items():
            cad_feature = {
                "id": feature_id,
                "parameters": feature.get("parameters", {})
            }
            cad_export["features"].append(cad_feature)
            
        # Add materials
        for material_name, material_data in self.current_design.get("materials", {}).items():
            cad_material = {
                "name": material_name,
                "coverage": material_data["coverage"],
                "properties": material_data["properties"]
            }
            cad_export["materials"].append(cad_material)
            
        return cad_export
        
    def generate_stealth_config(self) -> StealthSystemConfig:
        """
        Generate a stealth system configuration from the design.
        
        Returns:
            StealthSystemConfig for the designed system
        """
        if not self.current_design:
            # Return a default configuration instead of None
            return StealthSystemConfig(
                stealth_type=StealthType.RADAR_ABSORBENT_MATERIAL,  # Changed from COMPOSITE
                name="Default Stealth Design",
                description="Default configuration - no design available",
                weight_kg=0.0,
                power_requirements_kw=0.1,
                material_config=None,
                neuromorphic_enabled=False,
                signature_config=StealthSignatureConfig(
                    radar_cross_section_reduction=0.0,
                    infrared_signature_reduction=0.0,
                    acoustic_signature_reduction=0.0,
                    electromagnetic_signature_reduction=0.0,
                    visual_signature_reduction=0.0
                )
            )
            
        # Determine stealth type based on primary goal
        stealth_type = StealthType.RADAR_ABSORBENT_MATERIAL  # Default to RAM instead of COMPOSITE
        if self.current_design["primary_goal"] == DesignOptimizationGoal.RADAR_MINIMIZATION:
            stealth_type = StealthType.RADAR_ABSORBENT_MATERIAL
        elif self.current_design["primary_goal"] == DesignOptimizationGoal.INFRARED_REDUCTION:
            stealth_type = StealthType.INFRARED_SUPPRESSION
        elif self.current_design["primary_goal"] == DesignOptimizationGoal.ACOUSTIC_DAMPENING:
            stealth_type = StealthType.ACOUSTIC_DAMPENING
            
        # Create material config if materials are used
        material_config = None
        primary_material = None
        if self.current_design.get("materials"):
            # Use the material with highest coverage
            primary_material = max(
                self.current_design["materials"].items(),
                key=lambda x: x[1]["coverage"]
            )
            material_name = primary_material[0]
            material_data = primary_material[1]
            
            material_config = StealthMaterialConfig(
                material_type=material_name,
                thickness_mm=5.0,  # Default thickness
                coverage_percentage=material_data["coverage"] * 100.0,
                frequency_range_ghz={"min": 0.5, "max": 18.0}
            )
            
        # Create stealth system config
        performance = self.current_design["performance_estimates"]
        
        config = StealthSystemConfig(
            stealth_type=stealth_type,
            name=f"{self.current_design['platform_type']} Stealth Design",
            description=f"Parametric design optimized for {self.current_design['primary_goal'].name}",
            weight_kg=self.current_design.get("weight_impact_kg", 0.0),
            power_requirements_kw=0.1,  # Default value
            material_config=material_config,
            neuromorphic_enabled=True,
            signature_config=StealthSignatureConfig(
                radar_cross_section_reduction=performance.get("rcs_reduction", 0.0) * 100.0,
                infrared_signature_reduction=performance.get("ir_reduction", 0.0) * 100.0,
                acoustic_signature_reduction=performance.get("acoustic_reduction", 0.0) * 100.0,
                electromagnetic_signature_reduction=performance.get("em_reduction", 0.0) * 100.0,
                visual_signature_reduction=50.0  # Adding the missing parameter with a default value
            )
        )
        
        return config


class CADIntegration:
    """Integration with CAD systems for stealth design."""
    
    def __init__(self, designer: Optional[ParametricStealthDesigner] = None):
        """Initialize CAD integration."""
        self.designer = designer or ParametricStealthDesigner()
        self.cad_formats = ["json", "step", "stl", "iges"]
        
    def import_from_cad(self, cad_data: Dict[str, Any], format_type: str = "json") -> bool:
        """
        Import design from CAD data.
        
        Args:
            cad_data: CAD data to import
            format_type: Format of the CAD data
            
        Returns:
            Success status
        """
        if format_type not in self.cad_formats:
            return False
            
        try:
            # Extract platform type and dimensions
            platform_type = cad_data.get("metadata", {}).get("platform_type", "generic")
            dimensions = cad_data.get("dimensions", {})
            
            # Determine primary goal
            goal_name = cad_data.get("metadata", {}).get("primary_goal", "RADAR_MINIMIZATION")
            primary_goal = DesignOptimizationGoal.RADAR_MINIMIZATION
            for goal in DesignOptimizationGoal:
                if goal.name == goal_name:
                    primary_goal = goal
                    break
                    
            # Create new design
            self.designer.create_new_design(platform_type, dimensions, primary_goal)
            
            # Add features
            for feature in cad_data.get("features", []):
                feature_id = feature.get("id", "")
                if "_" in feature_id:
                    feature_type_name, feature_name = feature_id.split("_", 1)
                    
                    # Convert feature type name to enum
                    feature_type = None
                    for ft in StealthFeatureType:
                        if ft.name.lower() == feature_type_name:
                            feature_type = ft
                            break
                            
                    if feature_type:
                        self.designer.add_feature(
                            feature_type,
                            feature_name,
                            feature.get("parameters", {})
                        )
                        
            # Add materials
            for material in cad_data.get("materials", []):
                self.designer.add_material(
                    material.get("name", ""),
                    material.get("coverage", 0.0)
                )
                
            return True
            
        except Exception as e:
            print(f"Error importing CAD data: {e}")
            return False
            
    def generate_cad_model(self, 
                         platform_type: str,
                         dimensions: Dict[str, float],
                         stealth_config: StealthSystemConfig) -> Dict[str, Any]:
        """
        Generate a CAD model from stealth configuration.
        
        Args:
            platform_type: Type of platform
            dimensions: Platform dimensions
            stealth_config: Stealth system configuration
            
        Returns:
            CAD model data
        """
        # Determine primary goal from stealth type
        primary_goal = DesignOptimizationGoal.MULTI_SPECTRUM
        if stealth_config.stealth_type == StealthType.RADAR_ABSORBENT_MATERIAL:
            primary_goal = DesignOptimizationGoal.RADAR_MINIMIZATION
        elif stealth_config.stealth_type == StealthType.INFRARED_SUPPRESSION:
            primary_goal = DesignOptimizationGoal.INFRARED_REDUCTION
            
        # Create new design
        self.designer.create_new_design(platform_type, dimensions, primary_goal)
        
        # Add materials if specified
        if stealth_config.material_config:
            material_type = stealth_config.material_config.material_type
            coverage = stealth_config.material_config.coverage_percentage / 100.0
            self.designer.add_material(material_type, coverage)
            
        # Add appropriate features based on stealth type
        if stealth_config.stealth_type == StealthType.RADAR_ABSORBENT_MATERIAL:
            self.designer.add_feature(StealthFeatureType.EDGE_TREATMENT, "sawtooth")
            self.designer.add_feature(StealthFeatureType.SURFACE_CONTOUR, "faceted")
        elif stealth_config.stealth_type == StealthType.INFRARED_SUPPRESSION:
            self.designer.add_feature(StealthFeatureType.COATING_PATTERN, "gradient")
            
        # Export to CAD format
        return self.designer.export_to_cad()

    def generate_manufacturing_specs(self) -> Dict[str, Any]:
        """
        Generate manufacturing specifications for the current design.
        
        Returns:
            Dictionary with manufacturing specifications
        """
        if not self.designer.current_design:
            return {"error": "No active design"}
            
        # Validate manufacturing constraints first
        validation = self.validate_manufacturing_constraints()
        if not validation["valid"]:
            return {
                "error": "Design violates manufacturing constraints",
                "validation": validation
            }
            
        # Generate basic manufacturing specs
        manufacturing_specs = {
            "design_id": id(self.designer.current_design),
            "platform_type": self.designer.current_design["platform_type"],
            "primary_goal": self.designer.current_design["primary_goal"].name,
            "dimensions": self.designer.current_design["dimensions"],
            "materials": [],
            "features": [],
            "process_requirements": {
                "estimated_production_time": validation["manufacturing_estimates"]["estimated_production_hours"],
                "required_equipment": validation["manufacturing_estimates"]["required_equipment"],
                "quality_checks": [
                    "thickness_measurement",
                    "radar_absorption_test",
                    "surface_uniformity_inspection"
                ]
            }
        }
        
        # Add material specifications
        for material_name, material_data in self.designer.current_design.get("materials", {}).items():
            material_spec = {
                "material_id": f"{material_name}_{id(material_data)}",
                "material_type": material_name,
                "coverage_percentage": material_data["coverage"] * 100.0,
                "thickness_mm": 5.0,  # Default thickness
                "processing_requirements": {}
            }
            
            # Add material-specific processing requirements
            if material_name in self.designer.manufacturing_constraints["material_constraints"]:
                constraints = self.designer.manufacturing_constraints["material_constraints"][material_name]
                material_spec["processing_requirements"] = {
                    "curing_time_hours": constraints["curing_time_hours"],
                    "max_temperature_c": constraints["max_temperature_c"]
                }
                
            manufacturing_specs["materials"].append(material_spec)
            
        # Add feature specifications
        for feature_id, feature in self.designer.current_design.get("features", {}).items():
            feature_spec = {
                "feature_id": feature_id,
                "parameters": feature.get("parameters", {}),
                "manufacturing_notes": []
            }
            
            # Add feature-specific manufacturing notes
            if "edge_treatment" in feature_id:
                feature_spec["manufacturing_notes"].append("Requires precision CNC milling")
            elif "surface_contour" in feature_id:
                feature_spec["manufacturing_notes"].append("Requires 5-axis CNC machining")
            elif "coating_pattern" in feature_id:
                feature_spec["manufacturing_notes"].append("Apply using masked spray process")
                
            manufacturing_specs["features"].append(feature_spec)
            
        return manufacturing_specs

    def _get_required_manufacturing_equipment(self) -> List[str]:
        """Determine required manufacturing equipment based on design."""
        required_equipment = []
        
        # Check materials to determine equipment
        for material_name in self.designer.current_design.get("materials", {}).keys():
            if material_name == "carbon_composite":
                required_equipment.extend(["autoclave", "cnc_mill"])
            elif material_name == "ceramic_matrix":
                required_equipment.extend(["kiln", "spray_booth"])
            elif material_name == "metamaterial":
                required_equipment.extend(["laser_cutter", "precision_printer"])
                
        # Check features to determine equipment
        for feature_id in self.designer.current_design.get("features", {}).keys():
            if "edge_treatment" in feature_id:
                required_equipment.append("cnc_mill")
            elif "surface_contour" in feature_id:
                required_equipment.append("5_axis_cnc")
            elif "coating_pattern" in feature_id:
                required_equipment.append("spray_booth")
                
        # Return unique equipment list
        return list(set(required_equipment))
        
    def _estimate_surface_area(self, dimensions: Dict[str, float]) -> float:
        """Estimate surface area based on platform dimensions."""
        # Reuse the same method from the designer class
        return self.designer._estimate_surface_area(dimensions)
        
    def validate_manufacturing_constraints(self) -> Dict[str, Any]:
        """
        Validate the current design against manufacturing constraints.
        
        Returns:
            Dictionary with validation results
        """
        if not self.designer.current_design:
            return {"valid": False, "errors": ["No active design"]}
            
        errors = []
        warnings = []
        
        # Validate materials
        for material_name, material_data in self.designer.current_design.get("materials", {}).items():
            if material_name in self.designer.manufacturing_constraints["material_constraints"]:
                constraints = self.designer.manufacturing_constraints["material_constraints"][material_name]
                
                # Check for material thickness constraints
                for feature_id, feature in self.designer.current_design.get("features", {}).items():
                    if "parameters" in feature and "thickness_mm" in feature["parameters"]:
                        thickness = feature["parameters"]["thickness_mm"]
                        if thickness < constraints["min_thickness_mm"]:
                            errors.append(f"Material {material_name} thickness {thickness}mm is below minimum {constraints['min_thickness_mm']}mm")
                        elif thickness > constraints["max_thickness_mm"]:
                            errors.append(f"Material {material_name} thickness {thickness}mm exceeds maximum {constraints['max_thickness_mm']}mm")
            else:
                warnings.append(f"No manufacturing constraints defined for material {material_name}")
                
        # Validate features
        for feature_id, feature in self.designer.current_design.get("features", {}).items():
            if "_" in feature_id:
                feature_type, feature_name = feature_id.split("_", 1)
                
                if feature_type in self.designer.manufacturing_constraints["feature_constraints"]:
                    constraints = self.designer.manufacturing_constraints["feature_constraints"][feature_type]
                    
                    # Check feature-specific constraints
                    if feature_type == "edge_treatments" and "parameters" in feature:
                        if "spacing_mm" in feature["parameters"] and feature["parameters"]["spacing_mm"] < constraints["min_spacing_mm"]:
                            errors.append(f"Edge treatment spacing {feature['parameters']['spacing_mm']}mm is below minimum {constraints['min_spacing_mm']}mm")
                    
                    elif feature_type == "coating_patterns" and "parameters" in feature:
                        if "cell_size_mm" in feature["parameters"] and feature["parameters"]["cell_size_mm"] < constraints["min_cell_size_mm"]:
                            errors.append(f"Coating pattern cell size {feature['parameters']['cell_size_mm']}mm is below minimum {constraints['min_cell_size_mm']}mm")
                        
                        if "layers" in feature["parameters"] and feature["parameters"]["layers"] > constraints["max_layer_count"]:
                            errors.append(f"Coating pattern layer count {feature['parameters']['layers']} exceeds maximum {constraints['max_layer_count']}")
                else:
                    warnings.append(f"No manufacturing constraints defined for feature type {feature_type}")
                    
        # Validate process constraints
        process_constraints = self.designer.manufacturing_constraints["process_constraints"]
        
        # Estimate surface area
        surface_area = self._estimate_surface_area(self.designer.current_design["dimensions"]) / 10000  # Convert cm² to m²
        
        if surface_area > process_constraints["max_continuous_area_m2"]:
            warnings.append(f"Design surface area {surface_area:.2f}m² exceeds maximum continuous area {process_constraints['max_continuous_area_m2']}m²")
            warnings.append("Manufacturing may require segmentation and assembly")
            
        # Estimate production time
        production_days = surface_area / process_constraints["max_production_rate_m2_per_day"]
        
        # Return validation results
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "manufacturing_estimates": {
                "surface_area_m2": surface_area,
                "estimated_production_days": production_days,
                "estimated_production_hours": production_days * 24,
                "required_equipment": self._determine_required_equipment()
            }
        }
        
    # Keep only one implementation of _determine_required_equipment
    def _determine_required_equipment(self) -> List[str]:
        """Determine required manufacturing equipment based on design."""
        required_equipment = []
        
        # Check materials to determine equipment
        for material_name in self.designer.current_design.get("materials", {}).keys():
            if material_name == "carbon_composite":
                required_equipment.extend(["autoclave", "cnc_mill"])
            elif material_name == "ceramic_matrix":
                required_equipment.extend(["kiln", "spray_booth"])
            elif material_name == "metamaterial":
                required_equipment.extend(["laser_cutter", "precision_printer"])
                
        # Check features to determine equipment
        for feature_id in self.designer.current_design.get("features", {}).keys():
            if "edge_treatment" in feature_id:
                required_equipment.append("cnc_mill")
            elif "surface_contour" in feature_id:
                required_equipment.append("5_axis_cnc")
            elif "coating_pattern" in feature_id:
                required_equipment.append("spray_booth")
                
        # Return unique equipment list
        return list(set(required_equipment))