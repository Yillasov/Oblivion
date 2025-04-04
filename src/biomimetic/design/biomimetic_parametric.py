#!/usr/bin/env python3
"""
Biomimetic Parametric Design

This module extends the parametric design system to support biomimetic and organic shapes
for UCAV design, integrating biological reference models with engineering parameters.
"""

import sys
import os
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.manufacturing.cad.parametric import (
    ParametricDesign, DesignParameter, ParameterRelationship, 
    ParameterType, RelationshipType, UCAVParametricDesign
)
from src.biomimetic.design.principles import BiologicalReference, BiomimeticPrinciple
from src.biomimetic.design.parameter_mapping import BiomimeticParameterMapper

logger = get_logger("biomimetic_parametric")


class BiomimeticParametricDesign(UCAVParametricDesign):
    """
    Extends the UCAV parametric design system with biomimetic capabilities.
    
    This class integrates biological reference models and biomimetic design principles
    with the parametric design system, enabling the creation of organic and biomimetic
    shapes for UCAVs.
    """
    
    def __init__(self):
        """Initialize the biomimetic parametric design system."""
        super().__init__()
        self.parameter_mapper = BiomimeticParameterMapper()
        self.biological_reference: Optional[BiologicalReference] = None
        self.biomimetic_parameters: Dict[str, Any] = {}
        self.active_features: List[str] = []
        
        # Add biomimetic-specific parameters
        self._initialize_biomimetic_parameters()
        
        # Add flexible wing geometry parameters
        self._initialize_flexible_wing_parameters()
        
        logger.info("Initialized biomimetic parametric design system with flexible wing geometry")
    
    def _initialize_biomimetic_parameters(self):
        """Initialize biomimetic-specific design parameters."""
        # Organic form parameters
        self.add_parameter(DesignParameter(
            name="organic_form_factor",
            type=ParameterType.RATIO,
            value=0.5,  # 0.0 = mechanical, 1.0 = fully organic
            min_value=0.0,
            max_value=1.0
        ))
        
        # Surface curvature complexity
        self.add_parameter(DesignParameter(
            name="surface_complexity",
            type=ParameterType.RATIO,
            value=0.3,
            min_value=0.0,
            max_value=1.0
        ))
        
        # Asymmetry allowance
        self.add_parameter(DesignParameter(
            name="asymmetry_factor",
            type=ParameterType.RATIO,
            value=0.1,
            min_value=0.0,
            max_value=0.5
        ))
        
        # Morphing capability
        self.add_parameter(DesignParameter(
            name="morphing_capability",
            type=ParameterType.RATIO,
            value=0.2,
            min_value=0.0,
            max_value=1.0
        ))
        
        # Add parameter group for biomimetic parameters
        self.add_parameter_group("biomimetic_features", [
            "organic_form_factor", 
            "surface_complexity", 
            "asymmetry_factor",
            "morphing_capability"
        ])
    
    def _initialize_flexible_wing_parameters(self):
        """Initialize flexible wing geometry parameters for biomimetic designs."""
        # Wing curvature parameters
        self.add_parameter(DesignParameter(
            name="wing_camber_profile",
            type=ParameterType.ENUM,
            value="adaptive",
            allowed_values=["flat", "reflex", "adaptive", "s-curve", "bird-like"]
        ))
        
        self.add_parameter(DesignParameter(
            name="wing_camber_ratio",
            type=ParameterType.RATIO,
            value=0.04,  # 4% camber
            min_value=0.0,
            max_value=0.12
        ))
        
        # Wing twist parameters
        self.add_parameter(DesignParameter(
            name="wing_twist_angle",
            type=ParameterType.ANGLE,
            value=3.0,  # degrees
            min_value=-5.0,
            max_value=10.0
        ))
        
        # Wing planform parameters
        self.add_parameter(DesignParameter(
            name="wing_planform_shape",
            type=ParameterType.ENUM,
            value="elliptical",
            allowed_values=["rectangular", "tapered", "elliptical", "crescent", "bird-wing"]
        ))
        
        # Wingtip parameters
        self.add_parameter(DesignParameter(
            name="wingtip_morphology",
            type=ParameterType.ENUM,
            value="standard",
            allowed_values=["standard", "winglet", "raked", "feathered", "slotted"]
        ))
        
        self.add_parameter(DesignParameter(
            name="wingtip_cant_angle",
            type=ParameterType.ANGLE,
            value=30.0,  # degrees
            min_value=0.0,
            max_value=90.0
        ))
        
        # Leading edge parameters
        self.add_parameter(DesignParameter(
            name="leading_edge_radius",
            type=ParameterType.RATIO,
            value=0.02,  # 2% of chord
            min_value=0.005,
            max_value=0.05
        ))
        
        self.add_parameter(DesignParameter(
            name="leading_edge_droop",
            type=ParameterType.ANGLE,
            value=0.0,  # degrees
            min_value=0.0,
            max_value=15.0
        ))
        
        # Add articulated joint parameters
        self.add_parameter(DesignParameter(
            name="joint_count",
            type=ParameterType.INTEGER,
            value=3,
            min_value=0,
            max_value=8
        ))
        
        self.add_parameter(DesignParameter(
            name="joint_flexibility",
            type=ParameterType.RATIO,
            value=0.5,  # 0.0 = rigid, 1.0 = highly flexible
            min_value=0.0,
            max_value=1.0
        ))
        
        self.add_parameter(DesignParameter(
            name="joint_distribution",
            type=ParameterType.ENUM,
            value="uniform",
            allowed_values=["uniform", "leading_edge", "trailing_edge", "bird_like", "bat_like"]
        ))
        
        self.add_parameter(DesignParameter(
            name="joint_actuation_type",
            type=ParameterType.ENUM,
            value="passive",
            allowed_values=["passive", "active", "hybrid", "neuromorphic"]
        ))
        
        # Add parameter group for flexible wing parameters
        self.add_parameter_group("flexible_wing_geometry", [
            "wing_camber_profile",
            "wing_camber_ratio",
            "wing_twist_angle",
            "wing_planform_shape",
            "wingtip_morphology",
            "wingtip_cant_angle",
            "leading_edge_radius",
            "leading_edge_droop"
        ])
        
        # Add relationships between parameters
        self._add_flexible_wing_relationships()
    
    def _add_flexible_wing_relationships(self):
        """Add relationships between flexible wing parameters."""
        # Wing planform shape affects aspect ratio limits
        self.add_relationship(ParameterRelationship(
            source_param="wing_planform_shape",
            target_param="aspect_ratio",
            relationship_type=RelationshipType.CUSTOM,
            custom_function=self._adjust_aspect_ratio_limits
        ))
        
        # Wing camber profile affects camber ratio limits
        self.add_relationship(ParameterRelationship(
            source_param="wing_camber_profile",
            target_param="wing_camber_ratio",
            relationship_type=RelationshipType.CUSTOM,
            custom_function=self._adjust_camber_ratio_limits
        ))
        
        # Wingtip morphology affects cant angle limits
        self.add_relationship(ParameterRelationship(
            source_param="wingtip_morphology",
            target_param="wingtip_cant_angle",
            relationship_type=RelationshipType.CUSTOM,
            custom_function=self._adjust_cant_angle_limits
        ))
    
    def _adjust_aspect_ratio_limits(self, planform_shape: str) -> float:
        """Adjust aspect ratio limits based on wing planform shape."""
        aspect_ratio_param = self.parameters["aspect_ratio"]
        current_value = aspect_ratio_param.value
        
        # Adjust limits based on planform shape
        if planform_shape == "rectangular":
            aspect_ratio_param.min_value = 4.0
            aspect_ratio_param.max_value = 8.0
        elif planform_shape == "tapered":
            aspect_ratio_param.min_value = 5.0
            aspect_ratio_param.max_value = 9.0
        elif planform_shape == "elliptical":
            aspect_ratio_param.min_value = 6.0
            aspect_ratio_param.max_value = 10.0
        elif planform_shape == "crescent":
            aspect_ratio_param.min_value = 7.0
            aspect_ratio_param.max_value = 12.0
        elif planform_shape == "bird-wing":
            aspect_ratio_param.min_value = 8.0
            aspect_ratio_param.max_value = 15.0
        
        # Ensure current value is within new limits
        return max(aspect_ratio_param.min_value, min(current_value, aspect_ratio_param.max_value))
    
    def _adjust_camber_ratio_limits(self, camber_profile: str) -> float:
        """Adjust camber ratio limits based on wing camber profile."""
        camber_ratio_param = self.parameters["wing_camber_ratio"]
        current_value = camber_ratio_param.value
        
        # Adjust limits based on camber profile
        if camber_profile == "flat":
            camber_ratio_param.min_value = 0.0
            camber_ratio_param.max_value = 0.02
        elif camber_profile == "reflex":
            camber_ratio_param.min_value = 0.02
            camber_ratio_param.max_value = 0.06
        elif camber_profile == "adaptive":
            camber_ratio_param.min_value = 0.0
            camber_ratio_param.max_value = 0.08
        elif camber_profile == "s-curve":
            camber_ratio_param.min_value = 0.03
            camber_ratio_param.max_value = 0.10
        elif camber_profile == "bird-like":
            camber_ratio_param.min_value = 0.04
            camber_ratio_param.max_value = 0.12
        
        # Ensure current value is within new limits
        return max(camber_ratio_param.min_value, min(current_value, camber_ratio_param.max_value))
    
    def _adjust_cant_angle_limits(self, wingtip_morphology: str) -> float:
        """Adjust wingtip cant angle limits based on wingtip morphology."""
        cant_angle_param = self.parameters["wingtip_cant_angle"]
        current_value = cant_angle_param.value
        
        # Adjust limits based on wingtip morphology
        if wingtip_morphology == "standard":
            cant_angle_param.min_value = 0.0
            cant_angle_param.max_value = 15.0
        elif wingtip_morphology == "winglet":
            cant_angle_param.min_value = 45.0
            cant_angle_param.max_value = 90.0
        elif wingtip_morphology == "raked":
            cant_angle_param.min_value = 15.0
            cant_angle_param.max_value = 45.0
        elif wingtip_morphology == "feathered":
            cant_angle_param.min_value = 10.0
            cant_angle_param.max_value = 60.0
        elif wingtip_morphology == "slotted":
            cant_angle_param.min_value = 20.0
            cant_angle_param.max_value = 70.0
        
        # Ensure current value is within new limits
        return max(cant_angle_param.min_value, min(current_value, cant_angle_param.max_value))
    
    def apply_biological_reference(self, reference: BiologicalReference) -> bool:
        """
        Apply a biological reference model to the design.
        
        Args:
            reference: The biological reference model to apply
            
        Returns:
            True if applied successfully, False otherwise
        """
        self.biological_reference = reference
        self.active_features = reference.key_features
        
        # Map biological features to engineering parameters
        success = self._map_biological_features()
        
        if success:
            # Update design parameters based on mapped values
            self._update_design_parameters()
            
            # Update flexible wing parameters based on biological reference
            self._update_flexible_wing_parameters()
            
            logger.info(f"Applied biological reference: {reference.name}")
        
        return success
    
    def _map_biological_features(self) -> bool:
        """
        Map biological features to engineering parameters.
        
        Returns:
            True if mapped successfully, False otherwise
        """
        if not self.biological_reference:
            logger.warning("No biological reference model set")
            return False
        
        # Map features to wing design parameters
        wing_params = self.parameter_mapper.apply_mapping(
            self.biological_reference,
            "wing_design"
        )
        
        # Map features to fuselage design parameters
        fuselage_params = self.parameter_mapper.apply_mapping(
            self.biological_reference,
            "fuselage_design"
        )
        
        # Map features to control surfaces parameters
        control_params = self.parameter_mapper.apply_mapping(
            self.biological_reference,
            "control_surfaces"
        )
        
        # Combine all mapped parameters
        self.biomimetic_parameters = {
            **wing_params,
            **fuselage_params,
            **control_params
        }
        
        return len(self.biomimetic_parameters) > 0
    
    def _update_design_parameters(self) -> None:
        """Update design parameters based on mapped biological features."""
        # Update wingspan if available in mapped parameters
        if "wing_morphing_aspect_ratio_range" in self.biomimetic_parameters:
            aspect_ratio_range = self.biomimetic_parameters["wing_morphing_aspect_ratio_range"]
            # Use midpoint of range as default
            aspect_ratio = (aspect_ratio_range[0] + aspect_ratio_range[1]) / 2
            self.set_parameter("aspect_ratio", aspect_ratio)
        
        # Update wing sweep if available
        if "wing_morphing_sweep_angle_range_deg" in self.biomimetic_parameters:
            sweep_range = self.biomimetic_parameters["wing_morphing_sweep_angle_range_deg"]
            sweep_angle = (sweep_range[0] + sweep_range[1]) / 2
            self.set_parameter("wing_sweep", sweep_angle)
        
        # Update organic form factor based on biological reference
        if self.biological_reference and self.biological_reference.performance_metrics:
            # Calculate organic form factor based on performance metrics
            if "glide_ratio" in self.biological_reference.performance_metrics:
                glide_ratio = self.biological_reference.performance_metrics["glide_ratio"]
                # Higher glide ratio typically means more streamlined/organic form
                organic_factor = min(1.0, glide_ratio / 20.0)  # Normalize to 0-1 range
                self.set_parameter("organic_form_factor", organic_factor)
        
        # Update surface complexity based on biological features
        if "streamlined_body" in self.active_features:
            # Streamlined bodies typically have lower surface complexity
            self.set_parameter("surface_complexity", 0.2)
        elif "distributed_sensing" in self.active_features:
            # Distributed sensing often requires more complex surfaces
            self.set_parameter("surface_complexity", 0.7)
        
        # Update morphing capability if wing morphing is a feature
        if "wing_morphing" in self.active_features:
            self.set_parameter("morphing_capability", 0.8)
    
    def _update_flexible_wing_parameters(self) -> None:
        """Update flexible wing parameters based on biological reference."""
        if not self.biological_reference:
            return
            
        # Set wing planform shape based on biological reference
        if "bird_wing" in self.active_features:
            self.set_parameter("wing_planform_shape", "bird-wing")
            self.set_parameter("wing_camber_profile", "bird-like")
            self.set_parameter("wingtip_morphology", "feathered")
            self.set_parameter("wing_twist_angle", 5.0)
        elif "bat_wing" in self.active_features:
            self.set_parameter("wing_planform_shape", "elliptical")
            self.set_parameter("wing_camber_profile", "adaptive")
            self.set_parameter("wingtip_morphology", "slotted")
            self.set_parameter("wing_twist_angle", 8.0)
        elif "insect_wing" in self.active_features:
            self.set_parameter("wing_planform_shape", "tapered")
            self.set_parameter("wing_camber_profile", "s-curve")
            self.set_parameter("wingtip_morphology", "standard")
            self.set_parameter("wing_twist_angle", 2.0)
        elif "high_speed_flight" in self.active_features:
            self.set_parameter("wing_planform_shape", "crescent")
            self.set_parameter("wing_camber_profile", "reflex")
            self.set_parameter("wingtip_morphology", "raked")
            self.set_parameter("wing_twist_angle", 1.0)
        
        # Set leading edge parameters based on flight characteristics
        if self.biological_reference.performance_metrics:
            if "maneuverability" in self.biological_reference.performance_metrics:
                maneuverability = self.biological_reference.performance_metrics["maneuverability"]
                # Higher maneuverability often correlates with more leading edge droop
                droop_angle = min(15.0, maneuverability * 15.0)
                self.set_parameter("leading_edge_droop", droop_angle)
                
            if "cruise_speed" in self.biological_reference.performance_metrics:
                cruise_speed = self.biological_reference.performance_metrics["cruise_speed"]
                # Higher speeds typically require smaller leading edge radius
                radius_factor = max(0.005, 0.03 - (cruise_speed / 1000.0))
                self.set_parameter("leading_edge_radius", radius_factor)
    
    def generate_design(self) -> Dict[str, Any]:
        """
        Generate design data from parameters with biomimetic enhancements.
        
        Returns:
            Dictionary containing design data
        """
        # Get base design data from parent class
        design_data = super().generate_design()
        
        # If there was an error in base design generation, return it
        if "error" in design_data:
            return design_data
        
        # Add biomimetic-specific design data
        design_data["biomimetic"] = {
            "organic_form_factor": self.parameters["organic_form_factor"].value,
            "surface_complexity": self.parameters["surface_complexity"].value,
            "asymmetry_factor": self.parameters["asymmetry_factor"].value,
            "morphing_capability": self.parameters["morphing_capability"].value,
            "active_features": self.active_features,
            "biological_reference": self.biological_reference.name if self.biological_reference else None
        }
        
        # Add flexible wing geometry data
        design_data["flexible_wing"] = {
            "camber_profile": self.parameters["wing_camber_profile"].value,
            "camber_ratio": self.parameters["wing_camber_ratio"].value,
            "twist_angle": self.parameters["wing_twist_angle"].value,
            "planform_shape": self.parameters["wing_planform_shape"].value,
            "wingtip_morphology": self.parameters["wingtip_morphology"].value,
            "wingtip_cant_angle": self.parameters["wingtip_cant_angle"].value,
            "leading_edge_radius": self.parameters["leading_edge_radius"].value,
            "leading_edge_droop": self.parameters["leading_edge_droop"].value
        }
        
        # Enhance aerodynamics data with biomimetic factors
        if "aerodynamics" in design_data:
            design_data["aerodynamics"]["biomimetic_enhancement"] = self._calculate_biomimetic_enhancement()
            design_data["aerodynamics"]["flexible_wing_enhancement"] = self._calculate_flexible_wing_enhancement()
        
        return design_data
    
    def _calculate_biomimetic_enhancement(self) -> Dict[str, float]:
        """
        Calculate performance enhancements from biomimetic features.
        
        Returns:
            Dictionary of enhancement factors
        """
        enhancements = {
            "drag_reduction": 0.0,
            "lift_enhancement": 0.0,
            "maneuverability": 0.0,
            "stealth": 0.0
        }
        
        # Apply organic form factor to drag reduction
        organic_factor = self.parameters["organic_form_factor"].value
        enhancements["drag_reduction"] = organic_factor * 0.15  # Up to 15% drag reduction
        
        # Apply surface complexity to lift enhancement
        surface_complexity = self.parameters["surface_complexity"].value
        enhancements["lift_enhancement"] = surface_complexity * 0.1  # Up to 10% lift enhancement
        
        # Apply morphing capability to maneuverability
        morphing_capability = self.parameters["morphing_capability"].value
        enhancements["maneuverability"] = morphing_capability * 0.25  # Up to 25% maneuverability increase
        
        # Apply asymmetry factor to stealth (can be positive or negative)
        asymmetry_factor = self.parameters["asymmetry_factor"].value
        # Moderate asymmetry can enhance stealth, but too much can degrade it
        if asymmetry_factor < 0.3:
            enhancements["stealth"] = asymmetry_factor * 0.2  # Up to 6% stealth enhancement
        else:
            enhancements["stealth"] = (0.3 - asymmetry_factor) * 0.4  # Up to -8% stealth degradation
        
        return enhancements
    
    def _calculate_flexible_wing_enhancement(self) -> Dict[str, float]:
        """
        Calculate performance enhancements from flexible wing features.
        
        Returns:
            Dictionary of enhancement factors
        """
        enhancements = {
            "lift_to_drag_ratio": 0.0,
            "stall_resistance": 0.0,
            "control_effectiveness": 0.0,
            "gust_response": 0.0
        }
        
        # Apply wing camber profile and ratio to lift-to-drag ratio
        camber_profile = self.parameters["wing_camber_profile"].value
        camber_ratio = self.parameters["wing_camber_ratio"].value
        
        if camber_profile == "flat":
            enhancements["lift_to_drag_ratio"] = camber_ratio * 2.0
        elif camber_profile == "reflex":
            enhancements["lift_to_drag_ratio"] = camber_ratio * 3.0
        elif camber_profile == "adaptive":
            enhancements["lift_to_drag_ratio"] = camber_ratio * 5.0
        elif camber_profile == "s-curve":
            enhancements["lift_to_drag_ratio"] = camber_ratio * 4.0
        elif camber_profile == "bird-like":
            enhancements["lift_to_drag_ratio"] = camber_ratio * 6.0
        
        # Apply wing twist to stall resistance
        twist_angle = self.parameters["wing_twist_angle"].value
        enhancements["stall_resistance"] = min(0.3, twist_angle * 0.03)  # Up to 30% stall resistance
        
        # Apply wingtip morphology to control effectiveness
        wingtip_morphology = self.parameters["wingtip_morphology"].value
        if wingtip_morphology == "standard":
            enhancements["control_effectiveness"] = 0.0
        elif wingtip_morphology == "winglet":
            enhancements["control_effectiveness"] = 0.15
        elif wingtip_morphology == "raked":
            enhancements["control_effectiveness"] = 0.10
        elif wingtip_morphology == "feathered":
            enhancements["control_effectiveness"] = 0.25
        elif wingtip_morphology == "slotted":
            enhancements["control_effectiveness"] = 0.20
        
        # Apply leading edge parameters to gust response
        leading_edge_radius = self.parameters["leading_edge_radius"].value
        leading_edge_droop = self.parameters["leading_edge_droop"].value
        
        # Smaller radius and more droop generally improve gust response
        radius_factor = (0.05 - leading_edge_radius) / 0.045  # Normalize to 0-1
        droop_factor = leading_edge_droop / 15.0  # Normalize to 0-1
        
        enhancements["gust_response"] = 0.2 * radius_factor + 0.3 * droop_factor  # Up to 50% improvement
        
        return enhancements
        
        # Add flexible wing geometry data with articulated joints
        design_data["flexible_wing"] = {
            "camber_profile": self.parameters["wing_camber_profile"].value,
            "camber_ratio": self.parameters["wing_camber_ratio"].value,
            "twist_angle": self.parameters["wing_twist_angle"].value,
            "planform_shape": self.parameters["wing_planform_shape"].value,
            "wingtip_morphology": self.parameters["wingtip_morphology"].value,
            "wingtip_cant_angle": self.parameters["wingtip_cant_angle"].value,
            "leading_edge_radius": self.parameters["leading_edge_radius"].value,
            "leading_edge_droop": self.parameters["leading_edge_droop"].value
        }
        
        # Enhance aerodynamics data with biomimetic factors
        if "aerodynamics" in design_data:
            design_data["aerodynamics"]["biomimetic_enhancement"] = self._calculate_biomimetic_enhancement()
            design_data["aerodynamics"]["flexible_wing_enhancement"] = self._calculate_flexible_wing_enhancement()
        
        return design_data
    
    def _calculate_joint_positions(self) -> List[Dict[str, Any]]:
        """Calculate joint positions based on distribution type and count."""
        joint_count = self.parameters["joint_count"].value
        distribution = self.parameters["joint_distribution"].value
        
        if joint_count == 0:
            return []
            
        joints = []
        
        # Calculate positions based on distribution type
        if distribution == "uniform":
            # Evenly distribute joints along wingspan
            for i in range(joint_count):
                position = (i + 1) / (joint_count + 1)  # Normalized position (0-1)
                joints.append({
                    "position": position,
                    "axis": [0, 0, 1],  # Default rotation axis
                    "range": [-15, 15]   # Default rotation range in degrees
                })
                
        elif distribution == "leading_edge":
            # Concentrate joints along leading edge
            for i in range(joint_count):
                position = (i + 1) / (joint_count + 1)
                joints.append({
                    "position": position,
                    "axis": [0, 1, 0],  # Leading edge rotation axis
                    "range": [-20, 10]   # Asymmetric range
                })
                
        elif distribution == "trailing_edge":
            # Concentrate joints along trailing edge
            for i in range(joint_count):
                position = (i + 1) / (joint_count + 1)
                joints.append({
                    "position": position,
                    "axis": [0, -1, 0],  # Trailing edge rotation axis
                    "range": [-5, 25]    # Asymmetric range
                })
                
        elif distribution == "bird_like":
            # Bird-like joint distribution (shoulder, elbow, wrist, finger joints)
            positions = [0.2, 0.5, 0.75, 0.9][:joint_count]
            axes = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]][:joint_count]
            ranges = [[-30, 45], [-10, 20], [-15, 15], [-5, 10]][:joint_count]
            
            for i in range(joint_count):
                joints.append({
                    "position": positions[i],
                    "axis": axes[i],
                    "range": ranges[i]
                })
                
        elif distribution == "bat_like":
            # Bat-like joint distribution (many finger joints)
            base_positions = [0.15, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 0.95]
            positions = base_positions[:joint_count]
            
            for i in range(joint_count):
                # Alternate joint axes for complex movement
                axis = [0, 0, 1] if i % 3 == 0 else [0, 1, 0] if i % 3 == 1 else [1, 0, 0]
                joints.append({
                    "position": positions[i],
                    "axis": axis,
                    "range": [-25, 25]
                })
        
        return joints
    
    def _calculate_flexible_wing_enhancement(self) -> Dict[str, float]:
        """Calculate performance enhancements from flexible wing features."""
        enhancements = {
            "lift_to_drag_ratio": 0.0,
            "stall_resistance": 0.0,
            "control_effectiveness": 0.0,
            "gust_response": 0.0,
            "morphing_efficiency": 0.0  # New enhancement for articulated joints
        }
        
        # Apply wing camber profile and ratio to lift-to-drag ratio
        camber_profile = self.parameters["wing_camber_profile"].value
        camber_ratio = self.parameters["wing_camber_ratio"].value
        
        if camber_profile == "flat":
            enhancements["lift_to_drag_ratio"] = camber_ratio * 2.0
        elif camber_profile == "reflex":
            enhancements["lift_to_drag_ratio"] = camber_ratio * 3.0
        elif camber_profile == "adaptive":
            enhancements["lift_to_drag_ratio"] = camber_ratio * 5.0
        elif camber_profile == "s-curve":
            enhancements["lift_to_drag_ratio"] = camber_ratio * 4.0
        elif camber_profile == "bird-like":
            enhancements["lift_to_drag_ratio"] = camber_ratio * 6.0
        
        # Apply wing twist to stall resistance
        twist_angle = self.parameters["wing_twist_angle"].value
        enhancements["stall_resistance"] = min(0.3, twist_angle * 0.03)  # Up to 30% stall resistance
        
        # Apply wingtip morphology to control effectiveness
        wingtip_morphology = self.parameters["wingtip_morphology"].value
        if wingtip_morphology == "standard":
            enhancements["control_effectiveness"] = 0.0
        elif wingtip_morphology == "winglet":
            enhancements["control_effectiveness"] = 0.15
        elif wingtip_morphology == "raked":
            enhancements["control_effectiveness"] = 0.10
        elif wingtip_morphology == "feathered":
            enhancements["control_effectiveness"] = 0.25
        elif wingtip_morphology == "slotted":
            enhancements["control_effectiveness"] = 0.20
        
        # Apply leading edge parameters to gust response
        leading_edge_radius = self.parameters["leading_edge_radius"].value
        leading_edge_droop = self.parameters["leading_edge_droop"].value
        
        # Smaller radius and more droop generally improve gust response
        radius_factor = (0.05 - leading_edge_radius) / 0.045  # Normalize to 0-1
        droop_factor = leading_edge_droop / 15.0  # Normalize to 0-1
        
        enhancements["gust_response"] = 0.2 * radius_factor + 0.3 * droop_factor  # Up to 50% improvement
        
        return enhancements
        
        # Calculate joint-based enhancements
        joint_count = self.parameters["joint_count"].value
        joint_flexibility = self.parameters["joint_flexibility"].value
        joint_actuation = self.parameters["joint_actuation_type"].value
        
        # More joints and flexibility improve morphing efficiency
        base_morphing = min(0.8, joint_count * 0.1) * joint_flexibility
        
        # Actuation type affects efficiency
        actuation_factor = {
            "passive": 0.6,
            "active": 0.9,
            "hybrid": 1.0,
            "neuromorphic": 1.2  # Neuromorphic control provides additional benefits
        }
        
        enhancements["morphing_efficiency"] = base_morphing * actuation_factor.get(joint_actuation, 1.0)
        
        # Joints also affect other performance metrics
        enhancements["control_effectiveness"] += enhancements["morphing_efficiency"] * 0.5
        enhancements["gust_response"] += joint_flexibility * 0.2
        
        return enhancements