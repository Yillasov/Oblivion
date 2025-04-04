#!/usr/bin/env python3
"""
Biomimetic Parameter Mapping System

This module provides a system for mapping biological features and characteristics
to engineering parameters for biomimetic UCAV design.
"""

import sys
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.biomimetic.design.principles import BiologicalInspiration, BiologicalReference

logger = get_logger("biomimetic_mapping")


@dataclass
class ParameterMapping:
    """Mapping between biological features and engineering parameters."""
    biological_feature: str
    target_system: str
    engineering_parameters: Dict[str, Any]
    scaling_functions: Optional[Dict[str, Callable]] = None
    constraints: Optional[Dict[str, Any]] = None
    performance_impact: Optional[Dict[str, float]] = None
    implementation_notes: Optional[str] = None
    references: Optional[List[str]] = None


class BiomimeticParameterMapper:
    """
    System for mapping biological features to engineering parameters.
    
    This class provides methods for defining, storing, retrieving, and applying
    mappings between biological features and engineering parameters.
    """
    
    def __init__(self, mapping_file: Optional[str] = None):
        """
        Initialize the biomimetic parameter mapper.
        
        Args:
            mapping_file: Path to the mapping file
        """
        self.mappings: Dict[str, ParameterMapping] = {}
        self.mapping_file = mapping_file or os.path.join(
            project_root, 
            "data", 
            "biomimetic", 
            "parameter_mappings.json"
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.mapping_file), exist_ok=True)
        
        # Initialize with default mappings
        self._initialize_default_mappings()
        
        # Load mappings if file exists
        self._load_mappings()
        
        logger.info(f"Initialized biomimetic parameter mapper with {len(self.mappings)} mappings")
    
    def _initialize_default_mappings(self) -> None:
        """Initialize default parameter mappings."""
        # Wing morphing mapping
        self.add_mapping(ParameterMapping(
            biological_feature="wing_morphing",
            target_system="wing_design",
            engineering_parameters={
                "aspect_ratio_range": [2.0, 8.0],
                "sweep_angle_range_deg": [15.0, 45.0],
                "camber_range": [0.02, 0.08],
                "flexibility_factor": 0.7
            },
            performance_impact={
                "maneuverability": 0.8,
                "efficiency": 0.6,
                "speed": -0.2
            },
            implementation_notes="Requires flexible materials and actuators"
        ))
        
        # Streamlined body mapping
        self.add_mapping(ParameterMapping(
            biological_feature="streamlined_body",
            target_system="fuselage_design",
            engineering_parameters={
                "fineness_ratio_range": [5.0, 10.0],
                "cross_section_type": "elliptical",
                "surface_smoothness": 0.95
            },
            performance_impact={
                "drag_reduction": 0.7,
                "stability": 0.5,
                "internal_volume": -0.3
            }
        ))
        
        # High maneuverability mapping
        self.add_mapping(ParameterMapping(
            biological_feature="high_maneuverability",
            target_system="control_surfaces",
            engineering_parameters={
                "control_surface_area_ratio": 0.15,
                "actuation_speed_deg_per_sec": 60.0,
                "deflection_range_deg": [-30.0, 30.0]
            },
            performance_impact={
                "turn_rate": 0.9,
                "stability": -0.2,
                "control_precision": 0.7
            }
        ))
        
        # Distributed sensing mapping
        self.add_mapping(ParameterMapping(
            biological_feature="distributed_sensing",
            target_system="sensor_integration",
            engineering_parameters={
                "sensor_density": 0.05,  # sensors per square cm
                "sensor_types": ["pressure", "temperature", "flow"],
                "integration_depth_mm": 2.0
            },
            performance_impact={
                "situational_awareness": 0.8,
                "weight_penalty": -0.3,
                "structural_integrity": -0.1
            }
        ))
    
    def _load_mappings(self) -> None:
        """Load mappings from file if it exists."""
        if not os.path.exists(self.mapping_file):
            return
            
        try:
            with open(self.mapping_file, 'r') as f:
                data = json.load(f)
                
            for mapping_key, mapping_data in data.items():
                # Skip if already defined in default mappings
                if mapping_key in self.mappings:
                    continue
                    
                # Create mapping object
                mapping = ParameterMapping(
                    biological_feature=mapping_data["biological_feature"],
                    target_system=mapping_data["target_system"],
                    engineering_parameters=mapping_data["engineering_parameters"],
                    performance_impact=mapping_data.get("performance_impact"),
                    implementation_notes=mapping_data.get("implementation_notes"),
                    references=mapping_data.get("references")
                )
                
                # Add to mappings
                self.mappings[mapping_key] = mapping
                
            logger.info(f"Loaded {len(data)} mappings from file")
                
        except Exception as e:
            logger.error(f"Error loading mappings: {str(e)}")
    
    def save_mappings(self) -> bool:
        """
        Save mappings to file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert mappings to serializable format
            serializable_data = {}
            for mapping_key, mapping in self.mappings.items():
                mapping_dict = {
                    "biological_feature": mapping.biological_feature,
                    "target_system": mapping.target_system,
                    "engineering_parameters": mapping.engineering_parameters
                }
                
                if mapping.performance_impact:
                    mapping_dict["performance_impact"] = mapping.performance_impact
                    
                if mapping.implementation_notes:
                    mapping_dict["implementation_notes"] = mapping.implementation_notes
                    
                if mapping.references:
                    mapping_dict["references"] = mapping.references
                    
                serializable_data[mapping_key] = mapping_dict
            
            with open(self.mapping_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
            logger.info(f"Saved {len(self.mappings)} mappings to file")
            return True
            
        except Exception as e:
            logger.error(f"Error saving mappings: {str(e)}")
            return False
    
    def add_mapping(self, mapping: ParameterMapping) -> bool:
        """
        Add a parameter mapping.
        
        Args:
            mapping: The parameter mapping to add
            
        Returns:
            True if added successfully, False otherwise
        """
        mapping_key = f"{mapping.biological_feature}_{mapping.target_system}"
        
        if mapping_key in self.mappings:
            logger.warning(f"Mapping for {mapping_key} already exists")
            return False
            
        self.mappings[mapping_key] = mapping
        return True
    
    def get_mapping(self, biological_feature: str, target_system: str) -> Optional[ParameterMapping]:
        """
        Get a parameter mapping.
        
        Args:
            biological_feature: The biological feature
            target_system: The target engineering system
            
        Returns:
            ParameterMapping if found, None otherwise
        """
        mapping_key = f"{biological_feature}_{target_system}"
        return self.mappings.get(mapping_key)
    
    def apply_mapping(self, 
                     biological_reference: BiologicalReference, 
                     target_system: str,
                     feature: Optional[str] = None) -> Dict[str, Any]:
        """
        Apply parameter mapping to a biological reference.
        
        Args:
            biological_reference: The biological reference model
            target_system: The target engineering system
            feature: Specific feature to map (if None, maps all applicable features)
            
        Returns:
            Dictionary of mapped engineering parameters
        """
        result = {}
        
        # Determine features to map
        features_to_map = [feature] if feature else biological_reference.key_features
        
        # Apply mappings for each feature
        for bio_feature in features_to_map:
            mapping = self.get_mapping(bio_feature, target_system)
            
            if not mapping:
                continue
                
            # Apply mapping
            mapped_params = mapping.engineering_parameters.copy()
            
            # Apply scaling functions if available
            if mapping.scaling_functions:
                for param_name, scaling_func in mapping.scaling_functions.items():
                    if param_name in mapped_params:
                        # Get relevant biological metrics
                        if bio_feature in biological_reference.performance_metrics:
                            bio_value = biological_reference.performance_metrics[bio_feature]
                            mapped_params[param_name] = scaling_func(bio_value, mapped_params[param_name])
            
            # Add to result with feature as prefix to avoid parameter name collisions
            for param_name, param_value in mapped_params.items():
                result[f"{bio_feature}_{param_name}"] = param_value
                
            # Add performance impact if available
            if mapping.performance_impact:
                result[f"{bio_feature}_performance_impact"] = mapping.performance_impact
        
        return result
    
    def get_applicable_features(self, target_system: str) -> List[str]:
        """
        Get biological features applicable to a target system.
        
        Args:
            target_system: The target engineering system
            
        Returns:
            List of applicable biological features
        """
        applicable_features = []
        
        for mapping_key, mapping in self.mappings.items():
            if mapping.target_system == target_system:
                applicable_features.append(mapping.biological_feature)
                
        return applicable_features
    
    def get_applicable_systems(self, biological_feature: str) -> List[str]:
        """
        Get engineering systems applicable to a biological feature.
        
        Args:
            biological_feature: The biological feature
            
        Returns:
            List of applicable engineering systems
        """
        applicable_systems = []
        
        for mapping_key, mapping in self.mappings.items():
            if mapping.biological_feature == biological_feature:
                applicable_systems.append(mapping.target_system)
                
        return applicable_systems