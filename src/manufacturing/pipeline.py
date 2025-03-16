from typing import Dict, Any, List, Optional
import numpy as np
import os
import json
from enum import Enum

class ManufacturingProcess(Enum):
    ADDITIVE = "additive"
    SUBTRACTIVE = "subtractive"
    COMPOSITE_LAYUP = "composite_layup"
    INJECTION_MOLDING = "injection_molding"
    NANOFABRICATION = "nanofabrication"

class ManufacturingStage(Enum):
    DESIGN_VALIDATION = "design_validation"
    MATERIAL_SELECTION = "material_selection"
    TOOLING_GENERATION = "tooling_generation"
    FABRICATION = "fabrication"
    ASSEMBLY = "assembly"
    QUALITY_CONTROL = "quality_control"

class ManufacturingPipeline:
    """Pipeline for converting airframe designs to manufacturing instructions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get("output_dir", "/tmp/manufacturing")
        self.processes = {process.value: {} for process in ManufacturingProcess}
        self.current_stage = ManufacturingStage.DESIGN_VALIDATION
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def process_design(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an airframe design through the manufacturing pipeline."""
        results = {}
        
        # Design validation
        self.current_stage = ManufacturingStage.DESIGN_VALIDATION
        validation_results = self._validate_design(design_data)
        results["validation"] = validation_results
        
        if not validation_results.get("valid", False):
            return results
        
        # Material selection
        self.current_stage = ManufacturingStage.MATERIAL_SELECTION
        material_results = self._select_materials(design_data)
        results["materials"] = material_results
        
        # Tooling generation
        self.current_stage = ManufacturingStage.TOOLING_GENERATION
        tooling_results = self._generate_tooling(design_data, material_results)
        results["tooling"] = tooling_results
        
        # Fabrication instructions
        self.current_stage = ManufacturingStage.FABRICATION
        fabrication_results = self._generate_fabrication_instructions(
            design_data, material_results, tooling_results
        )
        results["fabrication"] = fabrication_results
        
        # Assembly instructions
        self.current_stage = ManufacturingStage.ASSEMBLY
        assembly_results = self._generate_assembly_instructions(
            design_data, fabrication_results
        )
        results["assembly"] = assembly_results
        
        # Quality control procedures
        self.current_stage = ManufacturingStage.QUALITY_CONTROL
        qc_results = self._generate_quality_control_procedures(design_data)
        results["quality_control"] = qc_results
        
        # Save manufacturing package
        package_path = self._save_manufacturing_package(results)
        results["package_path"] = package_path
        
        return results
    
    def _validate_design(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the design for manufacturability."""
        # Simple validation logic
        valid = True
        issues = []
        
        # Check for required design elements
        required_elements = ["airframe_type", "dimensions", "materials"]
        for element in required_elements:
            if element not in design_data:
                valid = False
                issues.append(f"Missing required element: {element}")
        
        return {
            "valid": valid,
            "issues": issues
        }
    
    def _select_materials(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate materials based on design requirements."""
        materials = {}
        
        # Get material requirements from design
        material_reqs = design_data.get("materials", {})
        
        # Map requirements to specific materials
        for component, requirements in material_reqs.items():
            if "type" in requirements:
                if requirements["type"] == "composite":
                    materials[component] = {
                        "material": "carbon_fiber_composite",
                        "properties": {
                            "density": 1.6,  # g/cm³
                            "tensile_strength": 3500,  # MPa
                            "process": ManufacturingProcess.COMPOSITE_LAYUP.value
                        }
                    }
                elif requirements["type"] == "metal":
                    materials[component] = {
                        "material": "titanium_alloy",
                        "properties": {
                            "density": 4.5,  # g/cm³
                            "tensile_strength": 1000,  # MPa
                            "process": ManufacturingProcess.SUBTRACTIVE.value
                        }
                    }
                elif requirements["type"] == "polymer":
                    materials[component] = {
                        "material": "high_performance_thermoplastic",
                        "properties": {
                            "density": 1.2,  # g/cm³
                            "tensile_strength": 85,  # MPa
                            "process": ManufacturingProcess.INJECTION_MOLDING.value
                        }
                    }
        
        return materials
    
    def _generate_tooling(self, design_data: Dict[str, Any], 
                         materials: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tooling requirements for manufacturing."""
        tooling = {}
        
        # Determine tooling based on materials and processes
        for component, material_info in materials.items():
            process = material_info.get("properties", {}).get("process")
            
            if process == ManufacturingProcess.COMPOSITE_LAYUP.value:
                tooling[component] = {
                    "tool_type": "mold",
                    "material": "invar",
                    "temperature_rating": 180  # °C
                }
            elif process == ManufacturingProcess.SUBTRACTIVE.value:
                tooling[component] = {
                    "tool_type": "cnc",
                    "cutting_tools": ["end_mill", "ball_mill"],
                    "fixturing": "vacuum_table"
                }
            elif process == ManufacturingProcess.INJECTION_MOLDING.value:
                tooling[component] = {
                    "tool_type": "injection_mold",
                    "material": "tool_steel",
                    "temperature_rating": 300  # °C
                }
        
        return tooling
    
    def _generate_fabrication_instructions(self, design_data: Dict[str, Any],
                                          materials: Dict[str, Any],
                                          tooling: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fabrication instructions for each component."""
        fabrication = {}
        
        # Generate instructions based on process
        for component, material_info in materials.items():
            process = material_info.get("properties", {}).get("process")
            
            if process == ManufacturingProcess.COMPOSITE_LAYUP.value:
                fabrication[component] = {
                    "process": process,
                    "steps": [
                        "Prepare mold surface with release agent",
                        "Cut carbon fiber fabric according to layup schedule",
                        "Apply resin to fabric",
                        "Layup fabric according to orientation schedule",
                        "Apply vacuum bagging",
                        "Cure in autoclave at specified temperature and pressure"
                    ],
                    "parameters": {
                        "cure_temp": 180,  # °C
                        "cure_pressure": 7,  # bar
                        "cure_time": 120  # minutes
                    }
                }
            elif process == ManufacturingProcess.SUBTRACTIVE.value:
                fabrication[component] = {
                    "process": process,
                    "steps": [
                        "Load raw material block",
                        "Set up fixturing",
                        "Run roughing operation",
                        "Run finishing operation",
                        "Inspect dimensions"
                    ],
                    "parameters": {
                        "feed_rate": 1000,  # mm/min
                        "spindle_speed": 10000,  # rpm
                        "tool_path": f"{component}_toolpath.nc"
                    }
                }
        
        return fabrication
    
    def _generate_assembly_instructions(self, design_data: Dict[str, Any],
                                       fabrication: Dict[str, Any]) -> Dict[str, Any]:
        """Generate assembly instructions for the airframe."""
        # Simple assembly sequence
        assembly = {
            "sequence": [
                "Assemble main structural components",
                "Install control surfaces",
                "Install propulsion system",
                "Install avionics and sensors",
                "Install power system",
                "Perform functional tests"
            ],
            "fixtures": {
                "main_assembly_jig": {
                    "type": "positioning_fixture",
                    "tolerance": 0.1  # mm
                }
            }
        }
        
        return assembly
    
    def _generate_quality_control_procedures(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality control procedures for the manufacturing process."""
        qc = {
            "inspections": [
                {
                    "stage": "post_fabrication",
                    "method": "dimensional_inspection",
                    "parameters": {
                        "tolerance": 0.1  # mm
                    }
                },
                {
                    "stage": "post_assembly",
                    "method": "functional_test",
                    "parameters": {
                        "test_procedures": ["control_surface_actuation", "power_system"]
                    }
                },
                {
                    "stage": "final",
                    "method": "non_destructive_testing",
                    "parameters": {
                        "techniques": ["ultrasonic", "x_ray"]
                    }
                }
            ]
        }
        
        return qc
    
    def _save_manufacturing_package(self, results: Dict[str, Any]) -> str:
        """Save the manufacturing package to disk."""
        # Create a unique filename based on timestamp
        import time
        timestamp = int(time.time())
        filename = f"manufacturing_package_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the package
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        return filepath