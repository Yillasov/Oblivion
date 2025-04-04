from typing import Dict, Any, List, Optional
import numpy as np
import os
import json
import logging as logger 
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
        try:
            # Try to import biomimetic materials if available
            from src.manufacturing.materials.biomimetic_materials import BiomimeticMaterialSelector
            
            # Use biomimetic material selector if available
            if not hasattr(self, 'material_selector') or not self.material_selector:
                self.material_selector = BiomimeticMaterialSelector(self.hardware_interface)
            
            # Extract requirements from design data
            requirements = design_data.get("requirements", {})
            stress_data = design_data.get("stress_analysis", {})
            geometry = design_data.get("geometry")
            
            # Determine biomimetic preference based on design goals
            biomimetic_preference = 0.5  # Default balanced approach
            if "performance_priority" in design_data:
                if design_data["performance_priority"] == "weight":
                    biomimetic_preference = 0.7  # Higher preference for biomimetic
                elif design_data["performance_priority"] == "cost":
                    biomimetic_preference = 0.3  # Lower preference for biomimetic
            
            # Select biomimetic materials
            if hasattr(self.material_selector, 'select_biomimetic_materials'):
                materials = self.material_selector.select_biomimetic_materials(
                    geometry, stress_data, requirements, biomimetic_preference
                )
            else:
                # Fall back to standard material selection
                materials = self.material_selector.select_materials(
                    geometry, stress_data, requirements
                )
            
            # Process anisotropic properties if needed
            self._process_anisotropic_properties(materials, design_data)
            
            # Simulate composite materials if needed
            if any(mat.get("material", "").endswith("composite") for mat in materials.values()):
                self._simulate_composite_materials(materials, design_data)
            
            return materials
        except ImportError:
            # Fall back to original implementation if biomimetic module not available
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
            
            # Simulate composite materials if needed
            if any(mat.get("material", "").endswith("composite") for mat in materials.values()):
                self._simulate_composite_materials(materials, design_data)
                
            return materials
    
    def _process_anisotropic_properties(self, materials: Dict[str, Any], 
                                   design_data: Dict[str, Any]) -> None:
        """Process anisotropic material properties."""
        try:
            from src.manufacturing.materials.anisotropic_materials import get_anisotropic_model, AnisotropyType
            
            # Get anisotropic model
            aniso_model = get_anisotropic_model()
            
            # Process each material
            for component, material_info in materials.items():
                material_name = material_info.get("material", "")
                
                # Skip if not a material that needs anisotropic processing
                if not any(keyword in material_name for keyword in 
                          ["composite", "fiber", "wood", "crystal", "anisotropic"]):
                    continue
                
                # Get anisotropic properties
                aniso_props = aniso_model.get_material(material_name)
                if not aniso_props:
                    # Try to find a similar material
                    if "carbon_fiber" in material_name:
                        aniso_props = aniso_model.get_material("carbon_fiber_composite")
                    elif "titanium" in material_name:
                        aniso_props = aniso_model.get_material("titanium_alloy")
                    else:
                        continue
                
                # Get load conditions for this component
                load_conditions = design_data.get("load_conditions", {}).get(component, {})
                
                # Calculate stiffness matrix
                stiffness_matrix = aniso_model.calculate_stiffness_matrix(
                    material_name if aniso_props else 
                    "carbon_fiber_composite" if "carbon_fiber" in material_name else
                    "titanium_alloy"
                )
                
                # Calculate thermal effects if temperature data is available
                temperature_change = 0.0
                if "operating_temperature" in design_data:
                    reference_temp = 25.0  # °C
                    operating_temp = design_data["operating_temperature"]
                    temperature_change = operating_temp - reference_temp
                
                # Add anisotropic properties to material info
                material_info["properties"]["anisotropic"] = True
                material_info["properties"]["anisotropy_type"] = (
                    aniso_props.anisotropy_type.value if aniso_props else "unknown"
                )
                
                # Add principal material directions if available
                if aniso_props and aniso_props.material_directions:
                    material_info["properties"]["principal_directions"] = [
                        dir.tolist() for dir in aniso_props.material_directions
                    ]
                
                # Add stiffness matrix (simplified for storage)
                material_info["properties"]["stiffness_matrix_diagonal"] = np.diag(stiffness_matrix).tolist()
                
                # Calculate and add thermal strain if temperature change is significant
                if abs(temperature_change) > 1.0 and aniso_props:
                    thermal_strain = aniso_model.calculate_thermal_strain(
                        material_name if aniso_props else 
                        "carbon_fiber_composite" if "carbon_fiber" in material_name else
                        "titanium_alloy",
                        temperature_change
                    )
                    material_info["properties"]["thermal_strain"] = thermal_strain.tolist()
                
                logger.info(f"Processed anisotropic properties for {component} ({material_name})")
                
        except ImportError as e:
            logger.warning(f"Could not process anisotropic properties: {e}")
    
    def _simulate_composite_materials(self, materials: Dict[str, Any], 
                                   design_data: Dict[str, Any]) -> None:
        """Simulate composite materials to validate performance."""
        try:
            from src.manufacturing.materials.composite_simulation import CompositeSimulator, CompositeType
            
            # Create simulator
            simulator = CompositeSimulator()
            
            # Process each composite material
            for component, material_info in materials.items():
                material_name = material_info.get("material", "")
                
                if not material_name.endswith("composite"):
                    continue
                    
                # Extract load conditions for this component
                load_conditions = design_data.get("load_conditions", {}).get(component, {})
                if not load_conditions:
                    # Default load conditions if none specified
                    load_conditions = {
                        "tension_x": 10.0,  # N/mm
                        "tension_y": 5.0,   # N/mm
                        "shear_xy": 2.0,    # N/mm
                        "bending_moment": 1.0,  # N·mm/mm
                        "span": 100.0       # mm
                    }
                
                # Create composite layup based on material type
                if "flexible" in material_name or "biomimetic" in material_name:
                    layup = [
                        {"material": "flexible_composite", "thickness": 0.5, "orientation": 0},
                        {"material": "aramid_fiber", "thickness": 0.2, "orientation": 45},
                        {"material": "flexible_composite", "thickness": 0.5, "orientation": 90}
                    ]
                    composite_type = CompositeType.FLEXIBLE
                else:
                    layup = [
                        {"material": "carbon_fiber", "thickness": 0.125, "orientation": 0},
                        {"material": "carbon_fiber", "thickness": 0.125, "orientation": 45},
                        {"material": "carbon_fiber", "thickness": 0.125, "orientation": 90},
                        {"material": "carbon_fiber", "thickness": 0.125, "orientation": -45}
                    ]
                    composite_type = CompositeType.LAMINATE
                
                # Create composite and simulate
                composite = simulator.create_composite(layup, composite_type)
                
                # Simulate mechanical response
                mech_results = simulator.simulate_mechanical_response(
                    load_conditions, 
                    temperature=design_data.get("operating_temperature", 25.0)
                )
                
                # Simulate impact if needed
                if "impact_energy" in design_data:
                    impact_results = simulator.simulate_impact(
                        design_data["impact_energy"],
                        impact_location=(0, 0)
                    )
                    
                    # Add impact results to material properties
                    material_info["properties"]["impact_resistance"] = 1.0 - impact_results["damage_ratio"]
                
                # Update material properties with simulation results
                material_info["properties"]["simulated"] = True
                material_info["properties"]["failure_index"] = mech_results["failure_index"]
                material_info["properties"]["max_deflection"] = mech_results["max_deflection"]
                material_info["properties"]["layup"] = layup
                material_info["properties"]["thickness"] = composite.total_thickness
                
                # Warn if failure predicted
                if mech_results["failure_predicted"]:
                    logger.warning(f"Potential failure predicted for {component} under specified loads")
                    
        except ImportError as e:
            logger.warning(f"Could not simulate composite materials: {e}")
    
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