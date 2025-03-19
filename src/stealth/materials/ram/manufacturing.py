"""
Manufacturing specifications for Radar-Absorbent Materials (RAM).
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.stealth.materials.ram.ram_system import RAMSystem, RAMMaterial


@dataclass
class RAMManufacturingSpec:
    """Manufacturing specifications for RAM materials."""
    material_id: str
    material_name: str
    thickness_mm: float
    density_kg_m3: float
    temperature_requirements: Dict[str, float]
    process_type: str
    curing_time_hours: float
    quality_checks: List[str]
    estimated_production_time_hours: float


class RAMManufacturingGenerator:
    """Generator for RAM manufacturing specifications."""
    
    def __init__(self, ram_system: RAMSystem):
        """
        Initialize the manufacturing specifications generator.
        
        Args:
            ram_system: RAM system to generate specifications for
        """
        self.ram_system = ram_system
        
    def generate_specs(self) -> Dict[str, Any]:
        """
        Generate manufacturing specifications for the current RAM material.
        
        Returns:
            Dictionary with manufacturing specifications
        """
        if not self.ram_system.active_material:
            return {"error": "No active material selected"}
            
        material = self.ram_system.active_material
        
        # Determine process type based on material properties
        process_type = self._determine_process_type(material)
        
        # Calculate estimated production time
        production_time = self._calculate_production_time(material, process_type)
        
        # Generate quality checks
        quality_checks = self._generate_quality_checks(material)
        
        # Generate temperature requirements
        temp_requirements = {
            "curing_min_temp": material.temperature_range[0] + 20,
            "curing_max_temp": material.temperature_range[1] - 10,
            "storage_temp": 20.0
        }
        
        # Create manufacturing spec
        spec = RAMManufacturingSpec(
            material_id=self._get_material_id(material),
            material_name=material.name,
            thickness_mm=material.thickness,
            density_kg_m3=material.density,
            temperature_requirements=temp_requirements,
            process_type=process_type,
            curing_time_hours=self._calculate_curing_time(material),
            quality_checks=quality_checks,
            estimated_production_time_hours=production_time
        )
        
        return {
            "material_specs": {
                "id": spec.material_id,
                "name": spec.material_name,
                "thickness_mm": spec.thickness_mm,
                "density_kg_m3": spec.density_kg_m3
            },
            "manufacturing_process": {
                "process_type": spec.process_type,
                "curing_time_hours": spec.curing_time_hours,
                "temperature_requirements": spec.temperature_requirements
            },
            "quality_control": {
                "checks": spec.quality_checks
            },
            "production": {
                "estimated_time_hours": spec.estimated_production_time_hours
            }
        }
        
    def _get_material_id(self, material: RAMMaterial) -> str:
        """Get material ID from the database."""
        for material_id, mat in self.ram_system.material_database.materials.items():
            if mat.name == material.name:
                return material_id
        return "unknown"
        
    def _determine_process_type(self, material: RAMMaterial) -> str:
        """Determine manufacturing process type based on material properties."""
        if "nanotube" in material.name.lower():
            return "chemical_vapor_deposition"
        elif "metamaterial" in material.name.lower():
            return "precision_lithography"
        elif "composite" in material.name.lower():
            return "composite_layup"
        else:
            return "standard_molding"
            
    def _calculate_production_time(self, material: RAMMaterial, process_type: str) -> float:
        """Calculate estimated production time in hours."""
        base_time = 4.0  # Base production time in hours
        
        # Adjust based on process type
        if process_type == "chemical_vapor_deposition":
            base_time *= 2.5
        elif process_type == "precision_lithography":
            base_time *= 3.0
        elif process_type == "composite_layup":
            base_time *= 1.5
            
        # Adjust based on material complexity (cost factor as proxy)
        base_time *= material.cost_factor
        
        # Adjust based on thickness
        base_time *= (material.thickness / 3.0)
        
        return round(base_time, 1)
        
    def _calculate_curing_time(self, material: RAMMaterial) -> float:
        """Calculate curing time in hours."""
        # Base curing time depends on material type
        if "nanotube" in material.name.lower():
            base_time = 8.0
        elif "metamaterial" in material.name.lower():
            base_time = 6.0
        elif "composite" in material.name.lower():
            base_time = 4.0
        else:
            base_time = 3.0
            
        # Adjust based on thickness
        return round(base_time * (material.thickness / 3.0), 1)
        
    def _generate_quality_checks(self, material: RAMMaterial) -> List[str]:
        """Generate list of quality checks based on material properties."""
        checks = [
            "thickness_measurement",
            "density_verification",
            "visual_inspection"
        ]
        
        # Add material-specific checks
        if material.cost_factor > 3.0:
            checks.append("microscopic_analysis")
            
        if "nanotube" in material.name.lower() or "metamaterial" in material.name.lower():
            checks.append("electron_microscopy")
            
        # Always add radar absorption test
        checks.append("radar_absorption_test")
        
        # Add weather resistance test if applicable
        if material.weather_resistance > 0.8:
            checks.append("environmental_exposure_test")
            
        return checks
        
    def export_manufacturing_data(self, output_dir: str) -> str:
        """
        Export manufacturing data to a file.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to the output file
        """
        import os
        import json
        
        specs = self.generate_specs()
        if "error" in specs:
            return f"Error: {specs['error']}"
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        material_id = specs["material_specs"]["id"]
        output_file = os.path.join(output_dir, f"{material_id}_manufacturing.json")
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(specs, f, indent=2)
            
        return output_file