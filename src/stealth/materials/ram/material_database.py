#!/usr/bin/env python3
"""
Database of Radar Absorbent Materials (RAM).
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from src.stealth.materials.ram.ram_material import RAMMaterial
from src.core.utils.logging_framework import get_logger

logger = get_logger("ram_database")


class RAMMaterialDatabase:
    """Database for managing and retrieving RAM material properties."""
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize the RAM material database.
        
        Args:
            database_path: Path to the material database file
        """
        self.materials: Dict[str, RAMMaterial] = {}
        self.database_path = database_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "data", 
            "ram_materials.json"
        )
        self.load_database()
        
    def load_database(self) -> bool:
        """
        Load material database from file.
        
        Returns:
            Success status
        """
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r') as f:
                    data = json.load(f)
                
                for material_id, material_data in data.items():
                    # Convert frequency response strings back to proper keys
                    freq_response = {str(k): v for k, v in material_data.get("frequency_response", {}).items()}
                    
                    # Create RAMMaterial instance
                    self.materials[material_id] = RAMMaterial(
                        name=material_data.get("name", "Unknown"),
                        density=material_data.get("density", 0.0),
                        thickness=material_data.get("thickness", 0.0),
                        frequency_response=freq_response,
                        temperature_range=(
                            material_data.get("temperature_range", [-40.0, 80.0])[0],
                            material_data.get("temperature_range", [-40.0, 80.0])[1]
                        ),
                        weather_resistance=material_data.get("weather_resistance", 0.5),
                        durability=material_data.get("durability", 0.5),
                        cost_factor=material_data.get("cost_factor", 1.0)
                    )
                return True
            else:
                # Initialize with default materials if database doesn't exist
                self._initialize_default_materials()
                self.save_database()
                return True
        except Exception as e:
            print(f"Error loading RAM material database: {e}")
            self._initialize_default_materials()
            return False
    
    def save_database(self) -> bool:
        """
        Save material database to file.
        
        Returns:
            Success status
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            
            # Convert materials to serializable format
            data = {}
            for material_id, material in self.materials.items():
                material_dict = asdict(material)
                # Convert tuple to list for JSON serialization
                material_dict["temperature_range"] = list(material_dict["temperature_range"])
                data[material_id] = material_dict
                
            with open(self.database_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving RAM material database: {e}")
            return False
    
    def _initialize_default_materials(self) -> None:
        """Initialize the database with default RAM materials."""
        self.materials = {
            "ferrite_composite": RAMMaterial(
                name="Ferrite Composite",
                density=2200.0,
                thickness=3.0,
                frequency_response={
                    "1.0": 10.0,  # 1 GHz: 10 dB attenuation
                    "5.0": 15.0,  # 5 GHz: 15 dB attenuation
                    "10.0": 12.0,  # 10 GHz: 12 dB attenuation
                    "15.0": 8.0,   # 15 GHz: 8 dB attenuation
                },
                temperature_range=(-40.0, 120.0),
                weather_resistance=0.7,
                durability=0.8,
                cost_factor=1.0
            ),
            "advanced_composite": RAMMaterial(
                name="Advanced Composite",
                density=1800.0,
                thickness=5.0,
                frequency_response={
                    "1.0": 15.0,   # 1 GHz: 15 dB attenuation
                    "5.0": 22.0,   # 5 GHz: 22 dB attenuation
                    "10.0": 20.0,  # 10 GHz: 20 dB attenuation
                    "15.0": 18.0,  # 15 GHz: 18 dB attenuation
                    "20.0": 15.0,  # 20 GHz: 15 dB attenuation
                },
                temperature_range=(-60.0, 150.0),
                weather_resistance=0.9,
                durability=0.85,
                cost_factor=2.5
            ),
            "carbon_nanotube": RAMMaterial(
                name="Carbon Nanotube Composite",
                density=1500.0,
                thickness=2.0,
                frequency_response={
                    "1.0": 18.0,   # 1 GHz: 18 dB attenuation
                    "5.0": 25.0,   # 5 GHz: 25 dB attenuation
                    "10.0": 30.0,  # 10 GHz: 30 dB attenuation
                    "15.0": 28.0,  # 15 GHz: 28 dB attenuation
                    "20.0": 22.0,  # 20 GHz: 22 dB attenuation
                },
                temperature_range=(-80.0, 200.0),
                weather_resistance=0.95,
                durability=0.9,
                cost_factor=5.0
            ),
            "metamaterial_absorber": RAMMaterial(
                name="Metamaterial Absorber",
                density=1200.0,
                thickness=1.5,
                frequency_response={
                    "1.0": 20.0,   # 1 GHz: 20 dB attenuation
                    "5.0": 28.0,   # 5 GHz: 28 dB attenuation
                    "10.0": 35.0,  # 10 GHz: 35 dB attenuation
                    "15.0": 32.0,  # 15 GHz: 32 dB attenuation
                    "20.0": 30.0,  # 20 GHz: 30 dB attenuation
                    "30.0": 25.0,  # 30 GHz: 25 dB attenuation
                },
                temperature_range=(-100.0, 250.0),
                weather_resistance=0.98,
                durability=0.85,
                cost_factor=8.0
            ),
            "magnetic_composite": RAMMaterial(
                name="Magnetic Composite",
                density=2500.0,
                thickness=4.0,
                frequency_response={
                    "0.5": 12.0,   # 0.5 GHz: 12 dB attenuation
                    "1.0": 16.0,   # 1 GHz: 16 dB attenuation
                    "2.0": 20.0,   # 2 GHz: 20 dB attenuation
                    "5.0": 18.0,   # 5 GHz: 18 dB attenuation
                    "10.0": 15.0,  # 10 GHz: 15 dB attenuation
                },
                temperature_range=(-20.0, 180.0),
                weather_resistance=0.75,
                durability=0.9,
                cost_factor=3.0
            )
        }
    
    def get_material(self, material_id: str) -> Optional[RAMMaterial]:
        """
        Get material by ID.
        
        Args:
            material_id: Material identifier
            
        Returns:
            RAMMaterial or None if not found
        """
        return self.materials.get(material_id)
    
    def add_material(self, material_id: str, material: RAMMaterial) -> bool:
        """
        Add a new material to the database.
        
        Args:
            material_id: Material identifier
            material: Material properties
            
        Returns:
            Success status
        """
        if material_id in self.materials:
            return False
            
        self.materials[material_id] = material
        self.save_database()
        return True
    
    def update_material(self, material_id: str, material: RAMMaterial) -> bool:
        """
        Update an existing material.
        
        Args:
            material_id: Material identifier
            material: Updated material properties
            
        Returns:
            Success status
        """
        if material_id not in self.materials:
            return False
            
        self.materials[material_id] = material
        self.save_database()
        return True
    
    def delete_material(self, material_id: str) -> bool:
        """
        Delete a material from the database.
        
        Args:
            material_id: Material identifier
            
        Returns:
            Success status
        """
        if material_id not in self.materials:
            return False
            
        del self.materials[material_id]
        self.save_database()
        return True
    
    def list_materials(self) -> List[str]:
        """
        List all material IDs.
        
        Returns:
            List of material IDs
        """
        return list(self.materials.keys())
    
    def get_material_properties(self, material_id: str) -> Dict[str, Any]:
        """
        Get material properties as a dictionary.
        
        Args:
            material_id: Material identifier
            
        Returns:
            Dictionary of material properties or empty dict if not found
        """
        material = self.get_material(material_id)
        if not material:
            return {}
            
        return asdict(material)
    
    def find_materials_by_property(self, 
                                  property_name: str, 
                                  min_value: Optional[float] = None,
                                  max_value: Optional[float] = None) -> List[str]:
        """
        Find materials by property value range.
        
        Args:
            property_name: Name of the property to search
            min_value: Minimum property value (inclusive)
            max_value: Maximum property value (inclusive)
            
        Returns:
            List of material IDs matching the criteria
        """
        results = []
        
        for material_id, material in self.materials.items():
            material_dict = asdict(material)
            
            # Handle special case for temperature range
            if property_name == "temperature_min":
                value = material.temperature_range[0]
            elif property_name == "temperature_max":
                value = material.temperature_range[1]
            # Handle special case for frequency response
            elif property_name.startswith("frequency_"):
                freq = property_name.split("_")[1]
                value = material.frequency_response.get(freq, 0.0)
            else:
                value = material_dict.get(property_name)
                
            if value is None:
                continue
                
            if min_value is not None and value < min_value:
                continue
                
            if max_value is not None and value > max_value:
                continue
                
            results.append(material_id)
            
        return results
    
    def get_optimal_material(self, 
                           frequency_ghz: float, 
                           environmental_conditions: Dict[str, float]) -> str:
        """
        Get the optimal material for given frequency and environmental conditions.
        
        Args:
            frequency_ghz: Target frequency in GHz
            environmental_conditions: Environmental conditions
            
        Returns:
            ID of the optimal material
        """
        best_material_id = None
        best_score = -1.0
        
        temperature = environmental_conditions.get("temperature", 20.0)
        humidity = environmental_conditions.get("humidity", 50.0)
        
        for material_id, material in self.materials.items():
            # Check if temperature is within range
            if temperature < material.temperature_range[0] or temperature > material.temperature_range[1]:
                continue
                
            # Find closest frequency in the response data
            closest_freq = min(material.frequency_response.keys(), 
                              key=lambda x: abs(float(x) - frequency_ghz))
            
            # Get attenuation at that frequency
            attenuation = material.frequency_response.get(closest_freq, 0.0)
            
            # Calculate score based on attenuation, weather resistance and durability
            weather_factor = 1.0
            if humidity > 80.0:
                weather_factor = material.weather_resistance
                
            score = attenuation * weather_factor * material.durability
            
            if score > best_score:
                best_score = score
                best_material_id = material_id
                
        return best_material_id or "ferrite_composite"  # Default if no match
