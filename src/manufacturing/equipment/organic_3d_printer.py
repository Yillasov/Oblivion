#!/usr/bin/env python3
"""
Controller for 3D printers specialized in organic and biomimetic structures.
Provides interfaces for printing complex organic structures with various materials.
"""

import os
import sys
import time
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.manufacturing.equipment.printer_controller import PrinterController, EquipmentStatus

logger = get_logger("organic_3d_printer")


class PrintMaterial(Enum):
    """Materials available for organic 3D printing."""
    FLEXIBLE_POLYMER = "flexible_polymer"
    RIGID_POLYMER = "rigid_polymer"
    COMPOSITE_FIBER = "composite_fiber"
    GRADIENT_MATERIAL = "gradient_material"
    BIOMIMETIC_RESIN = "biomimetic_resin"
    CONDUCTIVE_POLYMER = "conductive_polymer"


class PrinterMode(Enum):
    """Printing modes for organic structures."""
    SINGLE_MATERIAL = "single_material"
    MULTI_MATERIAL = "multi_material"
    VARIABLE_DENSITY = "variable_density"
    MICROSTRUCTURE = "microstructure"
    GRADIENT_PROPERTIES = "gradient_properties"
    VEIN_REINFORCEMENT = "vein_reinforcement"


class Organic3DPrinter(PrinterController):
    """Controller for 3D printers specialized in organic structures."""
    
    def __init__(self, printer_id: str = "organic_printer_01"):
        """Initialize organic 3D printer controller."""
        super().__init__(printer_id)
        self.materials = {}
        self.current_mode = PrinterMode.SINGLE_MATERIAL
        self.current_material = None
        self.print_resolution = 0.1  # mm
        self.temperature_zones = {
            "bed": 60.0,
            "extruder_1": 210.0,
            "extruder_2": 210.0,
            "chamber": 45.0
        }
        self.max_build_volume = (250.0, 250.0, 200.0)  # mm
        
        # Initialize available materials
        self._initialize_materials()
        
        logger.info(f"Initialized organic 3D printer controller: {printer_id}")
    
    def _initialize_materials(self) -> None:
        """Initialize available printing materials."""
        self.materials = {
            PrintMaterial.FLEXIBLE_POLYMER: {
                "name": "FlexiPoly",
                "temperature": 210.0,
                "density": 1.1,
                "elastic_modulus": 12.0,  # MPa
                "elongation": 350.0,  # %
                "color": "translucent"
            },
            PrintMaterial.RIGID_POLYMER: {
                "name": "RigidPoly",
                "temperature": 230.0,
                "density": 1.25,
                "elastic_modulus": 2200.0,  # MPa
                "elongation": 6.0,  # %
                "color": "white"
            },
            PrintMaterial.COMPOSITE_FIBER: {
                "name": "CompFiber",
                "temperature": 245.0,
                "density": 1.4,
                "elastic_modulus": 5500.0,  # MPa
                "elongation": 2.5,  # %
                "color": "black"
            },
            PrintMaterial.GRADIENT_MATERIAL: {
                "name": "GradMat",
                "temperature": 225.0,
                "density": 1.2,
                "elastic_modulus": "variable",
                "elongation": "variable",
                "color": "variable"
            },
            PrintMaterial.BIOMIMETIC_RESIN: {
                "name": "BioResin",
                "temperature": 200.0,
                "density": 1.05,
                "elastic_modulus": "variable",
                "elongation": 280.0,  # %
                "color": "amber"
            },
            PrintMaterial.CONDUCTIVE_POLYMER: {
                "name": "CondPoly",
                "temperature": 220.0,
                "density": 1.3,
                "elastic_modulus": 1800.0,  # MPa
                "elongation": 5.0,  # %
                "conductivity": 1.0e-4,  # S/cm
                "color": "black"
            }
        }
    
    def initialize(self) -> bool:
        """Initialize organic 3D printer."""
        try:
            self.status = EquipmentStatus(
                operational=True,
                temperature=25.0,
                power_state="standby",
                last_maintenance=datetime.now()
            )
            
            # Set default material
            self.current_material = PrintMaterial.FLEXIBLE_POLYMER
            
            # Set default mode
            self.current_mode = PrinterMode.SINGLE_MATERIAL
            
            logger.info(f"Organic 3D printer {self.equipment_id} initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize organic 3D printer: {e}")
            self.status.error_code = 1
            self.status.error_message = str(e)
            return False
    
    def set_print_mode(self, mode: PrinterMode) -> bool:
        """
        Set the printing mode.
        
        Args:
            mode: The printing mode to set
            
        Returns:
            bool: Success status
        """
        if not self.status.operational:
            logger.error("Cannot set print mode: printer not operational")
            return False
        
        self.current_mode = mode
        logger.info(f"Set print mode to {mode.value}")
        return True
    
    def load_material(self, material: PrintMaterial, slot: int = 1) -> bool:
        """
        Load material into a specific slot.
        
        Args:
            material: The material to load
            slot: The slot to load the material into (1 or 2)
            
        Returns:
            bool: Success status
        """
        if not self.status.operational:
            logger.error("Cannot load material: printer not operational")
            return False
        
        if slot not in [1, 2]:
            logger.error(f"Invalid slot: {slot}")
            return False
        
        if material not in self.materials:
            logger.error(f"Unknown material: {material}")
            return False
        
        # Set material temperature
        material_temp = self.materials[material]["temperature"]
        self.temperature_zones[f"extruder_{slot}"] = material_temp
        
        if slot == 1:
            self.current_material = material
        
        logger.info(f"Loaded {material.value} into slot {slot}")
        return True
    
    def set_resolution(self, resolution: float) -> bool:
        """
        Set the print resolution.
        
        Args:
            resolution: The resolution in mm
            
        Returns:
            bool: Success status
        """
        if not self.status.operational:
            logger.error("Cannot set resolution: printer not operational")
            return False
        
        if resolution < 0.01 or resolution > 0.5:
            logger.error(f"Resolution out of range: {resolution}")
            return False
        
        self.print_resolution = resolution
        logger.info(f"Set print resolution to {resolution} mm")
        return True
    
    def print_organic_structure(self, model_path: str, 
                              print_config: Dict[str, Any]) -> bool:
        """
        Print an organic structure.
        
        Args:
            model_path: Path to the model file
            print_config: Printing configuration
            
        Returns:
            bool: Success status
        """
        if not self.status.operational:
            logger.error("Cannot print: printer not operational")
            return False
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Extract configuration
        layer_height = print_config.get("layer_height", 0.1)
        infill_density = print_config.get("infill_density", 20)
        print_speed = print_config.get("print_speed", 50)
        
        # Set printer status to printing
        self.status.power_state = "printing"
        
        # Log print start
        logger.info(f"Starting organic print: {os.path.basename(model_path)}")
        logger.info(f"Mode: {self.current_mode.value}, Material: {self.current_material.value}")
        logger.info(f"Layer height: {layer_height}mm, Infill: {infill_density}%, Speed: {print_speed}mm/s")
        
        # Simulate printing process
        try:
            # Heat up
            logger.info("Heating bed and extruders...")
            time.sleep(0.1)  # Simulate heating (would be longer in real implementation)
            
            # Print
            logger.info("Printing organic structure...")
            time.sleep(0.2)  # Simulate printing (would be longer in real implementation)
            
            # Cool down
            logger.info("Print complete, cooling down...")
            time.sleep(0.1)  # Simulate cooling (would be longer in real implementation)
            
            # Reset printer status
            self.status.power_state = "standby"
            
            logger.info("Print job completed successfully")
            return True
        except Exception as e:
            logger.error(f"Print failed: {e}")
            self.status.power_state = "error"
            self.status.error_code = 2
            self.status.error_message = str(e)
            return False
    
    def print_biomimetic_wing(self, wing_type: str, 
                            venation_pattern: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                            wing_properties: Dict[str, Any]) -> bool:
        """
        Print a biomimetic wing structure.
        
        Args:
            wing_type: Type of wing (e.g., "dragonfly", "bat")
            venation_pattern: List of vein segments as pairs of (x,y) coordinates
            wing_properties: Wing material and structural properties
            
        Returns:
            bool: Success status
        """
        if not self.status.operational:
            logger.error("Cannot print wing: printer not operational")
            return False
        
        # Set appropriate mode for wing printing
        if wing_type.lower() == "dragonfly":
            self.set_print_mode(PrinterMode.VEIN_REINFORCEMENT)
            self.load_material(PrintMaterial.FLEXIBLE_POLYMER, 1)
            self.load_material(PrintMaterial.COMPOSITE_FIBER, 2)
        elif wing_type.lower() == "bat":
            self.set_print_mode(PrinterMode.GRADIENT_PROPERTIES)
            self.load_material(PrintMaterial.GRADIENT_MATERIAL, 1)
        else:
            logger.error(f"Unknown wing type: {wing_type}")
            return False
        
        # Set printer status to printing
        self.status.power_state = "printing"
        
        # Log print start
        logger.info(f"Starting biomimetic wing print: {wing_type}")
        logger.info(f"Mode: {self.current_mode.value}, Material: {self.current_material.value}")
        logger.info(f"Venation pattern: {len(venation_pattern)} segments")
        
        # Simulate printing process
        try:
            # Heat up
            logger.info("Heating bed and extruders...")
            time.sleep(0.1)  # Simulate heating
            
            # Print membrane
            logger.info("Printing wing membrane...")
            time.sleep(0.1)  # Simulate printing
            
            # Print venation pattern
            logger.info("Printing venation pattern...")
            for i, (start, end) in enumerate(venation_pattern):
                logger.debug(f"Printing vein segment {i+1}: {start} to {end}")
                time.sleep(0.01)  # Simulate printing each vein
            
            # Apply anisotropic properties if specified
            if wing_properties.get("anisotropic", False):
                logger.info("Applying anisotropic material properties...")
                time.sleep(0.05)  # Simulate processing
            
            # Cool down
            logger.info("Wing print complete, cooling down...")
            time.sleep(0.1)  # Simulate cooling
            
            # Reset printer status
            self.status.power_state = "standby"
            
            logger.info("Biomimetic wing print completed successfully")
            return True
        except Exception as e:
            logger.error(f"Wing print failed: {e}")
            self.status.power_state = "error"
            self.status.error_code = 3
            self.status.error_message = str(e)
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the printer."""
        try:
            logger.info(f"Shutting down organic 3D printer: {self.equipment_id}")
            self.status.operational = False
            self.status.power_state = "off"
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown printer: {e}")
            return False