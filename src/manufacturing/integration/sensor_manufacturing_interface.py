"""
Sensor Manufacturing Interface
Integrates sensor data with the manufacturing pipeline for real-time feedback and quality control.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from src.core.sdk.sensor_connector import get_sensor_connector
from src.core.sdk.sensor_config_manager import get_config_manager
from src.manufacturing.pipeline import ManufacturingPipeline, ManufacturingStage
from src.simulation.sensors.sensor_framework import SensorType, SensorConfig
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.core.neuromorphic.sensor_algorithms import create_algorithm_for_sensor

logger = logging.getLogger(__name__)

class SensorManufacturingInterface:
    """Interface between sensors and manufacturing pipeline."""
    
    def __init__(self, manufacturing_pipeline: Optional[ManufacturingPipeline] = None):
        self.sensor_connector = get_sensor_connector()
        self.config_manager = get_config_manager()
        self.manufacturing_pipeline = manufacturing_pipeline or ManufacturingPipeline({})
        self.quality_thresholds = {
            ManufacturingStage.MATERIAL_SELECTION: 0.8,
            ManufacturingStage.FABRICATION: 0.9,
            ManufacturingStage.QUALITY_CONTROL: 0.95
        }
        
        # Initialize neuromorphic system
        self.neuromorphic_system = NeuromorphicSystem()
        self.neuromorphic_system.initialize()
        
        # Initialize neuromorphic algorithms for each sensor type
        self.sensor_algorithms = {
            SensorType.TERAHERTZ: create_algorithm_for_sensor(SensorType.TERAHERTZ),
            SensorType.NEUROMORPHIC_VISION: create_algorithm_for_sensor(SensorType.NEUROMORPHIC_VISION)
        }
    
    def setup_manufacturing_sensors(self) -> bool:
        """Set up sensors for manufacturing monitoring."""
        try:
            # Add material analysis sensor
            material_sensor_config = SensorConfig(
                type=SensorType.TERAHERTZ,
                name="material_analyzer",
                update_rate=5.0,
                fov_horizontal=60.0,
                fov_vertical=60.0,
                max_range=1.0,
                accuracy=0.98,
                noise_factor=0.01
            )
            self.sensor_connector.add_sensor(material_sensor_config)
            
            # Add quality control sensor
            qc_sensor_config = SensorConfig(
                type=SensorType.NEUROMORPHIC_VISION,
                name="quality_inspector",
                update_rate=30.0,
                fov_horizontal=90.0,
                fov_vertical=90.0,
                max_range=2.0,
                accuracy=0.99,
                noise_factor=0.005
            )
            self.sensor_connector.add_sensor(qc_sensor_config)
            
            return True
        except Exception as e:
            logger.error(f"Failed to setup manufacturing sensors: {e}")
            return False
    
    def process_manufacturing_stage(self, stage: ManufacturingStage, 
                                  data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor data for current manufacturing stage."""
        # Get relevant sensor data
        sensor_data = self.sensor_connector.get_all_sensor_data()
        
        # Process sensor data using neuromorphic algorithms
        processed_data = {}
        for sensor_name, raw_data in sensor_data.items():
            sensor_type = raw_data.get('type')
            if sensor_type in self.sensor_algorithms:
                algorithm = self.sensor_algorithms[sensor_type]
                processed_data[sensor_name] = algorithm.process(raw_data)
        
        # Process based on stage
        if stage == ManufacturingStage.MATERIAL_SELECTION:
            return self._process_material_selection(processed_data, data)
        elif stage == ManufacturingStage.FABRICATION:
            return self._process_fabrication(processed_data, data)
        elif stage == ManufacturingStage.QUALITY_CONTROL:
            return self._process_quality_control(processed_data, data)
        
        return {"status": "skipped", "stage": stage.name}
    
    def _process_material_selection(self, sensor_data: Dict[str, Dict[str, Any]], 
                                  stage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process material selection stage."""
        material_data = sensor_data.get("material_analyzer", {})
        if not material_data:
            return {"status": "error", "message": "No material sensor data"}
        
        # Analyze material properties
        material_quality = material_data.get("material_quality", 0.0)
        composition_match = material_data.get("composition_match", 0.0)
        
        quality_score = float((material_quality + composition_match) / 2)
        passed = quality_score >= self.quality_thresholds[ManufacturingStage.MATERIAL_SELECTION]
        
        # Real-time feedback adjustment
        self._adjust_process_based_on_quality(quality_score, ManufacturingStage.MATERIAL_SELECTION)
        
        return {
            "status": "passed" if passed else "failed",
            "quality_score": quality_score,
            "details": {
                "material_quality": material_quality,
                "composition_match": composition_match
            }
        }
    
    def _process_fabrication(self, sensor_data: Dict[str, Dict[str, Any]], 
                           stage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process fabrication stage."""
        # Get processed sensor data
        vision_data = self.sensor_connector.get_processed_data("quality_inspector")
        material_data = self.sensor_connector.get_processed_data("material_analyzer")
        
        if vision_data.size == 0 or material_data.size == 0:
            return {"status": "error", "message": "Insufficient sensor data"}
        
        # Analyze fabrication quality
        dimensional_accuracy = np.mean(vision_data)
        material_integrity = np.mean(material_data)
        
        quality_score = float((dimensional_accuracy + material_integrity) / 2)
        passed = quality_score >= self.quality_thresholds[ManufacturingStage.FABRICATION]
        
        # Real-time feedback adjustment
        self._adjust_process_based_on_quality(quality_score, ManufacturingStage.FABRICATION)
        
        return {
            "status": "passed" if passed else "failed",
            "quality_score": quality_score,
            "details": {
                "dimensional_accuracy": float(dimensional_accuracy),
                "material_integrity": float(material_integrity)
            }
        }
    
    def _process_quality_control(self, sensor_data: Dict[str, Dict[str, Any]], 
                               stage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process quality control stage."""
        # Get fusion data for comprehensive analysis
        fusion_data = self.sensor_connector.get_fusion_data()
        
        if not fusion_data:
            return {"status": "error", "message": "No fusion data available"}
        
        # Analyze overall quality
        surface_quality = fusion_data.get("surface_quality", 0.0)
        structural_integrity = fusion_data.get("structural_integrity", 0.0)
        dimensional_compliance = fusion_data.get("dimensional_compliance", 0.0)
        
        quality_score = float((surface_quality + structural_integrity + dimensional_compliance) / 3)
        passed = quality_score >= self.quality_thresholds[ManufacturingStage.QUALITY_CONTROL]
        
        # Real-time feedback adjustment
        self._adjust_process_based_on_quality(quality_score, ManufacturingStage.QUALITY_CONTROL)
        
        return {
            "status": "passed" if passed else "failed",
            "quality_score": quality_score,
            "details": {
                "surface_quality": surface_quality,
                "structural_integrity": structural_integrity,
                "dimensional_compliance": dimensional_compliance
            }
        }
    
    def _adjust_process_based_on_quality(self, quality_score: float, stage: ManufacturingStage):
        """Adjust manufacturing process based on quality score."""
        logger.info(f"Adjusting process for {stage.name} with quality score: {quality_score}")
        # Implement specific adjustments based on quality score and stage
        if quality_score < self.quality_thresholds[stage]:
            logger.warning(f"Quality below threshold for {stage.name}. Initiating corrective actions.")
            # Example: Adjust machine parameters, alert operators, etc.
            # This is a placeholder for actual adjustment logic
            pass

# Create a singleton instance
_manufacturing_interface = None

def get_manufacturing_interface() -> SensorManufacturingInterface:
    """Get the global manufacturing interface instance."""
    global _manufacturing_interface
    if _manufacturing_interface is None:
        _manufacturing_interface = SensorManufacturingInterface()
    return _manufacturing_interface