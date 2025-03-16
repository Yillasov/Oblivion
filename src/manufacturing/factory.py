from typing import Dict, Any
from .pipeline import ManufacturingPipeline

class ManufacturingPipelineFactory:
    """Factory for creating manufacturing pipelines for different airframe types."""
    
    @staticmethod
    def create_pipeline(airframe_type: str, config: Dict[str, Any]) -> ManufacturingPipeline:
        """Create a manufacturing pipeline for the specified airframe type."""
        # Base configuration
        pipeline_config = config.copy()
        
        # Add airframe-specific configurations
        if airframe_type == "morphing_wing":
            pipeline_config.update({
                "specialized_processes": ["shape_memory_alloy_integration", "flexible_skin_fabrication"]
            })
        elif airframe_type == "hypersonic":
            pipeline_config.update({
                "specialized_processes": ["high_temperature_materials", "thermal_protection_system"]
            })
        elif airframe_type == "stealth":
            pipeline_config.update({
                "specialized_processes": ["radar_absorbing_material_application", "edge_alignment"]
            })
        
        return ManufacturingPipeline(pipeline_config)