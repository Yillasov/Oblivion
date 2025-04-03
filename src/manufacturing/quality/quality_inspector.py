"""
Quality inspection system for UCAV manufacturing.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.manufacturing.simulation.process_simulator import ManufacturingSimulator
from src.simulation.aerodynamics.ucav_model import UCAVGeometry
from src.core.utils.error_handling import (
    ErrorContext, handle_errors, QualityControlError
)

class QualityInspector:
    def __init__(self, hardware_interface=None):
        self.system = NeuromorphicSystem(hardware_interface)
        self.simulator = ManufacturingSimulator(hardware_interface)
        self.quality_threshold = 0.85
        self.inspection_aspects = ['structural', 'aerodynamic', 'material']
        
    @handle_errors(context={"operation": "quality_inspection"})
    def inspect_ucav(self, geometry: UCAVGeometry, 
                     manufacturing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform quality inspection on a UCAV design.
        
        Args:
            geometry: The UCAV geometry to inspect
            manufacturing_data: Manufacturing process data
            
        Returns:
            Dict[str, Any]: Inspection results
        """
        self.system.initialize()
        
        try:
            # Perform neuromorphic-based inspection
            inspection_results = {}
            detailed_results = {}
            
            # Inspect each aspect
            for aspect in self.inspection_aspects:
                with ErrorContext(context={"aspect": aspect}):
                    inspection_config = {
                        'computation': 'inspection',
                        'aspect': aspect,
                        'geometry': geometry.__dict__,
                        'manufacturing_data': manufacturing_data,
                        'parameters': {
                            'tolerance': 0.05,
                            'inspection_points': 12
                        }
                    }
                    aspect_results = self.system.process_data(inspection_config)
                    detailed_results[aspect] = aspect_results
            
            # Perform overall quality assessment
            assessment_config = {
                'computation': 'quality_assessment',
                'geometry': geometry.__dict__,
                'manufacturing_data': manufacturing_data,
                'inspection_results': detailed_results
            }
            quality_assessment = self.system.process_data(assessment_config)
            
            # Determine pass/fail status
            passed = quality_assessment.get('overall_quality', 0) >= self.quality_threshold
            
            # Compile final results
            inspection_results = {
                'quality_score': quality_assessment.get('overall_quality', 0),
                'passed': passed,
                'detailed_results': detailed_results,
                'defects': quality_assessment.get('detected_defects', []),
                'recommendations': quality_assessment.get('improvement_suggestions', [])
            }
            
            # If failed, raise quality control error (but still return results)
            if not passed:
                error = QualityControlError(
                    message="Quality inspection failed",
                    details={
                        "quality_score": inspection_results['quality_score'],
                        "threshold": self.quality_threshold,
                        "defects": inspection_results['defects']
                    }
                )
                error.log()
                
            return inspection_results
            
        finally:
            self.system.cleanup()