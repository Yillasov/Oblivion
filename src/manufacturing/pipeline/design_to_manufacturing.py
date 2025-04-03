"""
Neuromorphic-enabled design-to-manufacturing pipeline for UCAV production.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.manufacturing.workflow.production_automator import ProductionAutomator
from src.manufacturing.optimization.gradient_optimizer import GradientUCAVOptimizer
from src.simulation.aerodynamics.ucav_model import UCAVGeometry

class DesignToManufacturingPipeline:
    def __init__(self, hardware_interface=None):
        self.system = NeuromorphicSystem(hardware_interface)
        self.optimizer = GradientUCAVOptimizer(learning_rate=0.01, hardware_interface=hardware_interface)
        self.production = ProductionAutomator(hardware_interface)
        
    def execute_pipeline(self, design_requirements: Dict[str, Any]) -> Dict[str, Any]:
        self.system.initialize()
        pipeline_results = {}
        
        try:
            # Design Optimization
            optimized_geometry = self.optimizer.optimize(
                design_requirements.get('target_metrics', {})
            )
            pipeline_results['design'] = optimized_geometry.__dict__
            
            # Design Validation using neuromorphic processing
            validation_results = self.system.process_data({
                'geometry': optimized_geometry.__dict__,
                'requirements': design_requirements,
                'computation': 'design_validation'
            })
            pipeline_results['validation'] = validation_results
            
            if validation_results.get('design_valid', False):
                # Production Workflow
                production_results = self.production.run_production_workflow(
                    optimized_geometry,
                    design_requirements
                )
                pipeline_results.update(production_results)
                
                # Final feasibility assessment
                feasibility = self.system.process_data({
                    'pipeline_results': pipeline_results,
                    'computation': 'manufacturing_feasibility'
                })
                pipeline_results['feasibility'] = feasibility
                
            pipeline_results['status'] = 'completed'
            
        except Exception as e:
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            
        finally:
            self.system.cleanup()
            
        return pipeline_results

    def get_design_recommendations(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        return self.system.process_data({
            'results': pipeline_results,
            'computation': 'design_recommendations'
        })