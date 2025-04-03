"""
Neuromorphic-enabled cost estimation system for UCAV manufacturing.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any
import numpy as np
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.simulation.aerodynamics.ucav_model import UCAVGeometry
from src.manufacturing.materials.material_selector import NeuromorphicMaterialSelector
from src.manufacturing.simulation.process_simulator import ManufacturingSimulator

class CostEstimator:
    def __init__(self, hardware_interface=None):
        self.system = NeuromorphicSystem(hardware_interface)
        self.material_selector = NeuromorphicMaterialSelector(hardware_interface)
        self.simulator = ManufacturingSimulator(hardware_interface)
        self.base_costs = {
            'labor_rate': 75.0,  # $/hour
            'machine_rate': 150.0,  # $/hour
            'overhead_rate': 0.3,  # 30% of direct costs
            'tooling_cost': 25000.0,  # $ base tooling cost
            'testing_cost': 15000.0   # $ base testing cost
        }

    def estimate_costs(self, geometry: UCAVGeometry, 
                      manufacturing_data: Dict[str, Any]) -> Dict[str, float]:
        self.system.initialize()
        
        # Use neuromorphic processing for cost prediction
        cost_prediction = self.system.process_data({
            'geometry': geometry.__dict__,
            'manufacturing': manufacturing_data,
            'base_costs': self.base_costs,
            'computation': 'cost_estimation'
        })
        
        # Calculate detailed costs using neuromorphic insights
        material_costs = self._calculate_material_costs(
            manufacturing_data.get('materials', {}),
            cost_prediction.get('material_factors', {})
        )
        
        labor_costs = self._calculate_labor_costs(
            manufacturing_data.get('completion_time', 0),
            cost_prediction.get('labor_efficiency', 1.0)
        )
        
        self.system.cleanup()
        return {
            'material_cost': material_costs,
            'labor_cost': labor_costs,
            'tooling_cost': self.base_costs['tooling_cost'] * 
                          cost_prediction.get('tooling_factor', 1.0),
            'testing_cost': self.base_costs['testing_cost'] * 
                          cost_prediction.get('testing_factor', 1.0),
            'overhead_cost': (material_costs + labor_costs) * 
                           self.base_costs['overhead_rate'],
            'total_cost': cost_prediction.get('total_cost', 0.0),
            'confidence_score': cost_prediction.get('confidence', 0.0)
        }

    def _calculate_material_costs(self, materials: Dict[str, Any], 
                                factors: Dict[str, float]) -> float:
        return sum(
            mat.get('cost', 0.0) * factors.get(mat_type, 1.0)
            for mat_type, mat in materials.items()
        )

    def _calculate_labor_costs(self, hours: float, efficiency: float) -> float:
        return hours * self.base_costs['labor_rate'] * efficiency