"""
Simple neuromorphic manufacturing pipeline.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.manufacturing.optimization.gradient_optimizer import GradientUCAVOptimizer
from src.manufacturing.optimization.evolutionary_optimizer import EvolutionaryUCAVOptimizer
from src.core.hardware.config_manager import HardwareConfigManager

class ManufacturingPipeline:
    def __init__(self, hardware_interface=None):
        self.system = NeuromorphicSystem(hardware_interface)
        self.config_manager = HardwareConfigManager()
        self.optimizers = {
            'gradient': GradientUCAVOptimizer(learning_rate=0.01, hardware_interface=hardware_interface),
            'evolutionary': EvolutionaryUCAVOptimizer(population_size=20, hardware_interface=hardware_interface)
        }
    
    def run_manufacturing(self, design_specs: Dict[str, float], 
                         optimizer_type: str = 'gradient') -> Dict:
        # Initialize neuromorphic processing
        self.system.initialize()
        
        # Optimize design using neuromorphic hardware
        optimizer = self.optimizers[optimizer_type]
        geometry = optimizer.optimize(design_specs)
        
        # Quality control using neuromorphic processing
        qc_results = self.system.process_data({
            'design': geometry.__dict__,
            'specs': design_specs
        })
        
        # Generate manufacturing parameters
        return {
            'geometry': geometry.__dict__,
            'quality_metrics': qc_results,
            'status': 'completed'
        }

    def cleanup(self):
        self.system.cleanup()