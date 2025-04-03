"""
Simple evolutionary-based UCAV design optimizer.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Tuple
from src.simulation.aerodynamics.ucav_model import UCAVGeometry, UCAVAerodynamicsModel
from src.core.integration.neuromorphic_system import NeuromorphicSystem

class EvolutionaryUCAVOptimizer:
    def __init__(self, population_size: int = 20, hardware_interface=None):
        self.population_size = population_size
        self.aero_model = None
        self.system = NeuromorphicSystem(hardware_interface)
        self.param_ranges = {
            'length': (5.0, 15.0),
            'wingspan': (8.0, 20.0),
            'wing_area': (20.0, 60.0),
            'aspect_ratio': (4.0, 8.0),
            'taper_ratio': (0.2, 0.5),
            'sweep_angle': (20.0, 45.0)
        }

    def optimize(self, target_metrics: Dict[str, float], generations: int = 50) -> UCAVGeometry:
        population = self._initialize_population()
        best_geometry = self._create_geometry(population[0])
        
        for _ in range(generations):
            fitness_scores = [
                self._evaluate_design(self._create_geometry(params), target_metrics)
                for params in population
            ]
            
            # Select best performers
            elite_idx = np.argsort(fitness_scores)[-self.population_size//2:]
            elite = [population[i] for i in elite_idx]
            
            # Create new population through crossover
            population = elite.copy()
            while len(population) < self.population_size:
                parent1, parent2 = np.random.choice(elite, 2, replace=False)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                population.append(child)
            
            # Update best geometry
            best_idx = np.argmax(fitness_scores)
            best_geometry = self._create_geometry(population[best_idx])
            
        return best_geometry

    def _initialize_population(self) -> List[Dict[str, float]]:
        return [
            {param: np.random.uniform(range[0], range[1]) 
             for param, range in self.param_ranges.items()}
            for _ in range(self.population_size)
        ]

    # Remove the first _crossover method and keep only this one
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        # Use neuromorphic network for crossover decisions
        crossover_data = self.system.process_data({
            'parent1': list(parent1.values()),
            'parent2': list(parent2.values())
        })
        
        return {
            param: parent1[param] if crossover_data.get(f'select_{param}', 0.0) > 0.5 
                  else parent2[param]
            for param in self.param_ranges
        }

    def _mutate(self, params: Dict[str, float], mutation_rate: float = 0.1) -> Dict[str, float]:
        # Use neuromorphic hardware for mutation decisions
        spike_data = self.system.process_data({
            'input': list(params.values()),
            'mutation_rate': mutation_rate
        })
        
        for param in params:
            if spike_data.get(f'mutate_{param}', 0.0) > 0.5:
                min_val, max_val = self.param_ranges[param]
                # Use neuromorphic noise for parameter updates
                noise = self.system.process_data({'noise': True})['output']
                params[param] = np.clip(
                    params[param] + noise * (max_val - min_val),
                    min_val, max_val
                )
        return params

    def _create_geometry(self, params: Dict[str, float]) -> UCAVGeometry:
        return UCAVGeometry(
            length=params['length'],
            wingspan=params['wingspan'],
            wing_area=params['wing_area'],
            aspect_ratio=params['aspect_ratio'],
            taper_ratio=params['taper_ratio'],
            sweep_angle=params['sweep_angle']
        )

    def _evaluate_design(self, geometry: UCAVGeometry, target_metrics: Dict[str, float]) -> float:
        if not self.aero_model:
            self.aero_model = UCAVAerodynamicsModel(geometry)
        
        coeffs = self.aero_model.calculate_coefficients(alpha=2.0, beta=0.0, mach=0.8)
        
        fitness = 0.0
        for metric, target in target_metrics.items():
            if metric in coeffs:
                fitness -= abs(coeffs[metric] - target)
        
        return fitness