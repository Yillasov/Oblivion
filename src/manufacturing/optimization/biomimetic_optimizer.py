"""
Biomimetic UCAV design optimizer using neuromorphic computing.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from src.simulation.aerodynamics.ucav_model import UCAVGeometry, UCAVAerodynamicsModel
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.manufacturing.optimization.ucav_optimizer import NeuromorphicUCAVOptimizer
from src.biomimetic.design.principles import BiomimeticDesignFramework, BiomimeticPrinciple
from src.core.utils.logging_framework import get_logger

logger = get_logger("biomimetic_optimizer")

class BiomimeticUCAVOptimizer(NeuromorphicUCAVOptimizer):
    """UCAV optimizer that incorporates biomimetic design principles."""
    
    def __init__(self, hardware_interface=None, biological_reference: str = "peregrine_falcon", 
                 population_size: int = 20, mutation_rate: float = 0.1):
        """
        Initialize the biomimetic UCAV optimizer.
        
        Args:
            hardware_interface: Interface to neuromorphic hardware
            biological_reference: Name of biological reference model to use
            population_size: Size of the evolutionary population
            mutation_rate: Rate of mutation in evolutionary algorithm
        """
        super().__init__(hardware_interface)
        self.biomimetic_framework = BiomimeticDesignFramework()
        self.biological_reference = self.biomimetic_framework.biological_references.get(
            biological_reference, None)
        
        # Evolutionary algorithm parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        # Add biomimetic-specific parameters
        self.param_ranges.update({
            'wing_flexibility': (0.1, 0.9),
            'surface_texture': (0.01, 0.5),
            'adaptive_control': (0.0, 1.0)
        })
        
        # Biomimetic design principles to apply
        self.active_principles = [
            BiomimeticPrinciple.FORM_FOLLOWS_FUNCTION,
            BiomimeticPrinciple.ADAPTIVE_MORPHOLOGY,
            BiomimeticPrinciple.MATERIAL_EFFICIENCY
        ]
        
        logger.info(f"Initialized biomimetic optimizer with reference: {biological_reference}")
    
    def optimize(self, target_metrics: Dict[str, float], generations: int = 50) -> UCAVGeometry:
        """
        Optimize UCAV design using biomimetic principles and evolutionary algorithms.
        
        Args:
            target_metrics: Target performance metrics
            generations: Number of evolutionary generations
            
        Returns:
            Optimized UCAV geometry
        """
        # Apply biological scaling factors if reference exists
        if self.biological_reference:
            target_metrics = self._apply_biological_scaling(target_metrics)
        
        # Initialize population with biomimetic influence
        population = self._initialize_population()
        best_geometry = self._create_geometry(population[0])
        
        # Run evolutionary optimization
        for gen in range(generations):
            # Evaluate fitness for each individual
            fitness_scores = [
                self._evaluate_design(self._create_geometry(params), target_metrics)
                for params in population
            ]
            
            # Select elite individuals
            elite_idx = np.argsort(fitness_scores)[-self.population_size//4:]
            elite = [population[i] for i in elite_idx]
            
            # Create new population through crossover and mutation
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                # Select parents with preference for higher fitness
                parent1, parent2 = self._select_parents(population, fitness_scores)
                
                # Create child through crossover
                child = self._crossover(parent1, parent2)
                
                # Apply biomimetic mutation
                child = self._biomimetic_mutate(child)
                
                new_population.append(child)
            
            # Update population
            population = new_population
            
            # Update best geometry
            best_idx = np.argmax(fitness_scores)
            current_best = self._create_geometry(population[best_idx])
            
            # Apply biomimetic principles to best design
            current_best = self._apply_biomimetic_principles(current_best)
            
            # Update if better than previous best
            if self._evaluate_design(current_best, target_metrics) > self._evaluate_design(best_geometry, target_metrics):
                best_geometry = current_best
            
            logger.info(f"Generation {gen+1}/{generations}: Best fitness = {max(fitness_scores):.4f}")
        
        return best_geometry
    
    def _initialize_population(self) -> List[Dict[str, float]]:
        """Initialize population with biomimetic influence."""
        population = []
        for _ in range(self.population_size):
            params = self._sample_parameters()
            population.append(params)
        return population
    
    def _select_parents(self, population: List[Dict[str, float]], fitness_scores: List[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Select parents using tournament selection."""
        # Tournament selection
        tournament_size = 3
        selected_parents = []
        
        for _ in range(2):  # Select 2 parents
            # Random tournament
            candidates = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in candidates]
            winner_idx = candidates[np.argmax(tournament_fitness)]
            selected_parents.append(population[winner_idx])
        
        return selected_parents[0], selected_parents[1]
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Perform crossover between two parents."""
        child = {}
        
        # Uniform crossover with biomimetic bias
        for param in self.param_ranges:
            # Biomimetic parameters have higher chance to inherit from better parent
            if param in ['wing_flexibility', 'surface_texture', 'adaptive_control']:
                # Use neuromorphic system to determine inheritance
                if self.system.process_data({'sample': True})['output'] > 0.5:
                    child[param] = parent1[param]
                else:
                    child[param] = parent2[param]
            else:
                # Standard uniform crossover for other parameters
                if np.random.random() > 0.5:
                    child[param] = parent1[param]
                else:
                    child[param] = parent2[param]
        
        return child
    
    def _biomimetic_mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Apply mutation with biomimetic influence."""
        mutated = individual.copy()
        
        for param in self.param_ranges:
            # Apply mutation with probability based on mutation rate
            if np.random.random() < self.mutation_rate:
                min_val, max_val = self.param_ranges[param]
                
                # Biomimetic parameters have biologically-inspired mutations
                if param in ['wing_flexibility', 'surface_texture', 'adaptive_control']:
                    # Get biological reference value if available
                    bio_value = 0.5  # Default
                    if self.biological_reference and param in self.biological_reference.performance_metrics:
                        bio_value = self.biological_reference.performance_metrics[param]
                    
                    # Mutation biased toward biological reference
                    current = mutated[param]
                    # Move slightly toward biological reference
                    bias = 0.3 * (bio_value - current)
                    # Add random variation
                    noise = np.random.normal(0, 0.1 * (max_val - min_val))
                    mutated[param] = np.clip(current + bias + noise, min_val, max_val)
                else:
                    # Standard mutation for other parameters
                    noise = np.random.normal(0, 0.1 * (max_val - min_val))
                    mutated[param] = np.clip(mutated[param] + noise, min_val, max_val)
        
        return mutated
    
    def _create_geometry(self, params: Dict[str, float]) -> UCAVGeometry:
        """Create UCAVGeometry from parameters."""
        return UCAVGeometry(
            length=params['length'],
            wingspan=params['wingspan'],
            wing_area=params['wing_area'],
            aspect_ratio=params['aspect_ratio'],
            taper_ratio=params['taper_ratio'],
            sweep_angle=params['sweep_angle']
        )
    
    def _apply_biological_scaling(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Apply biological scaling factors to target metrics."""
        if not self.biological_reference:
            return metrics
            
        scaled_metrics = metrics.copy()
        
        # Scale aerodynamic metrics based on biological reference
        if 'lift_to_drag' in metrics and 'glide_ratio' in self.biological_reference.performance_metrics:
            bio_glide = self.biological_reference.performance_metrics['glide_ratio']
            scaled_metrics['lift_to_drag'] = metrics['lift_to_drag'] * (bio_glide / 10.0)
        
        # Scale maneuverability metrics
        if 'roll_rate' in metrics and 'wing_loading_n_per_sqm' in self.biological_reference.performance_metrics:
            bio_loading = self.biological_reference.performance_metrics['wing_loading_n_per_sqm']
            scaled_metrics['roll_rate'] = metrics['roll_rate'] * (100.0 / bio_loading)
            
        return scaled_metrics
    
    def _apply_biomimetic_principles(self, geometry: UCAVGeometry) -> UCAVGeometry:
        """Apply biomimetic design principles to refine geometry."""
        # Get current parameters as dictionary
        params = {
            'length': geometry.length,
            'wingspan': geometry.wingspan,
            'wing_area': geometry.wing_area,
            'aspect_ratio': geometry.aspect_ratio,
            'taper_ratio': geometry.taper_ratio,
            'sweep_angle': geometry.sweep_angle
        }
        
        # Apply form follows function principle
        if BiomimeticPrinciple.FORM_FOLLOWS_FUNCTION in self.active_principles:
            # Adjust sweep angle based on biological reference
            if self.biological_reference and 'sweep_angle_deg' in self.biological_reference.morphological_data:
                bio_sweep = self.biological_reference.morphological_data.get('sweep_angle_deg', 30.0)
                params['sweep_angle'] = (params['sweep_angle'] + bio_sweep) / 2.0
        
        # Apply adaptive morphology principle
        if BiomimeticPrinciple.ADAPTIVE_MORPHOLOGY in self.active_principles:
            # Adjust aspect ratio for better maneuverability
            if self.biological_reference and 'aspect_ratio' in self.biological_reference.performance_metrics:
                bio_aspect = self.biological_reference.performance_metrics['aspect_ratio']
                params['aspect_ratio'] = (params['aspect_ratio'] * 0.7 + bio_aspect * 0.3)
        
        # Create new geometry with biomimetic adjustments
        return UCAVGeometry(
            length=params['length'],
            wingspan=params['wingspan'],
            wing_area=params['wing_area'],
            aspect_ratio=params['aspect_ratio'],
            taper_ratio=params['taper_ratio'],
            sweep_angle=params['sweep_angle']
        )
    
    def _sample_parameters(self) -> Dict[str, float]:
        """Sample parameters with biomimetic influence."""
        params = super()._sample_parameters()
        
        # Add biomimetic-specific parameters
        for param in ['wing_flexibility', 'surface_texture', 'adaptive_control']:
            min_val, max_val = self.param_ranges[param]
            spike_rate = self.system.process_data({'sample': True, 'biomimetic': True})['output']
            params[param] = min_val + (max_val - min_val) * spike_rate
        
        return params
    
    def _evaluate_design(self, geometry: UCAVGeometry, target_metrics: Dict[str, float]) -> float:
        """Evaluate design with additional biomimetic fitness components."""
        # Get base fitness from standard evaluation
        fitness = super()._evaluate_design(geometry, target_metrics)
        
        # Add biomimetic fitness components
        if self.biological_reference:
            # Reward designs that match biological aspect ratio
            bio_aspect = self.biological_reference.performance_metrics.get('aspect_ratio', 0)
            if bio_aspect > 0:
                aspect_similarity = -abs(geometry.aspect_ratio - bio_aspect) / bio_aspect
                fitness += aspect_similarity * 0.2  # 20% weight to biological similarity
        
        return fitness