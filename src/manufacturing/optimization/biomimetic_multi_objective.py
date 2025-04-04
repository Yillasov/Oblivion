"""
Multi-objective optimization for biomimetic UCAV designs.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.manufacturing.optimization.biomimetic_optimizer import BiomimeticUCAVOptimizer
from src.simulation.aerodynamics.ucav_model import UCAVGeometry
from src.biomimetic.design.principles import BiomimeticPrinciple
from src.core.utils.logging_framework import get_logger

logger = get_logger("biomimetic_multi_objective")

class BiomimeticMultiObjectiveOptimizer(BiomimeticUCAVOptimizer):
    """Multi-objective optimizer for biomimetic UCAV designs."""
    
    def __init__(self, hardware_interface=None, biological_reference: str = "peregrine_falcon",
                 population_size: int = 30, mutation_rate: float = 0.1):
        """Initialize multi-objective biomimetic optimizer."""
        super().__init__(hardware_interface, biological_reference, population_size, mutation_rate)
        
        # Define objective weights (can be adjusted by user)
        self.objective_weights = {
            "aerodynamic_performance": 0.4,
            "biomimetic_fidelity": 0.3,
            "manufacturing_complexity": 0.2,
            "weight_efficiency": 0.1
        }
        
        # Track Pareto front
        self.pareto_front = []
        
    def set_objective_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for different objectives."""
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        self.objective_weights = {k: v/total for k, v in weights.items()}
        logger.info(f"Updated objective weights: {self.objective_weights}")
        
    def optimize(self, target_metrics: Dict[str, float], generations: int = 50) -> UCAVGeometry:
        """Run multi-objective optimization."""
        # Initialize population
        population = self._initialize_population()
        
        # Track Pareto front
        self.pareto_front = []
        
        for gen in range(generations):
            # Evaluate all objectives for each individual
            objective_scores = []
            for params in population:
                geometry = self._create_geometry(params)
                scores = self._evaluate_all_objectives(geometry, target_metrics)
                objective_scores.append(scores)
            
            # Update Pareto front
            self._update_pareto_front(population, objective_scores)
            
            # Select parents using Pareto dominance and crowding distance
            new_population = []
            
            # Keep elite solutions from Pareto front (25% of population)
            elite_count = min(len(self.pareto_front), self.population_size // 4)
            new_population.extend(self.pareto_front[:elite_count])
            
            # Fill rest of population with offspring
            while len(new_population) < self.population_size:
                # Tournament selection based on dominance
                parent1 = self._tournament_select(population, objective_scores)
                parent2 = self._tournament_select(population, objective_scores)
                
                # Create offspring
                child = self._crossover(parent1, parent2)
                child = self._biomimetic_mutate(child)
                
                new_population.append(child)
                
            population = new_population
            
            logger.info(f"Generation {gen+1}/{generations}: Pareto front size = {len(self.pareto_front)}")
            
        # Select best compromise solution from Pareto front
        best_solution = self._select_compromise_solution()
        return self._create_geometry(best_solution)
    
    def _evaluate_all_objectives(self, geometry: UCAVGeometry, 
                                target_metrics: Dict[str, float]) -> Dict[str, float]:
        """Evaluate all optimization objectives."""
        # Aerodynamic performance (from parent class)
        aero_fitness = self._evaluate_design(geometry, target_metrics)
        
        # Biomimetic fidelity (how well it matches biological reference)
        bio_fidelity = self._calculate_biomimetic_fidelity(geometry)
        
        # Manufacturing complexity (lower is better)
        manufacturing_complexity = self._estimate_manufacturing_complexity(geometry)
        
        # Weight efficiency
        weight_efficiency = self._calculate_weight_efficiency(geometry)
        
        return {
            "aerodynamic_performance": aero_fitness,
            "biomimetic_fidelity": bio_fidelity,
            "manufacturing_complexity": -manufacturing_complexity,  # Negative because lower is better
            "weight_efficiency": weight_efficiency
        }
    
    def _calculate_biomimetic_fidelity(self, geometry: UCAVGeometry) -> float:
        """Calculate how closely the design matches biological reference."""
        if not self.biological_reference:
            return 0.5  # Neutral score if no reference
            
        fidelity = 0.0
        
        # Compare aspect ratio
        bio_aspect = self.biological_reference.performance_metrics.get('aspect_ratio', 0)
        if bio_aspect > 0:
            aspect_similarity = 1.0 - min(1.0, abs(geometry.aspect_ratio - bio_aspect) / bio_aspect)
            fidelity += aspect_similarity * 0.3
            
        # Compare sweep angle
        bio_sweep = self.biological_reference.morphological_data.get('sweep_angle_deg', 0)
        if bio_sweep > 0:
            sweep_similarity = 1.0 - min(1.0, abs(geometry.sweep_angle - bio_sweep) / 45.0)
            fidelity += sweep_similarity * 0.3
            
        # Compare taper ratio
        bio_taper = self.biological_reference.morphological_data.get('taper_ratio', 0)
        if bio_taper > 0:
            taper_similarity = 1.0 - min(1.0, abs(geometry.taper_ratio - bio_taper) / 0.5)
            fidelity += taper_similarity * 0.4
            
        return fidelity
    
    def _estimate_manufacturing_complexity(self, geometry: UCAVGeometry) -> float:
        """Estimate manufacturing complexity (0-1 scale, lower is better)."""
        complexity = 0.0
        
        # More complex geometries are harder to manufacture
        if geometry.sweep_angle > 40:
            complexity += 0.2
            
        if geometry.taper_ratio < 0.25:
            complexity += 0.15
            
        if geometry.aspect_ratio > 8:
            complexity += 0.1
            
        # Add biomimetic complexity factors
        if BiomimeticPrinciple.ADAPTIVE_MORPHOLOGY in self.active_principles:
            complexity += 0.25
            
        if BiomimeticPrinciple.MATERIAL_EFFICIENCY in self.active_principles:
            complexity += 0.15
            
        return min(1.0, complexity)
    
    def _calculate_weight_efficiency(self, geometry: UCAVGeometry) -> float:
        """Calculate weight efficiency score."""
        # Simple model: higher aspect ratio generally means better efficiency
        # but extremely high values can be impractical
        if geometry.aspect_ratio < 4:
            return 0.3
        elif geometry.aspect_ratio < 6:
            return 0.6
        elif geometry.aspect_ratio < 10:
            return 0.9
        else:
            return 0.7  # Penalize extremely high aspect ratios
    
    def _dominates(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> bool:
        """Check if solution 1 dominates solution 2 (Pareto dominance)."""
        better_in_one = False
        for obj in scores1:
            if scores1[obj] < scores2[obj]:
                return False
            if scores1[obj] > scores2[obj]:
                better_in_one = True
        return better_in_one
    
    def _update_pareto_front(self, population: List[Dict[str, float]], 
                            objective_scores: List[Dict[str, float]]) -> None:
        """Update the Pareto front with non-dominated solutions."""
        self.pareto_front = []
        
        for i, (solution, scores) in enumerate(zip(population, objective_scores)):
            dominated = False
            
            for other_scores in objective_scores:
                if self._dominates(other_scores, scores):
                    dominated = True
                    break
                    
            if not dominated:
                self.pareto_front.append(solution)
                
        # Limit size of Pareto front
        if len(self.pareto_front) > self.population_size // 2:
            # Sort by weighted sum of objectives
            def weighted_sum(solution_idx):
                scores = objective_scores[population.index(self.pareto_front[solution_idx])]
                return sum(scores[obj] * self.objective_weights[obj] for obj in scores)
                
            indices = sorted(range(len(self.pareto_front)), key=weighted_sum, reverse=True)
            self.pareto_front = [self.pareto_front[i] for i in indices[:self.population_size // 2]]
    
    def _tournament_select(self, population: List[Dict[str, float]], 
                          objective_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Tournament selection based on Pareto dominance."""
        # Select 3 random candidates
        candidates = np.random.choice(len(population), 3, replace=False)
        
        # Find non-dominated among candidates
        non_dominated = []
        for i in candidates:
            dominated = False
            for j in candidates:
                if i != j and self._dominates(objective_scores[j], objective_scores[i]):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(i)
        
        # If multiple non-dominated, choose randomly
        if non_dominated:
            winner_idx = np.random.choice(non_dominated)
        else:
            # Fallback to weighted sum if all dominated
            def weighted_sum(idx):
                scores = objective_scores[idx]
                return sum(scores[obj] * self.objective_weights[obj] for obj in scores)
                
            winner_idx = max(candidates, key=weighted_sum)
            
        return population[winner_idx]
    
    def _select_compromise_solution(self) -> Dict[str, float]:
        """Select best compromise solution from Pareto front."""
        if not self.pareto_front:
            # Fallback if no Pareto solutions
            return self._sample_parameters()
            
        # Use weighted sum method for compromise
        best_solution = self.pareto_front[0]
        best_score = float('-inf')
        
        for solution in self.pareto_front:
            geometry = self._create_geometry(solution)
            scores = self._evaluate_all_objectives(geometry, {})
            
            # Calculate weighted sum
            weighted_sum = sum(scores[obj] * self.objective_weights[obj] for obj in scores)
            
            if weighted_sum > best_score:
                best_score = weighted_sum
                best_solution = solution
                
        return best_solution