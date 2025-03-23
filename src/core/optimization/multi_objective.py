"""
Multi-objective optimization framework for UCAV systems.
"""

from typing import Dict, List, Tuple, Callable, Any, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

class OptimizationObjective(Enum):
    """Common optimization objectives across UCAV systems."""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    STEALTH = "stealth"
    THERMAL = "thermal"
    WEIGHT = "weight"
    COST = "cost"
    RELIABILITY = "reliability"
    MANUFACTURABILITY = "manufacturability"

@dataclass
class ObjectiveConfig:
    """Configuration for an optimization objective."""
    objective: OptimizationObjective
    weight: float = 1.0
    min_value: float = 0.0
    max_value: float = 1.0
    target_value: Optional[float] = None
    is_minimize: bool = False  # True if we want to minimize this objective

@dataclass
class Solution:
    """A candidate solution in the optimization space."""
    parameters: Dict[str, float]
    objective_values: Dict[OptimizationObjective, float]
    dominated_by: int = 0  # Number of solutions that dominate this one
    
    def dominates(self, other: 'Solution') -> bool:
        """Check if this solution dominates another solution."""
        better_in_one = False
        for obj, value in self.objective_values.items():
            other_value = other.objective_values.get(obj, 0.0)
            # For minimization objectives, lower is better
            is_minimize = obj in [OptimizationObjective.WEIGHT, OptimizationObjective.COST, 
                                 OptimizationObjective.THERMAL]
            
            if is_minimize:
                if value > other_value:
                    return False
                if value < other_value:
                    better_in_one = True
            else:
                if value < other_value:
                    return False
                if value > other_value:
                    better_in_one = True
        
        return better_in_one

class MultiObjectiveOptimizer:
    """Multi-objective optimizer using Pareto front approach."""
    
    def __init__(self, 
                 objectives: List[ObjectiveConfig],
                 parameter_ranges: Dict[str, Tuple[float, float]],
                 evaluation_function: Callable[[Dict[str, float]], Dict[OptimizationObjective, float]]):
        """
        Initialize multi-objective optimizer.
        
        Args:
            objectives: List of optimization objectives
            parameter_ranges: Dictionary of parameter names and their ranges (min, max)
            evaluation_function: Function that evaluates a set of parameters and returns objective values
        """
        self.objectives = objectives
        self.parameter_ranges = parameter_ranges
        self.evaluation_function = evaluation_function
        self.population: List[Solution] = []
        self.pareto_front: List[Solution] = []
        
    def initialize_population(self, size: int = 50) -> None:
        """Initialize random population of solutions."""
        self.population = []
        for _ in range(size):
            # Generate random parameters within ranges
            params = {
                param: np.random.uniform(min_val, max_val)
                for param, (min_val, max_val) in self.parameter_ranges.items()
            }
            
            # Evaluate objectives
            objective_values = self.evaluation_function(params)
            
            # Create solution
            solution = Solution(params, objective_values)
            self.population.append(solution)
            
    def compute_pareto_front(self) -> List[Solution]:
        """Compute the Pareto front from the current population."""
        # Reset domination counters
        for solution in self.population:
            solution.dominated_by = 0
            
        # Count dominations
        for i, solution_i in enumerate(self.population):
            for j, solution_j in enumerate(self.population):
                if i != j and solution_j.dominates(solution_i):
                    solution_i.dominated_by += 1
        
        # Extract non-dominated solutions (Pareto front)
        self.pareto_front = [s for s in self.population if s.dominated_by == 0]
        return self.pareto_front
    
    def evolve_population(self, generations: int = 10) -> List[Solution]:
        """Evolve the population to find better solutions."""
        for _ in range(generations):
            # Compute current Pareto front
            pareto_front = self.compute_pareto_front()
            
            # Create new population with elitism (keep Pareto front)
            new_population = pareto_front.copy()
            
            # Fill the rest with crossover and mutation
            while len(new_population) < len(self.population):
                # Select parents (tournament selection)
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                child_params = self._crossover(parent1.parameters, parent2.parameters)
                
                # Mutation
                child_params = self._mutate(child_params)
                
                # Evaluate and add to new population
                objective_values = self.evaluation_function(child_params)
                new_population.append(Solution(child_params, objective_values))
            
            # Update population
            self.population = new_population
            
        # Final Pareto front
        return self.compute_pareto_front()
    
    def _tournament_selection(self, tournament_size: int = 3) -> Solution:
        """Select a solution using tournament selection."""
        candidates = [self.population[i] for i in np.random.choice(len(self.population), tournament_size, replace=False)]
        best = candidates[0]
        for candidate in candidates[1:]:
            if candidate.dominated_by < best.dominated_by:
                best = candidate
        return best
    
    def _crossover(self, params1: Dict[str, float], params2: Dict[str, float]) -> Dict[str, float]:
        """Perform crossover between two parameter sets."""
        child_params = {}
        for param in params1:
            # Uniform crossover
            if np.random.random() < 0.5:
                child_params[param] = params1[param]
            else:
                child_params[param] = params2[param]
        return child_params
    
    def _mutate(self, params: Dict[str, float], mutation_rate: float = 0.1) -> Dict[str, float]:
        """Mutate parameters with given probability."""
        for param in params:
            if np.random.random() < mutation_rate:
                min_val, max_val = self.parameter_ranges[param]
                # Gaussian mutation
                sigma = (max_val - min_val) * 0.1
                params[param] += np.random.normal(0, sigma)
                # Ensure within bounds
                params[param] = np.clip(params[param], min_val, max_val)
        return params
    
    def get_best_compromise(self) -> Solution:
        """Get the best compromise solution from the Pareto front."""
        if not self.pareto_front:
            self.compute_pareto_front()
            
        if not self.pareto_front:
            return Solution({}, {})  # Return empty solution instead of None
            
        # Normalize objective values
        normalized_scores = []
        for solution in self.pareto_front:
            score = 0
            for obj_config in self.objectives:
                obj = obj_config.objective
                value = solution.objective_values.get(obj, 0)
                
                # Normalize to 0-1 range
                norm_value = (value - obj_config.min_value) / (obj_config.max_value - obj_config.min_value)
                
                # Apply weight and direction (minimize/maximize)
                if obj_config.is_minimize:
                    score += obj_config.weight * (1 - norm_value)
                else:
                    score += obj_config.weight * norm_value
                    
            normalized_scores.append(score)
            
        # Return solution with highest score
        best_idx = np.argmax(normalized_scores)
        return self.pareto_front[best_idx]