"""
Continuous learning module for UCAV systems.
Implements simple machine learning techniques for ongoing adaptation and improvement.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable
import time
from collections import deque
import random

from src.swarm.adaptive_evolution import AdaptiveEvolution

logger = logging.getLogger(__name__)

class ContinuousLearning:
    """
    Simple machine learning system for continuous improvement of UCAV systems.
    Works alongside the adaptive evolution system to optimize performance.
    """
    
    def __init__(self, adaptive_evolution: AdaptiveEvolution, memory_size: int = 100):
        self.adaptive_evolution = adaptive_evolution
        self.experience_buffer = deque(maxlen=memory_size)
        self.learning_rate = 0.05
        self.exploration_rate = 0.2
        self.discount_factor = 0.9
        self.feature_weights = {}
        self.last_state = None
        self.last_action = None
        
        # Initialize feature weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize feature weights for learning."""
        # Get parameters from adaptive evolution
        for param_name in self.adaptive_evolution.parameters:
            self.feature_weights[param_name] = np.random.uniform(0.1, 0.5)
        
        # Add weights for environmental features
        env_features = ["threat_level", "target_density", "visibility", "terrain_complexity"]
        for feature in env_features:
            self.feature_weights[feature] = np.random.uniform(0.1, 0.5)
    
    def process_experience(self, state: Dict[str, Any], action: Dict[str, Any], 
                          reward: float, next_state: Dict[str, Any]) -> None:
        """
        Process a new experience for learning.
        
        Args:
            state: Current state information
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        # Store experience
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "timestamp": time.time()
        }
        self.experience_buffer.append(experience)
        
        # Update weights using simple TD learning
        self._update_weights(experience)
        
        # Store state and action for next update
        self.last_state = next_state
        self.last_action = action
    
    def _update_weights(self, experience: Dict[str, Any]) -> None:
        """Update feature weights based on experience."""
        state = experience["state"]
        action = experience["action"]
        reward = experience["reward"]
        next_state = experience["next_state"]
        
        # Extract features from state
        state_features = self._extract_features(state)
        
        # Calculate current Q value
        current_q = self._calculate_q_value(state_features, action)
        
        # Calculate next best Q value
        next_actions = self._get_possible_actions(next_state)
        next_features = self._extract_features(next_state)
        next_q_values = [self._calculate_q_value(next_features, a) for a in next_actions]
        max_next_q = max(next_q_values) if next_q_values else 0
        
        # Calculate TD error
        td_error = reward + self.discount_factor * max_next_q - current_q
        
        # Update weights
        for feature, value in state_features.items():
            if feature in self.feature_weights:
                self.feature_weights[feature] += self.learning_rate * td_error * value
    
    def _extract_features(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant features from state."""
        features = {}
        
        # Extract mission features
        if "mission_data" in state:
            mission_data = state["mission_data"]
            features["threat_level"] = mission_data.get("threat_level", 0.0)
            features["target_density"] = mission_data.get("target_density", 0.0)
            features["visibility"] = mission_data.get("visibility", 1.0)
            features["terrain_complexity"] = mission_data.get("terrain_complexity", 0.0)
        
        # Extract current parameter values
        if "parameters" in state:
            for param_name, param_value in state["parameters"].items():
                features[param_name] = param_value
        
        return features
    
    def _calculate_q_value(self, features: Dict[str, float], action: Dict[str, Any]) -> float:
        """Calculate Q value for a state-action pair."""
        q_value = 0.0
        
        # Linear function approximation
        for feature, value in features.items():
            if feature in self.feature_weights:
                q_value += self.feature_weights[feature] * value
        
        # Add action-specific components
        for param_name, param_value in action.items():
            feature_name = f"action_{param_name}"
            if feature_name in self.feature_weights:
                q_value += self.feature_weights[feature_name] * param_value
        
        return q_value
    
    def _get_possible_actions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get possible actions for a state."""
        # For simplicity, generate a few variations of parameter settings
        possible_actions = []
        
        # Get current parameters
        current_params = state.get("parameters", {})
        
        # Add current parameters as an action
        possible_actions.append(current_params)
        
        # Add variations
        for _ in range(3):
            variation = {}
            for param_name, param_value in current_params.items():
                # Add some noise to create variations
                variation[param_name] = max(0.1, min(0.9, param_value + np.random.normal(0, 0.1)))
            possible_actions.append(variation)
        
        return possible_actions
    
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Extract features
        features = self._extract_features(state)
        
        # Get possible actions
        possible_actions = self._get_possible_actions(state)
        
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return random.choice(possible_actions)
        
        # Exploitation: best action
        q_values = [self._calculate_q_value(features, action) for action in possible_actions]
        best_idx = np.argmax(q_values)
        return possible_actions[best_idx]
    
    def optimize_parameters(self, mission_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize parameters based on learned weights and mission data.
        
        Args:
            mission_data: Current mission information
            
        Returns:
            Optimized parameters
        """
        # Create state from mission data
        state = {
            "mission_data": mission_data,
            "parameters": {name: param.value for name, param in self.adaptive_evolution.parameters.items()}
        }
        
        # Select best action
        best_action = self.select_action(state)
        
        # Apply selected parameters
        for param_name, param_value in best_action.items():
            if param_name in self.adaptive_evolution.parameters:
                self.adaptive_evolution.parameters[param_name].value = param_value
        
        # Apply parameters to the system
        self.adaptive_evolution._apply_parameters()
        
        return best_action
    
    def learn_from_batch(self, batch_size: int = 10) -> float:
        """
        Learn from a batch of experiences.
        
        Args:
            batch_size: Number of experiences to learn from
            
        Returns:
            Average TD error
        """
        if len(self.experience_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(list(self.experience_buffer), batch_size)
        
        # Learn from each experience
        td_errors = []
        for experience in batch:
            state = experience["state"]
            action = experience["action"]
            reward = experience["reward"]
            next_state = experience["next_state"]
            
            # Extract features
            state_features = self._extract_features(state)
            
            # Calculate current Q value
            current_q = self._calculate_q_value(state_features, action)
            
            # Calculate next best Q value
            next_actions = self._get_possible_actions(next_state)
            next_features = self._extract_features(next_state)
            next_q_values = [self._calculate_q_value(next_features, a) for a in next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
            
            # Calculate TD error
            td_error = reward + self.discount_factor * max_next_q - current_q
            td_errors.append(abs(td_error))
            
            # Update weights
            for feature, value in state_features.items():
                if feature in self.feature_weights:
                    self.feature_weights[feature] += self.learning_rate * td_error * value
        
        return sum(td_errors) / len(td_errors) if td_errors else 0.0

# Example usage
if __name__ == "__main__":
    from src.swarm.neuromorphic_ai import NeuromorphicAI
    
    # Create neuromorphic AI
    ai_model = NeuromorphicAI(num_neurons=64)
    
    # Create adaptive evolution system
    from src.swarm.adaptive_evolution import AdaptiveEvolution, tactical_mission_fitness
    evolution = AdaptiveEvolution(ai_model)
    evolution.set_fitness_function(tactical_mission_fitness)
    
    # Create continuous learning system
    learning = ContinuousLearning(evolution)
    
    # Simulate mission data
    mission_data = {
        "mission_type": "tactical",
        "threat_level": 0.7,
        "target_density": 0.4,
        "visibility": 0.8,
        "terrain_complexity": 0.5
    }
    
    # Optimize parameters
    optimized_params = learning.optimize_parameters(mission_data)
    
    # Simulate reward
    reward = tactical_mission_fitness({name: value for name, value in optimized_params.items()})
    
    # Process experience
    state = {
        "mission_data": mission_data,
        "parameters": {name: param.value for name, param in evolution.parameters.items()}
    }
    next_state = state.copy()
    next_state["parameters"] = optimized_params
    
    learning.process_experience(state, optimized_params, reward, next_state)
    
    # Learn from batch
    avg_error = learning.learn_from_batch()
    
    logger.info(f"Optimized parameters: {optimized_params}")
    logger.info(f"Reward: {reward}")
    logger.info(f"Average TD error: {avg_error}")