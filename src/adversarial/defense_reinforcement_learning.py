"""
Reinforcement learning system for continuous improvement of UCAV defense strategies.
Adapts defensive tactics based on experience and outcomes.
"""

import numpy as np
import random
import time
from collections import deque
from typing import Dict, List, Any, Tuple, Optional
import logging
import sys
import os

# Fix import path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src.adversarial.real_time_adversarial_learning import RealTimeAdversarialLearner
except ImportError:
    # Fallback for direct script execution
    from real_time_adversarial_learning import RealTimeAdversarialLearner

logger = logging.getLogger(__name__)

class DefenseReinforcementLearner:
    """
    Reinforcement learning system for UCAV defense strategies.
    Continuously improves defensive capabilities through experience.
    """
    
    def __init__(self, 
                adversarial_learner: Optional[RealTimeAdversarialLearner] = None,
                learning_rate: float = 0.05,
                discount_factor: float = 0.9,
                exploration_rate: float = 0.2,
                memory_size: int = 1000):
        self.adversarial_learner = adversarial_learner
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.experience_buffer = deque(maxlen=memory_size)
        
        # Defense strategy parameters
        self.defense_params = {
            "evasion_intensity": 0.5,
            "countermeasure_usage": 0.3,
            "stealth_mode": 0.7,
            "formation_spacing": 0.6,
            "sensor_sensitivity": 0.5
        }
        
        # Q-values for state-action pairs
        self.q_values: Dict[str, Dict[str, float]] = {}
        
    def observe_threat(self, threat_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Observe a threat and determine optimal defense strategy.
        
        Args:
            threat_data: Data about the current threat
            
        Returns:
            Defense strategy parameters
        """
        # Convert threat data to state representation
        state = self._encode_state(threat_data)
        
        # Choose action (defense strategy) using epsilon-greedy policy
        if random.random() < self.exploration_rate:
            # Explore: random strategy with small variations from current
            strategy = self._explore_strategy()
        else:
            # Exploit: best known strategy for this state
            strategy = self._get_best_strategy(state)
            
        # If adversarial learner exists, incorporate its knowledge
        if self.adversarial_learner:
            self._incorporate_adversarial_knowledge(strategy, threat_data)
            
        return strategy
    
    def _encode_state(self, threat_data: Dict[str, Any]) -> str:
        """Convert threat data to a discrete state representation."""
        # Extract key features and discretize
        threat_type = threat_data.get("type", "unknown")
        threat_level = self._discretize(threat_data.get("level", 0.5), 5)
        distance = self._discretize(threat_data.get("distance", 1000), 5)
        speed = self._discretize(threat_data.get("speed", 0), 3)
        
        # Create state string
        return f"{threat_type}_{threat_level}_{distance}_{speed}"
    
    def _discretize(self, value: float, bins: int) -> int:
        """Discretize a continuous value into bins."""
        return min(bins - 1, max(0, int(value * bins)))
    
    def _explore_strategy(self) -> Dict[str, float]:
        """Generate an exploratory strategy with variations from current."""
        strategy = {}
        for param, value in self.defense_params.items():
            # Add random variation within bounds
            strategy[param] = max(0.0, min(1.0, value + random.uniform(-0.2, 0.2)))
        return strategy
    
    def _get_best_strategy(self, state: str) -> Dict[str, float]:
        """Get best known strategy for the given state."""
        if state not in self.q_values:
            # No experience with this state, use current strategy
            return dict(self.defense_params)
            
        # Find action with highest Q-value
        best_action = max(self.q_values[state].items(), key=lambda x: x[1])[0]
        return self._decode_action(best_action)
    
    def _incorporate_adversarial_knowledge(self, strategy: Dict[str, float], 
                                         threat_data: Dict[str, Any]) -> None:
        """Incorporate knowledge from adversarial learner."""
        if not self.adversarial_learner:
            return
            
        # Get adversarial strategy
        adv_strategy = self.adversarial_learner.strategy
        
        # Adjust defense based on adversarial knowledge
        if "defense" in adv_strategy and adv_strategy["defense"] > 0.7:
            # Increase defensive measures when adversary is defense-focused
            strategy["evasion_intensity"] = min(1.0, strategy["evasion_intensity"] * 1.2)
            strategy["countermeasure_usage"] = min(1.0, strategy["countermeasure_usage"] * 1.1)
    
    def update_from_outcome(self, 
                          threat_data: Dict[str, Any], 
                          strategy_used: Dict[str, float],
                          outcome: Dict[str, Any]) -> None:
        """
        Update learning based on defense outcome.
        
        Args:
            threat_data: Data about the threat
            strategy_used: Defense strategy that was used
            outcome: Outcome data including success metrics
        """
        # Calculate reward from outcome
        reward = self._calculate_reward(outcome)
        
        # Store experience
        state = self._encode_state(threat_data)
        action = self._encode_action(strategy_used)
        next_state = self._encode_state(outcome.get("next_threat", {}))
        
        self.experience_buffer.append((state, action, reward, next_state))
        
        # Update Q-values
        self._update_q_values(state, action, reward, next_state)
        
        # Update current defense parameters
        self.defense_params = strategy_used
        
        # Decay exploration rate over time
        self.exploration_rate = max(0.05, self.exploration_rate * 0.995)
        
        logger.info(f"Defense outcome: reward={reward:.2f}, exploration={self.exploration_rate:.2f}")
    
    def _calculate_reward(self, outcome: Dict[str, Any]) -> float:
        """Calculate reward from defense outcome."""
        # Base reward on success metrics
        success_rate = outcome.get("success_rate", 0.0)
        damage_avoided = outcome.get("damage_avoided", 0.0)
        resource_efficiency = outcome.get("resource_efficiency", 0.5)
        
        # Weighted reward calculation
        reward = (success_rate * 0.6 + 
                 damage_avoided * 0.3 + 
                 resource_efficiency * 0.1)
        
        return reward
    
    def _encode_action(self, strategy: Dict[str, float]) -> str:
        """Encode strategy as discrete action string."""
        # Discretize each parameter to create action string
        action_parts = []
        for param, value in sorted(strategy.items()):
            discrete_value = self._discretize(value, 5)
            action_parts.append(f"{param[:3]}{discrete_value}")
        
        return "_".join(action_parts)
    
    def _decode_action(self, action: str) -> Dict[str, float]:
        """Decode action string to strategy dict."""
        if not action or "_" not in action:
            return dict(self.defense_params)
            
        strategy = {}
        parts = action.split("_")
        
        # Map back to original parameters
        param_map = {
            "eva": "evasion_intensity",
            "cou": "countermeasure_usage",
            "ste": "stealth_mode",
            "for": "formation_spacing",
            "sen": "sensor_sensitivity"
        }
        
        for part in parts:
            if len(part) >= 4:
                prefix = part[:3]
                if prefix in param_map:
                    value = int(part[3]) / 4.0  # Convert back to [0,1] range
                    strategy[param_map[prefix]] = value
        
        # Fill in missing values
        for param in self.defense_params:
            if param not in strategy:
                strategy[param] = self.defense_params[param]
                
        return strategy
    
    def _update_q_values(self, state: str, action: str, reward: float, next_state: str) -> None:
        """Update Q-values using Q-learning algorithm."""
        # Initialize state in Q-table if needed
        if state not in self.q_values:
            self.q_values[state] = {}
        if next_state not in self.q_values:
            self.q_values[next_state] = {}
        
        # Initialize action in Q-table if needed
        if action not in self.q_values[state]:
            self.q_values[state][action] = 0.0
            
        # Find max Q-value for next state
        next_q_max = max(self.q_values[next_state].values()) if self.q_values[next_state] else 0.0
        
        # Q-learning update
        current_q = self.q_values[state][action]
        self.q_values[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q_max - current_q
        )

# Example usage
if __name__ == "__main__":
    # Create defense learner
    defense_learner = DefenseReinforcementLearner()
    
    # Example threat
    threat = {
        "type": "missile",
        "level": 0.8,
        "distance": 2000,
        "speed": 0.7
    }
    
    # Get defense strategy
    strategy = defense_learner.observe_threat(threat)
    print(f"Defense strategy: {strategy}")
    
    # Simulate outcome
    outcome = {
        "success_rate": 0.75,
        "damage_avoided": 0.8,
        "resource_efficiency": 0.6,
        "next_threat": {
            "type": "missile",
            "level": 0.6,
            "distance": 3000,
            "speed": 0.5
        }
    }
    
    # Update from outcome
    defense_learner.update_from_outcome(threat, strategy, outcome)