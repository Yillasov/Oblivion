#!/usr/bin/env python3
"""
Proactive decision-making algorithms for UCAV missions.
Enables anticipatory actions based on predictive behavior models.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import time

from src.swarm.predictive_behavior_modeling import PredictiveBehaviorModel

logger = logging.getLogger(__name__)

class ProactiveDecisionMaker:
    """
    Proactive decision-making system for UCAVs.
    Uses predictive models to make anticipatory decisions.
    """
    
    def __init__(self, prediction_model: PredictiveBehaviorModel, lookahead_time: float = 5.0):
        self.prediction_model = prediction_model
        self.lookahead_time = lookahead_time  # seconds
        self.decision_threshold = 0.7
        self.last_decision_time = 0
        self.decision_cooldown = 2.0  # seconds
    
    def evaluate_situation(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate current situation and predict future states.
        
        Args:
            current_data: Current sensor and mission data
            
        Returns:
            Situation assessment with threat probabilities
        """
        if not current_data:
            logger.warning("Empty data provided to situation evaluator")
            return {
                'predicted_position': np.zeros(3),
                'time_to_intercept': float('inf'),
                'threat_level': 0.0,
                'distance': float('inf'),
                'intercept_vector': np.zeros(3),
                'error': 'No data provided'
            }
        
        try:
            # Predict future target position
            future_position = self.prediction_model.predict(current_data)
            
            # Calculate distance to predicted position
            current_position = np.array(current_data.get('own_position', [0, 0, 0]))
            distance_vector = future_position - current_position
            distance = np.linalg.norm(distance_vector)
            
            # Calculate time to intercept
            own_speed = current_data.get('own_speed', 0.1)
            time_to_intercept = distance / max(own_speed, 0.1)
            
            # Assess threat level
            threat_level = self._calculate_threat_level(current_data, future_position)
            
            return {
                'predicted_position': future_position.tolist() if isinstance(future_position, np.ndarray) else future_position,
                'time_to_intercept': time_to_intercept,
                'threat_level': threat_level,
                'distance': distance,
                'intercept_vector': (distance_vector / max(distance, 0.1)).tolist() if isinstance(distance_vector, np.ndarray) else distance_vector
            }
        except Exception as e:
            logger.error(f"Error evaluating situation: {str(e)}")
            return {
                'error': f"Evaluation failed: {str(e)}",
                'predicted_position': np.zeros(3).tolist(),
                'time_to_intercept': float('inf'),
                'threat_level': 0.0,
                'distance': float('inf'),
                'intercept_vector': np.zeros(3).tolist()
            }
    
    def _calculate_threat_level(self, current_data: Dict[str, Any], 
                               predicted_position: np.ndarray) -> float:
        """Calculate threat level based on current data and predictions."""
        # Extract relevant data
        target_type = current_data.get('target_type', 'unknown')
        target_velocity = current_data.get('velocity', 0)
        target_heading = current_data.get('heading', 0)
        
        # Base threat level on target type
        base_threat = {
            'civilian': 0.1,
            'unknown': 0.5,
            'military': 0.7,
            'hostile': 0.9
        }.get(target_type, 0.5)
        
        # Adjust for velocity (faster = more threatening)
        velocity_factor = min(1.0, target_velocity / 500)
        
        # Adjust for heading (toward = more threatening)
        own_position = np.array(current_data.get('own_position', [0, 0, 0]))
        heading_vector = np.array([
            np.cos(np.radians(target_heading)),
            np.sin(np.radians(target_heading)),
            0
        ])
        to_own_vector = own_position - predicted_position
        to_own_vector = to_own_vector / max(np.linalg.norm(to_own_vector), 0.1)
        heading_alignment = np.dot(heading_vector, to_own_vector)
        heading_factor = (heading_alignment + 1) / 2  # Scale from [-1,1] to [0,1]
        
        # Combined threat level
        threat_level = 0.4 * base_threat + 0.3 * velocity_factor + 0.3 * heading_factor
        return min(1.0, max(0.0, threat_level))
    
    def make_decision(self, situation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Make proactive decision based on situation assessment.
        
        Args:
            situation: Assessed situation with predictions
            
        Returns:
            Decision type and parameters
        """
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_decision_time < self.decision_cooldown:
            return "maintain", {"reason": "cooldown"}
        
        threat_level = situation['threat_level']
        time_to_intercept = situation['time_to_intercept']
        distance = situation['distance']
        
        # Decision logic
        if threat_level > 0.8:
            if time_to_intercept < self.lookahead_time:
                self.last_decision_time = current_time
                return "engage", {
                    "target_position": situation['predicted_position'],
                    "urgency": "high",
                    "confidence": threat_level
                }
            else:
                return "track", {
                    "target_position": situation['predicted_position'],
                    "urgency": "medium",
                    "confidence": threat_level
                }
        elif threat_level > 0.5:
            if time_to_intercept < self.lookahead_time * 1.5:
                self.last_decision_time = current_time
                return "investigate", {
                    "target_position": situation['predicted_position'],
                    "urgency": "medium",
                    "confidence": threat_level
                }
            else:
                return "monitor", {
                    "target_position": situation['predicted_position'],
                    "urgency": "low",
                    "confidence": threat_level
                }
        elif distance < 1000:  # Close but not threatening
            return "evade", {
                "direction": -situation['intercept_vector'],
                "urgency": "medium",
                "confidence": 1.0 - threat_level
            }
        else:
            return "patrol", {
                "area": "current",
                "urgency": "low",
                "confidence": 1.0 - threat_level
            }

# Example usage
if __name__ == "__main__":
    # Create predictive model
    prediction_model = PredictiveBehaviorModel()
    
    # Example historical data for training
    historical_data = [
        {'velocity': 300, 'acceleration': 5, 'heading': 90, 'altitude': 1000, 
         'time_of_day': 12, 'target_position': [100, 200, 300]},
        {'velocity': 320, 'acceleration': 4, 'heading': 85, 'altitude': 1050, 
         'time_of_day': 13, 'target_position': [110, 210, 310]},
        {'velocity': 310, 'acceleration': 3, 'heading': 88, 'altitude': 1020, 
         'time_of_day': 14, 'target_position': [120, 220, 320]},
    ]
    
    # Train model
    prediction_model.train(historical_data)
    
    # Create proactive decision maker
    decision_maker = ProactiveDecisionMaker(prediction_model)
    
    # Current situation
    current_data = {
        'velocity': 330, 
        'acceleration': 6, 
        'heading': 92, 
        'altitude': 1100,
        'time_of_day': 15,
        'own_position': [0, 0, 0],
        'own_speed': 400,
        'target_type': 'military'
    }
    
    # Evaluate situation
    situation = decision_maker.evaluate_situation(current_data)
    
    # Make decision
    decision, params = decision_maker.make_decision(situation)
    
    logger.info(f"Situation assessment: {situation}")
    logger.info(f"Proactive decision: {decision}, params: {params}")