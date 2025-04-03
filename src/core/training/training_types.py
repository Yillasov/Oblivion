"""
Training Types

Common type definitions for the training system.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable


class TrainingMode(Enum):
    """Training mode enumeration."""
    ONLINE = "online"
    OFFLINE = "offline"
    HYBRID = "hybrid"
    TRANSFER = "transfer"


class TrainingMetrics:
    """Training metrics container."""
    
    def __init__(self):
        """Initialize training metrics."""
        self.loss_history = []
        self.accuracy_history = []
        self.validation_loss_history = []
        self.validation_accuracy_history = []
        self.learning_rate_history = []
        self.custom_metrics = {}
        
    def add_loss(self, loss: float, validation: bool = False):
        """Add loss value to history."""
        if validation:
            self.validation_loss_history.append(loss)
        else:
            self.loss_history.append(loss)
            
    def add_accuracy(self, accuracy: float, validation: bool = False):
        """Add accuracy value to history."""
        if validation:
            self.validation_accuracy_history.append(accuracy)
        else:
            self.accuracy_history.append(accuracy)
            
    def add_learning_rate(self, lr: float):
        """Add learning rate to history."""
        self.learning_rate_history.append(lr)
        
    def add_custom_metric(self, name: str, value: float):
        """Add custom metric value."""
        if name not in self.custom_metrics:
            self.custom_metrics[name] = []
        self.custom_metrics[name].append(value)