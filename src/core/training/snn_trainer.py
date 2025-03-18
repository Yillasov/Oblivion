"""
Enhanced SNN Trainer with improved modularity and training capabilities.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import time
from dataclasses import dataclass
from enum import Enum

from src.core.training.data_preprocessing import NeuromorphicPreprocessor, EncodingType
from src.core.training.snn_learning_rules import STDPLearningRule, RSTDPLearningRule
from src.core.training.snn_loss_functions import spike_count_loss, van_rossum_distance
from src.core.utils.logging_framework import get_logger

logger = get_logger("snn_trainer")

class LearningRule(Enum):
    STDP = "stdp"
    RSTDP = "rstdp"

class LossFunction(Enum):
    SPIKE_COUNT = "spike_count"
    VAN_ROSSUM = "van_rossum"

@dataclass
class TrainingMetrics:
    loss_history: List[float]
    accuracy_history: List[float]
    training_time: float
    best_epoch: int
    best_loss: float

class SNNTrainer:
    """Enhanced trainer for Spiking Neural Networks with improved features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._setup_training_parameters()
        self._setup_learning_components()
        
    def _setup_training_parameters(self) -> None:
        """Initialize training parameters."""
        self.epochs = self.config.get("epochs", 10)
        self.batch_size = self.config.get("batch_size", 32)
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.time_step = self.config.get("time_step", 1.0)
        self.early_stopping_patience = self.config.get("early_stopping_patience", 5)
        self.min_delta = self.config.get("min_delta", 1e-4)
        
    def _setup_learning_components(self) -> None:
        """Initialize learning rules and loss functions."""
        # Setup learning rule
        learning_rule = LearningRule(self.config.get("learning_rule", "stdp"))
        learning_params = self.config.get("learning_params", {})
        
        self.learning_rule = {
            LearningRule.STDP: lambda: STDPLearningRule(learning_params),
            LearningRule.RSTDP: lambda: RSTDPLearningRule(learning_params)
        }.get(learning_rule, lambda: STDPLearningRule(learning_params))()
        
        # Setup loss function
        loss_function = LossFunction(self.config.get("loss_function", "spike_count"))
        tau = self.config.get("van_rossum_tau", 10.0)
        
        self.loss_fn = {
            LossFunction.SPIKE_COUNT: spike_count_loss,
            LossFunction.VAN_ROSSUM: lambda x, y: van_rossum_distance(x, y, tau, self.time_step)
        }.get(loss_function, spike_count_loss)
        
        # Initialize preprocessor
        self.preprocessor = NeuromorphicPreprocessor(self.config.get("preprocessor", {}))
        
    def _should_stop_early(self, losses: List[float], patience: int, min_delta: float) -> bool:
        """Check if training should stop early."""
        if len(losses) < patience:
            return False
        
        recent_losses = losses[-patience:]
        min_loss = min(recent_losses)
        return all(loss - min_loss < min_delta for loss in recent_losses[1:])
    
    def train(self, model: Any, dataset: Any) -> TrainingMetrics:
        """
        Train an SNN model with enhanced monitoring and early stopping.
        
        Args:
            model: SNN model to train
            dataset: Training dataset
            
        Returns:
            TrainingMetrics: Detailed training metrics
        """
        logger.info("Starting enhanced SNN training")
        start_time = time.time()
        
        # Preprocess dataset
        processed_data = self.preprocessor.preprocess_dataset(dataset)
        inputs, targets = processed_data.inputs, processed_data.targets
        
        if inputs is None or targets is None:
            raise ValueError("No training data available")
        
        num_samples = len(inputs)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        
        metrics = TrainingMetrics(
            loss_history=[],
            accuracy_history=[],
            training_time=0.0,
            best_epoch=0,
            best_loss=float('inf')
        )
        
        # Training loop with early stopping
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            inputs_shuffled = inputs[indices]
            targets_shuffled = targets[indices]
            
            epoch_loss = self._train_epoch(
                model, inputs_shuffled, targets_shuffled, num_batches
            )
            
            metrics.loss_history.append(epoch_loss)
            
            # Update best metrics
            if epoch_loss < metrics.best_loss:
                metrics.best_loss = epoch_loss
                metrics.best_epoch = epoch
                self._save_model_state(model)
            
            # Calculate accuracy if model supports it
            if hasattr(model, "evaluate"):
                accuracy = model.evaluate(inputs, targets)
                metrics.accuracy_history.append(accuracy)
                logger.info(f"Epoch {epoch + 1}/{self.epochs}: "
                          f"loss={epoch_loss:.4f}, accuracy={accuracy:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}: loss={epoch_loss:.4f}")
            
            # Early stopping check
            if self._should_stop_early(
                metrics.loss_history, 
                self.early_stopping_patience, 
                self.min_delta
            ):
                logger.info("Early stopping triggered")
                break
        
        metrics.training_time = time.time() - start_time
        logger.info(f"Training completed in {metrics.training_time:.2f}s")
        
        # Restore best model state
        self._restore_model_state(model)
        
        return metrics
    
    def _train_epoch(self, model: Any, inputs: np.ndarray, 
                    targets: np.ndarray, num_batches: int) -> float:
        """Train one epoch and return epoch loss."""
        epoch_loss = 0.0
        
        for batch_idx in range(num_batches):
            batch_loss = self._train_batch(model, inputs, targets, batch_idx)
            epoch_loss += batch_loss
            
            if num_batches > 10 and (batch_idx + 1) % (num_batches // 10) == 0:
                logger.debug(f"Processed {batch_idx + 1}/{num_batches} batches")
        
        return epoch_loss / num_batches
    
    def _train_batch(self, model: Any, inputs: np.ndarray, 
                    targets: np.ndarray, batch_idx: int) -> float:
        """Train one batch and return batch loss."""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(inputs))
        
        batch_inputs = inputs[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]
        batch_loss = 0.0
        
        for input_data, target_data in zip(batch_inputs, batch_targets):
            # Forward pass
            output_spikes = model.forward(input_data)
            
            # Calculate loss
            sample_loss = self.loss_fn(output_spikes, target_data)
            batch_loss += sample_loss
            
            # Update weights for each layer
            self._update_layer_weights(model, input_data, target_data, output_spikes)
        
        return batch_loss / (end_idx - start_idx)
    
    def _update_layer_weights(self, model: Any, input_data: np.ndarray,
                            target_data: np.ndarray, output_spikes: np.ndarray) -> None:
        """Update weights for all layers."""
        for layer_idx in range(len(model.layers) - 1):
            pre_layer = model.layers[layer_idx]
            post_layer = model.layers[layer_idx + 1]
            
            pre_spikes = pre_layer.spike_history
            post_spikes = post_layer.spike_history
            
            if isinstance(self.learning_rule, RSTDPLearningRule):
                reward = model.get_reward(target_data, output_spikes)
                model.weights[layer_idx] = self.learning_rule.update_weights(
                    model.weights[layer_idx], pre_spikes, post_spikes,
                    reward, self.time_step
                )
            else:
                model.weights[layer_idx] = self.learning_rule.update_weights(
                    model.weights[layer_idx], pre_spikes, post_spikes,
                    self.time_step
                )
    
    def _save_model_state(self, model: Any) -> None:
        """Save best model state."""
        if hasattr(model, "get_state"):
            self.best_state = model.get_state()
    
    def _restore_model_state(self, model: Any) -> None:
        """Restore best model state."""
        if hasattr(model, "set_state") and hasattr(self, "best_state"):
            model.set_state(self.best_state)