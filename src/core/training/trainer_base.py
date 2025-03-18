"""
Neuromorphic Training Framework

Provides base classes and utilities for training spiking neural networks
on neuromorphic hardware.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable, Union, TYPE_CHECKING
import time
import uuid
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

# Add conditional import to avoid circular imports
if TYPE_CHECKING:
    from src.core.integration.neuromorphic_system import NeuromorphicSystem

from src.core.utils.logging_framework import get_logger
from src.core.hardware.hardware_abstraction import NeuromorphicHardware, HardwareFactory
from src.core.neuromorphic.primitives import NeuronModel
from src.core.training.session_manager import SessionManager, TrainingSession
from src.core.training.optimization import OptimizerRegistry, OptimizationAlgorithm, SGDOptimizer

logger = get_logger("neuromorphic_trainer")


class TrainingMode(Enum):
    """Training modes for neuromorphic networks."""
    ONLINE = "online"
    OFFLINE = "offline"
    HYBRID = "hybrid"
    TRANSFER = "transfer"


@dataclass
class TrainingConfig:
    """Configuration for neuromorphic training."""
    learning_rate: float = 0.01
    batch_size: int = 1
    epochs: int = 1
    mode: TrainingMode = TrainingMode.ONLINE
    hardware_type: str = "simulated"
    checkpoint_interval: int = 0  # 0 means no checkpointing
    early_stopping: bool = False
    patience: int = 5
    validation_split: float = 0.2
    shuffle: bool = True
    seed: Optional[int] = None
    optimizer: str = "sgd"  # Added optimizer field
    optimizer_params: Dict[str, Any] = field(default_factory=dict)  # Added optimizer params
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    training_time: float = 0.0
    epochs_completed: int = 0
    best_epoch: int = 0
    best_loss: float = float('inf')
    best_accuracy: float = 0.0


# Add these imports at the top of the file
import os
from dataclasses import asdict

# Add this import at the top of the file
from src.core.training.checkpoint_manager import CheckpointManager
from src.core.utils.model_serialization import ModelSerializer


class NeuromorphicTrainer(ABC):
    """
    Base class for training neuromorphic networks.
    
    This class provides the foundation for implementing training algorithms
    for spiking neural networks on neuromorphic hardware.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the neuromorphic trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.hardware = None
        self.model = None
        self.metrics = TrainingMetrics()
        self.training_id = str(uuid.uuid4())
        self.is_initialized = False
        self.current_epoch = 0
        self.best_weights = None
        self.session_dir = None
        self.session_manager = SessionManager()
        self.current_session = None
        
        # Add checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_interval=self.config.checkpoint_interval if hasattr(self.config, 'checkpoint_interval') else 10
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Set random seed if provided
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        logger.info(f"Initialized neuromorphic trainer with ID {self.training_id}")
    
    def initialize(self, model: Any, hardware: Optional[NeuromorphicHardware] = None) -> bool:
        """
        Initialize the trainer with a model and hardware.
        
        Args:
            model: Neural network model to train
            hardware: Optional hardware instance (will create based on config if None)
            
        Returns:
            bool: Success status
        """
        try:
            self.model = model
            
            # Initialize hardware if not provided
            if hardware is None:
                hardware_factory = HardwareFactory()
                self.hardware = hardware_factory.create_hardware(self.config.hardware_type)
                if self.hardware is None:
                    logger.error(f"Failed to create hardware of type {self.config.hardware_type}")
                    return False
                
                # Initialize the hardware
                if not self.hardware.initialize():
                    logger.error("Failed to initialize hardware")
                    return False
            else:
                self.hardware = hardware
            
            # Perform hardware-specific model initialization
            if not self._initialize_model_on_hardware():
                logger.error("Failed to initialize model on hardware")
                return False
            
            self.is_initialized = True
            logger.info(f"Trainer {self.training_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trainer: {str(e)}")
            return False
    
    @abstractmethod
    def _initialize_model_on_hardware(self) -> bool:
        """
        Initialize the model on the hardware.
        
        This method should be implemented by subclasses to handle
        hardware-specific model initialization.
        
        Returns:
            bool: Success status
        """
        pass
    
    def start_session(self, name: str) -> str:
        """
        Start a new training session.
        
        Args:
            name: Session name
            
        Returns:
            str: Session ID
        """
        self.current_session = self.session_manager.create_session(
            name=name,
            config=self.config,
            hardware_type=self.config.hardware_type
        )
        logger.info(f"Started training session: {name} ({self.current_session.session_id})")
        return self.current_session.session_id
    
    def update_session(self) -> None:
        """Update current session with latest metrics."""
        if self.current_session:
            self.session_manager.update_session(
                self.current_session.session_id,
                self.metrics,
                status="running"
            )
    
    def create_session(self, name: str, base_dir: str = "/Users/yessine/Oblivion/training_sessions") -> str:
        """
        Create a new training session.
        
        Args:
            name: Session name
            base_dir: Base directory for sessions
            
        Returns:
            str: Session directory path
        """
        # Create session directory with timestamp and name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_name = f"{timestamp}_{name}_{self.training_id[:8]}"
        self.session_dir = os.path.join(base_dir, session_name)
        
        # Create directories
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "logs"), exist_ok=True)
        
        # Save initial configuration
        self._save_session_config()
        
        logger.info(f"Created training session: {name} in {self.session_dir}")
        return self.session_dir
    
    def _save_session_config(self) -> None:
        """Save session configuration to file."""
        if not self.session_dir:
            return
            
        try:
            import json
            
            # Create config data
            config_data = {
                "training_id": self.training_id,
                "timestamp": time.time(),
                "config": asdict(self.config),
                "hardware_type": self.config.hardware_type
            }
            
            # Save to file
            config_path = os.path.join(self.session_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.debug(f"Saved session configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving session configuration: {str(e)}")
    
    def _save_session_metrics(self) -> None:
        """Save current metrics to session."""
        if not self.session_dir:
            return
            
        try:
            import json
            
            # Create metrics data
            metrics_data = asdict(self.metrics)
            metrics_data["timestamp"] = time.time()
            
            # Save to file
            metrics_path = os.path.join(self.session_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            logger.debug(f"Saved session metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving session metrics: {str(e)}")
    
    def _checkpoint(self, epoch: int) -> None:
        """
        Create a checkpoint of the current model state.
        
        Args:
            epoch: Current epoch number
        """
        if not self.session_dir:
            logger.warning("Cannot create checkpoint: No active session")
            return
            
        try:
            # Create checkpoint path
            checkpoint_path = os.path.join(
                self.session_dir, 
                "checkpoints", 
                f"checkpoint_epoch_{epoch+1}.ckpt"
            )
            
            # Save checkpoint
            self._save_checkpoint(checkpoint_path)
            
            # Save current metrics
            self._save_session_metrics()
            
            logger.info(f"Created checkpoint at epoch {epoch+1}: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error creating checkpoint: {str(e)}")
    
    def _save_checkpoint(self, path: str) -> bool:
        """
        Save model checkpoint to file.
        
        Args:
            path: Path to save checkpoint
            
        Returns:
            bool: Success status
        """
        try:
            # Create checkpoint directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Create checkpoint data
            checkpoint_data = {
                "model": self.model,
                "metrics": asdict(self.metrics),
                "training_id": self.training_id,
                "current_epoch": self.current_epoch,
                "hardware_type": self.config.hardware_type,
                "timestamp": time.time()
            }
            
            # Determine best format based on model type
            format = "pickle"  # Default format
            
            # Serialize checkpoint data
            success = ModelSerializer.serialize(checkpoint_data, path, format)
            
            if success:
                logger.info(f"Saved checkpoint to {path}")
                
                # If this is the best model so far, save it separately
                if self.best_weights is None or self.metrics.accuracy_history[-1] > max(self.metrics.accuracy_history[:-1], default=0):
                    self.best_weights = self.model
                    best_path = os.path.join(os.path.dirname(path), "best_model.ckpt")
                    ModelSerializer.serialize({"model": self.model}, best_path, format)
                    logger.info(f"Saved best model to {best_path}")
                    
            return success
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            return False
    
    def load_checkpoint(self, path: str) -> bool:
        """
        Load model checkpoint from file.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            bool: Success status
        """
        try:
            # Determine format based on file extension
            format = "pickle"  # Default format
            
            # Deserialize checkpoint data
            checkpoint_data = ModelSerializer.deserialize(path, format)
            
            if checkpoint_data is None:
                logger.error(f"Failed to load checkpoint from {path}")
                return False
                
            # Restore model and training state
            self.model = checkpoint_data.get("model")
            
            # Restore metrics if available
            if "metrics" in checkpoint_data:
                metrics_dict = checkpoint_data["metrics"]
                for key, value in metrics_dict.items():
                    if hasattr(self.metrics, key):
                        setattr(self.metrics, key, value)
            
            # Restore other training state
            self.current_epoch = checkpoint_data.get("current_epoch", 0)
            
            # Re-initialize model on hardware if needed
            if self.hardware and self.model:
                self._initialize_model_on_hardware()
                
            logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
            return True
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return False
    
    def resume_session(self, session_dir: str) -> bool:
        """
        Resume training from a previous session.
        
        Args:
            session_dir: Path to session directory
            
        Returns:
            bool: Success status
        """
        if not os.path.exists(session_dir):
            logger.error(f"Session directory not found: {session_dir}")
            return False
            
        try:
            import json
            
            # Set session directory
            self.session_dir = session_dir
            
            # Load configuration
            config_path = os.path.join(session_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                # Set training ID
                self.training_id = config_data.get("training_id", self.training_id)
                
                # Load metrics if available
                metrics_path = os.path.join(session_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics_data = json.load(f)
                    
                    # Update metrics
                    self.metrics.loss_history = metrics_data.get("loss_history", [])
                    self.metrics.accuracy_history = metrics_data.get("accuracy_history", [])
                    self.metrics.training_time = metrics_data.get("training_time", 0.0)
                    self.metrics.epochs_completed = metrics_data.get("epochs_completed", 0)
                    self.metrics.best_epoch = metrics_data.get("best_epoch", 0)
                    self.metrics.best_loss = metrics_data.get("best_loss", float('inf'))
                    self.metrics.best_accuracy = metrics_data.get("best_accuracy", 0.0)
                
                # Find latest checkpoint
                checkpoint_dir = os.path.join(session_dir, "checkpoints")
                if os.path.exists(checkpoint_dir):
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
                    if checkpoints:
                        # Sort by epoch number
                        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                        latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
                        
                        # Load checkpoint
                        if not self.load_checkpoint(latest_checkpoint):
                            logger.warning(f"Failed to load checkpoint: {latest_checkpoint}")
                
                logger.info(f"Resumed training session from {session_dir}")
                return True
            else:
                logger.error(f"Configuration file not found in session directory")
                return False
                
        except Exception as e:
            logger.error(f"Error resuming session: {str(e)}")
            return False
    
    def train(self, training_data: Any, validation_data: Optional[Any] = None, 
             session_name: Optional[str] = None) -> TrainingMetrics:
        """
        Train the model on the provided data.
        
        Args:
            training_data: Training dataset
            validation_data: Optional validation dataset
            session_name: Optional session name (creates a new session if provided)
            
        Returns:
            TrainingMetrics: Training metrics
        """
        if not self.is_initialized:
            logger.error("Trainer not initialized. Call initialize() first.")
            return self.metrics
        
        # Create session if name provided and no active session
        if session_name and not self.session_dir:
            self.create_session(session_name)
        
        start_time = time.time()
        
        try:
            # Extract inputs and targets from training data
            if isinstance(training_data, dict):
                inputs = training_data.get("inputs")
                targets = training_data.get("targets")
            else:
                # Assume training_data is a tuple or list of (inputs, targets)
                inputs, targets = training_data
            
            # Prepare validation data if provided
            val_inputs, val_targets = None, None
            if validation_data is not None:
                if isinstance(validation_data, dict):
                    val_inputs = validation_data.get("inputs")
                    val_targets = validation_data.get("targets")
                else:
                    val_inputs, val_targets = validation_data
            # If validation data not provided but validation split is set, create validation set
            elif self.config.validation_split > 0 and inputs is not None:
                split_idx = int(len(inputs) * (1 - self.config.validation_split))
                if self.config.shuffle:
                    # Create shuffled indices
                    indices = np.random.permutation(len(inputs))
                    train_indices = indices[:split_idx]
                    val_indices = indices[split_idx:]
                    
                    # Split data
                    val_inputs = inputs[val_indices]
                    val_targets = targets[val_indices] if targets is not None else None
                    inputs = inputs[train_indices]
                    targets = targets[train_indices] if targets is not None else None
                else:
                    # Split without shuffling
                    val_inputs = inputs[split_idx:]
                    val_targets = targets[split_idx:] if targets is not None else None
                    inputs = inputs[:split_idx]
                    targets = targets[:split_idx] if targets is not None else None
            
            # Log training information
            if inputs is not None:
                logger.info(f"Starting training with {len(inputs)} samples")
            else:
                logger.warning("No training samples provided")
            if val_inputs is not None:
                logger.info(f"Using validation set with {len(val_inputs)} samples")
            
            # Initialize training variables
            batch_size = self.config.batch_size
            num_samples = len(inputs) if inputs is not None else 0
            num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
            best_val_loss = float('inf')
            patience_counter = 0
            
            # Training loop
            for epoch in range(self.current_epoch, self.current_epoch + self.config.epochs):
                epoch_start_time = time.time()
                self.current_epoch = epoch
                
                # Shuffle data if configured
                if self.config.shuffle:
                    shuffle_indices = np.random.permutation(num_samples)
                    inputs_shuffled = inputs[shuffle_indices] if inputs is not None else None
                    targets_shuffled = targets[shuffle_indices] if targets is not None else None
                else:
                    inputs_shuffled = inputs
                    targets_shuffled = targets
                
                # Initialize epoch metrics
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                
                # Process batches
                for batch_idx in range(num_batches):
                    # Get batch data
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_inputs = inputs_shuffled[start_idx:end_idx] if inputs_shuffled is not None else None
                    batch_targets = targets_shuffled[start_idx:end_idx] if targets_shuffled is not None else None
                    
                    # Process batch (hardware-specific implementation)
                    batch_metrics = self._process_batch(batch_inputs, batch_targets)
                    
                    # Update epoch metrics
                    epoch_loss += batch_metrics.get("loss", 0.0) * (end_idx - start_idx)
                    epoch_accuracy += batch_metrics.get("accuracy", 0.0) * (end_idx - start_idx)
                    
                    # Log progress for large datasets
                    if num_batches > 10 and (batch_idx + 1) % (num_batches // 10) == 0:
                        logger.debug(f"Epoch {epoch+1}/{self.current_epoch + self.config.epochs}: "
                                    f"Processed {batch_idx+1}/{num_batches} batches")
                
                # Calculate average epoch metrics
                epoch_loss /= num_samples
                epoch_accuracy /= num_samples
                
                # Evaluate on validation set if available
                val_loss, val_accuracy = 0.0, 0.0
                if val_inputs is not None and val_targets is not None:
                    val_metrics = self._evaluate(val_inputs, val_targets)
                    val_loss = val_metrics.get("loss", 0.0)
                    val_accuracy = val_metrics.get("accuracy", 0.0)
                    
                    # Update metrics with validation results
                    self.metrics.loss_history.append(val_loss)
                    self.metrics.accuracy_history.append(val_accuracy)
                    
                    # Check for best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.metrics.best_loss = val_loss
                        self.metrics.best_accuracy = val_accuracy
                        self.metrics.best_epoch = epoch
                        self.best_weights = self._get_model_weights()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:
                    # Without validation, use training metrics
                    self.metrics.loss_history.append(epoch_loss)
                    self.metrics.accuracy_history.append(epoch_accuracy)
                    
                    # Check for best model
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        self.metrics.best_loss = epoch_loss
                        self.metrics.best_accuracy = epoch_accuracy
                        self.metrics.best_epoch = epoch
                        self.best_weights = self._get_model_weights()
                
                # Update training metrics
                self.metrics.epochs_completed = epoch + 1
                self.metrics.training_time = time.time() - start_time
                
                # Log epoch results
                epoch_time = time.time() - epoch_start_time
                if val_inputs is not None:
                    logger.info(f"Epoch {epoch+1}/{self.current_epoch + self.config.epochs}: "
                               f"loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}, "
                               f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}, "
                               f"time={epoch_time:.2f}s")
                else:
                    logger.info(f"Epoch {epoch+1}/{self.current_epoch + self.config.epochs}: "
                               f"loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}, "
                               f"time={epoch_time:.2f}s")
                
                # Create checkpoint if configured
                if self.config.checkpoint_interval > 0 and (epoch + 1) % self.config.checkpoint_interval == 0:
                    self._checkpoint(epoch)
                
                # Update session if active
                if self.current_session:
                    self.update_session()
                
                # Early stopping if configured
                if self.config.early_stopping and patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Restore best weights if validation was used
            if self.best_weights is not None:
                self._set_model_weights(self.best_weights)
                logger.info(f"Restored best model from epoch {self.metrics.best_epoch+1}")
            
            # Final evaluation
            if val_inputs is not None and val_targets is not None:
                final_metrics = self._evaluate(val_inputs, val_targets)
                logger.info(f"Final validation: loss={final_metrics.get('loss', 0.0):.4f}, "
                           f"accuracy={final_metrics.get('accuracy', 0.0):.4f}")
            
            # Update total training time
            self.metrics.training_time = time.time() - start_time
            logger.info(f"Training completed in {self.metrics.training_time:.2f}s")
            
            # Save final metrics if session is active
            if self.session_dir:
                self._save_session_metrics()
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            self.metrics.training_time = time.time() - start_time
            
            # Save metrics even on error
            if self.session_dir:
                self._save_session_metrics()
                
            return self.metrics

    def _process_batch(self, inputs: Optional[np.ndarray], targets: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Process a single batch of data.
        
        This method should be implemented by subclasses to handle
        hardware-specific batch processing.
        
        Args:
            inputs: Batch input data, can be None for special cases
            targets: Batch target data, can be None for special cases
                
        Returns:
            Dict[str, float]: Batch metrics (loss, accuracy, etc.)
        """
        # Default implementation (should be overridden by subclasses)
        logger.warning("Using default _process_batch implementation. This should be overridden.")
        return {"loss": 0.0, "accuracy": 0.0}

    def _evaluate(self, inputs: Optional[np.ndarray], targets: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model on the provided data.
        
        Args:
            inputs: Input data, can be None for special cases
            targets: Target data, can be None for special cases
                
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Default implementation uses batched evaluation
        if inputs is None or targets is None:
            logger.warning("Cannot evaluate with None inputs or targets")
            return {"loss": 0.0, "accuracy": 0.0}
            
        batch_size = self.config.batch_size
        num_samples = len(inputs)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        total_loss = 0.0
        total_accuracy = 0.0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_inputs = inputs[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            # Get predictions without updating weights
            batch_metrics = self._predict_batch(batch_inputs, batch_targets)
            
            # Update metrics
            total_loss += batch_metrics.get("loss", 0.0) * (end_idx - start_idx)
            total_accuracy += batch_metrics.get("accuracy", 0.0) * (end_idx - start_idx)
        
        # Calculate average metrics
        return {
            "loss": total_loss / num_samples,
            "accuracy": total_accuracy / num_samples
        }

    def _predict_batch(self, inputs: Optional[np.ndarray], targets: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Generate predictions for a batch without updating weights.
        
        Args:
            inputs: Batch input data, can be None for special cases
            targets: Batch target data, can be None for special cases
                
        Returns:
            Dict[str, float]: Batch metrics
        """
        # Default implementation (should be overridden by subclasses)
        logger.warning("Using default _predict_batch implementation. This should be overridden.")
        return {"loss": 0.0, "accuracy": 0.0}

    def _get_model_weights(self) -> Any:
        """
        Get current model weights.
        
        Returns:
            Any: Model weights in a format that can be restored
        """
        # Default implementation (should be overridden by subclasses)
        logger.warning("Using default _get_model_weights implementation. This should be overridden.")
        return self.model

    def _set_model_weights(self, weights: Any) -> None:
        """
        Set model weights.
        
        Args:
            weights: Model weights to set
        """
        # Default implementation (should be overridden by subclasses)
        logger.warning("Using default _set_model_weights implementation. This should be overridden.")
        self.model = weights

    def _create_optimizer(self) -> OptimizationAlgorithm:
        """Create optimizer based on configuration."""
        optimizer_name = self.config.optimizer
        optimizer_params = self.config.optimizer_params.copy()
        
        # Add learning rate if not in params
        if "learning_rate" not in optimizer_params:
            optimizer_params["learning_rate"] = self.config.learning_rate
            
        # Get optimizer from registry
        optimizer = OptimizerRegistry.get_optimizer(optimizer_name, **optimizer_params)
        
        # Always return a valid optimizer (use SGD as fallback)
        if optimizer is None:
            logger.warning(f"Optimizer '{optimizer_name}' not found, using SGD")
            optimizer = OptimizerRegistry.get_optimizer("sgd", learning_rate=self.config.learning_rate)
            # Ensure we have a valid optimizer even if registry fails
            if optimizer is None:
                logger.error("Failed to create SGD optimizer, using default implementation")
                optimizer = SGDOptimizer(learning_rate=self.config.learning_rate)
                
        logger.info(f"Using optimizer: {optimizer_name}")
        return optimizer
    
    def update_parameters(self, parameters: Dict[str, Any], gradients: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update parameters using the configured optimizer.
        
        Args:
            parameters: Current parameters
            gradients: Parameter gradients
            
        Returns:
            Dict[str, Any]: Updated parameters
        """
        return self.optimizer.optimize(parameters, gradients)

    # Add this method to the NeuromorphicTrainer class
    
    def integrate_with_system(self, system: 'NeuromorphicSystem', 
                            component_name: str, model_id: Optional[str] = None) -> bool:
        """
        Integrate the trained model with a neuromorphic system component.
        
        Args:
            system: Neuromorphic system instance
            component_name: Component name to integrate with
            model_id: Optional model identifier (defaults to training_id)
            
        Returns:
            bool: Success status
        """
        if not self.model:
            logger.error("No trained model available")
            return False
        
        # Create integration if needed
        from src.core.training.training_integration import TrainingIntegration
        integration = TrainingIntegration(self, system)
        
        # Use training ID if no model ID provided
        if model_id is None:
            model_id = f"model_{self.training_id[:8]}"
        
        # Register the model
        if not integration.register_model(model_id, self.model, {
            "training_id": self.training_id,
            "epochs": self.current_epoch,
            "hardware_type": self.config.hardware_type,
            "accuracy": self.metrics.best_accuracy
        }):
            return False
        
        # Deploy to component
        return integration.deploy_model_to_component(model_id, component_name)

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get current model parameters.
        
        Returns:
            Dict[str, np.ndarray]: Model parameters
        """
        # Implementation depends on the specific model representation
        # This is a placeholder that should be overridden by subclasses
        return {}
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]) -> bool:
        """
        Set model parameters.
        
        Args:
            parameters: Model parameters
            
        Returns:
            bool: Success status
        """
        # Implementation depends on the specific model representation
        # This is a placeholder that should be overridden by subclasses
        return True
    
    def train_batch_range(self, dataset: Dict[str, np.ndarray], 
                     start_batch: int, end_batch: int) -> Dict[str, float]:
        """
        Train on a specific range of batches.
        
        Args:
            dataset: Training dataset
            start_batch: Starting batch index
            end_batch: Ending batch index (exclusive)
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # Implementation depends on the specific training process
        # This is a placeholder that should be overridden by subclasses
        return {}

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current training metrics.
        
        Returns:
            Dict[str, Any]: Training metrics
        """
        return asdict(self.metrics)


def resume_training(self, session_id: str, dataset: Dict[str, np.ndarray], 
                   additional_epochs: int = 10) -> TrainingMetrics:
    """
    Resume training from the latest checkpoint.
    
    Args:
        session_id: Training session ID
        dataset: Training dataset
        additional_epochs: Number of additional epochs to train
        
    Returns:
        TrainingMetrics: Updated training metrics
    """
    # Load latest checkpoint
    checkpoint_data = self.checkpoint_manager.load_latest_checkpoint(session_id)
    if not checkpoint_data:
        logger.error(f"No checkpoint found for session {session_id}")
        return self.metrics
    
    # Restore model and training state
    self.model = checkpoint_data.get("model")
    self.training_id = checkpoint_data.get("training_id", self.training_id)
    self.current_epoch = checkpoint_data.get("current_epoch", 0)
    
    # Restore metrics
    metrics_dict = checkpoint_data.get("metrics", {})
    for key, value in metrics_dict.items():
        if hasattr(self.metrics, key):
            setattr(self.metrics, key, value)
    
    # Re-initialize model on hardware if needed
    if self.hardware and self.model:
        self._initialize_model_on_hardware()
    
    logger.info(f"Resumed training from epoch {self.current_epoch}")
    
    # Continue training
    return self.train(dataset, additional_epochs)