#!/usr/bin/env python3
"""
Training Integration System

Provides integration between the training framework and other system components.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import time
from typing import Dict, List, Any, Optional, Callable

from src.core.utils.logging_framework import get_logger
from src.core.training.trainer_base import NeuromorphicTrainer
from src.core.training.checkpoint_manager import CheckpointManager
from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.core.utils.model_serialization import ModelSerializer

logger = get_logger("training_integration")


class TrainingIntegration:
    """Integrates trained models with other system components."""
    
    def __init__(self, 
                 trainer: Optional[NeuromorphicTrainer] = None,
                 system: Optional[NeuromorphicSystem] = None):
        """
        Initialize training integration.
        
        Args:
            trainer: Neuromorphic trainer instance
            system: Neuromorphic system instance
        """
        self.trainer = trainer
        self.system = system
        self.checkpoint_manager = CheckpointManager()
        self.model_registry = {}
        self.component_mappings = {}
        
        logger.info("Initialized training integration system")
    
    def register_model(self, model_id: str, model: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a trained model.
        
        Args:
            model_id: Unique model identifier
            model: Trained model
            metadata: Optional model metadata
            
        Returns:
            bool: Success status
        """
        if model_id in self.model_registry:
            logger.warning(f"Model '{model_id}' already registered")
            return False
        
        self.model_registry[model_id] = {
            "model": model,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        logger.info(f"Registered model '{model_id}'")
        return True
    
    def load_model_from_checkpoint(self, session_id: str, model_id: str) -> bool:
        """
        Load model from checkpoint and register it.
        
        Args:
            session_id: Training session ID
            model_id: ID to register the model under
            
        Returns:
            bool: Success status
        """
        checkpoint_data = self.checkpoint_manager.load_latest_checkpoint(session_id)
        if not checkpoint_data:
            logger.error(f"No checkpoint found for session {session_id}")
            return False
        
        model = checkpoint_data.get("model")
        if not model:
            logger.error("No model found in checkpoint")
            return False
        
        # Extract metadata from checkpoint
        metadata = {
            "session_id": session_id,
            "version": checkpoint_data.get("version", 0),
            "epoch": checkpoint_data.get("current_epoch", 0),
            "hardware_type": checkpoint_data.get("hardware_type", "unknown"),
            "timestamp": checkpoint_data.get("timestamp", time.time())
        }
        
        # Register the model
        return self.register_model(model_id, model, metadata)
    
    def map_model_to_component(self, model_id: str, component_name: str, 
                              transform: Optional[Callable] = None) -> bool:
        """
        Map a model to a system component.
        
        Args:
            model_id: Model identifier
            component_name: Component name in the neuromorphic system
            transform: Optional transformation function
            
        Returns:
            bool: Success status
        """
        if model_id not in self.model_registry:
            logger.error(f"Model '{model_id}' not found in registry")
            return False
        
        if not self.system:
            logger.error("No neuromorphic system available")
            return False
        
        if component_name not in self.system.components:
            logger.error(f"Component '{component_name}' not found in system")
            return False
        
        self.component_mappings[component_name] = {
            "model_id": model_id,
            "transform": transform
        }
        
        logger.info(f"Mapped model '{model_id}' to component '{component_name}'")
        return True
    
    def deploy_model_to_component(self, model_id: str, component_name: str) -> bool:
        """
        Deploy a model to a system component.
        
        Args:
            model_id: Model identifier
            component_name: Component name in the neuromorphic system
            
        Returns:
            bool: Success status
        """
        if model_id not in self.model_registry:
            logger.error(f"Model '{model_id}' not found in registry")
            return False
        
        if not self.system:
            logger.error("No neuromorphic system available")
            return False
        
        if component_name not in self.system.components:
            logger.error(f"Component '{component_name}' not found in system")
            return False
        
        try:
            # Get the model
            model_data = self.model_registry[model_id]
            model = model_data["model"]
            
            # Get the component
            component = self.system.components[component_name]
            
            # Apply transformation if needed
            if component_name in self.component_mappings and self.component_mappings[component_name].get("transform"):
                transform_func = self.component_mappings[component_name]["transform"]
                model = transform_func(model)
            
            # Deploy model to component
            if hasattr(component, "load_model"):
                component.load_model(model)
            elif hasattr(component, "set_model"):
                component.set_model(model)
            elif hasattr(component, "model"):
                component.model = model
            else:
                logger.warning(f"Component '{component_name}' has no model loading interface")
                return False
            
            logger.info(f"Deployed model '{model_id}' to component '{component_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            return False
    
    def export_model(self, model_id: str, export_path: str, format: str = "pickle") -> bool:
        """
        Export a model to file.
        
        Args:
            model_id: Model identifier
            export_path: Path to export the model
            format: Export format
            
        Returns:
            bool: Success status
        """
        if model_id not in self.model_registry:
            logger.error(f"Model '{model_id}' not found in registry")
            return False
        
        try:
            # Get the model
            model_data = self.model_registry[model_id]
            model = model_data["model"]
            metadata = model_data["metadata"]
            
            # Create export data
            export_data = {
                "model": model,
                "metadata": metadata,
                "export_timestamp": time.time()
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            # Export the model
            success = ModelSerializer.serialize(export_data, export_path, format)
            
            if success:
                logger.info(f"Exported model '{model_id}' to {export_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            return False
    
    def import_model(self, import_path: str, model_id: str, format: str = "pickle") -> bool:
        """
        Import a model from file.
        
        Args:
            import_path: Path to import the model from
            model_id: Model identifier to register under
            format: Import format
            
        Returns:
            bool: Success status
        """
        try:
            # Import the model
            import_data = ModelSerializer.deserialize(import_path, format)
            
            if not import_data:
                logger.error(f"Failed to import model from {import_path}")
                return False
            
            # Extract model and metadata
            model = import_data.get("model")
            metadata = import_data.get("metadata", {})
            
            # Add import information to metadata
            metadata["imported_from"] = import_path
            metadata["import_timestamp"] = time.time()
            
            # Register the model
            return self.register_model(model_id, model, metadata)
            
        except Exception as e:
            logger.error(f"Error importing model: {str(e)}")
            return False
    
    def connect_to_control_system(self, model_id: str, control_system: Any) -> bool:
        """
        Connect a trained model to a control system.
        
        Args:
            model_id: Model identifier
            control_system: Control system instance
            
        Returns:
            bool: Success status
        """
        if model_id not in self.model_registry:
            logger.error(f"Model '{model_id}' not found in registry")
            return False
        
        try:
            # Get the model
            model_data = self.model_registry[model_id]
            model = model_data["model"]
            
            # Connect to control system
            if hasattr(control_system, "set_neural_model"):
                control_system.set_neural_model(model)
            elif hasattr(control_system, "load_model"):
                control_system.load_model(model)
            else:
                logger.warning("Control system has no model loading interface")
                return False
            
            logger.info(f"Connected model '{model_id}' to control system")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to control system: {str(e)}")
            return False