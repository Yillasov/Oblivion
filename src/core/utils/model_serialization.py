"""
Model Serialization Utilities

Provides functionality for serializing and deserializing neuromorphic models.
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, Any, Optional, Union, List

from src.core.utils.logging_framework import get_logger

logger = get_logger("model_serialization")


class ModelSerializer:
    """Handles serialization and deserialization of neuromorphic models."""
    
    @staticmethod
    def serialize(model: Any, path: str, format: str = "pickle") -> bool:
        """
        Serialize model to file.
        
        Args:
            model: Model to serialize
            path: Path to save serialized model
            format: Serialization format ('pickle', 'json', or 'numpy')
            
        Returns:
            bool: Success status
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            if format == "pickle":
                with open(path, 'wb') as f:
                    pickle.dump(model, f)
            elif format == "json":
                # Convert numpy arrays to lists for JSON serialization
                if hasattr(model, "to_json"):
                    # Use model's custom JSON serialization if available
                    json_data = model.to_json()
                    with open(path, 'w') as f:
                        json.dump(json_data, f, indent=2)
                else:
                    # Basic conversion for simple models
                    json_data = ModelSerializer._convert_numpy_to_lists(model)
                    with open(path, 'w') as f:
                        json.dump(json_data, f, indent=2)
            elif format == "numpy":
                # For models that are primarily numpy arrays
                if isinstance(model, dict) and all(isinstance(v, np.ndarray) for v in model.values()):
                    np.savez(path, **model)
                else:
                    logger.error("Model is not compatible with numpy format")
                    return False
            else:
                logger.error(f"Unsupported serialization format: {format}")
                return False
                
            logger.info(f"Serialized model to {path} using {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Error serializing model: {str(e)}")
            return False
    
    @staticmethod
    def deserialize(path: str, format: str = "pickle") -> Optional[Any]:
        """
        Deserialize model from file.
        
        Args:
            path: Path to serialized model
            format: Serialization format ('pickle', 'json', or 'numpy')
            
        Returns:
            Any: Deserialized model or None if failed
        """
        try:
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return None
                
            if format == "pickle":
                with open(path, 'rb') as f:
                    model = pickle.load(f)
            elif format == "json":
                with open(path, 'r') as f:
                    json_data = json.load(f)
                    
                # Convert lists back to numpy arrays if needed
                model = ModelSerializer._convert_lists_to_numpy(json_data)
            elif format == "numpy":
                model = dict(np.load(path))
            else:
                logger.error(f"Unsupported deserialization format: {format}")
                return None
                
            logger.info(f"Deserialized model from {path} using {format} format")
            return model
            
        except Exception as e:
            logger.error(f"Error deserializing model: {str(e)}")
            return None
    
    @staticmethod
    def _convert_numpy_to_lists(obj: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: ModelSerializer._convert_numpy_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ModelSerializer._convert_numpy_to_lists(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def _convert_lists_to_numpy(obj: Any) -> Any:
        """Convert lists back to numpy arrays."""
        if isinstance(obj, list):
            try:
                return np.array(obj)
            except:
                return [ModelSerializer._convert_lists_to_numpy(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: ModelSerializer._convert_lists_to_numpy(v) for k, v in obj.items()}
        else:
            return obj