#!/usr/bin/env python3
"""
Checkpoint Management for Long-Running Training

Provides functionality to manage checkpoints for neuromorphic training sessions.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import time
import shutil
import hashlib
import json
import pickle  # Add this import for pickle serialization
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from src.core.utils.logging_framework import get_logger
from src.core.utils.model_serialization import ModelSerializer
from src.core.training.trainer_base import NeuromorphicTrainer

logger = get_logger("checkpoint_manager")


class CheckpointManager:
    """Manages checkpoints for long-running training sessions."""
    
    def __init__(self, base_dir: str = "/Users/yessine/Oblivion/checkpoints", 
                max_checkpoints: int = 5, 
                checkpoint_interval: int = 10):
        """
        Initialize checkpoint manager.
        
        Args:
            base_dir: Base directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep per session
            checkpoint_interval: Checkpoint interval in epochs
        """
        self.base_dir = base_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.version_file = os.path.join(base_dir, "version_history.json")
        
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize version history
        self._init_version_history()
        
        logger.info(f"Initialized checkpoint manager with base directory: {base_dir}")
    
    def _init_version_history(self) -> None:
        """Initialize version history file if it doesn't exist."""
        if not os.path.exists(self.version_file):
            with open(self.version_file, 'w') as f:
                json.dump({
                    "models": {},
                    "latest_version": 0
                }, f, indent=2)
            logger.info(f"Created new version history file at {self.version_file}")
    
    def _get_model_hash(self, model: Any) -> str:
        """Generate a simple hash for model identification."""
        try:
            # Use pickle representation for hashing
            model_bytes = pickle.dumps(model)
            return hashlib.md5(model_bytes).hexdigest()[:10]
        except:
            # Fallback to timestamp if pickling fails
            return f"model_{int(time.time())}"
    
    def should_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be created at current epoch."""
        return epoch % self.checkpoint_interval == 0
    
    def create_checkpoint(self, trainer: NeuromorphicTrainer, session_id: str) -> Optional[str]:
        """
        Create a checkpoint for the current training state.
        
        Args:
            trainer: Neuromorphic trainer instance
            session_id: Training session ID
            
        Returns:
            Optional[str]: Path to checkpoint or None if failed
        """
        try:
            # Create session directory if it doesn't exist
            session_dir = os.path.join(self.base_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # Generate model version and hash
            model_hash = self._get_model_hash(trainer.model)
            model_version = self._register_model_version(session_id, model_hash, trainer)
            
            # Create checkpoint path with version
            timestamp = int(time.time())
            checkpoint_path = os.path.join(
                session_dir, 
                f"checkpoint_v{model_version}_epoch_{trainer.current_epoch}_{timestamp}.ckpt"
            )
            
            # Create checkpoint data
            checkpoint_data = {
                "model": trainer.model,
                "metrics": asdict(trainer.metrics),
                "training_id": trainer.training_id,
                "current_epoch": trainer.current_epoch,
                "hardware_type": trainer.config.hardware_type,
                "timestamp": timestamp,
                "version": model_version,
                "model_hash": model_hash
            }
            
            # Save checkpoint
            success = ModelSerializer.serialize(checkpoint_data, checkpoint_path)
            
            if success:
                logger.info(f"Created checkpoint for model v{model_version} at epoch {trainer.current_epoch}")
                
                # Manage checkpoint history
                self._manage_checkpoint_history(session_dir)
                
                # If this is the best model so far, save it separately
                if (not trainer.metrics.accuracy_history or 
                    trainer.metrics.accuracy_history[-1] > trainer.metrics.best_accuracy):
                    best_path = os.path.join(session_dir, f"best_model_v{model_version}.ckpt")
                    ModelSerializer.serialize({"model": trainer.model, "version": model_version}, best_path)
                    logger.info(f"Saved best model v{model_version} to {best_path}")
                
                return checkpoint_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating checkpoint: {str(e)}")
            return None
    
    def _register_model_version(self, session_id: str, model_hash: str, trainer: NeuromorphicTrainer) -> int:
        """Register model version in version history."""
        try:
            with open(self.version_file, 'r') as f:
                version_history = json.load(f)
            
            # Get latest version number
            latest_version = version_history.get("latest_version", 0)
            
            # Check if this model hash already exists
            for version_info in version_history.get("models", {}).values():
                if version_info.get("hash") == model_hash:
                    return version_info.get("version")
            
            # Create new version
            new_version = latest_version + 1
            
            # Add to version history
            if session_id not in version_history.get("models", {}):
                version_history["models"][session_id] = []
            
            # Add version info
            version_history["models"][session_id].append({
                "version": new_version,
                "hash": model_hash,
                "timestamp": int(time.time()),
                "epoch": trainer.current_epoch,
                "accuracy": trainer.metrics.accuracy_history[-1] if trainer.metrics.accuracy_history else 0,
                "hardware_type": trainer.config.hardware_type
            })
            
            # Update latest version
            version_history["latest_version"] = new_version
            
            # Save updated version history
            with open(self.version_file, 'w') as f:
                json.dump(version_history, f, indent=2)
            
            logger.info(f"Registered new model version v{new_version} for session {session_id}")
            return new_version
            
        except Exception as e:
            logger.error(f"Error registering model version: {str(e)}")
            return 0
    
    def get_version_history(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get version history for all models or a specific session."""
        try:
            with open(self.version_file, 'r') as f:
                version_history = json.load(f)
            
            if session_id:
                return {
                    "session_id": session_id,
                    "versions": version_history.get("models", {}).get(session_id, [])
                }
            
            return version_history
        except Exception as e:
            logger.error(f"Error getting version history: {str(e)}")
            return {}
    
    def load_model_version(self, version: int, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load a specific model version."""
        try:
            # Get version history
            version_history = self.get_version_history()
            
            # Find the session and checkpoint for this version
            target_session = session_id
            target_checkpoint = None
            
            if not target_session:
                # Search all sessions for this version
                for sess_id, versions in version_history.get("models", {}).items():
                    for version_info in versions:
                        if version_info.get("version") == version:
                            target_session = sess_id
                            break
                    if target_session:
                        break
            
            if not target_session:
                logger.error(f"Model version v{version} not found in any session")
                return None
            
            # Find the checkpoint file
            session_dir = os.path.join(self.base_dir, target_session)
            if not os.path.exists(session_dir):
                logger.error(f"Session directory not found: {session_dir}")
                return None
            
            # Look for checkpoint with this version
            for filename in os.listdir(session_dir):
                if filename.startswith(f"checkpoint_v{version}_") and filename.endswith(".ckpt"):
                    target_checkpoint = os.path.join(session_dir, filename)
                    break
            
            if not target_checkpoint:
                logger.error(f"Checkpoint for model v{version} not found in session {target_session}")
                return None
            
            # Load the checkpoint
            checkpoint_data = ModelSerializer.deserialize(target_checkpoint)
            if checkpoint_data:
                logger.info(f"Loaded model v{version} from {target_checkpoint}")
                return checkpoint_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading model version: {str(e)}")
            return None
    
    def load_latest_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint for a session.
        
        Args:
            session_id: Training session ID
            
        Returns:
            Optional[Dict[str, Any]]: Checkpoint data or None if not found
        """
        try:
            session_dir = os.path.join(self.base_dir, session_id)
            if not os.path.exists(session_dir):
                logger.warning(f"No checkpoints found for session {session_id}")
                return None
            
            # Find checkpoint files
            checkpoints = [f for f in os.listdir(session_dir) 
                          if f.startswith("checkpoint_") and f.endswith(".ckpt")]
            
            if not checkpoints:
                logger.warning(f"No checkpoints found in session directory {session_dir}")
                return None
            
            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True)
            latest_checkpoint = os.path.join(session_dir, checkpoints[0])
            
            # Load checkpoint
            checkpoint_data = ModelSerializer.deserialize(latest_checkpoint)
            if checkpoint_data:
                logger.info(f"Loaded checkpoint from {latest_checkpoint}")
                return checkpoint_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return None
    
    def _manage_checkpoint_history(self, session_dir: str) -> None:
        """Manage checkpoint history, keeping only the most recent ones."""
        try:
            # Find checkpoint files
            checkpoints = [f for f in os.listdir(session_dir) 
                          if f.startswith("checkpoint_") and f.endswith(".ckpt")]
            
            if len(checkpoints) <= self.max_checkpoints:
                return
            
            # Sort by timestamp (oldest first)
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            
            # Remove oldest checkpoints
            for i in range(len(checkpoints) - self.max_checkpoints):
                checkpoint_to_remove = os.path.join(session_dir, checkpoints[i])
                os.remove(checkpoint_to_remove)
                logger.info(f"Removed old checkpoint: {checkpoint_to_remove}")
                
        except Exception as e:
            logger.error(f"Error managing checkpoint history: {str(e)}")