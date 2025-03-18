"""
Training Session Management

Provides functionality for managing neuromorphic training sessions,
including saving, loading, and resuming training.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import uuid
from datetime import datetime

from src.core.utils.logging_framework import get_logger
from src.core.training.trainer_base import TrainingMetrics, TrainingConfig

logger = get_logger("training_session")


@dataclass
class TrainingSession:
    """Training session information and state."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unnamed_session"
    hardware_type: str = "simulated"
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    status: str = "created"  # created, running, paused, completed, failed
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    config: Dict[str, Any] = field(default_factory=dict)
    checkpoint_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        result = asdict(self)
        # Convert metrics to dict
        result["metrics"] = asdict(self.metrics)
        return result


class SessionManager:
    """Manages neuromorphic training sessions."""
    
    def __init__(self, sessions_dir: str = "/Users/yessine/Oblivion/training_sessions"):
        """
        Initialize session manager.
        
        Args:
            sessions_dir: Directory to store session data
        """
        self.sessions_dir = sessions_dir
        self.active_sessions: Dict[str, TrainingSession] = {}
        
        # Create sessions directory if it doesn't exist
        os.makedirs(sessions_dir, exist_ok=True)
        
        logger.info(f"Session manager initialized with directory: {sessions_dir}")
    
    def create_session(self, name: str, config: TrainingConfig, 
                      hardware_type: str = "simulated") -> TrainingSession:
        """
        Create a new training session.
        
        Args:
            name: Session name
            config: Training configuration
            hardware_type: Hardware type
            
        Returns:
            TrainingSession: New session
        """
        session = TrainingSession(
            name=name,
            hardware_type=hardware_type,
            config=asdict(config)
        )
        
        # Create session directory
        session_dir = os.path.join(self.sessions_dir, session.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save initial session state
        self._save_session_state(session)
        
        # Add to active sessions
        self.active_sessions[session.session_id] = session
        
        logger.info(f"Created training session: {session.name} ({session.session_id})")
        return session
    
    def update_session(self, session_id: str, metrics: TrainingMetrics, 
                      status: Optional[str] = None) -> bool:
        """
        Update session with latest metrics and status.
        
        Args:
            session_id: Session ID
            metrics: Updated metrics
            status: Optional new status
            
        Returns:
            bool: Success status
        """
        if session_id not in self.active_sessions:
            session = self.load_session(session_id)
            if not session:
                logger.error(f"Session not found: {session_id}")
                return False
        else:
            session = self.active_sessions[session_id]
        
        # Update session
        session.metrics = metrics
        session.last_update_time = time.time()
        if status:
            session.status = status
        
        # Save updated state
        self._save_session_state(session)
        
        logger.debug(f"Updated session {session_id} with status: {session.status}")
        return True
    
    def add_checkpoint(self, session_id: str, checkpoint_path: str) -> bool:
        """
        Add checkpoint to session.
        
        Args:
            session_id: Session ID
            checkpoint_path: Path to checkpoint file
            
        Returns:
            bool: Success status
        """
        if session_id not in self.active_sessions:
            session = self.load_session(session_id)
            if not session:
                logger.error(f"Session not found: {session_id}")
                return False
        else:
            session = self.active_sessions[session_id]
        
        # Add checkpoint
        session.checkpoint_paths.append(checkpoint_path)
        session.last_update_time = time.time()
        
        # Save updated state
        self._save_session_state(session)
        
        logger.info(f"Added checkpoint to session {session_id}: {checkpoint_path}")
        return True
    
    def get_latest_checkpoint(self, session_id: str) -> Optional[str]:
        """
        Get latest checkpoint path for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Optional[str]: Latest checkpoint path or None
        """
        if session_id not in self.active_sessions:
            session = self.load_session(session_id)
            if not session:
                logger.error(f"Session not found: {session_id}")
                return None
        else:
            session = self.active_sessions[session_id]
        
        if not session.checkpoint_paths:
            return None
        
        return session.checkpoint_paths[-1]
    
    def load_session(self, session_id: str) -> Optional[TrainingSession]:
        """
        Load session from disk.
        
        Args:
            session_id: Session ID
            
        Returns:
            Optional[TrainingSession]: Loaded session or None
        """
        session_file = os.path.join(self.sessions_dir, session_id, "session.json")
        if not os.path.exists(session_file):
            logger.error(f"Session file not found: {session_file}")
            return None
        
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            # Create metrics object
            metrics_data = data.pop("metrics", {})
            metrics = TrainingMetrics(
                loss_history=metrics_data.get("loss_history", []),
                accuracy_history=metrics_data.get("accuracy_history", []),
                training_time=metrics_data.get("training_time", 0.0),
                epochs_completed=metrics_data.get("epochs_completed", 0),
                best_epoch=metrics_data.get("best_epoch", 0),
                best_loss=metrics_data.get("best_loss", float('inf')),
                best_accuracy=metrics_data.get("best_accuracy", 0.0)
            )
            
            # Create session object
            session = TrainingSession(
                session_id=data.get("session_id", ""),
                name=data.get("name", ""),
                hardware_type=data.get("hardware_type", ""),
                start_time=data.get("start_time", 0.0),
                last_update_time=data.get("last_update_time", 0.0),
                status=data.get("status", ""),
                metrics=metrics,
                config=data.get("config", {}),
                checkpoint_paths=data.get("checkpoint_paths", []),
                metadata=data.get("metadata", {})
            )
            
            # Add to active sessions
            self.active_sessions[session_id] = session
            
            logger.info(f"Loaded session: {session.name} ({session.session_id})")
            return session
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {str(e)}")
            return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions.
        
        Returns:
            List[Dict[str, Any]]: List of session summaries
        """
        sessions = []
        
        # Check all subdirectories in sessions directory
        if os.path.exists(self.sessions_dir):
            for session_id in os.listdir(self.sessions_dir):
                session_dir = os.path.join(self.sessions_dir, session_id)
                session_file = os.path.join(session_dir, "session.json")
                
                if os.path.isdir(session_dir) and os.path.exists(session_file):
                    try:
                        with open(session_file, 'r') as f:
                            data = json.load(f)
                        
                        # Create summary
                        summary = {
                            "session_id": session_id,
                            "name": data.get("name", ""),
                            "hardware_type": data.get("hardware_type", ""),
                            "status": data.get("status", ""),
                            "start_time": data.get("start_time", 0),
                            "last_update_time": data.get("last_update_time", 0)
                        }
                        
                        sessions.append(summary)
                    except Exception as e:
                        logger.warning(f"Error reading session {session_id}: {str(e)}")
        
        return sessions
    
    def _save_session_state(self, session: TrainingSession) -> bool:
        """
        Save session state to disk.
        
        Args:
            session: Session to save
            
        Returns:
            bool: Success status
        """
        session_dir = os.path.join(self.sessions_dir, session.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        session_file = os.path.join(session_dir, "session.json")
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {str(e)}")
            return False