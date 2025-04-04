#!/usr/bin/env python3
"""
Manufacturing pipeline module for Oblivion project.
Provides classes and utilities for manufacturing processes.
"""

from enum import Enum
from typing import Dict, List, Any, Optional

class ManufacturingStage(Enum):
    """Manufacturing stages for production pipeline."""
    DESIGN = "design"
    PROTOTYPING = "prototyping"
    MATERIALS = "materials"
    FABRICATION = "fabrication"
    ASSEMBLY = "assembly"
    INTEGRATION = "integration"
    TESTING = "testing"
    QUALITY_CONTROL = "quality_control"
    PACKAGING = "packaging"
    DEPLOYMENT = "deployment"


class ManufacturingProcess:
    """Base class for manufacturing processes."""
    
    def __init__(self, name: str, stage: ManufacturingStage):
        """
        Initialize manufacturing process.
        
        Args:
            name: Process name
            stage: Manufacturing stage
        """
        self.name = name
        self.stage = stage
        self.status = "initialized"
        self.progress = 0.0
        self.metadata = {}
    
    def start(self) -> bool:
        """Start the manufacturing process."""
        self.status = "running"
        self.progress = 0.0
        return True
    
    def update(self, progress: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update process progress and metadata.
        
        Args:
            progress: Current progress (0-100)
            metadata: Optional metadata to update
            
        Returns:
            bool: Success status
        """
        self.progress = max(0.0, min(100.0, progress))
        
        if metadata:
            self.metadata.update(metadata)
        
        return True
    
    def complete(self) -> bool:
        """Mark process as complete."""
        self.status = "completed"
        self.progress = 100.0
        return True
    
    def fail(self, reason: str) -> bool:
        """
        Mark process as failed.
        
        Args:
            reason: Failure reason
            
        Returns:
            bool: Success status
        """
        self.status = "failed"
        self.metadata["failure_reason"] = reason
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current process status.
        
        Returns:
            Dict[str, Any]: Process status
        """
        return {
            "name": self.name,
            "stage": self.stage.value,
            "status": self.status,
            "progress": self.progress,
            "metadata": self.metadata
        }