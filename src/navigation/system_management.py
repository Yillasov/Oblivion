"""
Navigation system management utilities.

This module provides standardized methods for system initialization and status reporting.
"""

import logging
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)


class NavigationSystemManager:
    """Manages navigation system initialization and status reporting."""
    
    def __init__(self, integrator=None):
        """Initialize the system manager."""
        self.integrator = integrator
        self.initialization_time = None
        self.last_status_update = None
        self.status_history = []
        self.max_history_entries = 100
    
    def initialize_system(self, system_id: str) -> bool:
        """
        Initialize a specific navigation system.
        
        Args:
            system_id: ID of the system to initialize
            
        Returns:
            Success status
        """
        if not self.integrator:
            logger.error("No integrator available for initialization")
            return False
            
        if system_id not in self.integrator.navigation_systems:
            logger.error(f"System {system_id} not found")
            return False
            
        try:
            system = self.integrator.navigation_systems[system_id]
            success = system.initialize()
            
            if success:
                self.initialization_time = datetime.now()
                logger.info(f"Successfully initialized navigation system: {system_id}")
            else:
                logger.warning(f"Failed to initialize navigation system: {system_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error during initialization of {system_id}: {str(e)}")
            return False
    
    def get_system_status(self, system_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of navigation system(s).
        
        Args:
            system_id: Optional ID of specific system to check
            
        Returns:
            Status information dictionary
        """
        if not self.integrator:
            return {"error": "No integrator available"}
            
        self.last_status_update = datetime.now()
        
        # Return status for a specific system
        if system_id:
            if system_id not in self.integrator.navigation_systems:
                return {"error": f"System {system_id} not found"}
                
            try:
                system = self.integrator.navigation_systems[system_id]
                status = system.get_status()
                
                # Add metadata
                status.update({
                    "system_id": system_id,
                    "timestamp": self.last_status_update.isoformat()
                })
                
                # Add to history
                self._update_status_history(system_id, status)
                
                return status
                
            except Exception as e:
                logger.error(f"Error getting status for {system_id}: {str(e)}")
                return {"error": str(e), "system_id": system_id}
        
        # Return status for all systems
        all_statuses = {}
        for sys_id, system in self.integrator.navigation_systems.items():
            try:
                status = system.get_status()
                status["system_id"] = sys_id
                all_statuses[sys_id] = status
                
                # Add to history
                self._update_status_history(sys_id, status)
                
            except Exception as e:
                logger.error(f"Error getting status for {sys_id}: {str(e)}")
                all_statuses[sys_id] = {"error": str(e), "system_id": sys_id}
        
        # Add overall status
        all_statuses["_overall"] = {
            "total_systems": len(self.integrator.navigation_systems),
            "active_systems": sum(1 for s in all_statuses.values() if s.get("active", False)),
            "timestamp": self.last_status_update.isoformat(),
            "uptime": (datetime.now() - self.initialization_time).total_seconds() if self.initialization_time else 0
        }
        
        return all_statuses
    
    def _update_status_history(self, system_id: str, status: Dict[str, Any]) -> None:
        """Update status history for a system."""
        # Add timestamp if not present
        if "timestamp" not in status:
            status["timestamp"] = datetime.now().isoformat()
            
        # Add entry to history
        self.status_history.append({
            "system_id": system_id,
            "timestamp": status["timestamp"],
            "status": status
        })
        
        # Trim history if needed
        if len(self.status_history) > self.max_history_entries:
            self.status_history = self.status_history[-self.max_history_entries:]
    
    def get_status_history(self, system_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get status history for a system or all systems.
        
        Args:
            system_id: Optional ID to filter history by system
            
        Returns:
            List of status history entries
        """
        if system_id:
            return [entry for entry in self.status_history if entry["system_id"] == system_id]
        return self.status_history