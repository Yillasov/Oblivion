#!/usr/bin/env python3
"""
Base controller for 3D printers and manufacturing equipment.
"""

import os
import sys
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger

logger = get_logger("printer_controller")


@dataclass
class EquipmentStatus:
    """Status information for manufacturing equipment."""
    operational: bool = False
    temperature: float = 0.0
    power_state: str = "off"  # off, standby, printing, error
    error_code: int = 0
    error_message: str = ""
    last_maintenance: Optional[datetime] = None


class PrinterController:
    """Base controller for 3D printers."""
    
    def __init__(self, equipment_id: str):
        """
        Initialize printer controller.
        
        Args:
            equipment_id: Unique identifier for the printer
        """
        self.equipment_id = equipment_id
        self.status = EquipmentStatus()
        self.print_history = []
        logger.info(f"Initialized printer controller: {equipment_id}")
    
    def initialize(self) -> bool:
        """
        Initialize the printer.
        
        Returns:
            bool: Success status
        """
        try:
            self.status = EquipmentStatus(
                operational=True,
                temperature=25.0,
                power_state="standby",
                last_maintenance=datetime.now()
            )
            logger.info(f"Printer {self.equipment_id} initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize printer: {e}")
            self.status.error_code = 1
            self.status.error_message = str(e)
            return False
    
    def shutdown(self) -> bool:
        """
        Shutdown the printer.
        
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Shutting down printer: {self.equipment_id}")
            self.status.operational = False
            self.status.power_state = "off"
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown printer: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the printer.
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "equipment_id": self.equipment_id,
            "operational": self.status.operational,
            "temperature": self.status.temperature,
            "power_state": self.status.power_state,
            "error_code": self.status.error_code,
            "error_message": self.status.error_message,
            "last_maintenance": self.status.last_maintenance.isoformat() 
                if self.status.last_maintenance else None
        }
    
    def record_print_job(self, job_data: Dict[str, Any]) -> None:
        """
        Record a print job in the history.
        
        Args:
            job_data: Print job data
        """
        job_data["timestamp"] = datetime.now().isoformat()
        job_data["equipment_id"] = self.equipment_id
        self.print_history.append(job_data)
        logger.info(f"Recorded print job: {job_data.get('job_id', 'unknown')}")
    
    def get_print_history(self) -> list:
        """
        Get the print job history.
        
        Returns:
            list: List of print jobs
        """
        return self.print_history