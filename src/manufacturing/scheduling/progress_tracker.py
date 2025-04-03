import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
import threading

from src.core.utils.logging_framework import get_logger

logger = get_logger("progress_tracker")

class TaskStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    FAILED = "failed"

class ProgressTracker:
    def __init__(self):
        self.task_progress: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def start_task(self, task_id: str, expected_duration: float) -> None:
        with self._lock:
            self.task_progress[task_id] = {
                "status": TaskStatus.IN_PROGRESS,
                "start_time": datetime.now(),
                "expected_duration": expected_duration,
                "completion_percentage": 0,
                "actual_duration": 0.0
            }
    
    def update_progress(self, task_id: str, completion_percentage: float) -> None:
        with self._lock:
            if task_id not in self.task_progress:
                logger.warning(f"Task {task_id} not found in progress tracker")
                return
            
            progress = self.task_progress[task_id]
            progress["completion_percentage"] = min(completion_percentage, 100)
            
            if completion_percentage >= 100:
                self.complete_task(task_id)
    
    def complete_task(self, task_id: str) -> None:
        with self._lock:
            if task_id not in self.task_progress:
                return
                
            progress = self.task_progress[task_id]
            progress["status"] = TaskStatus.COMPLETED
            progress["completion_percentage"] = 100
            progress["actual_duration"] = (
                datetime.now() - progress["start_time"]
            ).total_seconds() / 3600  # Convert to hours
    
    def mark_delayed(self, task_id: str) -> None:
        with self._lock:
            if task_id in self.task_progress:
                self.task_progress[task_id]["status"] = TaskStatus.DELAYED
    
    def mark_failed(self, task_id: str, reason: str) -> None:
        with self._lock:
            if task_id in self.task_progress:
                progress = self.task_progress[task_id]
                progress["status"] = TaskStatus.FAILED
                progress["failure_reason"] = reason
    
    def get_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.task_progress.get(task_id)
    
    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        return self.task_progress.copy()