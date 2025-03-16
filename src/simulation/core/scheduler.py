"""
Real-Time Simulation Scheduler

A lightweight scheduler for managing real-time simulation components
with configurable update rates and priorities.
"""

import time
import threading
from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import heapq

from src.core.utils.logging_framework import get_logger

logger = get_logger("scheduler")


class SimulationState(Enum):
    """Simulation execution states."""
    STOPPED = 0
    RUNNING = 1
    PAUSED = 2
    STEPPING = 3


@dataclass
class TaskConfig:
    """Configuration for a scheduled task."""
    
    # Task identification
    name: str
    
    # Execution parameters
    update_rate: float  # Hz (0 = as fast as possible)
    priority: int = 0  # Higher values = higher priority
    
    # Time constraints
    max_execution_time: float = 0.0  # seconds (0 = no limit)
    
    # Execution group (tasks in same group run sequentially)
    group: str = "default"


class Task:
    """A scheduled simulation task."""
    
    def __init__(self, config: TaskConfig, callback: Callable[..., Any]):
        """Initialize a task."""
        self.config = config
        self.callback = callback
        self.last_execution_time = 0.0
        self.next_execution_time = 0.0
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.average_execution_time = 0.0
        self.is_enabled = True
        
        # Calculate update interval
        self.update_interval = 0.0 if config.update_rate <= 0 else 1.0 / config.update_rate
        
        logger.debug(f"Task '{config.name}' initialized with update rate {config.update_rate} Hz")
    
    def __lt__(self, other):
        """Compare tasks for priority queue."""
        # First by execution time
        if self.next_execution_time != other.next_execution_time:
            return self.next_execution_time < other.next_execution_time
        
        # Then by priority (higher priority first)
        return self.config.priority > other.config.priority
    
    def execute(self, sim_time: float, *args, **kwargs) -> bool:
        """
        Execute the task.
        
        Args:
            sim_time: Current simulation time
            *args, **kwargs: Arguments to pass to the callback
            
        Returns:
            bool: True if execution was successful
        """
        if not self.is_enabled:
            return False
        
        # Record execution time
        start_time = time.time()
        
        try:
            # Execute callback
            self.callback(sim_time, *args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Error executing task '{self.config.name}': {e}")
            success = False
        
        # Update statistics
        execution_time = time.time() - start_time
        self.last_execution_time = sim_time
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.execution_count
        
        # Schedule next execution
        if self.update_interval > 0:
            self.next_execution_time = sim_time + self.update_interval
        
        # Check if execution took too long
        if self.config.max_execution_time > 0 and execution_time > self.config.max_execution_time:
            logger.warning(f"Task '{self.config.name}' exceeded max execution time: {execution_time:.4f}s")
        
        return success
    
    def reset_statistics(self):
        """Reset task execution statistics."""
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.average_execution_time = 0.0


class SimulationScheduler:
    """
    Real-time simulation scheduler for coordinating simulation components.
    """
    
    def __init__(self, time_scale: float = 1.0):
        """
        Initialize the scheduler.
        
        Args:
            time_scale: Simulation time scale factor (1.0 = real-time)
        """
        self.tasks = {}
        self.task_queue = []
        self.task_groups = {}
        self.time_scale = time_scale
        self.state = SimulationState.STOPPED
        
        # Timing variables
        self.start_time = 0.0
        self.pause_time = 0.0
        self.sim_time = 0.0
        self.real_time = 0.0
        self.frame_count = 0
        
        # Performance metrics
        self.average_frame_time = 0.0
        self.total_frame_time = 0.0
        self.min_frame_time = float('inf')
        self.max_frame_time = 0.0
        
        # Threading
        self.thread = None
        self.stop_event = threading.Event()
        
        logger.info(f"Simulation scheduler initialized with time scale {time_scale}")
    
    def add_task(self, config: TaskConfig, callback: Callable[..., Any]) -> Task:
        """
        Add a task to the scheduler.
        
        Args:
            config: Task configuration
            callback: Function to call when task is executed
            
        Returns:
            Task: The created task
        """
        # Create task
        task = Task(config, callback)
        
        # Add to tasks dictionary
        self.tasks[config.name] = task
        
        # Add to group
        if config.group not in self.task_groups:
            self.task_groups[config.group] = []
        self.task_groups[config.group].append(task)
        
        # Add to priority queue if running
        if self.state == SimulationState.RUNNING:
            task.next_execution_time = self.sim_time
            heapq.heappush(self.task_queue, task)
        
        logger.info(f"Added task '{config.name}' to scheduler")
        return task
    
    def remove_task(self, name: str) -> bool:
        """
        Remove a task from the scheduler.
        
        Args:
            name: Task name
            
        Returns:
            bool: True if task was removed
        """
        if name not in self.tasks:
            return False
        
        # Get task
        task = self.tasks[name]
        
        # Remove from tasks dictionary
        del self.tasks[name]
        
        # Remove from group
        if task.config.group in self.task_groups:
            self.task_groups[task.config.group].remove(task)
            if not self.task_groups[task.config.group]:
                del self.task_groups[task.config.group]
        
        # Note: Task will be removed from queue when it's popped
        
        logger.info(f"Removed task '{name}' from scheduler")
        return True
    
    def get_task(self, name: str) -> Optional[Task]:
        """Get a task by name."""
        return self.tasks.get(name)
    
    def start(self, threaded: bool = True):
        """
        Start the simulation.
        
        Args:
            threaded: Whether to run in a separate thread
        """
        if self.state == SimulationState.RUNNING:
            return
        
        # Reset timing variables
        self.start_time = time.time()
        self.real_time = 0.0
        self.sim_time = 0.0
        self.frame_count = 0
        
        # Reset performance metrics
        self.average_frame_time = 0.0
        self.total_frame_time = 0.0
        self.min_frame_time = float('inf')
        self.max_frame_time = 0.0
        
        # Initialize task queue
        self.task_queue = []
        for task in self.tasks.values():
            task.next_execution_time = self.sim_time
            heapq.heappush(self.task_queue, task)
        
        # Set state
        self.state = SimulationState.RUNNING
        self.stop_event.clear()
        
        if threaded:
            # Start thread
            self.thread = threading.Thread(target=self._run_loop)
            self.thread.daemon = True
            self.thread.start()
        else:
            # Run in current thread
            self._run_loop()
    
    def stop(self):
        """Stop the simulation."""
        if self.state == SimulationState.STOPPED:
            return
        
        # Set state
        self.state = SimulationState.STOPPED
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        logger.info(f"Simulation stopped after {self.frame_count} frames, {self.sim_time:.2f}s sim time")
    
    def pause(self):
        """Pause the simulation."""
        if self.state != SimulationState.RUNNING:
            return
        
        # Set state
        self.state = SimulationState.PAUSED
        self.pause_time = time.time()
        
        logger.info(f"Simulation paused at {self.sim_time:.2f}s sim time")
    
    def resume(self):
        """Resume the simulation."""
        if self.state != SimulationState.PAUSED:
            return
        
        # Adjust start time to account for pause duration
        pause_duration = time.time() - self.pause_time
        self.start_time += pause_duration
        
        # Set state
        self.state = SimulationState.RUNNING
        
        logger.info(f"Simulation resumed at {self.sim_time:.2f}s sim time")
    
    def step(self, step_time: float = 0.01):
        """
        Step the simulation forward by a fixed amount of time.
        
        Args:
            step_time: Time to step forward (seconds)
        """
        # Set state
        prev_state = self.state
        self.state = SimulationState.STEPPING
        
        # Step simulation
        target_time = self.sim_time + step_time
        while self.sim_time < target_time and self.state == SimulationState.STEPPING:
            self._process_tasks()
        
        # Restore previous state
        self.state = prev_state
        
        logger.debug(f"Stepped simulation by {step_time:.4f}s to {self.sim_time:.4f}s")
    
    def set_time_scale(self, time_scale: float):
        """
        Set the simulation time scale.
        
        Args:
            time_scale: Simulation time scale factor (1.0 = real-time)
        """
        self.time_scale = max(0.01, time_scale)
        logger.info(f"Simulation time scale set to {self.time_scale}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.
        
        Returns:
            Dict[str, Any]: Scheduler statistics
        """
        return {
            'sim_time': self.sim_time,
            'real_time': self.real_time,
            'time_scale': self.time_scale,
            'frame_count': self.frame_count,
            'average_frame_time': self.average_frame_time,
            'min_frame_time': self.min_frame_time,
            'max_frame_time': self.max_frame_time,
            'fps': self.frame_count / max(0.001, self.real_time),
            'task_count': len(self.tasks),
            'state': self.state.name
        }
    
    def get_task_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all tasks.
        
        Returns:
            Dict[str, Dict[str, Any]]: Task statistics by name
        """
        stats = {}
        for name, task in self.tasks.items():
            stats[name] = {
                'execution_count': task.execution_count,
                'average_execution_time': task.average_execution_time,
                'update_rate': task.config.update_rate,
                'actual_rate': task.execution_count / max(0.001, self.sim_time),
                'enabled': task.is_enabled,
                'group': task.config.group,
                'priority': task.config.priority
            }
        return stats
    
    def _run_loop(self):
        """Main simulation loop."""
        logger.info("Starting simulation loop")
        
        while self.state == SimulationState.RUNNING and not self.stop_event.is_set():
            self._process_tasks()
        
        logger.info("Simulation loop ended")
    
    def _process_tasks(self):
        """Process due tasks."""
        # Calculate current time
        current_real_time = time.time() - self.start_time
        self.real_time = current_real_time
        
        # Calculate simulation time
        self.sim_time = current_real_time * self.time_scale
        
        # Start frame timing
        frame_start_time = time.time()
        
        # Process all tasks that are due
        tasks_executed = 0
        
        while self.task_queue and self.task_queue[0].next_execution_time <= self.sim_time:
            # Get next task
            task = heapq.heappop(self.task_queue)
            
            # Skip disabled tasks
            if not task.is_enabled:
                # Re-schedule with a small delay
                task.next_execution_time = self.sim_time + 0.1
                heapq.heappush(self.task_queue, task)
                continue
            
            # Execute task
            task.execute(self.sim_time)
            tasks_executed += 1
            
            # Re-schedule task
            if task.update_interval <= 0:
                # Schedule for immediate execution
                task.next_execution_time = self.sim_time
            else:
                # Schedule for next interval
                # Ensure we don't fall behind by scheduling based on last execution time
                task.next_execution_time = max(
                    self.sim_time,
                    task.last_execution_time + task.update_interval
                )
            
            # Add back to queue
            heapq.heappush(self.task_queue, task)
        
        # Update frame statistics
        self.frame_count += 1
        frame_time = time.time() - frame_start_time
        self.total_frame_time += frame_time
        self.average_frame_time = self.total_frame_time / self.frame_count
        self.min_frame_time = min(self.min_frame_time, frame_time)
        self.max_frame_time = max(self.max_frame_time, frame_time)
        
        # Sleep to maintain real-time if no tasks were executed
        if tasks_executed == 0:
            time.sleep(0.001)  # Small sleep to prevent CPU hogging


def create_default_scheduler() -> SimulationScheduler:
    """
    Create a default scheduler with real-time execution.
    
    Returns:
        SimulationScheduler: Default scheduler
    """
    return SimulationScheduler(time_scale=1.0)