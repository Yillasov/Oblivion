#!/usr/bin/env python3
"""
Real-time Optimization Monitor

Command-line tool to monitor and control real-time adaptive optimization.
"""

import argparse
import json
import sys
import os
import time
from typing import Dict, Any, List, Optional
import curses
import signal
import threading

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.optimization.adaptive_realtime_optimizer import (
    AdaptiveRealtimeOptimizer, 
    AdaptiveOptimizationConfig,
    OptimizationTarget
)
from src.core.hardware.neuromorphic_optimizer import NeuromorphicHardwareOptimizer
from src.core.utils.logging_framework import get_logger

logger = get_logger("optimization_monitor")

# Global variables for curses interface
stdscr = None
optimizers: Dict[str, NeuromorphicHardwareOptimizer] = {}
hardware_ids: Dict[str, List[str]] = {}
selected_hardware_type = 0
selected_hardware_id = 0
running = True
update_interval = 1.0
last_update_time = 0.0
optimization_history: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
auto_update = True

def setup_optimizers(args):
    """Set up optimizers for different hardware types."""
    global optimizers, hardware_ids
    
    # Create optimizers for each hardware type
    hardware_types = ["loihi", "spinnaker", "truenorth"]
    
    for hw_type in hardware_types:
        # Create optimizer
        optimizer = NeuromorphicHardwareOptimizer(hw_type)
        optimizers[hw_type] = optimizer
        hardware_ids[hw_type] = []
        
        # Register mock hardware instances for testing
        if args.mock:
            for i in range(1, 4):
                hardware_id = f"{hw_type}_{i}"
                mock_hardware = MockHardware(hw_type, hardware_id)
                mock_monitor = MockHardwareMonitor(hw_type, hardware_id)
                
                optimizer.register_hardware(hardware_id, mock_hardware, mock_monitor)
                hardware_ids[hw_type].append(hardware_id)
                
                # Initialize optimization history
                if hw_type not in optimization_history:
                    optimization_history[hw_type] = {}
                optimization_history[hw_type][hardware_id] = []

def update_optimizers():
    """Update all optimizers."""
    global optimizers, hardware_ids, optimization_history
    
    for hw_type, optimizer in optimizers.items():
        for hardware_id in hardware_ids[hw_type]:
            # Update optimization
            result = optimizer.update(hardware_id)
            
            # Store result in history
            if hw_type in optimization_history and hardware_id in optimization_history[hw_type]:
                optimization_history[hw_type][hardware_id].append({
                    "timestamp": time.time(),
                    "result": result
                })
                
                # Limit history size
                if len(optimization_history[hw_type][hardware_id]) > 100:
                    optimization_history[hw_type][hardware_id].pop(0)

def draw_interface(stdscr):
    """Draw the curses interface."""
    global optimizers, hardware_ids, selected_hardware_type, selected_hardware_id, running, auto_update
    
    # Clear screen
    stdscr.clear()
    
    # Get screen dimensions
    height, width = stdscr.getmaxyx()
    
    # Draw title
    title = "Real-time Neuromorphic Hardware Optimization Monitor"
    stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)
    
    # Draw hardware type selection
    stdscr.addstr(2, 2, "Hardware Types:", curses.A_BOLD)
    
    for i, hw_type in enumerate(optimizers.keys()):
        if i == selected_hardware_type:
            stdscr.addstr(3 + i, 4, f"> {hw_type}", curses.A_REVERSE)
        else:
            stdscr.addstr(3 + i, 4, f"  {hw_type}")
    
    # Draw hardware instances
    hw_type = list(optimizers.keys())[selected_hardware_type]
    stdscr.addstr(2, 25, f"{hw_type} Instances:", curses.A_BOLD)
    
    for i, hardware_id in enumerate(hardware_ids[hw_type]):
        if i == selected_hardware_id:
            stdscr.addstr(3 + i, 27, f"> {hardware_id}", curses.A_REVERSE)
        else:
            stdscr.addstr(3 + i, 27, f"  {hardware_id}")
    
    # Draw optimization stats
    if hardware_ids[hw_type]:
        hardware_id = hardware_ids[hw_type][selected_hardware_id]
        stats = optimizers[hw_type].get_optimization_stats(hardware_id)
        
        stdscr.addstr(2, 50, "Optimization Statistics:", curses.A_BOLD)
        
        if stats["success"]:
            # Draw metrics
            metrics_stats = stats.get("stats", {}).get("metrics", {})
            row = 3
            
            for metric, metric_stats in metrics_stats.items():
                stdscr.addstr(row, 52, f"{metric}:")
                stdscr.addstr(row, 70, f"Current: {metric_stats.get('current', 0.0):.3f}")
                stdscr.addstr(row, 90, f"Improvement: {metric_stats.get('improvement', 0.0):.3f}")
                row += 1
            
            # Draw overall score
            overall_score = stats.get("stats", {}).get("overall_score", {})
            stdscr.addstr(row + 1, 52, "Overall Score:", curses.A_BOLD)
            stdscr.addstr(row + 1, 70, f"Current: {overall_score.get('current', 0.0):.3f}")
            stdscr.addstr(row + 1, 90, f"Improvement: {overall_score.get('improvement', 0.0):.3f}")
            
            # Draw parameters
            parameters = stats.get("parameters", {})
            stdscr.addstr(row + 3, 52, "Current Parameters:", curses.A_BOLD)
            
            param_row = row + 4
            for param, value in parameters.items():
                stdscr.addstr(param_row, 54, f"{param}: {value:.4f}")
                param_row += 1
        else:
            stdscr.addstr(3, 52, f"Error: {stats.get('error', 'Unknown error')}")
    
    # Draw history graph if we have data
    hw_type = list(optimizers.keys())[selected_hardware_type]
    if hardware_ids[hw_type]:
        hardware_id = hardware_ids[hw_type][selected_hardware_id]
        
        if hw_type in optimization_history and hardware_id in optimization_history[hw_type]:
            history = optimization_history[hw_type][hardware_id]
            
            if history:
                # Draw graph title
                graph_title = "Performance History"
                stdscr.addstr(height - 15, (width - len(graph_title)) // 2, graph_title, curses.A_BOLD)
                
                # Draw graph axes
                for i in range(min(50, width - 10)):
                    stdscr.addstr(height - 3, 5 + i, "─")
                
                for i in range(10):
                    stdscr.addstr(height - 4 - i, 4, "│")
                
                # Draw data points
                scores = []
                for entry in history:
                    result = entry.get("result", {})
                    if "score" in result:
                        scores.append(result["score"])
                
                if scores:
                    # Scale scores to fit in graph
                    max_score = max(scores) if max(scores) > 0 else 1.0
                    scaled_scores = [min(10, int(score / max_score * 10)) for score in scores]
                    
                    # Draw points
                    for i, score in enumerate(scaled_scores[-min(50, width - 10):]):
                        if score > 0:
                            stdscr.addstr(height - 4 - score, 5 + i, "*")
    
    # Draw status and controls
    status_line = f"Auto-update: {'ON' if auto_update else 'OFF'} | Press 'q' to quit, 'a' to toggle auto-update, 'r' to reset, 'u' to update"
    stdscr.addstr(height - 1, 0, status_line)
    
    # Refresh screen
    stdscr.refresh()

def handle_input(stdscr):
    """Handle user input."""
    global selected_hardware_type, selected_hardware_id, running, auto_update
    
    # Get key
    key = stdscr.getch()
    
    # Handle key
    if key == ord('q'):
        running = False
    elif key == ord('a'):
        auto_update = not auto_update
    elif key == ord('u'):
        # Manual update
        hw_type = list(optimizers.keys())[selected_hardware_type]
        if hardware_ids[hw_type]:
            hardware_id = hardware_ids[hw_type][selected_hardware_id]
            update_optimizers()
    elif key == ord('r'):
        # Reset optimization
        hw_type = list(optimizers.keys())[selected_hardware_type]
        if hardware_ids[hw_type]:
            hardware_id = hardware_ids[hw_type][selected_hardware_id]
            optimizers[hw_type].reset(hardware_id)
            if hw_type in optimization_history and hardware_id in optimization_history[hw_type]:
                optimization_history[hw_type][hardware_id] = []
    elif key == curses.KEY_UP:
        # Move selection up
        if selected_hardware_id > 0:
            selected_hardware_id -= 1
    elif key == curses.KEY_DOWN:
        # Move selection down
        hw_type = list(optimizers.keys())[selected_hardware_type]
        if selected_hardware_id < len(hardware_ids[hw_type]) - 1:
            selected_hardware_id += 1
    elif key == curses.KEY_LEFT:
        # Move to previous hardware type
        if selected_hardware_type > 0:
            selected_hardware_type -= 1
            selected_hardware_id = 0
    elif key == curses.KEY_RIGHT:
        # Move to next hardware type
        if selected_hardware_type < len(optimizers) - 1:
            selected_hardware_type += 1
            selected_hardware_id = 0

def main_loop(stdscr):
    """Main application loop."""
    global running, last_update_time, update_interval, auto_update
    
    # Set up curses
    curses.curs_set(0)  # Hide cursor
    stdscr.timeout(100)  # Non-blocking input
    
    # Main loop
    while running:
        # Handle input
        handle_input(stdscr)
        
        # Update optimizers if auto-update is enabled
        current_time = time.time()
        if auto_update and current_time - last_update_time >= update_interval:
            update_optimizers()
            last_update_time = current_time
        
        # Draw interface
        draw_interface(stdscr)
        
        # Sleep to reduce CPU usage
        time.sleep(0.05)

class MockHardware:
    """Mock hardware for testing."""
    
    def __init__(self, hardware_type, hardware_id):
        self.hardware_type = hardware_type
        self.hardware_id = hardware_id
        self.settings = {}
    
    def apply_settings(self, settings):
        """Apply settings to hardware."""
        self.settings.update(settings)

class MockHardwareMonitor:
    """Mock hardware monitor for testing."""
    
    def __init__(self, hardware_type, hardware_id):
        self.hardware_type = hardware_type
        self.hardware_id = hardware_id
        self.base_metrics = {
            "performance": 50.0,
            "power_consumption": 5.0,
            "latency": 50.0,
            "throughput": 500.0,
            "error_rate": 0.05,
            "temperature": 60.0
        }
        self.improvement_factor = 0.0
    
    def get_metrics(self):
        """Get mock metrics with some randomness and gradual improvement."""
        import random
        
        # Increase improvement factor (simulates learning over time)
        self.improvement_factor = min(0.5, self.improvement_factor + 0.01)
        
        # Generate metrics with randomness and improvement
        metrics = {}
        for metric, base_value in self.base_metrics.items():
            # Add randomness
            randomness = random.uniform(-0.1, 0.1) * base_value
            
            # Apply improvement factor (better for all metrics except power_consumption and temperature)
            if metric in ["power_consumption", "latency", "error_rate", "temperature"]:
                # Lower is better for these metrics
                improvement = -self.improvement_factor * base_value * 0.5
            else:
                # Higher is better for these metrics
                improvement = self.improvement_factor * base_value * 0.5
            
            metrics[metric] = max(0, base_value + randomness + improvement)
        
        return metrics

def main():
    """Main entry point."""
    global stdscr
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Real-time Optimization Monitor")
    parser.add_argument("--mock", action="store_true", help="Use mock hardware for testing")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")
    args = parser.parse_args()
    
    # Set update interval
    global update_interval
    update_interval = args.interval
    
    # Set up optimizers
    setup_optimizers(args)
    
    # Run curses application
    try:
        stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        main_loop(stdscr)
    except Exception as e:
        # Clean up curses
        if stdscr:
            curses.endwin()
        print(f"Error: {str(e)}")
    finally:
        # Clean up curses
        if stdscr:
            curses.endwin()

if __name__ == "__main__":
    main()