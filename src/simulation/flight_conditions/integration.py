"""
Flight Conditions Integration

Integrates the edge case handler with the simulation framework.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any
from src.simulation.flight_conditions.edge_cases import (
    FlightConditionsManager, EdgeCaseType, EdgeCaseConfig,
    handle_turbulence, handle_icing
)
from src.simulation.core.scheduler import SimulationScheduler

def setup_flight_conditions(scheduler: SimulationScheduler) -> FlightConditionsManager:
    """
    Set up flight conditions manager with the simulation scheduler.
    
    Args:
        scheduler: Simulation scheduler
        
    Returns:
        Configured flight conditions manager
    """
    # Create manager
    manager = FlightConditionsManager()
    
    # Register default handlers
    manager.register_edge_case_handler(EdgeCaseType.EXTREME_TURBULENCE, handle_turbulence)
    manager.register_edge_case_handler(EdgeCaseType.ICING, handle_icing)
    
    def update_flight_conditions(sim_time, *args, **kwargs):
        # Access flight data from the manager's shared state
        flight_data = manager.get_shared_state()
        # Update flight conditions
        updated_data = manager.update(sim_time, flight_data)
        # Store updated data back to shared state
        manager.update_shared_state(updated_data)
        return updated_data
    
    # Add to scheduler
    scheduler.add_task(
        manager.get_task_config(),
        update_flight_conditions
    )
    
    return manager

def trigger_random_edge_case(manager: FlightConditionsManager) -> None:
    """Trigger a random edge case for testing."""
    import random
    
    # Select random edge case type
    edge_types = list(EdgeCaseType)
    selected_type = random.choice(edge_types)
    
    # Configure edge case
    config = EdgeCaseConfig(
        type=selected_type,
        severity=random.uniform(0.3, 0.9),
        duration=random.uniform(5.0, 15.0)
    )
    
    # Trigger edge case
    manager.trigger_edge_case(config)