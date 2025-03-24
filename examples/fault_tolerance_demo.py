#!/usr/bin/env python3
"""
Fault Tolerance Demonstration

Shows how to use the fault tolerance mechanisms for neuromorphic hardware.
"""

import sys
import os
import time
import random
import threading
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.core.hardware.hardware_monitor import HardwareMetricType
from src.core.hardware.fault_tolerance import FaultType, FaultSeverity

# Mock hardware classes
class MockNeuromorphicHardware:
    """Mock neuromorphic hardware for demonstration."""
    
    def __init__(self, hardware_id: str, hardware_type: str):
        """Initialize mock hardware."""
        self.hardware_id = hardware_id
        self.hardware_type = hardware_type
        self.settings = {}
        self.is_active = True
        self.error_rate = 0.0
    
    def apply_settings(self, settings: Dict[str, Any]):
        """Apply settings to hardware."""
        self.settings.update(settings)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get hardware metrics."""
        # Generate random metrics with some noise
        base_temp = 50.0 if self.hardware_type == "loihi" else 60.0
        base_power = 0.5 if self.hardware_type == "truenorth" else 0.7
        
        metrics = {
            "temperature": base_temp + 10.0 * random.random(),
            "power": base_power + 0.2 * random.random(),
            "utilization": 0.5 + 0.3 * random.random(),
            "error_rate": self.error_rate + 0.01 * random.random(),
            "response_time": 0.3 + 0.2 * random.random(),
            "memory_usage": 0.6 + 0.2 * random.random(),
            "spike_rate": 0.4 + 0.3 * random.random()
        }
        
        return metrics
    
    def simulate_fault(self, fault_type: str):
        """Simulate a hardware fault."""
        if fault_type == "thermal":
            # Simulate thermal issue
            return {"temperature": 95.0}
        elif fault_type == "power":
            # Simulate power issue
            return {"power": 0.98}
        elif fault_type == "computation":
            # Simulate computation error
            self.error_rate = 0.2
            return {"error_rate": 0.2}
        elif fault_type == "memory":
            # Simulate memory issue
            return {"memory_usage": 0.97}
        elif fault_type == "communication":
            # Simulate communication issue (no metrics returned)
            return None
        
        return {}

def main():
    """Main demonstration function."""
    # Create neuromorphic system
    system = NeuromorphicSystem()
    
    # Enable fault tolerance
    system.enable_fault_tolerance(True)
    
    # Create mock hardware instances
    hardware_types = ["loihi", "spinnaker", "truenorth"]
    mock_hardware = {}
    
    for hw_type in hardware_types:
        for i in range(1, 3):
            hardware_id = f"{hw_type}_{i}"
            mock_hw = MockNeuromorphicHardware(hardware_id, hw_type)
            
            # Register with system
            is_critical = (i == 1)  # First instance of each type is critical
            redundancy_group = hw_type if i > 1 else None
            
            system.register_hardware_component(
                hardware_id,
                mock_hw,
                hw_type,
                is_critical,
                redundancy_group
            )
            
            mock_hardware[hardware_id] = mock_hw
    
    # Start system
    system.start()
    
    # Monitoring thread function
    def update_metrics():
        while system.running:
            for hardware_id, hardware in mock_hardware.items():
                if hardware.is_active:
                    # Get metrics
                    metrics = hardware.get_metrics()
                    if metrics:
                        # Report metrics
                        system.report_hardware_metrics(hardware_id, metrics)
            
            # Sleep before next update
            time.sleep(0.5)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=update_metrics)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # Run demonstration
        print("Fault Tolerance Demonstration")
        print("----------------------------")
        print("Starting normal operation...")
        time.sleep(3)
        
        # Show initial system health
        health = system.get_hardware_status()
        print("\nInitial System Health:")
        print(f"Overall Health: {health['system_health']['overall_health']:.2f}")
        print(f"Critical Components Health: {health['system_health']['critical_health']:.2f}")
        print(f"Active Components: {health['system_health']['active_components']} / {health['system_health']['total_components']}")
        
        # Simulate faults
        print("\nSimulating faults...")
        
        # Thermal fault in loihi_1
        print("\n1. Simulating thermal fault in loihi_1 (critical component)...")
        mock_hardware["loihi_1"].simulate_fault("thermal")
        time.sleep(2)
        
        # Show system health after thermal fault
        health = system.get_hardware_status()
        print("\nSystem Health after Thermal Fault:")
        print(f"Overall Health: {health['system_health']['overall_health']:.2f}")
        print(f"Critical Components Health: {health['system_health']['critical_health']:.2f}")
        print(f"Active Faults: {health['system_health']['active_faults']}")
        
        # Show loihi_1 status
        loihi_status = system.get_hardware_status("loihi_1")
        print("\nloihi_1 Status:")
        print(f"Health Score: {loihi_status['health_score']:.2f}")
        print(f"Temperature: {loihi_status['current_metrics'].get('temperature', 'N/A')}")
        print(f"Active Faults: {len(loihi_status['active_faults'])}")
        
        # Resolve thermal fault
        if loihi_status['active_faults']:
            fault_id = loihi_status['active_faults'][0]['fault_id']
            print(f"\nResolving thermal fault ({fault_id})...")
            system.resolve_hardware_fault(fault_id, "Applied cooling measures")
            
            # Reset temperature
            mock_hardware["loihi_1"].simulate_fault("normal")
            time.sleep(2)
        
        # Computation fault in spinnaker_1
        print("\n2. Simulating computation fault in spinnaker_1 (critical component)...")
        mock_hardware["spinnaker_1"].simulate_fault("computation")
        time.sleep(2)
        
        # Show spinnaker_1 status
        spinnaker_status = system.get_hardware_status("spinnaker_1")
        print("\nspinnaker_1 Status:")
        print(f"Health Score: {spinnaker_status['health_score']:.2f}")
        print(f"Error Rate: {spinnaker_status['current_metrics'].get('error_rate', 'N/A')}")
        print(f"Active Faults: {len(spinnaker_status['active_faults'])}")
        
        # Communication fault in truenorth_1
        print("\n3. Simulating communication fault in truenorth_1 (critical component)...")
        mock_hardware["truenorth_1"].is_active = False  # Stop sending metrics
        time.sleep(6)  # Wait for communication fault to be detected
        
        # Show system health after multiple faults
        health = system.get_hardware_status()
        print("\nSystem Health after Multiple Faults:")
        print(f"Overall Health: {health['system_health']['overall_health']:.2f}")
        print(f"Critical Components Health: {health['system_health']['critical_health']:.2f}")
        print(f"Active Faults: {health['system_health']['total_active_faults']}")
        
        # Show active faults by severity
        print("\nActive Faults by Severity:")
        for severity, count in health['system_health']['active_faults'].items():
            if count > 0:
                print(f"  {severity}: {count}")
        
        # Restore communication
        print("\nRestoring communication with truenorth_1...")
        mock_hardware["truenorth_1"].is_active = True
        time.sleep(2)
        
        # Final system health
        health = system.get_hardware_status()
        print("\nFinal System Health:")
        print(f"Overall Health: {health['system_health']['overall_health']:.2f}")
        print(f"Critical Components Health: {health['system_health']['critical_health']:.2f}")
        print(f"Active Components: {health['system_health']['active_components']} / {health['system_health']['total_components']}")
        print(f"Active Faults: {health['system_health']['total_active_faults']}")
        
        print("\nDemonstration completed.")
    
    finally:
        # Stop system
        system.stop()

if __name__ == "__main__":
    main()