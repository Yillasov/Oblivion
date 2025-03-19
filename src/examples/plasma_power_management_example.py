"""
Example demonstrating plasma stealth power management.
"""

import time
from typing import Dict, Any
import matplotlib.pyplot as plt

# Add StealthType to the imports
from src.stealth.plasma.plasma_system import PlasmaStealthSystem
from src.stealth.plasma.plasma_generator import (
    PlasmaGenerator, 
    PlasmaControlSystem, 
    PlasmaGeneratorSpecs,
    PlasmaGenerationMethod,
    PlasmaPulsePattern
)
from src.stealth.plasma.power_management import (
    PlasmaStealthPowerManager,
    PowerProfile,
    PowerState,
    PowerOptimizer,
    PowerMonitor,
    create_default_power_profile
)
from src.stealth.base.config import StealthSystemConfig, StealthPowerMode, StealthType


def run_power_management_example():
    """Run a demonstration of plasma stealth power management."""
    print("Initializing plasma stealth power management demonstration...")
    
    # Create plasma stealth system - use the enum value instead of string
    config = StealthSystemConfig(
        stealth_type=StealthType.PLASMA_STEALTH,
        name="Advanced Plasma Stealth",
        description="High-performance plasma stealth system",
        weight_kg=150.0,
        power_requirements_kw=25.0,
        activation_time_seconds=2.0,
        cooldown_time_seconds=30.0,
        operational_duration_minutes=60.0
    )
    plasma_system = PlasmaStealthSystem(config)
    plasma_system.initialize()
    
    # Create plasma generator
    generator_specs = PlasmaGeneratorSpecs(
        max_power_kw=30.0,
        max_density=1.0e13,
        generation_method=PlasmaGenerationMethod.MAGNETRON,
        frequency_range=(0.5, 40.0),
        pulse_capabilities=[
            PlasmaPulsePattern.CONTINUOUS,
            PlasmaPulsePattern.REGULAR_PULSE,
            PlasmaPulsePattern.MODULATED
        ],
        startup_time_ms=500.0,
        cooldown_time_ms=2000.0,
        weight_kg=45.0,
        dimensions_cm=(80.0, 40.0, 25.0)
    )
    plasma_generator = PlasmaGenerator(generator_specs)
    
    # Create control system
    control_system = PlasmaControlSystem(plasma_system, plasma_generator)
    
    # Create power manager
    power_profile = create_default_power_profile()
    power_manager = PlasmaStealthPowerManager(control_system, power_profile, max_power_kw=50.0)
    
    # Create power optimizer and monitor
    power_optimizer = PowerOptimizer(power_manager)
    power_monitor = PowerMonitor(power_manager)
    
    # Power on the system
    print("Powering on plasma stealth system...")
    power_manager.power_on()
    
    # Display initial status
    status = power_manager.get_power_status()
    print(f"\nInitial power status: {status['power_state']}")
    
    # Activate the system in balanced mode
    print("Activating plasma stealth system in BALANCED mode...")
    power_manager.set_power_mode(StealthPowerMode.BALANCED)
    power_manager.activate()
    
    # Display active status
    status = power_manager.get_power_status()
    print(f"\nActive power status:")
    print(f"- State: {status['power_state']}")
    print(f"- Mode: {status['power_mode']}")
    print(f"- Current power: {status['current_power_kw']:.2f} kW")
    print(f"- Power allocation:")
    for component, power in status['power_allocation'].items():
        print(f"  - {component}: {power:.2f} kW")
    print(f"- Power efficiency: {status['power_efficiency']:.2f}")
    print(f"- Temperature: {status['temperature']:.1f}°C")
    print(f"- Remaining time: {status['remaining_operational_time_minutes']:.1f} minutes")
    
    # Simulate system operation with different power modes
    print("\nSimulating system operation with different power modes...")
    
    # Data for plotting
    timestamps = []
    power_values = []
    temperatures = []
    efficiencies = []
    
    # Simulate for 5 virtual minutes
    simulation_steps = 5
    for i in range(simulation_steps):
        # Switch power mode every minute
        if i == 1:
            print("Switching to ECO mode...")
            power_manager.set_power_mode(StealthPowerMode.ECO)
        elif i == 2:
            print("Switching to PERFORMANCE mode...")
            power_manager.set_power_mode(StealthPowerMode.PERFORMANCE)
        elif i == 3:
            print("Switching to MAXIMUM mode...")
            power_manager.set_power_mode(StealthPowerMode.MAXIMUM)
        elif i == 4:
            print("Switching back to BALANCED mode...")
            power_manager.set_power_mode(StealthPowerMode.BALANCED)
        
        # Simulate 60 seconds of operation
        for j in range(6):
            # Update power manager (10 seconds per step)
            power_manager.update(10.0)
            
            # Get status
            status = power_manager.get_power_status()
            
            # Record data
            timestamps.append(i * 60 + j * 10)
            power_values.append(status["current_power_kw"])
            temperatures.append(status["temperature"])
            efficiencies.append(status["power_efficiency"])
            
            # Check for alerts
            alerts = power_monitor.check_alerts()
            if alerts:
                print(f"\nALERTS at t={i * 60 + j * 10}s:")
                for alert in alerts:
                    print(f"- {alert['severity'].upper()} {alert['type']}: {alert['message']}")
    
    # Generate power report
    print("\nFinal power report:")
    report = power_monitor.get_power_report()
    print(f"- Power state: {report['power_state']}")
    print(f"- Power mode: {report['power_mode']}")
    print(f"- Current power: {report['current_power_kw']:.2f} kW")
    print(f"- Power distribution:")
    for component, percentage in report['power_distribution_percent'].items():
        print(f"  - {component}: {percentage:.1f}%")
    print(f"- Power efficiency: {report['power_efficiency']:.2f}")
    print(f"- Temperature: {report['temperature']:.1f}°C")
    print(f"- Total energy used: {report['total_energy_used_kwh']:.2f} kWh")
    print(f"- Remaining time: {report['remaining_time_minutes']:.1f} minutes")
    
    # Power off the system
    print("\nPowering off plasma stealth system...")
    power_manager.power_off()
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, power_values, 'b-', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Power (kW)')
    plt.title('Plasma Stealth Power Management')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, temperatures, 'r-', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(timestamps, efficiencies, 'g-', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Power Efficiency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nPower management demonstration completed.")


if __name__ == "__main__":
    run_power_management_example()