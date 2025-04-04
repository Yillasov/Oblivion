# Biomimetic Integration Framework

This tutorial explains how to use the Biomimetic Integration Framework to connect various biomimetic components into a cohesive system.

## Introduction

The Biomimetic Integration Framework provides a unified interface for connecting different biomimetic modules, including design principles, hardware components, sensors, power systems, and materials. It ensures cross-module compatibility and manages system configuration.

## Framework Architecture

The integration framework consists of several key components:

1. **Design Framework** - Implements biomimetic design principles
2. **Hardware Integration** - Connects to physical or simulated hardware
3. **Sensor Interface** - Manages biomimetic sensors
4. **Power Integrator** - Handles energy management
5. **Material Selector** - Selects appropriate biomimetic materials
6. **Configuration Manager** - Manages system configuration
7. **Validator** - Ensures cross-module compatibility

## Getting Started

### Basic Setup

```python
from src.biomimetic.integration.framework import BiomimeticIntegrationFramework

# Initialize the framework
framework = BiomimeticIntegrationFramework()

# Initialize specific modules
framework.initialize(modules=["design", "hardware", "sensors", "power"])

# Check if initialization was successful
if framework.initialized:
    print("Framework initialized successfully")
    print(f"Active modules: {', '.join(framework.active_modules)}")
else:
    print("Framework initialization failed")
```

### Module Dependencies

The framework manages dependencies between modules automatically. For example, the sensor module depends on the hardware module, so initializing sensors will automatically initialize hardware if it hasn't been initialized yet.

The dependency graph is as follows:

```
hardware → power
sensors → hardware
control → sensors, hardware
manufacturing → materials
```

## System Configuration

The framework uses a configuration manager to load, save, and validate system configurations.

### Loading Configuration

```python
from src.biomimetic.integration.framework import BiomimeticIntegrationFramework

# Initialize with a specific configuration file
framework = BiomimeticIntegrationFramework(
    config_path="/path/to/custom_config.json"
)

# Load configuration
config = framework.config_manager.load_configuration()
print(f"Loaded configuration version: {config.get('version', 'unknown')}")
```

### Applying Configuration

```python
# Create a new configuration
new_config = {
    "power": {
        "state": "high_performance",
        "energy_harvesting": True
    },
    "hardware": {
        "wing_flapping": {
            "frequency": 3.0,
            "amplitude": 0.7
        }
    }
}

# Apply the configuration
if framework.apply_configuration(new_config):
    print("Configuration applied successfully")
else:
    print("Failed to apply configuration")
```

## System Update Loop

The framework provides an update method that should be called periodically to update all active modules.

```python
import time

# Main update loop
try:
    while True:
        # Update the system with a time step of 10ms
        state = framework.update(dt=0.01)
        
        # Print some state information
        if "hardware" in state:
            hw_state = state["hardware"]
            if "actuator_states" in hw_state:
                for group, data in hw_state["actuator_states"].items():
                    print(f"{group} position: {data['position']:.2f}")
        
        # Sleep to maintain update rate
        time.sleep(0.01)
        
 except KeyboardInterrupt:
    print("Stopping update loop")
```

## Event Callbacks

You can register callbacks to be notified of specific events.

```python
def on_system_update(state):
    print(f"System updated at time: {state.get('time', 0.0)}")
    
    # Check sensor data
    if "sensors" in state:
        sensor_data = state["sensors"]
        if "processed" in sensor_data:
            for component, data in sensor_data["processed"].items():
                print(f"Sensor data for {component}: {data}")

# Register the callback
framework.register_callback("update", on_system_update)
```

## Accessing Design Principles

You can access the active biomimetic principles in your design:

```python
# Get active principles
active_principles = framework.get_active_principles()

print("Active Biomimetic Principles:")
for principle in active_principles:
    principle_info = framework.design_framework.get_principle(principle)
    print(f"- {principle.name}: {principle_info['description']}")
```

## Hardware Integration

The framework connects to hardware components through the hardware integration module.

```python
# Configure wing flapping parameters
if framework.hardware_integration:
    success = framework.hardware_integration.configure_wing_flapping(
        frequency=2.5,  # Hz
        amplitude=0.6    # 0.0-1.0
    )
    
    if success:
        print("Wing flapping configured successfully")
    else:
        print("Failed to configure wing flapping")
```

## Complete Example

Here's a complete example that initializes the framework, configures it, and runs a simple update loop:

```python
from src.biomimetic.integration.framework import BiomimeticIntegrationFramework
import time

# Initialize the framework
framework = BiomimeticIntegrationFramework()

# Initialize all modules
if not framework.initialize():
    print("Failed to initialize framework")
    exit(1)

# Configure the system
config = {
    "power": {
        "state": "balanced",
        "energy_harvesting": True
    },
    "hardware": {
        "wing_flapping": {
            "frequency": 2.0,
            "amplitude": 0.5
        }
    }
}

framework.apply_configuration(config)

# Define update callback
def on_update(state):
    print(f"System updated")

# Register callback
framework.register_callback("update", on_update)

# Run update loop for 5 seconds
print("Running update loop for 5 seconds...")
start_time = time.time()
while time.time() - start_time < 5.0:
    framework.update(dt=0.01)
    time.sleep(0.01)

print("Update loop completed")
```

## Next Steps

Now that you understand how to use the Biomimetic Integration Framework, proceed to the [Sensor Integration](./03_sensor_integration.md) tutorial to learn how to implement biomimetic sensor systems.