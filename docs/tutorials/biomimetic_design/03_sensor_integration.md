# Biomimetic Sensor Integration

This tutorial explains how to implement and integrate biomimetic sensor systems inspired by biological sensory organs.

## Introduction

Biological organisms use a variety of sensory systems to perceive their environment and internal state. The Oblivion SDK's biomimetic sensor framework allows you to create sensor systems that mimic these biological capabilities, providing rich environmental awareness for your UCAV.

## Sensor Types

The biomimetic sensor framework supports several types of sensors inspired by biological sensory systems:

1. **Proprioceptive** - Body position and movement (similar to muscle spindles)
2. **Exteroceptive** - External environment perception
3. **Interoceptive** - Internal state monitoring
4. **Tactile** - Touch and pressure sensing
5. **Visual** - Vision systems
6. **Auditory** - Sound detection
7. **Chemical** - Chemical sensing
8. **Thermal** - Temperature sensing
9. **Electromagnetic** - EM field detection
10. **Flow** - Air/fluid flow sensing

## Sensor Interface Architecture

The `BiomimeticSensorInterface` provides a unified framework for connecting various sensor types to your UCAV's control systems. It handles:

- Sensor registration and management
- Mapping sensors to specific components
- Sensor data processing and fusion
- Event callbacks for sensor data updates

## Basic Sensor Setup

```python
from src.biomimetic.sensors.integration_framework import BiomimeticSensorInterface
from src.biomimetic.sensors.integration_framework import SensorDataType, SensorMapping
from src.simulation.sensors.sensor_framework import Sensor, SensorConfig, SensorType

# Initialize the sensor interface
sensor_interface = BiomimeticSensorInterface()

# Create a simple airflow sensor
airflow_config = SensorConfig(
    name="wing_airflow_sensor",
    type=SensorType.PRESSURE,
    update_rate=100,  # Hz
    noise_level=0.02  # 2% noise
)
airflow_sensor = Sensor(airflow_config)

# Add the sensor to the interface
sensor_interface.add_sensor(airflow_sensor)

# Create a mapping between the sensor and a wing component
airflow_mapping = SensorMapping(
    sensor_id="wing_airflow_sensor",
    sensor_type=SensorType.PRESSURE,
    data_type=SensorDataType.FLOW,
    target_component="left_wing",
    data_field="pressure",
    scaling_factor=1.0,
    threshold=0.1  # Ignore small pressure changes
)

# Add the mapping
sensor_interface.add_mapping(airflow_mapping)
```

## Creating a Proprioceptive Sensor

Proprioceptive sensors monitor the position and movement of body parts, similar to how birds sense the position of their wings.

```python
from src.biomimetic.sensors.integration_framework import ProprioceptiveSensor

# Create a proprioceptive sensor for wing position
wing_position_sensor = ProprioceptiveSensor(
    group_id="left_wing_muscles",
    sensor_interface=sensor_interface
)

# The sensor is automatically registered with the sensor interface
print(f"Registered sensors: {list(sensor_interface.sensors.keys())}")
```

## Sensor Data Processing

The sensor interface processes raw sensor data and applies mappings to produce usable information for your control systems.

```python
# Create platform state and environment data
platform_state = {
    "time": 123.45,
    "hardware_state": {
        "actuator_states": {
            "left_wing_muscles": {
                "position": 0.75,
                "velocity": 0.1,
                "force": 0.5
            }
        }
    }
}

environment = {
    "wind": {
        "speed": 5.0,  # m/s
        "direction": [1.0, 0.0, 0.0]  # Unit vector
    },
    "temperature": 288.15,  # K
    "pressure": 101325  # Pa
}

# Update sensors and get processed data
sensor_data = sensor_interface.update(platform_state, environment)

# Access different levels of sensor data
raw_data = sensor_data["raw"]
fused_data = sensor_data["fused"]
processed_data = sensor_data["processed"]

print("Processed sensor data:")
for component, data in processed_data.items():
    print(f"{component}: {data}")
```

## Sensor Fusion

The sensor interface includes a fusion system that combines data from multiple sensors to provide more accurate and reliable information.

```python
from src.core.fusion.sensor_fusion import FusionConfig

# Configure sensor fusion
fusion_config = FusionConfig(
    fusion_method="kalman",
    time_window=0.1,  # 100ms fusion window
    confidence_threshold=0.7
)

# Create sensor interface with fusion configuration
sensor_interface = BiomimeticSensorInterface(fusion_config)

# Add multiple sensors that measure related quantities
# (e.g., IMU, visual odometry, and GPS for position)
# ...

# When updated, the fusion system will combine these measurements
sensor_data = sensor_interface.update(platform_state, environment)
fused_position = sensor_data["fused"].get("position")
```

## Biomimetic Sensor Callbacks

You can register callbacks to be notified when new sensor data is available for a specific component.

```python
def on_wing_sensor_update(data):
    position = data.get("position", 0.0)
    airflow = data.get("airflow", 0.0)
    print(f"Wing position: {position:.2f}, Airflow: {airflow:.2f}")
    
    # Implement control logic based on sensor data
    if airflow < 0.2:
        print("WARNING: Potential stall condition detected")

# Register the callback for the left wing component
sensor_interface.register_callback("left_wing", on_wing_sensor_update)
```

## Integration with the Biomimetic Framework

The sensor interface is designed to work seamlessly with the Biomimetic Integration Framework.

```python
from src.biomimetic.integration.framework import BiomimeticIntegrationFramework

# Initialize the framework
framework = BiomimeticIntegrationFramework()

# Initialize with sensors module
framework.initialize(modules=["hardware", "sensors"])

# The framework automatically creates and connects appropriate sensors
# based on the hardware configuration

# Update the framework to process sensor data
state = framework.update(dt=0.01)

# Access sensor data from the state
if "sensors" in state:
    sensor_data = state["sensors"]
    print(f"Sensor data: {sensor_data}")
```

## Creating a Bio-Inspired Visual System

This example shows how to create a visual system inspired by bird vision, with a wide field of view and specialized foveal region.

```python
from src.biomimetic.sensors.visual import BiomimeticVisualSensor
from src.biomimetic.sensors.integration_framework import SensorDataType, SensorMapping

# Create a biomimetic visual sensor with bird-like properties
visual_sensor = BiomimeticVisualSensor(
    name="raptor_vision",
    field_of_view=260,  # degrees (wider than human vision)
    resolution=[1024, 768],
    foveal_density=2.0,  # Higher resolution in the center
    peripheral_sensitivity=1.5,  # Enhanced motion detection in periphery
    update_rate=60  # Hz
)

# Add the sensor to the interface
sensor_interface.add_sensor(visual_sensor)

# Create mappings for different visual processing pathways
central_vision_mapping = SensorMapping(
    sensor_id="raptor_vision",
    sensor_type=SensorType.CAMERA,
    data_type=SensorDataType.VISUAL,
    target_component="target_tracking",
    data_field="central_image",
    scaling_factor=1.0
)

peripheral_vision_mapping = SensorMapping(
    sensor_id="raptor_vision",
    sensor_type=SensorType.CAMERA,
    data_type=SensorDataType.VISUAL,
    target_component="threat_detection",
    data_field="peripheral_image",
    scaling_factor=1.0,
    filter_function=lambda x: x.motion_vectors  # Extract motion data
)

# Add the mappings
sensor_interface.add_mapping(central_vision_mapping)
sensor_interface.add_mapping(peripheral_vision_mapping)
```

## Next Steps

Now that you understand how to implement biomimetic sensor systems, proceed to the [Adaptive Morphology](./04_adaptive_morphology.md) tutorial to learn how to create systems that can change their physical properties in response to environmental conditions.