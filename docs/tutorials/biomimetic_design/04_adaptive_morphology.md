# Adaptive Morphology

This tutorial explains how to implement adaptive morphology in your UCAV designs, allowing systems to change their physical properties in response to environmental conditions.

## Introduction

Adaptive morphology is a key biomimetic principle inspired by how living organisms modify their shape and structure to optimize performance in different conditions. Birds, for example, continuously adjust their wing shape during flight to optimize for different flight regimes (takeoff, cruising, landing).

## Biomimetic Adaptive Morphology

In the Oblivion SDK, adaptive morphology is implemented through several mechanisms:

1. **Morphing Structures** - Components that can change shape, size, or properties
2. **Adaptive Control Surfaces** - Control surfaces that adjust their characteristics based on flight conditions
3. **Reconfigurable Systems** - Systems that can reorganize their components for different missions

## Key Concepts

### Morphological Adaptation Types

- **Passive Adaptation** - Changes in response to external forces without active control
- **Active Adaptation** - Controlled changes using actuators and control systems
- **Hybrid Adaptation** - Combination of passive and active adaptation mechanisms

### Adaptation Timescales

- **Rapid Adaptation** - Millisecond to second timescale (e.g., wing flutter control)
- **Medium Adaptation** - Second to minute timescale (e.g., flight mode transitions)
- **Slow Adaptation** - Hour to day timescale (e.g., mission-specific reconfiguration)

## Implementing Adaptive Wing Morphology

### Basic Setup

```python
from src.biomimetic.integration.framework import BiomimeticIntegrationFramework
from src.biomimetic.actuators.morphing import MorphingWingController

# Initialize the framework
framework = BiomimeticIntegrationFramework()
framework.initialize(modules=["hardware", "sensors", "control"])

# Create a morphing wing controller
wing_controller = MorphingWingController(
    hardware_interface=framework.hardware_integration.hardware_interface
)

# Register the controller with the framework
framework.register_callback("update", wing_controller.update)
```

### Defining Morphing Parameters

```python
# Define morphing parameters for different flight regimes
morphing_configs = {
    "cruise": {
        "sweep_angle": 15.0,      # degrees
        "twist": 2.0,            # degrees
        "camber": 0.03,          # as fraction of chord
        "span_extension": 0.0    # as fraction of total span
    },
    "high_speed": {
        "sweep_angle": 35.0,
        "twist": 1.0,
        "camber": 0.01,
        "span_extension": -0.1   # Reduce span for high speed
    },
    "loiter": {
        "sweep_angle": 5.0,
        "twist": 3.0,
        "camber": 0.05,
        "span_extension": 0.1    # Increase span for efficiency
    },
    "landing": {
        "sweep_angle": 0.0,
        "twist": 5.0,
        "camber": 0.08,
        "span_extension": 0.0
    }
}

# Set initial configuration
wing_controller.set_morphing_config(morphing_configs["cruise"])
```

### Creating an Adaptive Controller

```python
class AdaptiveMorphologyController:
    """Controller that adapts wing morphology based on flight conditions."""
    
    def __init__(self, wing_controller, morphing_configs):
        self.wing_controller = wing_controller
        self.morphing_configs = morphing_configs
        self.current_mode = "cruise"
        self.transition_time = 0.0
        self.transition_duration = 2.0  # seconds
        self.transitioning = False
        self.start_config = None
        self.target_config = None
    
    def update(self, state):
        """Update the morphing controller based on current state."""
        # Extract relevant state information
        if "sensors" in state and "processed" in state["sensors"]:
            sensor_data = state["sensors"]["processed"]
            
            # Get airspeed if available
            airspeed = 0.0
            if "air_data" in sensor_data and "airspeed" in sensor_data["air_data"]:
                airspeed = sensor_data["air_data"]["airspeed"]
            
            # Determine appropriate flight mode based on airspeed
            new_mode = self._determine_flight_mode(airspeed)
            
            # If mode changed, start transition
            if new_mode != self.current_mode and not self.transitioning:
                self._start_transition(new_mode)
            
            # Update transition if in progress
            if self.transitioning:
                self._update_transition(state.get("time", 0.0))
    
    def _determine_flight_mode(self, airspeed):
        """Determine flight mode based on airspeed."""
        if airspeed > 250:  # km/h
            return "high_speed"
        elif airspeed < 100:
            return "landing" if self._is_landing_phase() else "loiter"
        else:
            return "cruise"
    
    def _is_landing_phase(self):
        """Determine if aircraft is in landing phase."""
        # This would use additional sensor data to determine landing phase
        # Simplified for this example
        return False
    
    def _start_transition(self, new_mode):
        """Start transition to a new morphing configuration."""
        self.transitioning = True
        self.transition_time = 0.0
        self.start_config = self._get_current_config()
        self.target_config = self.morphing_configs[new_mode]
        self.current_mode = new_mode
        print(f"Starting transition to {new_mode} mode")
    
    def _get_current_config(self):
        """Get current wing configuration."""
        return self.wing_controller.get_current_config()
    
    def _update_transition(self, current_time):
        """Update the transition between configurations."""
        self.transition_time += 0.01  # Assuming 10ms update rate
        
        # Calculate transition progress (0.0 to 1.0)
        progress = min(self.transition_time / self.transition_duration, 1.0)
        
        # Interpolate between start and target configurations
        current_config = {}
        for param in self.start_config:
            start_value = self.start_config[param]
            target_value = self.target_config[param]
            current_config[param] = start_value + progress * (target_value - start_value)
        
        # Apply the interpolated configuration
        self.wing_controller.set_morphing_config(current_config)
        
        # Check if transition is complete
        if progress >= 1.0:
            self.transitioning = False
            print(f"Completed transition to {self.current_mode} mode")

# Create and register the adaptive controller
adaptive_controller = AdaptiveMorphologyController(wing_controller, morphing_configs)
framework.register_callback("update", adaptive_controller.update)
```

## Implementing Biomimetic Actuators

Biomimetic actuators are essential for implementing adaptive morphology. These actuators are inspired by biological muscle systems.

```python
from src.biomimetic.actuators.muscle import BiomimeticMuscleActuator

# Create a biomimetic muscle actuator for wing morphing
wing_muscle = BiomimeticMuscleActuator(
    name="wing_morphing_muscle",
    max_force=500.0,         # Newtons
    contraction_ratio=0.25,   # Maximum contraction as fraction of length
    response_time=0.1,        # seconds
    energy_efficiency=0.7     # efficiency ratio
)

# Configure the actuator
wing_muscle.configure(
    activation_curve="sigmoid",  # Activation response curve
    fatigue_model="exponential",  # How the actuator fatigues with use
    recovery_rate=0.05           # Recovery rate when not in use
)

# Use the actuator to control wing twist
def control_wing_twist(target_twist):
    current_twist = wing_controller.get_current_config()["twist"]
    error = target_twist - current_twist
    
    # Calculate required muscle activation (0.0 to 1.0)
    activation = 0.5 + error * 0.1  # Simple proportional control
    activation = max(0.0, min(1.0, activation))  # Clamp to valid range
    
    # Activate the muscle
    force = wing_muscle.activate(activation)
    
    # The force would be applied to the wing structure
    # (simplified for this example)
    new_twist = current_twist + force * 0.01
    
    return new_twist
```

## Sensing for Adaptive Morphology

Effective adaptive morphology requires appropriate sensing to detect when adaptation is needed.

```python
from src.biomimetic.sensors.integration_framework import SensorMapping, SensorDataType
from src.simulation.sensors.sensor_framework import SensorType

# Create mappings for sensors relevant to morphing control
airflow_mapping = SensorMapping(
    sensor_id="wing_pressure_array",
    sensor_type=SensorType.PRESSURE_ARRAY,
    data_type=SensorDataType.FLOW,
    target_component="morphing_controller",
    data_field="pressure_distribution",
    scaling_factor=1.0
)

strain_mapping = SensorMapping(
    sensor_id="wing_strain_sensors",
    sensor_type=SensorType.STRAIN,
    data_type=SensorDataType.PROPRIOCEPTIVE,
    target_component="morphing_controller",
    data_field="structural_strain",
    scaling_factor=1.0
)

# Add mappings to the sensor interface
framework.sensor_interface.add_mapping(airflow_mapping)
framework.sensor_interface.add_mapping(strain_mapping)

# Create a callback to process the sensor data
def on_morphing_sensor_update(data):
    if "pressure_distribution" in data:
        # Analyze pressure distribution to detect flow separation
        pressure_data = data["pressure_distribution"]
        flow_separation = detect_flow_separation(pressure_data)
        
        if flow_separation:
            # Adjust camber to prevent separation
            current_config = wing_controller.get_current_config()
            current_config["camber"] += 0.01  # Increase camber slightly
            wing_controller.set_morphing_config(current_config)
    
    if "structural_strain" in data:
        # Monitor structural strain to prevent damage
        strain_data = data["structural_strain"]
        max_strain = max(strain_data.values())
        
        if max_strain > 0.8:  # 80% of maximum allowable strain
            # Reduce loading by adjusting configuration
            current_config = wing_controller.get_current_config()
            current_config["span_extension"] -= 0.05  # Reduce span extension
            wing_controller.set_morphing_config(current_config)

# Register the callback
framework.sensor_interface.register_callback("morphing_controller", on_morphing_sensor_update)

# Helper function to detect flow separation from pressure data
def detect_flow_separation(pressure_data):
    # Simplified algorithm to detect flow separation
    # In a real implementation, this would use more sophisticated analysis
    chord_positions = sorted(pressure_data.keys())
    pressure_gradient = []
    
    for i in range(1, len(chord_positions)):
        pos1 = chord_positions[i-1]
        pos2 = chord_positions[i]
        gradient = (pressure_data[pos2] - pressure_data[pos1]) / (pos2 - pos1)
        pressure_gradient.append(gradient)
    
    # Check for sign change in pressure gradient (simplified indicator of separation)
    for i in range(1, len(pressure_gradient)):
        if pressure_gradient[i-1] < 0 and pressure_gradient[i] > 0:
            return True
    
    return False
```

## Complete Example: Bird-Inspired Adaptive Wing

```python
from src.biomimetic.integration.framework import BiomimeticIntegrationFramework
from src.biomimetic.design.principles import BiomimeticDesignFramework, BiomimeticPrinciple

# Initialize frameworks
design_framework = BiomimeticDesignFramework()
integration_framework = BiomimeticIntegrationFramework()

# Initialize the integration framework
integration_framework.initialize(modules=["design", "hardware", "sensors", "power"])

# Get biological reference model for inspiration
swift = design_framework.get_biological_reference("common_swift")

# Create a swift-inspired adaptive wing system
class SwiftInspiredWing:
    def __init__(self, framework, swift_reference):
        self.framework = framework
        self.reference = swift_reference
        self.wing_aspect_ratio = swift_reference.performance_metrics["aspect_ratio"]
        self.wing_loading = swift_reference.performance_metrics["wing_loading_n_per_sqm"]
        
        # Configure wing parameters based on swift morphology (scaled)
        scale_factor = 10.0  # UCAV is 10x larger than a swift
        self.wingspan = swift_reference.morphological_data["wingspan_m"] * scale_factor
        self.wing_area = swift_reference.morphological_data["wing_area_sqm"] * (scale_factor ** 2)
        
        # Initialize morphing capabilities
        self.initialize_morphing_system()
    
    def initialize_morphing_system(self):
        # This would set up the actual morphing mechanisms
        # Simplified for this example
        print(f"Initializing Swift-inspired morphing wing system")
        print(f"Wingspan: {self.wingspan:.2f} m")
        print(f"Wing Area: {self.wing_area:.2f} m²")
        print(f"Aspect Ratio: {self.wing_aspect_ratio}")
        
        # Define morphing configurations based on swift flight modes
        self.morphing_configs = {
            "cruise": self._create_cruise_config(),
            "glide": self._create_glide_config(),
            "maneuver": self._create_maneuver_config(),
            "loiter": self._create_loiter_config()
        }
        
        # Set initial configuration
        self.current_config = "cruise"
        self.apply_configuration(self.current_config)
    
    def _create_cruise_config(self):
        # Based on swift's efficient cruising flight
        return {
            "sweep_angle": 15.0,
            "twist": 3.0,
            "camber": 0.04,
            "span_extension": 0.0
        }
    
    def _create_glide_config(self):
        # Based on swift's high-efficiency gliding
        return {
            "sweep_angle": 5.0,
            "twist": 2.0,
            "camber": 0.03,
            "span_extension": 0.1  # Extend span for better glide ratio
        }
    
    def _create_maneuver_config(self):
        # Based on swift's agile maneuvering
        return {
            "sweep_angle": 25.0,
            "twist": 4.0,
            "camber": 0.05,
            "span_extension": -0.05  # Slightly reduced span for agility
        }
    
    def _create_loiter_config(self):
        # Based on swift's energy-efficient slow flight
        return {
            "sweep_angle": 0.0,
            "twist": 5.0,
            "camber": 0.06,
            "span_extension": 0.15  # Maximum span extension for efficiency
        }
    
    def apply_configuration(self, config_name):
        if config_name in self.morphing_configs:
            config = self.morphing_configs[config_name]
            # In a real implementation, this would apply the configuration
            # to the actual hardware
            print(f"Applying {config_name} configuration:")
            for param, value in config.items():
                print(f"  {param}: {value}")
            self.current_config = config_name
            return True
        return False
    
    def update(self, state):
        # Process sensor data and update wing configuration
        if "sensors" in state and "processed" in state["sensors"]:
            sensor_data = state["sensors"]["processed"]
            
            # Determine appropriate configuration based on flight state
            new_config = self._determine_optimal_configuration(sensor_data)
            
            # Apply if different from current
            if new_config != self.current_config:
                self.apply_configuration(new_config)
    
    def _determine_optimal_configuration(self, sensor_data):
        # Simplified decision logic based on sensor data
        if "air_data" in sensor_data:
            air_data = sensor_data["air_data"]
            
            # Extract relevant parameters
            airspeed = air_data.get("airspeed", 150)  # km/h
            altitude = air_data.get("altitude", 1000)  # m
            vertical_speed = air_data.get("vertical_speed", 0)  # m/s
            
            # Decision logic
            if airspeed < 100 and abs(vertical_speed) < 1.0:
                return "loiter"  # Slow, level flight
            elif vertical_speed < -1.0 and airspeed < 150:
                return "glide"   # Descending, efficient glide
            elif abs(vertical_speed) > 5.0 or "high_g" in sensor_data:
                return "maneuver"  # Rapid climb/descent or turning
            else:
                return "cruise"  # Default cruising flight
        
        # Default to cruise if insufficient data
        return "cruise"

# Create the swift-inspired wing system
swift_wing = SwiftInspiredWing(integration_framework, swift)

# Register update callback
integration_framework.register_callback("update", swift_wing.update)

# In a real application, you would now run the system
print("Swift-inspired adaptive wing system initialized and ready")
```

## Next Steps

Now that you understand how to implement adaptive morphology, proceed to the [Energy Efficiency Patterns](./05_energy_efficiency.md) tutorial to learn how to apply biological energy conservation principles to optimize power usage.