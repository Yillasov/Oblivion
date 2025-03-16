# Core API Reference

This document provides detailed information about the core modules of the Oblivion SDK.

## Simulation Module

### Environment

```python
from oblivion.simulation import Environment
```

The Environment class provides the main simulation environment for UCAV testing.

#### Methods

- `create_scenario(config: dict) -> Scenario`
  - Creates a new flight scenario with the given configuration
  - Parameters:
    - `config`: Dictionary containing scenario parameters (altitude, speed, weather, etc.)

- `run_simulation(scenario: Scenario, controller: FlightController) -> SimulationResults`
  - Executes a simulation with the given scenario and controller
  - Parameters:
    - `scenario`: A Scenario object defining the simulation parameters
    - `controller`: A FlightController instance managing flight behavior

## Control Module

### FlightController

```python
from oblivion.control import FlightController
```

The FlightController class manages UCAV flight behavior and control systems.

#### Methods

- `attach_neural_network(network: NeuralNetwork) -> None`
  - Integrates a neural network for autonomous control
  - Parameters:
    - `network`: A NeuralNetwork instance for decision making

- `set_flight_parameters(params: dict) -> None`
  - Configures flight control parameters
  - Parameters:
    - `params`: Dictionary of flight parameters

## Neuromorphic Module

### NeuralNetwork

```python
from oblivion.neuromorphic import NeuralNetwork
```

The NeuralNetwork class handles neuromorphic computing integration.

#### Methods

- `load_model(model_name: str) -> bool`
  - Loads a pre-trained neural model
  - Parameters:
    - `model_name`: Name of the model to load

- `train(training_data: Dataset) -> TrainingResults`
  - Trains the neural network on provided data
  - Parameters:
    - `training_data`: Dataset object containing training examples

## Manufacturing Module

### DesignOptimizer

```python
from oblivion.manufacturing import DesignOptimizer
```

The DesignOptimizer class provides tools for UCAV design optimization.

#### Methods

- `optimize_design(constraints: dict) -> OptimizedDesign`
  - Optimizes UCAV design based on given constraints
  - Parameters:
    - `constraints`: Dictionary of design constraints

- `export_specifications(format: str) -> str`
  - Exports design specifications in the specified format
  - Parameters:
    - `format`: Output format ("pdf", "json", etc.)

## Error Handling

All core modules use the `OblivionError` base class for error handling:

```python
from oblivion.core.errors import OblivionError

try:
    env = Environment()
    # ... your code ...
except OblivionError as e:
    print(f"Error: {e}")
```

## Configuration

Core modules can be configured using YAML files in the `configs/` directory:

```yaml
# configs/simulation.yaml
simulation:
  update_rate: 60  # Hz
  physics_engine: "advanced"
  logging_level: "info"
```

## Best Practices

1. Always initialize the Environment before creating controllers
2. Use context managers for resource management
3. Implement proper error handling
4. Follow the configuration hierarchy

## See Also

- [Getting Started Guide](../tutorials/getting_started.md)
- [Hardware Integration](../specifications/hardware.md)
- [Example Projects](../examples/README.md)