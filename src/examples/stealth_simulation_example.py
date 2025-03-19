"""
Example demonstrating the use of stealth simulation capabilities.
"""

import numpy as np
from typing import Dict, Any

from src.simulation.stealth.rcs_simulator import (
    RCSSimulator, 
    RCSSimulationConfig, 
    RCSFrequencyBand
)
from src.simulation.stealth.ir_signature_simulator import (
    IRSignatureSimulator,
    IRSignatureConfig,
    IRBand
)
from src.stealth.infrared.infrared_suppression import InfraredSuppressionSystem
from src.stealth.base.config import StealthSystemConfig


def run_stealth_simulation_example():
    """Run a simple stealth simulation example."""
    # Create RCS simulator
    rcs_config = RCSSimulationConfig(
        frequency_band=RCSFrequencyBand.X_BAND,
        angle_resolution=5.0,
        include_propulsion_effects=True,
        include_material_effects=True,
        include_shape_effects=True
    )
    rcs_simulator = RCSSimulator(rcs_config)
    
    # Create IR signature simulator
    ir_config = IRSignatureConfig(
        ir_band=IRBand.MID_WAVE,
        resolution=0.5,
        include_propulsion_effects=True,
        include_surface_heating=True,
        include_atmospheric_effects=True,
        ambient_temperature=20.0
    )
    ir_simulator = IRSignatureSimulator(ir_config)
    
    # Create sample platform geometry
    platform_geometry = {
        "length": 15.0,  # meters
        "width": 10.0,   # meters
        "height": 3.0    # meters
    }
    ir_simulator.register_platform_geometry(platform_geometry)
    
    # Create sample RCS data (simplified for example)
    # In a real application, this would come from measurements or detailed simulations
    shape_id = "ucav_standard"
    rcs_data = np.ones((72, 37)) * 0.5  # Base RCS of 0.5 m²
    # Add some variation based on angle
    for i in range(72):  # azimuth
        for j in range(37):  # elevation
            # Front has lower RCS than sides
            angle_factor = abs(np.sin(np.radians(i * 5)))
            rcs_data[i, j] *= (0.2 + angle_factor * 0.8)
    
    # Register the RCS data
    rcs_simulator.load_shape_data(shape_id, rcs_data)
    
    # Create sample material properties
    material_properties = {
        "emissivity": 0.85,
        "thermal_conductivity": 237.0,  # W/(m·K) for aluminum
        "specific_heat": 897.0,  # J/(kg·K) for aluminum
        "density": 2700.0  # kg/m³ for aluminum
    }
    ir_simulator.register_material("aluminum_alloy", material_properties)
    
    # Create sample platform state
    platform_state = {
        "position": np.array([0.0, 0.0, 5000.0]),  # 5000m altitude
        "velocity": np.array([250.0, 0.0, 0.0]),   # 250 m/s forward
        "orientation": np.array([0.0, 0.0, 0.0]),  # Level flight
        "altitude": 5000.0,
        "speed": 250.0,
        "propulsion": {
            "engine1": {
                "power_level": 0.7,
                "temperature": 800.0,  # °C
                "thrust": 45000.0,     # N
                "fuel_flow": 0.8       # kg/s
            }
        }
    }
    
    # Create sample environmental conditions
    environmental_conditions = {
        "temperature": 15.0,  # °C
        "pressure": 101325.0,  # Pa (sea level)
        "humidity": 0.6,       # 60%
        "precipitation": 0.0,  # No rain
        "wind": np.array([5.0, 2.0, 0.0]),  # 5 m/s from west, 2 m/s from south
        "time_of_day": "day"
    }
    
    # Run RCS simulation
    rcs_results = rcs_simulator.calculate_rcs(
        shape_id=shape_id,
        azimuth=0.0,  # Looking from front
        elevation=0.0,  # Level
        propulsion_state=platform_state["propulsion"]
    )
    
    # Run IR signature simulation
    ir_results = ir_simulator.calculate_signature(
        platform_state=platform_state,
        environmental_conditions=environmental_conditions
    )
    
    # Generate IR image
    ir_image = ir_simulator.generate_ir_image(
        platform_state=platform_state,
        environmental_conditions=environmental_conditions,
        view_angle=(0.0, 0.0)  # Looking from front, level
    )
    
    # Print results
    print("\n=== Stealth Simulation Results ===")
    print("\nRCS Results:")
    print(f"  Base RCS: {rcs_results['base_rcs']:.4f} m²")
    print(f"  Material Factor: {rcs_results['material_factor']:.4f}")
    print(f"  Propulsion Factor: {rcs_results['propulsion_factor']:.4f}")
    print(f"  Final RCS: {rcs_results['final_rcs']:.4f} m²")
    print(f"  Frequency Band: {rcs_results['frequency_band']}")
    
    print("\nIR Signature Results:")
    print(f"  Total Signature: {ir_results['total_signature']:.4f}")
    print(f"  Propulsion Component: {ir_results['components']['propulsion']:.4f}")
    print(f"  Surface Component: {ir_results['components']['surface']:.4f}")
    print(f"  Suppression Factor: {ir_results['factors']['suppression']:.4f}")
    print(f"  Atmospheric Factor: {ir_results['factors']['atmospheric']:.4f}")
    print(f"  IR Band: {ir_results['ir_band']}")
    
    print("\nIR Detection Ranges:")
    for sensor_type, range_value in ir_results['detection_ranges'].items():
        print(f"  {sensor_type}: {range_value:.1f} km")
    
    print("\nIR Image Shape:", ir_image.shape)
    print(f"  Max Temperature: {np.max(ir_image):.1f}°C")
    print(f"  Min Temperature: {np.min(ir_image):.1f}°C")
    print(f"  Mean Temperature: {np.mean(ir_image):.1f}°C")


if __name__ == "__main__":
    run_stealth_simulation_example()