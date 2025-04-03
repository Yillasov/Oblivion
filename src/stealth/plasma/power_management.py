#!/usr/bin/env python3
"""
Power management system for plasma stealth technology.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum
from dataclasses import dataclass
import time

from src.stealth.plasma.plasma_system import PlasmaStealthSystem, PlasmaParameters
from src.stealth.plasma.plasma_generator import PlasmaGenerator, PlasmaControlSystem
from src.stealth.base.config import StealthPowerMode


@dataclass
class PowerProfile:
    """Power profile for plasma stealth operation."""
    base_power_kw: float
    peak_power_kw: float
    startup_power_kw: float
    cooldown_power_kw: float
    power_efficiency: float  # 0.0-1.0
    thermal_output_kw: float
    max_operational_time_minutes: float


class PowerState(Enum):
    """Power states for plasma stealth system."""
    OFF = "off"
    STARTING = "starting"
    IDLE = "idle"
    ACTIVE = "active"
    COOLDOWN = "cooldown"
    ERROR = "error"


class PlasmaStealthPowerManager:
    """Power management for plasma stealth systems."""
    
    def __init__(self, 
                plasma_control: PlasmaControlSystem, 
                power_profile: PowerProfile,
                max_power_kw: float = 50.0):
        """
        Initialize plasma stealth power manager.
        
        Args:
            plasma_control: Plasma control system
            power_profile: Power profile for the system
            max_power_kw: Maximum available power in kW
        """
        self.plasma_control = plasma_control
        self.power_profile = power_profile
        self.max_power_kw = max_power_kw
        
        self.power_state = PowerState.OFF
        self.current_power_kw = 0.0
        self.power_mode = StealthPowerMode.BALANCED
        self.power_allocation = {
            "generator": 0.0,  # kW
            "control_systems": 0.0,  # kW
            "cooling": 0.0,  # kW
        }
        
        self.power_history = []
        self.last_state_change = time.time()
        self.total_energy_used_kwh = 0.0
        self.last_power_update = time.time()
        
        # Power mode settings (power level multipliers)
        self.power_mode_settings = {
            StealthPowerMode.STANDBY: 0.1,
            StealthPowerMode.ECO: 0.4,
            StealthPowerMode.BALANCED: 0.7,
            StealthPowerMode.PERFORMANCE: 0.9,
            StealthPowerMode.MAXIMUM: 1.0
        }
        
        # Thermal management
        self.current_temperature = 20.0  # °C
        self.cooling_active = False
        
    def power_on(self) -> bool:
        """
        Power on the plasma stealth system.
        
        Returns:
            Success status
        """
        if self.power_state != PowerState.OFF:
            return False
            
        # Change state to starting
        self.power_state = PowerState.STARTING
        self.last_state_change = time.time()
        
        # Allocate startup power
        self.current_power_kw = self.power_profile.startup_power_kw
        self.power_allocation["generator"] = self.current_power_kw * 0.8
        self.power_allocation["control_systems"] = self.current_power_kw * 0.15
        self.power_allocation["cooling"] = self.current_power_kw * 0.05
        
        # Initialize plasma control system
        init_success = self.plasma_control.initialize()
        
        if init_success:
            # Move to idle state
            self.power_state = PowerState.IDLE
            self._update_power_usage()
            return True
        else:
            # Error during startup
            self.power_state = PowerState.ERROR
            return False
    
    def power_off(self) -> bool:
        """
        Power off the plasma stealth system.
        
        Returns:
            Success status
        """
        if self.power_state == PowerState.OFF:
            return True
            
        # Deactivate plasma system if active
        if self.power_state == PowerState.ACTIVE:
            self.plasma_control.deactivate()
        
        # Enter cooldown state
        self.power_state = PowerState.COOLDOWN
        self.last_state_change = time.time()
        
        # Allocate cooldown power
        self.current_power_kw = self.power_profile.cooldown_power_kw
        self.power_allocation["generator"] = 0.0
        self.power_allocation["control_systems"] = self.current_power_kw * 0.2
        self.power_allocation["cooling"] = self.current_power_kw * 0.8
        
        # After cooldown period, turn off
        # (In a real system, you'd use a timer or thread for this)
        self.power_state = PowerState.OFF
        self.current_power_kw = 0.0
        self.power_allocation = {k: 0.0 for k in self.power_allocation}
        
        return True
    
    def activate(self, power_level: Optional[float] = None) -> bool:
        """
        Activate plasma stealth with current power mode.
        
        Args:
            power_level: Optional override for power level (0.0-1.0)
            
        Returns:
            Success status
        """
        if self.power_state not in [PowerState.IDLE, PowerState.ACTIVE]:
            return False
            
        # Determine power level based on mode if not specified
        if power_level is None:
            power_level = self.power_mode_settings[self.power_mode]
        else:
            power_level = max(0.1, min(power_level, 1.0))
        
        # Calculate required power
        required_power = self.power_profile.base_power_kw + (
            (self.power_profile.peak_power_kw - self.power_profile.base_power_kw) * power_level
        )
        
        # Check if we have enough power
        if required_power > self.max_power_kw:
            # Not enough power, scale down
            power_level = (self.max_power_kw - self.power_profile.base_power_kw) / (
                self.power_profile.peak_power_kw - self.power_profile.base_power_kw
            )
            power_level = max(0.1, power_level)
            required_power = self.max_power_kw
        
        # Activate plasma system
        success = self.plasma_control.activate(
            power_level=power_level,
            pulse_mode=(self.power_mode == StealthPowerMode.ECO),
            pulse_frequency=1000.0
        )
        
        if success:
            self.power_state = PowerState.ACTIVE
            self.current_power_kw = required_power
            
            # Allocate power
            self.power_allocation["generator"] = required_power * 0.7
            self.power_allocation["control_systems"] = required_power * 0.15
            self.power_allocation["cooling"] = required_power * 0.15
            
            self._update_power_usage()
            return True
        else:
            return False
    
    def set_power_mode(self, mode: StealthPowerMode) -> bool:
        """
        Set power mode for plasma stealth system.
        
        Args:
            mode: Power mode to set
            
        Returns:
            Success status
        """
        if mode not in self.power_mode_settings:
            return False
            
        self.power_mode = mode
        
        # If system is active, adjust power level
        if self.power_state == PowerState.ACTIVE:
            power_level = self.power_mode_settings[mode]
            
            # Update plasma control system
            self.plasma_control.plasma_generator.set_power_level(power_level)
            
            # Update plasma system parameters
            self.plasma_control.plasma_system.adjust_parameters({
                "power_level": power_level,
                "pulse_mode": (mode == StealthPowerMode.ECO)
            })
            
            # Update power allocation
            required_power = self.power_profile.base_power_kw + (
                (self.power_profile.peak_power_kw - self.power_profile.base_power_kw) * power_level
            )
            self.current_power_kw = required_power
            
            self.power_allocation["generator"] = required_power * 0.7
            self.power_allocation["control_systems"] = required_power * 0.15
            self.power_allocation["cooling"] = required_power * 0.15
            
            self._update_power_usage()
        
        return True
    
    def get_power_status(self) -> Dict[str, Any]:
        """
        Get current power status.
        
        Returns:
            Power status dictionary
        """
        # Get plasma system status
        plasma_status = self.plasma_control.get_status()
        
        # Calculate remaining operational time
        remaining_time = self._calculate_remaining_time()
        
        return {
            "power_state": self.power_state.value,
            "power_mode": self.power_mode.name,
            "current_power_kw": self.current_power_kw,
            "power_allocation": self.power_allocation,
            "power_efficiency": self._calculate_current_efficiency(),
            "temperature": self.current_temperature,
            "cooling_active": self.cooling_active,
            "total_energy_used_kwh": self.total_energy_used_kwh,
            "remaining_operational_time_minutes": remaining_time,
            "plasma_system": {
                "active": plasma_status["stealth_system"]["active"],
                "power_level": plasma_status["stealth_system"]["power_level"],
                "plasma_density": plasma_status["plasma_generator"]["plasma_density"]
            }
        }
    
    def update(self, elapsed_seconds: float) -> None:
        """
        Update power management system.
        
        Args:
            elapsed_seconds: Elapsed time since last update in seconds
        """
        # Update energy usage
        self._update_energy_usage(elapsed_seconds)
        
        # Update thermal management
        self._update_thermal_management(elapsed_seconds)
        
        # Check if we need to transition from cooldown to off
        if self.power_state == PowerState.COOLDOWN:
            cooldown_time = time.time() - self.last_state_change
            if cooldown_time > 60:  # 1 minute cooldown
                self.power_state = PowerState.OFF
                self.current_power_kw = 0.0
                self.power_allocation = {k: 0.0 for k in self.power_allocation}
    
    def _update_power_usage(self) -> None:
        """Update power usage statistics."""
        now = time.time()
        elapsed = now - self.last_power_update
        self.last_power_update = now
        
        # Record power usage
        self.power_history.append({
            "timestamp": now,
            "power_kw": self.current_power_kw,
            "state": self.power_state.value,
            "mode": self.power_mode.name
        })
        
        # Trim history if it gets too long
        if len(self.power_history) > 1000:
            self.power_history = self.power_history[-1000:]
    
    def _update_energy_usage(self, elapsed_seconds: float) -> None:
        """
        Update energy usage based on elapsed time.
        
        Args:
            elapsed_seconds: Elapsed time in seconds
        """
        # Convert seconds to hours
        elapsed_hours = elapsed_seconds / 3600.0
        
        # Add energy usage
        self.total_energy_used_kwh += self.current_power_kw * elapsed_hours
    
    def _update_thermal_management(self, elapsed_seconds: float) -> None:
        """
        Update thermal management system.
        
        Args:
            elapsed_seconds: Elapsed time in seconds
        """
        # Simple thermal model
        if self.power_state == PowerState.ACTIVE:
            # Temperature increases based on power usage and cooling
            heat_generated = self.power_profile.thermal_output_kw * self.plasma_control.plasma_system.status["power_level"]
            cooling_power = self.power_allocation["cooling"]
            
            # Net temperature change
            temp_change = (heat_generated - cooling_power * 2.0) * 0.05 * elapsed_seconds
            self.current_temperature += temp_change
            
            # Activate cooling if temperature gets too high
            if self.current_temperature > 60.0:
                self.cooling_active = True
                # Increase cooling allocation
                cooling_increase = min(1.0, (self.current_temperature - 60.0) / 20.0) * 0.1 * self.current_power_kw
                self.power_allocation["cooling"] += cooling_increase
                self.power_allocation["generator"] -= cooling_increase
        else:
            # System cooling down
            if self.current_temperature > 20.0:
                self.current_temperature -= 0.1 * elapsed_seconds
                if self.current_temperature < 20.0:
                    self.current_temperature = 20.0
            
            if self.current_temperature < 40.0:
                self.cooling_active = False
    
    def _calculate_current_efficiency(self) -> float:
        """
        Calculate current power efficiency.
        
        Returns:
            Power efficiency (0.0-1.0)
        """
        if self.power_state != PowerState.ACTIVE:
            return 0.0
            
        # Base efficiency from profile
        efficiency = self.power_profile.power_efficiency
        
        # Adjust based on power level
        power_level = self.plasma_control.plasma_system.status["power_level"]
        
        # Power efficiency curve - typically more efficient at mid-range
        if power_level < 0.3:
            efficiency *= 0.7 + power_level
        elif power_level > 0.8:
            efficiency *= 1.1 - (power_level - 0.8)
            
        # Temperature affects efficiency
        if self.current_temperature > 50.0:
            temp_factor = 1.0 - ((self.current_temperature - 50.0) / 100.0)
            efficiency *= max(0.7, temp_factor)
            
        return min(1.0, efficiency)
    
    def _calculate_remaining_time(self) -> float:
        """
        Calculate remaining operational time based on current power usage.
        
        Returns:
            Remaining time in minutes
        """
        if self.power_state != PowerState.ACTIVE:
            return 0.0
            
        # Get max operational time from profile
        max_time = self.power_profile.max_operational_time_minutes
        
        # Adjust based on current power level
        power_level = self.plasma_control.plasma_system.status["power_level"]
        
        # Higher power levels drain energy faster
        adjusted_time = max_time * (1.0 - (power_level * 0.5))
        
        # Calculate time used so far
        time_active = (time.time() - self.last_state_change) / 60.0  # minutes
        
        remaining = max(0.0, adjusted_time - time_active)
        return remaining


def create_default_power_profile() -> PowerProfile:
    """
    Create default power profile for plasma stealth systems.
    
    Returns:
        Default power profile
    """
    return PowerProfile(
        base_power_kw=15.0,
        peak_power_kw=45.0,
        startup_power_kw=10.0,
        cooldown_power_kw=5.0,
        power_efficiency=0.85,
        thermal_output_kw=12.0,
        max_operational_time_minutes=60.0
    )


class PowerOptimizer:
    """Optimizer for plasma stealth power usage."""
    
    def __init__(self, power_manager: PlasmaStealthPowerManager):
        """
        Initialize power optimizer.
        
        Args:
            power_manager: Power manager to optimize
        """
        self.power_manager = power_manager
        self.optimization_active = False
        self.target_duration_minutes = 0.0
        self.last_optimization = time.time()
        self.optimization_interval = 10.0  # seconds
        
    def set_target_duration(self, duration_minutes: float) -> bool:
        """
        Set target operational duration.
        
        Args:
            duration_minutes: Target duration in minutes
            
        Returns:
            Success status
        """
        if duration_minutes <= 0:
            self.optimization_active = False
            return False
            
        self.target_duration_minutes = duration_minutes
        self.optimization_active = True
        return True
    
    def update(self) -> None:
        """Update power optimization."""
        if not self.optimization_active:
            return
            
        now = time.time()
        if now - self.last_optimization < self.optimization_interval:
            return
            
        self.last_optimization = now
        
        # Get current status
        status = self.power_manager.get_power_status()
        
        # If not active, nothing to optimize
        if status["power_state"] != PowerState.ACTIVE.value:
            return
            
        # Calculate optimal power level
        current_remaining = status["remaining_operational_time_minutes"]
        
        if current_remaining < self.target_duration_minutes * 0.9:
            # Need to conserve power
            self._reduce_power_usage()
        elif current_remaining > self.target_duration_minutes * 1.5:
            # Can use more power
            self._increase_power_usage()
    
    def _reduce_power_usage(self) -> None:
        """Reduce power usage to extend operational time."""
        # Get current power mode
        current_mode = self.power_manager.power_mode
        
        # Find next lower power mode
        modes = list(StealthPowerMode)
        current_index = modes.index(current_mode)
        
        if current_index > 0:
            # Switch to lower power mode
            new_mode = modes[current_index - 1]
            self.power_manager.set_power_mode(new_mode)
    
    def _increase_power_usage(self) -> None:
        """Increase power usage for better performance."""
        # Get current power mode
        current_mode = self.power_manager.power_mode
        
        # Find next higher power mode
        modes = list(StealthPowerMode)
        current_index = modes.index(current_mode)
        
        if current_index < len(modes) - 1:
            # Switch to higher power mode
            new_mode = modes[current_index + 1]
            self.power_manager.set_power_mode(new_mode)


class PowerMonitor:
    """Monitor for plasma stealth power system."""
    
    def __init__(self, power_manager: PlasmaStealthPowerManager):
        """
        Initialize power monitor.
        
        Args:
            power_manager: Power manager to monitor
        """
        self.power_manager = power_manager
        self.alert_thresholds = {
            "temperature": 70.0,  # °C
            "power_efficiency": 0.6,
            "remaining_time": 10.0  # minutes
        }
        self.alerts = []
        
    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for power system alerts.
        
        Returns:
            List of active alerts
        """
        # Clear previous alerts
        self.alerts = []
        
        # Get current status
        status = self.power_manager.get_power_status()
        
        # Check temperature
        if status["temperature"] > self.alert_thresholds["temperature"]:
            self.alerts.append({
                "type": "temperature",
                "severity": "high" if status["temperature"] > 80.0 else "medium",
                "message": f"System temperature high: {status['temperature']:.1f}°C",
                "value": status["temperature"]
            })
            
        # Check power efficiency
        if status["power_efficiency"] < self.alert_thresholds["power_efficiency"]:
            self.alerts.append({
                "type": "efficiency",
                "severity": "medium",
                "message": f"Power efficiency low: {status['power_efficiency']:.2f}",
                "value": status["power_efficiency"]
            })
            
        # Check remaining time
        if (status["power_state"] == PowerState.ACTIVE.value and 
            status["remaining_operational_time_minutes"] < self.alert_thresholds["remaining_time"]):
            self.alerts.append({
                "type": "remaining_time",
                "severity": "high" if status["remaining_operational_time_minutes"] < 5.0 else "medium",
                "message": f"Low operational time remaining: {status['remaining_operational_time_minutes']:.1f} minutes",
                "value": status["remaining_operational_time_minutes"]
            })
            
        return self.alerts
        
    def get_power_report(self) -> Dict[str, Any]:
        """
        Generate power usage report.
        
        Returns:
            Power usage report
        """
        status = self.power_manager.get_power_status()
        
        # Calculate power distribution percentages
        total_power = status["current_power_kw"]
        power_distribution = {}
        
        if total_power > 0:
            for key, value in status["power_allocation"].items():
                power_distribution[key] = (value / total_power) * 100.0
        else:
            power_distribution = {k: 0.0 for k in status["power_allocation"]}
            
        return {
            "timestamp": time.time(),
            "power_state": status["power_state"],
            "power_mode": status["power_mode"],
            "current_power_kw": status["current_power_kw"],
            "power_distribution_percent": power_distribution,
            "power_efficiency": status["power_efficiency"],
            "temperature": status["temperature"],
            "total_energy_used_kwh": status["total_energy_used_kwh"],
            "remaining_time_minutes": status["remaining_operational_time_minutes"],
            "alerts": self.check_alerts()
        }