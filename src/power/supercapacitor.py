from typing import Dict, Any, Tuple
import time
import logging

logger = logging.getLogger("supercapacitor")

class SupercapacitorSpecs:
    """Specifications for supercapacitor."""
    
    def __init__(self,
                 capacitance: float = 5000.0,  # Farads
                 max_charge_rate: float = 100.0,  # kW
                 max_discharge_rate: float = 100.0,  # kW
                 nominal_voltage: float = 2.7,  # V
                 energy_density: float = 10.0):  # Wh/kg
        """
        Initialize supercapacitor specifications.
        
        Args:
            capacitance: Capacitance in Farads
            max_charge_rate: Maximum charging rate in kW
            max_discharge_rate: Maximum discharging rate in kW
            nominal_voltage: Nominal voltage in V
            energy_density: Energy density in Wh/kg
        """
        self.capacitance = capacitance
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.nominal_voltage = nominal_voltage
        self.energy_density = energy_density
        
        # Calculate derived specifications
        self.weight = (capacitance * nominal_voltage) / energy_density  # kg
        self.volume = self.weight / 2.0  # Approximate volume in liters

class Supercapacitor:
    """Supercapacitor system."""
    
    def __init__(self, 
                 capacitor_id: str,
                 specs: SupercapacitorSpecs):
        """
        Initialize supercapacitor.
        
        Args:
            capacitor_id: Unique identifier
            specs: Supercapacitor specifications
        """
        self.capacitor_id = capacitor_id
        self.specs = specs
        self.charge_level = 0.5  # Initial charge at 50%
        self.current_energy = self.charge_level * self.specs.capacitance * self.specs.nominal_voltage
        self.temperature = 25.0  # °C
        self.status = {"active": False, "error": None}
        
        logger.info(f"Supercapacitor '{capacitor_id}' initialized with {self.specs.capacitance} Farads")

    def charge(self, power: float, duration: float) -> Dict[str, Any]:
        """
        Charge the supercapacitor.
        
        Args:
            power: Charging power in kW
            duration: Charging duration in hours
            
        Returns:
            Charging results
        """
        if not self.status["active"]:
            return {"success": False, "error": "Supercapacitor not active"}
        
        # Limit charging power to max rate
        actual_power = min(power, self.specs.max_charge_rate)
        
        # Calculate energy to add
        energy_to_add = actual_power * duration
        
        # Update energy and charge level
        prev_energy = self.current_energy
        self.current_energy = min(self.specs.capacitance * self.specs.nominal_voltage, self.current_energy + energy_to_add)
        self.charge_level = self.current_energy / (self.specs.capacitance * self.specs.nominal_voltage)
        
        # Update temperature (charging increases temperature)
        self._update_temperature(actual_power * 0.02)
        
        logger.info(f"Supercapacitor '{self.capacitor_id}' charged with {energy_to_add:.2f} kWh")
        
        return {
            "success": True,
            "energy_added": self.current_energy - prev_energy,
            "new_charge_level": self.charge_level,
            "temperature": self.temperature
        }

    def discharge(self, power: float, duration: float) -> Dict[str, Any]:
        """
        Discharge the supercapacitor.
        
        Args:
            power: Discharge power in kW
            duration: Discharge duration in hours
            
        Returns:
            Discharging results
        """
        if not self.status["active"]:
            return {"success": False, "error": "Supercapacitor not active"}
        
        # Limit discharge power to max rate
        actual_power = min(power, self.specs.max_discharge_rate)
        
        # Calculate energy to remove
        energy_to_remove = actual_power * duration
        
        # Check if we have enough energy
        if energy_to_remove > self.current_energy:
            actual_energy_removed = self.current_energy
            self.current_energy = 0
            self.charge_level = 0
        else:
            actual_energy_removed = energy_to_remove
            self.current_energy -= energy_to_remove
            self.charge_level = self.current_energy / (self.specs.capacitance * self.specs.nominal_voltage)
        
        # Update temperature (discharging increases temperature)
        self._update_temperature(actual_power * 0.01)
        
        logger.info(f"Supercapacitor '{self.capacitor_id}' discharged with {actual_energy_removed:.2f} kWh")
        
        return {
            "success": True,
            "energy_removed": actual_energy_removed,
            "new_charge_level": self.charge_level,
            "temperature": self.temperature
        }

    def _update_temperature(self, delta: float) -> None:
        """
        Update supercapacitor temperature.
        
        Args:
            delta: Temperature change in °C
        """
        self.temperature += delta
        ambient = 25.0
        cooling_factor = 0.05
        self.temperature += (ambient - self.temperature) * cooling_factor

class SupercapacitorArray:
    """Array of supercapacitors for increased capacity and redundancy."""
    
    def __init__(self, array_id: str, num_capacitors: int = 4):
        """
        Initialize supercapacitor array.
        
        Args:
            array_id: Unique identifier
            num_capacitors: Number of capacitors in array
        """
        self.array_id = array_id
        self.capacitors: Dict[str, Supercapacitor] = {}
        
        # Create capacitors
        for i in range(num_capacitors):
            capacitor_id = f"{array_id}_cap_{i}"
            self.capacitors[capacitor_id] = Supercapacitor(
                capacitor_id=capacitor_id,
                specs=SupercapacitorSpecs(
                    capacitance=5000.0,  # Farads
                    max_charge_rate=100.0,  # kW
                    max_discharge_rate=100.0  # kW
                )
            )
        
        self.active = False
        self.total_capacity = num_capacitors * 5000.0 * 2.7  # Total energy capacity in kWh
        self.current_energy = 0.0
        self.charge_level = 0.0
    
    def initialize(self) -> bool:
        """Initialize all capacitors in the array."""
        success = True
        for capacitor in self.capacitors.values():
            capacitor.status["active"] = True
        
        if success:
            self.active = True
            self._update_array_status()
        
        return success
    
    def _update_array_status(self) -> None:
        """Update array status based on individual capacitors."""
        total_energy = sum(c.current_energy for c in self.capacitors.values())
        self.current_energy = total_energy
        self.charge_level = total_energy / self.total_capacity if self.total_capacity > 0 else 0
    
    def charge(self, power: float, duration: float) -> Dict[str, Any]:
        """
        Charge the supercapacitor array.
        
        Args:
            power: Charging power in kW
            duration: Charging duration in hours
            
        Returns:
            Charging results
        """
        if not self.active:
            return {"success": False, "error": "Array not active"}
        
        active_capacitors = [c for c in self.capacitors.values() if c.status["active"]]
        if not active_capacitors:
            return {"success": False, "error": "No active capacitors in array"}
        
        sorted_capacitors = sorted(active_capacitors, key=lambda c: c.charge_level)
        
        results = []
        remaining_power = power
        
        for capacitor in sorted_capacitors:
            if remaining_power <= 0:
                break
                
            allocation_factor = 1.0 - (capacitor.charge_level / 2.0)
            capacitor_power = min(remaining_power, capacitor.specs.max_charge_rate * allocation_factor)
            
            result = capacitor.charge(capacitor_power, duration)
            results.append(result)
            
            if result["success"]:
                remaining_power -= capacitor_power
        
        self._update_array_status()
        
        return {
            "success": True,
            "energy_added": sum(r.get("energy_added", 0) for r in results if r["success"]),
            "new_charge_level": self.charge_level,
            "capacitor_results": results
        }
    
    def discharge(self, power: float, duration: float) -> Dict[str, Any]:
        """
        Discharge the supercapacitor array.
        
        Args:
            power: Discharge power in kW
            duration: Discharge duration in hours
            
        Returns:
            Discharging results
        """
        if not self.active:
            return {"success": False, "error": "Array not active"}
        
        active_capacitors = [c for c in self.capacitors.values() if c.status["active"]]
        if not active_capacitors:
            return {"success": False, "error": "No active capacitors in array"}
        
        sorted_capacitors = sorted(active_capacitors, key=lambda c: c.charge_level, reverse=True)
        
        results = []
        remaining_power = power
        
        for capacitor in sorted_capacitors:
            if remaining_power <= 0:
                break
                
            allocation_factor = capacitor.charge_level
            capacitor_power = min(remaining_power, capacitor.specs.max_discharge_rate * allocation_factor)
            
            result = capacitor.discharge(capacitor_power, duration)
            results.append(result)
            
            if result["success"]:
                remaining_power -= capacitor_power
        
        self._update_array_status()
        
        return {
            "success": True,
            "energy_removed": sum(r.get("energy_removed", 0) for r in results if r["success"]),
            "new_charge_level": self.charge_level,
            "capacitor_results": results
        }