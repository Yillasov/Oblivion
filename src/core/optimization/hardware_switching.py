"""
Standardized hardware switching optimizer for cross-subsystem optimization.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import time

from src.core.hardware.hardware_switcher import HardwareSwitcher, create_hardware_switcher
from src.core.hardware.exceptions import HardwareSwitchingError
from src.core.optimization.cross_subsystem import SubsystemOptimizer, OptimizationPriority

logger = logging.getLogger(__name__)

class HardwareSwitchingOptimizer(SubsystemOptimizer):
    """
    Optimizer for hardware switching decisions.
    
    Integrates with cross-subsystem optimization to make intelligent
    hardware switching decisions based on system-wide performance metrics.
    """
    
    def __init__(self, 
                 initial_hardware: Optional[str] = None,
                 available_hardware: Optional[List[str]] = None):
        """
        Initialize hardware switching optimizer.
        
        Args:
            initial_hardware: Initial hardware type
            available_hardware: List of available hardware types
        """
        super().__init__("hardware_switching")
        
        # Initialize hardware switcher
        self.hardware_switcher = create_hardware_switcher(initial_hardware)
        self.current_hardware = self.hardware_switcher.hardware_type
        
        # Available hardware platforms
        self.available_hardware = available_hardware or ["loihi", "spinnaker", "truenorth", "simulated"]
        
        # Performance tracking
        self.performance_history = {hw_type: [] for hw_type in self.available_hardware}
        self.switch_history = []
        self.last_switch_time = 0
        self.min_switch_interval = 30.0  # Minimum seconds between switches
        
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize hardware selection based on system-wide context.
        
        Args:
            context: Optimization context with system-wide information
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        current_time = time.time()
        
        # Don't switch too frequently
        if current_time - self.last_switch_time < self.min_switch_interval:
            return {
                "hardware_type": self.current_hardware,
                "action": "none",
                "reason": "cooling_period",
                "performance": self.get_current_performance()
            }
        
        # Get current workload characteristics
        workload = self._analyze_workload(context)
        
        # Determine optimal hardware for current workload
        optimal_hardware, score = self._select_optimal_hardware(workload, context)
        
        # Check if we should switch
        if optimal_hardware != self.current_hardware:
            switch_threshold = 0.2  # Minimum improvement to trigger switch
            
            if score > switch_threshold:
                # Attempt hardware switch
                success = self.hardware_switcher.switch_hardware(optimal_hardware)
                
                if success:
                    self.current_hardware = optimal_hardware
                    self.last_switch_time = current_time
                    self.switch_history.append({
                        "timestamp": current_time,
                        "from": self.current_hardware,
                        "to": optimal_hardware,
                        "score": score,
                        "workload": workload
                    })
                    
                    logger.info(f"Switched hardware to {optimal_hardware} (score: {score:.2f})")
                    
                    return {
                        "hardware_type": optimal_hardware,
                        "action": "switched",
                        "reason": "performance_optimization",
                        "score": score,
                        "performance": self.get_current_performance()
                    }
                else:
                    logger.warning(f"Failed to switch to {optimal_hardware}")
                    return {
                        "hardware_type": self.current_hardware,
                        "action": "failed_switch",
                        "target": optimal_hardware,
                        "reason": "switch_failure",
                        "performance": self.get_current_performance()
                    }
        
        # No switch needed
        return {
            "hardware_type": self.current_hardware,
            "action": "none",
            "reason": "optimal_hardware",
            "performance": self.get_current_performance()
        }
    
    def get_resource_requirements(self) -> Dict[str, float]:
        """Get resource requirements for hardware switching."""
        # Hardware switching has minimal resource requirements
        return {
            "cpu": 0.05,
            "memory": 0.1
        }
    
    def get_current_performance(self) -> Dict[str, float]:
        """Get current hardware performance metrics."""
        hw_info = self.hardware_switcher.active_hardware.get_hardware_info()
        
        return {
            "utilization": hw_info.get("utilization", 0.0),
            "power_efficiency": hw_info.get("power_efficiency", 0.0),
            "throughput": hw_info.get("throughput", 0.0)
        }
    
    def _analyze_workload(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze current workload characteristics.
        
        Args:
            context: Optimization context
            
        Returns:
            Dict[str, float]: Workload characteristics
        """
        # Extract workload characteristics from context
        subsystem_performance = context.get("performance", {})
        
        # Default workload profile
        workload = {
            "compute_intensity": 0.5,
            "memory_intensity": 0.5,
            "sparsity": 0.5,
            "precision_requirements": 0.5
        }
        
        # Analyze active subsystems to determine workload characteristics
        if subsystem_performance:
            # Compute intensity based on propulsion and sensor processing
            propulsion_perf = subsystem_performance.get("propulsion", {})
            sensor_perf = subsystem_performance.get("sensor_processing", {})
            
            if propulsion_perf or sensor_perf:
                workload["compute_intensity"] = max(
                    propulsion_perf.get("efficiency", 0.5),
                    sensor_perf.get("processing_load", 0.5)
                )
            
            # Memory intensity based on stealth and navigation
            stealth_perf = subsystem_performance.get("stealth", {})
            nav_perf = subsystem_performance.get("navigation", {})
            
            if stealth_perf or nav_perf:
                workload["memory_intensity"] = max(
                    stealth_perf.get("effectiveness", 0.5),
                    nav_perf.get("map_complexity", 0.5)
                )
        
        return workload
    
    def _select_optimal_hardware(self, 
                               workload: Dict[str, float], 
                               context: Dict[str, Any]) -> Tuple[str, float]:
        """
        Select optimal hardware for current workload.
        
        Args:
            workload: Workload characteristics
            context: Optimization context
            
        Returns:
            Tuple[str, float]: Selected hardware type and score
        """
        # Hardware capability profiles
        hardware_profiles = {
            "loihi": {
                "compute_efficiency": 0.9,
                "memory_efficiency": 0.6,
                "sparsity_handling": 0.8,
                "precision": 0.7,
                "power_efficiency": 0.9
            },
            "spinnaker": {
                "compute_efficiency": 0.7,
                "memory_efficiency": 0.8,
                "sparsity_handling": 0.6,
                "precision": 0.8,
                "power_efficiency": 0.7
            },
            "truenorth": {
                "compute_efficiency": 0.8,
                "memory_efficiency": 0.5,
                "sparsity_handling": 0.9,
                "precision": 0.5,
                "power_efficiency": 0.95
            },
            "simulated": {
                "compute_efficiency": 0.5,
                "memory_efficiency": 0.9,
                "sparsity_handling": 0.7,
                "precision": 0.9,
                "power_efficiency": 0.3
            }
        }
        
        # Calculate scores for each hardware type
        scores = {}
        for hw_type, profile in hardware_profiles.items():
            if hw_type not in self.available_hardware:
                continue
                
            # Calculate match score between workload and hardware profile
            compute_score = (1 - abs(workload["compute_intensity"] - profile["compute_efficiency"]))
            memory_score = (1 - abs(workload["memory_intensity"] - profile["memory_efficiency"]))
            sparsity_score = (1 - abs(workload["sparsity"] - profile["sparsity_handling"]))
            precision_score = (1 - abs(workload["precision_requirements"] - profile["precision"]))
            
            # Weighted score
            scores[hw_type] = (
                0.4 * compute_score +
                0.3 * memory_score +
                0.2 * sparsity_score +
                0.1 * precision_score
            )
            
            # Apply power efficiency bonus if power is constrained
            power_constraint = context.get("constraints", {}).get("power", 1.0)
            if power_constraint < 0.5:  # Power is constrained
                scores[hw_type] *= profile["power_efficiency"]
        
        # Find hardware with highest score
        best_hardware = max(scores.items(), key=lambda x: x[1])
        return best_hardware