"""
UCAV Hardware Profile Templates

Provides specialized hardware configuration templates for common UCAV operational scenarios.
"""

from typing import Dict, Any, Optional, List
from src.core.hardware.unified_config_manager import ConfigCategory
from src.core.optimization.multi_objective import OptimizationObjective

class UCAVHardwareProfiles:
    """Hardware profile templates for common UCAV configurations."""
    
    @staticmethod
    def get_profile_list() -> Dict[str, List[str]]:
        """
        Get list of available UCAV hardware profiles by hardware type.
        
        Returns:
            Dict[str, List[str]]: Hardware types and their UCAV profiles
        """
        return {
            "loihi": ["tactical", "stealth", "reconnaissance", "endurance"],
            "spinnaker": ["tactical", "swarm", "reconnaissance", "training"],
            "truenorth": ["tactical", "pattern_recognition", "target_acquisition"],
            "simulated": ["debug", "training", "mission_rehearsal"]
        }
    
    @staticmethod
    def get_profile(hardware_type: str, profile_name: str) -> Optional[Dict[str, Any]]:
        """
        Get UCAV hardware profile configuration.
        
        Args:
            hardware_type: Hardware type
            profile_name: Profile name
            
        Returns:
            Optional[Dict[str, Any]]: Profile configuration or None if not found
        """
        profiles = {
            # Loihi profiles for UCAV
            "loihi": {
                "tactical": {
                    "hardware_type": "loihi",
                    "board_id": 0,
                    "chip_id": 0,
                    "neurons_per_core": 1024,
                    "cores_per_chip": 128,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 50,
                        "metrics": ["neuron_activity", "power_usage", "temperature"]
                    },
                    "optimization": {
                        "placement_strategy": "density",
                        "connection_optimization": "throughput"
                    },
                    "ucav_specific": {
                        "mission_type": "tactical",
                        "sensor_priority": "radar",
                        "decision_latency_ms": 10,
                        "threat_response_ms": 5
                    },
                    "_metadata": {
                        "category": ConfigCategory.HARDWARE.value,
                        "objectives": [
                            OptimizationObjective.PERFORMANCE.value,
                            OptimizationObjective.RELIABILITY.value
                        ]
                    }
                },
                "stealth": {
                    "hardware_type": "loihi",
                    "board_id": 0,
                    "chip_id": 0,
                    "neurons_per_core": 512,
                    "cores_per_chip": 64,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 200,
                        "metrics": ["power_usage", "temperature"]
                    },
                    "power_mode": "efficient",
                    "clock_speed": "reduced",
                    "ucav_specific": {
                        "mission_type": "stealth",
                        "sensor_priority": "passive",
                        "emission_control": "strict",
                        "thermal_signature_reduction": True
                    },
                    "_metadata": {
                        "category": ConfigCategory.HARDWARE.value,
                        "objectives": [
                            OptimizationObjective.STEALTH.value,
                            OptimizationObjective.THERMAL.value
                        ]
                    }
                },
                "reconnaissance": {
                    "hardware_type": "loihi",
                    "board_id": 0,
                    "chip_id": 0,
                    "neurons_per_core": 1024,
                    "cores_per_chip": 128,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 100
                    },
                    "ucav_specific": {
                        "mission_type": "reconnaissance",
                        "sensor_priority": "optical",
                        "data_processing": "high_throughput",
                        "storage_priority": "high"
                    },
                    "_metadata": {
                        "category": ConfigCategory.HARDWARE.value,
                        "objectives": [
                            OptimizationObjective.PERFORMANCE.value,
                            OptimizationObjective.EFFICIENCY.value
                        ]
                    }
                },
                "endurance": {
                    "hardware_type": "loihi",
                    "board_id": 0,
                    "chip_id": 0,
                    "neurons_per_core": 512,
                    "cores_per_chip": 64,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 1000
                    },
                    "power_mode": "ultra_efficient",
                    "ucav_specific": {
                        "mission_type": "endurance",
                        "sensor_duty_cycle": 0.3,
                        "processing_priority": "essential_only",
                        "power_reserve": 0.25
                    },
                    "_metadata": {
                        "category": ConfigCategory.HARDWARE.value,
                        "objectives": [
                            OptimizationObjective.EFFICIENCY.value,
                            OptimizationObjective.RELIABILITY.value
                        ]
                    }
                }
            },
            
            # SpiNNaker profiles for UCAV
            "spinnaker": {
                "tactical": {
                    "hardware_type": "spinnaker",
                    "board_address": "192.168.1.1",
                    "neurons_per_core": 255,
                    "cores_per_chip": 16,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 50
                    },
                    "ucav_specific": {
                        "mission_type": "tactical",
                        "sensor_fusion": "high_priority",
                        "decision_latency_ms": 15
                    },
                    "_metadata": {
                        "category": ConfigCategory.HARDWARE.value,
                        "objectives": [
                            OptimizationObjective.PERFORMANCE.value,
                            OptimizationObjective.RELIABILITY.value
                        ]
                    }
                },
                "swarm": {
                    "hardware_type": "spinnaker",
                    "board_address": "192.168.1.1",
                    "neurons_per_core": 255,
                    "cores_per_chip": 16,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 100
                    },
                    "ucav_specific": {
                        "mission_type": "swarm",
                        "communication_priority": "high",
                        "coordination_algorithms": ["consensus", "formation"],
                        "swarm_size": "adaptive"
                    },
                    "_metadata": {
                        "category": ConfigCategory.HARDWARE.value,
                        "objectives": [
                            OptimizationObjective.PERFORMANCE.value,
                            OptimizationObjective.EFFICIENCY.value
                        ]
                    }
                }
            }
        }
        
        # Return requested profile if it exists
        if hardware_type in profiles and profile_name in profiles[hardware_type]:
            return profiles[hardware_type][profile_name]
        return None