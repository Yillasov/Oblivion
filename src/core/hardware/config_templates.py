"""
Hardware Configuration Templates

Provides predefined templates for common hardware configuration use cases.
"""

from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime

from src.core.utils.logging_framework import get_logger
from src.core.hardware.hardware_config import config_store

logger = get_logger("config_templates")


class ConfigTemplates:
    """Predefined hardware configuration templates for common use cases."""
    
    @staticmethod
    def get_template_list() -> Dict[str, List[str]]:
        """
        Get list of available templates by hardware type.
        
        Returns:
            Dict[str, List[str]]: Hardware types and their templates
        """
        return {
            "loihi": ["minimal", "research", "high_performance", "power_efficient"],
            "spinnaker": ["minimal", "research", "distributed", "real_time"],
            "truenorth": ["minimal", "classification", "pattern_recognition"],
            "simulated": ["debug", "benchmark", "training"]
        }
    
    @staticmethod
    def get_template(hardware_type: str, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration template.
        
        Args:
            hardware_type: Hardware type
            template_name: Template name
            
        Returns:
            Optional[Dict[str, Any]]: Template configuration or None if not found
        """
        templates = {
            # Loihi templates
            "loihi": {
                "minimal": {
                    "hardware_type": "loihi",
                    "board_id": 0,
                    "chip_id": 0,
                    "neurons_per_core": 1024,
                    "cores_per_chip": 128,
                    "monitoring": {
                        "enabled": False
                    }
                },
                "research": {
                    "hardware_type": "loihi",
                    "board_id": 0,
                    "chip_id": 0,
                    "neurons_per_core": 1024,
                    "cores_per_chip": 128,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 100,
                        "metrics": ["neuron_activity", "power_usage", "temperature"]
                    },
                    "debug_mode": True,
                    "logging_level": "debug"
                },
                "high_performance": {
                    "hardware_type": "loihi",
                    "board_id": 0,
                    "chip_id": 0,
                    "neurons_per_core": 1024,
                    "cores_per_chip": 128,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 500
                    },
                    "optimization": {
                        "placement_strategy": "density",
                        "connection_optimization": "throughput"
                    }
                },
                "power_efficient": {
                    "hardware_type": "loihi",
                    "board_id": 0,
                    "chip_id": 0,
                    "neurons_per_core": 512,  # Reduced for power efficiency
                    "cores_per_chip": 64,     # Using fewer cores
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 1000
                    },
                    "power_mode": "efficient",
                    "clock_speed": "reduced"
                }
            },
            
            # SpiNNaker templates
            "spinnaker": {
                "minimal": {
                    "hardware_type": "spinnaker",
                    "board_address": "192.168.1.1",
                    "neurons_per_core": 255,
                    "cores_per_chip": 16,
                    "monitoring": {
                        "enabled": False
                    }
                },
                "research": {
                    "hardware_type": "spinnaker",
                    "board_address": "192.168.1.1",
                    "neurons_per_core": 255,
                    "cores_per_chip": 16,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 100,
                        "metrics": ["neuron_activity", "packet_loss", "cpu_usage"]
                    },
                    "debug_mode": True,
                    "logging_level": "debug"
                },
                "distributed": {
                    "hardware_type": "spinnaker",
                    "board_address": "192.168.1.1",
                    "neurons_per_core": 255,
                    "cores_per_chip": 16,
                    "boards": [
                        {"id": 0, "address": "192.168.1.1"},
                        {"id": 1, "address": "192.168.1.2"},
                        {"id": 2, "address": "192.168.1.3"}
                    ],
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 200
                    },
                    "routing": {
                        "strategy": "shortest_path"
                    }
                },
                "real_time": {
                    "hardware_type": "spinnaker",
                    "board_address": "192.168.1.1",
                    "neurons_per_core": 100,  # Fewer neurons for real-time performance
                    "cores_per_chip": 16,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 50
                    },
                    "real_time": {
                        "enabled": True,
                        "max_delay_ms": 10,
                        "priority": "high"
                    }
                }
            },
            
            # TrueNorth templates
            "truenorth": {
                "minimal": {
                    "hardware_type": "truenorth",
                    "board_id": 0,
                    "neurons_per_core": 256,
                    "cores_per_chip": 4096,
                    "monitoring": {
                        "enabled": False
                    }
                },
                "classification": {
                    "hardware_type": "truenorth",
                    "board_id": 0,
                    "neurons_per_core": 256,
                    "cores_per_chip": 4096,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 500
                    },
                    "network_layout": "feedforward",
                    "classification": {
                        "output_mode": "winner_take_all",
                        "threshold_adaptation": True
                    }
                },
                "pattern_recognition": {
                    "hardware_type": "truenorth",
                    "board_id": 0,
                    "neurons_per_core": 256,
                    "cores_per_chip": 4096,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 200
                    },
                    "network_layout": "recurrent",
                    "pattern_recognition": {
                        "temporal_integration": True,
                        "adaptation_rate": 0.05
                    }
                }
            },
            
            # Simulated hardware templates
            "simulated": {
                "debug": {
                    "hardware_type": "simulated",
                    "neurons_per_core": 1000,
                    "cores_per_chip": 16,
                    "chips_available": 1,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 10
                    },
                    "debug_mode": True,
                    "simulation_speed": "slow",
                    "detailed_logging": True
                },
                "benchmark": {
                    "hardware_type": "simulated",
                    "neurons_per_core": 1000,
                    "cores_per_chip": 16,
                    "chips_available": 4,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 100,
                        "metrics": ["execution_time", "memory_usage", "throughput"]
                    },
                    "benchmark_mode": True,
                    "simulation_speed": "fast"
                },
                "training": {
                    "hardware_type": "simulated",
                    "neurons_per_core": 1000,
                    "cores_per_chip": 16,
                    "chips_available": 2,
                    "monitoring": {
                        "enabled": True,
                        "interval_ms": 500
                    },
                    "learning": {
                        "enabled": True,
                        "algorithm": "stdp",
                        "learning_rate": 0.01
                    },
                    "simulation_speed": "balanced"
                }
            }
        }
        
        if hardware_type not in templates:
            logger.warning(f"No templates available for hardware type: {hardware_type}")
            return None
            
        if template_name not in templates[hardware_type]:
            logger.warning(f"Template '{template_name}' not found for {hardware_type}")
            return None
            
        return templates[hardware_type][template_name]
    
    @staticmethod
    def apply_template(hardware_type: str, template_name: str, config_name: str) -> bool:
        """
        Apply template and save as a new configuration.
        
        Args:
            hardware_type: Hardware type
            template_name: Template name
            config_name: Name for the new configuration
            
        Returns:
            bool: Success status
        """
        template = ConfigTemplates.get_template(hardware_type, template_name)
        if not template:
            return False
            
        # Add metadata
        template["_metadata"] = {
            "created_from_template": template_name,
            "created_at": datetime.now().isoformat()
        }
        
        # Save configuration
        return config_store.save_config(hardware_type, config_name, template)